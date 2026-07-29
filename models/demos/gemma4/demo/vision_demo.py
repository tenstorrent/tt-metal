# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Gemma-4 vision demo: image + text prompt -> generated text.

Integrates the on-device Gemma-4 vision tower with the text model via the
multimodal extension of ``Gemma4Generator`` and drives a single-user
prefill + decode loop through the shared ``Generator`` interface (mirrors
``text_demo_v2.py``).

Flow:
    AutoProcessor (Gemma4Processor) -> input_ids (with image_token_id placeholders),
        pixel_values, image_position_ids
    Gemma4Generator.prefill_forward_multimodal(...) -> next-token logits
    Gemma4Generator.decode_forward(...) (inherited) -> greedy decode loop

Usage:
    HF_MODEL=google/gemma-4-31B-it pytest \\
        models/demos/gemma4/demo/vision_demo.py -k "1x8" -sv

    # Override the prompt / image / generation length:
    HF_MODEL=google/gemma-4-31B-it GEMMA4_VISION_PROMPT="Describe this image." \\
        GEMMA4_VISION_IMAGE=models/tt_transformers/demo/sample_prompts/llama_models/dog.jpg \\
        pytest models/demos/gemma4/demo/vision_demo.py -k "1x8" -sv
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from PIL import Image as PIL_Image

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig

IMG_PATH = Path("models/tt_transformers/demo/sample_prompts/llama_models").resolve()


def _model_path():
    return os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )


def _device_params():
    if is_blackhole():
        trace_region_size = int(os.environ.get("GEMMA4_TRACE_REGION_SIZE", 256_000_000))
        return {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": trace_region_size,
            "num_command_queues": 1,
        }
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30_000_000, "num_command_queues": 1}


def create_tt_page_table(batch_size, paged_attention_config: PagedAttentionConfig):
    if paged_attention_config is None:
        return None
    n_blocks = paged_attention_config.max_num_blocks
    cols = n_blocks // batch_size
    return torch.arange(n_blocks, dtype=torch.int32)[: batch_size * cols].reshape(batch_size, cols)


def _host_sample(logits, temperature, top_p):
    """Greedy argmax (temperature==0) or top-p sampling on host."""
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    if logits.dim() == 2:
        logits = logits.unsqueeze(0) if logits.shape[0] != 1 else logits
    if not temperature or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_idx, -1, choice)


def encode_multimodal(prompt, image, processor):
    """Tokenize an image+text prompt with the HF Gemma4Processor.

    Returns (input_ids, pixel_values, image_position_ids) as torch tensors.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    encoded = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cpu", dtype=torch.bfloat16)
    input_ids = encoded["input_ids"].to(torch.long)
    pixel_values = encoded["pixel_values"]
    image_position_ids = encoded["image_position_ids"].to(torch.long)
    return input_ids, pixel_values, image_position_ids


@pytest.mark.parametrize("batch_size", [32, 1])
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "T3K": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), (1, 8))
    ],
    indirect=True,
)
def test_demo_vision(mesh_device, batch_size, reset_seeds, is_ci_env):
    """Gemma-4 multimodal (image + text) generation demo.

    batch=1: single-user prefill + decode (vision encoded inline).
    batch=32: vision tower runs num_devices images at a time (data-parallel), then the
    text model prefills each user one at a time (single-user loop), then a batched
    decode loop. All 32 users share the same image+prompt (throughput demo).
    """
    import math

    from transformers import AutoProcessor

    model_path = _model_path()
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", 4096))
    max_generated_tokens = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", 128))
    num_layers = os.environ.get("GEMMA4_NUM_LAYERS")
    num_layers = int(num_layers) if num_layers else None
    temperature = float(os.environ.get("GEMMA4_TEMPERATURE", 0.0))
    top_p = float(os.environ.get("GEMMA4_TOP_P", 0.9))

    # Image + prompt (env-overridable).
    image_file = os.environ.get("GEMMA4_VISION_IMAGE", str(IMG_PATH / "dog.jpg"))
    prompt = os.environ.get("GEMMA4_VISION_PROMPT", "Write a short summary about this image.")
    logger.info(f"Vision demo: image={image_file}, prompt={prompt!r}")
    image = PIL_Image.open(image_file).convert("RGB")

    # Paged attention — KV pool sized to the ACTUAL need (batch * ceil(per_user_ctx /
    # block_size), per_user_ctx = prompt_len + max_new), which is ~10x smaller than
    # text_demo_v2's batch*ceil(max_seq_len/block_size) provisioning (text_demo_v2 can't
    # know prompt_len at build time; we encode first, so we can). Plus bounded sliding KV
    # support (mirrors text_demo_v2): auto-fall back to bounded above 64k context so the
    # 50 sliding layers cap at the 1024-token window. The cache is replicated across the
    # mesh, so batch=32 on a 4GB BH device still needs a reduced GEMMA4_MAX_SEQ_LEN /
    # GEMMA4_NUM_LAYERS to fit (or DP-sharded KV — future work).
    block_size = int(os.environ.get("GEMMA4_PAGE_BLOCK_SIZE", 64))

    # ── Processor + prompt encoding (needed to size the KV pool) ───────────
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, do_convert_rgb=True)
    input_ids, pixel_values, image_position_ids = encode_multimodal(prompt, image, processor)
    prompt_len = int(input_ids.shape[1])
    per_user_ctx = prompt_len + max_generated_tokens
    page_max_num_blocks = math.ceil(batch_size * per_user_ctx / block_size)
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=page_max_num_blocks)
    # Bounded sliding KV: full (unbounded) by default; auto-fall back to bounded above
    # 64k context so the 50 sliding layers cap at the 1024-token window (only the 10
    # full-attention layers grow). Override with GEMMA4_BOUNDED_SLIDING=0/1.
    _bs_env = os.environ.get("GEMMA4_BOUNDED_SLIDING")
    bounded_sliding = (max_seq_len > 65536) if _bs_env is None else _bs_env.lower() in ("1", "true", "yes")
    logger.info(
        f"KV pool: {page_max_num_blocks} blocks x {block_size} = {page_max_num_blocks * block_size} tokens "
        f"({per_user_ctx}/user x {batch_size} users, bounded_sliding={bounded_sliding})"
    )

    profiler = BenchmarkProfiler()
    profiler.start("run")

    # ── Model + vision tower + projector ───────────────────────────────────
    logger.info(f"Loading Gemma-4 multimodal model from {model_path}...")
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
        multimodal=True,
    )
    model_args = generator.model_args[0]
    page_table = create_tt_page_table(batch_size, paged_attention_config)

    # Bounded sliding needs per-layer page tables (sliding layers index their small
    # bounded pool, full layers the full pool). Build them once and stash on the model
    # so prefill/decode pick them up via _active_page_tables_per_layer (mirrors text_demo_v2).
    if bounded_sliding:
        from models.demos.gemma4.tt.attention.kv_cache_hybrid import build_hybrid_page_tables

        n_layers = num_layers or model_args.num_hidden_layers
        sliding_mask = [model_args.layer_types[i] == "sliding_attention" for i in range(n_layers)]
        per_layer_pts = build_hybrid_page_tables(
            n_layers,
            sliding_mask,
            num_users=batch_size,
            block_size=block_size,
            max_seq_len=max_seq_len,
            sliding_window=model_args.sliding_window,
        )
        generator.model[0]._active_page_tables_per_layer = per_layer_pts
        logger.info(f"Bounded sliding: installed {len(per_layer_pts)} per-layer page tables")

    # ── Warmup ─────────────────────────────────────────────────────────────
    # Compile the prefill path (non-traced — the vision encoder can't be traced)
    # + the vision tower on the real image. Decode is traced separately: its
    # Metal trace is auto-captured on the first decode iteration (iteration 0).
    generator.warmup_model_prefill(
        kv_cache=tt_kv_cache,
        enable_trace=False,
        can_sample_on_device=False,
        greedy_only=True,
    )
    # Vision compile on the actual image (also gives a soft-token count sanity check).
    enc = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cpu", dtype=torch.bfloat16)
    generator.warmup_vision(enc["pixel_values"], enc["image_position_ids"].to(torch.long))
    del enc

    # ── Encode the prompt (already done above for sizing; recompute for use) ─
    n_image_tokens = int((input_ids[0] == generator.image_token_id).sum().item())
    logger.info(
        f"Encoded prompt: {prompt_len} tokens ({n_image_tokens} image soft tokens, " f"{pixel_values.shape[1]} patches)"
    )
    assert (
        prompt_len + max_generated_tokens <= max_seq_len
    ), f"prompt ({prompt_len}) + max_generated_tokens ({max_generated_tokens}) > max_seq_len ({max_seq_len})"
    tokens = input_ids  # [1, prompt_len]

    # ── Prefill ───────────────────────────────────────────────────────────
    logger.info("Starting multimodal prefill...")
    profiler.start("inference_prefill")
    if batch_size == 1:
        prefill_logits = generator.prefill_forward_multimodal(
            tokens,
            pixel_values,
            image_position_ids,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=[prompt_len],
        )
        next_token = _host_sample(prefill_logits, temperature, top_p)  # [1,1]
    else:
        # Vision: tower runs num_devices images at a time (DP) -> per-user host embeds.
        pixel_values_batch = pixel_values.repeat(batch_size, 1, 1)
        image_position_ids_batch = image_position_ids.repeat(batch_size, 1, 1)
        per_user_embeds = generator.encode_vision_batch(pixel_values_batch, image_position_ids_batch)
        # Text: single-user prefill loop. Each user's KV is written to their own
        # page-table blocks (disjoint, via create_tt_page_table), so the subsequent
        # batched decode reads every user's KV at once.
        replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None
        next_tokens = torch.empty(batch_size, dtype=torch.long)
        for u in range(batch_size):
            image_embeds_tt = ttnn.from_torch(
                per_user_embeds[u],
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_mapper,
            )
            logits_u = generator.prefill_forward_multimodal(
                tokens,
                image_embeds_tt=image_embeds_tt,
                page_table=page_table[u : u + 1],
                kv_cache=tt_kv_cache,
                prompt_lens=[prompt_len],
            )
            # image_embeds_tt is deallocated inside prefill_forward_multimodal (scatter step).
            next_tokens[u] = int(_host_sample(logits_u, temperature, top_p).item())
        next_token = next_tokens.view(batch_size, 1)
    profiler.end("inference_prefill")
    logger.info("Prefill finished")

    # ── Decode loop (text-only, reused from Gemma4Generator) ──────────────
    stop_tokens = tokenizer.stop_tokens
    if batch_size == 1:
        all_tokens = input_ids[0].tolist() + [int(next_token.item())]
        current_pos = torch.tensor([prompt_len], dtype=torch.long)
        out_tok = next_token.view(1, 1).to(torch.long)
        iteration = 0

        logger.info("Starting decode loop...")
        profiler.start("inference_decode")
        while iteration < max_generated_tokens - 1:
            # Iteration 0 is the decode compile/trace-capture step (excluded from the
            # steady-state average below). Traced decode replays a single captured
            # Metal trace per token, matching text_demo_v2's decode path.
            profiler.start(f"inference_decode_time_{iteration}")
            logits, _ = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=True,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                sampling_params=None,
            )
            out_tok = _host_sample(logits, temperature, top_p)
            current_pos += 1
            tok = int(out_tok[0, 0].item())
            all_tokens.append(tok)
            profiler.end(f"inference_decode_time_{iteration}")
            iteration += 1
            if not is_ci_env and iteration % 16 == 0:
                partial = tokenizer.decode(all_tokens[prompt_len:])
                logger.info(f"[decode {iteration}] {partial.strip()!r}")
            if tok in stop_tokens:
                logger.info(f"Stop token {tok} hit at decode iteration {iteration}")
                break
        profiler.end("inference_decode")
        profiler.end("run")
        generated = tokenizer.decode(all_tokens[prompt_len:])
        logger.info(f"\n==VISION DEMO PROMPT\n{prompt}\n==VISION DEMO GENERATION\n{generated.strip()}\n")
    else:
        # Batched decode: all 32 users step together. Each iteration produces one token
        # per user; iteration 0 is the compile/trace-capture step (excluded from the
        # steady-state average). Users hit a stop token are marked done and stop appending.
        all_outputs = [input_ids[0].tolist() for _ in range(batch_size)]
        for u in range(batch_size):
            all_outputs[u].append(int(next_tokens[u].item()))
        current_pos = torch.tensor([prompt_len] * batch_size, dtype=torch.long)
        out_tok = next_tokens.view(batch_size, 1).to(torch.long)
        user_done = [False] * batch_size
        iteration = 0
        users_decoding = True

        logger.info("Starting decode loop...")
        profiler.start("inference_decode")
        while users_decoding and iteration < max_generated_tokens - 1:
            profiler.start(f"inference_decode_time_{iteration}")
            logits, _ = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=True,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                sampling_params=None,
            )
            out_tok = _host_sample(logits, temperature, top_p)
            profiler.end(f"inference_decode_time_{iteration}")
            current_pos += 1
            for u in range(batch_size):
                tok = int(out_tok[u, 0].item())
                if tok not in stop_tokens and not user_done[u]:
                    all_outputs[u].append(tok)
                elif not user_done[u]:
                    user_done[u] = True
                    logger.info(f"User {u} hit stop token {tok} at decode iteration {iteration + 1}")
            if all(user_done):
                users_decoding = False
            iteration += 1
            if not is_ci_env and iteration % 16 == 0:
                for u in range(batch_size):
                    partial = tokenizer.decode(all_outputs[u][prompt_len:])
                    logger.info(f"[decode {iteration} u{u}] {partial.strip()!r}")
        profiler.end("inference_decode")
        profiler.end("run")
        logger.info("Finished decoding. Final outputs:")
        for u in range(batch_size):
            gen_u = tokenizer.decode(all_outputs[u][prompt_len:])
            logger.info(f"\n==USER {u} - PROMPT\n{prompt}\n==USER {u} - GENERATION\n{gen_u.strip()}\n")

    # ── Metrics ───────────────────────────────────────────────────────────
    total_prefill = profiler.get_duration("inference_prefill")
    total_decode = profiler.get_duration("inference_decode")
    # Steady-state decode excludes iteration 0 (the compile/trace-capture step),
    # mirroring text_demo_v2 so the reported rate is comparable to the text demo.
    steady_decode = sum(profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, iteration))
    steady_iters = max(iteration - 1, 0)
    decode_tps_u = steady_iters / steady_decode if steady_decode > 0 else 0.0
    decode_tps = decode_tps_u * batch_size
    logger.info("")
    logger.info(f"=== Performance metrics (batch_size={batch_size}) ===")
    logger.info(f"Prompt tokens: {prompt_len} ({n_image_tokens} image), generated: {iteration + 1}")
    logger.info(f"Time to First Token (TTFT): {total_prefill * 1000:.1f} ms")
    if batch_size > 1:
        logger.info(f"Amortized prefill/user: {total_prefill / batch_size * 1000:.1f} ms")
    if steady_iters > 0:
        logger.info(
            f"Decode: {1000 / decode_tps_u:.2f} ms/token @ {decode_tps_u:.2f} tok/s/user "
            f"({decode_tps:.2f} tok/s aggregate, traced, steady-state excl. compile)"
        )
    else:
        logger.info("Decode: n/a (no steady-state decode iterations recorded)")
    logger.info(f"Full demo runtime: {profiler.get_duration('run'):.1f} s")

    if is_ci_env:
        measurements = {
            "inference_prefill": total_prefill,
            "inference_decode": total_decode,
            "prefill_time_to_token": total_prefill,
            "decode_t/s/u": decode_tps_u,
            "Full demo runtime": profiler.get_duration("run"),
        }
        benchmark_data = create_benchmark_data(
            profiler, measurements, {"inference_prefill": 0, "inference_decode": 0}, {}
        )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo",
            ml_model_name=f"{Path(model_path).name}-Vision",
            ml_model_type="vlm",
            batch_size=batch_size,
            config_params={},
            input_sequence_length=prompt_len,
            output_sequence_length=iteration + 1,
        )

    assert iteration >= 0, "decode produced no tokens"
