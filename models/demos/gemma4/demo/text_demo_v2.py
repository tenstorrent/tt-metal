# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generator-based Gemma4 text demo — structured after the Gemma3 text demo
(``models/demos/multimodal/gemma3/demo/text_demo.py``).

Unlike the original hand-rolled ``text_demo.py`` loop, this demo drives the
model through the shared ``Generator`` interface (via ``Gemma4Generator``),
mirroring how Gemma3 / tt_transformers models are run:

  from_pretrained → warmup_model_prefill → prefill_forward_text → decode_forward

Differences from the Gemma3 demo (Gemma4-specific):
  * Single model instance, no data-parallel submeshes (Gemma4 runs batch=1 per
    submesh today, so the demo focuses on the latency / long-context configs).
  * Host sampling (greedy argmax / top-p): Gemma4 does not expose on-device
    sampling through the demo path, so logits are read back and sampled on host.
  * No decode warmup (``warmup_model_decode`` is Gemma3-generator specific); the
    first decode iteration serves as the compile step and is excluded from the
    reported steady-state perf (matching the benchmark warmup convention).

All model-level optimizations (BFP8 weights via precision_overrides.json,
width-sharded RMSNorm, row-major RoPE caches, nlp_concat_heads_decode) are
applied automatically because they live in the shared model code that
``Gemma4Generator.from_pretrained`` builds.

Usage:
    HF_MODEL=google/gemma-4-31B-it pytest \
        models/demos/gemma4/demo/text_demo_v2.py -k "batch-1" -sv

    # Override prompts / lengths from the CLI:
    HF_MODEL=google/gemma-4-31B-it pytest \
        models/demos/gemma4/demo/text_demo_v2.py -k "batch-1" -sv \
        --max_generated_tokens 64
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import requests
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill
from models.tt_transformers.tt.model_config import determine_device_name

_CONTEXT_CACHE_DIR = Path("models/tt_transformers/demo/context_cache")


def _model_path():
    return os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )


def load_and_cache_context(context_url, cache_dir, max_length=None):
    """Fetch a long-context source from a URL with on-disk caching (mirrors gemma3)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()
    if cache_file.exists():
        context_text = cache_file.read_text()
        logger.info(f"Loaded context from cache: {context_url}")
    else:
        try:
            response = requests.get(context_url, timeout=60)
            if response.status_code == 200:
                context_text = response.text
                cache_file.write_text(context_text)
                logger.info(f"Downloaded and cached context: {context_url}")
            else:
                logger.warning(f"Failed to fetch context from {context_url}: {response.status_code}")
                context_text = ""
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error fetching context from {context_url}: {e}")
            context_text = ""
    if max_length:
        context_text = context_text[:max_length]
        logger.info(f"Clipped context to {max_length} chars")
    return context_text


def load_inputs(user_input, batch, instruct):
    """Load prompts from a json file (optionally fetching a gutenberg context), repeated to `batch`."""
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    if len(user_input) < batch:
        logger.warning(f"Fewer prompts ({len(user_input)}) than batch ({batch}); repeating to fill the batch.")
        user_input = user_input * batch

    in_prompt = []
    for i in range(batch):
        prompt = user_input[i]["prompt"]
        if "context" in user_input[i]:
            max_length = user_input[i].get("max_length")
            context_text = load_and_cache_context(user_input[i]["context"], _CONTEXT_CACHE_DIR, max_length=max_length)
            prompt = ("```" + context_text + "```\n\n" + prompt) if instruct else context_text
        in_prompt.append(prompt)
    return in_prompt


def create_tt_page_table(batch_size, paged_attention_config: PagedAttentionConfig):
    """Identity logical→physical page table [batch, n_blocks/batch] (single-DP)."""
    if paged_attention_config is None:
        return None
    n_blocks = paged_attention_config.max_num_blocks
    cols = n_blocks // batch_size
    return torch.arange(n_blocks, dtype=torch.int32)[: batch_size * cols].reshape(batch_size, cols)


def _host_sample(logits, temperature, top_p):
    """Sample next tokens on host. Greedy argmax for temperature==0, else top-p.

    logits: torch.Tensor shaped [B, vocab] or [B, 1, vocab].
    Returns: torch.LongTensor [B, 1].
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]
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


def _device_params():
    """Blackhole needs a larger trace region; keep a single command queue (host sampling)."""
    if is_blackhole():
        return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90_000_000, "num_command_queues": 1}
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30_000_000, "num_command_queues": 1}


# Parameters mirror the Gemma3 demo layout (subset): a latency config, a long-context
# config, and a CI config. Gemma4 runs batch=1, so throughput/DP rows are omitted.
@pytest.mark.parametrize(
    "input_prompts, instruct, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, "
    "sampling_params, stop_at_eos, ci_only, enable_trace",
    [
        (  # batch-1 (latency) — single user, short prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1024,
            1,
            200,
            True,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # batch-8 (throughput) — 8 concurrent users, short prompt
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1024,
            8,
            200,
            True,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # batch-32 (max throughput) — 32 concurrent users (decode batch ceiling)
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            1024,
            32,
            200,
            True,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # long-context-4k — single user, long prompt
            "models/tt_transformers/demo/sample_prompts/input_data_long_4k.json",
            True,
            4096,
            1,
            200,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 2048},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # long-context-64k — single user, ~64k prompt. max_seq_len is a power
            # of 2 so the prompt prefills in a single chunk (see the note below).
            "models/tt_transformers/demo/sample_prompts/input_data_long_128k.json",
            True,
            64 * 1024,
            1,
            200,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        # NOTE on the 128k/256k entries below:
        #   KV memory fits on QB2 (4x P300) via bounded sliding + right-sized paging,
        #   and single-chunk prefill runs at 128k without OOM. Long context is now
        #   coherent end-to-end (validated at 100k-token prompts) after fixing two
        #   bugs: (1) prefill ignored the Generator's chunk_start_idx -> forced a
        #   single prefill chunk + in-call chunked SDPA (full-attn via chunked SDPA
        #   over the paged cache, sliding via overlapping-window SDPA); (2) the
        #   bounded sliding cache wrote the prompt's padding tail over its real
        #   window -> capped the bounded-cache fill to the unpadded prompt length.
        #
        #   max_seq_len should be a power of 2: single-chunk prefill pads the prompt
        #   up to the next power of 2, which must fit the RoPE/KV caches.
        (  # long-context-128k — single user, 128k prompt (bounded sliding auto-on)
            "models/tt_transformers/demo/sample_prompts/input_data_long_128k.json",
            True,
            128 * 1024,
            1,
            200,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 2048},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # long-context-256k — single user, max context (uses the 128k prompt; KV pool sized for 256k)
            "models/tt_transformers/demo/sample_prompts/input_data_long_128k.json",
            True,
            256 * 1024,
            1,
            200,
            True,
            {"page_block_size": 64, "page_max_num_blocks": 4096},
            {"temperature": 0, "top_p": 0.08},
            True,
            False,
            True,
        ),
        (  # ci-1 — single user, fixed iteration count for perf tracking
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            8192,
            1,
            512,
            True,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            {"temperature": 0, "top_p": 0.08},
            False,
            True,
            True,
        ),
    ],
    ids=[
        "batch-1",
        "batch-8",
        "batch-32",
        "long-context-4k",
        "long-context-64k",
        "long-context-128k",
        "long-context-256k",
        "ci-1",
    ],
)
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
        }.get(os.environ.get("MESH_DEVICE"), (1, 4))
    ],
    indirect=True,
)
def test_demo_text(
    input_prompts,
    instruct,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    stop_at_eos,
    mesh_device,
    is_ci_env,
    ci_only,
    enable_trace,
    reset_seeds,
    request,
):
    """Gemma4 text generation through the Generator interface, modeled on the Gemma3 demo."""
    if is_ci_env and not ci_only:
        pytest.skip("CI only runs the CI-only configs")

    # Env overrides (Gemma4's conftest doesn't register the tt_transformers CLI
    # flags, so we keep overrides env-based). GEMMA4_NUM_LAYERS is a smoke-test
    # hook to build a few-layer model for a fast end-to-end wiring check.
    import math

    max_generated_tokens = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", max_generated_tokens))
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", max_seq_len))
    num_layers = os.environ.get("GEMMA4_NUM_LAYERS")
    num_layers = int(num_layers) if num_layers else None
    # Batch sweep hook: GEMMA4_BATCH overrides the config's batch_size so the
    # same config can probe batch-1 / 8 / 32 to find the QB2 ceiling.
    batch_size = int(os.environ.get("GEMMA4_BATCH", batch_size))

    model_path = _model_path()
    temperature = sampling_params.get("temperature", 0)
    top_p = sampling_params.get("top_p", 1.0)

    profiler = BenchmarkProfiler()
    profiler.start("run")

    # ── Inputs ────────────────────────────────────────────────────────────
    profiler.start("loading_inputs")
    prompts = load_inputs(input_prompts, batch_size, instruct)
    profiler.end("loading_inputs")

    # Right-size the paged KV pool to the actual context. The configs carried a
    # fixed page_max_num_blocks (e.g. 2048 = 131072 tokens) that over-allocates
    # KV ~16x vs max_seq_len and OOMs on long contexts; size it to exactly the
    # blocks needed (batch * ceil(max_seq_len / block_size)).
    block_size = page_params["page_block_size"]
    page_max_num_blocks = batch_size * math.ceil(max_seq_len / block_size)
    paged_attention_config = (
        PagedAttentionConfig(block_size=block_size, max_num_blocks=page_max_num_blocks) if paged_attention else None
    )

    # Bounded sliding KV cache: required for long context (>~64k). Without it,
    # the 50 sliding layers each allocate the *full* context KV (~25 GB at 128k)
    # and OOM; bounded mode caps them at the 1024-token sliding window so only
    # the 10 full-attention layers grow with context. Auto-enable for long
    # contexts; override with GEMMA4_BOUNDED_SLIDING=0/1.
    _bs_env = os.environ.get("GEMMA4_BOUNDED_SLIDING")
    bounded_sliding = (max_seq_len > 16384) if _bs_env is None else _bs_env.lower() in ("1", "true", "yes")
    bounded_sliding = bounded_sliding and paged_attention

    # ── Model (all optimizations applied inside create_tt_model) ───────────
    logger.info(
        f"Loading Gemma4 from {model_path} (layers={num_layers or 'all'}, max_seq_len={max_seq_len}, "
        f"bounded_sliding={bounded_sliding})..."
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
    )
    model_args_list = generator.model_args  # preprocess_inputs_prefill iterates this
    model_args = model_args_list[0]

    page_table = create_tt_page_table(batch_size, paged_attention_config)

    # Bounded sliding needs per-layer page tables (sliding layers index their
    # small bounded pool, full layers the full pool). Build them once and stash
    # on the model so prefill/decode pick them up via _active_page_tables_per_layer.
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

    # ── Warmup (prefill compile + optional trace) ──────────────────────────
    logger.info("Warming up prefill...")
    generator.warmup_model_prefill(
        kv_cache=tt_kv_cache,
        enable_trace=enable_trace,
        can_sample_on_device=False,
        non_greedy_decoding_on_device=False,
    )
    logger.info("Warmup complete")

    # ── Prefill ────────────────────────────────────────────────────────────
    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        prompts, tokenizer, model_args_list, instruct, max_generated_tokens, max_prefill_len=max_seq_len
    )
    max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
    assert max_generated_tokens + max_encoded_prompt_len <= max_seq_len, (
        f"prompt ({max_encoded_prompt_len}) + max_generated_tokens ({max_generated_tokens}) "
        f"must be <= max_seq_len ({max_seq_len})"
    )
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

    logger.info("Starting prefill...")
    profiler.start("inference_prefill")
    prefill_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        warmup_prefill=False,
    )
    prefilled_token = _host_sample(prefill_logits, temperature, top_p)
    profiler.end("inference_prefill")
    logger.info("Prefill finished")

    prefilled_flat = prefilled_token.view(batch_size, -1).squeeze(-1)
    all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
    for user in range(batch_size):
        all_outputs[user].append(int(prefilled_flat[user].item()))

    # ── Decode loop ─────────────────────────────────────────────────────────
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])
    out_tok = prefilled_flat.reshape(batch_size, 1)
    user_done = [False] * batch_size
    iteration = 0
    users_decoding = True

    logger.info("Starting decode loop...")
    profiler.start("inference_decode")
    while users_decoding:
        profiler.start(f"inference_decode_time_{iteration}")
        logits, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=enable_trace,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=None,  # host sampling
        )
        out_tok = _host_sample(logits, temperature, top_p)
        profiler.end(f"inference_decode_time_{iteration}")

        current_pos += 1
        for user in range(batch_size):
            tok = int(out_tok[user, 0].item())
            if tok not in tokenizer.stop_tokens and not user_done[user]:
                all_outputs[user].append(tok)
            elif stop_at_eos:
                user_done[user] = True
                if all(user_done):
                    users_decoding = False

        if not is_ci_env:
            for user in range(batch_size):
                text = "".join(tokenizer.decode(all_outputs[user]))
                text = ("..." + text[-97:]) if len(text) > 100 else text
                logger.info(f"[User {user}] {text.replace(chr(10), ' ')}")

        iteration += 1
        if iteration >= max_generated_tokens:
            users_decoding = False
    profiler.end("inference_decode")
    profiler.end("run")

    # ── Final outputs ────────────────────────────────────────────────────────
    # Print the GENERATED tokens separately from the prompt: all_outputs holds
    # the full prompt followed by generated tokens, so decoding the whole thing
    # is dominated by the prompt (misleading for long context — the model may be
    # echoing the prompt). Slice off the prompt to judge generation quality.
    logger.info("Finished decoding. Final outputs:")
    for i, (output, prompt) in enumerate(zip(all_outputs, prompts)):
        gen_text = tokenizer.decode(output[prefill_lens[i] :])
        short_prompt = (prompt[:100] + "\n<...>\n" + prompt[-100:]) if len(prompt) > 200 else prompt
        logger.info(f"\n==USER {i} - PROMPT\n{short_prompt}\n==USER {i} - GENERATION ONLY\n{gen_text.strip()}\n")

    # ── Performance metrics ───────────────────────────────────────────────────
    total_prefill = profiler.get_duration("inference_prefill")
    # Iteration 0 is the decode compile step — exclude from the steady-state average.
    steady_iters = max(iteration - 1, 1)
    total_decode = sum(profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, iteration))
    ttft_ms = total_prefill / batch_size * 1000
    decode_tps_u = steady_iters / total_decode if total_decode > 0 else 0
    decode_tps = decode_tps_u * batch_size

    logger.info("")
    logger.info("=== Performance metrics ===")
    logger.info(f"Prompt tokens: {prefill_lens[0]}, generated tokens: {iteration}")
    logger.info(f"Time to First Token (TTFT): {ttft_ms:.1f} ms")
    logger.info(
        f"Decode: {1000 / decode_tps_u:.2f} ms/token @ {decode_tps_u:.2f} tok/s/user "
        f"({decode_tps:.2f} tok/s throughput)"
    )
    logger.info(f"Full demo runtime: {profiler.get_duration('run'):.1f} s")

    if is_ci_env:
        measurements = {
            "inference_prefill": total_prefill,
            "inference_decode": total_decode,
            "prefill_time_to_token": total_prefill / batch_size,
            "decode_t/s/u": decode_tps_u,
            "decode_t/s": decode_tps,
            "Full demo runtime": profiler.get_duration("run"),
        }
        benchmark_data = create_benchmark_data(
            profiler, measurements, {"inference_prefill": 0, "inference_decode": 1}, {}
        )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo",
            ml_model_name=Path(model_path).name,
            ml_model_type="llm",
            device_name=determine_device_name(mesh_device),
            num_layers=num_layers or model_args.num_hidden_layers,
            batch_size=batch_size,
            config_params={},
            input_sequence_length=prefill_lens[0],
            output_sequence_length=iteration,
        )

    assert iteration > 0, "decode produced no tokens"
