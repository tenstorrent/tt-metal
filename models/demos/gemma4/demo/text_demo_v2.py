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
            repeat_context = int(user_input[i].get("repeat_context", 1))
            if repeat_context > 1:
                context_text = "\n\n".join([context_text] * repeat_context)
                logger.info(f"Repeated context {repeat_context}x ({len(context_text)} chars)")
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
    """Blackhole needs a larger trace region; keep a single command queue (host sampling).

    The batch-32 decode trace is the largest (~228 MB at capture), so the
    Blackhole trace region is sized above that with margin. ``GEMMA4_TRACE_REGION_SIZE``
    overrides it for configs that need a different budget.
    """
    if is_blackhole():
        trace_region_size = int(os.environ.get("GEMMA4_TRACE_REGION_SIZE", 256_000_000))
        return {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": trace_region_size,
            "num_command_queues": 1,
        }
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
            4096,
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
        (  # long-context-256k — single user, prompt is clipped to max_seq_len - max_generated_tokens
            "models/tt_transformers/demo/sample_prompts/input_data_long_256k.json",
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

    # ── Speculative-decoding dispatch ────────────────────────────────────────
    # `--speculative` reroutes the demo through the it-assistant drafter +
    # target verifier path. Delegated to _run_spec_decode, which builds its own
    # target+drafter, so we return before this test loads a model.
    if request.config.getoption("--speculative"):
        draft_len = request.config.getoption("--spec-draft-len")
        if draft_len is None:
            draft_len = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 3))
        if batch_size != 1:
            # Batched (B>1) spec-decode: drafts each user at batch=1 and runs ONE
            # batched packed verify over all users (KV-amortization win). Greedy,
            # ragged per-user acceptance. Currently UNTRACED (host-dispatch bound).
            prompts = load_inputs(input_prompts, batch_size, instruct)
            _run_spec_decode_batched(
                prompts=prompts,
                instruct=instruct,
                max_seq_len=max_seq_len,
                max_generated_tokens=max_generated_tokens,
                page_params=page_params,
                sampling_params=sampling_params,
                mesh_device=mesh_device,
                enable_trace=enable_trace,
                draft_len=draft_len,
                num_layers=num_layers,
            )
            return
        prompt = load_inputs(input_prompts, 1, instruct)[0]
        _run_spec_decode(
            prompt=prompt,
            instruct=instruct,
            max_seq_len=max_seq_len,
            max_generated_tokens=max_generated_tokens,
            page_params=page_params,
            sampling_params=sampling_params,
            mesh_device=mesh_device,
            enable_trace=enable_trace,
            draft_len=draft_len,
            num_layers=num_layers,
        )
        return

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

    # Sliding-cache mode. Default: FULL (unbounded) sliding KV, which stays
    # coherent at long context — the bounded circular ring corrupts the recent
    # window on padded >32k prefills (see docs/bounded_sliding_kv_cache_debug.md).
    # Full KV allocates every sliding layer at full length, so it only fits up to
    # ~64k on this board; above that we auto-fall back to bounded sliding to avoid
    # OOM (the 50 sliding layers cap at the 1024-token window; only the 10
    # full-attention layers grow), accepting bounded's known >~34k degradation
    # there. Override either way with GEMMA4_BOUNDED_SLIDING=0/1.
    _bs_env = os.environ.get("GEMMA4_BOUNDED_SLIDING")
    bounded_sliding = (max_seq_len > 65536) if _bs_env is None else _bs_env.lower() in ("1", "true", "yes")
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
    # Prefill tracing buys ~nothing (prefill runs only a handful of times) and its
    # trace buffers scale with chunk_size×batch, overflowing the trace region at
    # long context (≥4K) with no perf gain. Gate prefill tracing off above a
    # threshold (decode stays traced); GEMMA4_PREFILL_TRACE_MAX_SEQ overrides.
    prefill_trace_max = int(os.environ.get("GEMMA4_PREFILL_TRACE_MAX_SEQ", 4096))
    prefill_enable_trace = enable_trace and max_seq_len < prefill_trace_max
    if enable_trace and not prefill_enable_trace:
        logger.info(
            f"Prefill trace disabled (max_seq_len={max_seq_len} >= {prefill_trace_max}); "
            f"decode stays traced. Set GEMMA4_PREFILL_TRACE_MAX_SEQ to override."
        )
    logger.info("Warming up prefill...")
    generator.warmup_model_prefill(
        kv_cache=tt_kv_cache,
        enable_trace=prefill_enable_trace,
        can_sample_on_device=False,
        greedy_only=True,
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
        enable_trace=prefill_enable_trace,
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
    ttft_ms = total_prefill * 1000
    amortized_prefill_ms = total_prefill / batch_size * 1000
    decode_tps_u = steady_iters / total_decode if total_decode > 0 else 0
    decode_tps = decode_tps_u * batch_size

    logger.info("")
    logger.info("=== Performance metrics ===")
    logger.info(f"Prompt tokens: {prefill_lens[0]}, generated tokens: {iteration}")
    logger.info(f"Time to First Token (TTFT): {ttft_ms:.1f} ms")
    if batch_size > 1:
        logger.info(f"Amortized prefill/user: {amortized_prefill_ms:.1f} ms")
    if decode_tps_u > 0:
        logger.info(
            f"Decode: {1000 / decode_tps_u:.2f} ms/token @ {decode_tps_u:.2f} tok/s/user "
            f"({decode_tps:.2f} tok/s throughput)"
        )
    else:
        # No steady-state decode timing (e.g. EoS hit on the first token, so only
        # the compile iteration ran) — avoid dividing by zero.
        logger.info("Decode: n/a (no steady-state decode iterations recorded)")
    logger.info(f"Full demo runtime: {profiler.get_duration('run'):.1f} s")

    if is_ci_env:
        measurements = {
            "inference_prefill": total_prefill,
            "inference_decode": total_decode,
            "prefill_time_to_token": total_prefill,
            "prefill_time_to_token_per_user_amortized": total_prefill / batch_size,
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


# ══════════════════════════════════════════════════════════════════════════
# Speculative decoding (Gemma4 it-assistant drafter), batch=1
# ══════════════════════════════════════════════════════════════════════════


def _run_spec_decode(
    prompt,
    instruct,
    max_seq_len,
    max_generated_tokens,
    page_params,
    sampling_params,
    mesh_device,
    enable_trace=False,
    draft_len=None,
    num_layers=None,
):
    """Single-user speculative decode: target verifies the it-assistant drafter.

    The drafter defaults to ``<HF_MODEL>-assistant`` (e.g. HF_MODEL=
    google/gemma-4-12B-it -> google/gemma-4-12B-it-assistant); override with
    GEMMA4_ASSISTANT_MODEL. Greedy spec-decode is token-identical to greedy
    decode; we report the acceptance rate and decode tok/s/u (throughput).
    """
    import math
    import time

    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder

    model_path = _model_path()
    assistant_path = os.getenv("GEMMA4_ASSISTANT_MODEL")
    if not assistant_path:
        # Default to the matching it-assistant drafter so the demo runs without
        # an explicit env (e.g. google/gemma-4-12B-it -> ...-it-assistant).
        assistant_path = f"{model_path}-assistant"
        logger.info(f"GEMMA4_ASSISTANT_MODEL unset; defaulting drafter to {assistant_path}")
    temperature = sampling_params.get("temperature", 0)
    top_p = sampling_params.get("top_p", 1.0)
    top_k = sampling_params.get("top_k", 0)
    if draft_len is None:
        draft_len = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 3))
    batch_size = 1

    block_size = page_params["page_block_size"]
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=batch_size * math.ceil(max_seq_len / block_size)
    )

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,  # spec-decode needs unbounded sliding KV
    )
    target = generator.model[0]
    model_args = generator.model_args

    page_table = create_tt_page_table(batch_size, paged_attention_config)

    # Prefill tracing has ~no perf gain and OOMs the trace region at long context
    # (≥4K); gate it off above a threshold (decode/spec traces stay on).
    prefill_trace_max = int(os.environ.get("GEMMA4_PREFILL_TRACE_MAX_SEQ", 4096))
    prefill_enable_trace = enable_trace and max_seq_len < prefill_trace_max
    generator.warmup_model_prefill(
        kv_cache=tt_kv_cache, enable_trace=prefill_enable_trace, can_sample_on_device=False, greedy_only=True
    )

    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
    )
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

    logger.info("Spec-decode prefill...")
    prefill_t0 = time.perf_counter()
    prefill_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        warmup_prefill=False,
        enable_trace=prefill_enable_trace,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_elapsed = time.perf_counter() - prefill_t0
    if hasattr(prefill_logits, "deallocate"):
        prefill_logits.deallocate(True)

    prompt_len = int(decoding_pos[0])
    anchor_pos = prompt_len - 1
    anchor_token = int(encoded_prompts[0][anchor_pos])

    # Spec-decode drafts `draft_len` positions AHEAD of the committed position, so
    # the furthest position touched is (prompt_len-1) + generated + draft_len. The
    # RoPE / paged-attention structures are sized to max_seq_len, so overshooting
    # that bound indexes out of range and hangs the device (deterministically at
    # cur_pos == max_seq_len - draft_len). Reserve the speculative lookahead margin
    # by clamping generation to stay strictly within max_seq_len.
    _safe_gen = max_seq_len - prompt_len - (draft_len + 1)
    if max_generated_tokens > _safe_gen:
        logger.warning(
            f"Clamping max_generated_tokens {max_generated_tokens} -> {max(1, _safe_gen)} to keep "
            f"spec lookahead (draft_len={draft_len}) within max_seq_len={max_seq_len} "
            f"(prompt_len={prompt_len}); raise max_seq_len for more generated tokens."
        )
        max_generated_tokens = max(1, _safe_gen)

    # Load the assistant only after target prefill warmup/prefill is complete.
    # Loading it earlier makes the target prefill trace capture run with extra
    # assistant tensors resident and has been observed to trigger runtime
    # profiler sync timeouts in the speculative path while the plain path stays
    # clean.
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=assistant_path,
    )

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=draft_len,
    )

    # Greedy uses the fully on-device fused iteration (argmax + re-embed on
    # device, only 2K+1 ids read back per iter). With GEMMA4_SPEC_TRACE=1 the
    # whole iteration is ONE metal trace replayed per step (K draft steps +
    # verify fused — avoids the distinct-CCL-trace interleave deadlock). Sampling
    # (temp>0) falls back to the host-readback generate for batch=1.
    use_fused = batch_size == 1 and ((not temperature) or temperature <= 0)
    # The fused greedy path is HOST-DISPATCH bound when untraced (~10 tok/s/u —
    # SLOWER than plain decode); the single fused Metal trace removes that
    # overhead (>3x, exceeding plain decode). Default tracing to the demo's
    # `enable_trace` so spec-decode is fast out of the box; GEMMA4_SPEC_TRACE
    # overrides explicitly (=1 force on, =0 force off — e.g. to A/B the cost).
    if use_fused:
        _trace_env = os.environ.get("GEMMA4_SPEC_TRACE")
        spec._use_trace = enable_trace if _trace_env is None else (_trace_env == "1")
    logger.info(
        f"Spec-decode generate (draft_len={draft_len}, temp={temperature}, "
        f"path={'fused' if use_fused else 'host'}, trace={spec._use_trace}, "
        f"seed={'reseed' if spec._fused_reseed else 'shift'}, "
        f"shift_seed={getattr(spec, '_fused_shift_seed', 'n/a')})..."
    )
    t0 = time.time()
    if use_fused:
        generated, accepts = spec.generate_fused(
            anchor_token=anchor_token, anchor_pos=anchor_pos, max_new_tokens=max_generated_tokens
        )
    else:
        generated, accepts = spec.generate(
            anchor_token=anchor_token,
            anchor_pos=anchor_pos,
            max_new_tokens=max_generated_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    elapsed = time.time() - t0

    text = tokenizer.decode(generated)
    n_tokens = len(generated)
    n_iters = len(accepts)
    mean_accept = (sum(accepts) / n_iters) if n_iters else 0.0
    setup_elapsed = getattr(spec, "_last_fused_setup_s", 0.0) if use_fused else 0.0
    steady_elapsed = getattr(spec, "_last_fused_replay_s", elapsed) if use_fused else elapsed
    # batch=1 single-user: per-user rate == aggregate throughput (kept explicit
    # so the line is coherent with the plain-decode demo's metric format). Match
    # the plain demo's steady-state decode convention by excluding one-time spec
    # setup/trace capture from the main Decode line; report wall throughput too.
    tok_s_u = n_tokens / steady_elapsed if steady_elapsed > 0 else 0.0
    tok_s = tok_s_u * batch_size
    ms_per_token = (steady_elapsed * 1000.0 / n_tokens) if n_tokens else 0.0
    ms_per_iter = (steady_elapsed * 1000.0 / n_iters) if n_iters else 0.0
    wall_tok_s_u = n_tokens / elapsed if elapsed > 0 else 0.0

    logger.info(f"\n== SPEC-DECODE GENERATION ==\n{text.strip()}\n")
    logger.info("=== Speculative decoding metrics ===")
    logger.info(f"Prompt tokens: {prompt_len}, generated tokens: {n_tokens}")
    logger.info(f"Time to First Token (TTFT): {prefill_elapsed * 1000.0 / batch_size:.1f} ms")
    logger.info(
        f"Drafter: {draft_len} drafts/iter; mean accepted {mean_accept:.2f}/{draft_len} (tokens/iter: {mean_accept + 1:.2f})"
    )
    if setup_elapsed > 0:
        logger.info(
            f"Spec setup/trace capture: {setup_elapsed:.2f}s (wall decode incl. setup: {wall_tok_s_u:.2f} tok/s/user)"
        )
    logger.info(f"Verify iterations: {n_iters} ({ms_per_iter:.2f} ms/iter)")
    logger.info(f"Decode: {ms_per_token:.2f} ms/token @ {tok_s_u:.2f} tok/s/user " f"({tok_s:.2f} tok/s throughput)")
    assert n_tokens > 0, "speculative decode produced no tokens"
    return generated, accepts


def _run_spec_decode_batched(
    prompts,
    instruct,
    max_seq_len,
    max_generated_tokens,
    page_params,
    sampling_params,
    mesh_device,
    enable_trace,
    draft_len=None,
    num_layers=None,
):
    """Batched (B>1) greedy speculative decode: B independent users, one shared
    batched packed verify per iteration (KV-amortization), ragged per-user
    acceptance. Untraced (host-dispatch bound) — the device win is the batched
    verify; tracing the ragged loop is a follow-up.
    """
    import math
    import time

    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    B = len(prompts)
    model_path = _model_path()
    assistant_path = os.getenv("GEMMA4_ASSISTANT_MODEL")
    if not assistant_path:
        assistant_path = f"{model_path}-assistant"
        logger.info(f"GEMMA4_ASSISTANT_MODEL unset; defaulting drafter to {assistant_path}")
    temperature = sampling_params.get("temperature", 0)
    if temperature and temperature > 0:
        pytest.skip("batched spec-decode supports greedy only (set temperature=0)")
    if draft_len is None:
        draft_len = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 3))

    block_size = page_params["page_block_size"]
    blocks_per_user = math.ceil(max_seq_len / block_size)
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=B * blocks_per_user)

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=B,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,  # spec-decode needs unbounded sliding KV
    )
    target = generator.model[0]
    model_args = generator.model_args

    page_table = create_tt_page_table(B, paged_attention_config)  # [B, blocks_per_user]

    # Prefill tracing has ~no perf gain and OOMs the trace region at long context
    # (≥4K); gate it off above a threshold (the batched decode trace stays on).
    prefill_trace_max = int(os.environ.get("GEMMA4_PREFILL_TRACE_MAX_SEQ", 4096))
    prefill_enable_trace = enable_trace and max_seq_len < prefill_trace_max
    generator.warmup_model_prefill(
        kv_cache=tt_kv_cache, enable_trace=prefill_enable_trace, can_sample_on_device=False, greedy_only=True
    )

    # Per-user prefill into each user's own KV blocks (prompts have distinct lengths).
    logger.info(f"Spec-decode batched prefill for B={B} users...")
    anchor_tokens, anchor_positions, prompt_lens = [], [], []
    prefill_t0 = time.perf_counter()
    for b in range(B):
        in_pt, encoded, decoding_pos, p_lens = preprocess_inputs_prefill(
            [prompts[b]], tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
        )
        in_pt = torch.stack(in_pt).view(1, -1)
        prefill_logits = generator.prefill_forward_text(
            in_pt,
            page_table=page_table[b : b + 1],
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            warmup_prefill=False,
            enable_trace=prefill_enable_trace,
        )
        if hasattr(prefill_logits, "deallocate"):
            prefill_logits.deallocate(True)
        prompt_lens.append(int(decoding_pos[0]))
        anchor_positions.append(int(decoding_pos[0]) - 1)
        anchor_tokens.append(int(encoded[0][int(decoding_pos[0]) - 1]))
    ttnn.synchronize_device(mesh_device)
    prefill_elapsed = time.perf_counter() - prefill_t0

    # Clamp generation so the furthest spec position (pos + draft_len) stays in range.
    max_prompt = max(prompt_lens)
    _safe_gen = max_seq_len - max_prompt - (draft_len + 1)
    if max_generated_tokens > _safe_gen:
        logger.warning(
            f"Clamping max_generated_tokens {max_generated_tokens} -> {max(1, _safe_gen)} to keep spec "
            f"lookahead (draft_len={draft_len}) within max_seq_len={max_seq_len} (max prompt_len={max_prompt})."
        )
        max_generated_tokens = max(1, _safe_gen)

    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=assistant_path,
        max_local_batch_size=B,
    )

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=draft_len,
    )
    # The whole batched iteration (batched drafter chain + batched packed verify)
    # is captured as ONE metal trace and replayed per step; the one-time capture
    # (setup) is excluded from the steady decode rate. Prefill is NEVER traced.
    # Default tracing to the demo's enable_trace; GEMMA4_SPEC_TRACE overrides.
    _trace_env = os.environ.get("GEMMA4_SPEC_TRACE")
    spec._use_trace = enable_trace if _trace_env is None else (_trace_env == "1")

    logger.info(f"Spec-decode batched generate (B={B}, draft_len={draft_len}, greedy, trace={spec._use_trace})...")
    t0 = time.time()
    outs, accepts = spec.generate_batched(
        anchor_tokens=anchor_tokens,
        anchor_positions=anchor_positions,
        max_new_tokens=max_generated_tokens,
        max_seq_len=max_seq_len,
        temperature=0.0,
    )
    ttnn.synchronize_device(mesh_device)
    elapsed = time.time() - t0

    total_tokens = sum(len(o) for o in outs)
    all_accepts = [m for a in accepts for m in a]
    mean_accept = (sum(all_accepts) / len(all_accepts)) if all_accepts else 0.0
    # Steady decode excludes one-time trace capture (mirrors the single-user path).
    setup_s = getattr(spec, "_last_fused_setup_s", 0.0) if spec._use_trace else 0.0
    steady_s = getattr(spec, "_last_fused_replay_s", elapsed) if spec._use_trace else elapsed
    tok_s = total_tokens / steady_s if steady_s > 0 else 0.0

    logger.info("\n== BATCHED SPEC-DECODE GENERATION ==")
    for b in range(B):
        logger.info(f"[user {b}] {tokenizer.decode(outs[b]).strip()}")
    logger.info("=== Batched speculative decoding metrics ===")
    logger.info(f"Users (batch): {B}; prompt tokens (max): {max_prompt}; total generated tokens: {total_tokens}")
    logger.info(f"Time to First Token (TTFT, mean prefill/user): {prefill_elapsed * 1000.0 / B:.1f} ms")
    logger.info(
        f"Drafter: {draft_len} drafts/iter; mean accepted {mean_accept:.2f}/{draft_len} "
        f"(tokens/iter: {mean_accept + 1:.2f})"
    )
    if setup_s > 0:
        logger.info(f"Spec setup/trace capture: {setup_s:.2f}s (excluded from steady rate)")
    logger.info(
        f"Decode: {steady_s:.2f}s steady @ {tok_s:.2f} tok/s aggregate, {tok_s / B:.2f} tok/s/user "
        f"({'traced' if spec._use_trace else 'untraced'})"
    )
    assert total_tokens > 0, "batched speculative decode produced no tokens"
    return outs, accepts


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
def test_demo_spec_decode(mesh_device, reset_seeds):
    """Speculative-decode demo (target HF_MODEL + drafter GEMMA4_ASSISTANT_MODEL)."""
    _run_spec_decode(
        prompt=os.getenv("GEMMA4_SPEC_PROMPT", "Tell me about the history of computing in three sentences."),
        instruct=True,
        max_seq_len=int(os.environ.get("GEMMA4_MAX_SEQ_LEN", 4096)),
        max_generated_tokens=int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", 200)),
        page_params={"page_block_size": 64, "page_max_num_blocks": 2048},
        sampling_params={"temperature": float(os.environ.get("GEMMA4_TEMPERATURE", 0.0)), "top_p": 0.95, "top_k": 64},
        mesh_device=mesh_device,
    )
