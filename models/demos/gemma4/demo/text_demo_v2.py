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
  * On-device sampling by default (``GEMMA4_HOST_SAMPLE=0``) whenever the model
    exposes a sampling module — every mesh from 1x1 up, since the shard-width cap
    is now ``_MAX_SAMPLING_SHARD_WIDTH`` (256K) rather than 64K-and-TP>1.
    Host sampling all-gathers the full-vocab logits and reads them to CPU each
    token, which costs real time per step for token-for-token identical output.
    Set ``GEMMA4_HOST_SAMPLE=1`` to force the host path.
  * Decode token reads are pipelined one step deep (``GEMMA4_DECODE_PIPELINE=1``,
    default): the sampled token's DMA overlaps the next decode submit instead of
    blocking it. The step is device-bound, so this matters most where the model is
    small relative to the host round trip.
    ``GEMMA4_DECODE_PIPELINE=0`` restores the blocking loop.
  * No decode warmup (``warmup_model_decode`` is Gemma3-generator specific); the
    first decode iteration serves as the compile step and is excluded from the
    reported steady-state perf (matching the benchmark warmup convention).

All model-level optimizations (BFP8 weights via precision_overrides.json,
width-sharded RMSNorm, row-major RoPE caches, nlp_concat_heads_decode) are
applied automatically because they live in the shared model code that
``Gemma4Generator.from_pretrained`` builds.

Usage:
    HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=P150x8 pytest \
        models/demos/gemma4/tests/e2e/test_isl_sweep.py -k "batch-1" -sv

    # Long-context (defaults pick bounded/chunk for coherency):
    MESH_DEVICE=P150x8 HF_MODEL=google/gemma-4-31B-it pytest \
        models/demos/gemma4/tests/e2e/test_isl_sweep.py -k "long-context-128k" -s --timeout 1800

    # Override prompts / lengths from the CLI:
    HF_MODEL=google/gemma-4-31B-it pytest \
        models/demos/gemma4/tests/e2e/test_isl_sweep.py -k "batch-1" -sv \
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
from models.demos.gemma4.demo.sampling_utils import (
    build_device_sampling_params,
    device_tracks_decode_on_device,
    log_sampling_mode,
    model_can_sample_on_device,
)
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.generator_trace import resolve_gemma4_demo_long_context
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill
from models.tt_transformers.tt.model_config import determine_device_name

_CONTEXT_CACHE_DIR = Path("models/tt_transformers/demo/context_cache")

_MESH_DEVICE_SHAPES = {
    # Logical SKU names (same mapping as tt_transformers / gemma3 demos).
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P300x2": (1, 4),
    "P300X2": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}


def _mesh_device_param():
    """Resolve mesh shape from ``MESH_DEVICE`` as a hardcoded (rows, cols) tuple.

    Named SKUs map explicitly (so ``2x4`` / ``BHGLX`` stay DP layouts). Unset /
    unknown → ``(1, N)`` over all visible devices so a LoudBox opens full 1×8
    TP instead of a 4-chip subset. Set ``MESH_DEVICE`` for non-line meshes.
    """
    env = os.environ.get("MESH_DEVICE")
    if env in _MESH_DEVICE_SHAPES:
        return _MESH_DEVICE_SHAPES[env]
    if env and "x" in env.lower():
        try:
            rows, cols = env.lower().split("x", 1)
            return (int(rows), int(cols))
        except ValueError:
            pass
    try:
        n = len(ttnn.get_device_ids())
    except Exception:
        n = 4
    return (1, max(1, n))


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


# Tokens detokenized for the per-step progress line. The line keeps at most 97
# characters, and a token averages ~4, so this tail always covers it while making
# the decode-loop detokenize O(1) per step instead of O(tokens generated).
_LOG_TAIL_TOKENS = 48


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


def _prepare_demo_prefill_warmup(
    *,
    generator,
    tt_kv_cache,
    sampling_params,
    enable_trace,
    max_seq_len,
    model_args_list,
    batch_size,
    input_prompts=None,
):
    """Shared prefill trace buckets + on-device sampling warmup for demo paths."""
    from models.demos.gemma4.tt.generator_trace import (
        chunked_prefill_trace_enabled,
        enable_single_chunk_demo_prefill_trace_bucket,
        reset_trace_prefill_seq_lens_to_default,
        trim_demo_prefill_trace_buckets,
    )

    reset_trace_prefill_seq_lens_to_default()
    if input_prompts is not None:
        trim_demo_prefill_trace_buckets(input_prompts=input_prompts, max_seq_len=max_seq_len)

    prefill_trace_max = int(os.environ.get("GEMMA4_PREFILL_TRACE_MAX_SEQ", 4096))
    prefill_enable_trace = enable_trace and (max_seq_len < prefill_trace_max or chunked_prefill_trace_enabled())
    if enable_trace and not prefill_enable_trace:
        logger.info(
            f"Prefill trace disabled (max_seq_len={max_seq_len} >= {prefill_trace_max}); "
            f"decode stays traced. Set GEMMA4_PREFILL_TRACE_MAX_SEQ or "
            f"GEMMA4_CHUNKED_PREFILL_TRACE=1 to override."
        )
    if prefill_enable_trace:
        enable_single_chunk_demo_prefill_trace_bucket(
            max_seq_len=max_seq_len,
            max_prefill_chunk_size=int(getattr(model_args_list[0], "max_prefill_chunk_size", 0) or 0),
            model_args_list=model_args_list,
            batch_size=batch_size,
        )

    force_host = os.environ.get("GEMMA4_HOST_SAMPLE", "0").lower() in ("1", "true", "yes")
    can_sample = (not force_host) and model_can_sample_on_device(generator.model[0])
    device_sampling_params = build_device_sampling_params(sampling_params, can_sample=can_sample)
    temperature = sampling_params.get("temperature", 0)
    greedy_only = temperature <= 0
    log_sampling_mode(can_sample, sampling_params)

    logger.info("Warming up prefill...")
    generator.warmup_model_prefill(
        kv_cache=tt_kv_cache,
        enable_trace=prefill_enable_trace,
        can_sample_on_device=can_sample,
        greedy_only=greedy_only,
    )
    logger.info("Warmup complete")

    return prefill_enable_trace, device_sampling_params


def _run_demo_prefill(
    *,
    generator,
    input_tokens_prefill_pt,
    page_table,
    tt_kv_cache,
    decoding_pos,
    prefill_enable_trace,
    device_sampling_params,
    temperature,
    top_p,
):
    """Run measured prefill with the same sampling path as ``run_demo_text``."""
    import time

    logger.info("Starting prefill...")
    prefill_t0 = time.perf_counter()
    prefill_out = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        warmup_prefill=False,
        enable_trace=prefill_enable_trace,
        sampling_params=device_sampling_params,
    )
    if device_sampling_params is not None:
        prefill_tokens, _ = prefill_out
        prefilled_token = prefill_tokens.long()
    else:
        prefilled_token = _host_sample(prefill_out, temperature, top_p)
    prefill_elapsed = time.perf_counter() - prefill_t0
    logger.info("Prefill finished")
    return prefilled_token, prefill_out, prefill_elapsed


def _default_ccl_packet_bytes():
    """See :func:`models.demos.gemma4.tt.ccl.default_ccl_packet_bytes`."""
    from models.demos.gemma4.tt.ccl import default_ccl_packet_bytes

    return default_ccl_packet_bytes()


def _device_params():
    """Blackhole needs a larger trace region; CQ count is env-tunable.

    Default ``num_command_queues=1`` (safe with host or on-device sampling).
    Set ``GEMMA4_NUM_CQS=2`` for serving/H2D-bound workloads once sampling stays
    on-device (Phase D3); measure batched users, not single-stream 128k TTFT.
    ``GEMMA4_TRACE_REGION_SIZE`` overrides the BH trace budget.

    CCL residual knobs (set before pytest collection):
      ``GEMMA4_FABRIC=ring`` → ``FABRIC_1D_RING`` (default ``1d``; ring
        regressed TTFT ~28.8s→~30.9s on 31B/P150x8 — leave off).
      ``GEMMA4_CCL_PACKET_BYTES`` → FabricRouterConfig max payload.
        BH defaults: 5376 (31B) / 3840 (12B) to match CCL page packing.
        Set ``0`` / ``none`` / ``default`` to keep Fabric's default.
    ``l1_small_size`` is set so all_gather semaphores land in L1_SMALL (avoids
    fragmenting the main L1 pool).
    """
    num_cqs = max(1, int(os.environ.get("GEMMA4_NUM_CQS", "1")))
    fabric_env = os.environ.get("GEMMA4_FABRIC", "1d").strip().lower()
    if fabric_env in ("ring", "1d_ring", "fabric_1d_ring"):
        fabric_config = ttnn.FabricConfig.FABRIC_1D_RING
    else:
        fabric_config = ttnn.FabricConfig.FABRIC_1D

    params = {
        "fabric_config": fabric_config,
        "num_command_queues": num_cqs,
        # CCL all_gather allocates semaphores in L1_SMALL when this is > 0.
        "l1_small_size": int(os.environ.get("GEMMA4_L1_SMALL_SIZE", 24576)),
    }
    # ``trace_region_size`` must cover the CUMULATIVE size of every captured
    # trace, not the largest one (the limit is hit at the last end_trace_capture,
    # usually decode). Batched demos add B=2/4 prefill traces plus a larger decode
    # graph; WH 96 MB is not enough for 31B batch-8/32. BH stays at 256 MB.
    #
    # Do NOT lower the WH value back to 30 MB. That was the pre-35e0798bdee
    # default and it fails at end_trace_capture with "Creating trace buffers of
    # size N ... but only 30000000B is allocated for trace region" -- 35e0798bdee
    # raised it to 64 MB, 6473437bb9c to 96 MB for auto traced multi-chunk
    # prefill, and d5292dd1b31 to 192 MB for batched prefill. 2736fc46354 (a
    # model-load fix) reverted all three by accident; restored here.
    default_trace_region = 256_000_000 if is_blackhole() else 192_000_000
    params["trace_region_size"] = int(os.environ.get("GEMMA4_TRACE_REGION_SIZE", default_trace_region))

    pkt_env = os.environ.get("GEMMA4_CCL_PACKET_BYTES")
    if pkt_env is None:
        pkt_bytes = _default_ccl_packet_bytes() if is_blackhole() else None
    elif pkt_env.strip().lower() in ("0", "none", "default", ""):
        pkt_bytes = None
    else:
        pkt_bytes = max(4352, int(pkt_env))
    if pkt_bytes is not None:
        router = ttnn.FabricRouterConfig()
        router.max_packet_payload_size_bytes = pkt_bytes
        params["fabric_router_config"] = router
    return params


def run_demo_text(
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
    enable_trace,
    num_layers=None,
):
    """Run the Generator-based text demo (ISL sweep rows live in tests/e2e/)."""
    import math

    max_generated_tokens = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", max_generated_tokens))
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", max_seq_len))
    if num_layers is None:
        _num_layers = os.environ.get("GEMMA4_NUM_LAYERS")
        num_layers = int(_num_layers) if _num_layers else None
    batch_size = int(os.environ.get("GEMMA4_BATCH", batch_size))
    _decode_trace = os.environ.get("GEMMA4_DECODE_TRACE")
    if _decode_trace is not None:
        enable_trace = _decode_trace.lower() in ("1", "true", "yes")
        logger.info(f"GEMMA4_DECODE_TRACE override: enable_trace={enable_trace}")

    model_path = _model_path()
    temperature = sampling_params.get("temperature", 0)
    top_p = sampling_params.get("top_p", 1.0)

    profiler = BenchmarkProfiler()
    profiler.start("run")

    # ── Inputs ────────────────────────────────────────────────────────────
    profiler.start("loading_inputs")
    prompts = load_inputs(input_prompts, batch_size, instruct)
    profiler.end("loading_inputs")

    # Right-size the paged KV pool.
    #   * batch=1 long-context: configs sometimes over-allocate (e.g. 2048 blocks
    #     for a 64k run); shrink to exactly batch * ceil(max_seq_len / block).
    #   * batch>1 throughput: the row's page_max_num_blocks is the tuned shared
    #     pool (short prompts). Using B*max_seq_len here over-provisions and OOMs
    #     (batch-32 @ 4096 → 4096 blocks vs config 1024).
    block_size = page_params["page_block_size"]
    needed_blocks = batch_size * math.ceil(max_seq_len / block_size)
    configured_blocks = page_params.get("page_max_num_blocks")
    if batch_size <= 1 or configured_blocks is None:
        page_max_num_blocks = needed_blocks
    else:
        page_max_num_blocks = configured_blocks
    paged_attention_config = (
        PagedAttentionConfig(block_size=block_size, max_num_blocks=page_max_num_blocks) if paged_attention else None
    )

    # Sliding-cache + prefill chunk from GEMMA4_LONG_CONTEXT_POLICY (model × device).
    # Override: GEMMA4_BOUNDED_SLIDING, GEMMA4_GEN_PREFILL_CHUNK.
    lc = resolve_gemma4_demo_long_context(max_seq_len, mesh_device, model_path, paged_attention=paged_attention)
    bounded_sliding = lc["bounded_sliding"]
    from models.demos.gemma4.tt.generator_trace import (
        reset_trace_prefill_seq_lens_to_default,
        trim_demo_prefill_trace_buckets,
    )

    reset_trace_prefill_seq_lens_to_default()
    trim_demo_prefill_trace_buckets(input_prompts=input_prompts, max_seq_len=max_seq_len)

    # ── Model (all optimizations applied inside create_tt_model) ───────────
    logger.info(
        f"Loading Gemma4 from {model_path} (layers={num_layers or 'all'}, max_seq_len={max_seq_len}, "
        f"bounded_sliding={bounded_sliding}, prefill_chunk={lc['prefill_chunk']}, "
        f"policy={lc['policy_source']})..."
    )
    profiler.start("model_load")
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
    )
    profiler.end("model_load")
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

    profiler.start("warmup_prefill")
    prefill_enable_trace, device_sampling_params = _prepare_demo_prefill_warmup(
        generator=generator,
        tt_kv_cache=tt_kv_cache,
        sampling_params=sampling_params,
        enable_trace=enable_trace,
        max_seq_len=max_seq_len,
        model_args_list=model_args_list,
        batch_size=batch_size,
        input_prompts=input_prompts,
    )
    profiler.end("warmup_prefill")

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
    prefill_out = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        warmup_prefill=False,
        enable_trace=prefill_enable_trace,
        sampling_params=device_sampling_params,
    )
    if device_sampling_params is not None:
        prefill_tokens, _ = prefill_out
        prefilled_token = prefill_tokens.long()
    else:
        prefilled_token = _host_sample(prefill_out, temperature, top_p)
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

    # Pipelined token readback: submit step j+1 before syncing step j's token.
    # Only possible with device sampling, where the sampled token is written
    # straight into the trace's token input buffer, so the next submit needs
    # nothing from host (see Generator._decode_forward_trace_text: reset_inputs
    # is False once decode is steady). The host loop then sits one step behind
    # and EOS is seen one step late — that extra token is discarded below, so
    # the emitted text is unchanged. GEMMA4_DECODE_PIPELINE=0 restores the
    # blocking loop.
    pipeline_reads = device_sampling_params is not None and os.environ.get("GEMMA4_DECODE_PIPELINE", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    device_tracks_pos = device_tracks_decode_on_device(
        generator.model[0],
        device_sampling=device_sampling_params is not None,
        enable_trace=enable_trace,
    )
    pending = []

    def _fold_tokens(toks, *, log_progress=True):
        """Fold one step's sampled tokens into the output; True to keep going."""
        toks = toks.long().view(batch_size, -1)
        keep_going = True
        for user in range(batch_size):
            tok = int(toks[user, 0])
            if tok not in tokenizer.stop_tokens and not user_done[user]:
                all_outputs[user].append(tok)
            elif stop_at_eos:
                user_done[user] = True
                if all(user_done):
                    keep_going = False
        if log_progress and not is_ci_env:
            for user in range(batch_size):
                # Detokenize only the tail that survives the clamp below. Decoding
                # the whole generated slice is O(generated) host work per token —
                # quadratic over a run — to print a line that keeps 97 characters.
                # (Decoding all_outputs whole was worse still: O(prompt) per token,
                # which at long context dominated the decode step itself.)
                # Runs inside the timed decode window, so this is measured time.
                generated = all_outputs[user][prefill_lens[user] :]
                text = tokenizer.decode(generated[-_LOG_TAIL_TOKENS:])
                if len(generated) > _LOG_TAIL_TOKENS or len(text) > 100:
                    text = "..." + text[-97:]
                logger.info(f"[User {user}] {text.replace(chr(10), ' ')}")
        return keep_going

    def _log_decode_progress():
        if is_ci_env:
            return
        for user in range(batch_size):
            generated = all_outputs[user][prefill_lens[user] :]
            text = tokenizer.decode(generated[-_LOG_TAIL_TOKENS:])
            if len(generated) > _LOG_TAIL_TOKENS or len(text) > 100:
                text = "..." + text[-97:]
            logger.info(f"[User {user}] {text.replace(chr(10), ' ')}")

    def _consume_tokens(host_out, read_events):
        """Wait for one pipelined read, then fold its tokens into the output."""
        for event in read_events:
            ttnn.event_synchronize(event)
        toks, _ = generator.process_decode_output_host(host_out, is_tokens=True)
        return _fold_tokens(toks, log_progress=False)

    logger.info(
        "Starting decode loop... (pipelined token reads: {}, device_pos_on_device: {})",
        pipeline_reads,
        device_tracks_pos,
    )
    profiler.start("inference_decode")
    while users_decoding:
        # One timer per loop pass, closed at the bottom. In the pipelined path a
        # pass is "submit step j, then sync step j-1", so the window still measures
        # one token of steady-state wall time. Ending it right after the submit
        # would time the enqueue only and report a fictitious tok/s.
        step = iteration
        profiler.start(f"inference_decode_time_{step}")
        decode_out = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=enable_trace,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=device_sampling_params,
            read_from_device=not pipeline_reads,
        )

        if pipeline_reads:
            # Start the DMA and record an event; do NOT wait on it. The device
            # already holds the token it needs for the next step.
            host_out, read_events = generator.read_decode_output(decode_out, async_read=True)
            pending.append((host_out, read_events))
        else:
            decode_out, _ = decode_out
            if device_sampling_params is not None:
                out_tok = decode_out.long().view(batch_size, 1)
            else:
                out_tok = _host_sample(decode_out, temperature, top_p)

        if not device_tracks_pos:
            current_pos += 1
        iteration += 1

        consumed = False
        if pipeline_reads:
            # One step of slack: the read issued last iteration has had a full
            # decode submit to land, so this sync is off the critical path.
            if len(pending) > 1:
                users_decoding = _consume_tokens(*pending.pop(0))
                consumed = True
        else:
            users_decoding = _fold_tokens(out_tok, log_progress=False)
            consumed = True

        profiler.end(f"inference_decode_time_{step}")
        if consumed and not is_ci_env:
            _log_decode_progress()
        if iteration >= max_generated_tokens:
            users_decoding = False

    # Drain the in-flight reads so the emitted text holds every submitted step.
    for entry in pending:
        _consume_tokens(*entry)
    pending.clear()
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
    logger.info(f"Model load: {profiler.get_duration('model_load'):.1f} s")
    logger.info(f"Prefill warmup: {profiler.get_duration('warmup_prefill'):.1f} s")
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
    input_prompts=None,
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
    model_args_list = model_args if isinstance(model_args, (list, tuple)) else [model_args]

    page_table = create_tt_page_table(batch_size, paged_attention_config)

    prefill_enable_trace, device_sampling_params = _prepare_demo_prefill_warmup(
        generator=generator,
        tt_kv_cache=tt_kv_cache,
        sampling_params=sampling_params,
        enable_trace=enable_trace,
        max_seq_len=max_seq_len,
        model_args_list=model_args_list,
        batch_size=batch_size,
        input_prompts=input_prompts,
    )

    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
    )
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

    logger.info("Spec-decode prefill...")
    _, prefill_out, prefill_elapsed = _run_demo_prefill(
        generator=generator,
        input_tokens_prefill_pt=input_tokens_prefill_pt,
        page_table=page_table,
        tt_kv_cache=tt_kv_cache,
        decoding_pos=decoding_pos,
        prefill_enable_trace=prefill_enable_trace,
        device_sampling_params=device_sampling_params,
        temperature=temperature,
        top_p=top_p,
    )
    if device_sampling_params is None and hasattr(prefill_out, "deallocate"):
        prefill_out.deallocate(True)

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
    # Capture the fused graph after prefill / assistant load, before the decode
    # timer. Decode: already excludes this; wall should too so 200-token demos
    # are not dominated by one-time compile. Setup is still logged separately.
    if use_fused and spec._use_trace:
        spec.prepare_fused_trace(anchor_token, anchor_pos)
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
    wall_elapsed = elapsed + (setup_elapsed if use_fused and spec._use_trace else 0.0)
    wall_tok_s_u = n_tokens / wall_elapsed if wall_elapsed > 0 else 0.0

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
    if spec._verify_time_s > 0 or spec._draft_time_s > 0:
        logger.info(
            f"Target verify: {spec._verify_time_s * 1000.0:.1f} ms total "
            f"({spec._verify_time_s * 1000.0 / n_iters:.2f} ms/iter)"
        )
        logger.info(
            f"MTP (drafter) parallel-token generation: {spec._draft_time_s * 1000.0:.1f} ms total "
            f"({spec._draft_time_s * 1000.0 / n_iters:.2f} ms/iter, {draft_len} tokens/iter)"
        )
    else:
        logger.info(
            "Target-verify/MTP time split not available: draft+verify ran as ONE fused Metal "
            "trace replay per iteration (GEMMA4_SPEC_TRACE=1/enable_trace), so the two phases "
            "aren't separately observable on the host. Set GEMMA4_SPEC_TRACE=0 for the breakdown."
        )
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
    input_prompts=None,
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
    model_args_list = model_args if isinstance(model_args, (list, tuple)) else [model_args]
    top_p = sampling_params.get("top_p", 1.0)

    page_table = create_tt_page_table(B, paged_attention_config)  # [B, blocks_per_user]

    prefill_enable_trace, device_sampling_params = _prepare_demo_prefill_warmup(
        generator=generator,
        tt_kv_cache=tt_kv_cache,
        sampling_params=sampling_params,
        enable_trace=enable_trace,
        max_seq_len=max_seq_len,
        model_args_list=model_args_list,
        batch_size=B,
        input_prompts=input_prompts,
    )

    # Per-user prefill into each user's own KV blocks (prompts have distinct lengths).
    logger.info(f"Spec-decode batched prefill for B={B} users...")
    anchor_tokens, anchor_positions, prompt_lens = [], [], []
    prefill_elapsed = 0.0
    for b in range(B):
        in_pt, encoded, decoding_pos, p_lens = preprocess_inputs_prefill(
            [prompts[b]], tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
        )
        in_pt = torch.stack(in_pt).view(1, -1)
        _, prefill_out, user_prefill_elapsed = _run_demo_prefill(
            generator=generator,
            input_tokens_prefill_pt=in_pt,
            page_table=page_table[b : b + 1],
            tt_kv_cache=tt_kv_cache,
            decoding_pos=decoding_pos,
            prefill_enable_trace=prefill_enable_trace,
            device_sampling_params=device_sampling_params,
            temperature=temperature,
            top_p=top_p,
        )
        prefill_elapsed += user_prefill_elapsed
        if device_sampling_params is None and hasattr(prefill_out, "deallocate"):
            prefill_out.deallocate(True)
        prompt_lens.append(int(decoding_pos[0]))
        anchor_positions.append(int(decoding_pos[0]) - 1)
        anchor_tokens.append(int(encoded[0][int(decoding_pos[0]) - 1]))

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
    n_batched_iters = max(max((len(a) for a in accepts), default=0), 1)
    if spec._verify_time_s > 0 or spec._draft_time_s > 0:
        logger.info(
            f"Target verify: {spec._verify_time_s * 1000.0:.1f} ms total "
            f"({spec._verify_time_s * 1000.0 / n_batched_iters:.2f} ms/iter)"
        )
        logger.info(
            f"MTP (drafter) parallel-token generation: {spec._draft_time_s * 1000.0:.1f} ms total "
            f"({spec._draft_time_s * 1000.0 / n_batched_iters:.2f} ms/iter, {draft_len} tokens/iter/user)"
        )
    else:
        logger.info(
            "Target-verify/MTP time split not available: draft+verify ran as ONE fused Metal "
            "trace replay per iteration (GEMMA4_SPEC_TRACE=1/enable_trace), so the two phases "
            "aren't separately observable on the host. Set GEMMA4_SPEC_TRACE=0 for the breakdown."
        )
    assert total_tokens > 0, "batched speculative decode produced no tokens"
    return outs, accepts


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [_mesh_device_param()],
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
