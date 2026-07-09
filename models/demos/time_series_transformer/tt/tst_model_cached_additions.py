# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
KV-cache single-token decode + N per-layer traces + optional 2CQ.

NOTE ON WHICH GENERATION PATH IS CANONICAL (added after session-end review):
  This module's generate_traced_cached() is the Stage 1 deliverable path.
  tst_model.py separately defines generate() (untraced reference) and
  generate_traced() (an earlier, KV-cache-less traced path with no 2CQ
  support — confirmed via grep: no use_2cq/num_command_queues/num_cqs
  anywhere in tst_model.py). generate_traced() and its TracedDecoderContext
  remain in tst_model.py because test_tst_e2e_traced.py still uses them as
  a real correctness gate (tracing the decoder stack preserves correctness
  vs. generate()), but generate_traced() is NOT benchmarked against any
  bounty Stage 1 target (throughput, sample-generation-under-1s, single
  -sequence-latency) and should not be confused with generate_traced_cached()
  below.

KEY CONSTRAINT (discovered from hardware error this session):
  ttnn.experimental.slice_write is a device WRITE and is forbidden inside
  begin_trace_capture / end_trace_capture.  The error is:
    TT_FATAL: Writes are not supported during trace capture. trace id: 1

  Fix: split each decode step into two phases:
    Phase 1 (untraced): extract Q/K/V from new token; slice_write K and V
                         into the caches at position `step`.
    Phase 2 (traced):   Q @ full_K_cache, softmax+mask, context @ full_V_cache,
                         cross-attn with precomputed KV, FFN, layernorms.

  This is logically equivalent to having the writes inside the trace because:
  - Before Phase 2 executes, k_cache[:,:,:,0..step] and v_cache[:,:,0..step,:]
    already contain the correct values (written in Phase 1).
  - The causal mask zeros out columns/rows beyond `step`, so the extra zero
    padding in positions step+1..T_max-1 has no effect on the output.
  - Phase 2 only reads the caches (pure compute), so it is traceable.

CORRECTNESS FIX (this revision -- confirmed via real hardware PCC run,
~0.31-0.33 against the HF reference at every step tested, and via code
inspection of the previous version of this file):

  The PREVIOUS version of this module captured ONE trace for the whole
  2-layer decoder stack, and computed EVERY layer's Q (inside the trace,
  via _extract_q_only) and EVERY layer's K/V (outside the trace, via
  _extract_and_write_kv) from `captured_dec_input` -- the raw single-token
  embedding. That is correct for layer 0, but WRONG for every layer after
  it: layer 1's self-attention Q/K/V must be projected from layer 0's real
  output hidden state, not from the raw input. Layer 1 was therefore
  attending over content it never actually received. This bug was invisible
  to every existing test in this repo: test_tst_pcc.py and test_tst_e2e.py
  only exercise the untraced generate()/run_decoder_step() path, and
  test_tst_e2e_traced.py only exercises the older, KV-cache-less
  generate_traced() -- neither ever calls generate_traced_cached() or
  build_traced_decoder_context_cached().

  FIX: one trace PER DECODER LAYER instead of one trace for the whole
  stack, chained through real device buffers:
    - Untraced: write layer i's K/V from layer i's REAL input
                (captured_dec_input for i=0, layer i-1's real trace
                output buffer for i>0).
    - Trace i:  layer i self-attn(from cache) + cross-attn + FFN. Q is
                computed INSIDE this trace via _extract_q_only on layer
                i's real input buffer, so every replay picks up whatever
                is actually in that buffer. Produces layer i's output at
                a FIXED device buffer address (same mechanism the
                original code already relied on for captured_dec_input
                and the old single traced_out).
    - Repeat for layer i+1, reading layer i's real output buffer.
    - After the last layer, its output buffer is read for the
      distribution head (ctx.traced_out now points at this buffer).

  This means num_decoder_layers separate trace captures and
  num_decoder_layers separate execute_trace calls per autoregressive step,
  instead of 1. More host-dispatch overhead per step than the (broken)
  single-trace version -- this works against the Stage 1 latency target
  and will need to be re-measured and possibly optimized (e.g. batching
  independent per-sample work, or investigating whether consecutive
  layer traces can be merged once a write-free formulation is found) once
  correctness is confirmed on hardware. Correctness comes first.

  THIS HAS NOT BEEN HARDWARE-VERIFIED YET. Before trusting output from
  generate_traced_cached() again:
    1. Re-run test_tst_perf.py::test_2cq_matches_single_queue_output --
       it already does real NaN/Inf/shape/distributional checks against
       this exact function and will catch gross breakage.
    2. Write and run a PCC-level gate (mirroring test_tst_e2e_traced.py's
       pattern, but calling generate_traced_cached()/
       build_traced_decoder_context_cached() instead of generate_traced())
       comparing per-step distribution params against generate(). No such
       test currently exists for this path -- that gap is exactly why the
       original bug shipped undetected.

Architecture:
  - num_decoder_layers separate traces, one per layer, captured once and
    reused across all `prediction_length` autoregressive steps.
  - Cross-attn KV precomputed once (encoder hidden is fixed).
  - CPU embedding prep per step (no device ops between trace replays).

NOTE ON A REVERTED EXPERIMENT (prior revision):
  A prior revision attempted to replace the manual Q@K_cache attend-from-cache
  below with ttnn.transformer.scaled_dot_product_attention_decode (FlashDecode)
  plus a cur_pos_tensor, to try to collapse per-step host overhead further.
  That was reverted: FlashDecode parallelizes one core (or core-group) per
  batch element ("The op parallelizes over b and K/V/Mask's s dimension." —
  tt-metal FlashDecode tech report), and hits
  `TT_FATAL: num_cores_available >= B` on an 8x7=56-core Wormhole grid once
  B = batch * num_parallel_samples exceeds the core count — which is exactly
  the regime the bounty's "100 samples in <1s" / "100+ seq/s" targets require.
  FlashDecode is built for small-batch/long-context LLM serving; this model's
  access pattern is the inverse (large effective batch, T_max=24 context), so
  it is not a fit here. The manual matmul attend-from-cache below is PCC-
  verified (0.9999927) against the FlashDecode path it replaces and has no
  batch-size ceiling, so it is restored as the Stage 1 implementation.

2CQ EVENT CHOREOGRAPHY:
  Per the tt-metal tech report "Advanced Performance Optimizations for Models"
  (tech_reports/AdvancedPerformanceOptimizationsForModels/
  AdvancedPerformanceOptimizationsForModels.md, sections 2.2/2.3/3.3 — this is
  the bounty issue's own primary reference), event creation and recording are
  the SAME call:

      event = ttnn.record_event(device, cq_id)
      ttnn.wait_for_event(cq_id, event)

  There is no separate ttnn.create_event() and no ttnn.event_synchronize() in
  this pattern.

  This revision follows the report's "ops + readback on CQ0, input writes on
  CQ1" layout (section 2.3.2 / 3.3.1.1), adapted to this model's per-step,
  PER-LAYER loop:
    - CQ1: copy_host_to_device_tensor for the step input, and the K/V
           slice_write cache updates (the "write" work) -- now once PER
           LAYER, not once per step, because layer i+1's write depends on
           layer i's real trace output.
    - CQ0: execute_trace (the "compute" work, pure read-from-cache) -- also
           once per layer.
  Two events, now recorded once per LAYER (not once per step):
    - `op_event` (recorded on CQ0 after each layer's execute_trace):
      signals CQ0 has finished producing that layer's output, so CQ1 may
      read it to write the NEXT layer's K/V (or, after the last layer,
      overwrite step k+1's input).
    - `write_event` (recorded on CQ1 after each layer's writes): signals
      CQ1 has finished writing this layer's K/V (and, for layer 0, the
      step input and mask), so CQ0 may run that layer's trace.
  IMPORTANT CAVEAT vs. the previous (single-trace, buggy) version: because
  layer i+1's write now genuinely depends on layer i's compute output,
  the per-step critical path is more serialized than before -- the old
  version could pipeline all-layers'-writes against nothing, this version
  cannot skip the write->compute->write->compute chain within a step.
  Cross-step overlap (layer 0's write for step k+1 against the last
  layer's compute for step k) should still be possible in principle since
  layer 0's input never depends on other layers, but this has NOT been
  verified. use_2cq=False remains the default until test_2cq_matches_
  single_queue_output has been re-run against THIS version and confirmed
  passing.
"""

import time

import torch
import torch.nn.functional as F

import ttnn

from .tst_attention import HEAD_DIM_PADDED, HEAD_DIM_TRUE, NUM_HEADS, allocate_kv_cache, precompute_cross_attn_kv
from .tst_decoder_layer import tst_ffn
from .tst_embedding import prepare_encoder_input
from .tst_model import (
    CONTEXT_LENGTH,
    D_MODEL,
    NUM_PARALLEL_SAMPLES,
    PADDED_WIDTH,
    PREDICTION_LENGTH,
    _apply_layernorm_ttnn,
    _build_static_feat_cpu,
    _distribution_head,
    _future_time_to_ttnn,
    _inputs_to_ttnn,
    _past_values_repeated_to_ttnn,
    _sample_next_step,
    run_encoder,
)
from .ttnn_utils import layer_norm_padded

_NEG_INF = -1e9

# ── Sub-op timing breakdown for write_prep (diagnostic only) ──────────────
_OP_TIMES = {}


def _reset_op_times():
    _OP_TIMES.clear()


def _accum(name, dt):
    _OP_TIMES[name] = _OP_TIMES.get(name, 0.0) + dt


def _print_op_times():
    if not _OP_TIMES:
        return
    parts = " ".join(f"{k}={v*1000:.1f}ms" for k, v in sorted(_OP_TIMES.items(), key=lambda kv: -kv[1]))
    print(f"[OP BREAKDOWN] {parts}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 helpers: extract Q/K/V and write K/V to cache (untraced)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_and_write_kv(
    hidden_1tok: ttnn.Tensor,  # [BS, 1, PADDED_WIDTH]
    w: dict,  # self_attn weights for one layer
    k_cache: ttnn.Tensor,  # [BS, H, D, T_max] ROW_MAJOR
    v_cache: ttnn.Tensor,  # [BS, H, T_max, D] ROW_MAJOR
    step: int,
    cq_id: int = 0,
) -> ttnn.Tensor:
    """
    Project hidden_1tok → Q/K/V, write K and V into caches at position `step`,
    return Q in tile layout for use in Phase 2 (traced attend).

    This runs OUTSIDE the trace capture (no writes allowed inside trace).

    cq_id: which command queue dispatches the linear/split/slice_write ops
    here. For the 2CQ path this should be the WRITER queue (CQ1), so these
    writes can be issued/run while CQ0 is still executing the previous
    step's trace.

    CALLER RESPONSIBILITY (post layer-threading fix): hidden_1tok must be
    THIS layer's real input -- captured_dec_input for layer 0, or the
    PREVIOUS layer's real trace output buffer for layer i>0. Passing the
    same raw embedding for every layer reproduces the confirmed
    layer-threading bug this file was rewritten to fix.
    """
    BS = hidden_1tok.shape[0]

    t0 = time.perf_counter()
    fused_qkv = ttnn.linear(hidden_1tok, w["qkv_weight"], bias=w["qkv_bias"], queue_id=cq_id)
    _accum("qkv_linear", time.perf_counter() - t0)

    t0 = time.perf_counter()
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        fused_qkv,
        num_heads=NUM_HEADS,
        queue_id=cq_id,
    )
    _accum("qkv_split", time.perf_counter() - t0)
    # query: [BS, H, 1, D]
    # key:   [BS, H, D, 1]  (already transposed by TTNN — matches k_cache's
    #         [B, H, D, T_max] layout directly, no permute needed)
    # value: [BS, H, 1, D]

    t0 = time.perf_counter()
    key_rm = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT, queue_id=cq_id)
    # V: only convert to ROW_MAJOR when v_cache is ROW_MAJOR (BS > 1, original
    # slice_write path). When BS == 1, v_cache is TILE_LAYOUT (see
    # allocate_kv_cache) and `value` is already TILE straight out of
    # split_query_key_value_and_split_heads -- no conversion needed, and
    # skipping it is the entire point of this path.
    use_update_cache_for_v = BS == 1
    if not use_update_cache_for_v:
        value_rm = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT, queue_id=cq_id)
    _accum("to_layout_kv", time.perf_counter() - t0)

    _step = [1, 1, 1, 1]
    t0 = time.perf_counter()
    ttnn.experimental.slice_write(
        key_rm,
        k_cache,
        [0, 0, 0, step],
        [BS, NUM_HEADS, HEAD_DIM_PADDED, step + 1],
        _step,
        queue_id=cq_id,
    )
    if use_update_cache_for_v:
        # BS == 1 only: ttnn.update_cache hard-asserts
        # input_tensor.padded_shape()[0] == 1 (update_cache_device_operation.cpp:53,
        # confirmed empirically -- B=4/32/33/100 all TT_FATAL on this check,
        # B=1 passes). Verified in isolation
        # (tests/diagnostics/probe_v_cache_update_cache*.py): exact match vs
        # slice_write ground truth across multiple steps, and correct behavior
        # when captured once and replayed inside a trace with update_idx frozen
        # as a Python int (same constraint slice_write's `step` already has
        # here) -- confirmed the trace re-reads `value`'s live contents on each
        # replay rather than baking in a stale reference. queue_id confirmed
        # accepted by ttnn.update_cache (tested against a 2-CQ device).
        ttnn.update_cache(v_cache, value, update_idx=step, batch_offset=0, queue_id=cq_id)
    else:
        ttnn.experimental.slice_write(
            value_rm,
            v_cache,
            [0, 0, step, 0],
            [BS, NUM_HEADS, step + 1, HEAD_DIM_PADDED],
            _step,
            queue_id=cq_id,
        )
    _accum("slice_write_kv", time.perf_counter() - t0)

    return query  # [BS, H, 1, D] TILE_LAYOUT


def _extract_q_only(
    hidden_1tok: ttnn.Tensor,
    w: dict,
) -> ttnn.Tensor:
    """
    Phase 1, trace-capture version: project hidden_1tok → Q/K/V but
    return Q only (K/V already written before begin_trace_capture).

    CALLER RESPONSIBILITY (post layer-threading fix): hidden_1tok is
    captured and re-read fresh on every trace replay -- it must be THIS
    layer's real input buffer (captured_dec_input for layer 0, the
    previous layer's real trace-output buffer for layer i>0), so that
    every replay computes Q from whatever real hidden state is currently
    sitting there.
    """
    fused_qkv = ttnn.linear(hidden_1tok, w["qkv_weight"], bias=w["qkv_bias"])
    query, _, _ = ttnn.transformer.split_query_key_value_and_split_heads(
        fused_qkv,
        num_heads=NUM_HEADS,
    )
    return query


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: traced attend-from-cache (pure compute, no writes)
# ─────────────────────────────────────────────────────────────────────────────


def _attend_from_cache(
    query: ttnn.Tensor,  # [BS, H, 1, D] TILE
    k_cache: ttnn.Tensor,  # [BS, H, D, T_max] ROW_MAJOR (read as TILE)
    v_cache: ttnn.Tensor,  # [BS, H, T_max, D] ROW_MAJOR when BS>1 (read as TILE); already TILE_LAYOUT when BS==1
    causal_mask_1tok: ttnn.Tensor,  # [1, 1, 1, T_max] TILE
    w: dict,
) -> ttnn.Tensor:
    """Pure compute: Q @ K_cache -> softmax -> V_cache -> out_proj. No writes."""
    k_tile = ttnn.to_layout(k_cache, ttnn.TILE_LAYOUT)  # [BS, H, D, T_max]
    scores = ttnn.matmul(query, k_tile)  # [BS, H, 1, T_max]
    scale = HEAD_DIM_TRUE**-0.5
    scaled = ttnn.multiply(scores, scale)
    masked = ttnn.add(scaled, causal_mask_1tok)
    probs = ttnn.softmax(masked, dim=-1)

    # v_cache is already TILE_LAYOUT when BS == 1 (see allocate_kv_cache) --
    # skip the conversion in that case, it's already the layout matmul needs.
    # For BS > 1, v_cache stays ROW_MAJOR and still needs this conversion,
    # unchanged from the original path.
    v_tile = (
        v_cache if v_cache.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(v_cache, ttnn.TILE_LAYOUT)
    )  # [BS, H, T_max, D]
    context = ttnn.matmul(probs, v_tile)  # [BS, H, 1, D]
    context = ttnn.transformer.concatenate_heads(context)  # [BS, 1, PADDED_WIDTH]
    return ttnn.linear(context, w["out_proj_weight"], bias=w["out_proj_bias"])


from .tst_attention import tst_cross_attention_with_kv  # noqa: E402  (used below)

# ─────────────────────────────────────────────────────────────────────────────
# Single-layer traced compute: reads caches (pre-written), no slice_write
# inside. THIS REPLACES THE OLD FUSED-2-LAYER _run_traced_decoder_stack --
# see module docstring "CORRECTNESS FIX" section for why the fused version
# was wrong.
# ─────────────────────────────────────────────────────────────────────────────


def _run_single_layer_compute(
    hidden_in: ttnn.Tensor,  # [BS, 1, PADDED_WIDTH] -- THIS layer's real input
    query: ttnn.Tensor,  # [BS, H, 1, D] -- Q projected from hidden_in
    k_cache: ttnn.Tensor,
    v_cache: ttnn.Tensor,
    k_pre: ttnn.Tensor,
    v_pre: ttnn.Tensor,
    causal_mask_1tok: ttnn.Tensor,  # [1, 1, 1, T_max]
    w: dict,  # THIS layer's full weight dict (self_attn/encoder_attn/fc.../ln...)
) -> ttnn.Tensor:
    """
    Pure-compute single decoder layer: self-attn(from cache) + cross-attn
    (from precomputed KV) + FFN, with residual/layernorm after each.
    No writes -- safe to trace.

    Each layer gets its OWN trace (see build_traced_decoder_context_cached).
    `hidden_in` and `query` must both be derived from the SAME real hidden
    state for this layer -- captured_dec_input for layer 0, the previous
    layer's real output buffer for layer i>0. This is what makes it
    possible for layer i>0 to actually receive layer i-1's output, instead
    of every layer independently re-deriving Q/K/V from the raw embedding
    (the bug this file was rewritten to fix).
    """
    self_attn_out = _attend_from_cache(query, k_cache, v_cache, causal_mask_1tok, w["self_attn"])
    residual = ttnn.add(hidden_in, self_attn_out)
    hidden = layer_norm_padded(
        residual,
        w["self_attn_layer_norm_weight"],
        w["self_attn_layer_norm_bias"],
        orig_dim=D_MODEL,
    )

    cross_out = tst_cross_attention_with_kv(hidden, k_pre, v_pre, w["encoder_attn"])
    residual = ttnn.add(hidden, cross_out)
    hidden = layer_norm_padded(
        residual,
        w["encoder_attn_layer_norm_weight"],
        w["encoder_attn_layer_norm_bias"],
        orig_dim=D_MODEL,
    )

    ffn_out = tst_ffn(hidden, w)
    residual = ttnn.add(hidden, ffn_out)
    hidden = layer_norm_padded(
        residual,
        w["final_layer_norm_weight"],
        w["final_layer_norm_bias"],
        orig_dim=D_MODEL,
    )

    return hidden  # [BS, 1, PADDED_WIDTH] -- fixed-address buffer for THIS layer


# ─────────────────────────────────────────────────────────────────────────────
# Causal mask helper
# ─────────────────────────────────────────────────────────────────────────────


def _update_causal_mask(device: ttnn.Device, mask_tensor: ttnn.Tensor, step: int, T_max: int, cq_id: int = 0) -> None:
    """
    Update the shared causal mask buffer in-place for the given step.

    PERF/CORRECTNESS FIX: previously hardcoded cq_id=0 unconditionally, so in
    the use_2cq=True path this write landed on the COMPUTE queue (CQ0)
    instead of the WRITER queue (CQ1/CQ_WRITE) that every other per-step
    write (input, K/V cache) uses. Callers now pass cq_id=CQ_WRITE
    explicitly, matching how the input buffer and K/V caches are already
    written on CQ_WRITE in run_traced_generation_cached.
    """
    t0 = time.perf_counter()
    mask = torch.full((1, 1, 1, T_max), _NEG_INF)
    mask[:, :, :, : step + 1] = 0.0
    new_tt = ttnn.from_torch(
        mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=None,
    )
    _accum("mask_build_host", time.perf_counter() - t0)
    t0 = time.perf_counter()
    ttnn.copy_host_to_device_tensor(new_tt, mask_tensor, cq_id=cq_id)
    _accum("mask_copy_to_device", time.perf_counter() - t0)


def _update_causal_mask_precomputed(
    device: ttnn.Device, mask_tensor: ttnn.Tensor, precomputed_host_tensor, cq_id: int = 0
) -> None:
    """Hot-loop version: copy an already-built host tensor, no torch work."""
    t0 = time.perf_counter()
    ttnn.copy_host_to_device_tensor(precomputed_host_tensor, mask_tensor, cq_id=cq_id)
    _accum("mask_copy_to_device", time.perf_counter() - t0)


def _build_causal_mask_1tok(device: ttnn.Device, step: int, T_max: int) -> ttnn.Tensor:
    """[1, 1, 1, T_max]: 0 for positions 0..step, NEG_INF beyond."""
    mask = torch.full((1, 1, 1, T_max), _NEG_INF)
    mask[:, :, :, : step + 1] = 0.0
    return ttnn.from_torch(
        mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _precompute_causal_masks_host(T_max: int) -> list:
    """
    PERF FIX: build all T_max host-side ttnn tensors ONCE here; per-step
    work becomes a single copy_host_to_device_tensor from an already-built
    host tensor, no torch construction in the hot loop.
    """
    masks = []
    for step in range(T_max):
        mask = torch.full((1, 1, 1, T_max), _NEG_INF)
        mask[:, :, :, : step + 1] = 0.0
        masks.append(ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None))
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# CPU single-token embedding prep
# ─────────────────────────────────────────────────────────────────────────────


def _prepare_dec_step_cpu_1tok(
    k: int,
    future_samples_so_far: list,
    past_values_cpu: torch.Tensor,
    future_time_cpu: torch.Tensor,
    static_feat_cpu: torch.Tensor,
    loc_cpu: torch.Tensor,
    scale_cpu: torch.Tensor,
    value_proj_cpu: torch.Tensor,
    dec_ln_w_cpu: torch.Tensor,
    dec_ln_b_cpu: torch.Tensor,
    pos_emb_cpu: torch.Tensor,
    context_length: int,
    T_max: int,
) -> torch.Tensor:
    """
    CPU: single-token decoder embedding for step k.
    Returns [BS, 1, PADDED_WIDTH] bfloat16.
    """
    BS = past_values_cpu.shape[0]
    lags = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 23, 24, 25, 35, 36, 37]

    if k == 0:
        future_vals_k = torch.zeros(BS, 1)
    else:
        prev = torch.stack(future_samples_so_far, dim=1)  # [BS, k]
        future_vals_k = torch.cat([prev, torch.zeros(BS, 1)], dim=1)

    full_seq = torch.cat([past_values_cpu, future_vals_k], dim=1)
    full_seq_scaled = (full_seq - loc_cpu.squeeze(-1)) / scale_cpu.squeeze(-1)

    full_len = full_seq_scaled.shape[1]
    lagged = torch.zeros(BS, 1, len(lags))
    for li, lag in enumerate(lags):
        src = full_len - 1 - lag
        if 0 <= src < full_len:
            lagged[:, 0, li] = full_seq_scaled[:, src]

    time_feat_k = future_time_cpu[:, k : k + 1, :]
    expanded_static = static_feat_cpu.unsqueeze(1)
    features = torch.cat([expanded_static, time_feat_k], dim=-1)
    transformer_in = torch.cat([lagged, features], dim=-1)

    emb = transformer_in @ value_proj_cpu
    pos = pos_emb_cpu[context_length + k : context_length + k + 1]
    emb = emb + pos.unsqueeze(0)
    emb = F.layer_norm(emb.float(), [D_MODEL], weight=dec_ln_w_cpu.float(), bias=dec_ln_b_cpu.float())

    pad_feat = PADDED_WIDTH - D_MODEL
    if pad_feat > 0:
        emb = F.pad(emb, (0, pad_feat))

    return emb.to(torch.bfloat16)  # [BS, 1, PADDED_WIDTH]


# ─────────────────────────────────────────────────────────────────────────────
# Traced decoder context
# ─────────────────────────────────────────────────────────────────────────────


class TracedDecoderContextCached:
    __slots__ = (
        "device",
        "trace_ids",
        "captured_dec_input",
        "layer_hidden_buffers",
        "traced_out",
        "kv_caches",
        "precomputed_kv",
        "enc_hidden_rep",
        "repeated_loc_tt",
        "repeated_scale_tt",
        "repeated_past_raw_tt",
        "repeated_future_time_tt",
        "repeated_static_cat_tt",
        "repeated_static_real_tt",
        "_lc",
        "_sc",
        "BS",
        "T_max",
        "dist_type",
        "num_decoder_layers",
        "_released",
        "shared_causal_mask",
        "precomputed_masks_host",
    )

    def release(self):
        if not self._released:
            for trace_id in self.trace_ids:
                ttnn.release_trace(self.device, trace_id)
            self._released = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()


def build_traced_decoder_context_cached(
    device,
    weights,
    past_values,
    past_time_features,
    future_time_features,
    past_observed_mask,
    static_categorical_features,
    static_real_features,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    num_parallel_samples=NUM_PARALLEL_SAMPLES,
) -> TracedDecoderContextCached:
    """
    One-time setup: encoder, cross-attn KV, KV-cache allocation, ONE trace
    PER DECODER LAYER (not one trace for the whole stack -- see module
    docstring "CORRECTNESS FIX" for why).

    Each layer's trace captures ONLY pure compute (attend-from-cache +
    cross-attn + FFN), reading Q computed inside the trace from that
    layer's REAL input. The slice_write cache updates for layer i happen
    outside trace i, immediately before it, in run_traced_generation_cached.
    """
    B = past_values.shape[0]
    S = num_parallel_samples
    BS = B * S
    T_max = prediction_length
    dt = weights.get("dist_type", "student_t")
    num_decoder_layers = sum(1 for k in weights if k.startswith("decoder.layers."))

    # ── Encoder ──────────────────────────────────────────────────────────
    pv_tt, pt_tt, pm_tt, sc_tt, sr_tt = _inputs_to_ttnn(
        device,
        past_values,
        past_time_features,
        past_observed_mask,
        static_categorical_features,
        static_real_features,
    )
    enc_emb, loc, scale = prepare_encoder_input(
        device,
        past_values=pv_tt,
        past_time_features=pt_tt,
        past_observed_mask=pm_tt,
        static_cat_features=sc_tt,
        static_real_features=sr_tt,
        cat_embedder_weight=weights["cat_embedder"],
        value_proj_weight=weights["encoder_value_proj"],
        pos_emb_weight=weights["encoder_pos_emb"],
        context_length=context_length,
    )
    enc_emb = _apply_layernorm_ttnn(enc_emb, weights["encoder_layernorm_ttnn"])
    encoder_hidden = run_encoder(device, enc_emb, weights)

    loc_t = ttnn.to_torch(loc).float()
    scale_t = ttnn.to_torch(scale).float()
    rep_loc = loc_t.repeat_interleave(S, dim=0)
    rep_scale = scale_t.repeat_interleave(S, dim=0)
    _sc = rep_scale.squeeze(-1).squeeze(-1)
    _lc = rep_loc.squeeze(-1).squeeze(-1)

    rep_past_tt = _past_values_repeated_to_ttnn(device, past_values.repeat_interleave(S, dim=0).float())
    rep_ftime_tt = _future_time_to_ttnn(device, future_time_features.repeat_interleave(S, dim=0))
    rep_scat_tt = ttnn.from_torch(
        static_categorical_features.repeat_interleave(S, dim=0).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    rep_sreal_tt = ttnn.from_torch(
        static_real_features.repeat_interleave(S, dim=0).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    rep_loc_tt = ttnn.from_torch(rep_loc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    rep_scale_tt = ttnn.from_torch(rep_scale, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    enc_hidden_t = ttnn.to_torch(encoder_hidden).float()
    enc_hidden_rep = ttnn.from_torch(
        enc_hidden_t.repeat_interleave(S, dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # ── Cross-attention KV (fixed) ────────────────────────────────────────
    precomputed_kv = []
    for i in range(num_decoder_layers):
        k_pre, v_pre = precompute_cross_attn_kv(enc_hidden_rep, weights[f"decoder.layers.{i}"]["encoder_attn"])
        precomputed_kv.append((k_pre, v_pre))

    # ── KV caches ─────────────────────────────────────────────────────────
    kv_caches = []
    for _ in range(num_decoder_layers):
        k_cache, v_cache = allocate_kv_cache(device, BS, T_max=T_max)
        kv_caches.append((k_cache, v_cache))

    # ── Persistent input buffer for layer 0 (allocated before any trace
    #    capture) ──────────────────────────────────────────────────────────
    captured_dec_input = ttnn.from_torch(
        torch.zeros(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ── Shared causal mask buffer (fixed address; contents updated per step
    #    via _update_causal_mask outside any trace) ─────────────────────────
    shared_causal_mask = _build_causal_mask_1tok(device, 0, T_max)
    precomputed_masks_host = _precompute_causal_masks_host(T_max)

    # ── Per-layer warmup + trace capture, chained through real buffers ─────
    # THIS is the fix: hidden_in starts as captured_dec_input (layer 0's
    # real input) and after each layer is captured, hidden_in is
    # reassigned to THAT layer's real output buffer -- so layer i+1's
    # warmup/capture genuinely reads layer i's real output, not the raw
    # embedding.
    trace_ids = []
    layer_hidden_buffers = []

    hidden_in = captured_dec_input
    for layer_idx in range(num_decoder_layers):
        w_layer = weights[f"decoder.layers.{layer_idx}"]
        w_self = w_layer["self_attn"]
        k_cache, v_cache = kv_caches[layer_idx]
        k_pre, v_pre = precomputed_kv[layer_idx]

        # Untraced warmup at step 0: write this layer's K/V from its REAL
        # input, then run the layer once untraced to compile kernels
        # before capture (mirrors the original single-trace warmup).
        q_warmup = _extract_and_write_kv(hidden_in, w_self, k_cache, v_cache, step=0, cq_id=0)
        _ = _run_single_layer_compute(hidden_in, q_warmup, k_cache, v_cache, k_pre, v_pre, shared_causal_mask, w_layer)
        ttnn.synchronize_device(device)

        # Capture this layer's trace. Q is computed INSIDE the trace via
        # _extract_q_only(hidden_in, ...) -- hidden_in is this layer's
        # real input (captured_dec_input for layer 0, the previous
        # layer's real output buffer for layer i>0).
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        q_traced = _extract_q_only(hidden_in, w_self)
        layer_out = _run_single_layer_compute(
            hidden_in, q_traced, k_cache, v_cache, k_pre, v_pre, shared_causal_mask, w_layer
        )
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        trace_ids.append(trace_id)
        layer_hidden_buffers.append(layer_out)
        hidden_in = layer_out  # next layer's real input

    # Reset KV caches to zero before actual generation
    _zero_kv_caches(kv_caches, device, BS, T_max)
    ttnn.synchronize_device(device)

    ctx = TracedDecoderContextCached()
    ctx.device = device
    ctx.trace_ids = trace_ids  # length == num_decoder_layers
    ctx.shared_causal_mask = shared_causal_mask
    ctx.precomputed_masks_host = precomputed_masks_host
    ctx.captured_dec_input = captured_dec_input
    ctx.layer_hidden_buffers = layer_hidden_buffers
    ctx.traced_out = layer_hidden_buffers[-1]  # final layer's real output
    ctx.kv_caches = kv_caches
    ctx.precomputed_kv = precomputed_kv
    ctx.enc_hidden_rep = enc_hidden_rep
    ctx.repeated_loc_tt = rep_loc_tt
    ctx.repeated_scale_tt = rep_scale_tt
    ctx.repeated_past_raw_tt = rep_past_tt
    ctx.repeated_future_time_tt = rep_ftime_tt
    ctx.repeated_static_cat_tt = rep_scat_tt
    ctx.repeated_static_real_tt = rep_sreal_tt
    ctx._lc = _lc
    ctx._sc = _sc
    ctx.BS = BS
    ctx.T_max = T_max
    ctx.dist_type = dt
    ctx.num_decoder_layers = num_decoder_layers
    ctx._released = False
    return ctx


def _zero_kv_caches(kv_caches, device, BS, T_max):
    """Zero-fill KV caches via slice_write (outside any trace).

    When BS == 1, v_cache is TILE_LAYOUT (see allocate_kv_cache) and is
    zeroed via ttnn.update_cache, called once per T_max position, writing
    IN PLACE into the SAME tensor object already referenced by any trace
    captured earlier in build_traced_decoder_context_cached. This must stay
    in-place: allocating a new device tensor here instead would zero a
    different buffer than the one execute_trace actually reads from later,
    silently breaking correctness without raising an error. K cache and,
    for BS > 1, V cache are unchanged: zeroed via slice_write exactly as
    before, which already writes into the given tensor in place.
    """
    for k_cache, v_cache in kv_caches:
        zero_k = ttnn.from_torch(
            torch.zeros(BS, NUM_HEADS, HEAD_DIM_PADDED, T_max, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        ttnn.experimental.slice_write(
            zero_k,
            k_cache,
            [0, 0, 0, 0],
            [BS, NUM_HEADS, HEAD_DIM_PADDED, T_max],
            [1, 1, 1, 1],
        )

        if v_cache.layout == ttnn.TILE_LAYOUT:
            # BS == 1 path: zero via update_cache, one call per position,
            # writing in place into the same v_cache tensor object.
            zero_v_1tok = ttnn.from_torch(
                torch.zeros(BS, NUM_HEADS, 1, HEAD_DIM_PADDED, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            for pos in range(T_max):
                ttnn.update_cache(v_cache, zero_v_1tok, update_idx=pos, batch_offset=0)
        else:
            zero_v = ttnn.from_torch(
                torch.zeros(BS, NUM_HEADS, T_max, HEAD_DIM_PADDED, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            ttnn.experimental.slice_write(
                zero_v,
                v_cache,
                [0, 0, 0, 0],
                [BS, NUM_HEADS, T_max, HEAD_DIM_PADDED],
                [1, 1, 1, 1],
            )


# ─────────────────────────────────────────────────────────────────────────────
# Generation loop
# ─────────────────────────────────────────────────────────────────────────────


def run_traced_generation_cached(
    ctx: TracedDecoderContextCached,
    weights: dict,
    context_length: int = CONTEXT_LENGTH,
    prediction_length: int = PREDICTION_LENGTH,
    use_2cq: bool = True,
) -> torch.Tensor:
    """
    Per step, per decoder layer i (0..num_layers-1):
      1. CPU (once per step, before the layer loop): build single-token
         embedding.
      2. Device: write this layer's input (raw embedding for i=0, the
         PREVIOUS layer's real output buffer for i>0) + slice_write K/V
         into layer i's cache + (i==0 only) update the shared causal mask.
      3. Device (traced): execute layer i's trace -- reads the
         just-written K/V cache, pure compute, writes layer i's fixed
         output buffer.
    After the last layer:
      4. Host: read the last layer's output buffer, sample from
         distribution.

    use_2cq=False: everything blocking on CQ0, strictly ordered layer by
                   layer -- no race possible by construction.
    use_2cq=True:  writes on CQ1, execute_trace on CQ0, handoff via
                   record_event/wait_for_event -- now once PER LAYER, not
                   once per step, because layer i+1's write genuinely
                   depends on layer i's real output. See module docstring
                   2CQ section for the caveat this implies. UNVERIFIED for
                   this multi-layer version -- re-run
                   test_2cq_matches_single_queue_output before trusting
                   use_2cq=True here.

    Returns [BS, prediction_length] float32.
    """
    device = ctx.device
    BS = ctx.BS
    T_max = ctx.T_max
    dt = ctx.dist_type
    num_layers = ctx.num_decoder_layers

    CQ_COMPUTE = 0  # runs execute_trace, always
    CQ_WRITE = 1 if use_2cq else 0  # writes input + KV cache updates + mask

    def _layer_input(i):
        """Layer i's real input buffer for the step currently being written."""
        return ctx.captured_dec_input if i == 0 else ctx.layer_hidden_buffers[i - 1]

    past_values_cpu = ttnn.to_torch(ctx.repeated_past_raw_tt).float()
    future_time_cpu = ttnn.to_torch(ctx.repeated_future_time_tt).float()
    loc_cpu = ttnn.to_torch(ctx.repeated_loc_tt).float()
    scale_cpu = ttnn.to_torch(ctx.repeated_scale_tt).float()
    static_cat_cpu = ttnn.to_torch(ctx.repeated_static_cat_tt).long()
    static_real_cpu = ttnn.to_torch(ctx.repeated_static_real_tt).float()
    cat_emb_w_cpu = ttnn.to_torch(weights["cat_embedder"]).float()
    value_proj_cpu = ttnn.to_torch(weights["decoder_value_proj"]).float()
    pos_emb_cpu = ttnn.to_torch(weights["decoder_pos_emb"]).float()
    dec_ln = weights["decoder_layernorm_ttnn"]
    dec_ln_w_cpu = ttnn.to_torch(dec_ln["weight"]).float().squeeze()
    dec_ln_b_cpu = ttnn.to_torch(dec_ln["bias"]).float().squeeze()

    static_feat_cpu = _build_static_feat_cpu(
        loc_cpu,
        scale_cpu,
        static_real_cpu,
        static_cat_cpu,
        cat_emb_w_cpu,
    )

    _reset_op_times()
    future_samples = []
    cpu_prep_time = 0.0
    device_enqueue_time = 0.0
    write_prep_time = 0.0
    trace_exec_time = 0.0
    readback_time = 0.0
    sample_time = 0.0

    op_event = None
    if use_2cq:
        # Warmup pass through every layer's write at step 0, on CQ_WRITE,
        # so the first real step's kernels are already resident there
        # (same rationale as the original single-trace warmup).
        for layer_idx in range(num_layers):
            w_self = weights[f"decoder.layers.{layer_idx}"]["self_attn"]
            k_cache, v_cache = ctx.kv_caches[layer_idx]
            _extract_and_write_kv(
                _layer_input(layer_idx),
                w_self,
                k_cache,
                v_cache,
                step=0,
                cq_id=CQ_WRITE,
            )
            if layer_idx == 0:
                _update_causal_mask(device, ctx.shared_causal_mask, step=0, T_max=ctx.T_max, cq_id=CQ_WRITE)
            ttnn.synchronize_device(device, cq_id=CQ_WRITE)
            op_event = ttnn.record_event(device, CQ_COMPUTE)

    for step in range(prediction_length):
        # ── 1. CPU embedding ──────────────────────────────────────────────
        t_cpu0 = time.perf_counter()
        step_input = _prepare_dec_step_cpu_1tok(
            k=step,
            future_samples_so_far=future_samples,
            past_values_cpu=past_values_cpu,
            future_time_cpu=future_time_cpu,
            static_feat_cpu=static_feat_cpu,
            loc_cpu=loc_cpu,
            scale_cpu=scale_cpu,
            value_proj_cpu=value_proj_cpu,
            dec_ln_w_cpu=dec_ln_w_cpu,
            dec_ln_b_cpu=dec_ln_b_cpu,
            pos_emb_cpu=pos_emb_cpu,
            context_length=context_length,
            T_max=T_max,
        )  # [BS, 1, PADDED_WIDTH] bfloat16 CPU
        cpu_prep_time += time.perf_counter() - t_cpu0

        t_dev0 = time.perf_counter()
        host_tt = ttnn.from_torch(
            step_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
        )

        if use_2cq:
            # Layer 0: write step k's raw input + K/V + mask, then trace.
            ttnn.wait_for_event(CQ_WRITE, op_event)
            ttnn.copy_host_to_device_tensor(host_tt, ctx.captured_dec_input, cq_id=CQ_WRITE)
            w_self0 = weights["decoder.layers.0"]["self_attn"]
            k_cache0, v_cache0 = ctx.kv_caches[0]
            _extract_and_write_kv(ctx.captured_dec_input, w_self0, k_cache0, v_cache0, step, cq_id=CQ_WRITE)
            _update_causal_mask_precomputed(
                device, ctx.shared_causal_mask, ctx.precomputed_masks_host[step], cq_id=CQ_WRITE
            )
            write_event = ttnn.record_event(device, CQ_WRITE)
            write_prep_time += time.perf_counter() - t_dev0

            t_trace0 = time.perf_counter()
            ttnn.wait_for_event(CQ_COMPUTE, write_event)
            ttnn.execute_trace(device, ctx.trace_ids[0], cq_id=CQ_COMPUTE, blocking=False)
            op_event = ttnn.record_event(device, CQ_COMPUTE)
            trace_exec_time += time.perf_counter() - t_trace0

            # Layers 1..N-1: this layer's write reads the PREVIOUS layer's
            # real output, only valid once that layer's trace finished --
            # hence waiting on op_event, recorded right after that
            # execute_trace.
            for layer_idx in range(1, num_layers):
                t_w0 = time.perf_counter()
                ttnn.wait_for_event(CQ_WRITE, op_event)
                w_self = weights[f"decoder.layers.{layer_idx}"]["self_attn"]
                k_cache, v_cache = ctx.kv_caches[layer_idx]
                _extract_and_write_kv(
                    _layer_input(layer_idx),
                    w_self,
                    k_cache,
                    v_cache,
                    step,
                    cq_id=CQ_WRITE,
                )
                write_event = ttnn.record_event(device, CQ_WRITE)
                write_prep_time += time.perf_counter() - t_w0

                t_tr = time.perf_counter()
                ttnn.wait_for_event(CQ_COMPUTE, write_event)
                ttnn.execute_trace(device, ctx.trace_ids[layer_idx], cq_id=CQ_COMPUTE, blocking=False)
                op_event = ttnn.record_event(device, CQ_COMPUTE)
                trace_exec_time += time.perf_counter() - t_tr
        else:
            # Single-queue path: everything blocking on CQ0, strictly
            # ordered layer by layer.
            ttnn.copy_host_to_device_tensor(host_tt, ctx.captured_dec_input, cq_id=CQ_WRITE)
            t_mask0 = time.perf_counter()
            _update_causal_mask_precomputed(
                device, ctx.shared_causal_mask, ctx.precomputed_masks_host[step], cq_id=CQ_WRITE
            )
            write_prep_time += time.perf_counter() - t_mask0

            for layer_idx in range(num_layers):
                t_w0 = time.perf_counter()
                w_self = weights[f"decoder.layers.{layer_idx}"]["self_attn"]
                k_cache, v_cache = ctx.kv_caches[layer_idx]
                _extract_and_write_kv(
                    _layer_input(layer_idx),
                    w_self,
                    k_cache,
                    v_cache,
                    step,
                    cq_id=CQ_WRITE,
                )
                write_prep_time += time.perf_counter() - t_w0

                t_tr = time.perf_counter()
                ttnn.execute_trace(device, ctx.trace_ids[layer_idx], cq_id=CQ_COMPUTE, blocking=True)
                trace_exec_time += time.perf_counter() - t_tr

        device_enqueue_time += time.perf_counter() - t_dev0
        t_readback0 = time.perf_counter()
        # ── 4. Read + sample ──────────────────────────────────────────────
        dec_out = ttnn.to_torch(ctx.traced_out).float()[..., :D_MODEL]
        readback_time += time.perf_counter() - t_readback0
        t_sample0 = time.perf_counter()
        params = _distribution_head(dec_out, weights)
        next_sample = _sample_next_step(params, dt, ctx._lc, ctx._sc)
        future_samples.append(next_sample)
        sample_time += time.perf_counter() - t_sample0

    total_time = cpu_prep_time + device_enqueue_time + readback_time + sample_time
    print(
        f"\n[TIMING] cpu_prep={cpu_prep_time*1000:.1f}ms device_enqueue={device_enqueue_time*1000:.1f}ms "
        f"(write_prep={write_prep_time*1000:.1f}ms trace_exec={trace_exec_time*1000:.1f}ms) "
        f"readback={readback_time*1000:.1f}ms sample={sample_time*1000:.1f}ms total={total_time*1000:.1f}ms"
    )
    _print_op_times()
    return torch.stack(future_samples, dim=1)  # [BS, prediction_length]


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point (mirrors generate_traced signature)
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_traced_cached(
    device,
    weights,
    past_values,
    past_time_features,
    future_time_features,
    past_observed_mask,
    static_categorical_features,
    static_real_features,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    num_parallel_samples=NUM_PARALLEL_SAMPLES,
    use_2cq=False,  # default False until 2CQ path is hardware-verified
    traced_ctx=None,
) -> torch.Tensor:
    """
    KV-cache traced generation, one trace per decoder layer. Mirrors
    generate_traced() signature.

    Pass a pre-built TracedDecoderContextCached via traced_ctx to reuse
    across multiple calls. Without it, builds/releases context per call.

    Returns [B, num_parallel_samples, prediction_length] float32.

    NOTE: use_2cq defaults to False. Neither queue mode of the per-layer
    chained version has been hardware-verified as of this revision --
    use_2cq=False first (test_2cq_matches_single_queue_output treats it
    as ground truth), then compare use_2cq=True against it, before
    trusting either for the bounty's perf numbers.
    """
    B = past_values.shape[0]
    S = num_parallel_samples

    owns_ctx = traced_ctx is None
    if owns_ctx:
        traced_ctx = build_traced_decoder_context_cached(
            device=device,
            weights=weights,
            past_values=past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            context_length=context_length,
            prediction_length=prediction_length,
            num_parallel_samples=num_parallel_samples,
        )

    try:
        samples = run_traced_generation_cached(
            ctx=traced_ctx,
            weights=weights,
            context_length=context_length,
            prediction_length=prediction_length,
            use_2cq=use_2cq,
        )
    finally:
        if owns_ctx:
            traced_ctx.release()

    return samples.reshape(B, S, prediction_length)
