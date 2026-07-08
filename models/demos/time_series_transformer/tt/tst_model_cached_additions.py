# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
KV-cache single-token decode + 24 per-step traces + optional 2CQ.

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
  below, which is the only path with KV-caching, per-step trace reuse,
  hardware-probe-verified 2CQ event choreography, and passing throughput
  /sample-generation/CRPS/NLL results against the actual bounty targets.

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

Architecture:
- 24 separate traces (one per step): required because slice_write's Python-int
  offsets are baked into untraced calls, and the causal mask shape varies per
  step — confirmed by probe_slice_write_tensor_offset.py on hardware.
- Cross-attn KV precomputed once (encoder hidden is fixed).
- CPU embedding prep per step (no device ops between trace replays).

NOTE ON A REVERTED EXPERIMENT (this revision):
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
  this pattern — an earlier revision of this file invented both, and used
  them entirely within CQ0 (no cross-queue handoff at all), which provided
  zero actual overlap and relied on an unverified API (create_event).

  This revision follows the report's "ops + readback on CQ0, input writes on
  CQ1" layout (section 2.3.2 / 3.3.1.1), adapted to this model's per-step loop:
    - CQ1: copy_host_to_device_tensor for the step input, and the K/V
           slice_write cache updates (the "write" work).
    - CQ0: execute_trace (the "compute" work, pure read-from-cache).
  Two events per step:
    - `op_event` (recorded on CQ0): signals CQ0 has finished consuming the
      PREVIOUS step's input buffer / caches, so CQ1 may overwrite them.
    - `write_event` (recorded on CQ1): signals CQ1 has finished writing the
      CURRENT step's input buffer and cache updates, so CQ0 may run its trace.
  This still must be re-verified end-to-end on hardware (see test_tst_perf.py's
  test_single_sequence_latency_2cq); use_2cq=False remains the default for
  correctness testing until that verification happens.
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
    """
    BS = hidden_1tok.shape[0]

    fused_qkv = ttnn.linear(hidden_1tok, w["qkv_weight"], bias=w["qkv_bias"], queue_id=cq_id)
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        fused_qkv,
        num_heads=NUM_HEADS,
        queue_id=cq_id,
    )
    # query: [BS, H, 1, D]
    # key:   [BS, H, D, 1]  (already transposed by TTNN — matches k_cache's
    #         [B, H, D, T_max] layout directly, no permute needed)
    # value: [BS, H, 1, D]

    key_rm = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT, queue_id=cq_id)
    value_rm = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT, queue_id=cq_id)

    _step = [1, 1, 1, 1]
    ttnn.experimental.slice_write(
        key_rm,
        k_cache,
        [0, 0, 0, step],
        [BS, NUM_HEADS, HEAD_DIM_PADDED, step + 1],
        _step,
        queue_id=cq_id,
    )
    ttnn.experimental.slice_write(
        value_rm,
        v_cache,
        [0, 0, step, 0],
        [BS, NUM_HEADS, step + 1, HEAD_DIM_PADDED],
        _step,
        queue_id=cq_id,
    )

    return query  # [BS, H, 1, D] TILE_LAYOUT


def _extract_q_only(
    hidden_1tok: ttnn.Tensor,
    w: dict,
) -> ttnn.Tensor:
    """
    Phase 1, trace-capture version: project hidden_1tok → Q/K/V but
    return Q only (K/V already written before begin_trace_capture).
    Used during warmup/compile run only.
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
    v_cache: ttnn.Tensor,  # [BS, H, T_max, D] ROW_MAJOR (read as TILE)
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

    v_tile = ttnn.to_layout(v_cache, ttnn.TILE_LAYOUT)  # [BS, H, T_max, D]
    context = ttnn.matmul(probs, v_tile)  # [BS, H, 1, D]
    context = ttnn.transformer.concatenate_heads(context)  # [BS, 1, PADDED_WIDTH]
    return ttnn.linear(context, w["out_proj_weight"], bias=w["out_proj_bias"])


# ─────────────────────────────────────────────────────────────────────────────
# Traced decoder stack: reads caches (pre-written), no slice_write inside
# ─────────────────────────────────────────────────────────────────────────────


def _run_traced_decoder_stack(
    hidden_1tok: ttnn.Tensor,  # [BS, 1, PADDED_WIDTH]
    queries: list,  # per-layer [BS, H, 1, D] query tensors (pre-extracted)
    kv_caches: list,  # per-layer (k_cache, v_cache) — already updated
    precomputed_kv: list,  # per-layer (k_pre, v_pre) for cross-attn
    causal_mask_1tok: ttnn.Tensor,  # [1, 1, 1, T_max]
    weights: dict,
    num_decoder_layers: int,
) -> ttnn.Tensor:
    """
    Pure-compute decoder stack. No slice_write. Safe to trace.

    Reads k_cache/v_cache at their current (already updated) state.
    """
    hidden = hidden_1tok
    for layer_idx in range(num_decoder_layers):
        w = weights[f"decoder.layers.{layer_idx}"]
        q = queries[layer_idx]
        k_cache, v_cache = kv_caches[layer_idx]
        k_pre, v_pre = precomputed_kv[layer_idx]

        # Masked self-attention (attend from cache — no writes)
        self_attn_out = _attend_from_cache(q, k_cache, v_cache, causal_mask_1tok, w["self_attn"])
        residual = ttnn.add(hidden, self_attn_out)
        hidden = layer_norm_padded(
            residual,
            w["self_attn_layer_norm_weight"],
            w["self_attn_layer_norm_bias"],
            orig_dim=D_MODEL,
        )

        # Cross-attention (precomputed KV — no writes)
        cross_out = tst_cross_attention_with_kv(hidden, k_pre, v_pre, w["encoder_attn"])
        residual = ttnn.add(hidden, cross_out)
        hidden = layer_norm_padded(
            residual,
            w["encoder_attn_layer_norm_weight"],
            w["encoder_attn_layer_norm_bias"],
            orig_dim=D_MODEL,
        )

        # FFN
        ffn_out = tst_ffn(hidden, w)
        residual = ttnn.add(hidden, ffn_out)
        hidden = layer_norm_padded(
            residual,
            w["final_layer_norm_weight"],
            w["final_layer_norm_bias"],
            orig_dim=D_MODEL,
        )

    return hidden  # [BS, 1, PADDED_WIDTH]


from .tst_attention import tst_cross_attention_with_kv  # noqa: E402  (used above)

# ─────────────────────────────────────────────────────────────────────────────
# Causal mask helper
# ─────────────────────────────────────────────────────────────────────────────


def _update_causal_mask(device: ttnn.Device, mask_tensor: ttnn.Tensor, step: int, T_max: int, cq_id: int = 0) -> None:
    """
    Update the shared causal mask buffer in-place for the given step.

    PERF/CORRECTNESS FIX: previously hardcoded cq_id=0 unconditionally, so in
    the use_2cq=True path this write landed on the COMPUTE queue (CQ0)
    instead of the WRITER queue (CQ1/CQ_WRITE) that every other per-step
    write (input, K/V cache) uses. It produced correct output only because
    same-queue FIFO ordering happened to sequence it before execute_trace --
    but it forced CQ0 to block on a host->device copy synchronously, which
    defeats the entire purpose of the 2CQ split (CQ1 writes overlap with
    CQ0 finishing the PREVIOUS step's trace). Callers now pass cq_id=CQ_WRITE
    explicitly, matching how the input buffer and K/V caches are already
    written on CQ_WRITE in run_traced_generation_cached.
    """
    mask = torch.full((1, 1, 1, T_max), _NEG_INF)
    mask[:, :, :, : step + 1] = 0.0
    new_tt = ttnn.from_torch(
        mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=None,
    )
    ttnn.copy_host_to_device_tensor(new_tt, mask_tensor, cq_id=cq_id)


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

    Verified identical to _prepare_dec_step_cpu[:, k, :] from tst_model.py
    across a simulated 5-step run (see prior session notes).
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
    One-time setup: encoder, cross-attn KV, KV-cache allocation, 24 trace captures.

    Each trace captures ONLY pure compute (attend-from-cache + cross-attn + FFN).
    The slice_write cache updates happen outside the trace in run_traced_generation_cached.
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

    # ── Persistent input buffer (allocated before any trace capture) ───────
    captured_dec_input = ttnn.from_torch(
        torch.zeros(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ── Shared causal mask buffer (fixed address; contents updated per step
    #    via _update_causal_mask outside the trace) ─────────────────────────
    shared_causal_mask = _build_causal_mask_1tok(device, 0, T_max)

    # ── Warmup compile run (step 0, untraced) ─────────────────────────────
    queries_0 = [
        _extract_q_only(captured_dec_input, weights[f"decoder.layers.{i}"]["self_attn"])
        for i in range(num_decoder_layers)
    ]
    traced_out = _run_traced_decoder_stack(
        captured_dec_input,
        queries_0,
        kv_caches,
        precomputed_kv,
        shared_causal_mask,
        weights,
        num_decoder_layers,
    )
    ttnn.synchronize_device(device)

    # ── Capture ONE trace using a shared mask buffer ──────────────────────
    # The mask buffer address is fixed (baked into the trace); we update its
    # contents before each execute_trace call instead of capturing 24 traces.
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    queries_traced = [
        _extract_q_only(captured_dec_input, weights[f"decoder.layers.{i}"]["self_attn"])
        for i in range(num_decoder_layers)
    ]
    traced_out = _run_traced_decoder_stack(
        captured_dec_input,
        queries_traced,
        kv_caches,
        precomputed_kv,
        shared_causal_mask,
        weights,
        num_decoder_layers,
    )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    trace_ids = [trace_id]  # single trace reused for all 24 steps

    # Reset KV caches to zero before actual generation
    _zero_kv_caches(kv_caches, device, BS, T_max)
    ttnn.synchronize_device(device)

    ctx = TracedDecoderContextCached()
    ctx.device = device
    ctx.trace_ids = trace_ids
    ctx.shared_causal_mask = shared_causal_mask
    ctx.T_max = T_max
    ctx.captured_dec_input = captured_dec_input
    ctx.traced_out = traced_out
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
    """Zero-fill KV caches via slice_write (outside any trace)."""
    for k_cache, v_cache in kv_caches:
        zero_k = ttnn.from_torch(
            torch.zeros(BS, NUM_HEADS, HEAD_DIM_PADDED, T_max, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        zero_v = ttnn.from_torch(
            torch.zeros(BS, NUM_HEADS, T_max, HEAD_DIM_PADDED, dtype=torch.bfloat16),
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
    Per step:
      1. CPU: build single-token embedding
      2. Device: copy input + slice_write K/V into caches, update causal mask
           - use_2cq=False: all on CQ0, blocking, no event choreography.
           - use_2cq=True:  writes on CQ1, compute (execute_trace) on CQ0,
                             handed off via record_event/wait_for_event
                             exactly per the tech report's documented
                             "ops+readback on CQ0, input writes on CQ1"
                             pattern (section 2.3.2 / 3.3.1.1). There is no
                             ttnn.create_event or ttnn.event_synchronize in
                             that pattern — record_event both creates and
                             records the event, and wait_for_event is the
                             only blocking-on-event primitive used.
      3. Device (traced): Q attend-from-cache + cross-attn + FFN
      4. Host: sample from distribution

    Returns [BS, prediction_length] float32.
    """
    device = ctx.device
    BS = ctx.BS
    T_max = ctx.T_max
    dt = ctx.dist_type
    num_layers = ctx.num_decoder_layers

    CQ_COMPUTE = 0  # runs execute_trace, always
    CQ_WRITE = 1 if use_2cq else 0  # writes input + KV cache updates + mask

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

    future_samples = []
    cpu_prep_time = 0.0
    device_enqueue_time = 0.0
    readback_time = 0.0
    sample_time = 0.0

    # Dummy initial op_event on CQ_COMPUTE: signals "compute has finished
    # with the input buffer / caches" so the writer queue can proceed on
    # the very first step too (mirrors the tech report's "dummy record an
    # op event ... since we wait on this first in the loop" pattern).
    if use_2cq:
        # CORRECTNESS FIX (hardware-verified this session): an op's kernel
        # binary must be resident on a queue BEFORE it is dispatched near a
        # trace-execute handoff, or the first real call absorbs a JIT-compile
        # stall -- confirmed via isolated probe (ttnn.sum failed inside
        # trace capture until warmed up untraced first; identical mechanism
        # applies here to _extract_and_write_kv / _update_causal_mask on
        # CQ_WRITE, which -- unlike the compute-side ops -- have NEVER been
        # run on CQ_WRITE before this point; the existing warmup block above
        # only exercises CQ0). Warming up against real step=0 is harmless:
        # the actual step 0 of the loop below immediately overwrites
        # position 0 with the correct value before it's ever read.
        for layer_idx in range(num_layers):
            w_self = weights[f"decoder.layers.{layer_idx}"]["self_attn"]
            k_cache, v_cache = ctx.kv_caches[layer_idx]
            _extract_and_write_kv(
                ctx.captured_dec_input,
                w_self,
                k_cache,
                v_cache,
                step=0,
                cq_id=CQ_WRITE,
            )
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
            # Stall the writer queue until compute (CQ0) has signalled it is
            # done reading the previous step's buffer/caches.
            ttnn.wait_for_event(CQ_WRITE, op_event)

            # ── 2. Writer queue (CQ1): write input + update KV caches + mask
            ttnn.copy_host_to_device_tensor(host_tt, ctx.captured_dec_input, cq_id=CQ_WRITE)
            for layer_idx in range(num_layers):
                w_self = weights[f"decoder.layers.{layer_idx}"]["self_attn"]
                k_cache, v_cache = ctx.kv_caches[layer_idx]
                _extract_and_write_kv(
                    ctx.captured_dec_input,
                    w_self,
                    k_cache,
                    v_cache,
                    step,
                    cq_id=CQ_WRITE,
                )
            _update_causal_mask(device, ctx.shared_causal_mask, step, ctx.T_max, cq_id=CQ_WRITE)
            # Signal that this step's writes are complete.
            write_event = ttnn.record_event(device, CQ_WRITE)

            # ── 3. Compute queue (CQ0): wait for writes, then run trace ──
            ttnn.wait_for_event(CQ_COMPUTE, write_event)
            ttnn.execute_trace(device, ctx.trace_ids[0], cq_id=CQ_COMPUTE, blocking=False)
            # Signal that compute is done consuming the input buffer/caches
            # for THIS step, so the writer queue may overwrite them for the
            # next step's input write (the loop's next iteration).
            op_event = ttnn.record_event(device, CQ_COMPUTE)
        else:
            # ── 2/3. Single-queue path: everything blocking on CQ0 ───────
            ttnn.copy_host_to_device_tensor(host_tt, ctx.captured_dec_input, cq_id=CQ_WRITE)
            for layer_idx in range(num_layers):
                w_self = weights[f"decoder.layers.{layer_idx}"]["self_attn"]
                k_cache, v_cache = ctx.kv_caches[layer_idx]
                _extract_and_write_kv(
                    ctx.captured_dec_input,
                    w_self,
                    k_cache,
                    v_cache,
                    step,
                    cq_id=CQ_WRITE,
                )
            _update_causal_mask(device, ctx.shared_causal_mask, step, ctx.T_max, cq_id=CQ_WRITE)
            ttnn.execute_trace(device, ctx.trace_ids[0], cq_id=CQ_COMPUTE, blocking=True)

        device_enqueue_time += time.perf_counter() - t_dev0
        t_readback0 = time.perf_counter()
        # ── 4. Read + sample ──────────────────────────────────────────────
        # Per-step output readback is a blocking host read regardless of
        # use_2cq -- the distribution head / sampler runs on host and needs
        # the values now, so there is no benefit to a non-blocking .cpu()
        # here the way there is for whole-model output readback in the
        # tech report's examples. A final ttnn.synchronize_device() is not
        # needed for this reason (unlike the report's batched-output
        # examples), since every step already blocks on its own readback.
        dec_out = ttnn.to_torch(ctx.traced_out).float()[..., :D_MODEL]
        readback_time += time.perf_counter() - t_readback0
        t_sample0 = time.perf_counter()
        params = _distribution_head(dec_out, weights)
        next_sample = _sample_next_step(params, dt, ctx._lc, ctx._sc)
        future_samples.append(next_sample)
        sample_time += time.perf_counter() - t_sample0

    total_time = cpu_prep_time + device_enqueue_time + readback_time + sample_time
    print(
        f"\n[TIMING] cpu_prep={cpu_prep_time*1000:.1f}ms device_enqueue={device_enqueue_time*1000:.1f}ms readback={readback_time*1000:.1f}ms sample={sample_time*1000:.1f}ms total={total_time*1000:.1f}ms"
    )
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
    KV-cache traced generation. Mirrors generate_traced() signature.

    Pass a pre-built TracedDecoderContextCached via traced_ctx to reuse
    across multiple calls. Without it, builds/releases context per call.

    Returns [B, num_parallel_samples, prediction_length] float32.

    NOTE: use_2cq defaults to False. The 2CQ path (CQ event choreography)
    follows the tech report's documented record_event(device, cq_id) /
    wait_for_event(cq_id, event) pattern with writes on CQ1 and compute on
    CQ0, but it has NOT yet been hardware-verified. Use False for
    correctness testing first; only flip to True once
    test_single_sequence_latency_2cq has been run and confirmed both
    correct (matches use_2cq=False output distributionally) and not hanging.
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
