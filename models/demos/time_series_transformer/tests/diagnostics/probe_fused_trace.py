# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
DIAGNOSTIC PROBE 3 -- NOT for the PR, lives in tests/diagnostics/ only.

QUESTION THIS ANSWERS:
Probes 1 and 2 proved the overlay technique (cache-read + one-hot-selector
overlay, no write) produces bit-identical output to the write-then-read
cache pattern, first via host round-trip (probe 1), then fully on-device
with zero host round-trips (probe 2). Neither probe touched a trace.

This probe is the actual point of the investigation: capture ALL decoder
layers for one step inside a SINGLE begin_trace_capture/end_trace_capture
region (using the overlay instead of the write that currently forces one
trace per layer), replay that one trace across multiple autoregressive
steps, and confirm the result still matches -- plus get the first real
timing numbers against today's num_decoder_layers-separate-traces design.

SCOPING DECISION (read before extending this probe):
This probe compares the FUSED single trace against a PER-LAYER-TRACE
baseline built inside this probe (mirroring the canonical warmup/capture
pattern in tst_model_cached_additions.py lines ~700-745), NOT against a
fresh eager ground truth. Eager-vs-overlay correctness was already proven
twice (probe 1: host round-trip, probe 2: device-side). Production
PCC/e2e tests already establish the per-layer-trace path as correct
against the HF reference. Re-deriving eager correctness a third time here
would be redundant. What's new and unproven until this probe is: (a) does
the overlay survive actual trace capture (no host callback opportunities
mid-capture), and (b) does chaining num_decoder_layers layers inside ONE
trace, then reading back each layer's real input buffer AFTER execute_trace
for write-back, actually work across multiple replay steps.

KEY MECHANISM THIS RELIES ON (verified by reading, not assumed):
tst_model_cached_additions.py's existing per-layer-trace capture (lines
~700-745) already proves that a tensor produced INSIDE a trace capture
(layer_out from _run_single_layer_compute) gets a stable, fixed device
address that persists after end_trace_capture and can be fed as the real
input to the NEXT layer's trace capture. This probe extends that same
proven mechanism: chain all layers' hidden states inside ONE trace instead
of stopping at layer boundaries, then after execute_trace runs, read back
each layer's real input buffer (still populated with this step's real
value) to do the actual persistent cache write OUTSIDE the trace, via the
real _extract_and_write_kv (never reimplemented -- imported as-is).

WRITE-BACK ORDERING (the part that makes fusion legal):
Inside the fused trace: overlay only, no write, so cache position `step`
must be zero when the overlay reads it -- true for every layer because
nothing writes to that position until AFTER the whole trace finishes.
After execute_trace(blocking=True) returns: loop over layers, call the
real _extract_and_write_kv using that layer's real input buffer (which
the trace populated this step) to persist K/V for FUTURE steps' overlay
reads. This must complete before the NEXT step's execute_trace runs (the
next step's overlay reads the same cache buffers) -- guaranteed here by
staying on a single command queue (cq_id=0) with blocking calls throughout.
2CQ overlap of write-back with the next step's compute is a real future
optimization but explicitly NOT attempted in this probe -- correctness
and a first honest timing number come first.

WHY THIS PROBE NOW DOES TWO WARMUP PASSES BEFORE CAPTURE (fix applied
after the first hardware run of this probe):
The first hardware run of this file hit
    TT_FATAL: Writes are not supported during trace capture. trace id: 2
on the very first ttnn.mul(key, k_selector_dev) call inside
begin_trace_capture/end_trace_capture. The backtrace showed the failing
write was inside MeshWorkloadImpl::load_binaries -> write_shard_to_device,
i.e. a program-cache MISS triggering a kernel-binary load/write, not a
fundamental "ttnn.mul is untraceable" limitation. Tenstorrent's own trace
tech report states the requirement directly: "to capture the trace of a
sequence of operations, we must run with program cache and have already
compiled the target operations before we capture them." --
https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md
The original warmup loop below only ran `_run_single_layer_compute` (the
write-then-read path) -- it never once ran `_overlay_kv_device` or
`_run_single_layer_compute_prebuilt_kv`, the two functions actually used
INSIDE the fused trace, so their specific op configurations (these shapes,
these selector tensors) had never been compiled into the program cache.
Fix: add a second eager warmup pass that runs the exact overlay path once
before capture begins, then re-synchronize. No arithmetic changed --
this only adds compilation coverage.

WHY THIS PROBE ALSO CONVERTS k_cache TO TILE_LAYOUT BEFORE THE OVERLAY ADD
(fix applied after the trace-capture fix produced max_abs_diff ~0.8-0.9
from step 0): allocate_kv_cache in tst_attention.py allocates k_cache
ROW_MAJOR unconditionally -- K's transposed [B,H,D,T_max] layout is
incompatible with ttnn.update_cache's fixed axis contract, so K always
stays on the ROW_MAJOR slice_write path, for every batch size. `key`
coming out of split_query_key_value_and_split_heads is TILE_LAYOUT. The
real _attend_from_cache converts k_cache to TILE via
`ttnn.to_layout(k_cache, ttnn.TILE_LAYOUT)` before ever using it in a
matmul -- the overlay function originally skipped that conversion and did
`ttnn.add(k_cache, k_overlay)` directly, silently mixing ROW_MAJOR and
TILE tensors (this does not raise an error; it just produces wrong
values). v_cache is TILE_LAYOUT only when B==1 (per allocate_kv_cache);
this probe is BS=1-only so v_cache was already TILE and never actually
broken, but the same conditional conversion _attend_from_cache uses is
included in the overlay function for parity.

KEY MECHANISM THIS RELIES ON (verified by reading, not assumed):
tst_model_cached_additions.py's existing per-layer-trace capture (lines
~700-745) already proves that a tensor produced INSIDE a trace capture
(layer_out from _run_single_layer_compute) gets a stable, fixed device
address that persists after end_trace_capture and can be fed as the real
input to the NEXT layer's trace capture. This probe extends that same
proven mechanism.

WHAT THIS PROBE ASSUMES WITHOUT RE-VERIFYING:
ttnn.mul's NumPy-style broadcast (confirmed via help() output on hardware
in probe 2, now treated as an established fact, not re-derived here).
_extract_q_only and _extract_and_write_kv's internal correctness (both
imported as-is from the canonical module, never reimplemented -- if
either were wrong, production PCC/e2e tests would already be failing,
which per current memory they are not). ttnn.to_layout converting an
already-TILE tensor to TILE_LAYOUT is a safe no-op -- this is the same
call structurally made every replay inside the real _attend_from_cache
against the real ROW_MAJOR k_cache in the (working) baseline arm, so this
is exercised, not a new risk.

TOLERANCE: 0.01, same as probe 2 (device-side, no host round-trip
expected in the fused arm's hot path; write-back after execute_trace does
call real ttnn ops but no to_torch/from_torch round-trip either).

TIMING METHODOLOGY: WARMUP_REPLAYS=5 discarded before measuring, per this
project's own documented learning that fewer warmup replays understate
steady-state latency. Reports median device-side ms/step for both arms
over TIMING_STEPS replays. This times mask/selector update through
write-back completion for both arms -- NOT host-side step-input prep,
which is identical for both and not what's being changed.
"""

import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[5]  # .../tt-metal (verified path depth from probe 1)
sys.path.insert(0, str(REPO_ROOT))

from models.demos.time_series_transformer.tt.tst_attention import (  # noqa: E402
    NUM_HEADS,
    allocate_kv_cache,
    precompute_cross_attn_kv,
    tst_cross_attention_with_kv,
)
from models.demos.time_series_transformer.tt.tst_decoder_layer import tst_ffn  # noqa: E402
from models.demos.time_series_transformer.tt.tst_embedding import prepare_encoder_input  # noqa: E402
from models.demos.time_series_transformer.tt.tst_model import (  # noqa: E402
    CONTEXT_LENGTH,
    D_MODEL,
    PADDED_WIDTH,
    _apply_layernorm_ttnn,
    _build_static_feat_cpu,
    _inputs_to_ttnn,
    load_weights,
    run_encoder,
)
from models.demos.time_series_transformer.tt.tst_model_cached_additions import (  # noqa: E402
    _attend_from_cache,
    _build_causal_mask_1tok,
    _extract_and_write_kv,
    _extract_q_only,
    _precompute_causal_masks_host,
    _prepare_dec_step_cpu_1tok,
    _run_single_layer_compute,
    _update_causal_mask_precomputed,
    _zero_kv_caches,
)
from models.demos.time_series_transformer.tt.ttnn_utils import layer_norm_padded  # noqa: E402

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
MODEL_REVISION = "2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35"
REFERENCE_DIR = REPO_ROOT / "models" / "demos" / "time_series_transformer" / "reference"

T_MAX = 8
N_CORRECTNESS_STEPS = 5
WARMUP_REPLAYS = 5
TIMING_STEPS = 30
BS = 1
TOLERANCE = 0.01


# ── Overlay primitives (proven correct on hardware in probe 2) ─────────────


def _overlay_kv_device(hidden_1tok, w_self, k_cache, v_cache, k_selector_dev, v_selector_dev):
    print(">>> RUNNING PATCHED _overlay_kv_device <<<", flush=True)

    """Pure device-side overlay, zero host round-trips. Must be called
    INSIDE trace capture in this probe -- that is the new thing being
    tested versus probe 2, which ran this eagerly.

    LAYOUT FIX (applied after the first hardware run of the trace-capture
    fix produced max_abs_diff ~0.8-0.9 from step 0): allocate_kv_cache
    allocates k_cache ROW_MAJOR unconditionally -- K's transposed
    [B,H,D,T_max] layout is incompatible with ttnn.update_cache's fixed
    axis contract, so K always stays on the ROW_MAJOR slice_write path,
    for every batch size (see allocate_kv_cache's docstring in
    tst_attention.py). `key` coming out of
    split_query_key_value_and_split_heads is TILE_LAYOUT. The real
    _attend_from_cache converts k_cache to TILE via
    `ttnn.to_layout(k_cache, ttnn.TILE_LAYOUT)` before ever using it in a
    matmul -- this function skipped that conversion and did
    `ttnn.add(k_cache, k_overlay)` directly, mixing ROW_MAJOR and TILE
    tensors. That does not raise an error; it silently produces wrong
    values, which is what the large, step-0-present diff actually was.

    v_cache is TILE_LAYOUT only when B==1 (per allocate_kv_cache); this
    probe is BS=1-only so v_cache was already TILE and never actually
    broken, but the same conditional conversion _attend_from_cache uses
    is included here for parity, in case BS is ever changed.
    """
    fused_qkv = ttnn.linear(hidden_1tok, w_self["qkv_weight"], bias=w_self["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)

    k_cache_tile = ttnn.to_layout(k_cache, ttnn.TILE_LAYOUT)
    k_overlay = ttnn.mul(key, k_selector_dev)
    k_full = ttnn.add(k_cache_tile, k_overlay)

    v_cache_tile = v_cache if v_cache.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(v_cache, ttnn.TILE_LAYOUT)
    v_overlay = ttnn.mul(value, v_selector_dev)
    v_full = ttnn.add(v_cache_tile, v_overlay)

    return query, k_full, v_full


def _run_single_layer_compute_prebuilt_kv(hidden_in, query, k_full, v_full, k_pre, v_pre, causal_mask_1tok, w):
    self_attn_out = _attend_from_cache(query, k_full, v_full, causal_mask_1tok, w["self_attn"])
    residual = ttnn.add(hidden_in, self_attn_out)
    hidden = layer_norm_padded(
        residual, w["self_attn_layer_norm_weight"], w["self_attn_layer_norm_bias"], orig_dim=D_MODEL
    )

    cross_out = tst_cross_attention_with_kv(hidden, k_pre, v_pre, w["encoder_attn"])
    residual = ttnn.add(hidden, cross_out)
    hidden = layer_norm_padded(
        residual, w["encoder_attn_layer_norm_weight"], w["encoder_attn_layer_norm_bias"], orig_dim=D_MODEL
    )

    ffn_out = tst_ffn(hidden, w)
    residual = ttnn.add(hidden, ffn_out)
    hidden = layer_norm_padded(residual, w["final_layer_norm_weight"], w["final_layer_norm_bias"], orig_dim=D_MODEL)
    return hidden


def _precompute_kv_selectors_host(T_max):
    k_selectors, v_selectors = [], []
    for step in range(T_max):
        k_sel = torch.zeros(1, 1, 1, T_max, dtype=torch.bfloat16)
        k_sel[..., step] = 1.0
        k_selectors.append(k_sel)
        v_sel = torch.zeros(1, 1, T_max, 1, dtype=torch.bfloat16)
        v_sel[:, :, step, :] = 1.0
        v_selectors.append(v_sel)
    return k_selectors, v_selectors


def _update_selector_precomputed(shared_dev_buf, selector_host_list, step, cq_id=0):
    host_tensor = ttnn.from_torch(selector_host_list[step], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None)
    ttnn.copy_host_to_device_tensor(host_tensor, shared_dev_buf, cq_id=cq_id)


# ── Per-layer-trace baseline (self-contained mirror of the canonical
#    warmup/capture pattern -- NOT importing production ctx/2CQ machinery,
#    out of scope for this probe) ───────────────────────────────────────


def _build_per_layer_traces(
    device, weights, num_layers, kv_caches, precomputed_kv, captured_dec_input, shared_causal_mask
):
    """Mirrors tst_model_cached_additions.py lines ~700-745: one trace per
    layer, chained through real fixed-address buffers. Returns
    (trace_ids, layer_hidden_buffers) where layer_hidden_buffers[i] is
    layer i's real input (index num_layers is the final output)."""
    trace_ids = []
    layer_hidden_buffers = [captured_dec_input]
    hidden_in = captured_dec_input
    for layer_idx in range(num_layers):
        w_layer = weights[f"decoder.layers.{layer_idx}"]
        w_self = w_layer["self_attn"]
        k_cache, v_cache = kv_caches[layer_idx]
        k_pre, v_pre = precomputed_kv[layer_idx]

        q_warmup = _extract_and_write_kv(hidden_in, w_self, k_cache, v_cache, step=0, cq_id=0)
        _ = _run_single_layer_compute(hidden_in, q_warmup, k_cache, v_cache, k_pre, v_pre, shared_causal_mask, w_layer)
        ttnn.synchronize_device(device)

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        q_traced = _extract_q_only(hidden_in, w_self)
        layer_out = _run_single_layer_compute(
            hidden_in, q_traced, k_cache, v_cache, k_pre, v_pre, shared_causal_mask, w_layer
        )
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        trace_ids.append(trace_id)
        layer_hidden_buffers.append(layer_out)
        hidden_in = layer_out

    return trace_ids, layer_hidden_buffers


def _run_per_layer_traces_one_step(device, weights, num_layers, kv_caches, trace_ids, layer_hidden_buffers, step):
    """One autoregressive step: write real K/V for each layer using that
    layer's real input, then execute that layer's trace -- exactly the
    canonical per-layer sequencing."""
    for layer_idx in range(num_layers):
        w_self = weights[f"decoder.layers.{layer_idx}"]["self_attn"]
        k_cache, v_cache = kv_caches[layer_idx]
        _extract_and_write_kv(layer_hidden_buffers[layer_idx], w_self, k_cache, v_cache, step=step, cq_id=0)
        ttnn.execute_trace(device, trace_ids[layer_idx], cq_id=0, blocking=True)
    return layer_hidden_buffers[num_layers]  # final output


# ── Fused single-trace arm (the thing under test) ──────────────────────


def _build_fused_trace(
    device,
    weights,
    num_layers,
    kv_caches,
    precomputed_kv,
    captured_dec_input,
    shared_causal_mask,
    shared_k_selector,
    shared_v_selector,
):
    """ALL layers captured inside ONE begin_trace_capture/end_trace_capture
    region via the overlay (no write inside capture).

    Two warmup passes run BEFORE capture starts, each followed by a
    synchronize_device:

      Pass 1 (unchanged from the original probe): the real write-then-
      compute path (_extract_and_write_kv + _run_single_layer_compute).
      This exists purely to compile kernels shared with the per-layer
      baseline before capture -- warmup itself is never traced.

      Pass 2 (this is the trace-capture fix): the actual overlay path
      used INSIDE the trace (_overlay_kv_device + _run_single_layer_
      compute_prebuilt_kv), run once eagerly with program cache enabled
      so that all ops in the overlay path, including the k_cache
      TILE_LAYOUT conversion, are fully compiled before
      begin_trace_capture.

    Pass 2 only reads kv_caches (overlay, never writes), so it is safe to
    run against the already-zeroed caches without corrupting state that
    the correctness/timing loops depend on -- kv_caches are re-zeroed by
    the caller after this function returns, before either arm is used
    for real work.
    """
    hidden_in = captured_dec_input
    for layer_idx in range(num_layers):
        w_layer = weights[f"decoder.layers.{layer_idx}"]
        w_self = w_layer["self_attn"]
        k_cache, v_cache = kv_caches[layer_idx]
        k_pre, v_pre = precomputed_kv[layer_idx]
        q_warmup = _extract_and_write_kv(hidden_in, w_self, k_cache, v_cache, step=0, cq_id=0)
        hidden_in = _run_single_layer_compute(
            hidden_in, q_warmup, k_cache, v_cache, k_pre, v_pre, shared_causal_mask, w_layer
        )
    ttnn.synchronize_device(device)

    hidden_in = captured_dec_input
    for layer_idx in range(num_layers):
        w_layer = weights[f"decoder.layers.{layer_idx}"]
        w_self = w_layer["self_attn"]
        k_cache, v_cache = kv_caches[layer_idx]
        k_pre, v_pre = precomputed_kv[layer_idx]
        q_new, k_full, v_full = _overlay_kv_device(
            hidden_in, w_self, k_cache, v_cache, shared_k_selector, shared_v_selector
        )
        hidden_in = _run_single_layer_compute_prebuilt_kv(
            hidden_in, q_new, k_full, v_full, k_pre, v_pre, shared_causal_mask, w_layer
        )
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    hidden_in = captured_dec_input
    layer_hidden_buffers = [captured_dec_input]
    for layer_idx in range(num_layers):
        w_layer = weights[f"decoder.layers.{layer_idx}"]
        w_self = w_layer["self_attn"]
        k_cache, v_cache = kv_caches[layer_idx]
        k_pre, v_pre = precomputed_kv[layer_idx]
        q_new, k_full, v_full = _overlay_kv_device(
            hidden_in, w_self, k_cache, v_cache, shared_k_selector, shared_v_selector
        )
        hidden_in = _run_single_layer_compute_prebuilt_kv(
            hidden_in, q_new, k_full, v_full, k_pre, v_pre, shared_causal_mask, w_layer
        )
        layer_hidden_buffers.append(hidden_in)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    return trace_id, layer_hidden_buffers


def _run_fused_trace_one_step(device, weights, num_layers, kv_caches, trace_id, layer_hidden_buffers, step):
    """execute_trace once (all layers), THEN write-back per layer using
    each layer's real input buffer -- now populated by the trace this step
    -- so future steps' overlays see this step's real K/V."""
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    for layer_idx in range(num_layers):
        w_self = weights[f"decoder.layers.{layer_idx}"]["self_attn"]
        k_cache, v_cache = kv_caches[layer_idx]
        _extract_and_write_kv(layer_hidden_buffers[layer_idx], w_self, k_cache, v_cache, step=step, cq_id=0)
    return layer_hidden_buffers[num_layers]


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24_576)
    trace_ids_to_release = []
    try:
        hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        hf_model.eval()
        weights = load_weights(hf_model, device)
        num_layers = sum(1 for k in weights if k.startswith("decoder.layers."))
        print(f"[setup] num_decoder_layers={num_layers}")

        tensors = {}
        with safe_open(str(REFERENCE_DIR / "inputs.safetensors"), framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)[:BS]

        pv_tt, pt_tt, pm_tt, sc_tt, sr_tt = _inputs_to_ttnn(
            device,
            tensors["input_past_values"],
            tensors["input_past_time_features"],
            tensors["input_past_observed_mask"],
            tensors["input_static_categorical_features"],
            tensors["input_static_real_features"],
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
            context_length=CONTEXT_LENGTH,
        )
        enc_emb = _apply_layernorm_ttnn(enc_emb, weights["encoder_layernorm_ttnn"])
        encoder_hidden = run_encoder(device, enc_emb, weights)

        precomputed_kv = []
        for i in range(num_layers):
            k_pre, v_pre = precompute_cross_attn_kv(encoder_hidden, weights[f"decoder.layers.{i}"]["encoder_attn"])
            precomputed_kv.append((k_pre, v_pre))

        loc_cpu = ttnn.to_torch(loc).float()
        scale_cpu = ttnn.to_torch(scale).float()
        past_values_cpu = tensors["input_past_values"].float()
        future_time_cpu = tensors["input_future_time_features"].float()
        static_cat_cpu = tensors["input_static_categorical_features"].long()
        static_real_cpu = tensors["input_static_real_features"].float()
        cat_emb_w_cpu = ttnn.to_torch(weights["cat_embedder"]).float()
        value_proj_cpu = ttnn.to_torch(weights["decoder_value_proj"]).float()
        pos_emb_cpu = ttnn.to_torch(weights["decoder_pos_emb"]).float()
        dec_ln = weights["decoder_layernorm_ttnn"]
        dec_ln_w_cpu = ttnn.to_torch(dec_ln["weight"]).float().squeeze()
        dec_ln_b_cpu = ttnn.to_torch(dec_ln["bias"]).float().squeeze()
        static_feat_cpu = _build_static_feat_cpu(loc_cpu, scale_cpu, static_real_cpu, static_cat_cpu, cat_emb_w_cpu)

        # ── Shared buffers, one causal mask object used by BOTH arms
        #    (content-identical at each step; canonical design already
        #    reuses one shared_causal_mask across all per-layer traces) ──
        shared_causal_mask = _build_causal_mask_1tok(device, 0, T_MAX)
        precomputed_masks_host = _precompute_causal_masks_host(T_MAX)
        k_selectors_host, v_selectors_host = _precompute_kv_selectors_host(T_MAX)
        shared_k_selector = ttnn.from_torch(
            k_selectors_host[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        shared_v_selector = ttnn.from_torch(
            v_selectors_host[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # ── Baseline arm: per-layer traces ──────────────────────────────
        kv_caches_baseline = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(kv_caches_baseline, device, BS, T_MAX)
        captured_dec_input_baseline = ttnn.from_torch(
            torch.zeros(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        print("[build] capturing per-layer baseline traces...")
        trace_ids_baseline, layer_hidden_buffers_baseline = _build_per_layer_traces(
            device,
            weights,
            num_layers,
            kv_caches_baseline,
            precomputed_kv,
            captured_dec_input_baseline,
            shared_causal_mask,
        )
        trace_ids_to_release.extend(trace_ids_baseline)
        _zero_kv_caches(kv_caches_baseline, device, BS, T_MAX)

        # ── New arm: one fused trace ─────────────────────────────────────
        kv_caches_fused = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(kv_caches_fused, device, BS, T_MAX)
        captured_dec_input_fused = ttnn.from_torch(
            torch.zeros(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        print("[build] capturing ONE fused trace across all layers...")
        try:
            trace_id_fused, layer_hidden_buffers_fused = _build_fused_trace(
                device,
                weights,
                num_layers,
                kv_caches_fused,
                precomputed_kv,
                captured_dec_input_fused,
                shared_causal_mask,
                shared_k_selector,
                shared_v_selector,
            )
        except Exception as e:
            print(f"[FAIL] Fused trace capture raised: {e}")
            print("  If this is STILL 'Writes are not supported during trace capture' after")
            print("  the two-pass warmup fix, the op is genuinely untraceable as used here --")
            print("  grep its source for explicit trace-capture constraints before retrying.")
            sys.exit(1)
        trace_ids_to_release.append(trace_id_fused)
        _zero_kv_caches(kv_caches_fused, device, BS, T_MAX)
        print("[build] Fused trace captured successfully.\n")

        # ── Correctness: multi-step replay, fused vs per-layer-trace ────
        fixed_future_samples = [torch.full((BS,), 1000.0 + 37.0 * k) for k in range(N_CORRECTNESS_STEPS)]
        max_diffs = []
        for step in range(N_CORRECTNESS_STEPS):
            _update_causal_mask_precomputed(device, shared_causal_mask, precomputed_masks_host[step], cq_id=0)
            _update_selector_precomputed(shared_k_selector, k_selectors_host, step)
            print(f"[step={step}] k_selector on-device:", ttnn.to_torch(shared_k_selector).flatten()[:T_MAX])
            print(f"[step={step}] k_selector expected:      ", k_selectors_host[step].flatten())
            _update_selector_precomputed(shared_v_selector, v_selectors_host, step)
            print(f"[step={step}] v_selector on-device:", ttnn.to_torch(shared_v_selector).flatten()[:T_MAX])
            print(f"[step={step}] v_selector expected:      ", v_selectors_host[step].flatten())

            step_input_np = _prepare_dec_step_cpu_1tok(
                k=step,
                future_samples_so_far=fixed_future_samples[:step],
                past_values_cpu=past_values_cpu,
                future_time_cpu=future_time_cpu,
                static_feat_cpu=static_feat_cpu,
                loc_cpu=loc_cpu,
                scale_cpu=scale_cpu,
                value_proj_cpu=value_proj_cpu,
                dec_ln_w_cpu=dec_ln_w_cpu,
                dec_ln_b_cpu=dec_ln_b_cpu,
                pos_emb_cpu=pos_emb_cpu,
                context_length=CONTEXT_LENGTH,
                T_max=T_MAX,
            )
            step_input_tt = ttnn.from_torch(step_input_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None)
            ttnn.copy_host_to_device_tensor(step_input_tt, captured_dec_input_baseline)
            ttnn.copy_host_to_device_tensor(step_input_tt, captured_dec_input_fused)

            out_baseline = _run_per_layer_traces_one_step(
                device, weights, num_layers, kv_caches_baseline, trace_ids_baseline, layer_hidden_buffers_baseline, step
            )
            out_fused = _run_fused_trace_one_step(
                device, weights, num_layers, kv_caches_fused, trace_id_fused, layer_hidden_buffers_fused, step
            )

            base_t = ttnn.to_torch(out_baseline).float()[..., :D_MODEL]
            fused_t = ttnn.to_torch(out_fused).float()[..., :D_MODEL]
            diff = (base_t - fused_t).abs().max().item()
            max_diffs.append(diff)
            print(f"[correctness step={step}] max_abs_diff={diff:.6f}")

        overall_max = max(max_diffs)
        print(f"\n[RESULT] overall max_abs_diff across {N_CORRECTNESS_STEPS} steps: {overall_max:.6f}")
        if overall_max >= TOLERANCE:
            print(f"[FAIL] Fused trace diverges from per-layer-trace baseline beyond tolerance ({TOLERANCE}).")
            print("  Do NOT proceed to integrating this into tst_model_cached_additions.py.")
            print("  If step=0 is already wrong: overlay-inside-trace itself is broken.")
            print("  If step=0 is right but step>=1 diverges: write-back-after-execute_trace")
            print("  ordering is wrong, or a buffer isn't actually staying at a fixed address")
            print("  across replay the way the canonical per-layer design assumes.")
            sys.exit(1)
        print(
            f"[PASS] Fused single trace matches per-layer-trace baseline across {N_CORRECTNESS_STEPS} steps, "
            f"exact to {TOLERANCE} tolerance.\n"
        )

        # ── Timing: per-layer-trace baseline vs fused single trace ──────
        print(f"[timing] {WARMUP_REPLAYS} warmup replays discarded, then {TIMING_STEPS} measured replays per arm.")
        _zero_kv_caches(kv_caches_baseline, device, BS, T_MAX)
        _zero_kv_caches(kv_caches_fused, device, BS, T_MAX)

        def _timing_step_input(step):
            samples = [torch.full((BS,), 1000.0 + 37.0 * k) for k in range(step)]
            step_input_np = _prepare_dec_step_cpu_1tok(
                k=step,
                future_samples_so_far=samples,
                past_values_cpu=past_values_cpu,
                future_time_cpu=future_time_cpu,
                static_feat_cpu=static_feat_cpu,
                loc_cpu=loc_cpu,
                scale_cpu=scale_cpu,
                value_proj_cpu=value_proj_cpu,
                dec_ln_w_cpu=dec_ln_w_cpu,
                dec_ln_b_cpu=dec_ln_b_cpu,
                pos_emb_cpu=pos_emb_cpu,
                context_length=CONTEXT_LENGTH,
                T_max=T_MAX,
            )
            return ttnn.from_torch(step_input_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None)

        total_replays = WARMUP_REPLAYS + TIMING_STEPS
        baseline_times_ms, fused_times_ms = [], []

        for replay_idx in range(total_replays):
            cache_step = replay_idx % T_MAX
            step_input_tt = _timing_step_input(cache_step)
            _update_causal_mask_precomputed(device, shared_causal_mask, precomputed_masks_host[cache_step], cq_id=0)
            _update_selector_precomputed(shared_k_selector, k_selectors_host, cache_step)
            _update_selector_precomputed(shared_v_selector, v_selectors_host, cache_step)
            ttnn.copy_host_to_device_tensor(step_input_tt, captured_dec_input_baseline)

            t0 = time.perf_counter()
            _run_per_layer_traces_one_step(
                device,
                weights,
                num_layers,
                kv_caches_baseline,
                trace_ids_baseline,
                layer_hidden_buffers_baseline,
                cache_step,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if replay_idx >= WARMUP_REPLAYS:
                baseline_times_ms.append(elapsed_ms)

        _zero_kv_caches(kv_caches_baseline, device, BS, T_MAX)
        _zero_kv_caches(kv_caches_fused, device, BS, T_MAX)

        for replay_idx in range(total_replays):
            cache_step = replay_idx % T_MAX
            step_input_tt = _timing_step_input(cache_step)
            _update_causal_mask_precomputed(device, shared_causal_mask, precomputed_masks_host[cache_step], cq_id=0)
            _update_selector_precomputed(shared_k_selector, k_selectors_host, cache_step)
            _update_selector_precomputed(shared_v_selector, v_selectors_host, cache_step)
            ttnn.copy_host_to_device_tensor(step_input_tt, captured_dec_input_fused)

            t0 = time.perf_counter()
            _run_fused_trace_one_step(
                device, weights, num_layers, kv_caches_fused, trace_id_fused, layer_hidden_buffers_fused, cache_step
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if replay_idx >= WARMUP_REPLAYS:
                fused_times_ms.append(elapsed_ms)

        baseline_median = statistics.median(baseline_times_ms)
        fused_median = statistics.median(fused_times_ms)
        speedup = baseline_median / fused_median if fused_median > 0 else float("nan")

        print(
            f"\n[TIMING RESULT] per-layer-trace baseline: median={baseline_median:.3f}ms "
            f"mean={statistics.mean(baseline_times_ms):.3f}ms over {len(baseline_times_ms)} replays"
        )
        print(
            f"[TIMING RESULT] fused single trace:        median={fused_median:.3f}ms "
            f"mean={statistics.mean(fused_times_ms):.3f}ms over {len(fused_times_ms)} replays"
        )
        print(
            f"[TIMING RESULT] speedup: {speedup:.3f}x  (num_decoder_layers={num_layers}, "
            f"so max possible trace-call reduction this step is {num_layers}->1)"
        )
        print("\nNOTE: this measures per-step decode latency only, not full single-sequence")
        print("latency (which also includes encoder + embedding + distribution head). Not a")
        print("substitute for test_single_sequence_latency -- treat as a directional signal")
        print("that integrating this into tst_model_cached_additions.py is worth doing, not")
        print("as a passing/failing number against the 50ms target.")

    finally:
        for trace_id in trace_ids_to_release:
            try:
                ttnn.release_trace(device, trace_id)
            except Exception as e:
                print(f"[cleanup] release_trace failed for {trace_id}: {e}")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
