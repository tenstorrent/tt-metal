# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
DIAGNOSTIC PROBE -- NOT for the PR, lives in tests/diagnostics/ only.

QUESTION THIS ANSWERS:
Can layer i's current-step K/V be folded into attention as a pure-compute
elementwise overlay (cache-read + add, no slice_write/update_cache) instead
of the current write-then-read-cache pattern, WITHOUT changing the output?

WHY THIS MATTERS:
_extract_and_write_kv currently does a real device WRITE
(slice_write / update_cache) into the persistent K/V cache before
_attend_from_cache reads that same cache back. Writes are banned inside
begin_trace_capture/end_trace_capture, which is exactly why today's code
captures ONE TRACE PER DECODER LAYER: layer i+1's write depends on layer
i's real trace output, so a write has to happen between every layer's
trace, forcing a separate execute_trace() call per layer.

If current-step K/V can instead be injected via elementwise arithmetic
(overlay) rather than a write, there is no write occurring between
layers' compute -- meaning all layers become a single uninterrupted
compute DAG, capturable as ONE trace instead of num_decoder_layers.
The actual persistent cache write (needed so *future* steps see this
step's K/V) can then happen once, AFTER all layers finish, instead of
gating layer i+1 within the same step.

WHY THE OVERLAY IS VALID (not just a hack):
At the moment layer i attends for step k, cache position k is guaranteed
to be exactly zero -- nothing has written it yet this step (the real
write for step k only happens after this step's compute, in both the
baseline and the proposed new path). So:

    k_full = k_cache_asof_before_this_steps_write   (positions 0..k-1 real, k..T_max-1 zero)
           + k_cur_padded_at_position_k              (zero everywhere except column k)

is mathematically identical to what slice_write would have produced,
without ever writing. Same construction for v_full.

WHAT THIS PROBE DOES NOT YET PROVE:
- Trace-capturability of the overlay construction itself (this probe runs
  everything eagerly, no begin_trace_capture/end_trace_capture). Compute-only
  ops used here (linear, split, matmul, add, softmax, layer_norm) are all
  already proven traceable elsewhere in this codebase -- what's new here is
  ONLY the overlay arithmetic, which is also plain compute (read + add), so
  it should be traceable, but that must be verified as this probe's
  immediate follow-up before touching tst_model_cached_additions.py.
- Multi-layer trace capture with the overlay's required per-step "current
  step index" buffer (a shape-invariant analogue of the existing causal
  mask, needed since the overlay's placement position depends on `step`
  and traces must be shape/structure-invariant across replays).

TEACHER FORCING: both baseline and new paths are driven by IDENTICAL,
externally-fixed decoder input embeddings (not autoregressively sampled),
so a mismatch can only come from the attention math itself, not from
divergent sampling. This mirrors how test_tst_pcc.py already validates
the decoder (teacher-forced against the HF reference).

PASS CRITERION: per-layer, per-step hidden state must match between
baseline and new path to a tight tolerance (bf16 round-trip noise only --
this probe round-trips through torch for the overlay construction itself,
which the real device-side implementation would not need to do, so a
LOOSER tolerance than pure bit-exactness is expected and accepted here).
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[5]  # .../tt-metal
sys.path.insert(0, str(REPO_ROOT))

from models.demos.time_series_transformer.tt.tst_attention import (  # noqa: E402
    HEAD_DIM_PADDED,
    HEAD_DIM_TRUE,
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
    _prepare_dec_step_cpu_1tok,
    _run_single_layer_compute,
    _zero_kv_caches,
)
from models.demos.time_series_transformer.tt.ttnn_utils import layer_norm_padded  # noqa: E402

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
MODEL_REVISION = "2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35"
REFERENCE_DIR = REPO_ROOT / "models" / "demos" / "time_series_transformer" / "reference"

T_MAX = 6  # small, just enough steps to exercise the recurrence -- not a perf test
N_STEPS = 5
BS = 1  # matches the primary Stage 1 target config (B=1, S=1)


def _run_single_layer_compute_prebuilt_kv(hidden_in, query, k_full, v_full, k_pre, v_pre, causal_mask_1tok, w):
    """Identical to _run_single_layer_compute, except attention reads a
    pre-built k_full/v_full (this step's K/V already overlaid in) instead
    of reading k_cache/v_cache directly. No cache access here at all --
    this function performs zero device writes."""
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


def _overlay_kv_no_write(hidden_1tok, w_self, k_cache, v_cache, step, device):
    """
    Compute this layer's Q/K/V from hidden_1tok WITHOUT writing anything to
    the cache, then build k_full/v_full = (cache read, position `step`
    still zero) + (this step's K/V placed at position `step`).

    Uses a torch round-trip for the overlay placement -- acceptable for
    this eager, math-only probe. A trace-compatible device-side version
    (pad + add, or a dedicated one-hot-broadcast op) is the next step
    once this confirms the math itself is sound.
    """
    fused_qkv = ttnn.linear(hidden_1tok, w_self["qkv_weight"], bias=w_self["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)
    # query: [BS,H,1,D]  key: [BS,H,D,1]  value: [BS,H,1,D]

    k_cache_t = ttnn.to_torch(k_cache).float()  # [BS,H,D,T_max]
    k_cur_t = ttnn.to_torch(key).float()  # [BS,H,D,1]
    k_full_t = k_cache_t.clone()
    assert torch.all(
        k_full_t[..., step] == 0
    ), "cache position `step` must be zero before overlay -- write happened too early"
    k_full_t[..., step] = k_cur_t[..., 0]
    k_full_tt = ttnn.from_torch(k_full_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    v_cache_t = ttnn.to_torch(v_cache).float()  # [BS,H,T_max,D]
    v_cur_t = ttnn.to_torch(value).float()  # [BS,H,1,D]
    v_full_t = v_cache_t.clone()
    assert torch.all(
        v_full_t[:, :, step, :] == 0
    ), "cache position `step` must be zero before overlay -- write happened too early"
    v_full_t[:, :, step, :] = v_cur_t[:, :, 0, :]
    v_full_tt = ttnn.from_torch(v_full_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return query, k_full_tt, v_full_tt


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24_576)
    try:
        hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        hf_model.eval()
        weights = load_weights(hf_model, device)
        num_layers = sum(1 for k in weights if k.startswith("decoder.layers."))

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

        # -- baseline arm: exactly today's write-then-read-cache path --
        kv_caches_base = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(kv_caches_base, device, BS, T_MAX)
        captured_dec_input_base = ttnn.from_torch(
            torch.zeros(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # -- new arm: overlay path, separate buffers so it can't leak into baseline --
        kv_caches_new = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(kv_caches_new, device, BS, T_MAX)
        captured_dec_input_new = ttnn.from_torch(
            torch.zeros(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

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

        # Fixed, externally-supplied "teacher forcing" future values -- identical
        # for both arms, so any divergence can only come from attention math,
        # never from autoregressive feedback differing between the two paths.
        fixed_future_samples = [torch.full((BS,), 1000.0 + 37.0 * k) for k in range(N_STEPS)]

        max_diffs = []
        for step in range(N_STEPS):
            causal_mask = _build_causal_mask_1tok(device, step, T_MAX)

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
            ttnn.copy_host_to_device_tensor(step_input_tt, captured_dec_input_base)
            ttnn.copy_host_to_device_tensor(step_input_tt, captured_dec_input_new)

            hidden_base = captured_dec_input_base
            hidden_new = captured_dec_input_new

            for layer_idx in range(num_layers):
                w_layer = weights[f"decoder.layers.{layer_idx}"]
                w_self = w_layer["self_attn"]
                k_cache_b, v_cache_b = kv_caches_base[layer_idx]
                k_cache_n, v_cache_n = kv_caches_new[layer_idx]
                k_pre, v_pre = precomputed_kv[layer_idx]

                # -- baseline: write first, then read the (now-updated) cache --
                q_base = _extract_and_write_kv(hidden_base, w_self, k_cache_b, v_cache_b, step=step, cq_id=0)
                hidden_base = _run_single_layer_compute(
                    hidden_base, q_base, k_cache_b, v_cache_b, k_pre, v_pre, causal_mask, w_layer
                )

                # -- new: overlay (no write), compute, THEN write for future steps --
                layer_input_new = (
                    hidden_new  # this layer's real input, BEFORE it gets reassigned to this layer's output below
                )
                q_new, k_full, v_full = _overlay_kv_no_write(
                    layer_input_new, w_self, k_cache_n, v_cache_n, step, device
                )
                hidden_new = _run_single_layer_compute_prebuilt_kv(
                    layer_input_new, q_new, k_full, v_full, k_pre, v_pre, causal_mask, w_layer
                )
                _extract_and_write_kv(  # real persistent write, now AFTER this step's compute -- uses the SAME input the overlay used
                    layer_input_new,
                    w_self,
                    k_cache_n,
                    v_cache_n,
                    step=step,
                    cq_id=0,
                )

                base_t = ttnn.to_torch(hidden_base).float()[..., :D_MODEL]
                new_t = ttnn.to_torch(hidden_new).float()[..., :D_MODEL]
                diff = (base_t - new_t).abs().max().item()
                max_diffs.append(diff)
                print(f"[step={step} layer={layer_idx}] max_abs_diff={diff:.6f}")

        overall_max = max(max_diffs)
        print(f"\n[RESULT] overall max_abs_diff across all steps/layers: {overall_max:.6f}")
        TOLERANCE = 0.05  # loose: bf16 + extra torch round-trip in the overlay construction
        if overall_max < TOLERANCE:
            print(f"[PASS] Overlay path matches write-then-read path within tolerance ({TOLERANCE}).")
        else:
            print(f"[FAIL] Overlay path diverges beyond tolerance ({TOLERANCE}) -- do NOT proceed to trace fusion yet.")
            sys.exit(1)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
