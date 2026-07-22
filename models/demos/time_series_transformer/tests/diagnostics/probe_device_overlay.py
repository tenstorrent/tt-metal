# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
DIAGNOSTIC PROBE 2 -- NOT for the PR, lives in tests/diagnostics/ only.

QUESTION THIS ANSWERS:
Probe 1 (probe_single_trace_fusion.py) proved the write-then-read cache
pattern and a host-round-trip overlay produce identical output. That
overlay used ttnn.to_torch / ttnn.from_torch mid-computation, which is
NOT trace-compatible (traces cannot call back to host during capture).

This probe replaces the host round-trip with a pure device-side overlay:
    k_overlay = ttnn.mul(k_cur, k_selector)   # broadcast to [BS,H,D,T_max]
    k_full    = ttnn.add(k_cache, k_overlay)  # cache position `step` is
                                               # zero (proven in probe 1),
                                               # so this equals a write
    v_overlay = ttnn.mul(v_cur, v_selector)   # broadcast to [BS,H,T_max,D]
    v_full    = ttnn.add(v_cache, v_overlay)

k_selector / v_selector are one-hot tensors (1 at position `step`, 0
elsewhere), precomputed on host for all T_max steps and swapped into a
FIXED-ADDRESS device buffer via copy_host_to_device_tensor -- exactly the
same lifecycle already proven for shared_causal_mask in
_update_causal_mask_precomputed. Nothing new architecturally; this is
that pattern applied to two more buffers.

WHY THIS MATTERS:
If this overlay is provably (a) correct and (b) built from zero host
round-trips, it is a pure compute op sequence and therefore a candidate
for living inside begin_trace_capture/end_trace_capture across all
decoder layers -- which is the actual blocker on the 50ms latency target.
This probe does NOT capture a trace yet. It only proves the device-side
overlay arithmetic is correct in eager mode. Trace capture is probe 3.

PREFLIGHT CHECK (runs first, aborts before touching the model if it fails):
Verifies ttnn.mul actually broadcasts [BS,H,D,1] x [1,1,1,T_max] and
[BS,H,1,D] x [1,1,T_max,1] the way NumPy-style broadcasting would. This
assumption was NOT verified against source before writing this probe --
if the assertion fails, do not patch around it here. Stop, grep
ttnn.mul's implementation, and find the actual supported broadcast shape,
then fix _overlay_kv_device accordingly.

TEACHER FORCING: identical to probe 1 -- both arms driven by the same
fixed, externally-supplied future values. Any mismatch can only come
from the overlay arithmetic itself.

TOLERANCE: tighter than probe 1 (0.05) because there is no host round
trip in the new arm anymore. Expect near bit-exact bf16 agreement.
Set to 0.01 to allow for op-ordering rounding (mul+add vs write), not
for round-trip slop.
"""

import sys
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
    _prepare_dec_step_cpu_1tok,
    _run_single_layer_compute,
    _zero_kv_caches,
)
from models.demos.time_series_transformer.tt.ttnn_utils import layer_norm_padded  # noqa: E402

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
MODEL_REVISION = "2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35"
REFERENCE_DIR = REPO_ROOT / "models" / "demos" / "time_series_transformer" / "reference"

T_MAX = 6
N_STEPS = 5
BS = 1
TOLERANCE = 0.01


def _verify_broadcast_mul_semantics(device):
    """
    Preflight gate. Proves ttnn.mul broadcasts the two shapes this probe
    depends on. Aborts loudly and specifically if not -- no silent fallback.
    """
    print("[preflight] Verifying ttnn.mul broadcast semantics on real hardware...")

    # K-axis case: [BS,H,D,1] x [1,1,1,T_max] -> expect [BS,H,D,T_max]
    a_np = torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 4, 1)  # [1,1,4,1]
    b_np = torch.zeros(1, 1, 1, 3, dtype=torch.float32)
    b_np[..., 1] = 1.0  # one-hot at index 1
    a_tt = ttnn.from_torch(a_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        out_tt = ttnn.mul(a_tt, b_tt)
    except Exception as e:
        print(f"[preflight] FAIL: ttnn.mul raised on K-axis broadcast shape: {e}")
        print("  grep ttnn.mul's implementation for supported broadcast rank/shape rules")
        print("  before touching _overlay_kv_device.")
        sys.exit(1)
    out_np = ttnn.to_torch(out_tt).float()
    expected = a_np * b_np  # NumPy-style broadcast reference
    diff = (out_np - expected).abs().max().item()
    if diff > 1e-3 or out_np.shape != expected.shape:
        print(
            f"[preflight] FAIL: K-axis broadcast mismatch. shape={out_np.shape} "
            f"expected={expected.shape} max_diff={diff:.6f}"
        )
        print("  ttnn.mul did not broadcast the way NumPy would here.")
        print("  Do not proceed -- fix _overlay_kv_device's shapes first.")
        sys.exit(1)
    print(f"[preflight] K-axis broadcast OK. shape={out_np.shape} max_diff={diff:.6f}")

    # V-axis case: [BS,H,1,D] x [1,1,T_max,1] -> expect [BS,H,T_max,D]
    c_np = torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 1, 4)  # [1,1,1,4]
    d_np = torch.zeros(1, 1, 3, 1, dtype=torch.float32)
    d_np[:, :, 1, :] = 1.0
    c_tt = ttnn.from_torch(c_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    d_tt = ttnn.from_torch(d_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        out2_tt = ttnn.mul(c_tt, d_tt)
    except Exception as e:
        print(f"[preflight] FAIL: ttnn.mul raised on V-axis broadcast shape: {e}")
        sys.exit(1)
    out2_np = ttnn.to_torch(out2_tt).float()
    expected2 = c_np * d_np
    diff2 = (out2_np - expected2).abs().max().item()
    if diff2 > 1e-3 or out2_np.shape != expected2.shape:
        print(
            f"[preflight] FAIL: V-axis broadcast mismatch. shape={out2_np.shape} "
            f"expected={expected2.shape} max_diff={diff2:.6f}"
        )
        sys.exit(1)
    print(f"[preflight] V-axis broadcast OK. shape={out2_np.shape} max_diff={diff2:.6f}")
    print("[preflight] All broadcast assumptions verified. Proceeding.\n")


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
    """
    One-hot selector per step, for both cache axes. Same lifecycle as
    _precompute_causal_masks_host: build T_max fixed tensors on host once,
    swap the active one into a fixed device buffer per step.
    """
    k_selectors = []  # each [1,1,1,T_max], one-hot at position `step`
    v_selectors = []  # each [1,1,T_max,1], one-hot at position `step`
    for step in range(T_max):
        k_sel = torch.zeros(1, 1, 1, T_max, dtype=torch.bfloat16)
        k_sel[..., step] = 1.0
        k_selectors.append(k_sel)

        v_sel = torch.zeros(1, 1, T_max, 1, dtype=torch.bfloat16)
        v_sel[:, :, step, :] = 1.0
        v_selectors.append(v_sel)
    return k_selectors, v_selectors


def _update_selector_precomputed(shared_dev_buf, selector_host_list, step):
    """
    Mirrors _update_causal_mask_precomputed's fixed-address swap pattern.
    NOTE: verify this matches the real _update_causal_mask_precomputed
    signature/behavior (grepped above) before trusting this in a trace --
    for this eager probe it's equivalent, just written inline instead of
    imported, since the K/V selector buffers are new and have no existing
    helper.
    """
    host_tensor = ttnn.from_torch(selector_host_list[step], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None)
    ttnn.copy_host_to_device_tensor(host_tensor, shared_dev_buf)


def _overlay_kv_device(hidden_1tok, w_self, k_cache, v_cache, k_selector_dev, v_selector_dev):
    """
    Pure device-side overlay. Zero ttnn.to_torch / ttnn.from_torch calls.
    Depends entirely on the preflight-verified broadcast behavior above.
    """
    fused_qkv = ttnn.linear(hidden_1tok, w_self["qkv_weight"], bias=w_self["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)
    # query: [BS,H,1,D]  key: [BS,H,D,1]  value: [BS,H,1,D]

    k_overlay = ttnn.mul(key, k_selector_dev)  # [BS,H,D,1] x [1,1,1,T_max] -> [BS,H,D,T_max]
    k_full = ttnn.add(k_cache, k_overlay)  # cache position `step` is zero (proven in probe 1)

    v_overlay = ttnn.mul(value, v_selector_dev)  # [BS,H,1,D] x [1,1,T_max,1] -> [BS,H,T_max,D]
    v_full = ttnn.add(v_cache, v_overlay)

    return query, k_full, v_full


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24_576)
    try:
        _verify_broadcast_mul_semantics(device)

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

        # -- new arm: device-side overlay, no host round-trip --
        kv_caches_new = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(kv_caches_new, device, BS, T_MAX)
        captured_dec_input_new = ttnn.from_torch(
            torch.zeros(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Selector precompute + fixed-address device buffers, shared across
        # all layers within a step (position `step` doesn't vary per layer).
        k_selectors_host, v_selectors_host = _precompute_kv_selectors_host(T_MAX)
        shared_k_selector = ttnn.from_torch(
            k_selectors_host[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        shared_v_selector = ttnn.from_torch(
            v_selectors_host[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
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

        fixed_future_samples = [torch.full((BS,), 1000.0 + 37.0 * k) for k in range(N_STEPS)]

        max_diffs = []
        for step in range(N_STEPS):
            causal_mask = _build_causal_mask_1tok(device, step, T_MAX)
            _update_selector_precomputed(shared_k_selector, k_selectors_host, step)
            _update_selector_precomputed(shared_v_selector, v_selectors_host, step)

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

                q_base = _extract_and_write_kv(hidden_base, w_self, k_cache_b, v_cache_b, step=step, cq_id=0)
                hidden_base = _run_single_layer_compute(
                    hidden_base, q_base, k_cache_b, v_cache_b, k_pre, v_pre, causal_mask, w_layer
                )

                layer_input_new = hidden_new
                q_new, k_full, v_full = _overlay_kv_device(
                    layer_input_new, w_self, k_cache_n, v_cache_n, shared_k_selector, shared_v_selector
                )
                hidden_new = _run_single_layer_compute_prebuilt_kv(
                    layer_input_new, q_new, k_full, v_full, k_pre, v_pre, causal_mask, w_layer
                )
                _extract_and_write_kv(layer_input_new, w_self, k_cache_n, v_cache_n, step=step, cq_id=0)

                base_t = ttnn.to_torch(hidden_base).float()[..., :D_MODEL]
                new_t = ttnn.to_torch(hidden_new).float()[..., :D_MODEL]
                diff = (base_t - new_t).abs().max().item()
                max_diffs.append(diff)
                print(f"[step={step} layer={layer_idx}] max_abs_diff={diff:.6f}")

        overall_max = max(max_diffs)
        print(f"\n[RESULT] overall max_abs_diff across all steps/layers: {overall_max:.6f}")
        if overall_max < TOLERANCE:
            print(
                f"[PASS] Device-side overlay matches baseline within tolerance ({TOLERANCE}). No host round-trip needed."
            )
            print("[NEXT] Probe 3: wrap this per-step, all-layer sequence in begin_trace_capture/end_trace_capture.")
        else:
            print(f"[FAIL] Device-side overlay diverges beyond tolerance ({TOLERANCE}).")
            print("  Check preflight output above -- if broadcast passed but this failed,")
            print("  the divergence is likely op-ordering (mul+add vs write) exceeding bf16 slop,")
            print("  or the selector buffer isn't actually being swapped before each step's use.")
            sys.exit(1)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
