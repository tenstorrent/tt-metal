# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
DIAGNOSTIC PROBE -- NOT for the PR, lives in tests/diagnostics/ only.

QUESTION THIS ANSWERS:
probe_fused_trace.py's overlay arm diverges from the per-layer-trace
baseline by ~0.8-0.9 max_abs_diff, present at step 0, unchanged after
fixing the k_cache ROW_MAJOR/TILE layout mismatch. The remaining
unverified piece: `key` out of split_query_key_value_and_split_heads has
LOGICAL shape [BS,H,D,1] but TILE layout pads the last dim to 32. probe 2
validated ttnn.mul's NumPy-style broadcast only on a from-scratch mock
tensor built at full width via ttnn.from_torch(..., layout=TILE_LAYOUT) --
never on this op's genuinely narrow, padded-to-32 real output. This probe
isolates exactly that: real split_query_key_value_and_split_heads output x
ttnn.mul(., one_hot_selector), nothing else -- no cache, no trace, no
multi-layer chaining.

PASS CONDITION: for a chosen step index, ttnn.mul(key, k_selector) must be
exactly zero in every column except `step`, and column `step` must exactly
equal key's own values (same check for value/v_selector on the row axis).
FAIL on either narrows the root cause to this exact op pair.
"""

import sys
from pathlib import Path

import torch

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[5]  # .../tt-metal
sys.path.insert(0, str(REPO_ROOT))

from models.demos.time_series_transformer.tt.tst_attention import NUM_HEADS  # noqa: E402
from models.demos.time_series_transformer.tt.tst_model import D_MODEL, PADDED_WIDTH  # noqa: E402

BS = 1
T_MAX = 8
STEP = 3  # arbitrary, non-zero on purpose -- step 0 alone can't rule out an
# off-by-one in selector placement
HEAD_DIM_PADDED = PADDED_WIDTH // NUM_HEADS
TOLERANCE = 1e-3


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24_576)
    try:
        torch.manual_seed(0)
        hidden_cpu = torch.randn(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16)
        qkv_weight_cpu = torch.randn(PADDED_WIDTH, 3 * PADDED_WIDTH, dtype=torch.bfloat16) * 0.02
        qkv_bias_cpu = torch.randn(3 * PADDED_WIDTH, dtype=torch.bfloat16) * 0.02

        hidden = ttnn.from_torch(hidden_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        qkv_weight = ttnn.from_torch(qkv_weight_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        qkv_bias = ttnn.from_torch(
            qkv_bias_cpu.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        fused_qkv = ttnn.linear(hidden, qkv_weight, bias=qkv_bias)
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)

        print(f"[shapes] key.shape={tuple(key.shape)}  value.shape={tuple(value.shape)}")
        print(f"[shapes] key.padded_shape={tuple(key.padded_shape)}  value.padded_shape={tuple(value.padded_shape)}")

        key_cpu = ttnn.to_torch(key).float()  # expect [BS, NUM_HEADS, HEAD_DIM_PADDED, 1]
        value_cpu = ttnn.to_torch(value).float()  # expect [BS, NUM_HEADS, 1, HEAD_DIM_PADDED]

        # ── K: selector along last axis (T_max), one-hot at STEP ──────────
        k_sel_cpu = torch.zeros(1, 1, 1, T_MAX, dtype=torch.bfloat16)
        k_sel_cpu[..., STEP] = 1.0
        k_sel = ttnn.from_torch(k_sel_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        k_overlay = ttnn.mul(key, k_sel)
        k_overlay_cpu = ttnn.to_torch(k_overlay).float()  # expect [BS, NUM_HEADS, HEAD_DIM_PADDED, T_MAX]
        print(f"[shapes] k_overlay.shape={tuple(k_overlay_cpu.shape)}")

        k_selected_col = k_overlay_cpu[..., STEP]
        k_other_cols = torch.cat([k_overlay_cpu[..., :STEP], k_overlay_cpu[..., STEP + 1 :]], dim=-1)

        k_col_diff = (k_selected_col - key_cpu[..., 0]).abs().max().item()
        k_leak = k_other_cols.abs().max().item()
        print(f"[K] selected-column vs real key max_abs_diff={k_col_diff:.6f}")
        print(f"[K] non-selected-columns max_abs leak={k_leak:.6f}")

        # ── V: selector along the T_max axis (second-to-last), one-hot at STEP ─
        v_sel_cpu = torch.zeros(1, 1, T_MAX, 1, dtype=torch.bfloat16)
        v_sel_cpu[:, :, STEP, :] = 1.0
        v_sel = ttnn.from_torch(v_sel_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        v_overlay = ttnn.mul(value, v_sel)
        v_overlay_cpu = ttnn.to_torch(v_overlay).float()  # expect [BS, NUM_HEADS, T_MAX, HEAD_DIM_PADDED]
        print(f"[shapes] v_overlay.shape={tuple(v_overlay_cpu.shape)}")

        v_selected_row = v_overlay_cpu[:, :, STEP, :]
        v_other_rows = torch.cat([v_overlay_cpu[:, :, :STEP, :], v_overlay_cpu[:, :, STEP + 1 :, :]], dim=2)

        v_row_diff = (v_selected_row - value_cpu[:, :, 0, :]).abs().max().item()
        v_leak = v_other_rows.abs().max().item()
        print(f"[V] selected-row vs real value max_abs_diff={v_row_diff:.6f}")
        print(f"[V] non-selected-rows max_abs leak={v_leak:.6f}")

        ok = k_col_diff < TOLERANCE and k_leak < TOLERANCE and v_row_diff < TOLERANCE and v_leak < TOLERANCE
        print(f"\n[RESULT] {'PASS' if ok else 'FAIL'} (tolerance={TOLERANCE})")
        if not ok:
            print("  ttnn.mul broadcast on the real split_query_key_value_and_split_heads")
            print("  output does NOT behave as a clean logical-shape broadcast -- this is")
            print("  the root cause of probe_fused_trace.py's step-0 divergence.")
            sys.exit(1)
        print("  Broadcast is clean on this op's real output -- the padding hypothesis is")
        print("  falsified. Root cause is still open; check k_selector_dev's actual on-device")
        print("  shape/dtype next.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
