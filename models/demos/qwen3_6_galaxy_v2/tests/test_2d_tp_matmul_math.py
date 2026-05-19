# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only pre-flight for 2D-TP DeltaNet matmul math.

Validates that splitting a matmul ``y = W @ x`` across both mesh axes
(rows on output dim, cols on input dim) gives the same result as the
unsplit baseline, within bf16 noise.

Specifically checks the three DeltaNet projections that get re-sharded
in V2-DN-TP:

  - ``w_qkvz``: [16384, 5120] @ [5120] → [16384]
    rows split output 8-way, cols split input 4-way
  - ``w_ba``:   [96, 5120]    @ [5120] → [96]
    rows split output 8-way, cols split input 4-way (only 8 N tiles, so
    rows still split)
  - ``w_out``:  [5120, 6144]  @ [6144] → [5120]
    rows split input 8-way, cols split output 4-way

For each weight the check is:

  baseline:  y = W @ x                        (fp32 reference)
  2D-TP:     per-chip y_partial = W_chip @ x_chip
             col-axis sum → y_row
             concat rows → y_2d
  assert     allclose(y_2d, y, atol_bf16)

Runs on CPU in <1 s.  Use as a sanity check BEFORE the device-level
re-sharding work in V2-DN-TP.
"""
from __future__ import annotations

import pytest
import torch

# --- Mesh + DeltaNet shape constants ----------------------------------------
ROWS = 8  # cluster_axis=0
COLS = 4  # cluster_axis=1
H = 5120  # hidden dim
QKV_DIM = 16 * 128 + 16 * 128 + 48 * 128  # 10240
Z_DIM = 48 * 128  # 6144
QKVZ_OUT = QKV_DIM + Z_DIM  # 16384
BA_OUT = 2 * 48  # 96
VAL_DIM = 48 * 128  # 6144 (DeltaNet value dim → w_out input)

# bf16 tolerance: ~2^-7 per element, accumulated over K terms.
# rms((W@x).bf16 vs (W@x).fp32) typically ~K * 2^-7 * sigma^2.
_BF16_ATOL = 5e-2  # generous; we don't need bit-identity here, just direction


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def _1d_tp_rowsplit(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Today's DeltaNet: rows split output dim, cols replicate the input."""
    # W: [N_out, K_in].  Split N_out across rows → N_out/ROWS per row chip.
    # Each row chip computes y_row = W_row @ x_full.
    # 4 col chips all see the SAME x_full and produce the SAME y_row.
    # The y_row's then concat across rows → full y.
    N_out = W.shape[0]
    assert N_out % ROWS == 0
    rows_y = []
    for r in range(ROWS):
        W_row = W[r * (N_out // ROWS) : (r + 1) * (N_out // ROWS), :]
        # Single col chip's matmul (cols just replicate; pick col 0).
        y_row = _bf16(W_row) @ _bf16(x)
        rows_y.append(y_row)
    return torch.cat(rows_y, dim=0)


def _2d_tp_proposed(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """V2-DN-TP: rows split output dim, cols split INPUT dim.

    Each of 32 chips computes a partial: W_chip @ x_chip.
    Cols sum on the K axis → y_row.  Rows concat → y.
    """
    N_out, K_in = W.shape
    assert N_out % ROWS == 0, f"N={N_out} not divisible by ROWS={ROWS}"
    assert K_in % COLS == 0, f"K={K_in} not divisible by COLS={COLS}"
    rows_y = []
    for r in range(ROWS):
        W_row = W[r * (N_out // ROWS) : (r + 1) * (N_out // ROWS), :]  # [N/8, K]
        # cols split the K-axis: each col-chip gets W_row[:, c*K/4:(c+1)*K/4] @ x[c*K/4:(c+1)*K/4]
        col_partials = []
        for c in range(COLS):
            W_chip = W_row[:, c * (K_in // COLS) : (c + 1) * (K_in // COLS)]
            x_chip = x[c * (K_in // COLS) : (c + 1) * (K_in // COLS)]
            y_partial = _bf16(W_chip) @ _bf16(x_chip)  # [N/8]
            col_partials.append(y_partial)
        # all_reduce across cols (sum) → y_row for this row's 4 col chips
        y_row = torch.stack(col_partials, dim=0).sum(dim=0)
        rows_y.append(y_row)
    return torch.cat(rows_y, dim=0)


def _2d_tp_proposed_row_input_col_output(W: torch.Tensor, x_row_sharded: torch.Tensor) -> torch.Tensor:
    """V2-DN-TP w_out: rows split INPUT dim, cols split OUTPUT dim.

    Input ``x_row_sharded`` is the **post-readout, per-head** tensor —
    different on each row (heads are split across rows).  W is the output
    projection ``[H, VAL_DIM]`` and the per-chip slice is
    ``[H/COLS, VAL_DIM/ROWS]`` — rows pick the input slice for that
    row's heads, cols pick the output slice (H/4 = 1280).

    Per chip: y_partial = W_chip @ x_row_chip
    Rows sum on N=VAL_DIM axis (the head reduction) → y_col for each
    col chip.  Result: each col-chip holds H/4 of the full output;
    different cols hold different H/4 slices.

    Returns the assembled full-H output (col-sharded slices concatenated).
    """
    H_out, K_in = W.shape  # [5120, 6144]
    assert H_out % COLS == 0
    assert K_in % ROWS == 0
    cols_y = []
    for c in range(COLS):
        W_col = W[c * (H_out // COLS) : (c + 1) * (H_out // COLS), :]  # [H/4, K]
        row_partials = []
        for r in range(ROWS):
            W_chip = W_col[:, r * (K_in // ROWS) : (r + 1) * (K_in // ROWS)]
            x_chip = x_row_sharded[r * (K_in // ROWS) : (r + 1) * (K_in // ROWS)]
            y_partial = _bf16(W_chip) @ _bf16(x_chip)  # [H/4]
            row_partials.append(y_partial)
        y_col = torch.stack(row_partials, dim=0).sum(dim=0)  # row all-reduce
        cols_y.append(y_col)
    return torch.cat(cols_y, dim=0)


@pytest.mark.cpu_only
def test_2d_tp_wqkvz_matches_baseline():
    """w_qkvz: 2D-TP (rows on N, cols on K) === 1D-TP (rows on N only)."""
    torch.manual_seed(0)
    W = torch.randn(QKVZ_OUT, H, dtype=torch.float32) * 0.02
    x = torch.randn(H, dtype=torch.float32) * 0.5

    y_baseline = _bf16(W) @ _bf16(x)
    y_1d = _1d_tp_rowsplit(W, x)
    y_2d = _2d_tp_proposed(W, x)

    diff_1d = (y_1d - y_baseline).abs().max().item()
    diff_2d = (y_2d - y_baseline).abs().max().item()
    print(f"[w_qkvz] max|y_1d - y_baseline| = {diff_1d:.4e}")
    print(f"[w_qkvz] max|y_2d - y_baseline| = {diff_2d:.4e}")
    # 1D-TP and 2D-TP should both match the baseline within bf16 noise.
    assert diff_1d < _BF16_ATOL, f"1D-TP diff {diff_1d:.4e} > tol {_BF16_ATOL:.4e}"
    assert diff_2d < _BF16_ATOL, f"2D-TP diff {diff_2d:.4e} > tol {_BF16_ATOL:.4e}"


@pytest.mark.cpu_only
def test_2d_tp_wqkvz_per_chip_shapes():
    """Sanity: per-chip slice sizes match the documented shapes."""
    assert QKVZ_OUT % ROWS == 0
    assert H % COLS == 0
    n_per_row = QKVZ_OUT // ROWS
    k_per_chip = H // COLS
    assert n_per_row == 2048, f"per-row N = {n_per_row}, expected 2048"
    assert k_per_chip == 1280, f"per-chip K = {k_per_chip}, expected 1280"


@pytest.mark.cpu_only
def test_2d_tp_wba_matches_baseline():
    """w_ba is the smallest matmul — verify the split math still holds.

    N_out=96 doesn't divide evenly by 8 → per row 12 outputs.  Cols=4 still
    works for K=5120.
    """
    torch.manual_seed(1)
    W = torch.randn(BA_OUT, H, dtype=torch.float32) * 0.02
    x = torch.randn(H, dtype=torch.float32) * 0.5

    y_baseline = _bf16(W) @ _bf16(x)
    y_2d = _2d_tp_proposed(W, x)
    diff = (y_2d - y_baseline).abs().max().item()
    print(f"[w_ba] max|y_2d - y_baseline| = {diff:.4e}")
    assert diff < _BF16_ATOL, f"w_ba 2D-TP diff {diff:.4e} > tol {_BF16_ATOL:.4e}"


@pytest.mark.cpu_only
def test_2d_tp_wout_matches_baseline():
    """w_out: rows split input (head reduction), cols split output (H/4)."""
    torch.manual_seed(2)
    W = torch.randn(H, VAL_DIM, dtype=torch.float32) * 0.02
    # The "input" to w_out is per-head (VAL_DIM=6144).  In production, each
    # row chip already holds its head subset — meaning rows 0..7 each have
    # 768 of VAL_DIM, and within a row cols replicate.  To mimic this for
    # the math check we build a full VAL_DIM-sized input and slice per-row
    # inside ``_2d_tp_proposed_row_input_col_output``.
    x = torch.randn(VAL_DIM, dtype=torch.float32) * 0.5

    y_baseline = _bf16(W) @ _bf16(x)
    y_2d = _2d_tp_proposed_row_input_col_output(W, x)
    diff = (y_2d - y_baseline).abs().max().item()
    print(f"[w_out] max|y_2d - y_baseline| = {diff:.4e}")
    assert diff < _BF16_ATOL, f"w_out 2D-TP diff {diff:.4e} > tol {_BF16_ATOL:.4e}"


@pytest.mark.cpu_only
def test_2d_tp_ccl_volume_estimate():
    """Sanity-check the CCL volume we'd add per DeltaNet layer.

    Each col-axis all_reduce moves the per-chip output through a 4-way
    ring.  After all_reduce, each col chip holds the same result.  The
    payload per chip per ring-pass is ``per_chip_N * 2 bytes (bf16)``.
    """
    n_qkvz = QKVZ_OUT // ROWS  # 2048
    n_ba = BA_OUT // ROWS  # 12 (padded up for tile alignment in production)
    n_wout = H // COLS  # 1280

    payload_qkvz_bytes = n_qkvz * 2
    payload_ba_bytes = n_ba * 2
    payload_wout_bytes = n_wout * 2

    # Rough latency: BH inter-chip eth is ~25 GB/s sustained per link;
    # 4-way ring needs 3 hops; per-call overhead ~30 µs.
    def _ring_latency_us(payload_bytes, ring_size, bw_gbs=25.0, overhead_us=30.0):
        hops = ring_size - 1
        return overhead_us + hops * (payload_bytes / (bw_gbs * 1e3))

    lat_qkvz = _ring_latency_us(payload_qkvz_bytes, COLS)
    lat_ba = _ring_latency_us(payload_ba_bytes, COLS)
    lat_wout = _ring_latency_us(payload_wout_bytes, ROWS)  # existing 8-way

    per_layer_added_us = lat_qkvz + lat_ba  # both NEW col-axis all_reduces
    print(f"[CCL est] new col all_reduce after w_qkvz: ~{lat_qkvz:.1f} us")
    print(f"[CCL est] new col all_reduce after w_ba:   ~{lat_ba:.1f} us")
    print(f"[CCL est] existing row all_reduce after w_out: ~{lat_wout:.1f} us")
    print(f"[CCL est] added per DeltaNet layer: ~{per_layer_added_us:.1f} us")
    print(f"[CCL est] added across 48 DeltaNet layers: ~{per_layer_added_us * 48 / 1000:.2f} ms")
    # Just informational; don't assert.


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-x"])
