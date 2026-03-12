# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DPRINT investigation: fp32 Wt=2, detailed error analysis.
Run with: TT_METAL_DPRINT_CORES=0,0 TT_METAL_DPRINT_RISCVS=TR0,TR2 scripts/tt-test.sh ...
"""

import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

SHAPE = (1, 1, 32, 64)  # Wt=2, 1 tile-row


def pytorch_rms_norm(x, epsilon=1e-6):
    x = x.float()
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + epsilon)
    return x / rms


def test_fp32_rm_wt2_error_map(device):
    """Detailed error analysis for fp32 RM Wt=2."""
    torch.manual_seed(42)
    x = torch.randn(*SHAPE, dtype=torch.float32)
    expected = pytorch_rms_norm(x)

    x_tt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(x_tt, gamma=None, epsilon=1e-6)
    actual = ttnn.to_torch(out).float()

    diff = (actual - expected).abs()
    rel_err = diff / (expected.abs() + 1e-10)

    print(f"\n=== ERROR MAP (1x1x32x64) ===")
    print(f"Max abs error: {diff.max().item():.6e}")
    print(f"Zero fraction: {(actual.abs() < 1e-30).float().mean().item():.4f}")
    print(f"")

    # Print per-row max error and which column has the max
    print(f"Row | MaxErr col0-31 | MaxErr col32-63 | Tile0 ok? | Tile1 ok?")
    print(f"----|----------------|-----------------|-----------|----------")
    for row in range(32):
        t0_err = diff[0, 0, row, :32].max().item()
        t1_err = diff[0, 0, row, 32:].max().item()
        t0_ok = "YES" if t0_err < 0.1 else f"NO ({t0_err:.2e})"
        t1_ok = "YES" if t1_err < 0.1 else f"NO ({t1_err:.2e})"
        print(f" {row:2d} | {t0_err:14.6e} | {t1_err:15.6e} | {t0_ok:9s} | {t1_ok}")

    print(f"\n=== SAMPLE: row 0, all 64 cols ===")
    print(f"actual:   {actual[0,0,0,:8].tolist()}")
    print(f"expected: {expected[0,0,0,:8].tolist()}")
    print(f"actual[32:40]:   {actual[0,0,0,32:40].tolist()}")
    print(f"expected[32:40]: {expected[0,0,0,32:40].tolist()}")

    print(f"\n=== SAMPLE: row 16, all 64 cols ===")
    print(f"actual:   {actual[0,0,16,:8].tolist()}")
    print(f"expected: {expected[0,0,16,:8].tolist()}")
    print(f"actual[32:40]:   {actual[0,0,16,32:40].tolist()}")
    print(f"expected[32:40]: {expected[0,0,16,32:40].tolist()}")
