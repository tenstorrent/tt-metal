# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for nomic_moe_dispatch (verifier).

The op is a gather plus a tilize: no arithmetic, so the expected result is bit-exact (PCC 1, every error 0, every
got/true ratio exactly 1). Shapes cover a ragged window, the k = 1 (whole-row) range, a k = 3 perf shape and the
k = 2 perf focus. The compute-config cases check that the values do not depend on compute_kernel_config: the
standard (not fast) tilize under dst_full_sync_en=True, and fp32 DEST accumulation, both still bit-exact.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.nomic_moe_dispatch import default_compute_kernel_config, nomic_moe_dispatch, plan

HIDDEN = 768
TILE = 32


def rows_for_tokens(tokens: int) -> int:
    return TILE * (-(-2 * tokens // TILE) + 7)


def _index(tokens: int, rows: int, seed: int = 7) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, tokens, (rows,), generator=generator).tolist()


def _config(variant: str):
    config = default_compute_kernel_config()
    if variant == "full_sync":
        config.dst_full_sync_en = True  # the tilize helper falls back to the standard tilize
    elif variant == "fp32_dest":
        config.fp32_dest_acc_en = True
        config.math_fidelity = ttnn.MathFidelity.HiFi4
    return config


def _measure(device, tokens: int, rows: int, variant: str = "default"):
    torch.manual_seed(1234)
    table = torch.randn(1, 1, tokens, HIDDEN).to(torch.bfloat16)
    gather = _index(tokens, rows)
    tt_table = ttnn.from_torch(
        table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_index = ttnn.from_torch(
        torch.tensor(gather, dtype=torch.int32).reshape(1, rows),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = nomic_moe_dispatch(tt_table, tt_index, compute_kernel_config=_config(variant))
    got = ttnn.to_torch(output).reshape(rows, HIDDEN)
    want = table.reshape(-1, HIDDEN)[torch.tensor(gather, dtype=torch.int64)]
    for tensor in (tt_table, tt_index, output):
        ttnn.deallocate(tensor)
    return got, want


def _report(tag: str, got: torch.Tensor, want: torch.Tensor):
    _, pcc = assert_with_pcc(want.float(), got.float(), pcc=0.99999)
    allclose_ok, allclose_msg = comp_allclose(got.float(), want.float(), rtol=0.0, atol=0.0)
    diff = (got.float() - want.float()).abs()
    rel_rms = float(diff.pow(2).mean().sqrt() / want.float().pow(2).mean().sqrt())
    bits_equal = got.view(torch.int16) == want.view(torch.int16)
    ulp_max = int((got.view(torch.int16).int() - want.view(torch.int16).int()).abs().max())
    nonzero = want.float() != 0
    ratio = got.float()[nonzero] / want.float()[nonzero]
    print(
        f"PRECISION {tag}: pcc={pcc} max_abs={float(diff.max()):.3e} mean_abs={float(diff.mean()):.3e} "
        f"rel_rms={rel_rms:.3e} ulp_max={ulp_max} bit_exact={bool(bits_equal.all())} "
        f"ratio_median={float(ratio.median()):.6f} ratio_p5={float(ratio.quantile(0.05)):.6f} "
        f"ratio_p95={float(ratio.quantile(0.95)):.6f} allclose(0,0)={allclose_ok} [{allclose_msg}]"
    )
    assert bool(bits_equal.all()), f"{tag}: {int((~bits_equal).sum())} values differ bit-wise"
    assert ulp_max == 0 and float(diff.max()) == 0.0


# (tokens, spare tile rows): R, k
SHAPES = [
    (1000, 0),  # R = 2240, k = 3, ragged window (smallest R)
    (1536, 0),  # R = 3296, k = 1 (whole rows, 1 block a core)
    (2048, 0),  # R = 4320, k = 3 (perf guard)
    (4096, 0),  # R = 8416, k = 2 (perf focus, largest perf shape)
]


@pytest.mark.parametrize("shape", SHAPES, ids=[f"T{t}_R{rows_for_tokens(t) + TILE * s}" for t, s in SHAPES])
def test_precision_baseline(shape, device):
    tokens, spare = shape
    rows = rows_for_tokens(tokens) + TILE * spare
    got, want = _measure(device, tokens, rows)
    _report(f"T={tokens} R={rows} k={plan(rows).pieces_per_row}", got, want)


@pytest.mark.parametrize("variant", ["full_sync", "fp32_dest"])
@pytest.mark.parametrize("tokens", [1024, 4096])
def test_values_independent_of_compute_config(tokens, variant, device):
    rows = rows_for_tokens(tokens)
    got, want = _measure(device, tokens, rows, variant)
    _report(f"T={tokens} R={rows} config={variant}", got, want)
