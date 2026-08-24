# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.argmax(..., use_rvv=True): the opt-in Blackhole TILE-layout
last-dim argmax path that runs the reduction on TRISC2's Zve32f vector unit.

Gating: the RVV kernels JIT-compile with the in-tree opt-in
(ComputeConfigDescriptor::enable_trisc2_rvv, which adds zve32f to the TRISC2
compile). Compile-side coverage runs device-free in CI
(tests/tt_metal/tt_metal/jit_build/test_trisc2_rvv.cpp); these tests execute
on silicon, so they stay behind TT_ENABLE_RVV_TESTS=1 until a Blackhole
runner picks them up; otherwise they skip.
"""

import os

import numpy as np
import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

pytestmark = [
    pytest.mark.skipif(not is_blackhole(), reason="ttnn.argmax use_rvv=True is Blackhole-only (TRISC2 Zve32f)"),
    pytest.mark.skipif(
        os.environ.get("TT_ENABLE_RVV_TESTS") != "1",
        reason="RVV device-test gate: set TT_ENABLE_RVV_TESTS=1 to run the RVV argmax tests on a Blackhole device",
    ),
]


def _monotone(bits: np.ndarray) -> np.ndarray:
    """Monotone uint image of the bfloat16_greater sign-magnitude total order."""
    bits = bits.astype(np.uint32)
    return np.where(bits >= 0x8000, (~bits) & 0xFFFF, bits | 0x8000).astype(np.uint32)


_MONO_NEG_INF = 0x007F  # monotone(0xFF80): the incumbent argmax kernel's -inf init


def _ref_argmax_row(bits_row: np.ndarray):
    """Incumbent ttnn.argmax semantics: bfloat16_greater total order, smallest
    index on ties, -inf init (a row that never beats -inf reports (0, 0xFF80))."""
    m = _monotone(bits_row)
    if int(m.max()) <= _MONO_NEG_INF:
        return 0, 0xFF80
    i = int(np.argmax(m))  # first occurrence == smallest-index tie-break
    return i, int(bits_row[i])


def _bits_of(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().astype(np.uint16)


def _make_case(name: str, v: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """Row-major [b, v] bf16 bit patterns for one battery case."""
    x = rng.standard_normal((b, v), dtype=np.float32) * 4.0
    bits = _bits_of(torch.from_numpy(x).bfloat16()).reshape(b, v)
    kmax, kdecoy = 0x7F7F, 0x7F7E  # largest finite bf16 + decoy the RNG cannot reach
    if name == "random":
        pass
    elif name == "unique_max":
        bits[:, 5 * v // 8] = kmax
        bits[:, v // 3] = kdecoy
    elif name == "tie_first_wins":
        bits[:, 5 * v // 8] = kmax
        bits[:, 7 * v // 8] = kmax
    elif name == "max_at_end":
        bits[:, v - 1] = kmax
    elif name == "max_at_zero":
        bits[:, 0] = kmax
    elif name == "denormal":
        small = rng.integers(0, 0x0080, size=(b, v), dtype=np.uint16)  # denormals and +/-0
        sign = (rng.integers(0, 2, size=(b, v), dtype=np.uint16) << 15).astype(np.uint16)
        bits = (small | sign).astype(np.uint16)
    elif name == "nan_bearing":
        bits[:, v // 4] = 0x7FC0  # +NaN sorts above +inf in the bit order
        bits[:, v // 2] = 0xFFC0  # -NaN payload sorts below -inf
    elif name == "all_negative":
        bits = (bits | 0x8000).astype(np.uint16)
        bits[bits == 0xFF80] = 0xBF80  # avoid accidental -inf
    elif name == "all_neginf":
        bits = np.full((b, v), 0xFF80, dtype=np.uint16)  # the -inf init corner
    else:
        raise ValueError(name)
    return bits


CASES = [
    "random",
    "unique_max",
    "tie_first_wins",
    "max_at_end",
    "max_at_zero",
    "denormal",
    "nan_bearing",
    "all_negative",
    "all_neginf",
]


# v boundaries: 32 = single tile (chunk_pages = min(64, w_tiles) = 1);
# 2016 = 63 tiles, exercising the w_tiles - tiles_done < chunk_pages remainder
# branch; 2048/8192 = exact multiples of the 64-tile chunk.
@pytest.mark.parametrize("v", [32, 2016, 2048, 8192])
@pytest.mark.parametrize("b", [1, 5, 32])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("with_maxval", [True, False])
def test_argmax_rvv_battery(device, v, b, keepdim, with_maxval):
    """Bit-exact index (and optional max-value bits) against the incumbent
    semantics across planted-max / tie / special-value cases."""
    rng = np.random.default_rng(1234 + v + 100 * b)
    for name in CASES:
        bits = _make_case(name, v, b, rng)
        x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, b, v)

        t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        mv = None
        if with_maxval:
            out_shape = (1, 1, b, 1) if keepdim else (1, 1, b)
            mv = ttnn.from_torch(
                torch.zeros(out_shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

        idx_t = ttnn.argmax(t_tile, dim=3, keepdim=keepdim, use_rvv=True, maxval_tensor=mv)
        got_idx = ttnn.to_torch(idx_t).flatten().numpy().astype(np.uint32)
        got_val = _bits_of(ttnn.to_torch(mv).flatten()) if with_maxval else None

        for r in range(b):
            ref_idx, ref_val = _ref_argmax_row(bits[r])
            assert int(got_idx[r]) == ref_idx, f"case {name} row {r}: idx {int(got_idx[r])} != {ref_idx}"
            if with_maxval:
                assert int(got_val[r]) == ref_val, f"case {name} row {r}: val {int(got_val[r]):#06x} != {ref_val:#06x}"


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("v", [32, 2016, 4096])
@pytest.mark.parametrize("b", [1, 32])
def test_argmax_rvv_matches_upstream_tile_path(device, name, v, b):
    """Index cross-check against the incumbent ttnn.argmax on the same TILE
    tensor — not just random data: every planted-max / tie / special-value
    case from CASES runs through both paths."""
    rng = np.random.default_rng(42 + v + 100 * b)
    bits = _make_case(name, v, b, rng)
    x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, b, v)
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    idx_rvv = ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True, use_rvv=True))
    idx_ref = ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True))
    assert torch.equal(
        idx_rvv.to(torch.int64), idx_ref.to(torch.int64)
    ), f"RVV/incumbent index mismatch on case {name!r} (v={v}, b={b})"


def test_argmax_rvv_rejects_row_major_input(device, expect_error):
    """use_rvv=True is a TILE-layout path; ROW_MAJOR input must be rejected."""
    x = torch.randn(1, 1, 32, 2048).bfloat16()
    t_rm = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "requires TILE layout input"):
        ttnn.argmax(t_rm, dim=3, keepdim=True, use_rvv=True)
