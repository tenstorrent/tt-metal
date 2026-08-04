# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1b — wide-`W` reduce precision under `fp32_dest_acc_en=False`.

Pins the 10 cells Refinement 1 left failing: every `fp32_dest_acc_en=False`
loose case with `W >= 5120`, at the perf cases' declared config
(`bfloat16 / TILE / INTERLEAVED / fp32_dest_acc_en=False / HiFi2`), swept over
both gamma layouts.  The binding metric is the golden suite's
`rms <= 0.04` component of `TOLERANCES[bfloat16]` — PCC was never the thing that
missed (it stayed at 0.99993+ while rel RMS reached 0.127).

The fix is the reduce DATAPATH (descriptor D7 /
`ReduceAlgorithm::AccumulateViaAdd`), so this file also pins:
  * the narrow-`W` control, which stays on the ReduceTile datapath
    (`WT_CHUNK < REDUCE_ACC_VIA_ADD_MIN_WT`) and must not move; and
  * a masked-reduce guard on the pad-poison widths — AccumulateViaAdd swaps the
    partial-`W` mechanism from a scaler tile to a 0/1 mask tile, and a padding
    leak on a wide row is a near-uniform scale error PCC is largely blind to, so
    the assertion is on the got/true RATIO, not on PCC.

DO NOT DELETE.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import REDUCE_ACC_VIA_ADD_MIN_WT

TILE_DIM = 32

# Golden-suite bfloat16 thresholds (eval/golden_tests/rms_norm/helpers.py).
PCC_BF16 = 0.995
RMS_BF16 = 0.04

# The shapes Refinement 1's `severity=precision` failures came from: every
# fp32_dest_acc_en=False loose case with W >= 5120 (interleaved half; the sharded
# half is Refinement 2's).
WIDE_SHAPES = [
    pytest.param((1, 1, 32, 5120), id="decode_w5120"),
    pytest.param((1, 1, 32, 7168), id="decode_w7168"),
    pytest.param((1, 1, 96, 6144), id="resilience_w6144"),
    pytest.param((1, 1, 160, 11008), id="resilience_w11008"),
    pytest.param((1, 224, 11008), id="resilience_3d_w11008"),
    pytest.param((1, 1, 8192, 5120), id="prefill_w5120"),
    pytest.param((1, 1, 8192, 7168), id="prefill_w7168"),
]

GAMMA_LAYOUTS = [
    pytest.param(ttnn.TILE_LAYOUT, id="gTILE"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="gRM"),
]


def _perf_case_config():
    """The `_perf_case` table's pinned compute config — the datapath under test."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _torch_rms_norm(x, gamma, epsilon=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    return (xf / rms) * gamma.to(torch.float32).reshape(-1)


def _run(device, shape, gamma_layout, *, poison=None, seed=0):
    """Run the op at the perf-case config; return (got, expected) as fp32 torch."""
    torch.manual_seed(seed)
    W = shape[-1]
    x = torch.randn(*shape, dtype=torch.bfloat16)
    gamma = torch.randn(W, dtype=torch.bfloat16)

    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    gamma_host = gamma.reshape(1, -1) if gamma_layout == ttnn.TILE_LAYOUT else gamma
    gamma_dev = ttnn.from_torch(gamma_host, dtype=ttnn.bfloat16, layout=gamma_layout, device=device)

    if poison is not None:
        # Same mechanism the golden harness uses (helpers.py `poison_padding`):
        # fill the implicit tile padding with a loud value, so a leak into
        # sum(x^2) is catastrophic rather than marginal.
        assert W % TILE_DIM or shape[-2] % TILE_DIM, "poison guard needs a non-tile-aligned dim"
        x_dev = ttnn.fill_implicit_tile_padding(x_dev, poison)
        if gamma_layout == ttnn.TILE_LAYOUT:
            gamma_dev = ttnn.fill_implicit_tile_padding(gamma_dev, poison)

    out = rms_norm(x_dev, gamma=gamma_dev, compute_kernel_config=_perf_case_config())
    got = ttnn.to_torch(out).to(torch.float32)[..., :W]
    return got, _torch_rms_norm(x, gamma)


def _metrics(got, expected):
    diff = got - expected
    denom = expected.pow(2).mean().sqrt()
    rel_rms = (diff.pow(2).mean().sqrt() / denom).item()
    pcc = torch.corrcoef(torch.stack([got.flatten(), expected.flatten()]))[0, 1].item()
    return pcc, rel_rms


@pytest.mark.parametrize("shape", WIDE_SHAPES)
@pytest.mark.parametrize("gamma_layout", GAMMA_LAYOUTS)
def test_wide_w_reduce_precision_no_fp32_dest(device, shape, gamma_layout):
    """rel RMS <= 0.04 on the wide-`W` bf16-DEST reduce (was 0.041-0.127)."""
    got, expected = _run(device, shape, gamma_layout)
    pcc, rel_rms = _metrics(got, expected)
    print(f"\n{shape} gamma={gamma_layout}: PCC={pcc:.7f} rel_rms={rel_rms:.5f}")
    assert pcc >= PCC_BF16, f"PCC {pcc:.7f} < {PCC_BF16}"
    assert rel_rms <= RMS_BF16, f"rel RMS {rel_rms:.5f} > {RMS_BF16}"


@pytest.mark.parametrize("shape", [(1, 1, 32, 1024), (1, 1, 32, 64), (1, 1, 32, 96)])
def test_narrow_w_control_no_fp32_dest(device, shape):
    """Narrow `W` keeps its Refinement-1 accuracy (and its ReduceTile datapath)."""
    got, expected = _run(device, shape, ttnn.TILE_LAYOUT)
    pcc, rel_rms = _metrics(got, expected)
    print(f"\n{shape}: PCC={pcc:.7f} rel_rms={rel_rms:.5f}")
    assert pcc >= PCC_BF16, f"PCC {pcc:.7f} < {PCC_BF16}"
    assert rel_rms <= RMS_BF16, f"rel RMS {rel_rms:.5f} > {RMS_BF16}"


# The pad-poison widths, chosen so BOTH partial-`W` mechanisms are covered:
# Wt < REDUCE_ACC_VIA_ADD_MIN_WT keeps the [full, partial] SCALER pair, Wt above
# it takes the 0/1 MASK tile.  A leak of the poisoned padding into sum(x^2) is a
# near-uniform SCALE error, so the assertion is on the got/true ratio.
POISON_SHAPES = [
    pytest.param((1, 1, 32, 40), id="wt2_scaler"),
    pytest.param((1, 1, 32, 72), id="wt3_scaler"),
    pytest.param((1, 1, 32, 136), id="wt5_mask"),
    pytest.param((1, 1, 32, 200), id="wt7_mask"),
    pytest.param((1, 1, 224, 72), id="wt3_manyrows"),
    pytest.param((1, 1, 40, 40), id="wt2_h_and_w"),
    # Wide + non-aligned: the mask path with many full tiles before the masked one.
    pytest.param((1, 1, 32, 4095), id="wt128_mask_wide"),
    pytest.param((3, 1, 736, 5119), id="wt160_mask_wide"),
]


@pytest.mark.parametrize("shape", POISON_SHAPES)
def test_pad_poison_masked_reduce(device, shape):
    """Poisoned tile padding must not leak into the reduction, on either mechanism."""
    Wt = (shape[-1] + TILE_DIM - 1) // TILE_DIM
    mechanism = "mask" if Wt >= REDUCE_ACC_VIA_ADD_MIN_WT else "partial-scaler"
    got, expected = _run(device, shape, ttnn.TILE_LAYOUT, poison=1000.0)
    pcc, rel_rms = _metrics(got, expected)

    # Scale-error detector: a padding leak scales every element of the row by
    # sqrt(W_true / W_leaked) — a tight ratio cluster away from 1.0.
    keep = expected.abs().flatten() > 1e-2  # skip near-zero denominators
    ratio = got.flatten()[keep] / expected.flatten()[keep]
    median_ratio = ratio.median().item()

    print(f"\n{shape} Wt={Wt} ({mechanism}): PCC={pcc:.7f} rel_rms={rel_rms:.5f} median_ratio={median_ratio:.5f}")
    assert pcc >= PCC_BF16, f"PCC {pcc:.7f} < {PCC_BF16}"
    assert rel_rms <= RMS_BF16, f"rel RMS {rel_rms:.5f} > {RMS_BF16}"
    assert abs(median_ratio - 1.0) <= 0.02, f"median got/true ratio {median_ratio:.5f} — uniform scale error"
