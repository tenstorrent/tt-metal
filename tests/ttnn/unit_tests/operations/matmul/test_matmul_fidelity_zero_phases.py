# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fidelity phases that only multiply zero mantissa bits.

Each fidelity phase multiplies a 5-bit slice of SrcA (in1) by a 7-bit slice of SrcB (in0):
LoFi = A hi x B hi, HiFi2 adds A lo x B hi, HiFi3 adds A hi x B lo, HiFi4 adds A lo x B lo.
bfloat4_b in1 fits in SrcA's high slice, and bfloat8_b or bfloat4_b in0 fits in SrcB's high slice, so the
phases that read those low slices add zero. ttnn picks the lowest equivalent fidelity for its default and
warns when compute_kernel_config asks for more (get_lowest_equivalent_matmul_fidelity).
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_wormhole_b0

LoFi, HiFi2, HiFi3, HiFi4 = (
    ttnn.MathFidelity.LoFi,
    ttnn.MathFidelity.HiFi2,
    ttnn.MathFidelity.HiFi3,
    ttnn.MathFidelity.HiFi4,
)

# (in0 dtype, in1 dtype, requested fidelity, lowest fidelity with bit-identical output)
ZERO_PHASE_CASES = [
    (ttnn.bfloat16, ttnn.bfloat4_b, HiFi2, LoFi),
    (ttnn.bfloat16, ttnn.bfloat4_b, HiFi4, HiFi3),
    (ttnn.float32, ttnn.bfloat4_b, HiFi2, LoFi),
    (ttnn.bfloat8_b, ttnn.bfloat4_b, HiFi2, LoFi),
    (ttnn.bfloat8_b, ttnn.bfloat4_b, HiFi3, LoFi),
    (ttnn.bfloat8_b, ttnn.bfloat4_b, HiFi4, LoFi),
    (ttnn.bfloat8_b, ttnn.bfloat8_b, HiFi3, HiFi2),
    (ttnn.bfloat8_b, ttnn.bfloat8_b, HiFi4, HiFi2),
    (ttnn.bfloat8_b, ttnn.bfloat16, HiFi4, HiFi2),
    (ttnn.bfloat4_b, ttnn.bfloat16, HiFi3, HiFi2),
]

# Every phase up to the requested one contributes: dropping to the lower fidelity must change the output.
# These show that the bit-identity checks above can fail.
CONTRIBUTING_PHASE_CASES = [
    (ttnn.bfloat16, ttnn.bfloat4_b, HiFi3, LoFi),  # the HiFi3 phase is needed even though HiFi2 is zero (#39627)
    (ttnn.bfloat16, ttnn.bfloat16, HiFi2, LoFi),
    (ttnn.bfloat8_b, ttnn.bfloat16, HiFi2, LoFi),
    (ttnn.bfloat16, ttnn.bfloat8_b, HiFi4, HiFi3),
]

M, K, N = 64, 2048, 128


def _id(case):
    in0, in1, requested, lower = case
    return f"{in0}-{in1}-{requested}-vs-{lower}".replace("DataType.", "").replace("MathFidelity.", "")


def _inputs(device, in0_dtype, in1_dtype):
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K)
    b = torch.randn(1, 1, K, N) / K**0.5
    a_tt = ttnn.from_torch(a, dtype=in0_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return a_tt, b_tt


def _matmul(device, a_tt, b_tt, fidelity, fp32_dest_acc_en):
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )
    out = ttnn.matmul(a_tt, b_tt, dtype=ttnn.float32, compute_kernel_config=config)
    return ttnn.to_torch(out)


def _skip_wormhole_hifi4_fp32_acc(fidelities, fp32_dest_acc_en):
    # HiFi4 with fp32 accumulation can give wrong results on Wormhole (#38306).
    if is_wormhole_b0() and fp32_dest_acc_en and HiFi4 in fidelities:
        pytest.skip("HiFi4 + fp32_dest_acc_en is unreliable on Wormhole (#38306)")


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
@pytest.mark.parametrize("case", ZERO_PHASE_CASES, ids=_id)
def test_zero_phases_do_not_change_output(device, case, fp32_dest_acc_en):
    in0_dtype, in1_dtype, requested, lower = case
    _skip_wormhole_hifi4_fp32_acc((requested, lower), fp32_dest_acc_en)
    a_tt, b_tt = _inputs(device, in0_dtype, in1_dtype)
    out_requested = _matmul(device, a_tt, b_tt, requested, fp32_dest_acc_en)
    out_lower = _matmul(device, a_tt, b_tt, lower, fp32_dest_acc_en)
    assert torch.equal(out_requested, out_lower), (
        f"{requested} and {lower} differ in {(out_requested != out_lower).sum().item()} of {out_lower.numel()} "
        f"elements for in0={in0_dtype}, in1={in1_dtype}"
    )


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
@pytest.mark.parametrize("case", CONTRIBUTING_PHASE_CASES, ids=_id)
def test_contributing_phases_change_output(device, case, fp32_dest_acc_en):
    in0_dtype, in1_dtype, requested, lower = case
    _skip_wormhole_hifi4_fp32_acc((requested, lower), fp32_dest_acc_en)
    a_tt, b_tt = _inputs(device, in0_dtype, in1_dtype)
    out_requested = _matmul(device, a_tt, b_tt, requested, fp32_dest_acc_en)
    out_lower = _matmul(device, a_tt, b_tt, lower, fp32_dest_acc_en)
    assert not torch.equal(out_requested, out_lower)


def test_lowest_equivalent_fidelity_binding(device):
    for in0_dtype, in1_dtype, requested, lower in ZERO_PHASE_CASES:
        assert ttnn.get_lowest_equivalent_matmul_fidelity(device.arch(), in0_dtype, in1_dtype, requested) == lower
    for in0_dtype, in1_dtype, requested, _ in CONTRIBUTING_PHASE_CASES:
        assert ttnn.get_lowest_equivalent_matmul_fidelity(device.arch(), in0_dtype, in1_dtype, requested) == requested


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, expected",
    [
        (ttnn.bfloat16, ttnn.bfloat4_b, LoFi),  # was HiFi2, which is bit-identical to LoFi here
        (ttnn.bfloat16, ttnn.bfloat8_b, HiFi2),
        (ttnn.bfloat16, ttnn.bfloat16, HiFi2),
        (ttnn.bfloat8_b, ttnn.bfloat8_b, LoFi),
    ],
)
def test_default_fidelity_skips_zero_phases(device, in0_dtype, in1_dtype, expected):
    a_tt, b_tt = _inputs(device, in0_dtype, in1_dtype)
    attributes = ttnn.create_matmul_attributes(a_tt, b_tt, ttnn.MatmulParams(), [])
    assert attributes.compute_kernel_config.math_fidelity == expected


def test_warns_on_zero_phases(device, capfd):
    # A dtype/fidelity combination no other test here uses: the warning is logged once per combination.
    a_tt, b_tt = _inputs(device, ttnn.bfloat4_b, ttnn.bfloat8_b)
    _matmul(device, a_tt, b_tt, HiFi4, fp32_dest_acc_en=False)
    out, err = capfd.readouterr()
    assert "only the first 2 can change the result" in out + err
