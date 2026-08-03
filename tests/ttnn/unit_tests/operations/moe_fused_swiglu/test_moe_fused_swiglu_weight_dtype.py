# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight dtypes other than bfp4.

The weight stride and the weight CB format are one number, taken from `w_gate.dtype`, so widening
the axis is mostly plumbing. What is NOT plumbing, and what this pins:

  * ACCURACY MUST IMPROVE WITH THE FORMAT. Each dtype is gated against its OWN quantized floor —
    the reference recomputed with the weights round-tripped through that dtype — because a fixed
    PCC gate would pass a bf16 run that had silently quantized to bfp4 somewhere. The floors are
    then asserted to be ORDERED, which is the check that actually catches a wrong CB format: if
    the kernel read bf16 weights as bfp4 the output would track the bfp4 floor, not the bf16 one.
  * L1. Weight CBs are resident, so a wider dtype costs proportionally: bf16 is 3.56x bfp4's
    bytes. The op drops W_down residency to fit, and refuses with the numbers when even that is
    not enough — which at emb 7168 / N 2048 is exactly what bf16 does.

All three weights must share one dtype; a mixed set is rejected, because the stride and the CB
format are derived once.
"""

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_geometry as geo

TILE = 32
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137

DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}

#: emb 7168 / N 2048 is the graded shape but does not fit at bfp8 or bf16 (weight CBs are
#: resident). N 1024 fits all three, which is what makes the ordering check below possible.
EMB, HIDDEN, CAPACITY, COUNT = 7168, 1024, 1024, 256

#: How close the op must get to its OWN format's floor. Slack, not an absolute gate: the floor
#: already carries the format's error, so what is left is the op's own accumulation.
FLOOR_SLACK = 0.004


def _pcc(a, b):
    a, b = a.flatten().to(torch.float32), b.flatten().to(torch.float32)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _swiglu(x, wg, wu, wd, count):
    xs = x[0, 0, :count].to(torch.float32)
    h = torch.nn.functional.silu(xs @ wg.to(torch.float32)) * (xs @ wu.to(torch.float32))
    return h @ wd.to(torch.float32)


def _roundtrip(t, dtype, device):
    """`t` as the device would store it in `dtype`, brought back to torch — the format's floor."""
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device))


def _build(device, dtype):
    torch.manual_seed(42)
    x = torch.randn((1, 1, CAPACITY, EMB), dtype=torch.float32)
    x[:, :, COUNT:, :] = 100.0
    xb = x.to(torch.bfloat16)
    wg, wu = (torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) for _ in range(2))
    wd = torch.randn((HIDDEN, EMB), dtype=torch.bfloat16)
    d = lambda t, dt, l: ttnn.from_torch(t, dtype=dt, layout=l, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = COUNT
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    return (
        (xb, wg, wu, wd),
        d(xb, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        [d(w, dtype, ttnn.TILE_LAYOUT) for w in (wg, wu, wd)],
        d(counts, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        d(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    )


@pytest.mark.parametrize("name", list(DTYPES))
def test_weight_dtype(device, name):
    dtype = DTYPES[name]
    (xb, wg, wu, wd), tt_x, tt_w, tt_counts, tt_idx = _build(device, dtype)

    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID, core_grid=(11, 8))
    got = ttnn.to_torch(out)[0, 0, :COUNT]

    exact = _swiglu(xb, wg, wu, wd, COUNT)
    floor = _swiglu(xb, *(_roundtrip(w, dtype, device) for w in (wg, wu, wd)), COUNT)
    pcc_exact, pcc_floor = _pcc(got, exact), _pcc(floor, exact)

    assert torch.isfinite(got).all(), f"{name}: non-finite output"
    assert pcc_exact >= pcc_floor - FLOOR_SLACK, (
        f"{name}: pcc {pcc_exact:.6f} is more than {FLOOR_SLACK} below this format's own floor "
        f"{pcc_floor:.6f} — the op is losing accuracy the format does not explain"
    )
    print(f"[wdtype] {name:>5} tile={ttnn.tile_size(dtype):>4}B  pcc={pcc_exact:.6f}  floor={pcc_floor:.6f}")


def test_accuracy_is_ordered_by_format(device):
    """bfp4 < bfp8 <= bf16, which is what catches a weight CB left on the wrong format.

    A fixed PCC gate cannot: a bf16 run that quantized to bfp4 somewhere still clears 0.975. What
    it cannot do is land on the bf16 floor.
    """
    seen = {}
    for name, dtype in DTYPES.items():
        (xb, wg, wu, wd), tt_x, tt_w, tt_counts, tt_idx = _build(device, dtype)
        out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID, core_grid=(11, 8))
        seen[name] = _pcc(ttnn.to_torch(out)[0, 0, :COUNT], _swiglu(xb, wg, wu, wd, COUNT))
    print(f"[wdtype] ordering: " + "  ".join(f"{k}={v:.6f}" for k, v in seen.items()))
    assert seen["bfp4"] < seen["bfp8"], f"bfp8 did not beat bfp4: {seen}"
    assert seen["bfp8"] <= seen["bf16"] + 1e-6, f"bf16 did not match or beat bfp8: {seen}"


def test_mixed_weight_dtypes_are_rejected(device, expect_error):
    """The stride and the CB format are ONE number, derived from w_gate."""
    (_, tt_x, tt_w, tt_counts, tt_idx) = _build(device, ttnn.bfloat4_b)[0:1] + _build(device, ttnn.bfloat4_b)[1:]
    wd8 = ttnn.from_torch(
        torch.randn((HIDDEN, EMB), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(ValueError, r"(?i)share one dtype"):
        moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], wd8, tt_counts, tt_idx, LOCAL_EXPERT_ID, core_grid=(11, 8))


def test_bf16_weights_report_l1_at_the_graded_shape(device, expect_error):
    """Weight CBs are resident, so bf16 costs 3.56x bfp4. At emb 7168 / N 2048 it does not fit,
    and the op must say so with the numbers rather than let the allocator throw."""
    blk = geo.Blocking(11, 8, 7168, 2048, 32, w_tile=ttnn.tile_size(ttnn.bfloat16), x_stick=28 * 64)
    assert blk.l1_bytes(True) > geo.L1_CB_BUDGET - geo.L1_CB_RESERVE, "expected bf16 not to fit here"
    assert not blk.wd_resident, "residency should already have been given up before refusing"
