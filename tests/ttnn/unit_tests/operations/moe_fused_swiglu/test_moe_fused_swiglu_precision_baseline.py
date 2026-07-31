# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for moe_fused_swiglu (verifier-authored, Phase 0).

Records PCC / max-abs / mean-abs / relative-RMS **and** the got/true ratio spread on a
handful of shapes, against two references:

  1. ``fp32``  — the unquantized oracle the golden suite grades against
     (``eval/golden_tests/moe_fused_swiglu/helpers.py``; its gate is imported below).
  2. ``floor`` — the SAME torch chain with only the bfp4_b weight quantization applied
     (weights round-tripped through ``from_torch``/``to_torch``). This is the ceiling
     helpers.py itself calls the number "no correct implementation can beat", and it is
     the recipe helpers.py prescribes before blaming a kernel.

FINDING THIS FILE PINS: the measured floor is **0.9797-0.9799**, so the ORIGINAL 0.98
golden gate sat ABOVE the unbeatable format ceiling for these randn fixtures and was
unreachable by any correct implementation. That finding was acted on — the operator
relaxed the golden gate to 0.975 on 2026-07-31 — and this file's job is unchanged and
is the reason the relaxation is safe: the device lands 0.0005-0.0007 under the floor,
and it is that *kernel-attributable* margin, not the absolute gate, that ``FLOOR_SLACK``
holds. A real precision regression trips here long before it trips a 0.975 gate.

The got/true ratio spread is the scale-bug detector: a tight cluster of ``actual/expected``
around a non-1.0 constant would mean a uniform scale/structural bug (fp32 intermediates
would not fix it); a broad spread centred on 1.0 is ordinary quantization noise.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu

TILE = 32
HIDDEN = 2048
NUM_GLOBAL_EXPERTS = 256
NUM_LOCAL_EXPERTS = 8
LOCAL_EXPERT_ID = 3
GLOBAL_EXPERT_ID = 137
PADDING_SENTINEL = 100.0

# The golden suite's gate. Recorded for context; NOT asserted here (what IS asserted
# is FLOOR_SLACK below). Imported rather than copied — this was a `0.98` literal that
# went stale when the operator relaxed the golden gate to 0.975, which made the
# printout below state the opposite of the truth.
try:
    from eval.golden_tests.moe_fused_swiglu.feature_spec import _PCC_GATE as GOLDEN_PCC_GATE
except ImportError:  # a checkout without the eval harness
    GOLDEN_PCC_GATE = 0.975

# What the kernel is actually held to: how far below the measured bfp4-weight format
# floor the device may land. Phase 0 measures 0.0005-0.0007, so 0.0015 is a
# regression tripwire, not a fresh bar.
FLOOR_SLACK = 0.0015

# (emb, capacity, count) — small / medium / multi-M-block / a larger allocation.
SHAPES = [
    (6144, 1024, 32),  # one tile-row, tile aligned
    (7168, 1024, 255),  # non-tile-aligned tail (the phantom-row seam)
    (7168, 2048, 512),  # crosses the internal M block (m_blocks > 1)
    (7168, 5120, 256),  # the graded perf point's allocation
]


def _reference(x_rows, w_gate, w_up, w_down):
    """h = SiLU(x @ W_gate) * (x @ W_up); out = h @ W_down, in fp32."""
    xf = x_rows.to(torch.float32)
    h = torch.nn.functional.silu(torch.matmul(xf, w_gate.to(torch.float32)))
    h = h * torch.matmul(xf, w_up.to(torch.float32))
    return torch.matmul(h, w_down.to(torch.float32))


def _count_tensors(count, device):
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    to_dev = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return to_dev(counts), to_dev(idx)


def _quantized(t, device, dtype):
    """Round-trip through a device format so torch sees exactly the bytes the kernel sees."""
    tt = ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.to_torch(tt).to(torch.float32)


def _pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _metrics(expected, actual):
    diff = (actual - expected).abs()
    denom = expected.norm().item()
    ratio = (actual[expected.abs() > 0] / expected[expected.abs() > 0]).to(torch.float64)
    q = torch.quantile(ratio, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    return {
        "pcc": _pcc(expected, actual),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_rms": ((actual - expected).norm().item() / denom) if denom else float("nan"),
        "r_p5": q[0].item(),
        "r_med": q[1].item(),
        "r_p95": q[2].item(),
    }


@pytest.mark.parametrize("emb, capacity, count", SHAPES)
@pytest.mark.parametrize("input_format", ["bf16_rm", "bfp8_tile"])
def test_precision_baseline(device, emb, capacity, count, input_format):
    torch.manual_seed(42)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    if count < capacity:
        x[:, :, count:, :] = PADDING_SENTINEL
    w_gate = torch.randn((emb, HIDDEN), dtype=torch.float32)
    w_up = torch.randn((emb, HIDDEN), dtype=torch.float32)
    w_down = torch.randn((HIDDEN, emb), dtype=torch.float32)

    act_dtype, act_layout = (
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) if input_format == "bf16_rm" else (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    )
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=act_dtype,
        layout=act_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_w = [
        ttnn.from_torch(
            w.to(torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in (w_gate, w_up, w_down)
    ]
    tt_counts, tt_idx = _count_tensors(count, device)

    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID)
    actual = ttnn.to_torch(out)[0, 0, :count, :].to(torch.float32)

    x_rows = x[0, 0, :count, :].to(torch.bfloat16)
    expected = _reference(x_rows, w_gate, w_up, w_down)

    # The unbeatable ceiling: same chain, bfp4_b weights only.
    floor = _reference(
        x_rows,
        _quantized(w_gate, device, ttnn.bfloat4_b),
        _quantized(w_up, device, ttnn.bfloat4_b),
        _quantized(w_down, device, ttnn.bfloat4_b),
    )
    floor_pcc = _pcc(expected, floor)

    dev = _metrics(expected, actual)
    _, allclose_msg = comp_allclose(expected, actual, rtol=0.3, atol=1e-2)

    print(
        f"\n[precision] {input_format} emb={emb} cap={capacity} count={count}\n"
        f"  vs fp32 oracle : pcc={dev['pcc']:.6f} max_abs={dev['max_abs']:.4g} "
        f"mean_abs={dev['mean_abs']:.4g} rel_rms={dev['rel_rms']:.4f}\n"
        f"  got/true ratio : median={dev['r_med']:.4f} p5={dev['r_p5']:.4f} p95={dev['r_p95']:.4f}"
        f"  (broad spread centred on 1.0 == quantization noise, NOT a scale bug)\n"
        f"  bfp4 weight floor (torch chain) : pcc={floor_pcc:.6f}   "
        f"kernel-attributable dpcc={floor_pcc - dev['pcc']:.6f}\n"
        f"  golden gate {GOLDEN_PCC_GATE} is {'ABOVE' if floor_pcc < GOLDEN_PCC_GATE else 'below'} the floor "
        f"-> the gate is {'UNREACHABLE' if floor_pcc < GOLDEN_PCC_GATE else 'reachable'}\n"
        f"  {allclose_msg}"
    )

    assert torch.isfinite(actual).all(), "non-finite value in a defined output row"
    # The scale-bug tripwire: a uniform got/true ratio pinned away from 1.0.
    assert 0.9 < dev["r_med"] < 1.1, f"got/true median {dev['r_med']} — uniform scale error, not noise"
    # The graded gate must be ACHIEVABLE. A gate above the format ceiling grades the
    # bfp4_b weight quantization instead of the kernel, and no correct implementation
    # can pass it — that is exactly the condition Refinement 1b was opened to fix.
    assert GOLDEN_PCC_GATE < floor_pcc, (
        f"the graded PCC gate {GOLDEN_PCC_GATE} sits ABOVE the measured format ceiling "
        f"{floor_pcc:.6f} for (emb={emb}, capacity={capacity}, count={count}) — it is "
        f"unreachable by any correct implementation and grades the weight format, not the op"
    )
    # The kernel is held to the measured format floor, which is far tighter than the gate.
    assert_with_pcc(expected, actual, floor_pcc - FLOOR_SLACK)


def test_pcc_gate_has_one_source_of_truth():
    """The acceptance gate must BE the golden suite's gate, not a copy of it.

    Refinement 1b existed because those drifted: the golden gate was relaxed
    0.98 -> 0.975 (below the format ceiling this file measures) and the acceptance
    test kept the stale 0.98 literal, which left all 12 of its numerics cells
    unsatisfiable. All three now resolve to one definition; this keeps them there.
    Host-only — no device needed.
    """
    from eval.golden_tests.moe_fused_swiglu.feature_spec import _PCC_GATE as spec_gate
    from eval.golden_tests.moe_fused_swiglu.helpers import _PCC_GATE as helpers_gate
    from tests.ttnn.unit_tests.operations.moe_fused_swiglu.test_moe_fused_swiglu import PCC_GATE as acceptance_gate

    assert (
        helpers_gate == spec_gate
    ), f"golden suite disagrees with itself: helpers.py {helpers_gate} vs feature_spec.py {spec_gate}"
    assert acceptance_gate == spec_gate, (
        f"the acceptance test's gate ({acceptance_gate}) has drifted from the golden "
        f"suite's ({spec_gate}); it must track it by import, not by a copied literal"
    )
