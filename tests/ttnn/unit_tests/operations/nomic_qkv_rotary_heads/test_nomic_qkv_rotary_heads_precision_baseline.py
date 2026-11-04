# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for nomic_qkv_rotary_heads (verifier measurement; numbers go to verification_report.md).

Per output (q, k, v) against the float64 reference: PCC, max / mean absolute error, relative RMS (RMS error over
the reference's std), bf16 ULP error (against the reference rounded to bf16), and the got/true ratio spread (median,
p5, p95) over elements whose reference magnitude is above 10% of the reference std. A tight ratio cluster at a value
other than 1.0 would mean a scale / structural bug rather than rounding.

Run with -s to see the table rows ("PRECISION ...").
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.nomic_qkv_rotary_heads import (
    default_compute_kernel_config,
    head_dim_order,
    nomic_qkv_rotary_heads,
    prepare_qkv_weights,
    prepare_rotary_tables,
)


def _load_acceptance_module():
    """The acceptance test's reference and input builders (sibling file; this test dir is not a package)."""
    path = Path(__file__).with_name("test_nomic_qkv_rotary_heads.py")
    spec = importlib.util.spec_from_file_location("_nomic_qkv_rotary_heads_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_acceptance = _load_acceptance_module()
allocate_heads = _acceptance.allocate_heads
make_inputs = _acceptance.make_inputs
torch_reference = _acceptance.torch_reference
x_to_device = _acceptance.x_to_device

PCC_GATE = 0.99998
RMS_GATE = 0.0050
RATIO_MEDIAN_TOL = {ttnn.MathFidelity.HiFi2: 5e-3, ttnn.MathFidelity.HiFi4: 1e-3}

SHAPES = [
    (2, 1, 512, 768),  # M = 32 (1 M block a core); the 1k-token shape
    (3, 1, 400, 768),  # M = 39, seq_padded, 2 M blocks, 10 grid columns
    (4, 1, 512, 768),  # M = 64; the 2k-token shape
    (8, 1, 512, 768),  # M = 128, 4 M blocks; the 4k-token perf focus
]


def _bf16_ordinal(t):
    """Monotone integer ordinal of bf16 values (for ULP distances)."""
    bits = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _metrics(actual, expected):
    actual = actual.double()
    expected = expected.double()
    diff = actual - expected
    rel_rms = (diff.pow(2).mean().sqrt() / expected.std()).item()
    mask = expected.abs() > 0.1 * expected.std()
    # ULP distance in bf16 away from zero crossings (near-zero references make ordinal distance meaningless).
    ulp = (_bf16_ordinal(actual[mask].float()) - _bf16_ordinal(expected[mask].float())).abs().double()
    ratio = actual[mask] / expected[mask]
    q = torch.quantile(ratio.float()[: 1 << 24], torch.tensor([0.05, 0.5, 0.95]))
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rel_rms": rel_rms,
        "ulp_mean": ulp.mean().item(),
        "ulp_max": ulp.max().item(),
        "ratio_p5": q[0].item(),
        "ratio_med": q[1].item(),
        "ratio_p95": q[2].item(),
    }


FIDELITIES = [
    ttnn.MathFidelity.HiFi2,  # the op's default (the model's QKV group): the baseline
    ttnn.MathFidelity.HiFi4,  # diagnostic: separates HiFi2's srcB mantissa truncation from kernel error
]


@pytest.mark.parametrize("fidelity", FIDELITIES, ids=["HiFi2", "HiFi4"])
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{s[0]}x{s[2]}" for s in SHAPES])
def test_precision_baseline(shape, fidelity, device):
    x, weight, bias = make_inputs(shape)
    expected = torch_reference(x, weight, bias)
    provided = allocate_heads(shape, device, ttnn.L1_MEMORY_CONFIG)
    config = default_compute_kernel_config()
    config.math_fidelity = fidelity
    outputs = nomic_qkv_rotary_heads(
        x_to_device(x, device),
        prepare_qkv_weights(weight, bias, device),
        prepare_rotary_tables(shape[-2], device),
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tensors=provided,
        compute_kernel_config=config,
    )
    order = list(head_dim_order())
    results = {}
    for name, got, ref in zip("qkv", outputs, expected):
        reference = (ref[..., order] if name in "qk" else ref).float()
        actual = ttnn.to_torch(got).float()
        assert_with_pcc(reference, actual, PCC_GATE)
        _, allclose_msg = comp_allclose(reference, actual)
        a, e = actual.double().flatten(), reference.double().flatten()
        pcc = torch.corrcoef(torch.stack((a, e)))[0, 1].item()
        m = _metrics(actual, reference)
        results[name] = m
        print(
            f"\nPRECISION {shape[0]}x{shape[2]} {fidelity.name} {name} pcc={pcc:.7f} max_abs={m['max_abs']:.4e} "
            f"mean_abs={m['mean_abs']:.4e} rel_rms={m['rel_rms']:.5f} ulp_mean={m['ulp_mean']:.3f} "
            f"ulp_max={m['ulp_max']:.0f} ratio[p5,med,p95]=[{m['ratio_p5']:.5f},{m['ratio_med']:.6f},"
            f"{m['ratio_p95']:.5f}] ({allclose_msg})"
        )
    for name, m in results.items():
        assert m["rel_rms"] <= RMS_GATE, f"{name}: relative RMS {m['rel_rms']:.5f} > {RMS_GATE}"
        # Scale-bug detector: the ratio must be centered near 1 and broad (rounding noise), not a tight cluster at
        # another constant. HiFi2 drops x's last mantissa bit in srcB (truncation toward zero), which biases every
        # product's magnitude down by ~2^-9 on average; the tolerance admits that and nothing structural.
        assert math.isclose(
            m["ratio_med"], 1.0, abs_tol=RATIO_MEDIAN_TOL[fidelity]
        ), f"{name}: got/true median {m['ratio_med']:.6f} is off 1.0 (scale/structural bug?)"
