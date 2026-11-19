# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: device RMSNorm on a hidden-split tensor (tt/norm.py) vs the pure-torch ground truth
(reference.attention.rms_norm, HF's HYV4RMSNorm), tenstorrent/tt-metal#65893.

On one chip first (the arithmetic), then on the 8x4 Galaxy with the hidden dimension split over
the 4-chip axis (the SC1 layout) and over the 8-chip axis (the same code on the other axis), and last
with the checkpoint's norm weights of layers 0-5 on the hidden states the real model fed them.

PCC is blind to a wrong scale, and a norm is all scale: a divisor that used the chip's slice width
instead of the hidden size would be off by sqrt(tp) on every token and still correlate perfectly.
So every case also bounds the error of each token's row relative to the reference, on inputs whose
rows span five orders of magnitude, from the regime where eps dominates the mean of squares to
far above it.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import rms_norm
from models.experimental.hy4_preview.tests.common import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH,
    SC1_HIDDEN_AXIS,
    check_pcc,
    hy4_tensors,
)
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.decoder_layer import hidden_to_device, hidden_to_torch
from models.experimental.hy4_preview.tt.norm import TtRMSNorm

EPS = 1e-5  # rms_norm_eps
PCC = 0.9999
# Per-token bound on |device - reference| / |reference|: fp32 input and output (the device measures
# 2e-7), and BF16 input or output (3e-3, BF16's own rounding).
ROW_ERROR = {"fp32": 1e-5, "bf16": 1e-2}
HIDDEN = {"H768": 768, "H6144": 6144}
DTYPES = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16}


def _case(tokens, hidden, seed):
    """A norm weight spread around one, and rows whose RMS runs from 1e-3 (mean of squares 1e-6,
    below eps) to 1e2."""
    g = torch.Generator().manual_seed(seed)
    weight = 1.0 + 0.25 * torch.randn(hidden, generator=g)
    magnitude = torch.logspace(-3, 2, tokens).reshape(tokens, 1) if tokens > 1 else torch.ones(1, 1)
    x = torch.randn(tokens, hidden, generator=g) * magnitude
    return weight, x


def _row_error(ref, dev):
    """max over tokens of |dev - ref| / |ref|, row by row."""
    return ((dev.double() - ref.double()).norm(dim=-1) / ref.double().norm(dim=-1)).max().item()


def _check(plan, weight, x, in_dtype, out_dtype, label, ref=None, bound=None):
    """Run the device norm on x [T, hidden] and compare with the reference (or ``ref``)."""
    ref = rms_norm(x.float(), weight.float(), EPS) if ref is None else ref
    norm = TtRMSNorm(plan, weight, EPS, output_dtype=out_dtype)
    d_out = norm(hidden_to_device(plan, x, dtype=in_dtype))
    assert d_out.dtype == (out_dtype or in_dtype)
    assert tuple(d_out.shape) == (1, 1, x.shape[0], x.shape[1] // plan.hidden_tp)
    shards = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(d_out)]
    out = hidden_to_torch(plan, d_out)
    check_pcc(label, ref.float(), out, PCC)
    error = _row_error(ref.float(), out)
    exact = in_dtype == ttnn.float32 and (out_dtype or in_dtype) == ttnn.float32
    bound = bound or ROW_ERROR["fp32" if exact else "bf16"]
    logger.info(f"{label}: worst row error={error:.2e} (bound {bound:.0e})")
    assert error <= bound, f"{label}: worst row error {error:.2e} > {bound:.0e}"
    return shards


# ---------------------------------------------------------------------------------------------
# One chip
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tokens", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("hidden", list(HIDDEN), ids=list(HIDDEN))
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_norm_one_chip(device, tokens, hidden, dtype):
    plan = MeshPlan(device, hidden_axis=None)
    weight, x = _case(tokens, HIDDEN[hidden], seed=1)
    _check(plan, weight, x, DTYPES[dtype], None, f"one chip norm {hidden} {dtype}")


def test_norm_one_chip_fp32_in_bf16_out(device):
    """How the decoder layer uses it: an fp32 collapsed stream in, a BF16 sublayer input out."""
    plan = MeshPlan(device, hidden_axis=None)
    weight, x = _case(32, 768, seed=2)
    _check(plan, weight, x, ttnn.float32, ttnn.bfloat16, "one chip norm fp32 to bf16")


# ---------------------------------------------------------------------------------------------
# 8x4 Galaxy: the hidden dimension split across a mesh axis
# ---------------------------------------------------------------------------------------------
galaxy = [
    pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True),
    pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True),
]


def _galaxy(fn):
    for mark in reversed(galaxy):
        fn = mark(fn)
    return fn


def _replicas_agree(plan, shards, label):
    """Chips that hold the same hidden slice must hold the same values."""
    rows, cols = plan.mesh_shape
    for index, shard in enumerate(shards):
        r, c = divmod(index, cols)
        first = shards[r * cols] if plan.hidden_axis == 0 else shards[c]
        assert torch.equal(shard, first), f"{label}: chip ({r}, {c}) differs from the first replica of its slice"


@_galaxy
# 1: hidden over the 4-chip axis, replicated across the 8-chip axis (SC1). 0: the other axis.
@pytest.mark.parametrize("hidden_axis", [1, 0], ids=["axis1-4chips-sc1", "axis0-8chips"])
@pytest.mark.parametrize("tokens", [1, 32, 96], ids=["T1", "T32", "T96"])
@pytest.mark.parametrize("hidden", list(HIDDEN), ids=list(HIDDEN))
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_norm_hidden_split(mesh_device, hidden_axis, tokens, hidden, dtype):
    """Each chip sums the squares of its own slice; one all-reduce makes the mean global."""
    plan = MeshPlan(mesh_device, hidden_axis=hidden_axis)
    weight, x = _case(tokens, HIDDEN[hidden], seed=3)
    label = f"split norm {hidden} {dtype}"
    shards = _check(plan, weight, x, DTYPES[dtype], None, label)
    assert len(shards) == 32
    _replicas_agree(plan, shards, label)


@_galaxy
def test_norm_sc1_fp32_in_bf16_out(mesh_device):
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    weight, x = _case(96, 6144, seed=4)
    _check(plan, weight, x, ttnn.float32, ttnn.bfloat16, "sc1 norm fp32 to bf16")


@_galaxy
def test_inverse_rms_uses_the_global_width(mesh_device):
    """The scale itself, where a chip-local divisor would show as a factor of sqrt(tp) = 2."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    weight, x = _case(32, 6144, seed=5)
    norm = TtRMSNorm(plan, weight, EPS)
    d_scale = norm.inverse_rms(hidden_to_device(plan, x))
    shards = [ttnn.to_torch(s).float().reshape(-1) for s in ttnn.get_device_tensors(d_scale)]
    ref = torch.rsqrt(x.pow(2).mean(-1) + EPS)
    ratio = shards[0] / ref
    logger.info(f"inverse rms device / reference: min={ratio.min():.6f} max={ratio.max():.6f}")
    assert (ratio - 1).abs().max() < 1e-5, ratio
    for shard in shards[1:]:
        assert torch.equal(shard, shards[0]), "the all-reduce must leave the same scale on every chip"


# ---------------------------------------------------------------------------------------------
# The checkpoint's norm weights, on the hidden states the real model fed them
# ---------------------------------------------------------------------------------------------
@_galaxy
@pytest.mark.parametrize("layer", range(6), ids=lambda layer: f"layer{layer}")
@pytest.mark.parametrize("which", ["input_layernorm", "post_attention_layernorm"], ids=["attn", "mlp"])
def test_norm_sc1_real_weights(mesh_device, layer, which):
    """Layers 0-5: fp32 in, BF16 out, against the reference and against what HF's own norm returned
    for the same input (BF16, so that comparison carries HF's rounding)."""
    golden = hf_harness.find_golden()
    if golden is None:
        pytest.skip(f"no golden under {hf_harness.GOLDEN_ROOT}: record one with reference/hf_harness.py")
    site = "attn" if which == "input_layernorm" else "mlp"
    rec = golden["layers"][layer]
    (weight,) = hy4_tensors([f"model.layers.{layer}.{which}.weight"])
    x = rec[f"{site}_hc"].float()
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    label = f"sc1 real L{layer} {which}"
    _check(plan, weight, x, ttnn.float32, ttnn.bfloat16, label)
    _check(plan, weight, x, ttnn.float32, ttnn.bfloat16, f"{label} vs HF", ref=rec[f"{site}_in"].float(), bound=2e-2)
