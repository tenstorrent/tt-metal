# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 3 — Input dtype coverage for backward_softmax.

Exercises the relaxed dtype contract:
  - float32   : Phase-0 dtype (unchanged compute config: HiFi4 + fp32_dest_acc)
  - bfloat16  : 7-bit mantissa; HiFi2 + fp32_dest_acc=False
  - bfloat8_b : 8-bit block format with shared exponent; LoFi + fp32_dest_acc=False

The matmul-based REDUCE_ROW SUM path runs the same kernel logic regardless of
input dtype — only the CB format descriptors (page sizes) and the
ComputeConfigDescriptor differ. The accuracy floor moves with the input
quantisation:
  fp32   inputs  → carries the Phase-0 numerical baseline.
  bf16   inputs  → quantised to ~3 decimal digits before pass 1.
  bfp8_b inputs  → shared-exponent compression, ~2 decimal digits effective.

Output dtype follows the input dtype (allocate_tensor_on_device uses
grad_output.dtype), so this file also asserts that contract.

NOTE: the `(dy − s)` catastrophic-cancellation site that gates the fp32
precision floor (Refinement 5 territory) is even more visible at lower
precision; the per-dtype tolerances below allow PCC ≥ 0.999 / 0.99 / 0.95
respectively, and use rel-rms (not abs) as the secondary check because
absolute error scales with the input range, which randn does not bound.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from ttnn.operations.backward_softmax import backward_softmax

# Per-dtype precision contract.
#
# The kernel path is the same for all three dtypes — only the CB format and
# compute config differ. So the *only* knob moving the floor is input
# quantisation:
#   fp32   : matches the Phase-0 baseline (PCC ≥ 0.999, rms_rel ≤ 0.01).
#   bf16   : ~7 mantissa bits → expect ~0.5× decade looser rms_rel.
#   bfp8_b : ~5-6 bits + shared exponent → another 0.5-1 decade looser.
#
# These thresholds were calibrated against probe_003.py / the strategy
# correctness regime; they're 1-2× the empirically observed worst-case rms_rel
# on N(0,1) inputs, so a real regression would trip them while normal run-to-
# run fluctuation stays under.
_DTYPE_TOLERANCES = {
    # dtype           : (pcc_threshold, rms_rel_threshold)
    ttnn.float32: (0.999, 0.01),
    ttnn.bfloat16: (0.999, 0.05),
    ttnn.bfloat8_b: (0.95, 0.15),
}

_DTYPE_IDS = {
    ttnn.float32: "float32",
    ttnn.bfloat16: "bfloat16",
    ttnn.bfloat8_b: "bfloat8_b",
}


def _torch_reference(grad_output: torch.Tensor, output: torch.Tensor, dim: int) -> torch.Tensor:
    """grad_input = output * (grad_output - sum(output * grad_output, dim))."""
    grad_output = grad_output.float()
    output = output.float()
    s = (output * grad_output).sum(dim=dim, keepdim=True)
    return output * (grad_output - s)


def _torch_input_for(dtype: ttnn.DataType, t: torch.Tensor) -> torch.Tensor:
    """
    Quantise a host-side reference input down to whatever the on-device dtype
    can actually represent, so the reference and the device path operate on
    the SAME numerical input. Without this, bf16's pcc looks worse than it is
    because the reference saw fp32 inputs while the device saw bf16.
    """
    if dtype == ttnn.bfloat16:
        return t.to(torch.bfloat16).to(torch.float32)
    # bfp8_b is a packed device-only format with no host counterpart. Use the
    # fp32 host tensor as the reference (slightly pessimistic for bfp8 — the
    # input itself loses ~2-3 bits of precision when ttnn.from_torch packs it
    # to bfp8).
    return t


def _to_device(t: torch.Tensor, device, dtype: ttnn.DataType) -> ttnn.Tensor:
    """Build an on-device TILE_LAYOUT tensor of the requested dtype."""
    if dtype == ttnn.bfloat16:
        host_t = t.to(torch.bfloat16)
    else:
        host_t = t  # fp32 + bfp8_b accept fp32 host tensors
    return ttnn.from_torch(
        host_t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# Shape set: covers single-tile minimal, multi-tile along each reduce axis,
# non-square, and multi-batch. Kept small enough that the 3-dtype × 2-dim
# parametrisation doesn't blow up runtime, but rich enough to exercise the
# whole-row caching paths and multi-batch lane distribution.
_SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 32, 256), id="multi_tile_W"),
    pytest.param((1, 1, 256, 32), id="multi_tile_H"),
    pytest.param((1, 1, 64, 128), id="non_square_64x128"),
    pytest.param((2, 4, 64, 128), id="multi_batch"),
]


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim=-1", "dim=-2"])
@pytest.mark.parametrize(
    "dtype",
    [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    ids=lambda d: _DTYPE_IDS[d],
)
def test_backward_softmax_dtype_correctness(device, shape, dim, dtype):
    """
    backward_softmax handles every supported dtype within its per-dtype PCC /
    rel-rms band. Output dtype contract (output dtype == input dtype) is
    asserted as well.
    """
    torch.manual_seed(42)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)

    # Use the dtype-quantised reference so we measure the kernel's error, not
    # the host-side dtype conversion's error.
    expected = _torch_reference(_torch_input_for(dtype, torch_dy), _torch_input_for(dtype, torch_y), dim=dim)

    ttnn_dy = _to_device(torch_dy, device, dtype)
    ttnn_y = _to_device(torch_y, device, dtype)

    ttnn_grad_input = backward_softmax(ttnn_dy, ttnn_y, dim=dim)

    # Output dtype must match input dtype.
    assert ttnn_grad_input.dtype == dtype, (
        f"[{_DTYPE_IDS[dtype]} shape={shape} dim={dim}] " f"output dtype {ttnn_grad_input.dtype} != input dtype {dtype}"
    )
    assert tuple(ttnn_grad_input.shape) == tuple(shape)

    actual = ttnn.to_torch(ttnn_grad_input).float()

    pcc_threshold, rms_rel_threshold = _DTYPE_TOLERANCES[dtype]

    pcc_ok, pcc_msg = check_with_pcc(expected, actual, pcc=pcc_threshold)
    rms_rel = (actual - expected).pow(2).mean().sqrt().item() / max(expected.pow(2).mean().sqrt().item(), 1e-12)

    assert pcc_ok, f"[{_DTYPE_IDS[dtype]} shape={shape} dim={dim}] PCC failed: {pcc_msg} (rms_rel={rms_rel:.4f})"
    assert rms_rel <= rms_rel_threshold, (
        f"[{_DTYPE_IDS[dtype]} shape={shape} dim={dim}] " f"rms_rel={rms_rel:.4f} > {rms_rel_threshold:.4f}"
    )


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=lambda d: _DTYPE_IDS[d],
)
def test_backward_softmax_dtype_against_real_softmax(device, dtype):
    """
    End-to-end sanity: when `output` is the actual softmax of some logits, the
    result of backward_softmax(dy, output) matches torch.autograd's softmax
    backward. This guards against sign/scaling bugs that might cancel out on
    randn-only inputs.
    """
    torch.manual_seed(7)
    shape = (1, 1, 64, 128)
    dim = -1

    # Reference path: compute torch.autograd softmax-backward in fp32, then
    # quantise to the device dtype's precision floor for fair comparison.
    logits = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    softmax_out = torch.nn.functional.softmax(logits, dim=dim)
    grad_output = torch.randn(shape, dtype=torch.float32)
    softmax_out.backward(grad_output)
    expected = _torch_input_for(dtype, logits.grad.detach().float())

    ttnn_dy = _to_device(grad_output, device, dtype)
    ttnn_y = _to_device(softmax_out.detach(), device, dtype)

    ttnn_grad_input = backward_softmax(ttnn_dy, ttnn_y, dim=dim)
    assert ttnn_grad_input.dtype == dtype

    actual = ttnn.to_torch(ttnn_grad_input).float()

    pcc_threshold, _ = _DTYPE_TOLERANCES[dtype]
    pcc_ok, pcc_msg = check_with_pcc(expected, actual, pcc=pcc_threshold)
    assert pcc_ok, f"[{_DTYPE_IDS[dtype]}] PCC vs torch.autograd softmax-backward: {pcc_msg}"


@pytest.mark.parametrize(
    "dtype",
    [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    ids=lambda d: _DTYPE_IDS[d],
)
def test_backward_softmax_dtype_zero_input_invariant(device, dtype):
    """
    Invariant: if grad_output == 0, then grad_input == 0 identically. Holds
    for every dtype because the formula is multiplicatively linear in dy:

        grad_input = output * (0 - sum(output * 0)) = 0

    A loss of this invariant under bf16/bfp8 would point at a non-zero
    quantisation drift through the multiply or reduce path.
    """
    shape = (1, 1, 32, 64)
    torch_dy = torch.zeros(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)

    ttnn_dy = _to_device(torch_dy, device, dtype)
    ttnn_y = _to_device(torch_y, device, dtype)

    out = backward_softmax(ttnn_dy, ttnn_y, dim=-1)
    actual = ttnn.to_torch(out).float()

    # Tolerance is the same per-dtype rel-rms band, but applied to absolute
    # values since the reference is identically zero.
    _, rms_rel_threshold = _DTYPE_TOLERANCES[dtype]
    max_abs = actual.abs().max().item()
    # Per-dtype slack — bfp8's shared exponent can produce small "ghost" values
    # near zero when rescaling.
    abs_tol = {ttnn.float32: 1e-4, ttnn.bfloat16: 1e-2, ttnn.bfloat8_b: 1e-1}[dtype]
    assert max_abs <= abs_tol, f"[{_DTYPE_IDS[dtype]}] zero-dy invariant violated: max_abs={max_abs:.4f} > {abs_tol}"


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=lambda d: _DTYPE_IDS[d],
)
def test_backward_softmax_dtype_determinism(device, dtype):
    """
    Same inputs produce bit-identical output across two invocations for every
    dtype. Refinement 1 (multi-core) guarantees this for fp32; the dtype-aware
    compute config in R3 shouldn't change it.
    """
    torch.manual_seed(99)
    shape = (1, 1, 32, 128)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)

    ttnn_dy = _to_device(torch_dy, device, dtype)
    ttnn_y = _to_device(torch_y, device, dtype)

    out1 = ttnn.to_torch(backward_softmax(ttnn_dy, ttnn_y, dim=-1)).float()
    out2 = ttnn.to_torch(backward_softmax(ttnn_dy, ttnn_y, dim=-1)).float()
    assert torch.equal(out1, out2), f"[{_DTYPE_IDS[dtype]}] non-deterministic across runs"


def test_backward_softmax_dtype_rejects_unsupported_dtypes(device):
    """
    Validator still rejects integer / uint32 dtypes. Only fp32 / bf16 / bfp8_b
    pass.
    """
    # uint32 is the canonical "not a float" sentinel TTNN exposes.
    shape = (1, 1, 32, 32)
    torch_input = torch.randint(0, 100, shape, dtype=torch.int32)
    bad_grad = ttnn.from_torch(
        torch_input,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bad_out = ttnn.from_torch(
        torch_input,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises((ValueError, RuntimeError)):
        backward_softmax(bad_grad, bad_out)


@pytest.mark.parametrize(
    "dtype_a,dtype_b",
    [
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.float32, ttnn.bfloat8_b),
    ],
    ids=["fp32_vs_bf16", "bf16_vs_bfp8", "fp32_vs_bfp8"],
)
def test_backward_softmax_dtype_rejects_dtype_mismatch_across_supported(device, dtype_a, dtype_b):
    """
    Even within the supported dtype set, grad_output dtype and output dtype
    must match. This guarantees the kernel's CB format descriptors all derive
    from a single dtype.
    """
    torch.manual_seed(0)
    shape = (1, 1, 32, 32)
    torch_t = torch.randn(shape, dtype=torch.float32)
    grad = _to_device(torch_t, device, dtype_a)
    out = _to_device(torch_t, device, dtype_b)
    with pytest.raises((ValueError, RuntimeError)):
        backward_softmax(grad, out)
