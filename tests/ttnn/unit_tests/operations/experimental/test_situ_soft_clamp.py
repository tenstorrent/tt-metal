# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SiTU SFPU ops.

Two layers are covered:
  * the compute API (api/compute/eltwise_unary/situ.h) driven through generic_op
    with a minimal test kernel, which reaches both dst-accumulator modes;
  * the public ttnn ops (ttnn.soft_clamp / ttnn.situ_gate).

Both are compared against the torch reference transcribed from Moonshot's
SituAndMul module:
    soft_clamp(x) = beta * tanh(x / beta)               -- up half
    situ_gate(x)  = beta * tanh(x / beta) * sigmoid(x)  -- gate half
"""

import struct

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

# activation_situ_beta / activation_situ_linear_beta from the Kimi K3 config.
BETA_GATE = 4.0
BETA_LINEAR = 25.0

TILE_ELEMS = 32 * 32
TILE_BYTES = TILE_ELEMS * 2  # bf16

SMALLEST_NORMAL_BF16 = torch.finfo(torch.bfloat16).smallest_normal
# Below this, an output is irrelevant to the activation and flush-to-zero is fine.
FLUSH_TOLERANT_FLOOR = 1e-30


def situ_and_mul_reference(x, beta, linear_beta):
    """Transcription of Moonshot's SituAndMul.forward (fp32 internally)."""
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def soft_clamp_reference(x, beta):
    """The `up = linear_beta * tanh(up / linear_beta)` line, on its own."""
    return (beta * torch.tanh(x.to(torch.float32) / beta)).to(x.dtype)


def situ_gate_reference(x, beta):
    """The `situ_a = beta * tanh(gate / beta) * sigmoid(gate)` line, on its own."""
    xf = x.to(torch.float32)
    return (beta * torch.tanh(xf / beta) * torch.sigmoid(xf)).to(x.dtype)


# (label, compute-API tile fn, its init fn, model beta, torch reference). The ttnn
# op name matches the label.
OP_CASES = [
    ("soft_clamp", "soft_clamp_tile", "soft_clamp_tile_init", BETA_LINEAR, soft_clamp_reference),
    ("situ_gate", "situ_gate_tile", "situ_gate_tile_init", BETA_GATE, situ_gate_reference),
]
OP_CASE_IDS = [c[0] for c in OP_CASES]
TILE_FNS = {c[0]: (c[1], c[2]) for c in OP_CASES}


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def beta_bit_patterns(beta):
    """beta and 1/beta as fp32 bit patterns. The reciprocal is computed in fp32 so
    it matches the `1.0f / beta` that the ttnn codegen path emits."""
    beta32 = torch.tensor(beta, dtype=torch.float32)
    recip32 = torch.tensor(1.0, dtype=torch.float32) / beta32
    return f32_bits(beta32.item()), f32_bits(recip32.item())


def run_situ_op_on_device(device, torch_input, op_label, beta, fp32_dest_acc_en):
    """Apply one SiTU compute-API op to torch_input on a single core."""
    num_tiles = torch_input.numel() // TILE_ELEMS
    shape = [1, num_tiles, 32, 32]
    tile_fn, tile_init_fn = TILE_FNS[op_label]
    beta_bits, recip_bits = beta_bit_patterns(beta)

    input_tensor = ttnn.from_torch(
        torch_input.reshape(shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    in_cb, out_cb = 0, 16
    # Double-buffered so the reader/writer overlap the single-tile compute loop.
    cb_descriptors = [
        ttnn.CBDescriptor(
            total_size=2 * TILE_BYTES,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
            ],
        )
        for idx in (in_cb, out_cb)
    ]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[0][0] = [input_tensor.buffer_address(), num_tiles, 0]
    writer_rt_args[0][0] = [output_tensor.buffer_address(), num_tiles, 0]

    writer_compile_time_args = [out_cb]
    writer_compile_time_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
            core_ranges=core_grid,
            compile_time_args=ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args(),
            runtime_args=reader_rt_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            core_ranges=core_grid,
            compile_time_args=writer_compile_time_args,
            runtime_args=writer_rt_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/compute/situ_soft_clamp.cpp",
            core_ranges=core_grid,
            compile_time_args=[num_tiles, beta_bits, recip_bits],
            defines=[("situ_tile", tile_fn), ("situ_tile_init", tile_init_fn)],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc_en),
        ),
    ]

    program_descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cb_descriptors)
    output = ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
    return ttnn.to_torch(output).reshape(torch_input.shape)


def situ_test_input(beta, num_tiles, seed=0):
    """A COVERAGE generator, not a representative one — see
    test_situ_realistic_distribution for the plausible-activation case.

    Half a linear sweep to +/-6*beta, half a heavy-tailed draw, shuffled so every
    tile spans the full range. The sweep is what forces the polynomial tanh's
    saturation clamp to be entered at all: a realistic narrow distribution never
    reaches it, which would leave that branch and the |out| <= beta assertion
    untested."""
    n = num_tiles * TILE_ELEMS
    torch.manual_seed(seed)
    sweep = torch.linspace(-6.0 * beta, 6.0 * beta, n // 2)
    tail = torch.randn(n - n // 2) * (2.0 * beta)
    return torch.cat([sweep, tail])[torch.randperm(n)].to(torch.bfloat16)


def report_error(name, golden, actual):
    golden_f = golden.to(torch.float32)
    actual_f = actual.to(torch.float32)
    abs_err = (actual_f - golden_f).abs()
    # Relative error only where the reference is large enough to be meaningful;
    # near a zero crossing bf16 quantisation dominates the ratio.
    mask = golden_f.abs() > 1e-2
    rel_err = (abs_err[mask] / golden_f[mask].abs()).max().item() if mask.any() else 0.0
    logger.info(f"{name}: max abs err {abs_err.max().item():.4e}, max rel err {rel_err:.4e}")
    return rel_err


# bf16 pack rounding alone costs ~3.9e-3 relative; the polynomial tanh adds ~2.3e-3
# on top. 1.5e-2 leaves headroom without accepting a broken op.
REL_ERR_LIMIT = 1.5e-2


@pytest.mark.parametrize("beta", [BETA_LINEAR, BETA_GATE], ids=["beta_25", "beta_4"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["bf16_dst", "fp32_dst"])
@pytest.mark.parametrize("num_tiles", [1, 8])
def test_situ_soft_clamp(device, beta, fp32_dest_acc_en, num_tiles):
    """beta * tanh(x / beta) on device vs torch."""
    torch_input = situ_test_input(beta, num_tiles)
    golden = soft_clamp_reference(torch_input, beta)
    actual = run_situ_op_on_device(device, torch_input, "soft_clamp", beta, fp32_dest_acc_en)

    # tanh is bounded by 1, so beta is a hard bound on the output; an overshoot
    # means the polynomial's clamp is not holding.
    assert actual.to(torch.float32).abs().max().item() <= beta * (1.0 + 1e-3)

    rel_err = report_error(f"soft_clamp beta={beta}", golden, actual)
    assert rel_err < REL_ERR_LIMIT
    assert_with_pcc(golden.to(torch.float32), actual.to(torch.float32), pcc=0.999)


@pytest.mark.parametrize("beta", [BETA_GATE, BETA_LINEAR], ids=["beta_4", "beta_25"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["bf16_dst", "fp32_dst"])
@pytest.mark.parametrize("num_tiles", [1, 8])
def test_situ_gate(device, beta, fp32_dest_acc_en, num_tiles):
    """beta * tanh(x / beta) * sigmoid(x) on device vs torch.

    Exercises the tanh/sigmoid pairing under a single init — the failure mode this
    guards is the two halves fighting over vConstFloatPrgm0, which would show up as
    a gross error rather than a tolerance miss.
    """
    torch_input = situ_test_input(beta, num_tiles)
    golden = situ_gate_reference(torch_input, beta)
    actual = run_situ_op_on_device(device, torch_input, "situ_gate", beta, fp32_dest_acc_en)

    # sigmoid <= 1 and the clamp caps at beta, so beta bounds the product.
    assert actual.to(torch.float32).abs().max().item() <= beta * (1.0 + 1e-3)
    # A dropped sigmoid would leave the deeply-negative tail at the full soft-clamp
    # magnitude (~-beta). Require >10x suppression there instead of an absolute
    # bound, which would scale with beta. Real suppression is 400-2000x.
    tail = torch_input.to(torch.float32) < -6.0
    ungated = (beta * torch.tanh(torch_input.to(torch.float32) / beta))[tail].abs().max().item()
    assert actual.to(torch.float32)[tail].abs().max().item() < 0.1 * ungated

    rel_err = report_error(f"situ_gate beta={beta}", golden, actual)
    assert rel_err < REL_ERR_LIMIT
    assert_with_pcc(golden.to(torch.float32), actual.to(torch.float32), pcc=0.999)


@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["bf16_dst", "fp32_dst"])
@pytest.mark.parametrize("num_tiles", [8])
def test_situ_and_mul(device, num_tiles, fp32_dest_acc_en):
    """Full SituAndMul with both halves computed by the SFPU ops.

    The final gate*up multiply stays in torch — in the real kernel it is a plain
    mul_tiles, not part of either SFPU op.
    """
    gate = situ_test_input(BETA_GATE, num_tiles, seed=0)
    up = situ_test_input(BETA_LINEAR, num_tiles, seed=1)
    x = torch.cat([gate, up], dim=-1).reshape(1, -1)

    golden = situ_and_mul_reference(x, BETA_GATE, BETA_LINEAR)

    gate_device = run_situ_op_on_device(device, gate, "situ_gate", BETA_GATE, fp32_dest_acc_en)
    up_device = run_situ_op_on_device(device, up, "soft_clamp", BETA_LINEAR, fp32_dest_acc_en)
    actual = (gate_device.to(torch.float32) * up_device.to(torch.float32)).to(torch.bfloat16).reshape(1, -1)

    rel_err = report_error("SituAndMul", golden, actual)
    # Both halves' errors compound through the product, so this is looser than the
    # per-op limit.
    assert rel_err < 2.5e-2
    assert_with_pcc(golden.to(torch.float32), actual.to(torch.float32), pcc=0.999)


@pytest.mark.parametrize("label, tile_fn, tile_init_fn, beta, reference", OP_CASES, ids=OP_CASE_IDS)
@pytest.mark.parametrize("sigma", [1.0, 4.0, 16.0], ids=["sigma_1", "sigma_4", "sigma_16"])
def test_situ_realistic_distribution(device, label, tile_fn, tile_init_fn, beta, reference, sigma):
    """Same tolerance against a plausible activation distribution rather than the
    engineered sweep, so the reported accuracy is not an artifact of the input.

    The sigmas are a spread (deep-linear / moderate / reaching saturation at
    beta=4), not measured from the model. bf16 dst only: this test is about the
    input distribution, and both dst modes are covered above.
    """
    torch.manual_seed(1234)
    torch_input = (torch.randn(8 * TILE_ELEMS) * sigma).to(torch.bfloat16)

    golden = reference(torch_input, beta)
    actual = run_situ_op_on_device(device, torch_input, label, beta, fp32_dest_acc_en=False)

    rel_err = report_error(f"{label} N(0,{sigma}) beta={beta}", golden, actual)
    assert rel_err < REL_ERR_LIMIT
    assert_with_pcc(golden.to(torch.float32), actual.to(torch.float32), pcc=0.999)


def edge_value_inputs(beta):
    """Values the coverage generator never produces. Inf matters here specifically:
    a stale-Inf reaching this activation must clamp to +/-beta rather than escape.

    NaN is deliberately absent. The gate/up matmul cannot produce it in a used
    element — its K dim is unpadded and the model dims are tile-aligned — and an
    Inf input clamps rather than becoming NaN. See ckernel_sfpu_situ.h for the
    behaviour the SFPU actually exhibits if one is forced in.
    """
    subnormal = SMALLEST_NORMAL_BF16 / 2
    return [
        0.0,
        -0.0,
        float("inf"),
        float("-inf"),
        SMALLEST_NORMAL_BF16,
        -SMALLEST_NORMAL_BF16,
        subnormal,
        -subnormal,
        1.0,
        -1.0,
        beta,
        -beta,
        6.0 * beta,
        -6.0 * beta,
    ]


def edge_mismatch(golden, actual, rtol):
    """None if this element is acceptable, else why not."""
    # Every input here has a finite reference, so any NaN out is a regression —
    # a broken Inf clamp or an 0*Inf introduced into the SFPU sequence.
    if torch.isnan(actual):
        return f"unexpected NaN (golden {golden.item():.6e})"
    if golden == 0:
        # Sign of zero is not preserved (torch gives -0.0, device +0.0). Harmless:
        # 0 == -0 numerically and nothing downstream reads the sign bit.
        return None if actual == 0 else f"expected 0, got {actual.item():.6e}"
    if golden.abs() < FLUSH_TOLERANT_FLOOR:
        # Dividing by beta turns even the smallest NORMAL input into a subnormal
        # intermediate, which the SFPU may flush. Outputs this small cannot matter
        # to the activation, so accept either outcome and let the caller log which.
        if actual == 0 or (actual - golden).abs() <= rtol * golden.abs():
            return None
        return f"tiny: expected {golden.item():.6e} or 0, got {actual.item():.6e}"
    if (actual - golden).abs() > rtol * golden.abs():
        return f"expected {golden.item():.6e}, got {actual.item():.6e}"
    return None


@pytest.mark.parametrize("label, tile_fn, tile_init_fn, beta, reference", OP_CASES, ids=OP_CASE_IDS)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["bf16_dst", "fp32_dst"])
def test_situ_edge_values(device, label, tile_fn, tile_init_fn, beta, reference, fp32_dest_acc_en):
    """Zeros, signed zeros, infinities, subnormals and the saturation points.

    Elementwise rather than PCC: the tile is mostly a single padding constant, so a
    correlation over it would be dominated by that and say nothing about the edges.
    """
    edges = edge_value_inputs(beta)
    # Pad to a full tile with a benign value; the padding is compared too.
    values = edges + [1.0] * (TILE_ELEMS - len(edges))
    torch_input = torch.tensor(values, dtype=torch.bfloat16)

    golden = reference(torch_input, beta)
    actual = run_situ_op_on_device(device, torch_input, label, beta, fp32_dest_acc_en)

    failures = []
    for i, x in enumerate(edges):
        why = edge_mismatch(golden[i], actual[i], rtol=2.0e-2)
        logger.info(
            f"{label} beta={beta}  x={x!s:>12}  golden={golden[i].item():>13.6e}  got={actual[i].item():>13.6e}"
        )
        if why is not None:
            failures.append(f"x={x!r}: {why}")

    for name, value in (("smallest normal", SMALLEST_NORMAL_BF16), ("subnormal", SMALLEST_NORMAL_BF16 / 2)):
        slot = edges.index(value)
        logger.info(f"{label}: {name} input {'FLUSHED to zero' if actual[slot] == 0 else 'preserved'}")

    # The padding must be clean too — a wrong lane count would show up here.
    pad_why = edge_mismatch(golden[len(edges)], actual[len(edges)], rtol=2.0e-2)
    if pad_why is not None:
        failures.append(f"padding: {pad_why}")

    assert not failures, "edge-value mismatches:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("label, tile_fn, tile_init_fn, beta, reference", OP_CASES, ids=OP_CASE_IDS)
@pytest.mark.parametrize("num_tiles", [1, 8])
def test_ttnn_situ_op(device, label, tile_fn, tile_init_fn, beta, reference, num_tiles):
    """The public ttnn op, exercising the whole host path: UnaryOpType -> codegen ->
    the same compute-API call the kernel test drives directly.

    The ttnn path fixes dst mode via the op's own ComputeConfig, so only the
    default is reachable here; the generic_op tests above cover fp32 dst.
    """
    torch_input = situ_test_input(beta, num_tiles)
    golden = reference(torch_input, beta)

    tt_input = ttnn.from_torch(
        torch_input.reshape([1, num_tiles, 32, 32]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(getattr(ttnn, label)(tt_input, beta)).reshape(torch_input.shape)

    assert actual.to(torch.float32).abs().max().item() <= beta * (1.0 + 1e-3)

    rel_err = report_error(f"ttnn.{label} beta={beta}", golden, actual)
    assert rel_err < REL_ERR_LIMIT
    assert_with_pcc(golden.to(torch.float32), actual.to(torch.float32), pcc=0.999)
