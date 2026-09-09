# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone device test for the fused clamped_silu_glu binary SFPU op (api/compute/clamped_silu_glu.h).

Drives the op through ttnn.generic_op with a minimal binary test kernel, reaching both
dst-accumulator modes, and compares against the torch reference:

    gate_c = min(gate, limit)
    up_c   = clamp(up, -limit, limit)
    result = gate_c * sigmoid(gate_c) * up_c

This is DeepSeek-V4's routed-expert activation.
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from models.common.utility_functions import is_blackhole

LIMIT = 10.0  # DeepSeek-V4 swiglu_limit, baked as ClampedSiluGluConfigDsV4
TILE_ELEMS = 32 * 32

# (ttnn dtype, tile page bytes). bfp8_b: 1 mantissa byte/datum + 1 shared exp byte / 16.
IN_DTYPES = {
    "bf16": (ttnn.bfloat16, TILE_ELEMS * 2),
    "bfp8_b": (ttnn.bfloat8_b, TILE_ELEMS + TILE_ELEMS // 16),
}
OUT_PAGE_BYTES = TILE_ELEMS * 2  # the output CB is bf16

# The op always packs bf16, so the bf16 arm is gated in ULP (measured worst case: 0.89).
# bfp8_b quantizes the inputs before the op runs, so that arm is gated by PCC only.
BF16_ULP = 2
BF16_PCC = 0.999
BFP8_PCC = 0.99


def clamped_silu_glu_reference(gate, up):
    g = gate.to(torch.float32)
    u = up.to(torch.float32)
    gate_c = torch.clamp(g, max=LIMIT)
    up_c = torch.clamp(u, min=-LIMIT, max=LIMIT)
    return gate_c * torch.sigmoid(gate_c) * up_c


def _coverage_inputs(num_tiles, seed=0):
    """Sweeps that force the gate's upper clamp and BOTH of the up clamp's tails, plus the
    SiLU tail on the unclamped side, shuffled so every tile spans the range. gate and up are
    permuted independently over different endpoints so they do not correlate.
    """
    n = num_tiles * TILE_ELEMS
    torch.manual_seed(seed)
    gate = torch.cat([torch.linspace(-3 * LIMIT, 3 * LIMIT, n // 2), torch.randn(n - n // 2) * LIMIT])
    up = torch.cat([torch.linspace(-2 * LIMIT, 4 * LIMIT, n // 2), torch.randn(n - n // 2) * LIMIT])
    return gate[torch.randperm(n)].to(torch.bfloat16), up[torch.randperm(n)].to(torch.bfloat16)


def _run(device, gate_t, up_t, in_dtype, page_bytes, fp32_dest, dst_gate=0, dst_up=1, dst_out=0, skip_init=False):
    num_tiles = gate_t.numel() // TILE_ELEMS
    shape = [1, num_tiles, 32, 32]

    gate = ttnn.from_torch(
        gate_t.reshape(shape),
        dtype=in_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    up = ttnn.from_torch(
        up_t.reshape(shape),
        dtype=in_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cb_gate, cb_up, cb_out = 0, 1, 16

    def cb(idx, fmt, page):
        return ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(cb_gate, in_dtype, page_bytes),
        cb(cb_up, in_dtype, page_bytes),
        cb(cb_out, ttnn.bfloat16, OUT_PAGE_BYTES),
    ]

    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [gate.buffer_address(), up.buffer_address(), num_tiles, 0]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [output.buffer_address(), num_tiles, 0]

    reader_cta = (
        ttnn.TensorAccessorArgs(gate).get_compile_time_args() + ttnn.TensorAccessorArgs(up).get_compile_time_args()
    )
    writer_cta = [cb_out] + ttnn.TensorAccessorArgs(output).get_compile_time_args()

    kernels = [
        # Generic two-tensor reader (gate -> c_0, up -> c_1); only its name says situ_glu.
        ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/dataflow/reader_situ_glu.cpp",
            core_ranges=core,
            compile_time_args=reader_cta,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=writer_cta,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/compute/clamped_silu_glu.cpp",
            core_ranges=core,
            compile_time_args=[num_tiles, dst_gate, dst_up, dst_out, int(skip_init)],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest),
        ),
    ]

    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    out = ttnn.generic_op([gate, up, output], program)
    return ttnn.to_torch(out).reshape(gate_t.shape)


@pytest.mark.skipif(not is_blackhole(), reason="clamped_silu_glu SFPU op is implemented for Blackhole only")
@pytest.mark.parametrize("in_name", list(IN_DTYPES), ids=list(IN_DTYPES))
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["bf16_dst", "fp32_dst"])
# (gate, up, out) dst placements; "production" mirrors the fused call BINARY_ACT_TILE(j, c + j, j).
# Every index stays under 4: fp32 dest accumulate holds only 4 dst tiles, and both modes run here.
@pytest.mark.parametrize(
    "dst_gate, dst_up, dst_out",
    [
        pytest.param(0, 1, 0, id="out_aliases_gate"),
        pytest.param(0, 1, 2, id="out_separate"),
        pytest.param(1, 3, 1, id="production"),
    ],
)
def test_clamped_silu_glu_sfpu(device, in_name, fp32_dest, dst_gate, dst_up, dst_out):
    in_dtype, page_bytes = IN_DTYPES[in_name]
    num_tiles = 8
    gate_t, up_t = _coverage_inputs(num_tiles)

    golden = clamped_silu_glu_reference(gate_t, up_t)
    actual = _run(device, gate_t, up_t, in_dtype, page_bytes, fp32_dest, dst_gate, dst_up, dst_out)

    is_bfp8 = in_name == "bfp8_b"
    # |gate_c * sigmoid(gate_c)| <= limit and |up_c| <= limit -> |result| <= their product.
    bound = LIMIT * LIMIT * (1.0 + (5e-2 if is_bfp8 else 2**-8))
    assert actual.to(torch.float32).abs().max().item() <= bound

    g = golden.to(torch.float32)
    a = actual.to(torch.float32)
    logger.debug(f"{in_name} fp32_dst={fp32_dest}: max abs err {(a - g).abs().max().item():.4e}")

    if is_bfp8:
        # bfp8_b inputs quantize before the op runs, so the output carries hundreds of bf16 ULP
        # no matter how accurate the SFPU is; ULP only says something about the bf16 arm.
        assert_with_pcc(g, a, pcc=BFP8_PCC)
    else:
        assert_with_ulp(golden, actual, ulp_threshold=BF16_ULP)
        assert_with_pcc(g, a, pcc=BF16_PCC)


@pytest.mark.skipif(not is_blackhole(), reason="clamped_silu_glu SFPU op is implemented for Blackhole only")
def test_clamped_silu_glu_init_is_required(device, expect_error):
    """The op's sigmoid reaches sfpu_reciprocal_iter, which needs 2.0f in vConstFloatPrgm0. The
    test kernel leaves a wrong value there, so skipping the init disables the Newton step.
    """
    in_dtype, page_bytes = IN_DTYPES["bf16"]
    gate_t, up_t = _coverage_inputs(8)
    golden = clamped_silu_glu_reference(gate_t, up_t)

    assert_with_ulp(golden, _run(device, gate_t, up_t, in_dtype, page_bytes, False), ulp_threshold=BF16_ULP)
    with expect_error(AssertionError, "Max ULP Delta"):
        assert_with_ulp(
            golden,
            _run(device, gate_t, up_t, in_dtype, page_bytes, False, skip_init=True),
            ulp_threshold=BF16_ULP,
        )
