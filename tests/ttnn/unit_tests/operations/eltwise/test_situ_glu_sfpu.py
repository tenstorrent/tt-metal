# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone device test for the fused situ_glu binary SFPU op (api/compute/situ_glu.h).

Drives the op through ttnn.generic_op with a minimal binary test kernel (no
production op wired), reaching both dst-accumulator modes, and compares against
the torch reference:

    situ_a  = beta_gate * tanh(gate / beta_gate) * sigmoid(gate)
    up_half = beta_up * tanh(up / beta_up)
    result  = situ_a * up_half
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from models.common.utility_functions import is_blackhole

BETA_GATE = 4.0  # Kimi K3 gate-half beta
BETA_UP = 25.0  # Kimi K3 up-half beta
TILE_ELEMS = 32 * 32

# (ttnn dtype, tile page bytes). bfp8_b: 1 mantissa byte/datum + 1 shared exp byte / 16.
IN_DTYPES = {
    "bf16": (ttnn.bfloat16, TILE_ELEMS * 2),
    "bfp8_b": (ttnn.bfloat8_b, TILE_ELEMS + TILE_ELEMS // 16),
}
OUT_PAGE_BYTES = TILE_ELEMS * 2  # op rounds its result to bf16

# The op always packs bf16, so the bf16 arm is gated in ULP (measured worst case: 1.4).
BF16_ULP = 4
BF16_PCC = 0.999
BFP8_PCC = 0.99


def situ_glu_reference(gate, up):
    g = gate.to(torch.float32)
    u = up.to(torch.float32)
    situ_a = BETA_GATE * torch.tanh(g / BETA_GATE) * torch.sigmoid(g)
    up_half = BETA_UP * torch.tanh(u / BETA_UP)
    return situ_a * up_half


def _coverage_inputs(num_tiles, seed=0):
    """Sweeps that force each half's tanh saturation clamp to be entered, plus a
    heavy tail, shuffled so every tile spans the range."""
    n = num_tiles * TILE_ELEMS
    torch.manual_seed(seed)
    gate = torch.cat([torch.linspace(-6 * BETA_GATE, 6 * BETA_GATE, n // 2), torch.randn(n - n // 2) * (2 * BETA_GATE)])
    up = torch.cat([torch.linspace(-6 * BETA_UP, 6 * BETA_UP, n // 2), torch.randn(n - n // 2) * (2 * BETA_UP)])
    perm = torch.randperm(n)
    return gate[perm].to(torch.bfloat16), up[perm].to(torch.bfloat16)


def _run(device, gate_t, up_t, in_dtype, page_bytes, fp32_dest, dst_out=0):
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
            kernel_source="tests/tt_metal/tt_metal/test_kernels/compute/situ_glu.cpp",
            core_ranges=core,
            compile_time_args=[num_tiles, dst_out],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest),
        ),
    ]

    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    out = ttnn.generic_op([gate, up, output], program)
    return ttnn.to_torch(out).reshape(gate_t.shape)


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu SFPU op is implemented for Blackhole only")
@pytest.mark.parametrize("in_name", list(IN_DTYPES), ids=list(IN_DTYPES))
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["bf16_dst", "fp32_dst"])
# out_tile_idx aliasing the gate operand is what an expert kernel does; a separate dst slot is
# what catches a kernel that ignores out_tile_idx.
@pytest.mark.parametrize("dst_out", [0, 2], ids=["out_aliases_gate", "out_separate"])
def test_situ_glu_sfpu(device, in_name, fp32_dest, dst_out):
    in_dtype, page_bytes = IN_DTYPES[in_name]
    num_tiles = 8
    gate_t, up_t = _coverage_inputs(num_tiles)

    golden = situ_glu_reference(gate_t, up_t)
    actual = _run(device, gate_t, up_t, in_dtype, page_bytes, fp32_dest, dst_out)

    is_bfp8 = in_name == "bfp8_b"
    # |situ_a| <= beta_gate, |up_half| <= beta_up -> |result| <= their product.
    bound = BETA_GATE * BETA_UP * (1.0 + (5e-2 if is_bfp8 else 2**-8))
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
