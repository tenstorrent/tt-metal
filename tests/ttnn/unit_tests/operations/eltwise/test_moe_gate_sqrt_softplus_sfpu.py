# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone device test for the fused moe_gate_sqrt_softplus_add_bias binary SFPU op
(api/compute/moe_gate_sqrt_softplus.h).

Drives the op through ttnn.generic_op with a minimal binary test kernel and compares both of its
in-place results against the torch reference:

    score = sqrt(softplus(x))     (beta = 1, threshold = 20)
    key   = score + bias

This is DeepSeek-V4's MoE gate activation (moe_grouped_topk, score_func="sqrtsoftplus").
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from tt_py_test_utils_common.utility_functions import is_blackhole

TILE_ELEMS = 32 * 32

# (ttnn dtype, tile page bytes, unpack-to-DEST mode). The gate feeds bf16 or fp32 logits and bias:
# bf16 is exact through SrcA, fp32 unpacks straight to DEST (SrcA would truncate it to TF32).
IN_DTYPES = {
    "bf16": (ttnn.bfloat16, TILE_ELEMS * 2, ttnn.UnpackToDestMode.Default),
    "fp32": (ttnn.float32, TILE_ELEMS * 4, ttnn.UnpackToDestMode.UnpackToDestFp32),
}
OUT_PAGE_BYTES = TILE_ELEMS * 4  # the output CB is fp32, as the gate's score and key CBs are

# Both results are fp32, so they are gated in fp32 ULP (measured worst case: 48.5 on the score, 24.4
# on the key); the stock bf16 softplus is ~13000 ULP off. The bar also catches an init that leaves one
# of the sqrt's programmable constants stale.
FP32_ULP = 96
PCC = 0.999


def moe_gate_sqrt_softplus_reference(x, bias):
    score = torch.sqrt(torch.nn.functional.softplus(x.to(torch.float64)))
    return score, score + bias.to(torch.float64)


def _coverage_inputs(num_tiles, seed=0):
    """A sweep through the far negative tail (softplus ~ e^x down to 2^-115), zero and past the
    softplus threshold (20), plus a normal draw around the production logit range, shuffled so every
    tile spans the range. The bias is non-negative so score + bias never cancels: ULP is measured
    against the key, and a cancelling sum would report the score's error relative to ~0.
    """
    n = num_tiles * TILE_ELEMS
    torch.manual_seed(seed)
    x = torch.cat([torch.linspace(-80.0, 30.0, n // 2), torch.randn(n - n // 2) * 8.0])
    bias = torch.rand(n) * 2.0
    return x[torch.randperm(n)], bias


def _run(device, x_t, bias_t, in_dtype, page_bytes, unpack_mode, dst_x=0, dst_bias=1, pack_key=False):
    num_tiles = x_t.numel() // TILE_ELEMS
    shape = [1, num_tiles, 32, 32]

    x = ttnn.from_torch(
        x_t.reshape(shape),
        dtype=in_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias = ttnn.from_torch(
        bias_t.reshape(shape),
        dtype=in_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cb_x, cb_bias, cb_out = 0, 1, 16

    def cb(idx, fmt, page):
        return ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(cb_x, in_dtype, page_bytes),
        cb(cb_bias, in_dtype, page_bytes),
        cb(cb_out, ttnn.float32, OUT_PAGE_BYTES),
    ]

    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [x.buffer_address(), bias.buffer_address(), num_tiles, 0]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [output.buffer_address(), num_tiles, 0]

    reader_cta = (
        ttnn.TensorAccessorArgs(x).get_compile_time_args() + ttnn.TensorAccessorArgs(bias).get_compile_time_args()
    )
    writer_cta = [cb_out] + ttnn.TensorAccessorArgs(output).get_compile_time_args()

    # The op needs fp32 DEST (it has no bf16 rounding step), as the gate's factory configures it.
    compute_config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)
    # Host descriptors use the maximum CB count so this vector covers both Wormhole (32) and Blackhole (64).
    unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
    unpack_modes[cb_x] = unpack_mode
    unpack_modes[cb_bias] = unpack_mode
    compute_config.unpack_to_dest_mode = unpack_modes

    kernels = [
        # Generic two-tensor reader (x -> c_0, bias -> c_1); only its name says situ_glu.
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
            kernel_source="tests/tt_metal/tt_metal/test_kernels/compute/moe_gate_sqrt_softplus.cpp",
            core_ranges=core,
            compile_time_args=[num_tiles, dst_x, dst_bias, int(pack_key)],
            runtime_args=[],
            config=compute_config,
        ),
    ]

    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    out = ttnn.generic_op([x, bias, output], program)
    return ttnn.to_torch(out).reshape(x_t.shape)


@pytest.mark.skipif(not is_blackhole(), reason="moe_gate_sqrt_softplus SFPU op is implemented for Blackhole only")
@pytest.mark.parametrize("in_name", list(IN_DTYPES), ids=list(IN_DTYPES))
@pytest.mark.parametrize("pack_key", [False, True], ids=["score", "key"])
# (x, bias) dst placements; "production" is the gate's call (0, 1). Every index stays under 4: fp32
# dest accumulate holds only 4 dst tiles.
@pytest.mark.parametrize(
    "dst_x, dst_bias",
    [
        pytest.param(0, 1, id="production"),
        pytest.param(1, 0, id="swapped"),
        pytest.param(1, 3, id="strided"),
    ],
)
def test_moe_gate_sqrt_softplus_sfpu(device, in_name, pack_key, dst_x, dst_bias):
    in_dtype, page_bytes, unpack_mode = IN_DTYPES[in_name]
    num_tiles = 8
    x_t, bias_t = _coverage_inputs(num_tiles)
    if in_dtype == ttnn.bfloat16:
        # The golden must see exactly what the device sees.
        x_t, bias_t = x_t.to(torch.bfloat16).float(), bias_t.to(torch.bfloat16).float()

    score, key = moe_gate_sqrt_softplus_reference(x_t, bias_t)
    golden = key if pack_key else score
    actual = _run(device, x_t, bias_t, in_dtype, page_bytes, unpack_mode, dst_x, dst_bias, pack_key)

    a = actual.to(torch.float32)
    # sqrt(softplus(x)) >= 0, so the score is non-negative and the key is at least the bias.
    floor = bias_t if pack_key else torch.zeros_like(a)
    assert (a >= floor).all()

    _, ulp_message = assert_with_ulp(expected_result=golden, actual_result=a, ulp_threshold=FP32_ULP)
    logger.debug(
        f"{in_name} {'key' if pack_key else 'score'}: max abs err {(a.double() - golden).abs().max().item():.4e}, "
        f"{ulp_message}"
    )
    assert_with_pcc(golden.float(), a, pcc=PCC)
