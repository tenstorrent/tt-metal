# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("factory", ["reuse", "mcast_1d", "mcast_2d"])
@pytest.mark.parametrize("packer_l1_acc", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_subblock_writer_chunks(device, factory, packer_l1_acc, has_bias):
    # Two batches exercise reuse of aliased output/partials storage. Multicast
    # shapes leave partial subblocks and fully padded tiles at the block edges.
    m = n = 128 if factory == "reuse" else 224
    k = 256
    generator = torch.Generator().manual_seed(42)
    a = torch.randint(-2, 3, (2, 1, m, k), generator=generator).to(torch.bfloat16)
    b = torch.eye(k, n, dtype=torch.bfloat16).repeat(2, 1, 1, 1)
    b[1].neg_()
    expected = a @ b
    config_args = dict(in0_block_w=2, out_subblock_h=2, out_subblock_w=2, per_core_M=4, per_core_N=4)
    if factory == "reuse":
        config = ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(1, 1), **config_args)
    elif factory == "mcast_1d":
        config_args["per_core_M"] = 8
        config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(2, 1), fuse_batch=False, mcast_in0=True, **config_args
        )
    else:
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(2, 2), fuse_batch=False, transpose_mcast=False, **config_args
        )

    def to_device(value):
        return ttnn.from_torch(
            value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    bias = None
    if has_bias:
        bias_shape = (2, 1, m, n) if factory == "reuse" else (1, 1, 1, n)
        bias_host = torch.randint(-2, 3, bias_shape, generator=generator).to(torch.bfloat16)
        expected += bias_host
        bias = to_device(bias_host)
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=packer_l1_acc
    )
    output = ttnn.linear(
        to_device(a),
        to_device(b),
        bias=bias,
        program_config=config,
        compute_kernel_config=compute_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch.testing.assert_close(ttnn.to_torch(output), expected, rtol=0, atol=0)


@pytest.mark.parametrize("packer_l1_acc", [False, True])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "relu", "relu6"])
@pytest.mark.parametrize("untilize_out", [False, True])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["bf16-dst", "fp32-dst"])
def test_fused_output_block_state(
    device, packer_l1_acc, transpose_a, has_bias, activation, untilize_out, fp32_dest_acc_en
):
    # Two batches and 2x2 output blocks per core exercise state restoration and
    # streamed width-block bias. Each transposed K block has only two tiles,
    # exercising the transpose helper's partial-DST-chunk path as well. FP32
    # accumulation makes the partials and untilized output formats differ.
    generator = torch.Generator().manual_seed(43)
    a = torch.randint(-8, 9, (2, 1, 128, 96), generator=generator).to(torch.bfloat16)
    # Unbatched weights keep bias inside the fused compute kernel; batched
    # weights route bias through a separate host-selected add operation.
    b = torch.eye(96, 128, dtype=torch.bfloat16).reshape(1, 1, 96, 128)
    expected = a @ b
    bias = torch.randint(-2, 3, (1, 1, 1, 128), generator=generator).to(torch.bfloat16) if has_bias else None
    if has_bias:
        expected += bias
    unary_op = None
    if activation == "relu":
        expected = expected.relu()
        unary_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    elif activation == "relu6":
        expected = expected.clamp(0, 6)
        unary_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6)
    config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=1,
        out_subblock_h=2,
        out_subblock_w=2,
        out_block_h=2,
        out_block_w=2,
        per_core_M=4,
        per_core_N=4,
        fuse_batch=False,
        mcast_in0=True,
        fused_activation=unary_op,
        untilize_out=untilize_out,
    )

    def to_device(value):
        return ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.linear(
        to_device(a.transpose(-1, -2).contiguous() if transpose_a else a),
        to_device(b),
        bias=to_device(bias) if has_bias else None,
        transpose_a=transpose_a,
        program_config=config,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest_acc_en, packer_l1_acc=packer_l1_acc
        ),
    )
    torch.testing.assert_close(ttnn.to_torch(output), expected, rtol=0, atol=0)
