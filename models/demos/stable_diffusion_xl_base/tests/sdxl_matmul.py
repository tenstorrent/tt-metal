# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from tt_py_test_utils_common.matmul_determinism import ttnn_linear
from tt_py_test_utils_common.utils_for_testing import assert_numeric_metrics


def test_sdxl_matmul(
    device,
    core_grid,
    M,
    K,
    N,
    in0_block_w,
    out_subblock_h,
    out_subblock_w,
    per_core_M,
    per_core_N,
    has_gelu,
    perf_test_mode=False,
):
    torch.manual_seed(0)

    act_shape = (1, 1, M, K)
    weights_shape = (1, 1, K, N)
    bias_shape = (1, 1, N)

    torch_act = torch.randn(act_shape, dtype=torch.bfloat16)
    torch_weights = torch.randn(weights_shape, dtype=torch.bfloat16)
    torch_bias = torch.randn(bias_shape, dtype=torch.bfloat16)

    tt_act = ttnn.from_torch(torch_act, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_weights = ttnn.from_torch(torch_weights, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)
    tt_bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)

    if not perf_test_mode:
        torch_output_tensor = torch_act @ torch_weights + torch_bias
        if has_gelu:
            torch_output_tensor = torch.nn.functional.gelu(torch_output_tensor)

    sharded_mem_config = ttnn.create_sharded_memory_config(
        act_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_act_block_sharded = ttnn.to_memory_config(tt_act, sharded_mem_config)
    prog_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fuse_batch=True,
        fused_activation=[ttnn.UnaryOpType.GELU, True] if has_gelu else None,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    output_tensor = ttnn_linear(
        tt_act_block_sharded,
        tt_weights,
        bias=tt_bias,
        program_config=prog_config,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.synchronize_device(device)

    if not perf_test_mode:
        output_tensor = ttnn.to_torch(output_tensor)
        assert_numeric_metrics(
            torch_output_tensor,
            output_tensor,
            check_allclose=False,
            check_frobenius=False,
            pcc_threshold=0.999,
            check_ulp=False,
        )
