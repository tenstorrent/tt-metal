# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


@pytest.mark.parametrize("N", [8, 16])
@pytest.mark.parametrize("in_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_reduce_h(N, in_sharded, out_sharded, dtype, device, function_level_defaults):
    torch.manual_seed(0)
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    C = 1
    H = 64
    W = 2048

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttnn.TILE_LAYOUT,
    ).to(
        device,
        interleaved_mem_config,
    )

    if in_sharded:
        xt = ttnn.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    yt = ttnn.max(xt, 2, memory_config=out_mem_config)

    if out_sharded:
        yt = ttnn.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    y = torch.amax(x, 2)

    if dtype == ttnn.bfloat16:
        pcc_threshold = 1
        rtol = 1e-06
        atol = 1e-06
        frobenius_threshold = 1e-09
    else:
        pcc_threshold = 0.999
        rtol = 0.032
        atol = 0.039
        frobenius_threshold = 0.005

    # test for equivalance
    assert_numeric_metrics(
        y,
        tt_got_back,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("op", ["max", "var"])
def test_nd_sharded_reduce_h_no_output_shard_spec(op, device, function_level_defaults):
    """Reduce with ND_SHARDED output MemoryConfig that omits nd_shard_spec.

    The shard spec is optional in MemoryConfig. When absent, the reduce operation
    should infer grid and shard shape from the input tensor.
    Parametrized over max (ReduceDeviceOperation) and var (WelfordReduceDeviceOperation)
    to cover both code paths.
    """
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    N = 1
    C = 1
    H = 64
    W = 2048

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    nd_sharded_out_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.ND_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
    ).to(device, interleaved_mem_config)

    xt = ttnn.interleaved_to_sharded(
        xt,
        grid_size,
        [N * C * H, W // (grid_size[0] * grid_size[1])],
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )

    ttnn_op = getattr(ttnn, op)
    torch_op_name = {"max": "amax", "min": "amin"}.get(op, op)
    torch_op = getattr(torch, torch_op_name)

    yt = ttnn_op(xt, 2, memory_config=nd_sharded_out_config)
    y = torch_op(x, 2)

    yt = ttnn.sharded_to_interleaved(yt, interleaved_mem_config)
    tt_got_back = yt.cpu().to_torch()

    assert_numeric_metrics(
        y,
        tt_got_back,
        pcc_threshold=0.999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=0.02,
    )
