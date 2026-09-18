# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import ttnn_max, TTNN_REDUCTION_WRAPPERS

# Module-scoped device: these tests all run with the default device config, so the device is
# opened once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device


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

    yt = ttnn_max(xt, 2, memory_config=out_mem_config)

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

    ttnn_op = TTNN_REDUCTION_WRAPPERS[op]
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


# Tall-H ROW_MAJOR reduces: Ht_rm >= 16 splits the H reduce into FP32 partials collapsed by a second
# stage. Post-commit covers fp32 / keepdim=False; the bf16 and keepdim variants live here.
@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 3136, 144),  # EfficientNetB0 global-pool; Wt=5, split fills the grid
        (1, 1, 12544, 32),  # very tall, Wt=1, S=64 — deepest per-slice accumulation here
        (1, 1, 3137, 144),  # non-aligned H → last shard overhang (identity pad)
        (1, 1, 3136, 145),  # non-aligned W → last-tile clamp
    ],
)
def test_rm_reduce_h_axis_split(device, reduce_op, dtype, keepdim, shape):
    """H reduce on tall ROW_MAJOR input — exercises the multi-shard H-axis-split + combine path."""
    if dtype == ttnn.bfloat16 and shape == (1, 1, 12544, 32):
        pytest.skip("bf16 accumulation-limited at H=12544; covered by the FP32 variant")
    torch.manual_seed(0)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.rand(shape, dtype=torch_dtype)
    torch_op = torch.mean if reduce_op == "mean" else torch.sum
    torch_ref = torch_op(torch_input.float(), dim=-2, keepdim=keepdim).to(torch_dtype)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_op = ttnn.mean if reduce_op == "mean" else ttnn.sum
    tt_output = ttnn_op(tt_input, dim=-2, keepdim=keepdim)
    assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT

    if dtype == ttnn.float32:
        # Only mean has an accurate fp32 SFPU reduce; sum goes through the TF32-truncating FPU.
        rtol = 0.002 if reduce_op == "mean" else 0.004
        pcc_threshold, atol, frobenius_threshold = 0.999, 1e-3, 0.003
    else:
        pcc_threshold, rtol, atol, frobenius_threshold = 0.97, 0.01, 0.02, 0.005
    assert_numeric_metrics(
        torch_ref,
        ttnn.to_torch(tt_output),
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
        check_ulp=False,
    )


# Partials are always ROW_MAJOR, so output_layout only selects what the final combine stage emits —
# orthogonal to dtype, so bfloat16 alone covers it.
@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["rm", "tile"])
@pytest.mark.parametrize("shape", [(1, 1, 784, 32), (1, 1, 1281, 144)])
def test_rm_reduce_h_axis_split_output_layout(device, reduce_op, keepdim, output_layout, shape):
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_op = torch.mean if reduce_op == "mean" else torch.sum
    torch_ref = torch_op(torch_input.float(), dim=-2, keepdim=keepdim).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_op = ttnn.mean if reduce_op == "mean" else ttnn.sum
    tt_output = ttnn_op(tt_input, dim=-2, keepdim=keepdim, output_layout=output_layout)
    assert tt_output.layout == output_layout

    assert_numeric_metrics(
        torch_ref,
        ttnn.to_torch(tt_output),
        pcc_threshold=0.97,
        rtol=0.01,
        atol=0.02,
        frobenius_threshold=0.005,
        check_ulp=False,
    )
