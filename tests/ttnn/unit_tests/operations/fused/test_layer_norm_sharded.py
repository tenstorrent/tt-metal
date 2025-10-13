# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.test_utils import TILE_HEIGHT, TILE_WIDTH


def skip_welford_blackhole(use_welford):
    return pytest.mark.skipif(
        use_welford and is_blackhole(), reason="Welford's algorithm is not supported on Blackhole"
    )


def generate_input_tensor(h, w, type, dtype):
    """
    Generate various torch tensors
    Returns:
        A torch tensor of shape (h, w) of the given type and dtype.
    """
    if type == "random":
        return torch.rand((h, w), dtype=dtype)
    elif type == "random_normal":
        return torch.randn((h, w), dtype=dtype)
    elif type == "ascending_values_repeated_rows":
        return torch.arange(w).repeat(h, 1).to(dtype)
    elif type == "monotonically_ascending_values":
        return torch.arange(h * w).reshape(h, w).to(dtype)
    else:
        raise ValueError(f"Invalid tensor type: {type}")


def create_shard_spec(shard_height, shard_width, num_cores_h, num_cores_w):
    return ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1),
                )
            }
        ),
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )


def create_single_stage_shard_spec(h, w, num_cores_h, num_cores_w):
    shard_height = h // num_cores_h
    shard_width = w // num_cores_w
    return create_shard_spec(shard_height, shard_width, num_cores_h, num_cores_w)


def create_two_stage_shard_spec(h, w, num_cores_h, num_cores_w):
    shard_height = h
    shard_width = w // (num_cores_w * num_cores_h)
    return create_shard_spec(shard_height, shard_width, num_cores_h, num_cores_w)


def create_sharded_mem_config(h, w, num_cores_h, num_cores_w, two_stage):
    mem_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED if two_stage else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    if two_stage:
        shard_spec = create_two_stage_shard_spec(h, w, num_cores_h, num_cores_w)
    else:
        shard_spec = create_single_stage_shard_spec(h, w, num_cores_h, num_cores_w)
    return ttnn.MemoryConfig(memory_layout=mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec)


def layer_norm_sharded(
    device, tt_input_tensor, use_welford, block_ht, block_wt, subblock_w=1, residual=None, weight=None, bias=None
):
    """
    Run layer norm sharded on a TTNN tensor.
    Args:
        device: The device to run the layer norm on.
        tt_input_tensor: The TTNN tensor to run the layer norm on.
        use_welford: Whether to use Welford's algorithm.
        block_ht: The height of the block in tiles.
        block_wt: The width of the block in tiles.
        residual: The residual tensor to add to the input tensor.
        weight: The weight tensor to use for the layer norm.
        bias: The bias tensor to use for the layer norm.
    Returns:
        The output tensor as a torch tensor.
    """
    # Create output memory config (same sharding as input)
    output_memory_config = ttnn.get_memory_config(tt_input_tensor)

    # Run layernorm
    output_ttnn = ttnn.layer_norm(
        tt_input_tensor,
        residual_input_tensor=residual,
        weight=weight,
        bias=bias,
        memory_config=output_memory_config,
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=block_ht,
            block_w=block_wt,
            subblock_w=subblock_w,
            use_welford=use_welford,
            inplace=False,
        ),
    )

    output_ttnn = ttnn.to_layout(output_ttnn, ttnn.ROW_MAJOR_LAYOUT)
    output_ttnn = ttnn.from_device(output_ttnn)
    return ttnn.to_torch(output_ttnn)


def torch_layer_norm(torch_input_tensor, residual=None, weight=None, bias=None):
    """
    Run layer norm in torch
    Args:
        torch_input_tensor: The input tensor to run the layer norm on.
        residual: The residual tensor to add to the input tensor.
        weight: The weight tensor to use for the layer norm.
        bias: The bias tensor to use for the layer norm.
    Returns:
        The output tensor as a torch tensor.
    """
    if residual is not None:
        torch_input_tensor += residual
    return torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[torch_input_tensor.shape[1]], weight=weight, bias=bias
    )


def single_stage_param_sets():
    """
    Generate valid single-stage reduction tensor,block and shard shapes
    for input h,w
    Returns:
        List[Tuple] of (h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt) sets
    """
    hs = [32 * 2, 32 * 8]
    ws = [32 * 2, 32 * 16]
    num_cores_hs = [4, 8]
    num_cores_ws = [4, 8]
    block_ht_mults = [1, 2, 4, 8]
    block_wt_mults = [1, 2, 4, 8]
    possible_subblock_wts = [1, 2]
    param_sets = []
    for h in hs:
        for w in ws:
            for num_cores_h in num_cores_hs:
                for num_cores_w in num_cores_ws:
                    h_per_core = h // num_cores_h
                    w_per_core = w // num_cores_w
                    ht_per_core = h_per_core // TILE_HEIGHT
                    wt_per_core = w_per_core // TILE_WIDTH
                    possible_block_hts = [
                        v for v in [h_per_core // (TILE_HEIGHT * m) for m in block_ht_mults] if v >= 1
                    ]
                    possible_block_wts = [v for v in [w_per_core // (TILE_WIDTH * m) for m in block_wt_mults] if v >= 1]
                    for block_ht in possible_block_hts:
                        for block_wt in possible_block_wts:
                            for subblock_wt in possible_subblock_wts:
                                block_ht_valid = block_ht >= 1 and block_ht == ht_per_core
                                block_wt_valid = block_wt >= 1 and block_wt == wt_per_core
                                subblock_wt_valid = subblock_wt >= 1 and subblock_wt <= block_wt
                                if block_ht_valid and block_wt_valid and subblock_wt_valid:
                                    param_sets.append((h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt))
    return param_sets


def layernorm_test_main(
    device,
    h,
    w,
    num_cores_h,
    num_cores_w,
    block_ht,
    block_wt,
    subblock_wt,
    use_welford,
    two_stage,
    tensor_type,
    dtype,
    residual=None,
    weight=None,
    bias=None,
):
    torch.manual_seed(12345)

    # Run torch layernorm to get the reference tensor
    torch_input_tensor = generate_input_tensor(h, w, tensor_type, dtype)
    torch_output_tensor = torch_layer_norm(torch_input_tensor, residual=residual, weight=weight, bias=bias)

    # Generate the tt tensor based on the inputs
    sharded_mem_config = create_sharded_mem_config(h, w, num_cores_h, num_cores_w, two_stage)
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=sharded_mem_config,
    )

    # Run layernorm
    if residual is not None:
        residual = ttnn.from_torch(residual, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem_config)
    if weight is not None:
        weight = ttnn.from_torch(weight, layout=ttnn.TILE_LAYOUT, device=device)
    if bias is not None:
        bias = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=device)
    output_ttnn = layer_norm_sharded(
        device,
        tt_input_tensor,
        use_welford,
        block_ht,
        block_wt,
        subblock_wt,
        residual=residual,
        weight=weight,
        bias=bias,
    )

    # Check PCC
    assert_with_pcc(torch_output_tensor, output_ttnn, 0.9998)


@pytest.mark.parametrize("h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt", single_stage_param_sets())
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm_sharded_single_stage(
    device, h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt, use_welford, two_stage, tensor_type, dtype
):
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
    )


@pytest.mark.parametrize(
    "h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt",
    [(32 * 2, 32 * 4, 2, 2, 2, 1, 1), (32 * 4, 32 * 8, 4, 2, 4, 1, 1), (32 * 8, 32 * 16, 2, 4, 8, 2, 1)],
)
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [True])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm_sharded_two_stage(
    device, h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt, use_welford, two_stage, tensor_type, dtype
):
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
    )


def simple_size_params(two_stage):
    h = 32 * 8
    w = 32 * 10
    num_cores_h = 2
    num_cores_w = 5
    if two_stage:
        block_ht = 8
        block_wt = 1
        subblock_wt = 1
    else:
        block_ht = 4
        block_wt = 2
        subblock_wt = 1

    return h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt


@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random_normal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm_sharded_with_residual(device, use_welford, two_stage, tensor_type, dtype):
    if tensor_type == "random" or tensor_type == "random_normal":
        pytest.skip("Low PCC, see <Ryan to create issue and add number here>")

    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    residual = generate_input_tensor(h, w, "random_normal", dtype)
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
        residual=residual,
        weight=None,
        bias=None,
    )


@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random_normal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm_sharded_with_weight_and_bias(device, use_welford, two_stage, tensor_type, dtype):
    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    weight = generate_input_tensor(1, w, "random", dtype)
    bias = generate_input_tensor(1, w, "random_normal", dtype)
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
        residual=None,
        weight=weight[0],
        bias=bias[0],
    )


@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm_sharded_with_weight_and_bias_and_residual(device, use_welford, two_stage, tensor_type, dtype):
    if tensor_type == "random" or tensor_type == "random_normal":
        pytest.skip("Low PCC, see <Ryan to create issue and add number here>")

    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    residual = generate_input_tensor(h, w, "random_normal", dtype)
    weight = generate_input_tensor(1, w, "random", dtype)
    bias = generate_input_tensor(1, w, "random_normal", dtype)
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
        residual=residual,
        weight=weight[0],
        bias=bias[0],
    )
