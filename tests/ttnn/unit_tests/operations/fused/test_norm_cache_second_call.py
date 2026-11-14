# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A second call through the program cache must read and write the buffers of that call.

On a cache hit the compiled program is reused and only the buffer addresses are patched. A stale
address (#49159, #43204) makes the second call read or write the buffer of the first. Counting cache
entries does not see this, so each test runs the op twice and checks the second output:

1. run the op on tensors A: a cache miss that compiles and caches the program;
2. allocate a spacer tensor and build new tensors B from new data: the first tensors are still alive,
   so B lands at other addresses;
3. run the op on B: the program cache must not grow;
4. compare the output with torch, and bit for bit with a cold-cache rerun of the same data.

Steps 1 to 4 are check_second_call_through_cache in tests/ttnn/unit_tests/operations/test_utils.py. The
in-place softmax cases are in test_softmax_program_cache.py.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.unit_tests.operations.test_utils import check_second_call_through_cache

EPS = 1e-5


@pytest.fixture
def isolate_program_cache(device):
    """Start each test with an empty program cache and clean up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


@pytest.mark.parametrize("op", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("residual_aliases_input", [False, True], ids=["residual", "residual_is_input"])
def test_norm_second_call_new_weight_bias_residual(device, isolate_program_cache, op, residual_aliases_input):
    """New weight, bias and residual buffers on the second call. With residual_is_input the first call has
    a separate residual and the second passes the input tensor as its own residual: same program, but two
    bindings that now point at one buffer."""
    batch, h, w = 2, 64, 128
    is_layer_norm = op == "layer_norm"

    def build(seed, second_call):
        torch.manual_seed(seed)
        torch_input = torch.rand((batch, h, w), dtype=torch.bfloat16)
        torch_residual = torch.rand((batch, h, w), dtype=torch.bfloat16)
        torch_weight = torch.rand((w,), dtype=torch.bfloat16)
        torch_bias = torch.rand((w,), dtype=torch.bfloat16)

        alias = residual_aliases_input and second_call
        torch_added = torch_input + (torch_input if alias else torch_residual)
        added = torch_added.float()
        if is_layer_norm:
            golden = F.layer_norm(added, (w,), torch_weight.float(), torch_bias.float(), EPS)
        else:
            golden = added * torch.rsqrt(added.pow(2).mean(-1, keepdim=True) + EPS) * torch_weight.float()

        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        tt_residual = tt_input if alias else ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)
        tt_weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
        tt_bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)
        return (tt_input, tt_residual, tt_weight, tt_bias), golden

    def run(inputs):
        tt_input, tt_residual, tt_weight, tt_bias = inputs
        if is_layer_norm:
            tt_output = ttnn.layer_norm(
                tt_input, epsilon=EPS, weight=tt_weight, bias=tt_bias, residual_input_tensor=tt_residual
            )
        else:
            tt_output = ttnn.rms_norm(tt_input, epsilon=EPS, weight=tt_weight, residual_input_tensor=tt_residual)
        return ttnn.to_torch(tt_output).float()

    check_second_call_through_cache(device, build, run)


def test_group_norm_height_sharded_second_call(device, isolate_program_cache):
    """Height-sharded group_norm (Welford, in place): the input shard lives in L1, so the second call's
    shard is at another L1 address, and its mask, gamma and beta are new DRAM buffers."""
    N, C, H, W, num_groups = 1, 320, 32, 32, 16
    grid_size = ttnn.CoreGrid(y=1, x=8)

    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
    )
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    def build(seed, second_call):
        torch.manual_seed(seed)
        torch_input = torch.rand((N, C, H, W), dtype=torch.bfloat16)
        torch_weight = torch.rand((C,), dtype=torch.bfloat16)
        torch_bias = torch.rand((C,), dtype=torch.bfloat16)
        golden = F.group_norm(torch_input.float(), num_groups, torch_weight.float(), torch_bias.float())
        golden = golden.permute(0, 2, 3, 1).reshape(N, 1, W * H, C)

        tt_input = ttnn.from_torch(
            torch_input.permute(0, 2, 3, 1).reshape(N, 1, W * H, C),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_input = ttnn.to_memory_config(tt_input, sharded_mem_config)

        tt_mask = ttnn.to_device(
            ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y, ttnn.DataType.BFLOAT8_B), device
        )
        tt_gamma, tt_beta = (
            ttnn.from_torch(
                ttnn.create_group_norm_weight_bias_rm(torch_tensor, C, grid_size.y),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for torch_tensor in (torch_weight, torch_bias)
        )
        return (tt_input, tt_mask, tt_gamma, tt_beta), golden

    def run(inputs):
        tt_input, tt_mask, tt_gamma, tt_beta = inputs
        tt_output = ttnn.group_norm(
            tt_input,
            num_groups=num_groups,
            input_mask=tt_mask,
            weight=tt_gamma,
            bias=tt_beta,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
            use_welford=True,
        )
        return ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(tt_output, ttnn.DRAM_MEMORY_CONFIG))).float()

    check_second_call_through_cache(device, build, run)
