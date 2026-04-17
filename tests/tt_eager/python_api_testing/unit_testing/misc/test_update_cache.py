# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from models.common.utility_functions import nearest_32, pad_by_zero
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from models.common.utility_functions import skip_for_blackhole


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("max_seq_len", [4096])
@pytest.mark.parametrize("num_users", [8, 16, 32, 64])
@pytest.mark.parametrize("num_heads", [1, 2, 8])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
class TestUpdateCache:
    @pytest.mark.parametrize("seq_len", [32, 512, 2048, 4096])
    def test_fill_cache(self, seq_len, head_dim, max_seq_len, num_users, num_heads, in_sharded, input_dtype, device):
        cache_dtype = input_dtype
        input_shape = [1, num_heads, seq_len, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        for i in range(num_users):
            x = torch.randn(input_shape).bfloat16().float()
            xt = ttnn.Tensor(x, input_dtype).to(ttnn.TILE_LAYOUT)
            if in_sharded:
                compute_grid_size = device.compute_with_storage_grid_size()
                num_cores = min(seq_len // 32 * num_heads, 32)  # Always use max 32 cores for testing
                shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
                input_shard_spec = ttnn.ShardSpec(
                    shard_grid,
                    [
                        xt.volume() // xt.padded_shape[-1] // num_cores,
                        xt.padded_shape[-1],
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                )
                input_mem_config = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
                )
                xt = xt.to(device, input_mem_config)
            else:
                xt = xt.to(device)

            cachett = ttnn.fill_cache(cachett, xt, i)
            cache[i : i + 1, :, : x.shape[-2], :] = x

        tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
            eq, output = comp_equal(cache, tt_got_back)
        else:
            eq, output = comp_pcc(cache, tt_got_back)
        logger.info(output)
        assert eq

    @pytest.mark.parametrize("cache_idx", [0, 1, 127, 1057])
    @pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
    @pytest.mark.parametrize(
        "batch_offset", [0, 16]
    )  # Only used when num_users < 32 and batch_offset + num_users <= 32
    def test_update_cache_decode(
        self,
        cache_idx,
        head_dim,
        max_seq_len,
        num_users,
        batch_offset,
        num_heads,
        in_sharded,
        input_dtype,
        cache_dtype,
        device,
    ):
        if num_users > 32 or (num_users + batch_offset) > 32:
            pytest.skip("Batch offset is only used when num_users < 32 and batch_offset + num_users <= 32")
        if cache_dtype != ttnn.bfloat16:
            pytest.skip(
                "#12931: Update Cache currently produces non-deterministic output on GS when converting data types for cache tensor"
            )
        input_shape = [num_users, num_heads, 1, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        x = torch.randn(input_shape).bfloat16().float()
        # pad dim0 of x to 32 if batch size is less than 32, make 0-batch_offset elements 0, batch_offset-batch_offset+num_users elements non-zero, and rest 0
        x_new = x.clone()
        if num_users < 32:
            x_new = torch.cat((torch.zeros(batch_offset, num_heads, 1, head_dim), x_new), dim=0)
            x_new = torch.cat((x_new, torch.zeros(32 - num_users - batch_offset, num_heads, 1, head_dim)), dim=0)
            assert x_new.shape[0] == 32, f"Expected x.shape[0] to be 32, got {x_new.shape[0]}"
        xt = ttnn.Tensor(x_new.permute(2, 1, 0, 3), input_dtype).to(ttnn.TILE_LAYOUT)
        if in_sharded:
            compute_grid_size = device.compute_with_storage_grid_size()
            num_cores = min(max(num_users, 32) // 32 * num_heads, compute_grid_size.x * compute_grid_size.y)
            shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
            input_shard_spec = ttnn.ShardSpec(
                shard_grid,
                [
                    xt.volume() // xt.padded_shape[-1] // num_cores,
                    xt.padded_shape[-1],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            input_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
        else:
            xt = xt.to(device)

        cachett = ttnn.update_cache(cachett, xt, cache_idx, batch_offset=batch_offset)
        cache[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]] = x

        tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
            eq_cache, output_cache = comp_equal(cache, tt_got_back)  # checks the entire kv cache
            eq_update, output_update = comp_equal(
                x, tt_got_back[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]]
            )  # checks the updated parts
        else:
            eq_cache, output_cache = comp_pcc(cache, tt_got_back)  # checks the entire kv cache
            eq_update, output_update = comp_pcc(
                x, tt_got_back[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]]
            )  # checks the updated parts
        logger.info(output_cache)
        logger.info(output_update)
        assert eq_cache and eq_update


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [8, 16, 32, 64])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
class TestUpdateCacheFP32:
    @pytest.mark.parametrize("seq_len", [32, 512, 1024])
    def test_fill_cache_fp32(
        self, seq_len, head_dim, max_seq_len, num_users, num_heads, in_sharded, input_dtype, device
    ):
        cache_dtype = input_dtype

        input_shape = [1, num_heads, seq_len, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        for i in range(num_users):
            x = torch.randn(input_shape).bfloat16().float()
            xt = ttnn.Tensor(x, input_dtype).to(ttnn.TILE_LAYOUT)
            if in_sharded:
                compute_grid_size = device.compute_with_storage_grid_size()
                num_cores = min(seq_len // 32 * num_heads, 32)  # Always use max 32 cores for testing
                shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
                input_shard_spec = ttnn.ShardSpec(
                    shard_grid,
                    [
                        xt.volume() // xt.padded_shape[-1] // num_cores,
                        xt.padded_shape[-1],
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                )
                input_mem_config = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
                )
                xt = xt.to(device, input_mem_config)
            else:
                xt = xt.to(device)

            cachett = ttnn.fill_cache(cachett, xt, i)
            cache[i : i + 1, :, : x.shape[-2], :] = x

        tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
            eq, output = comp_equal(cache, tt_got_back)
        else:
            eq, output = comp_pcc(cache, tt_got_back)
        logger.info(output)
        assert eq

    @pytest.mark.parametrize("cache_idx", [0, 1, 127, 1057])
    @pytest.mark.parametrize("cache_dtype", [ttnn.float32])
    @pytest.mark.parametrize(
        "batch_offset", [0, 16]
    )  # Only used when num_users < 32 and batch_offset + num_users <= 32
    def test_update_cache_decode_fp32(
        self,
        cache_idx,
        head_dim,
        max_seq_len,
        num_users,
        batch_offset,
        num_heads,
        in_sharded,
        input_dtype,
        cache_dtype,
        device,
    ):
        if num_users > 32 or (num_users + batch_offset) > 32:
            pytest.skip("Batch offset is only used when num_users < 32 and batch_offset + num_users <= 32")
        input_shape = [num_users, num_heads, 1, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        x = torch.randn(input_shape).bfloat16().float()
        # pad dim0 of x to 32 if batch size is less than 32, make 0-batch_offset elements 0, batch_offset-batch_offset+num_users elements non-zero, and rest 0
        x_new = x.clone()
        if num_users < 32:
            x_new = torch.cat((torch.zeros(batch_offset, num_heads, 1, head_dim), x_new), dim=0)
            x_new = torch.cat((x_new, torch.zeros(32 - num_users - batch_offset, num_heads, 1, head_dim)), dim=0)
            assert x_new.shape[0] == 32, f"Expected x.shape[0] to be 32, got {x_new.shape[0]}"
        xt = ttnn.Tensor(x_new.permute(2, 1, 0, 3), input_dtype).to(ttnn.TILE_LAYOUT)
        if in_sharded:
            compute_grid_size = device.compute_with_storage_grid_size()
            num_cores = min(max(num_users, 32) // 32 * num_heads, compute_grid_size.x * compute_grid_size.y)
            shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
            input_shard_spec = ttnn.ShardSpec(
                shard_grid,
                [
                    xt.volume() // xt.padded_shape[-1] // num_cores,
                    xt.padded_shape[-1],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            input_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
        else:
            xt = xt.to(device)

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            fp32_dest_acc_en=True,
        )

        cachett = ttnn.update_cache(
            cachett, xt, cache_idx, batch_offset=batch_offset, compute_kernel_config=compute_kernel_config
        )
        cache[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]] = x

        tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
            eq_cache, output_cache = comp_equal(cache, tt_got_back)  # checks the entire kv cache
            eq_update, output_update = comp_equal(
                x, tt_got_back[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]]
            )  # checks the updated parts
        else:
            eq_cache, output_cache = comp_pcc(cache, tt_got_back)  # checks the entire kv cache
            eq_update, output_update = comp_pcc(
                x, tt_got_back[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]]
            )  # checks the updated parts
        logger.info(output_cache)
        logger.info(output_update)
        assert eq_cache and eq_update


@pytest.mark.parametrize(
    "mesh_shape,seq_len,expected_seq_len_local,expected_layout",
    [
        ((2, 4), 1024, 512, "ND_SHARDED"),  # 512/32 = 16 shards across 8 banks → 2 shards/bank → ND_SHARDED
        ((8, 4), 1024, 128, "HEIGHT_SHARDED"),  # 128/32 = 4 shards across 8 banks → ≤1 shard/bank → HEIGHT_SHARDED
    ],
)
@pytest.mark.parametrize("num_heads", [128])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("op_type", ["fill", "fill_cache_for_user", "update"])
class TestUpdateCacheWithKVPE:
    def test_kvpe_cache_op(
        self, op_type, mesh_shape, seq_len, expected_seq_len_local, expected_layout, num_heads, input_dtype, device
    ):
        """Test fill/update cache operations with KVPE cache created by init_kvpe_cache.

        Uses ND_SHARDED cache with ROUND_ROBIN_1D across 8 DRAM banks.
        The cache layout behavior depends on mesh size:
        - 2x4 mesh (seq_len=1024, seq_len_local=512): 512/32 = 16 shards → 2 shards/bank → ND_SHARDED
        - 8x4 mesh (seq_len=1024, seq_len_local=128): 128/32 = 4 shards → ≤1 shard/bank → HEIGHT_SHARDED
        """

        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

        kvpe_cache_head_dim = 32
        sp_axis = 0
        num_kvpe_cache_layers = 2
        mesh_device = device

        # Initialize KVPE cache using the utility function
        # This creates cache with shape [num_layers, 1, seq_len_local, kvpe_cache_head_dim]
        # with ND_SHARDED memory config using ROUND_ROBIN_1D across 8 DRAM banks
        tt_cache = init_kvpe_cache(
            kvpe_cache_head_dim=kvpe_cache_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=num_kvpe_cache_layers,
        )

        if expected_layout == "ND_SHARDED":
            expected_layout_enum = ttnn.TensorMemoryLayout.ND_SHARDED
        elif expected_layout == "HEIGHT_SHARDED":
            expected_layout_enum = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        else:
            raise ValueError(f"Unexpected expected_layout value: {expected_layout}")
        assert tt_cache.memory_config().memory_layout == expected_layout_enum, (
            f"Expected cache layout to be {expected_layout_enum}, " f"but got {tt_cache.memory_config().memory_layout}"
        )
        assert tt_cache.shape[-2] == expected_seq_len_local, (
            f"Expected local sequence length to be {expected_seq_len_local}, " f"but got {tt_cache.shape[-2]}"
        )

        logger.info(f"Cache memory config: {tt_cache.memory_config()}")
        logger.info(f"Cache shape: {tt_cache.shape}")
        logger.info(f"Expected layout: {expected_layout}")
        logger.info(f"Operation type: {op_type}")

        seq_len_local = seq_len // mesh_shape[sp_axis]

        if op_type == "fill":
            input_shape = [1, 1, seq_len_local, kvpe_cache_head_dim]
            x = torch.randn(input_shape).bfloat16()
            tt_x = ttnn.from_torch(x, dtype=input_dtype, device=device, layout=ttnn.TILE_LAYOUT)
            ttnn.fill_cache(tt_cache, tt_x, batch_idx=0)
            tt_got_back = tt_cache.to_torch().float()
            eq, output = comp_pcc(x.float(), tt_got_back[0, :, :, :])

        elif op_type == "fill_cache_for_user":
            user_idx = 1
            input_shape = [1, 1, seq_len_local, kvpe_cache_head_dim]
            user = torch.randn(input_shape).bfloat16()
            tt_user = ttnn.from_torch(user, dtype=input_dtype, device=device, layout=ttnn.TILE_LAYOUT)
            ttnn.kv_cache.fill_cache_for_user_(tt_cache, tt_user, user_idx)

            tt_got_back = tt_cache.to_torch().float()
            eq, output = comp_pcc(user.float(), tt_got_back[user_idx, :, :, :])
            eq_z, output_z = comp_pcc(torch.zeros_like(user).float(), tt_got_back[0, :, :, :])
            eq = eq and eq_z
            output = f"User match: {output}, Zero match: {output_z}"
        elif op_type == "update":
            batch_offset = 0
            update_idx = 2
            cache_entry = torch.randn([1, 1, 1, kvpe_cache_head_dim]).bfloat16()
            tt_cache_entry = ttnn.from_torch(cache_entry, dtype=input_dtype, device=device, layout=ttnn.TILE_LAYOUT)
            ttnn.update_cache(tt_cache, tt_cache_entry, update_idx=update_idx, batch_offset=batch_offset)

            tt_got_back = tt_cache.to_torch().float()
            eq, output = comp_pcc(cache_entry.float(), tt_got_back[batch_offset, :, update_idx, :])

        else:
            raise ValueError(f"Unexpected op_type value: {op_type}")

        logger.info(output)
        assert eq, output
