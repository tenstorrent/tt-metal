# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
@pytest.mark.parametrize("max_seq_len", [4096])
@pytest.mark.parametrize("num_users", [8, 32])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
class TestFillCacheWithOffset:
    @pytest.mark.parametrize("seq_len", [32, 128])
    @pytest.mark.parametrize("update_idx", [32, 64, 512])
    def test_fill_cache_with_update_idx(
        self, seq_len, head_dim, max_seq_len, num_users, num_heads, in_sharded, input_dtype, update_idx, device
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
                num_cores = min(seq_len // 32 * num_heads, 32)
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

            cachett = ttnn.fill_cache(cachett, xt, i, update_idx=update_idx)
            cache[i : i + 1, :, update_idx : update_idx + x.shape[-2], :] = x

        tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
            eq, output = comp_equal(cache, tt_got_back)
        else:
            eq, output = comp_pcc(cache, tt_got_back)
        logger.info(output)
        assert eq

    @pytest.mark.parametrize("seq_len", [128])
    def test_fill_cache_continuation(
        self, seq_len, head_dim, max_seq_len, num_users, num_heads, in_sharded, input_dtype, device
    ):
        """Simulates continuation chat: fill at 0, then fill more at seq_len offset."""
        cache_dtype = input_dtype
        input_shape = [1, num_heads, seq_len, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]

        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)

        for i in range(num_users):
            x1 = torch.randn(input_shape).bfloat16().float()
            x2 = torch.randn(input_shape).bfloat16().float()

            def to_device(x):
                xt = ttnn.Tensor(x, input_dtype).to(ttnn.TILE_LAYOUT)
                if in_sharded:
                    compute_grid_size = device.compute_with_storage_grid_size()
                    num_cores = min(seq_len // 32 * num_heads, 32)
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
                    return xt.to(device, input_mem_config)
                return xt.to(device)

            cachett = ttnn.fill_cache(cachett, to_device(x1), i)
            cache[i : i + 1, :, :seq_len, :] = x1

            cachett = ttnn.fill_cache(cachett, to_device(x2), i, update_idx=seq_len)
            cache[i : i + 1, :, seq_len : 2 * seq_len, :] = x2

        tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
            eq, output = comp_equal(cache, tt_got_back)
        else:
            eq, output = comp_pcc(cache, tt_got_back)
        logger.info(output)
        assert eq


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("max_seq_len", [4096])
@pytest.mark.parametrize("num_users", [8, 32])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
class TestFillCacheSubTileOffset:
    """Tests fill_cache with non-tile-aligned update_idx (sub-tile offsets).
    Only interleaved input is supported for non-aligned offsets."""

    @pytest.mark.parametrize("seq_len", [32, 128])
    @pytest.mark.parametrize("update_idx", [1, 10, 16, 25, 31])
    def test_fill_cache_subtile_offset(
        self, seq_len, head_dim, max_seq_len, num_users, num_heads, input_dtype, update_idx, device
    ):
        cache_dtype = input_dtype
        input_shape = [1, num_heads, seq_len, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]

        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)

        for i in range(num_users):
            x = torch.randn(input_shape).bfloat16().float()
            xt = ttnn.Tensor(x, input_dtype).to(ttnn.TILE_LAYOUT).to(device)

            cachett = ttnn.fill_cache(cachett, xt, i, update_idx=update_idx)
            cache[i : i + 1, :, update_idx : update_idx + x.shape[-2], :] = x

        tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
            eq, output = comp_equal(cache, tt_got_back)
        else:
            eq, output = comp_pcc(cache, tt_got_back)
        logger.info(output)
        assert eq

    @pytest.mark.parametrize("seq_len", [128])
    @pytest.mark.parametrize("decode_tokens", [5, 17])
    def test_fill_cache_continuation_subtile(
        self, seq_len, head_dim, max_seq_len, num_users, num_heads, input_dtype, decode_tokens, device
    ):
        """Simulates continuation after decode: prefill at 0, then fill at seq_len + decode_tokens (non-aligned)."""
        cache_dtype = input_dtype
        input_shape = [1, num_heads, seq_len, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        continuation_offset = seq_len + decode_tokens

        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)

        for i in range(num_users):
            x1 = torch.randn(input_shape).bfloat16().float()
            x2 = torch.randn(input_shape).bfloat16().float()

            xt1 = ttnn.Tensor(x1, input_dtype).to(ttnn.TILE_LAYOUT).to(device)
            cachett = ttnn.fill_cache(cachett, xt1, i)
            cache[i : i + 1, :, :seq_len, :] = x1

            xt2 = ttnn.Tensor(x2, input_dtype).to(ttnn.TILE_LAYOUT).to(device)
            cachett = ttnn.fill_cache(cachett, xt2, i, update_idx=continuation_offset)
            cache[i : i + 1, :, continuation_offset : continuation_offset + seq_len, :] = x2

        tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
            eq, output = comp_equal(cache, tt_got_back)
        else:
            eq, output = comp_pcc(cache, tt_got_back)
        logger.info(output)
        assert eq


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
