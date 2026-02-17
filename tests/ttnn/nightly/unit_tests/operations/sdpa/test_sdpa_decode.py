# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    run_test_sdpa_decode_multi_pos,
    run_test_sdpa_decode_single_iter,
    run_test_sdpa_decode_paged_attention,
    run_test_sdpa_decode_ndpcc,
    num_to_corerange,
)


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.bfloat8_b, ttnn.bfloat16],
        [ttnn.bfloat4_b, ttnn.bfloat16],
    ],
    ids=[
        "all_bfp8",
        "all_bfp16",
        "kv_bfp8",
        "kv_bfp4",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, single_iter, cur_pos_tensor",
    (
        # [32, 8, 1, 32768, 128, (8, 6), True, True],  # Llama2-70B
        # [16, 8, 1, 32768, 128, (8, 6), False, False],  # Llama2-70B
        [8, 8, 1, 32768, 128, (8, 6), True, False],  # Llama2-70B
        # [4, 8, 1, 32768, 128, (8, 6), True, False],  # Llama2-70B
        [32, 8, 1, 8192, 128, (8, 8), True, True],  # Mixtral8x7b
        # [32, 8, 1, 32768, 128, (8, 6), True, False],  # Llama2-70B
        # [4, 32, 8, 32768, 128, (8, 8), True, False],  # llama 3.1 8b
        [4, 32, 8, 8192, 128, (8, 8), True, True],  # llama 3.1 8b
        [32, 32, 8, 8192, 128, (8, 8), True, False],  # llama 3.1 8b
        # [4, 16, 4, 32768, 128, (8, 8), False, False],  # llama 3.1 8b
        # [1, 8, 1, 8192*16, 128, (1, 1), False, True],  # llama2-70B long seqlen
    ),
)
@pytest.mark.timeout(120)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, single_iter, cur_pos_tensor):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    if single_iter:
        run_test_sdpa_decode_single_iter(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
        )
    else:
        run_test_sdpa_decode_multi_pos(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
        )


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        # [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        # "all_bfp16",
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    (
        [32, 32, 8, 4224, 128, (8, 8)],  # llama3.2 vision encoder on n150
        [8, 16, 4, 4224, 128, (8, 8)],  # llama3.2 vision encoder on n300
        [32, 4, 1, 4224, 128, (8, 8)],  # llama3.2 vision encoder on n300
        [1, 64, 8, 2048, 128, (8, 8)],  # num q heads greater than 32
    ),
)
@pytest.mark.timeout(120)
def test_sdpa_decode_non_causal(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    for _ in range(2):
        run_test_sdpa_decode_single_iter(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=False, sharded_out=False, causal=False
        )
    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat16, ttnn.bfloat16],
    ],
    ids=[
        "all_bfp16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, single_iter, cur_pos_tensor",
    ([32, 8, 1, 32768, 128, (8, 6), True, True],),  # Llama2-70B
)
def test_sdpa_decode_ignore_users(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, single_iter, cur_pos_tensor):
    # Set odd users to -1 to test skipping users
    start_indices = [100 if bb % 2 == 0 else -1 for bb in range(b)]

    run_test_sdpa_decode_single_iter(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        dtype,
        grid_size,
        q_dtype,
        cur_pos_tensor,
        sharded_in=False,
        sharded_out=False,
        start_indices=start_indices,
    )


@pytest.mark.parametrize(
    "kv_dtype, q_dtype",
    [
        # [ttnn.bfloat8_b, ttnn.bfloat8_b],
        # [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.bfloat8_b, ttnn.bfloat16],
        # [ttnn.bfloat4_b, ttnn.bfloat16],
    ],
    ids=[
        # "all_bfp8",
        # "all_bfp16",
        "kv_bfp8_q_bf16",
        # "kv_bfp4",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, cur_pos_tensor, sliding_window_size",
    (
        # [32, 8, 1, 32768, 128, (8, 6), True, None],  # Llama2-70B
        # [4, 32, 8, 4096, 128, (8, 8), True, None],  # llama 3.1 8b
        # [4, 16, 4, 32768, 128, (8, 8), True, None],
        # [32, 32, 8, 4096, 128, (8, 8), True, None],  # llama 3.1 8b
        [8, 16, 4, 4096, 128, (8, 2), True, None],  # llama 3.1 8b N300
        [1, 8, 1, 128 * 1024, 128, (8, 4), True, None],  # llama 3.1 8b N300
        [1, 32, 8, 32 * 1024, 128, (8, 1), True, None],  # llama3.1 8b (performance-batch-1 settings)
        [1, 4, 2, 1024 * 128, 128, (8, 8), True, 1024],  # gemma-3-27b on T3K
        [1, 8, 1, 1024 * 128, 64, (8, 8), True, 128],  # GPT-OSS
        # [32, 32, 8, 1024, 128, (8, 8), True, None],  # llama 3.1 8b (performance-batch-32 settings) -- Issue 21534: Breaking blackhole post commit tests
        # [1, 8, 1, 32768, 128, (8, 1), True, None],  # Llama2-70B
        # [16, 8, 1, 32768, 128, (8, 6), False, False, None],  # Llama2-70B
        # [8, 8, 1, 32768, 128, (8, 6), True, False, None],  # Llama2-70B
        # [4, 8, 1, 32768, 128, (8, 6), True, False, None],  # Llama2-70B
        # [32, 8, 1, 32768, 128, (8, 8), True, True, None],  # Mixtral8x7b
    ),
    ids=["llama3.1-a", "llama3.1-b", "llama3.1-c", "gemma-3-27b", "GPT-OSS"],
)
@pytest.mark.parametrize("block_size", (32, 64, 128), ids=["paged_32", "paged_64", "paged_128"])
def test_sdpa_decode_paged_attention(
    device, b, nh, nkv, s, d, kv_dtype, grid_size, q_dtype, cur_pos_tensor, sliding_window_size, block_size, reset_seeds
):
    if s == 128 * 1024 and block_size != 64:
        # 128k sequence, block_size 64 tests the sizing of the page table CB
        pytest.skip("Skipping test for seq_len=128k with block_size!=64")
    run_test_sdpa_decode_paged_attention(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        kv_dtype,
        grid_size,
        q_dtype,
        cur_pos_tensor,
        block_size=block_size,
        sharded_in=True,
        sharded_out=False,
        sliding_window_size=sliding_window_size,
    )

    assert device.num_program_cache_entries() == 4


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.bfloat16, ttnn.bfloat16],
    ],
    ids=[
        "all_bfp8",
        "all_bfp16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    (
        [1, 8, 1, 32768, 128, (8, 8)],
        [16, 8, 1, 32768, 128, (8, 8)],
    ),  # Llama2-70B
)
def test_sdpa_decode_sharded(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=True, sharded_out=False
    )
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=True, sharded_out=True
    )
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=False, sharded_out=True
    )


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "bfp8_cache_bf16_act",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    (
        [8, 8, 1, 2048, 128, (8, 4)],
        [8, 8, 1, 256, 128, (8, 4)],
    ),  # Llama2-70B
)
@pytest.mark.parametrize(
    "start_core, sub_core_grids",
    [
        (
            ttnn.CoreCoord(1, 0),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                ]
            ),
        ),
    ],
)
def test_sdpa_decode_sharded_on_subcoregrids(
    device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, start_core, sub_core_grids
):
    run_test_sdpa_decode_single_iter(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        dtype,
        grid_size,
        q_dtype,
        sharded_in=True,
        sharded_out=True,
        start_core=start_core,
        sub_core_grids=sub_core_grids,
    )
    run_test_sdpa_decode_single_iter(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        dtype,
        grid_size,
        q_dtype,
        sharded_in=True,
        sharded_out=True,
        start_core=start_core,
        sub_core_grids=sub_core_grids,
    )
    assert device.num_program_cache_entries() == 1


@pytest.mark.skip("Skipping Perf Test in CI")
def test_sdpa_decode_perf(device):
    dtype = ttnn.bfloat8_b
    q_dtype = ttnn.bfloat16
    nh = 8
    nkv = 1
    d = 128
    grid_size = (8, 8)

    bs_combs = [
        (32, 2048),
        (16, 2048),
        (32, 8192),
        (16, 8192),
        (8, 8192),
        (16, 8192 * 2),
        (8, 8192 * 2),
        (16, 8192 * 4),
        (8, 8192 * 4),
        (4, 8192 * 4),
    ]

    for b, s in bs_combs:
        run_test_sdpa_decode_single_iter(  # different user pos
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            grid_size,
            q_dtype,
            sharded_in=True,
            sharded_out=True,
            start_indices=np.linspace(0, s - 1, b, dtype=np.int32).tolist(),
        )
        run_test_sdpa_decode_single_iter(  # all same user pos
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            grid_size,
            q_dtype,
            sharded_in=True,
            sharded_out=True,
            start_indices=[s - 1 for _ in range(b)],
        )


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
    ids=["bfp8", "bf16"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([16, 8, 1, 8192, 128],),  # Llama2-70B
)
def test_sdpa_decode_program_cache(device, b, nh, nkv, s, d, dtype):
    import torch

    dummy_tensors = []
    for i in range(2):
        # generate random start indices from 0 to s-1
        start_indices = np.random.randint(0, s - 1, b).tolist()
        start_indices[0] = s - 1

        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(32, 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(1, 1, 32, 32 * 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({num_to_corerange(32)}),
                        (32, 32),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=False,
            sharded_out=False,
            start_indices=start_indices,
            cur_pos_tensor=True,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=True,
            sharded_out=False,
            start_indices=start_indices,
            cur_pos_tensor=False,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=True,
            sharded_out=True,
            start_indices=start_indices,
            cur_pos_tensor=False,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=False,
            sharded_out=True,
            start_indices=start_indices,
            cur_pos_tensor=True,
        )

    assert device.num_program_cache_entries() == 4


@pytest.mark.timeout(600)
@pytest.mark.skip("Skipping due to causing 45 minutes timeout on tt eager unit tests")
@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.bfloat4_b, ttnn.bfloat4_b],
    ],
    ids=[
        "bfp16_bfp16",
        "bfp8_bfp8",
        "bfp4_bfp4",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    (
        # [16, 8, 1, 32768, 128, (8, 6)],  # Llama2-70B
        # [32, 8, 1, 32768, 128, (8, 8)],  # Llama2-70B
        [16, 8, 1, 32768, 128, (8, 6)],
    ),
)
def test_sdpa_decode_ndpcc(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    run_test_sdpa_decode_ndpcc(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype)


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "all_bfp16",
        "kv_bfp8_q_bf16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, sliding_window_size",
    [
        # Test different sliding window sizes
        [1, 4, 2, 1024 * 16, 128, (8, 8), 1024],  # Gemma test
        [1, 8, 1, 1024 * 16, 128, (8, 8), 128],  # GPT-OSS test
        [4, 8, 1, 1024, 128, (8, 4), 64],  # Small window
        [4, 8, 1, 1024, 128, (8, 4), 128],  # Medium window
        [4, 8, 1, 1024, 128, (8, 4), 256],  # Large window
        [8, 16, 4, 2048, 128, (8, 8), 128],  # Multi-head with window
        [1, 8, 1, 4096, 128, (8, 4), 512],  # Long sequence with window
    ],
)
@pytest.mark.parametrize("cur_pos_tensor", [False, True])
@pytest.mark.timeout(120)
def test_sdpa_decode_sliding_window(
    device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sliding_window_size, cur_pos_tensor
):
    """Test sliding window attention functionality."""
    from loguru import logger

    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    # Ensure sliding window is smaller than sequence length
    if sliding_window_size >= s:
        pytest.skip(f"Sliding window {sliding_window_size} must be smaller than sequence length {s}")

    # Test different window start positions to ensure all fill tile code paths are hit
    # when generating the sliding window
    k_values = [5, 10, 20, 30]

    test_positions = [
        *[
            (sliding_window_size + offset - 1) + 32 * k
            for k in k_values
            for offset in (15, 16, 17)
            # in first face, in second face (1st face completely filled), in second face (1st face completely filled + 2nd face partially filled)
        ],
        sliding_window_size * 2,
        sliding_window_size // 2,
        sliding_window_size - 1,
        s // 2,
        s - 10,
    ]
    for cur_pos in test_positions:
        if cur_pos >= s:
            continue

        logger.info(f"Testing sliding window={sliding_window_size} at position {cur_pos}")

        # Test both cur_pos and cur_pos_tensor modes
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            grid_size,
            q_dtype,
            cur_pos_tensor=cur_pos_tensor,
            sharded_in=False,
            sharded_out=False,
            start_indices=[cur_pos + i for i in range(b)],  # test a batch with different start positions
            sliding_window_size=sliding_window_size,
        )
