# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Fast post-commit tests for SDPA decode with a representative subset of parametrizations.
"""

import random

import torch
import numpy as np
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    num_to_corerange,
    run_test_sdpa_decode_single_iter,
    run_test_sdpa_decode_multi_pos,
    run_test_sdpa_decode_paged_attention,
    run_test_sdpa_decode_broadcast_mask_batch,
)


@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)
    yield


def test_sdpa_decode_golden_preserves_layout_and_batch_positions():
    q = torch.randn(1, 2, 2, 4)
    k = torch.randn(2, 1, 5, 4)
    v = torch.randn(2, 1, 5, 4)
    cur_pos = torch.tensor([1, 3], dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn.transformer.scaled_dot_product_attention_decode)

    actual = golden_function(q, k, v, cur_pos_tensor=cur_pos)

    query = q.permute(1, 2, 0, 3)
    key = k.repeat_interleave(2, dim=1)
    value = v.repeat_interleave(2, dim=1)
    mask = torch.zeros(2, 1, 1, 5)
    mask[0, ..., 2:] = float("-inf")
    mask[1, ..., 4:] = float("-inf")
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask).permute(2, 0, 1, 3)

    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected)


def test_chunked_sdpa_golden_gathers_pages_and_offsets_causal_mask():
    q = torch.randn(1, 1, 2, 4)
    dense_k = torch.randn(1, 1, 4, 4)
    dense_v = torch.randn(1, 1, 4, 4)
    logical_k_pages = dense_k.reshape(1, 1, 2, 2, 4).permute(0, 2, 1, 3, 4).squeeze(0)
    logical_v_pages = dense_v.reshape(1, 1, 2, 2, 4).permute(0, 2, 1, 3, 4).squeeze(0)
    page_table = torch.tensor([[1, 0]], dtype=torch.int64)
    paged_k = logical_k_pages.flip(0)
    paged_v = logical_v_pages.flip(0)
    golden_function = ttnn.get_golden_function(ttnn.transformer.chunked_scaled_dot_product_attention)

    actual = golden_function(q, paged_k, paged_v, page_table, 2)

    mask = torch.zeros(2, 4)
    mask[0, 3] = float("-inf")
    expected = torch.nn.functional.scaled_dot_product_attention(q, dense_k, dense_v, attn_mask=mask)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, single_iter, cur_pos_tensor",
    (
        [8, 8, 1, 32768, 128, (8, 6), True, False],  # Llama2-70B
        [4, 32, 8, 8192, 128, (8, 8), True, True],  # llama 3.1 8b
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
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, cur_pos_tensor",
    ([2, 20, 20, 512, 64, (8, 8), True],),  # Whisper-large (nh not multiple of 32; grid must give num_cores/b <= nkv)
)
@pytest.mark.timeout(120)
def test_sdpa_decode_non_tile_aligned_heads(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor):
    """Regression test for models with num_heads not a multiple of 32 (e.g. Whisper-large with 20 heads).

    The output logical shape must preserve the unpadded head count so that downstream
    ops like nlp_concat_heads produce the correct hidden dimension.
    """
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
    )


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    ([1, 64, 8, 2048, 128, (8, 8)],),  # num q heads greater than 32
)
@pytest.mark.timeout(120)
def test_sdpa_decode_non_causal(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    for _ in range(2):
        run_test_sdpa_decode_single_iter(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=False, sharded_out=False, causal=False
        )
    assert device.cache_entries_counter.total == 1


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
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8_q_bf16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, cur_pos_tensor, sliding_window_size",
    ([8, 16, 4, 4096, 128, (8, 2), True, None],),  # llama 3.1 8b N300
    ids=["llama3.1-a"],
)
@pytest.mark.parametrize("block_size", (64,), ids=["paged_64"])
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
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    [
        [32, 32, 8, 2048, 128, (10, 11)],
    ],
    ids=["blackhole_b32_nkv8"],
)
@pytest.mark.timeout(120)
def test_sdpa_decode_kv_head_core_divisibility(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    """Regression test for github.com/tenstorrent/tt-metal/issues/40978.

    When floor(num_cores_available / B) yields a value that makes ceil(num_kv_heads / uncapped)
    a non-divisor of num_kv_heads, the old core allocation produced inconsistent counts causing
    a TT_FATAL crash at output core indexing.
    """
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor=True, sharded_in=False, sharded_out=False
    )


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
    ],
    ids=[
        "all_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    ([1, 8, 1, 32768, 128, (8, 8)],),
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


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat8_b],
    ids=["bfp8"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([16, 8, 1, 8192, 128],),  # Llama2-70B
)
def test_sdpa_decode_program_cache(device, b, nh, nkv, s, d, dtype, reset_seeds):
    dummy_tensors = []
    # One cur_pos vector for both outer passes: compute_program_hash includes cur_pos, so resampling
    # per iteration would compile extra programs for cur_pos_tensor=False paths (expected cache size 4).
    start_indices = np.random.randint(0, s - 1, b).tolist()
    start_indices[0] = s - 1
    for i in range(2):
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


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8_q_bf16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, sliding_window_size",
    [
        [4, 8, 1, 1024, 128, (8, 4), 128],  # Medium window
    ],
)
@pytest.mark.parametrize("cur_pos_tensor", [True], ids=["cur_pos_tensor"])
@pytest.mark.timeout(120)
def test_sdpa_decode_sliding_window(
    device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sliding_window_size, cur_pos_tensor
):
    """Test sliding window attention functionality."""

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


@pytest.mark.parametrize("mask_dtype", [ttnn.bfloat16, ttnn.bfloat4_b])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, dtype, grid_size",
    [
        (32, 8, 1, 2048, 128, ttnn.bfloat8_b, (8, 8)),
    ],
)
def test_sdpa_decode_broadcast_mask_batch(device, b, nh, nkv, s, d, dtype, grid_size, mask_dtype):
    run_test_sdpa_decode_broadcast_mask_batch(
        device,
        b=b,
        nh=nh,
        nkv=nkv,
        s=s,
        d=d,
        dtype=dtype,
        grid_size=grid_size,
        mask_dtype=mask_dtype,
    )
