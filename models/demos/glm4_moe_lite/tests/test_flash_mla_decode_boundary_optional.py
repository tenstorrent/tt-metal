# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _nearest_n(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def _create_paged_kv_cache(
    *,
    num_users: int,
    max_seq_len: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    page_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (paged_cache, dense_cache) consistent with `page_table`."""
    seq_len_per_user = (num_blocks * block_size) // num_users
    assert seq_len_per_user == max_seq_len, (seq_len_per_user, max_seq_len)

    dense_cache = torch.randn((num_users, 1, seq_len_per_user, head_dim), dtype=torch.bfloat16) * 0.1

    # Reshape dense -> paged physical layout, then permute physical blocks to match page_table mapping.
    paged_cache = dense_cache.reshape(num_users, 1, -1, block_size, head_dim)
    paged_cache = paged_cache.transpose(1, 2)
    paged_cache = paged_cache.reshape(num_blocks, 1, block_size, head_dim)
    inverse_mapping = torch.argsort(page_table.view(-1))
    paged_cache = paged_cache[inverse_mapping]
    return paged_cache, dense_cache


def _sdpa_reference_mla(
    *,
    q: torch.Tensor,  # [1, B, H, DH]
    kv: torch.Tensor,  # [B, 1, S, DH]
    cur_pos: torch.Tensor,  # [B]
    head_dim_v: int,
    k_chunk_size: int,
) -> torch.Tensor:
    """PyTorch reference for MLA decode: V is the first `head_dim_v` dims of KV."""
    b = int(q.shape[1])
    num_heads = int(q.shape[2])
    head_dim = int(q.shape[3])
    assert list(kv.shape[:2]) == [b, 1], kv.shape
    assert int(kv.shape[3]) == head_dim, (kv.shape, head_dim)
    assert int(cur_pos.shape[0]) == b, (cur_pos.shape, b)

    # Kernel processes nearest_n(cur_pos+1, k_chunk_size) tokens.
    padded_layer_len = _nearest_n(int(cur_pos.max().item()) + 1, k_chunk_size)

    q_ref = q.permute(1, 2, 0, 3)  # [B, H, 1, DH]
    k_ref = kv[:, :, :padded_layer_len, :]  # [B, 1, S, DH]
    v_ref = kv[:, :, :padded_layer_len, :head_dim_v]  # [B, 1, S, DV]

    # Expand GQA KV heads to match Q heads (nkv==1).
    k_ref = k_ref.repeat(1, num_heads, 1, 1)
    v_ref = v_ref.repeat(1, num_heads, 1, 1)

    attn_mask = torch.zeros((b, num_heads, 1, padded_layer_len), dtype=torch.float32)
    for i in range(b):
        pos = int(cur_pos[i].item())
        attn_mask[i, :, :, pos + 1 :] = torch.finfo(torch.float32).min

    out = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask, is_causal=False)
    out = out.permute(2, 0, 1, 3)  # [1, B, H, DV]
    return out


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["cache_bf16", "cache_bf8"])
def test_paged_flash_mla_decode_boundary_matches_reference(cache_dtype: ttnn.DataType) -> None:
    """Regression: decode output should remain correct across the first k_chunk boundary (64 tokens)."""
    torch.manual_seed(0)

    # Keep shapes small, but aligned to TT tile requirements (heads multiple of 32).
    num_users = 1
    num_heads = 32
    # Match GLM-4.7 MLA shapes: KVPE dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576.
    head_dim = 576
    head_dim_v = 512
    block_size = 64
    num_blocks = 4
    max_seq_len = (num_blocks * block_size) // num_users
    k_chunk_size = 64

    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(num_users, num_blocks)
    paged_cache, dense_cache = _create_paged_kv_cache(
        num_users=num_users,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        page_table=page_table,
    )

    q = torch.randn((1, num_users, num_heads, head_dim), dtype=torch.bfloat16) * 0.1

    device = ttnn.open_device(
        device_id=0,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        tt_q = ttnn.from_torch(
            q,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_kv = ttnn.from_torch(
            paged_cache,
            dtype=cache_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_page_table = ttnn.from_torch(
            page_table,
            dtype=ttnn.int32,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        grid_size = device.compute_with_storage_grid_size()
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=0,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        for pos in (63, 64):
            cur_pos = torch.tensor([pos], dtype=torch.int32)
            tt_cur_pos = ttnn.from_torch(
                cur_pos,
                dtype=ttnn.int32,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                tt_q,
                tt_kv,
                page_table_tensor=tt_page_table,
                cur_pos_tensor=tt_cur_pos,
                head_dim_v=head_dim_v,
                program_config=sdpa_program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.synchronize_device(device)
            tt_out_torch = ttnn.to_torch(ttnn.from_device(tt_out))

            ref = _sdpa_reference_mla(
                q=q,
                kv=dense_cache,
                cur_pos=cur_pos,
                head_dim_v=head_dim_v,
                k_chunk_size=k_chunk_size,
            )

            expected_pcc = 0.98 if cache_dtype == ttnn.bfloat16 else 0.95
            ok, msg = comp_pcc(tt_out_torch, ref, pcc=expected_pcc)
            assert ok, f"MLA paged decode mismatch at cur_pos={pos}: {msg}"
    finally:
        ttnn.close_device(device)


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["cache_bf16", "cache_bf8"])
def test_paged_update_cache_block_boundary_updates_correct_block(cache_dtype: ttnn.DataType) -> None:
    """Regression: paged_update_cache must correctly write the first token of a new KV block (idx == block_size)."""
    torch.manual_seed(0)

    batch = 1
    num_heads = 1  # KV heads
    block_size = 64
    head_dim = 576  # kv_lora_rank + rope_dim (GLM-4.7 KVPE dim)
    max_seq_len = 128  # 2 blocks
    blocks_per_seq = max_seq_len // block_size
    num_blocks = batch * blocks_per_seq

    # Identity mapping: virtual block i -> physical block i.
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(batch, blocks_per_seq)

    device = ttnn.open_device(
        device_id=0,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        cache_host = torch.zeros((num_blocks, num_heads, block_size, head_dim), dtype=torch.bfloat16)
        cache_tt = ttnn.from_torch(
            cache_host,
            device=device,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Update tensor follows GLM layout: [1, 1, B, head_dim] then padded/permuted to [1, B, 32, head_dim].
        kvpe_new = torch.randn((1, 1, batch, head_dim), dtype=torch.bfloat16) * 0.1
        kvpe_new_tt = ttnn.from_torch(
            kvpe_new,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kvpe_padded_view = ttnn.pad(kvpe_new_tt, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)
        kvpe_padded = ttnn.clone(kvpe_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kvpe_perm_view = ttnn.permute(kvpe_padded, (0, 2, 1, 3))  # [1, B, 32, head_dim]
        kvpe_perm = ttnn.clone(kvpe_perm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(kvpe_padded, force=False)

        grid_size = device.compute_with_storage_grid_size()
        user_grid = ttnn.num_cores_to_corerangeset(int(batch), grid_size, row_wise=True)
        sharded_cfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, int(head_dim)),
            core_grid=user_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        kvpe_sharded_view = ttnn.to_memory_config(kvpe_perm, sharded_cfg)
        kvpe_sharded = ttnn.clone(kvpe_sharded_view, memory_config=sharded_cfg)
        ttnn.deallocate(kvpe_perm, force=False)

        for update_idx in (63, 64):
            update_idxs = torch.tensor([update_idx], dtype=torch.int32)
            update_idxs_tt = ttnn.from_torch(
                update_idxs,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            ttnn.experimental.paged_update_cache(
                cache_tt,
                kvpe_sharded,
                update_idxs_tensor=update_idxs_tt,
                page_table=page_table_tt,
            )
            ttnn.synchronize_device(device)

            cache_back = ttnn.to_torch(ttnn.from_device(cache_tt))
            # cache_back: [num_blocks, num_heads, block_size, head_dim]
            block = update_idx // block_size
            offset = update_idx % block_size
            got = cache_back[block, 0, offset : offset + 1, :]
            want = kvpe_new[:, 0, 0:1, :]
            ok, msg = comp_pcc(got, want, pcc=0.98)
            assert ok, f"paged_update_cache mismatch at update_idx={update_idx} ({cache_dtype}): {msg}"
            ttnn.deallocate(update_idxs_tt, force=False)

        ttnn.deallocate(kvpe_sharded, force=False)
        ttnn.deallocate(kvpe_new_tt, force=False)
        ttnn.deallocate(page_table_tt, force=False)
        ttnn.deallocate(cache_tt, force=False)
    finally:
        ttnn.close_device(device)
