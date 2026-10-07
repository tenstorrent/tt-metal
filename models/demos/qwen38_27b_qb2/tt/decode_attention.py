# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit decode SDPA policy; all preparation stays inside device traces."""

import ttnn


def paged_decode(query, key, value, *, positions, page_table, policy, page_size=32):
    batch, heads, width = tuple(query.shape)[1:]
    grid = query.device().compute_with_storage_grid_size()
    accurate = policy.get("decode_attention", "native") == "accurate_full_tile"
    short_cache = page_table.shape[-1] < 16
    chunk = (
        256
        if accurate
        else (32 if policy.get("adaptive_sdpa", False) and short_cache else policy.get("sdpa_k", 32) or 32)
    )
    while (page_table.shape[-1] * page_size) % chunk:
        chunk //= 2
    decode_grid = policy.get("sdpa_grid", [grid.x, grid.y])
    if batch == 1 and short_cache:
        decode_grid = policy.get("sdpa_short_grid", decode_grid)
    extra = {}
    if accurate:
        # The pinned partial-tile SDPA helper forces approximate exp regardless
        # of exp_approx_mode. Pad independent query heads to a full tile so that
        # accurate exponentiation is actually selected. TP4 has one KV head;
        # adding Q rows preserves that Q->KV mapping. Other TP layouts must be
        # qualified separately rather than silently changing their GQA groups.
        if heads != 6 or key.shape[1] != 1 or width != 256:
            raise ValueError("Accurate decode attention currently requires TP4 Qwen head geometry")
        query = ttnn.pad(query, [(0, 0), (0, 0), (0, 32 - heads), (0, 0)], 0.0)
        extra["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    result = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        query,
        key,
        value,
        cur_pos_tensor=positions,
        page_table_tensor=page_table,
        scale=width**-0.5,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=decode_grid,
            q_chunk_size=policy.get("sdpa_q", 32),
            k_chunk_size=chunk,
            exp_approx_mode=not accurate,
            max_cores_per_head_batch=16,
        ),
        **extra,
    )
    return result[:, :, :heads, :] if accurate else result
