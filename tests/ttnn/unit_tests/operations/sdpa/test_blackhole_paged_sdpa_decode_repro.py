# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


pytestmark = pytest.mark.use_module_device


@pytest.mark.timeout(120)
def test_blackhole_paged_sdpa_decode_repro(device):
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("This repro is intended for Blackhole devices.")

    torch.manual_seed(0)

    # Match the failing TTNN op shape captured from ttnn_graph.mlir.
    num_users = 1
    total_num_blocks = 893
    max_num_blocks_per_seq = 4
    num_query_heads = 32
    num_kv_heads = 4
    block_size = 32
    head_dim = 64

    query = torch.randn(1, num_users, num_query_heads, head_dim, dtype=torch.bfloat16)
    key = torch.randn(total_num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16)
    value = torch.randn(total_num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16)
    page_table = torch.arange(max_num_blocks_per_seq, dtype=torch.int32).reshape(num_users, max_num_blocks_per_seq)
    cur_pos_tensor = torch.ones(num_users, dtype=torch.int32)

    tt_query = ttnn.from_torch(
        query,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_key = ttnn.from_torch(
        key,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_value = ttnn.from_torch(
        value,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_page_table = ttnn.from_torch(
        page_table,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_cur_pos = ttnn.from_torch(
        cur_pos_tensor,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Intentionally omit program_config so the default approximate exponential
    # path is used.
    output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_query,
        tt_key,
        tt_value,
        tt_page_table,
        is_causal=True,
        cur_pos_tensor=tt_cur_pos,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_torch = ttnn.to_torch(output)
    assert tuple(output_torch.shape) == (1, 1, 32, 64)
