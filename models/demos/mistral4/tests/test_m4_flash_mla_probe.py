# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A6 feasibility probe: does ttnn.transformer.paged_flash_multi_latent_attention_decode run on
mistral4's MLA dims (kv_lora=256, rope=64, v=128, nh=32 -> head_dim=320) + HEIGHT_SHARDED layout on
the standard 1x8 mesh? Random data, runs/finite check — gates the full compressed-latent decode build.
"""
import pytest
import torch

import ttnn

B, NH, KVL, ROPE = 4, 32, 256, 64
HEAD_DIM = KVL + ROPE  # 320
MAX_SEQ, BLOCK = 1024, 32
NUM_BLOCKS = MAX_SEQ * B // BLOCK


def _repl(t, mesh, layout=ttnn.TILE_LAYOUT, dt=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        dtype=dt,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_flash_mla_probe(mesh_device):
    dev = mesh_device
    torch.manual_seed(0)
    q = torch.randn(1, B, NH, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    page_table = torch.randperm(NUM_BLOCKS, dtype=torch.int32).reshape(B, NUM_BLOCKS // B)
    cache = torch.randn(NUM_BLOCKS, 1, BLOCK, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    pos = torch.arange(B, dtype=torch.int32) * 8 + 16

    tt_pt = _repl(page_table, dev, layout=ttnn.ROW_MAJOR_LAYOUT, dt=ttnn.int32)
    tt_cache = _repl(cache, dev)
    tt_pos = _repl(pos, dev, layout=ttnn.ROW_MAJOR_LAYOUT, dt=ttnn.int32)

    tt_q = _repl(q, dev)
    grid = dev.compute_with_storage_grid_size()
    rows_per_core = 32  # HEIGHT_SHARDED shard height must be tile-aligned (32)
    num_cores = max(1, (B * NH) // rows_per_core)  # 128 rows / 32 = 4 cores
    cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    q_mem = ttnn.create_sharded_memory_config(
        shape=[rows_per_core, HEAD_DIM],
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_q = ttnn.to_memory_config(tt_q, q_mem)
    out_mem = ttnn.create_sharded_memory_config(
        shape=[rows_per_core, KVL],
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
    )
    ckcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
        tt_q,
        tt_cache,
        page_table_tensor=tt_pt,
        cur_pos_tensor=tt_pos,
        head_dim_v=KVL,
        scale=128**-0.5,
        program_config=prog,
        compute_kernel_config=ckcfg,
        memory_config=out_mem,
    )
    t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1]
    assert torch.isfinite(t).all(), "flash-MLA op produced non-finite output"
    print(f"flash-MLA op OK on mistral4 dims: out {list(t.shape)} finite (rows_per_core={rows_per_core})")
