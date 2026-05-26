# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Variant E/F: paged_update_cache + paged_SDPA in front of axis-0 RS.

Earlier micro-benches isolated the FA WO 1280×BF16 cluster_axis=0 RS (slow at
666 µs/call mean in the model) and ruled out: bf16-vs-bf8, num_links, L1-shard,
chain-position, inter-axis CCL contention, back-to-back axis-0 contention.

The only major FA-path-specific ops remaining as candidates are:
  - paged_update_cache (×2 for K and V; KV heads sharded across chips)
  - paged_scaled_dot_product_attention_decode (1×1 program cfg, per-head work)

This test adds those between a prior matmul + axis-1 RS and the final axis-0 RS.

Variants:
  E: matmul → axis-1 RS → matmul → paged_update_cache → matmul → axis-0 RS
  F: matmul → axis-1 RS → matmul → paged_update_cache + paged_SDPA → matmul → axis-0 RS

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m tracy -p -v -r -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_rs_kv_chain_micro.py -v -s
"""
from __future__ import annotations

import pytest
import torch

import ttnn

_MESH = (8, 4)
_M = 32
_PER_CHIP_W_0 = 1280
_PER_CHIP_W_1 = 2048
_K = 768
_K1 = 1280
_HD = 256
_N_KV_PER_CHIP = 1
_N_Q_PER_CHIP = 3
_MAX_SEQ = 2048  # page_block_size 32 × max_num_blocks 64
_PAGE_BLOCK_SIZE = 32
_MAX_NUM_BLOCKS = 64
_N_WARM = 4
_CUR_POS = 128


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(*_MESH),
        trace_region_size=184915840,
        worker_l1_size=1345000,
    )
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _make_weight(mesh, k, n, seed=7):
    torch.manual_seed(seed)
    w = torch.randn(_MESH[0], _MESH[1], k, n, dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(
        w,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=_MESH),
    )


def _make_act(mesh, m, k, seed=99):
    torch.manual_seed(seed)
    a = torch.randn(_MESH[0], _MESH[1], m, k, dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(
        a,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=_MESH),
    )


def _make_kv_cache(mesh):
    """Per-chip paged KV cache [max_num_blocks=64, n_kv_per_chip=1,
    page_block_size=32, hd=256] BFLOAT16. Replicated across all chips.

    Paged KV layout (matches `ttnn.experimental.paged_update_cache` expectations):
    dim 0 = num_blocks, dim 1 = n_kv_heads, dim 2 = block_size, dim 3 = hd.
    """
    torch.manual_seed(43)
    cache = torch.randn(_MAX_NUM_BLOCKS, _N_KV_PER_CHIP, _PAGE_BLOCK_SIZE, _HD, dtype=torch.bfloat16) * 0.05
    return ttnn.from_torch(
        cache,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _make_page_table(mesh):
    """Page table per chip: max_batch × num_blocks, int32, col-sharded."""
    torch.manual_seed(44)
    permutation = torch.randperm(_MAX_NUM_BLOCKS)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(1, _MAX_NUM_BLOCKS)
    return ttnn.from_torch(
        page_table,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=_MESH),
    )


def _make_cur_pos(mesh, pos=_CUR_POS):
    """Replicated current position tensor."""
    t = torch.tensor([pos], dtype=torch.int32)
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _make_k_input(mesh):
    """K input to paged_update_cache: HEIGHT_SHARDED [tile=32, hd=256] on (1,1) core grid.
    Per chip: 1 n_kv_head × hd=256 entry packed in tile."""
    torch.manual_seed(55)
    k = torch.randn(_MESH[0], _MESH[1], 1, _N_KV_PER_CHIP, _M, _HD, dtype=torch.bfloat16) * 0.1
    k = k.reshape(_MESH[0], _MESH[1], _M, _HD)  # collapse to [M=32, hd]
    height_shard = ttnn.create_sharded_memory_config(
        shape=[_M, _HD],
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(
        k,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=height_shard,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=_MESH),
    )


@pytest.mark.hardware
def test_rs_with_paged_kv_chain(bh_glx_mesh):
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *a, **k: None  # noqa: E731

    mesh = bh_glx_mesh

    # Activations + weights
    a0 = _make_act(mesh, _M, _K, seed=11)
    w0 = _make_weight(mesh, _K, _PER_CHIP_W_0, seed=21)
    a1 = _make_act(mesh, _M, _K1, seed=12)
    w1 = _make_weight(mesh, _K1, _PER_CHIP_W_1, seed=22)

    # KV cache + page table for paged_update_cache + paged_SDPA
    k_cache = _make_kv_cache(mesh)
    v_cache = _make_kv_cache(mesh)
    page_table = _make_page_table(mesh)
    cur_pos = _make_cur_pos(mesh)
    k_input = _make_k_input(mesh)
    v_input = _make_k_input(mesh)

    def axis0_chain():
        out = ttnn.linear(a0, w0, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r = ttnn.all_reduce(out, cluster_axis=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        ttnn.deallocate(r)

    def axis1_chain():
        out = ttnn.linear(a1, w1, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r = ttnn.all_reduce(out, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        ttnn.deallocate(r)

    def variant_E():
        """matmul → axis-1 RS → matmul → paged_update_cache ×2 → matmul → axis-0 RS"""
        axis1_chain()
        ttnn.experimental.paged_update_cache(k_cache, k_input, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v_input, update_idxs_tensor=cur_pos, page_table=page_table)
        axis0_chain()

    def variant_F():
        """matmul → axis-1 RS → matmul → paged_update_cache ×2 → paged_SDPA → matmul → axis-0 RS"""
        axis1_chain()
        ttnn.experimental.paged_update_cache(k_cache, k_input, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v_input, update_idxs_tensor=cur_pos, page_table=page_table)
        # Q for SDPA: per-chip [1, B=1, n_q_per_chip=3, hd=256]. Build via matmul + permute.
        q_4d = ttnn.linear(
            a0,  # reusing a0 for simplicity
            w0,  # output (B*T=32, 1280) ... need to adapt
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Adapt q shape to [1, B=1, n_q_per_chip=3, hd=256] = [1, 1, 3, 256]
        # Use a synthetic q tensor instead — simpler than retrofitting matmul output.
        ttnn.deallocate(q_4d)
        # Build q_1bnd directly
        torch.manual_seed(77)
        q_host = torch.randn(_MESH[0], _MESH[1], 1, 1, _N_Q_PER_CHIP, _HD, dtype=torch.bfloat16) * 0.1
        q_host = q_host.reshape(_MESH[0], _MESH[1], 1, 1, _N_Q_PER_CHIP, _HD)
        q_1bnd = ttnn.from_torch(
            q_host[:, :, 0, ...],
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=_MESH),
        )
        paged_sdpa_prog_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(1, 1),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_1bnd,
            k_cache,
            v_cache,
            page_table,
            cur_pos_tensor=cur_pos,
            scale=0.125,
            program_config=paged_sdpa_prog_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_1bnd)
        ttnn.deallocate(attn_out)
        axis0_chain()

    # === Run variant E ===
    print("\n=== Variant E: matmul → axis-1 RS → matmul → paged_update_cache ×2 → matmul → axis-0 RS ===")
    variant_E()
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    signpost("E_warm_start")
    for _ in range(_N_WARM):
        variant_E()
    ttnn.synchronize_device(mesh)
    signpost("E_warm_done")
    print(f"  {_N_WARM} warm iters done")

    # === Run variant F ===
    print("\n=== Variant F: variant E + paged_SDPA ===")
    variant_F()
    ttnn.synchronize_device(mesh)
    signpost("F_warm_start")
    for _ in range(_N_WARM):
        variant_F()
    ttnn.synchronize_device(mesh)
    signpost("F_warm_done")
    print(f"  {_N_WARM} warm iters done")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
