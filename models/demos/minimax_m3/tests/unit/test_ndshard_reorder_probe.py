# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe: localize the chunked cache-read reorder scramble to an exact permutation.

Write a POSITION RAMP (token t -> value t, fp32 so it is exact) into the REAL cache via the real
``update_padded_kv_cache`` write path, then read it back two ways on the same cache:
  (A) host ``gather_layer``-style: per-shard ``to_torch`` (delinearizes the slab) + ``blockcyclic_positions``
      scatter -> must be identity natural order [0,1,...,T-1].
  (B) on-device ``gather_natural``: ``to_memory_config(DRAM)`` + AllGather(SP) + ``_blockcyclic_to_natural``.
Comparing (B) to identity reveals the exact permutation the on-device path gets wrong, and whether it is a
regular reshape/transpose we can undo on-device (no host round-trip).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import BH_NUM_DRAM_BANKS, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
from models.demos.minimax_m3.config import MeshConfig, ModeConfig
from models.demos.minimax_m3.tt.attention.kv_cache import allocate_kv_caches, write_index_k_chunk
from models.demos.minimax_m3.tt.attention.msa import _blockcyclic_to_natural
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

HEAD_DIM = 128


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("chunk_local,n_prior", [(640, 1)], ids=["chunk640_prior1"])
def test_ndshard_reorder_permutation(mesh_device, device_params, chunk_local, n_prior, reset_seeds):
    rows, cols = mesh_device.shape
    sp, sp_axis = rows, 0
    chunk = sp * chunk_local
    n_chunks = n_prior + 1
    T = n_chunks * chunk
    seq_local = n_chunks * chunk_local
    logger.info(
        f"probe: T={T} sp={sp} chunk={chunk} chunk_local={chunk_local} n_chunks={n_chunks} "
        f"seq_local={seq_local} banks={BH_NUM_DRAM_BANKS} shard_tok={NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}"
    )

    mesh_config = MeshConfig((rows, cols), decode=ModeConfig(tp=cols, ep=sp))
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    # natural ramp: token t -> value t in all channels (fp32 exact)
    ramp = torch.arange(T, dtype=torch.float32).reshape(1, 1, T, 1).expand(1, 1, T, HEAD_DIM).contiguous()

    def shard(t):  # index_k: replicate heads on cols, contiguous SP split on seq (dim2)
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=[2, None]),
        )

    kv = allocate_kv_caches(
        mesh_device, num_layers=1, max_seq_len=T, sp_axis=sp_axis, head_dim=HEAD_DIM, cache_dtype=ttnn.float32
    )
    for c in range(n_chunks):
        sl = slice(c * chunk, (c + 1) * chunk)
        write_index_k_chunk(kv, shard(ramp[:, :, sl, :]), slot_idx=0, layer_idx=0, kv_actual=c * chunk, sp_axis=sp_axis)
    ik = ttnn.slice(kv.index_k, (0, 0, 0, 0), (1, 1, seq_local, HEAD_DIM))

    # (A) host gather_layer-style read -> expect identity natural order
    p = blockcyclic_positions(sp, chunk, T)
    dts = ttnn.get_device_tensors(kv.index_k)
    dev_bc = torch.cat([ttnn.to_torch(dts[r * cols + 0])[0, 0].float() for r in range(sp)], dim=0)  # [T, hd] blk-cyclic
    natA = torch.empty_like(dev_bc)
    natA[p] = dev_bc
    obsA = natA[:, 0].round().to(torch.long)
    logger.info(f"[A host] identity={bool((obsA == torch.arange(T)).all())} n_wrong={int((obsA != torch.arange(T)).sum())}")

    # Hypothesis probes: does the SLICE (vs full tensor) or a clone/copy change to_memory_config correctness?
    def dev0_seq(t):  # device-0 seq column (block-cyclic) after a device read
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])[0, 0, :seq_local, 0].round().to(torch.long)

    # per-device expected block-cyclic (device 0 holds chip-0 rows): p restricted to rows < seq_local
    dev0_bc_expected = p[:seq_local]
    full_tt = ttnn.to_torch(dts[0])[0, 0, :, 0].round().to(torch.long)
    logger.info(f"[to_torch full dev0] n_wrong-vs-bc={int((full_tt != dev0_bc_expected).sum())}/{seq_local}")

    memcfg_full = ttnn.to_memory_config(kv.index_k, ttnn.DRAM_MEMORY_CONFIG)
    mf = ttnn.to_torch(ttnn.get_device_tensors(memcfg_full)[0])[0, 0, :seq_local, 0].round().to(torch.long)
    logger.info(f"[to_memcfg FULL dev0] n_wrong-vs-bc={int((mf != dev0_bc_expected).sum())}/{seq_local}  first16={mf[:16].tolist()}")

    memcfg_slice = ttnn.to_memory_config(ik, ttnn.DRAM_MEMORY_CONFIG)
    ms = ttnn.to_torch(ttnn.get_device_tensors(memcfg_slice)[0])[0, 0, :seq_local, 0].round().to(torch.long)
    logger.info(f"[to_memcfg SLICE dev0] n_wrong-vs-bc={int((ms != dev0_bc_expected).sum())}/{seq_local}  first16={ms[:16].tolist()}")
    logger.info(f"[expected bc dev0] first16={dev0_bc_expected[:16].tolist()}")

    # ===== FIX CANDIDATE: to_memory_config(FULL) FIRST, then slice, then AllGather + reorder =====
    full_int = ttnn.to_memory_config(kv.index_k, ttnn.DRAM_MEMORY_CONFIG)  # correct (round-robin intact)
    ik_fix = ttnn.slice(full_int, (0, 0, 0, 0), (1, 1, seq_local, HEAD_DIM))  # slice the INTERLEAVED result
    full_bc_fix = mesh_config.allgather(ik_fix, ccl, axis=sp_axis, dim=2)
    nat_fix = _blockcyclic_to_natural(full_bc_fix, sp, n_chunks, chunk_local)
    obs_fix = ttnn.to_torch(ttnn.get_device_tensors(nat_fix)[0])[0, 0, :, 0].round().to(torch.long)
    logger.info(
        f"[FIX full->slice->AG->reorder] identity={bool((obs_fix == torch.arange(T)).all())} "
        f"n_wrong={int((obs_fix != torch.arange(T)).sum())}/{T}  first16={obs_fix[:16].tolist()}"
    )

    # outer-dim slot slice sanity: num_layers=2, write ramp to slot 1, convert-full then slice slot 1.
    kv2 = allocate_kv_caches(
        mesh_device, num_layers=2, max_seq_len=T, sp_axis=sp_axis, head_dim=HEAD_DIM, cache_dtype=ttnn.float32
    )
    for c in range(n_chunks):
        sl = slice(c * chunk, (c + 1) * chunk)
        write_index_k_chunk(kv2, shard(ramp[:, :, sl, :]), slot_idx=0, layer_idx=1, kv_actual=c * chunk, sp_axis=sp_axis)
    fi2 = ttnn.to_memory_config(kv2.index_k, ttnn.DRAM_MEMORY_CONFIG)
    ik2 = ttnn.slice(fi2, (1, 0, 0, 0), (2, 1, seq_local, HEAD_DIM))  # slot 1
    bc2 = mesh_config.allgather(ik2, ccl, axis=sp_axis, dim=2)
    nat2 = _blockcyclic_to_natural(bc2, sp, n_chunks, chunk_local)
    obs2 = ttnn.to_torch(ttnn.get_device_tensors(nat2)[0])[0, 0, :, 0].round().to(torch.long)
    logger.info(
        f"[FIX slot1 (num_layers=2)] identity={bool((obs2 == torch.arange(T)).all())} "
        f"n_wrong={int((obs2 != torch.arange(T)).sum())}/{T}"
    )

    # (B) on-device gather_natural read -> compare to identity
    t_int = ttnn.to_memory_config(ik, ttnn.DRAM_MEMORY_CONFIG)
    full_bc = mesh_config.allgather(t_int, ccl, axis=sp_axis, dim=2)
    full_bc_t = ttnn.to_torch(ttnn.get_device_tensors(full_bc)[0])[0, 0, :, 0].round().to(torch.long)  # [T] blk-cyclic
    nat = _blockcyclic_to_natural(full_bc, sp, n_chunks, chunk_local)
    obsB = ttnn.to_torch(ttnn.get_device_tensors(nat)[0])[0, 0, :, 0].round().to(torch.long)
    logger.info(f"[B dev ] identity={bool((obsB == torch.arange(T)).all())} n_wrong={int((obsB != torch.arange(T)).sum())}")

    # Structure: the AllGather'd block-cyclic order (pre-reorder) vs the expected block-cyclic order.
    # expected_bc[row] = natural token stored at block-cyclic row `row` = p (blockcyclic_positions).
    exp_bc = p  # p[row] = natural position at that block-cyclic row
    bc_wrong = int((full_bc_t != exp_bc).sum())
    logger.info(f"[B pre-reorder] AllGather'd-vs-expected-blockcyclic n_wrong={bc_wrong}/{T}")
    logger.info(f"[B pre-reorder] full_bc first 40: {full_bc_t[:40].tolist()}")
    logger.info(f"[B pre-reorder] expect   first 40: {exp_bc[:40].tolist()}")
    logger.info(f"[B post-reorder] obsB first 40: {obsB[:40].tolist()}")
    logger.info(f"[B post-reorder] obsB rows 620-660: {obsB[620:660].tolist()}")
    # per-32 block: which natural shard sits at each physical block, in the AllGather'd order
    blocks = full_bc_t.reshape(T // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK)
    logger.info(f"[B pre-reorder] block first-toks: {blocks[:, 0].tolist()}")
