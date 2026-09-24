# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MSA cross-chunk cache read from a MID-SLAB chunk start at TP=4 x SP=8, vs a torch golden at the true positions.

A multi-turn continuation resumes at a 32-token boundary, so the chunk's cached_len need not be a whole number of
5120-token chunks. The KV writer then rotates which SP rank holds the chunk's first 640-token block, the boundary
rank's rows straddle two slab blocks, and the ranks end up unevenly filled. msa_sp_attention_cache_read must size
its gather for the fullest rank (msa_cache_read_extent), and the indexer / sparse_sdpa_msa must mask every query at
its rotated position (rotated_chip_positions), not the linear cached_len + rank*640.

The inputs are built so a wrong position cannot hide under averaging:
  - index keys: every 128-block has a distinct strength (geometric, 3% apart -- far above bf16 rounding), index
    queries are all the same direction, so the top-16 selection is unambiguous and a causal-mask error in the
    indexer would pick a (stronger) future block;
  - attention keys: box-filtered random walks, so a query's logit peaks at its own position and decays over +-32
    tokens -- each query attends mostly to its neighbours, and future neighbours leaking through a mis-placed
    diagonal-block mask would carry as much weight as the past ones.
The chunk-aligned case is kept as the regression anchor; every mid-slab case also asserts that masking at the old
linear positions would fail the threshold, so the test keeps discriminating.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.reference.model import msa_block_selection_chunk
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

NQ, NKV, NIDX, HEAD_DIM = 64, 4, 4, 128
BLOCK, TOPK = 128, 16
CHUNK_LOCAL = 640
WINDOW = 32  # attention keys: correlation 1 - |dt| / (2*WINDOW + 1)
LOGIT = 10.0  # scaled q.k at dt == 0
PCC = 0.99


def local_keys(n, gen):
    """[n, HEAD_DIM] unit vectors; u_s . u_t decays linearly from 1 to 0 over |s - t| = 2*WINDOW + 1."""
    w = torch.randn(n + 2 * WINDOW, HEAD_DIM, generator=gen)
    c = torch.cumsum(torch.cat([torch.zeros(1, HEAD_DIM), w]), dim=0)
    u = c[2 * WINDOW + 1 :] - c[:n]  # u_t = sum(w[t : t + 2*WINDOW + 1])
    return u / u.norm(dim=-1, keepdim=True)


def make_inputs(capacity, chunk_global, cached_len, seed):
    """Natural-order K/V/index_k over the cache capacity, and the chunk's q/index_q over [cached_len, +chunk)."""
    gen = torch.Generator().manual_seed(seed)
    scale = HEAD_DIM**-0.5
    k = torch.stack([local_keys(capacity, gen) for _ in range(NKV)]).unsqueeze(0)  # [1, NKV, cap, HD]
    v = torch.randn(1, NKV, capacity, HEAD_DIM, generator=gen)
    base = torch.randn(HEAD_DIM, generator=gen)
    base = base / base.norm()
    strength = 1.03 ** torch.randperm(capacity // BLOCK, generator=gen).float()
    ik = (strength.repeat_interleave(BLOCK)[:, None] * base).reshape(1, 1, capacity, HEAD_DIM)
    group = NQ // NKV
    q_dir = k[:, :, cached_len : cached_len + chunk_global].repeat_interleave(group, dim=1)
    q = (LOGIT / scale) * q_dir + 0.1 * torch.randn(1, NQ, chunk_global, HEAD_DIM, generator=gen)
    iq = base.expand(1, NIDX, chunk_global, HEAD_DIM).clone()
    return q, iq, k, v, ik


def golden_rank(q, iq, k, v, ik, positions, cached_len, kv_len, *, heads, mask_positions=None):
    """Golden MSA output [1, len(heads), R, HD] for one rank's query rows at global `positions` (their KV group is
    heads[0] // (NQ // NKV); one index head per group, as the TP=4 deployment has). Selection follows the reference
    model (causal index scores, block max-pool, forced local block, top-k) at the TRUE positions. The token-level
    causal cut is at the true positions, or -- to emulate sparse_sdpa_msa masking at wrong positions -- only inside
    the block holding mask_positions[i] and only past it, which is what the kernel's diagonal-block mask does."""
    g = heads[0] // (NQ // NKV)
    scale = HEAD_DIM**-0.5
    kpos = torch.arange(kv_len)
    sel = torch.empty(len(positions), kv_len, dtype=torch.bool)
    # The selection op wants contiguous query runs; a rank's rows are at most two (the straddle).
    start = 0
    while start < len(positions):
        stop = start + 1
        while stop < len(positions) and positions[stop] == positions[stop - 1] + 1:
            stop += 1
        rows = positions[start:stop] - cached_len
        sel[start:stop] = msa_block_selection_chunk(
            iq[0, g, rows].float(), ik[0, 0, :kv_len].float(), scale, BLOCK, TOPK, int(positions[start]), kv_len
        )
        start = stop
    if mask_positions is None:
        allowed = sel & (kpos[None, :] <= positions[:, None])
    else:
        mp = mask_positions[:, None]
        allowed = sel & ~((kpos[None, :] // BLOCK == mp // BLOCK) & (kpos[None, :] > mp))
    qh = q[0, heads][:, positions - cached_len].float()  # [h, R, HD]
    scores = scale * qh @ k[0, g, :kv_len].float().T
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    return (torch.softmax(scores, dim=-1) @ v[0, g, :kv_len].float()).unsqueeze(0)


def rank_positions(cached_len, sp):
    return [torch.tensor(p) for p in rotated_chip_positions(cached_len, sp, CHUNK_LOCAL)]


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize(
    "start_offset",
    [0, 32, 128, 640, 736],
    ids=["slab_aligned", "mid_block_straddle", "block_aligned_straddle", "rotated", "rotated_straddle"],
)
def test_msa_sp_cache_read_mid_slab(mesh_device, device_params, start_offset, reset_seeds):
    from models.demos.minimax_m3.tt.attention.kv_cache import allocate_kv_caches, write_index_k_chunk, write_kv_chunk
    from models.demos.minimax_m3.tt.attention.msa import msa_cache_read_extent, msa_sp_attention_cache_read

    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    chunk_global = sp * CHUNK_LOCAL
    cached_len = chunk_global + start_offset  # one whole prior chunk, then a 32-aligned resume point
    capacity = 3 * chunk_global
    kv_len, _ = msa_cache_read_extent(cached_len, CHUNK_LOCAL, sp, BLOCK, capacity // sp)
    q, iq, k, v, ik = make_inputs(capacity, chunk_global, cached_len, seed=start_offset)
    positions = rank_positions(cached_len, sp)
    order = torch.cat(positions)  # chip-major: the rows each SP rank holds, in its local-row order
    group = NQ // tp

    mesh_config = MeshConfig((rows, cols), tp=tp)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    def shard(t, split_heads):
        dims = [None, None]
        dims[sp_axis] = 2
        dims[1] = 1 if split_heads else None
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims),
        )

    kv = allocate_kv_caches(
        mesh_device, num_layers=1, max_seq_len=capacity, sp_axis=sp_axis, head_dim=HEAD_DIM, cache_dtype=ttnn.bfloat16
    )

    def write(kv_actual, idx):
        write_kv_chunk(
            kv,
            shard(k[:, :, idx], True),
            shard(v[:, :, idx], True),
            slot_idx=0,
            layer_idx=0,
            kv_actual=kv_actual,
            sp_axis=sp_axis,
        )
        write_index_k_chunk(
            kv, shard(ik[:, :, idx], False), slot_idx=0, layer_idx=0, kv_actual=kv_actual, sp_axis=sp_axis
        )

    # The previous turn: whole chunks from 0 (their tail past cached_len is overwritten by the resumed chunk).
    for c in range(-(-cached_len // chunk_global)):
        write(c * chunk_global, torch.cat(rank_positions(c * chunk_global, sp)))
    write(cached_len, order)

    out = msa_sp_attention_cache_read(
        shard(q[:, :, order - cached_len], True),
        shard(iq[:, :, order - cached_len], True),
        kv,
        slot=0,
        mesh_config=mesh_config,
        ccl_manager=ccl,
        cached_len=cached_len,
        chunk_local=CHUNK_LOCAL,
        scale=HEAD_DIM**-0.5,
        block_size=BLOCK,
        topk_blocks=TOPK,
        num_groups=1,
    )
    dts = ttnn.get_device_tensors(out)

    worst, worst_linear = 1.0, 1.0
    for r in range(rows):
        for c in range(cols):
            heads = [c * group]  # one head per KV group keeps the golden cheap; the group shares its selection
            gold = golden_rank(q, iq, k, v, ik, positions[r], cached_len, kv_len, heads=heads)
            dev = ttnn.to_torch(dts[r * cols + c]).float()[:, :1]
            _, p = comp_pcc(gold, dev, PCC)
            worst = min(worst, p)
            if start_offset % chunk_global:
                linear = cached_len + r * CHUNK_LOCAL + torch.arange(CHUNK_LOCAL)
                bug = golden_rank(q, iq, k, v, ik, positions[r], cached_len, kv_len, heads=heads, mask_positions=linear)
                _, p_lin = comp_pcc(gold, bug, PCC)
                worst_linear = min(worst_linear, p_lin)
    logger.info(
        f"[msa-mid-slab] cached_len={cached_len} kv_len={kv_len}: worst rank pcc={worst:.5f} "
        f"(old linear mask would give {worst_linear:.5f})"
    )
    if start_offset % chunk_global:
        assert worst_linear < PCC, "test lost its power: the old linear mask would pass too"
    assert worst >= PCC, f"mid-slab MSA cache read diverges from the golden (worst rank pcc={worst:.5f})"
