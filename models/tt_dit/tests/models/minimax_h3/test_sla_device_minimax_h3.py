# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""SLA mode's device ops (coarse stage, topk_large_indices, vsa_sdpa) on the 4x8 galaxy at the 10 s 768p geometry."""

import math

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3 import vsa_stages_minimax_h3
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, build_sla_geometry
from ....utils.tensor import bf16_tensor_2dshard
from .common import GALAXY_RING


def _attend(q, k, v, keys):
    p = torch.softmax(q @ k[keys].T / math.sqrt(q.shape[-1]), dim=-1)
    return p @ v[keys]


def _mesh_concat(mesh_device, x, sp_axis, tp_axis):
    dims = [None, None]
    dims[sp_axis], dims[tp_axis] = 2, 1
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    return ttnn.to_torch(x, mesh_composer=composer)


@GALAXY_RING
@pytest.mark.parametrize("seq_len", [75449], ids=["10s_768p"])
def test_sla_stage_and_vsa_sdpa_on_device(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, seq_len):
    """The device SLA path (coarse stage + vsa_sdpa, raw selection) on random Q/K/V at the 10 s 768p geometry.

    Three checks tell the ops apart: topk_large_indices alone at the stage's k_pad against torch.topk, the
    device selection against LightX2V's (set overlap per row; bf16 near-ties keep it below 1), and vsa_sdpa's
    output against block-sparse attention over the ids the device itself selected.
    """
    sp, tp = tuple(mesh_device.shape)[sp_axis], tuple(mesh_device.shape)[tp_axis]
    if sp != 8:
        pytest.skip("4x8 only")
    heads, d, sparsity, q_block_tiles = 2 * tp, 128, 0.85, 2
    g = build_sla_geometry(seq_len, (1, 1, 1), sp_factor=sp, q_block_tiles=q_block_tiles)
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    cfg = vsa_stages_minimax_h3.MiniMaxH3VSAConfig(sparsity=sparsity, selection="sla")
    stage = vsa_stages_minimax_h3.MiniMaxH3VSACoarseStage(
        g,
        sparsity=sparsity,
        selection="sla",
        sla_q_block_tiles=q_block_tiles,
        padded_pooling=True,
        head_dim=d,
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        ccl_manager=ccl,
    )
    tiles, slots, k, k_pad = g.tiles_per_shard, stage.slots_per_shard, stage.k, stage.k_pad
    logger.info(f"T455 geometry n_tiles={g.n_tiles} tiles/shard={tiles} slots={slots} k={k} k_pad={k_pad}")

    # (1) topk_large_indices alone on stage-shaped rows: [1, H, slots, sp * slots], -inf on pad slots
    gen = torch.Generator().manual_seed(1)
    rows = torch.randn(1, heads // tp, slots, sp * slots, generator=gen).bfloat16()
    rows.view(1, heads // tp, slots, sp, slots)[..., tiles:] = float("-inf")
    tt_rows = ttnn.from_torch(
        rows, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )  # fmt: skip
    topk_ok = {}
    for kk in sorted({16, 64, 128, 160, k_pad, 192, 256}):
        got = ttnn.to_torch(ttnn.get_device_tensors(ttnn.experimental.topk_large_indices(tt_rows, k=kk))[0])
        ref = torch.topk(rows.float(), kk, dim=-1).indices
        same = torch.sort(got.to(torch.int64), -1).values.eq(torch.sort(ref, -1).values).all(-1)
        topk_ok[kk] = same.float().mean().item()
    logger.info(f"T455 topk_large_indices rows with the exact top-k set, by k: {topk_ok}")

    # (2) + (3) the stage and vsa_sdpa on random Q/K/V
    q, kt, v = (torch.randn(heads, seq_len, d, generator=gen).bfloat16().float() for _ in range(3))
    qp, kp, vp = (g.pack_rows(x, dim=-2) for x in (q, kt, v))
    up = lambda x: bf16_tensor_2dshard(x.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tq, tk, tv = up(qp), up(kp), up(vp)
    _, tt_ids = stage(tq, tk, tv, compute_o_c=False, raw_selection=True)
    tk_g = ccl.all_gather_persistent_buffer(tk, dim=2, mesh_axis=sp_axis)
    tv_g = ccl.all_gather_persistent_buffer(tv, dim=2, mesh_axis=sp_axis)
    tt_out = ttnn.transformer.vsa_sdpa(
        tq, tk_g, tv_g, tt_ids, stage.block_counts_tensor(),
        k_chunk_blocks=cfg.k_chunk_blocks, streaming=cfg.streaming, distributed=cfg.distributed, v2=cfg.v2,
        heads_per_group=cfg.heads_per_group, mcast_log=cfg.mcast_log, list_len=k, exempt_ids=stage.exempt_ids,
        dense_row_mask=stage.dense_row_mask_tensor(), coarse_slots_shift=stage.coarse_slots_shift,
        coarse_real_per_shard=tiles, dense_row_hint=stage.dense_row_hint,
    )  # fmt: skip
    raw = _mesh_concat(mesh_device, tt_ids, sp_axis, tp_axis)[0].to(torch.int64)  # [H, sp * tiles, k_pad]
    out = _mesh_concat(mesh_device, tt_out, sp_axis, tp_axis)[0].float()  # [H, padded_len, d]
    dev_ids = (raw - (raw >> stage.coarse_slots_shift) * (slots - tiles))[..., :k]

    n_real = -(-seq_len // VSA_TILE_TOKENS)
    blkq = q_block_tiles * VSA_TILE_TOKENS
    pool = lambda x, blk: torch.stack(
        [x[:, i : i + blk].mean(1) for i in range(0, seq_len, blk)], dim=1
    )  # [H, blocks, d]; the tail block averages its real rows
    scores = pool(q, blkq) @ pool(kt - kt.mean(1, keepdim=True), VSA_TILE_TOKENS).transpose(-1, -2)
    ref_ids = torch.topk(scores, k, dim=-1).indices  # [H, q_blocks, k]
    in_range = bool(((dev_ids >= 0) & (dev_ids < n_real)).all())
    overlap = torch.zeros(heads, n_real)
    for h in range(heads):
        for t in range(n_real):
            overlap[h, t] = len(set(dev_ids[h, t].tolist()) & set(ref_ids[h, t // q_block_tiles].tolist())) / k
    dups = sum(len(set(dev_ids[h, t].tolist())) < k for h in range(heads) for t in range(n_real))
    logger.info(
        f"T455 selection: ids in range {in_range}, rows with duplicate ids {dups}, overlap with LightX2V "
        f"mean {overlap.mean():.4f} min {overlap.min():.4f}, rows under 0.5: {int((overlap < 0.5).sum())}"
    )

    counts = g.valid_counts
    pcc = lambda a, b: torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
    sample = torch.randperm(n_real, generator=gen)[:48].tolist() + [0, n_real - 1]
    worst_dev, worst_ref = 1.0, 1.0
    for h in range(heads):
        for t in sample:
            r = slice(t * VSA_TILE_TOKENS, min(seq_len, (t + 1) * VSA_TILE_TOKENS))
            keys = lambda blocks: torch.cat(
                [torch.arange(b * VSA_TILE_TOKENS, b * VSA_TILE_TOKENS + int(counts[b])) for b in sorted(blocks)]
            )
            got = out[h, r]
            worst_dev = min(worst_dev, pcc(got, _attend(qp[h, r], kp[h], vp[h], keys(dev_ids[h, t].tolist()))))
            ids_ref = ref_ids[h, t // q_block_tiles].tolist()
            worst_ref = min(worst_ref, pcc(got, _attend(q[h, r], kt[h], v[h], keys(ids_ref))))
    logger.info(f"T455 output PCC, worst sampled q tile: vs device ids {worst_dev:.5f}, vs LightX2V {worst_ref:.5f}")

    assert all(x == 1.0 for x in topk_ok.values()), topk_ok
    assert in_range and dups == 0
    assert worst_dev > 0.999
    assert overlap.mean() > 0.9
