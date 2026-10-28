# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 index keys, indexer, candidates, top-k (beads F4, F10) vs the reference at real dims, and the
selection kernels (``tt/v41/indexer_kernels.py``) vs torch.

``test_v41_indexer`` uses the oracle's V4.1 layers 2 -> 3 -> 20 -> 21 -> 24 reference (synthetic weights, 96
candidate blocks) and tests layers 2 (ratio-2 KV+index source), 20 (ratio-1 candidate source), 24
(candidate-constrained index source). Bars (G1): index scores >= 0.999; selection exact given identical scores
(checked on tie-free scores fed to both sides, through the module's ``select``: candidate block sets and top-k row
sets, with 96 candidate blocks (one-level candidates) and 32 (two-level)); determinism. Recall of the device
selection against the reference under device scores is reported as a diagnostic only.

``test_v41_indexer_selection_long`` feeds tie-free scores of chunks starting at 0 / 20,480 / 65,536 with the
production 2048 candidate blocks (all blocks candidates / one-level / two-level candidates and the gathered
candidate top-k, ``segment_max`` / ``gather_runs`` underneath) and checks the same exactness against ``v41.select_candidate_blocks`` + ``torch.topk``.
"""

import time

import pytest
import torch

import ttnn
from models.common.timing_events import phase
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.kernel_cpu import fp4_act_quant
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.reference_weights import dequant
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41ChunkTables
from models.demos.deepseek_v3_d_p.tt.v41.indexer import (
    SENTINEL,
    CandidateBlocks,
    DecodePins,
    TtV41Indexer,
    TtV41IndexKeys,
)
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import (
    RANGE_CHUNK,
    candidate_scores,
    gather_runs,
    range_scores,
    segment_max,
    split_geometry,
)
from tests.ttnn.utils_for_testing import comp_pcc

SCORE_PCC = 0.999
LAYERS = (2, 3, 20, 21, 24)
SEQ = 2048
MESH_2X4 = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
]
# the kernel tests derive every shape from the mesh: they run on the Galaxy layouts too (the 2x4 cases skip there)
KERNEL_MESHES = MESH_2X4 + galaxy_meshes()


def _per_query(t):
    """Per-query device outputs (queries split over SP then TP) -> host [1, 1, S, W] in token order."""
    return torch.cat([ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)], dim=2)


def _feed(mesh_device, scores: torch.Tensor):
    """Host [S, W] scores -> per-chip row-major [1, 1, S/(sp*tp), W] bf16 (the indexer's score layout)."""
    sp, tp = tuple(mesh_device.shape)
    seq, width = scores.shape
    return ttnn.from_torch(
        scores.to(torch.bfloat16).reshape(sp, tp, seq // (sp * tp), width),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
    )


def _rows(t: torch.Tensor) -> list[set]:
    """[S, k] device picks (sentinel = none) -> per-query sets; asserts the sentinels form a tail."""
    t = t.long()
    out = []
    for row in t:
        valid = row != SENTINEL
        n = int(valid.sum())
        assert bool(valid[:n].all()), "sentinel picks are not a tail"
        out.append(set(row[:n].tolist()))
    return out


def _ref_rows(scores: torch.Tensor, k: int) -> list[set]:
    top = scores.topk(k, dim=-1)
    return [set(i[v > float("-inf")].tolist()) for i, v in zip(top.indices, top.values)]


def _ref_blocks(mask: torch.Tensor, block: int) -> list[set]:
    """``select_candidate_blocks`` row mask [S, W] -> per-query kept block sets."""
    return [set((torch.nonzero(row[::block]).flatten()).tolist()) for row in mask]


def _check_selection(mesh_device, cfg, tables, scores: torch.Tensor, start: int, results: dict, tag: str):
    """Exactness of the candidate source's blocks and top-k and of a candidate index source's top-k on the same
    host ``scores`` [S, W] (-inf where not visible), through ``TtV41Indexer.select``."""
    seq, width = scores.shape
    visible = start + seq
    weights = {
        "wq_b": torch.zeros(cfg.INDEX_N_HEADS * cfg.INDEX_HEAD_DIM, cfg.Q_LORA_RANK),
        "weights_proj": torch.zeros(cfg.INDEX_N_HEADS, cfg.EMB_SIZE),
    }
    source = TtV41Indexer(mesh_device, cfg, cfg.CANDIDATE_SOURCE_LAYER, weights)
    consumer = TtV41Indexer(mesh_device, cfg, 24, weights)
    feed = _feed(mesh_device, scores)
    k = min(cfg.INDEX_TOPK, visible)
    compress_lens = (torch.arange(start + 1, start + seq + 1)).unsqueeze(-1)
    ref_mask = v41.select_candidate_blocks(
        scores.clone(), compress_lens, cfg.CANDIDATE_TOPK_BLOCKS, cfg.CANDIDATE_BLOCK_SIZE
    )
    idx, published = source.select(feed, tables, start, visible)
    assert isinstance(published, CandidateBlocks)
    if published.ids is None:
        results[f"{tag}_candidates_exact"] = bool(ref_mask[torch.isfinite(scores)].all())
    else:
        dev_blocks = _rows(_per_query(published.ids)[0, 0])
        results[f"{tag}_candidates_exact"] = dev_blocks == _ref_blocks(ref_mask, cfg.CANDIDATE_BLOCK_SIZE)
    ref_rows = _ref_rows(scores, k)
    results[f"{tag}_source_topk_exact"] = _rows(_per_query(idx)[0, 0]) == ref_rows
    if published.ids is not None and cfg.CANDIDATE_TOPK_BLOCKS > k:
        # the source's top-k among its own candidate blocks (its path on rows wider than SUBSET_TOPK_MIN_WIDTH)
        idx = source._topk_in_blocks(feed, published.ids, k)
        results[f"{tag}_source_topk_in_blocks_exact"] = _rows(_per_query(idx)[0, 0]) == ref_rows
    idx, _ = consumer.select(feed, tables, start, visible, published)
    masked = scores.masked_fill(~ref_mask, float("-inf"))
    ref_masked = _ref_rows(masked, k)
    results[f"{tag}_consumer_topk_exact"] = _rows(_per_query(idx)[0, 0]) == ref_masked
    return published


def _reference_scores(attn, x, qr):
    """The reference Indexer.forward score for a single-shot prefill (start 0), before candidate masking."""
    ind = attn.indexer
    seq, ratio, rd = x.shape[1], ind.compress_ratio, ind.rope_head_dim
    q = ind.wq_b(qr).unflatten(-1, (ind.n_local_heads, ind.index_head_dim))
    v41.apply_rotary_emb(q[..., -rd:], ind.freqs_cis[:seq])
    fp4_act_quant(q, 32, True)
    index_k = v41.shared_attn.index_k[:1, : seq // ratio]
    weights = ind.weights_proj(x) * (ind.softmax_scale * ind.n_heads**-0.5)
    score = torch.einsum("bshd,btd->bsht", q, index_k)
    score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    compress_lens = (torch.arange(1, seq + 1) // ratio).unsqueeze(-1)
    score.masked_fill_(torch.arange(seq // ratio) >= compress_lens, -torch.inf)
    return score[0].float()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device, device_params", MESH_2X4, indirect=True)
def test_v41_indexer(mesh_device, device_params):
    cfg = type("V41TestConfig", (C,), {"CANDIDATE_TOPK_BLOCKS": 96})  # match the oracle's candidate count
    spec = orc.real_spec(LAYERS, SEQ, candidate_topk_blocks=96)
    reference = orc.build_reference(spec)
    result = orc.oracle(spec, orc.random_tokens(spec), model=reference)
    shape, (sp, tp) = tuple(mesh_device.shape), tuple(mesh_device.shape)
    seq = SEQ
    down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))

    per_query = _per_query

    def up(t, dims=(2, None), dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=dims),
        )

    results, real_candidates = {}, None
    tables = V41ChunkTables(mesh_device, cfg, seq, seq, list(LAYERS))  # the forward's position tables
    for layer in (2, 20, 24):
        attn = reference.layers[LAYERS.index(layer)].attn
        x = result["blocks"][layer]["attn_in"][None]
        with v41.set_dtype(torch.bfloat16):
            qr = attn.q_norm(attn.wq_a(x))
            _, ref_idx = attn._compress_kv(x, qr, 0, seq)  # runs the reference compressor / keys / indexer
        ref_idx = ref_idx[0].long() - seq  # [S, k] rows, -1 - seq for none
        ratio = cfg.compress_ratio(layer)
        ind_w = {"wq_b": dequant(attn.indexer.wq_b), "weights_proj": attn.indexer.weights_proj.weight.detach()}
        indexer = TtV41Indexer(mesh_device, cfg, layer, ind_w)

        # index keys (KV sources) from the reference latent -> compare with the reference's index-K rows
        src = cfg.kv_source(layer)
        ref_k = v41.shared_attn.index_k[0, : seq // ratio].float()
        if layer in cfg.KV_SOURCE_LAYERS:
            keys = TtV41IndexKeys(
                mesh_device,
                cfg,
                layer,
                {"wk": attn.indexer.wk.weight.detach(), "k_norm": attn.indexer.k_norm.weight.detach()},
            )
            with v41.set_dtype(torch.bfloat16):
                latent = attn.compressor(x, 0)[0]
            dev_k = down(keys(up(latent[None, None]), tables.rope(True, ratio, 0)))[0, 0, :, : cfg.INDEX_HEAD_DIM]
            results[f"L{layer}_index_keys"] = comp_pcc(ref_k, dev_k.float(), 0.0)[1]
        index_k = ttnn.from_torch(
            ref_k.to(torch.bfloat16)[None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_x, tt_qr = up(x[0][None, None], dims=(2, 3)), up(qr[0][None, None])

        score, visible = indexer.scores(tt_x, tt_qr, index_k, tables, 0, seq)
        dev_score = per_query(score)[0, 0, :, :visible].float()
        with v41.set_dtype(torch.bfloat16):
            ref_score = _reference_scores(attn, x, qr)
        finite = torch.isfinite(ref_score)
        assert torch.equal(finite, torch.isfinite(dev_score)), f"L{layer}: visibility masks differ"
        results[f"L{layer}_scores"] = comp_pcc(ref_score[finite], dev_score[finite], 0.0)[1]

        if layer > cfg.CANDIDATE_SOURCE_LAYER and real_candidates.ids is not None:
            # candidate scoring (the long-row path of a candidate index source) vs the reference and the full-row
            # score, both over the candidate rows
            ids = real_candidates.ids
            sparse = per_query(indexer.candidate_rows(tt_x, tt_qr, index_k, tables, 0, ids))[0, 0].float()
            dense = per_query(gather_runs(score, ids, cfg.CANDIDATE_BLOCK_SIZE))[0, 0].float()
            host_ids = per_query(ids)[0, 0].long() & 0xFFFFFFFF
            cols = (host_ids.unsqueeze(-1) * cfg.CANDIDATE_BLOCK_SIZE + torch.arange(cfg.CANDIDATE_BLOCK_SIZE)).flatten(
                1
            )
            padded = torch.cat([ref_score, torch.full((seq, 1), float("-inf"))], dim=1)
            ref_rows = torch.gather(padded, 1, cols.clamp(max=ref_score.shape[1]))
            cand_finite = torch.isfinite(ref_rows)
            results[f"L{layer}_candidate_rows_mask_exact"] = torch.equal(torch.isfinite(sparse), cand_finite) and (
                torch.equal(torch.isfinite(dense), cand_finite)
            )
            results[f"L{layer}_candidate_scores"] = comp_pcc(ref_rows[cand_finite], sparse[cand_finite], 0.0)[1]
            results[f"L{layer}_full_row_candidate_scores_diag"] = comp_pcc(
                ref_rows[cand_finite], dense[cand_finite], 0.0
            )[1]

        # selection given identical, tie-free scores
        g = torch.Generator().manual_seed(layer)
        # distinct bf16 values per row: distinct positive bf16 bit patterns (integers > 256 would collide)
        codes = torch.stack([torch.randperm(0x4000, generator=g)[:visible] + 0x3000 for _ in range(seq)])
        rows = codes.to(torch.int16).view(torch.bfloat16).float()
        tie_free = torch.where(finite, rows, float("-inf"))
        if layer == cfg.CANDIDATE_SOURCE_LAYER:
            # candidate source + candidate index source on the same scores: one-level (96 blocks of 256 per
            # row, 64 superblocks) and two-level (32 blocks) candidate selection
            for blocks in (96, 32):
                sel_cfg = type(f"V41TestConfigK{blocks}", (C,), {"CANDIDATE_TOPK_BLOCKS": blocks})
                _check_selection(mesh_device, sel_cfg, tables, tie_free, 0, results, f"K{blocks}")
            real_candidates = indexer.candidates(score, tables, 0)
        elif layer < cfg.CANDIDATE_SOURCE_LAYER:
            idx, _ = indexer.select(_feed(mesh_device, tie_free), tables, 0, visible)
            k = min(cfg.INDEX_TOPK, visible)
            results[f"L{layer}_selection_exact"] = _rows(per_query(idx)[0, 0]) == _ref_rows(tie_free, k)

        # determinism of the full indexer on real scores
        cands = real_candidates if layer > cfg.CANDIDATE_SOURCE_LAYER else None
        idx1, _ = indexer(tt_x, tt_qr, index_k, tables, 0, seq, cands)
        idx2, _ = indexer(tt_x, tt_qr, index_k, tables, 0, seq, cands)
        a, b = per_query(idx1)[0, 0], per_query(idx2)[0, 0]
        results[f"L{layer}_deterministic"] = bool(torch.equal(a, b))
        k = min(cfg.INDEX_TOPK, visible)
        dev_rows = torch.where(a[:, :k].long() == SENTINEL, -1, a[:, :k].long())
        ref_rows = torch.where(ref_idx >= 0, ref_idx, -1)
        recall = [
            len(set(r[r >= 0].tolist()) & set(d[d >= 0].tolist())) / max(1, int((r >= 0).sum()))
            for r, d in zip(ref_rows, dev_rows)
        ]
        results[f"L{layer}_recall_diag"] = sum(recall) / len(recall)
    print(f"indexer: {results}")
    for key, value in results.items():
        if key.endswith("_scores") or key.endswith("_index_keys"):
            assert value >= SCORE_PCC, (key, results)
        if key.endswith("_exact") or key.endswith("_deterministic"):
            assert value, (key, results)


def _tie_free_scores(seq: int, start: int, g) -> torch.Tensor:
    """[seq, W] scores of a ratio-1 chunk at ``start`` (-inf where not visible, W = visible rows rounded to 32):
    per row a random 30 % (at least 4096) of the visible rows get distinct positive bf16 values and the rest one lower
    value, so block maxima and the top picks are tie-free while rows may hold more elements than bf16 has distinct
    values."""
    width = _round32(start + seq)
    p = torch.arange(start, start + seq).view(-1, 1)
    t = torch.arange(width).view(1, -1)
    out = torch.full((seq, width), -100.0)
    for i in range(seq):
        n = start + i + 1
        top = torch.randperm(n, generator=g)[: min(n, max(4096, (3 * n) // 10))]
        codes = torch.randperm(0x7F00 - 0x0080, generator=g)[: top.numel()] + 0x0080
        out[i, top] = codes.to(torch.int16).view(torch.bfloat16).float()
    return torch.where(t <= p, out, float("-inf"))


def _round32(x: int) -> int:
    return -(-x // 32) * 32


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("start", [0, 20480, 65536], ids=lambda s: f"start{s}")
@pytest.mark.parametrize("mesh_device, device_params", MESH_2X4, indirect=True)
def test_v41_indexer_selection_long(mesh_device, device_params, start):
    """Production candidate count (2048 blocks) on long rows: all-candidates / one-level / two-level paths."""
    seq = 1024
    g = torch.Generator().manual_seed(start)
    with phase("reference", what="tie-free scores"):
        scores = _tie_free_scores(seq, start, g)
    results = {}
    with phase("compute", what="tables + select + reference selection"):
        tables = V41ChunkTables(mesh_device, C, start + seq, seq, [2, 20, 24])
        _check_selection(mesh_device, C, tables, scores, start, results, f"start{start}")
    print(f"indexer selection: {results}")
    assert all(results.values()), results


def _decode_scores(rows: int, cap: int, visible: torch.Tensor, g) -> torch.Tensor:
    """[rows, cap + 32] scores of a decode tile over a pool of ``cap`` rows (the pad tile after them): row i sees
    its first ``visible[i]`` rows, -inf beyond; an idle row (0) is all -inf. Tie-free where it matters: up to 4096
    of a row's blocks hold one distinct positive bf16 code each (at a random row of the block), the rest of the
    visible rows are -100, so the top-2048 blocks by maximum, the top-k rows and the top-k inside the candidate
    blocks are unambiguous. The codes stay below 2^32: the segment max's -inf column masks saturate against values
    above ~1e36 on the FPU (`-inf + x` -> `-bf16max + x`), which real scores never reach."""
    width = cap + 32
    out = torch.full((rows, width), float("-inf"))
    for i in range(rows):
        n = int(visible[i])
        if n == 0:
            continue
        out[i, :n] = -100.0
        nblocks = -(-n // 8)
        blocks = torch.randperm(nblocks, generator=g)[: min(nblocks, 4096)]
        pos = (blocks * 8 + torch.randint(0, 8, (blocks.numel(),), generator=g)).clamp(max=n - 1)
        # and enough distinct positives for a tie-free top-k (INDEX_TOPK) wherever the row has that many rows
        pos = torch.unique(torch.cat([pos, torch.randperm(n, generator=g)[: min(n, 8192)]]))
        codes = torch.randperm(0x4F00 - 0x0080, generator=g)[: pos.numel()] + 0x0080
        out[i, pos] = codes.to(torch.int16).view(torch.bfloat16).float()
    return out


def _replicated(mesh_device, host: torch.Tensor, dtype):
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _chip0(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])[0, 0]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("cap", [8192, 16384, 20480, 69632, 1048576], ids=lambda c: f"cap{c}")
@pytest.mark.parametrize("mesh_device, device_params", KERNEL_MESHES, indirect=True)
def test_v41_decode_selection(mesh_device, device_params, cap):
    """The decode tile's selection (``TtV41Indexer.select`` with ``DecodePins``) against the reference: 32 rows at
    their own visible counts (idle rows, a row below INDEX_TOPK, block and superblock boundaries, the full pool),
    the candidate source's published blocks vs ``select_candidate_blocks`` with that row's count, its top-k vs the
    dense top-k, a consumer's top-k vs its dense score masked by the published blocks. Capacities on the
    all-candidates, one-level and two-level paths (the last two below and above SUBSET_TOPK_MIN_WIDTH)."""
    rows, cfg = 32, C
    g = torch.Generator().manual_seed(cap)
    visible = torch.randint(1, cap + 1, (rows,), generator=g)
    visible[0], visible[1] = 0, 0  # idle users: the sentinel pin, an all -inf row
    visible[2] = 300  # fewer visible rows than INDEX_TOPK: a sentinel tail
    visible[3] = cap  # the whole pool
    visible[4] = 8 * (cap // 16)  # a block boundary (the newest block is full)
    visible[5] = 8 * (cap // 16) + 1  # the row after it (a one-row newest block)
    visible[6] = 32 * (cap // 64) + 5  # inside a superblock
    visible[7] = min(cap, 2048 * 8 + 3)  # just past the budget of one level
    with phase("reference", what="tie-free decode scores + reference selection"):
        scores = _decode_scores(rows, cap, visible, g)
        k = cfg.INDEX_TOPK
        ref_blocks, ref_src, ref_cons = [], [], []
        for i in range(rows):
            n = int(visible[i])
            if n == 0:
                ref_blocks.append(None), ref_src.append(None), ref_cons.append(None)
                continue
            mask = v41.select_candidate_blocks(
                scores[i : i + 1].clone(), n, cfg.CANDIDATE_TOPK_BLOCKS, cfg.CANDIDATE_BLOCK_SIZE
            )
            ref_blocks.append(_ref_blocks(mask, cfg.CANDIDATE_BLOCK_SIZE)[0])
            ref_src.append(_ref_rows(scores[i : i + 1], k)[0])
            ref_cons.append(_ref_rows(scores[i : i + 1].masked_fill(~mask, float("-inf")), k)[0])
    with phase("compute", what="device selection"):
        weights = {
            "wq_b": torch.zeros(cfg.INDEX_N_HEADS * cfg.INDEX_HEAD_DIM, cfg.Q_LORA_RANK),
            "weights_proj": torch.zeros(cfg.INDEX_N_HEADS, cfg.EMB_SIZE),
        }
        source = TtV41Indexer(mesh_device, cfg, cfg.CANDIDATE_SOURCE_LAYER, weights)
        consumer = TtV41Indexer(mesh_device, cfg, 24, weights)
        feed = _replicated(mesh_device, scores.to(torch.bfloat16)[None, None], ttnn.bfloat16)
        newest = torch.where(visible > 0, visible - 1, torch.full_like(visible, -1)).to(torch.int32)
        pins = DecodePins(_replicated(mesh_device, newest.view(1, 1, 1, rows), ttnn.uint32))
        idx, published = source.select(feed, pins, 0, k)
        assert isinstance(published, CandidateBlocks)
        src_rows = _rows(_chip0(idx))
        dev_blocks = None if published.ids is None else _rows(_chip0(published.ids))
        idx, _ = consumer.select(feed, pins, 0, k, published)
        cons_rows = _rows(_chip0(idx))
    results = {}
    for i in range(rows):
        if ref_src[i] is None:
            continue
        if dev_blocks is None:
            results[f"row{i}_blocks"] = cap // cfg.CANDIDATE_BLOCK_SIZE <= cfg.CANDIDATE_TOPK_BLOCKS
        else:
            results[f"row{i}_blocks"] = dev_blocks[i] == ref_blocks[i]
        results[f"row{i}_source"] = src_rows[i] == ref_src[i]
        results[f"row{i}_consumer"] = cons_rows[i] == ref_cons[i]
    bad = {key: value for key, value in results.items() if not value}
    print(f"decode selection cap {cap}: {len(results)} checks, {len(bad)} wrong: {sorted(bad)[:12]}")
    assert not bad, sorted(bad)


def _pin_table(mesh_device, rows: int):
    """Per-chip uint32 [1, 1, 1, 8] first chunk query index, as ``V41ChunkTables.query_first``."""
    sp, tp = tuple(mesh_device.shape)
    first = torch.zeros(sp, tp, 1, 8, dtype=torch.int32)
    first[..., 0] = (torch.arange(sp * tp, dtype=torch.int32) * rows).reshape(sp, tp, 1)
    return ttnn.from_torch(
        first,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
    )


def _pin_rows(mesh_device, newest: torch.Tensor, rows: int):
    """Per-chip uint32 [1, 1, 1, rows] per-row pin page: entry i the chip's row i's value of ``newest`` [seq]."""
    sp, tp = tuple(mesh_device.shape)
    return ttnn.from_torch(
        newest.to(torch.int64).reshape(sp, tp, 1, rows).to(torch.int32),  # the sentinel wraps to -1: its bits
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
    )


def _kernel_rows(seq: int, width: int, g) -> torch.Tensor:
    """[seq, width] bf16 rows with negatives, zeros, repeated values and -inf runs (whole blocks, partial blocks and
    a -inf row tail)."""
    x = (torch.randn(seq, width, generator=g) * 4).to(torch.bfloat16).float()
    x[:, ::7] = -x[:, ::7].abs() - 1  # plenty of all-negative blocks
    x[::5, 3:40] = float("-inf")
    x[1::3, width // 2 :] = float("-inf")
    x[2, :] = -3.0  # a row of ties
    x[3, 10:20] = 0.0
    return x.to(torch.bfloat16).float()


@pytest.mark.parametrize("width", [1056, 8288, 16384], ids=lambda w: f"W{w}")
@pytest.mark.parametrize("mesh_device, device_params", KERNEL_MESHES, indirect=True)
def test_v41_segment_max(mesh_device, device_params, width):
    """segment_max vs torch (exact, every chip): seg 8 / 32, partial last chunk and padded output widths,
    negative / -inf / tied values, pins inside, at the edge of and past the row, and the rank-0 (masked) pin."""
    sp, tp = tuple(mesh_device.shape)
    rows = 64  # per chip
    seq = rows * sp * tp
    g = torch.Generator().manual_seed(width)
    x = _kernel_rows(seq, width, g)
    feed = _feed(mesh_device, x)
    pin = _pin_table(mesh_device, rows)
    q = torch.arange(seq)
    for seg in (8, 32):
        out_width = -(-(width // seg) // 32) * 32
        base = x.reshape(seq, width // seg, seg).amax(dim=-1)
        for pin_base, pin_mask in ((width - seq // 2, 0xFFFFFFFF), (0, 31), (width, 0xFFFFFFFF)):
            ref = torch.full((seq, out_width), float("-inf"))
            ref[:, : width // seg] = base
            e = (pin_base + q) & pin_mask
            hit = e < width
            ref[q[hit], e[hit] // seg] = float("inf")
            out = segment_max(feed, seg, pin, pin_base, pin_mask)
            assert list(out.shape) == [1, 1, rows, out_width]
            dev = _per_query(out)[0, 0].float()
            assert torch.equal(dev, ref), (seg, pin_base, pin_mask, (dev != ref).nonzero()[:8].tolist())
            again = _per_query(segment_max(feed, seg, pin, pin_base, pin_mask))[0, 0].float()
            assert torch.equal(again, dev), "segment_max is not deterministic"
        # the per-row pin page (the decode tile): row i pinned at its own entry, sentinel rows not pinned
        newest = torch.randint(0, width + 64, (seq,))  # some past the row: not pinned
        newest[1::7] = SENTINEL
        for pin_base, pin_mask in ((0, 0xFFFFFFFF), (0, 31)):
            ref = torch.full((seq, out_width), float("-inf"))
            ref[:, : width // seg] = base
            e = (pin_base + newest) & pin_mask
            hit = (e < width) & (newest != SENTINEL)
            ref[q[hit], e[hit] // seg] = float("inf")
            dev = _per_query(segment_max(feed, seg, _pin_rows(mesh_device, newest, rows), pin_base, pin_mask))
            dev = dev[0, 0].float()
            assert torch.equal(dev, ref), ("per-row", seg, pin_mask, (dev != ref).nonzero()[:8].tolist())


@pytest.mark.parametrize("k", [96, 2048], ids=lambda k: f"K{k}")
@pytest.mark.parametrize("mesh_device, device_params", MESH_2X4, indirect=True)
def test_v41_gather_runs(mesh_device, device_params, k):
    """gather_runs vs torch (exact, every chip): runs of 8 and 32, ids in any order with repeats, the sentinel and
    ids just past the row reading -inf."""
    sp, tp = tuple(mesh_device.shape)
    rows, width = 64, 65536 + 96
    seq = rows * sp * tp
    g = torch.Generator().manual_seed(k)
    x = _kernel_rows(seq, width, g)
    feed = _feed(mesh_device, x)
    for run in (8, 32):
        nruns = width // run
        ids = torch.randint(0, nruns, (seq, k), generator=g)
        ids[:, -3] = nruns  # first id past the row
        ids[::2, -1] = SENTINEL
        ids = ids & 0xFFFFFFFF
        ids[:, 0] = nruns - 1  # the row's last run
        ids[:, 1] = ids[:, 2]  # a repeat
        dev_ids = ttnn.from_torch(
            ids.to(torch.int32).reshape(sp, tp, rows, k),  # the sentinel wraps to -1: its uint32 bits
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
        )
        cols = ids.clamp(max=nruns).unsqueeze(-1) * run + torch.arange(run)  # [seq, k, run]
        padded = torch.cat([x, torch.full((seq, run), float("-inf"))], dim=1)
        ref = torch.gather(padded, 1, cols.clamp(max=width).reshape(seq, -1))
        out = gather_runs(feed, dev_ids, run)
        assert list(out.shape) == [1, 1, rows, k * run]
        dev = _per_query(out)[0, 0].float()
        assert torch.equal(dev, ref), (run, (dev != ref).nonzero()[:8].tolist())
        assert torch.equal(_per_query(gather_runs(feed, dev_ids, run))[0, 0].float(), dev), "not deterministic"


def _fp4_like(shape, g) -> torch.Tensor:
    """bf16 values with at most 2 significant bits (FP4 e2m1 codes times a power of two), as QDQ'd q and index-K."""
    codes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    v = codes[torch.randint(0, 8, shape, generator=g)] * torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
    return (v * 2.0 ** torch.randint(-4, 0, shape[:-1] + (1,), generator=g)).to(torch.bfloat16)


def _candidate_score_ref(q, w, k, ids, positions) -> torch.Tensor:
    """[S, K * 8] reference of ``candidate_scores``: per head a bf16 q . k (exact fp32 dot, one rounding), relu,
    weighted sum over the heads in fp32, one bf16 rounding; -inf for sentinel ids and rows past the query. Also
    returns sum_h |w_h relu(q_h . k)| (the scale of the head sum's accumulation error)."""
    seq, nk = ids.shape
    nblocks = k.shape[0] // 8
    valid = ids < nblocks
    rows = (ids.clamp(max=nblocks - 1).unsqueeze(-1) * 8 + torch.arange(8)).reshape(seq, -1)  # [S, K * 8]
    out, magnitude = torch.empty(seq, nk * 8), torch.empty(seq, nk * 8)
    for i in range(seq):
        kr = k[rows[i]].float()  # [K * 8, 128]
        s = torch.einsum("hd,td->ht", q[i].float(), kr).to(torch.bfloat16).float().relu()
        terms = w[i].float().unsqueeze(-1) * s
        out[i] = terms.sum(0).to(torch.bfloat16).float()
        magnitude[i] = terms.abs().sum(0)
    visible = valid.repeat_interleave(8, dim=1) & (rows <= positions.view(-1, 1))
    return torch.where(visible, out, float("-inf")), magnitude


@pytest.mark.parametrize("nk, start", [(96, 2048), (2048, 131072)], ids=["K96", "K2048"])
@pytest.mark.parametrize("mesh_device, device_params", KERNEL_MESHES, indirect=True)
def test_v41_candidate_scores(mesh_device, device_params, nk, start):
    """candidate_scores vs torch (every chip): the newest block pinned first (partly past the query), random blocks
    with a repeat, a sentinel tail on some queries; FP4-like q / index-K, bf16 head weights. The finite pattern must
    match exactly; values within one bf16 rounding plus 2^-6 of the head terms' magnitude of the fp32-accumulated
    reference, PCC >= 0.9999; deterministic."""
    sp, tp = tuple(mesh_device.shape)
    rows = 32  # per chip
    seq = rows * sp * tp
    g = torch.Generator().manual_seed(nk)
    with phase("reference", what="inputs + torch candidate scores"):
        t_rows = start + seq
        q = _fp4_like((seq, 32, 128), g)
        k = _fp4_like((t_rows, 128), g)
        w = (torch.randn(seq, 32, generator=g) * 0.05).to(torch.bfloat16)
        positions = start + torch.arange(seq)
        ids = torch.stack([torch.randperm(start // 8, generator=g)[:nk] for _ in range(seq)])
        ids[:, 0] = positions // 8  # the pinned newest block: rows past the query are masked
        ids[:, 2] = ids[:, 3]  # a repeat
        ids[::3, nk - 5 :] = SENTINEL
        ref, magnitude = _candidate_score_ref(q, w, k, ids, positions)
    with phase("compute", what="device candidate scores"):
        shard = ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1))

        def up(t, dtype):
            return ttnn.from_torch(
                t.reshape(sp, tp, rows, -1),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=shard,
            )

        dev_k = ttnn.from_torch(
            k[None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        dev_ids = up(ids.to(torch.int32), ttnn.uint32)  # the sentinel wraps to -1: its uint32 bits
        args = (
            up(q.reshape(seq, -1), ttnn.bfloat16),
            up(w, ttnn.bfloat16),
            dev_k,
            dev_ids,
            _pin_table(mesh_device, rows),
        )
        out = candidate_scores(*args, start, t_rows)
        assert list(out.shape) == [1, 1, rows, nk * 8]
        dev = _per_query(out)[0, 0].float()
        again = _per_query(candidate_scores(*args, start, t_rows))[0, 0].float()
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(dev), finite), (torch.isfinite(dev) != finite).nonzero()[:8].tolist()
    diff = (dev[finite] - ref[finite]).abs()
    exact = float((diff == 0).float().mean())
    pcc = comp_pcc(ref[finite], dev[finite], 0.0)[1]
    print(f"candidate_scores K{nk}: exact {exact:.4f}, max abs diff {float(diff.max()):.3e}, pcc {pcc:.6f}")
    # the device's head sum is not an exact fp32 sum (measured: within 2^-7.6 of sum_h |terms| beyond the output's
    # bf16 rounding); bound it by one bf16 rounding of the output plus 2^-6 of the terms' magnitude
    bound = ref[finite].abs() * 2**-7 + magnitude[finite] * 2**-6 + 1e-6
    assert bool((diff <= bound).all()), float((diff / bound).max())
    assert pcc >= 0.9999, pcc
    assert torch.equal(again, dev), "candidate_scores is not deterministic"
    # the per-row pin page: each row's own position (here: the chunk's, so the same finite pattern), then a row
    # whose entry is the sentinel sees every candidate row, and rows with earlier positions see fewer
    with phase("per-row pins", what="device candidate scores with a per-row pin page"):
        newest = positions - start
        dev2 = _per_query(candidate_scores(*args[:4], _pin_rows(mesh_device, newest, rows), start, t_rows))
        assert torch.equal(dev2[0, 0].float(), dev), "the per-row page with the chunk's positions differs"
        newest = newest.clone()
        newest[::5] = SENTINEL
        newest[1::5] = (newest[1::5] - 37).clamp(min=0)
        ref2, magnitude2 = _candidate_score_ref(q, w, k, ids, torch.where(newest == SENTINEL, 1 << 40, start + newest))
        dev2 = _per_query(candidate_scores(*args[:4], _pin_rows(mesh_device, newest, rows), start, t_rows))
        dev2 = dev2[0, 0].float()
        finite2 = torch.isfinite(ref2)
        assert torch.equal(torch.isfinite(dev2), finite2), (torch.isfinite(dev2) != finite2).nonzero()[:8].tolist()
        assert torch.equal(dev2[finite2 & finite], dev[finite2 & finite]), "visible scores changed with the pin page"
        diff2 = (dev2[finite2] - ref2[finite2]).abs()
        bound2 = ref2[finite2].abs() * 2**-7 + magnitude2[finite2] * 2**-6 + 1e-6
        assert bool((diff2 <= bound2).all()), float((diff2 / bound2).max())
    # the index-K as persistent block pages (the decode pool's form), the queried slot's block 0 at a page offset
    with phase("block pages", what="device candidate scores from block pages at a base"):
        base_blocks = 3 * 256  # another slot's pages before this one's
        pages = torch.cat(
            [torch.randn(base_blocks * 8, 128).to(torch.bfloat16), k, torch.zeros(32, 128).to(torch.bfloat16)]
        )
        dev_pages = ttnn.from_torch(
            pages.reshape(1, 1, -1, 8 * 128),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        dev3 = _per_query(
            candidate_scores(args[0], args[1], dev_pages, args[3], args[4], start, t_rows, block_base=base_blocks)
        )
        assert torch.equal(dev3[0, 0].float(), dev), "the block pages at a base differ from the row-major cache"


def _range_score_ref(q_u: torch.Tensor, w_u: torch.Tensor, k_u: torch.Tensor, newest: int):
    """[W] reference of ``range_scores`` for one user (``_candidate_score_ref``'s arithmetic over every key): per
    head a bf16 q . k, relu, weighted sum over the heads in fp32, one bf16 rounding; -inf past ``newest`` (every
    key for the sentinel). Also returns sum_h |w_h relu(q_h . k)|."""
    s = torch.einsum("hd,td->ht", q_u.float(), k_u.float()).to(torch.bfloat16).float().relu()
    terms = w_u.float().unsqueeze(-1) * s
    out, magnitude = terms.sum(0).to(torch.bfloat16).float(), terms.abs().sum(0)
    visible = torch.arange(k_u.shape[0]) <= (newest if newest != SENTINEL else -1)
    return torch.where(visible, out, float("-inf")), magnitude


def _unsplit(rows: torch.Tensor, width: int) -> torch.Tensor:
    """A split-layout score [R, Wr] (``split_geometry``) -> the one row [width] of global key order."""
    n_rows, row_width = rows.shape
    cw = RANGE_CHUNK * 32
    per_row = row_width // cw
    # (r, slot) -> chunk slot * R + r; chunk c holds keys [c * cw, (c + 1) * cw)
    chunks = rows.view(n_rows, per_row, cw).permute(1, 0, 2).reshape(-1)  # chunk-major, then rows, then keys
    assert bool((chunks[width:] == float("-inf")).all()), "the slots past the last chunk are not -inf"
    return chunks[:width]


def _valid_topk(picked: set, scores: torch.Tensor, k: int) -> bool:
    """``picked`` is a top-k of ``scores`` [W] up to ties: the right count and no pick below the k-th value."""
    finite = int(torch.isfinite(scores).sum())
    n = min(k, finite)
    if len(picked) != n:
        return False
    if n == 0:
        return True
    kth = float(scores.topk(n).values[-1])
    return all(float(scores[t]) >= kth for t in picked)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("split", [1, 32], ids=lambda r: f"rows{r}")
@pytest.mark.parametrize("mesh_device, device_params", KERNEL_MESHES, indirect=True)
def test_v41_range_scores(mesh_device, device_params, split):
    """``range_scores`` (the single-query scorer) vs torch on every chip: four users of a
    20,480-row pool at a position that is a multiple of neither 32 nor 8, below INDEX_TOPK, on a tile boundary,
    idle (sentinel) and at the full pool, in the one-row and the 32-row split layouts; the finite pattern exact, the values within ``candidate_scores``'s bound,
    deterministic, bit-equal to ``candidate_scores`` over the same keys from block pages; then the one-row selection
    (``TtV41Indexer.select`` with a one-entry pin page) on the device score is a top-k of it up to ties, the
    consumer's inside the published blocks, and the dense path (``indexer_score_dsa`` + mask) agrees on the
    selection up to the scorers' rounding (the overlap is printed; at least 90 % is demanded)."""
    cfg, rows = C, 32
    cap = 20480
    width = cap + 32
    users = 5
    g = torch.Generator().manual_seed(58)
    newest = torch.full((rows,), -1, dtype=torch.int64)  # the sentinel's bits
    newest[:users] = torch.tensor([19999, 299, 4095, -1, cap - 1])
    with phase("inputs", what="FP4-like query / index-K, head weights"):
        q = _fp4_like((rows, 32, 128), g)
        w = (torch.randn(rows, 32, generator=g) * 0.05).to(torch.bfloat16)
        k = _fp4_like((users * width, 128), g)
        k.view(users, width, 128)[:, cap:] = 0  # the pad tile is never written
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        dev_q = ttnn.from_torch(
            q.reshape(1, 1, rows, -1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )
        dev_w = ttnn.from_torch(
            w.reshape(1, 1, rows, 32),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )
        dev_k = ttnn.from_torch(
            k.reshape(users, 1, width, 128),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )
        dev_pin = _replicated(mesh_device, newest.view(1, 1, 1, rows).to(torch.int32), ttnn.uint32)
    source = TtV41Indexer(
        mesh_device,
        cfg,
        cfg.CANDIDATE_SOURCE_LAYER,
        {
            "wq_b": torch.zeros(cfg.INDEX_N_HEADS * cfg.INDEX_HEAD_DIM, cfg.Q_LORA_RANK),
            "weights_proj": torch.zeros(cfg.INDEX_N_HEADS, cfg.EMB_SIZE),
        },
    )
    consumer = TtV41Indexer(
        mesh_device,
        cfg,
        24,
        {
            "wq_b": torch.zeros(cfg.INDEX_N_HEADS * cfg.INDEX_HEAD_DIM, cfg.Q_LORA_RANK),
            "weights_proj": torch.zeros(cfg.INDEX_N_HEADS, cfg.EMB_SIZE),
        },
    )
    topk = cfg.INDEX_TOPK
    overlaps = []
    for u in range(users):
        n_u = int(newest[u]) if newest[u] >= 0 else SENTINEL
        k_u = k.view(users, width, 128)[u]
        with phase(f"user{u}", what=f"range_scores of user {u} (newest {n_u})"):
            out = range_scores(dev_q, dev_w, dev_k, dev_pin, u, width, rows=split)
            n_rows, _, _, row_width = split_geometry(width, split)
            assert n_rows == split, (n_rows, split)
            assert list(out.shape) == [1, 1, n_rows, row_width], (tuple(out.shape), n_rows, row_width)
            devs = [_unsplit(ttnn.to_torch(d)[0, 0].float(), width) for d in ttnn.get_device_tensors(out)]
            dev = devs[0]
            for i, other in enumerate(devs[1:], 1):
                assert torch.equal(other, dev), f"chip {i} differs from chip 0"
            again = _unsplit(_chip0(range_scores(dev_q, dev_w, dev_k, dev_pin, u, width, rows=split)).float(), width)
            assert torch.equal(again, dev), "range_scores is not deterministic"
        ref, magnitude = _range_score_ref(q[u], w[u], k_u, n_u)
        finite = torch.isfinite(ref)
        assert torch.equal(torch.isfinite(dev), finite), (
            (torch.isfinite(dev) != finite).nonzero()[:8].flatten().tolist()
        )
        if n_u == SENTINEL:
            print(f"range_scores user {u}: idle, all -inf")
            continue
        diff = (dev[finite] - ref[finite]).abs()
        bound = ref[finite].abs() * 2**-7 + magnitude[finite] * 2**-6 + 1e-6
        pcc = comp_pcc(ref[finite], dev[finite], 0.0)[1]
        print(
            f"range_scores user {u} (newest {n_u}): exact {float((diff == 0).float().mean()):.4f}, max abs diff {float(diff.max()):.3e}, pcc {pcc:.6f}"
        )
        assert bool((diff <= bound).all()), float((diff / bound).max())
        assert pcc >= 0.9999, pcc
        # the same arithmetic as candidate_scores over the same keys (every block in order, from block pages)
        with phase(f"user{u}-cand", what="candidate_scores over every block"):
            nb = width // 8
            pages = ttnn.from_torch(
                k_u.reshape(1, 1, nb, 8 * 128),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=rep,
            )
            ids = _replicated(mesh_device, torch.arange(nb, dtype=torch.int32).view(1, 1, 1, nb), ttnn.uint32)
            q1 = ttnn.from_torch(
                q[u].reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=rep,
            )
            w1 = ttnn.from_torch(
                w[u].reshape(1, 1, 1, 32),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=rep,
            )
            pin1 = ttnn.slice(dev_pin, [0, 0, 0, u], [1, 1, 1, u + 1])
            cand = _chip0(candidate_scores(q1, w1, pages, ids, pin1, 0, width))[0].float()
        assert torch.equal(cand, dev), f"candidate_scores differs at {(cand != dev).nonzero()[:8].flatten().tolist()}"
        # the split selection on the device score (the rows merged), with the user's one-entry pin page
        with phase(f"user{u}-select", what="split selection"):
            idx, published = source.select_split(out, pin1, width, topk)
            picks = _rows(_chip0(idx))[0]
            blocks = None if published.ids is None else _rows(_chip0(published.ids))[0]
            assert published.ids is not None and list(published.ids.shape) == [1, 1, 1, cfg.CANDIDATE_TOPK_BLOCKS]
            cidx = consumer.select_user_in_blocks(dev_q, dev_w, pages, pin1, published.ids, u, width, 0, topk)
            cpicks = _rows(_chip0(cidx))[0]
        assert _valid_topk(picks, dev, topk), f"user {u}: the source's picks are not a top-{topk}"
        mask = v41.select_candidate_blocks(
            dev[None].clone(), n_u + 1, cfg.CANDIDATE_TOPK_BLOCKS, cfg.CANDIDATE_BLOCK_SIZE
        )
        assert blocks is not None
        ref_blocks = _ref_blocks(mask, cfg.CANDIDATE_BLOCK_SIZE)[0]
        if blocks != ref_blocks:
            # ties among block maxima at the boundary: every device block must reach the reference's k-th maximum
            bmax = torch.nn.functional.pad(dev, (0, -width % 8), value=float("-inf")).view(-1, 8).amax(-1)
            kth = min(float(bmax[b]) for b in ref_blocks if b != n_u // 8)
            extra = sorted(blocks - ref_blocks)[:8]
            missing = sorted(ref_blocks - blocks)[:8]
            detail = f"{len(blocks)} vs {len(ref_blocks)} blocks, kth {kth}, extra {[(b, float(bmax[b])) for b in extra]}, missing {[(b, float(bmax[b])) for b in missing]}"
            assert len(blocks) == len(ref_blocks) and all(
                float(bmax[b]) >= kth for b in blocks if b != n_u // 8
            ), f"user {u}: blocks {detail}"
        assert _valid_topk(cpicks, dev.masked_fill(~mask[0], float("-inf")), topk), f"user {u}: the consumer's picks"
        # the dense reference path: indexer_score_dsa on the 32-row tile + the mask
        with phase(f"user{u}-dense", what="indexer_score_dsa + mask"):
            qt = ttnn.from_torch(
                q.permute(1, 0, 2).reshape(1, 32, rows, 128),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=rep,
            )
            wt = ttnn.from_torch(
                w.reshape(1, 1, rows, 32),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=rep,
            )
            config = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(256, width), head_group_size=0)
            dense = ttnn.experimental.indexer_score_dsa(
                qt, dev_k, wt, chunk_start_idx=cap, seq_shard_axes=[], program_config=config, cache_batch_idx=u
            )
            dense_u = _chip0(dense)[u].float()
            dense_u[n_u + 1 :] = float("-inf")
        fin = torch.isfinite(dense_u) & finite
        pcc_dense = comp_pcc(dense_u[fin], dev[fin], 0.0)[1]
        dense_top = set(dense_u.topk(min(topk, n_u + 1)).indices.tolist())
        overlap = len(picks & dense_top) / max(1, len(picks))
        overlaps.append(overlap)
        print(f"range_scores user {u}: vs indexer_score_dsa pcc {pcc_dense:.6f}, top-{topk} overlap {overlap:.4f}")
        assert overlap >= 0.9, overlap
    # timing of the full-pool user: the launch and the split selection (with the host's dispatch)
    with phase("timing", what="range_scores and select_split at the full pool"):
        u = users - 1
        pin1 = ttnn.slice(dev_pin, [0, 0, 0, u], [1, 1, 1, u + 1])
        out = range_scores(dev_q, dev_w, dev_k, dev_pin, u, width, rows=split)
        for what, fn in (
            ("range_scores", lambda: range_scores(dev_q, dev_w, dev_k, dev_pin, u, width, rows=split)),
            ("select_split (source)", lambda: source.select_split(out, pin1, width, topk)),
            ("select_split (plain)", lambda: consumer.select_split(out, pin1, width, topk, CandidateBlocks(None))),
        ):
            fn()
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(10):
                fn()
            ttnn.synchronize_device(mesh_device)
            print(f"{what} at {cap} keys: {(time.perf_counter() - t0) / 10 * 1e3:.3f} ms a call (with host)")
    print(f"range_scores overlaps with the dense path: {overlaps}")


@pytest.mark.parametrize("split", [1, 32], ids=lambda r: f"rows{r}")
@pytest.mark.parametrize("mesh_device, device_params", KERNEL_MESHES, indirect=True)
def test_v41_select_split_rows(mesh_device, device_params, split):
    """``TtV41Indexer.select_split_rows`` (the live rows' selections merged in one chain) on
    four tile rows' split-layout scores stacked along the rows, against ``select_split`` row by row: the top-k ids
    and the candidate source's published blocks bit-equal, for the candidate source and for a plain consumer, in
    the one-row and the 32-row layouts; the rows at distinct positions (one below INDEX_TOPK, one idle). Then the
    per-row chain and the merged one are timed over four rows."""
    cfg, rows = C, 32
    cap = 20480
    width = cap + 32
    users = 4
    g = torch.Generator().manual_seed(135)
    newest = torch.full((rows,), -1, dtype=torch.int64)
    newest[:users] = torch.tensor([19999, 299, 4095, -1])
    tile_rows = [0, 4, 5, 6]  # a verify block's rows under the c1 MoE: the user's own, then the rows past the users'
    with phase("inputs", what="FP4-like query / index-K, head weights"):
        q = _fp4_like((rows, 32, 128), g)
        w = (torch.randn(rows, 32, generator=g) * 0.05).to(torch.bfloat16)
        k = _fp4_like((users * width, 128), g)
        k.view(users, width, 128)[:, cap:] = 0
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        dev_q = ttnn.from_torch(
            q.reshape(1, 1, rows, -1), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
        )
        dev_w = ttnn.from_torch(
            w.reshape(1, 1, rows, 32), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
        )
        dev_k = ttnn.from_torch(
            k.reshape(users, 1, width, 128), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep
        )
        # the tile's pin page: row t's newest (the verify rows read their user's keys with their own pin entry)
        page = torch.full((rows,), -1, dtype=torch.int64)
        for u, t in enumerate(tile_rows):
            page[t] = newest[u]
        dev_pin = _replicated(mesh_device, page.view(1, 1, 1, rows).to(torch.int32), ttnn.uint32)
    weights = {
        "wq_b": torch.zeros(cfg.INDEX_N_HEADS * cfg.INDEX_HEAD_DIM, cfg.Q_LORA_RANK),
        "weights_proj": torch.zeros(cfg.INDEX_N_HEADS, cfg.EMB_SIZE),
    }
    source = TtV41Indexer(mesh_device, cfg, cfg.CANDIDATE_SOURCE_LAYER, weights)
    consumer = TtV41Indexer(mesh_device, cfg, 24, weights)
    topk = cfg.INDEX_TOPK
    with phase("scores", what="range_scores of the four rows (user u's keys, tile row t's query and pin)"):
        scores = [range_scores(dev_q, dev_w, dev_k, dev_pin, u, width, rows=split, row=t) for u, t in enumerate(tile_rows)]
        n_rows = scores[0].shape[2]
        assert n_rows == split, (n_rows, split)
        stacked = ttnn.concat(scores, dim=2)
        idx = _replicated(
            mesh_device,
            torch.tensor([t for t in tile_rows for _ in range(n_rows)], dtype=torch.int32).view(1, 1, 1, -1),
            ttnn.uint32,
        )
        wide = ttnn.concat([dev_pin] * (-(-idx.shape[3] // rows)), dim=3)
        newest_page = ttnn.gather(wide, -1, idx)
        got = [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(newest_page)[0])[0, 0, 0].tolist()]
        want = [int(page[t]) & 0xFFFFFFFF for t in tile_rows for _ in range(n_rows)]
        assert got == want, (got[:8], want[:8])
    for name, indexer, cand in (("source", source, None), ("consumer", consumer, CandidateBlocks(None))):
        with phase(f"{name}-merged", what="select_split_rows over the stacked rows"):
            merged, published = indexer.select_split_rows(stacked, newest_page, users, width, topk, cand)
            assert list(merged.shape) == [1, 1, users, topk], tuple(merged.shape)
            merged_t = _chip0(merged)
            blocks_t = None if published is None or published.ids is None else _chip0(published.ids)
        for u, t in enumerate(tile_rows):
            with phase(f"{name}-row{u}", what="select_split of the row alone"):
                pin_t = ttnn.slice(dev_pin, [0, 0, 0, t], [1, 1, 1, t + 1])
                one, pub = indexer.select_split(scores[u], pin_t, width, topk, cand)
                one_t = _chip0(one)[0]
            assert torch.equal(merged_t[u], one_t), f"{name}: row {u}'s top-k differs at {(merged_t[u] != one_t).nonzero()[:8].flatten().tolist()}"
            if blocks_t is not None:
                pub_t = _chip0(pub.ids)[0]
                assert torch.equal(blocks_t[u], pub_t), f"{name}: row {u}'s blocks differ"
            n_valid = len(_rows(one_t.view(1, -1))[0])
            print(f"select_split_rows {name} row {u} (newest {int(newest[u])}): {n_valid} picks, bit-equal to the row's own")
    with phase("timing", what="the per-row chain against the merged one, four rows (with the host's dispatch)"):
        pins = [ttnn.slice(dev_pin, [0, 0, 0, t], [1, 1, 1, t + 1]) for t in tile_rows]
        for what, fn in (
            ("select_split x 4 (source)", lambda: [source.select_split(s, p, width, topk) for s, p in zip(scores, pins)]),
            ("select_split_rows (source)", lambda: source.select_split_rows(stacked, newest_page, users, width, topk)),
            (
                "select_split x 4 (plain)",
                lambda: [consumer.select_split(s, p, width, topk, CandidateBlocks(None)) for s, p in zip(scores, pins)],
            ),
            (
                "select_split_rows (plain)",
                lambda: consumer.select_split_rows(stacked, newest_page, users, width, topk, CandidateBlocks(None)),
            ),
        ):
            fn()
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(10):
                fn()
            ttnn.synchronize_device(mesh_device)
            print(f"{what} at {cap} keys, {split} split rows: {(time.perf_counter() - t0) / 10 * 1e3:.3f} ms a call (with host)")
