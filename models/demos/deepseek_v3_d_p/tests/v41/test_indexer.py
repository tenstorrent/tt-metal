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

import pytest
import torch

import ttnn
from models.common.timing_events import phase
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.kernel_cpu import fp4_act_quant
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.reference_weights import dequant
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41ChunkTables
from models.demos.deepseek_v3_d_p.tt.v41.indexer import SENTINEL, CandidateBlocks, TtV41Indexer, TtV41IndexKeys
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import gather_runs, segment_max
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
@pytest.mark.parametrize("mesh_device, device_params", MESH_2X4, indirect=True)
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
