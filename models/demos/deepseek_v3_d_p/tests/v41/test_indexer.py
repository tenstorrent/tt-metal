# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 index keys, indexer, candidates, top-k (bead F4) vs the reference at real dims.

Uses the oracle's V4.1 layers 2 -> 3 -> 20 -> 21 -> 24 reference (synthetic weights, 96 candidate blocks) and tests
layers 2 (ratio-2 KV+index source), 20 (ratio-1 candidate source), 24 (candidate-constrained index source). Bars (G1): index scores >= 0.999; selection exact given identical
scores (checked on tie-free scores fed to both sides); determinism. Recall of the device selection against
the reference under device scores is reported as a diagnostic only.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.kernel_cpu import fp4_act_quant
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.reference_weights import dequant
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41ChunkTables
from models.demos.deepseek_v3_d_p.tt.v41.indexer import SENTINEL, TtV41Indexer, TtV41IndexKeys
from tests.ttnn.utils_for_testing import comp_pcc

SCORE_PCC = 0.999
LAYERS = (2, 3, 20, 21, 24)
SEQ = 2048


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
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        )
    ],
    indirect=True,
)
def test_v41_indexer(mesh_device, device_params):
    cfg = type("V41TestConfig", (C,), {"CANDIDATE_TOPK_BLOCKS": 96})  # match the oracle's candidate count
    spec = orc.real_spec(LAYERS, SEQ, candidate_topk_blocks=96)
    reference = orc.build_reference(spec)
    result = orc.oracle(spec, orc.random_tokens(spec), model=reference)
    shape, (sp, tp) = tuple(mesh_device.shape), tuple(mesh_device.shape)
    seq = SEQ
    down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))

    def per_query(t):
        """Per-query device outputs (queries split over SP then TP) -> host [1, 1, S, W] in token order."""
        return torch.cat([ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)], dim=2)

    def up(t, dims=(2, None), dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=dims),
        )

    results, ref_candidates, real_candidates, synthetic_candidates = {}, None, None, None
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
        feed = ttnn.from_torch(
            tie_free.to(torch.bfloat16).reshape(sp, tp, seq // (sp * tp), visible),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(0, 1)),
        )
        # reference selection on the same scores
        compress_lens = (torch.arange(1, seq + 1) // ratio).unsqueeze(-1)
        ref_scores_sel = tie_free.clone()
        if layer == cfg.CANDIDATE_SOURCE_LAYER:
            ref_candidates = v41.select_candidate_blocks(
                ref_scores_sel, compress_lens, cfg.CANDIDATE_TOPK_BLOCKS, cfg.CANDIDATE_BLOCK_SIZE
            )
            dev_mask = per_query(indexer.candidates(feed, tables, 0, seq // (sp * tp), visible))[0, 0, :, :visible]
            assert torch.equal(ref_candidates, torch.isfinite(dev_mask)), "candidate blocks differ"
            real_candidates = indexer.candidates(score, tables, 0, seq // (sp * tp), visible)
            synthetic_candidates = indexer.candidates(feed, tables, 0, seq // (sp * tp), visible)
        elif layer > cfg.CANDIDATE_SOURCE_LAYER:
            ref_scores_sel = ref_scores_sel.masked_fill(~ref_candidates, float("-inf"))
            feed = ttnn.to_layout(
                ttnn.add(ttnn.to_layout(feed, ttnn.TILE_LAYOUT), synthetic_candidates), ttnn.ROW_MAJOR_LAYOUT
            )
        k = min(cfg.INDEX_TOPK, visible)
        ref_sel = ref_scores_sel.topk(k, dim=-1).indices
        ref_sel = torch.where(torch.gather(ref_scores_sel, -1, ref_sel) > float("-inf"), ref_sel, -1)
        dev_sel = per_query(ttnn.experimental.topk_large_indices(feed, k=max(16, -(-k // 16) * 16)))[0, 0, :, :k].long()
        dev_sel = torch.where(dev_sel == SENTINEL, -1, dev_sel)
        same = all(set(a.tolist()) == set(b.tolist()) for a, b in zip(ref_sel, dev_sel))
        results[f"L{layer}_selection_exact"] = same

        # determinism of the full indexer on real scores
        idx1, _ = indexer(
            tt_x, tt_qr, index_k, tables, 0, seq, real_candidates if layer > cfg.CANDIDATE_SOURCE_LAYER else None
        )
        idx2, _ = indexer(
            tt_x, tt_qr, index_k, tables, 0, seq, real_candidates if layer > cfg.CANDIDATE_SOURCE_LAYER else None
        )
        a, b = per_query(idx1)[0, 0], per_query(idx2)[0, 0]
        results[f"L{layer}_deterministic"] = bool(torch.equal(a, b))
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
