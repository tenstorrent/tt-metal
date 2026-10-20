# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``rows=`` in the GLM-5.3 MTP predictor reference against a full run, on CPU at toy dims."""

import types

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference, random_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_3.mtp import glm_mtp_predictor_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config

HIDDEN = 64
CHUNK = 96
CACHE_LEN = 3 * CHUNK
TOTAL = 2 * CHUNK + 40  # MHA first chunk, MQA after it, the last one partial
SAMPLED_PCC = 0.99999


def _config():
    """GLM-5.3 config at toy dims; index_topk stays below every chunk so the top-k really selects."""
    return types.SimpleNamespace(
        **{
            **vars(glm_5_3_hf_config(CACHE_LEN)),
            "hidden_size": HIDDEN,
            "num_attention_heads": 4,
            "q_lora_rank": 32,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "index_n_heads": 4,
            "index_head_dim": 16,
            "index_topk": 24,
        }
    )


def _weights(seed: int = 11):
    """``(attn_norm, ffn_norm, moe, mtp)`` weights, same conventions as the MTP device test's helpers."""
    g = torch.Generator().manual_seed(seed)
    h, inter, n_routed = HIDDEN, 32, 16

    def rand(*shape, scale):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    def gain():
        return (torch.randn(h, generator=g) * 0.1 + 1.0).to(torch.bfloat16)

    def expert():
        return {
            "gate_proj": rand(inter, h, scale=h**-0.5),
            "up_proj": rand(inter, h, scale=h**-0.5),
            "down_proj": rand(h, inter, scale=inter**-0.5),
        }

    moe = {
        "gate_weights": {
            "weight": rand(n_routed, h, scale=h**-0.5),
            "e_score_correction_bias": torch.randn(n_routed, generator=g) * 0.01,
        },
        "routed_expert_weights": [expert() for _ in range(n_routed)],
        "shared_expert_weights": expert(),
    }
    mtp = {
        "enorm": gain(),
        "hnorm": gain(),
        "eh_proj": rand(h, 2 * h, scale=(2 * h) ** -0.5),
        "shared_head_norm": gain(),
    }
    return gain(), gain(), moe, mtp


def _run(config, mla_weights, weights, embeds, hiddens, rows_of=None):
    """The chunked, teacher-forced predictor reference; ``rows_of(start, end)`` picks each chunk's rows.
    Returns the chunks' results, the per-level references and each chunk's level-1 top-k."""
    attn_norm, ffn_norm, moe, mtp = weights
    refs = [SparseMLAReference(config, mla_weights, seq_len=CACHE_LEN) for _ in embeds]
    chunks, topks = [], []
    for start in range(0, TOTAL, CHUNK):
        end = min(start + CHUNK, TOTAL)
        chunks.append(
            glm_mtp_predictor_reference(
                config,
                mla_weights,
                mtp,
                attn_norm,
                ffn_norm,
                [e[:, start:end] for e in embeds],
                hiddens[0][:, start:end],
                CACHE_LEN,
                moe_weights=moe,
                hiddens=[h[:, start:end] for h in hiddens],
                mla_refs=refs,
                actual_start=start,
                actual_end=end,
                rows=None if rows_of is None else rows_of(start, end),
            )
        )
        topks.append(refs[0].indexer_topk[0].clone())
    return chunks, refs, topks


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


@pytest.mark.parametrize("num_levels", [1, 3])
@pytest.mark.parametrize("mode", ["sampled", "every"])
def test_rows_match_full_reference(mode: str, num_levels: int) -> None:
    config = _config()
    mla_weights = random_mla_weights(config)
    weights = _weights()
    g = torch.Generator().manual_seed(7)
    embeds = [torch.randn(1, TOTAL, HIDDEN, generator=g).to(torch.bfloat16) for _ in range(num_levels)]
    hiddens = [torch.randn(1, TOTAL, HIDDEN, generator=g).to(torch.bfloat16) for _ in range(num_levels)]

    picked = []

    def rows_of(start: int, end: int) -> torch.Tensor:
        keep = torch.rand(end - start, generator=g) < 0.25 if mode == "sampled" else torch.ones(end - start, dtype=bool)
        keep[[0, -1]] = True
        picked.append(keep.nonzero().flatten())
        return picked[-1]

    full, full_refs, full_topks = _run(config, mla_weights, weights, embeds, hiddens)
    part, part_refs, part_topks = _run(config, mla_weights, weights, embeds, hiddens, rows_of)

    for i, (rows, f, p) in enumerate(zip(picked, full_topks, part_topks)):
        assert torch.equal(f[rows].sort(-1).values, p.sort(-1).values), f"chunk {i}: the kept rows select other top-k"
    for rows, (f_x, f_out, f_normed, _), (p_x, p_out, p_normed, _) in zip(picked, full, part):
        for k in range(num_levels):
            assert torch.equal(f_x[k], p_x[k]), f"level {k}: the MTP projection must not depend on rows"
            for name, f, p in (("out", f_out[k][:, rows], p_out[k]), ("normed", f_normed[k][:, rows], p_normed[k])):
                if mode == "every":
                    assert torch.equal(f, p), f"level {k} {name}: rows=arange must reproduce the full run exactly"
                else:
                    assert _pcc(f, p) >= SAMPLED_PCC, f"level {k} {name}: PCC {_pcc(f, p):.7f} < {SAMPLED_PCC}"
    for k, (f, p) in enumerate(zip(full_refs, part_refs)):
        assert torch.equal(f.kvpe_cache, p.kvpe_cache), f"level {k}: KVPE cache must take every row"
        assert torch.equal(f.index_cache, p.index_cache), f"level {k}: index-K cache must take every row"
