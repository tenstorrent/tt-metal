# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Algebra lock for the CSA compressor / indexer decomposition (tt/v4/attention/csa_math.py) against the reference
DeepseekV4CSACompressor + DeepseekV4Indexer, device-free, fp32: single call, two chunks through the reference
cache (the overlap prior), and the top-k selection mask vs the reference block bias."""

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4CSACompressor,
    apply_rotary_pos_emb,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.v4.attention import csa_math as C


class _Cache:
    def __init__(self, layer):
        self.layers = [layer]


def _cfg():
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    cfg.hidden_size = 1024  # smaller projections; the math is dimension-agnostic
    return cfg


def _ref(cfg, seed=3):
    torch.manual_seed(seed)
    ref = DeepseekV4CSACompressor(cfg).eval()
    with torch.no_grad():
        ref.position_bias.normal_(0.0, 0.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.indexer.position_bias.normal_(0.0, 0.5)
        ref.indexer.kv_norm.weight.uniform_(0.5, 1.5)
    return ref


def _mirror_entries(ref, hidden, prior, first_window_position):
    with torch.no_grad():
        kv, gate = ref.kv_proj(hidden)[0], ref.gate_proj(hidden)[0]
    return C.compress_chunk(
        kv,
        gate,
        ref.position_bias,
        ref.kv_norm.weight,
        ref.kv_norm.variance_epsilon,
        ref.rotary_emb,
        prior,
        first_window_position,
    )


def test_single_call_entries_match_reference():
    cfg = _cfg()
    ref = _ref(cfg)
    S = 512
    torch.manual_seed(1)
    hidden = torch.randn(1, S, cfg.hidden_size)
    q_res = torch.randn(1, S, cfg.q_lora_rank)
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        entries_ref, _ = ref(hidden, q_res, pos, None, 0)  # [1, 1, S/4, 512]
    entries, _ = _mirror_entries(ref, hidden, C.empty_prior(cfg.head_dim), 0)
    torch.testing.assert_close(entries, entries_ref[0, 0], rtol=1e-4, atol=1e-4)


def test_two_chunks_carry_the_overlap_prior():
    cfg = _cfg()
    ref = _ref(cfg)
    S1, S2 = 256, 320
    torch.manual_seed(2)
    hidden = torch.randn(1, S1 + S2, cfg.hidden_size)
    q_res = torch.randn(1, S1 + S2, cfg.q_lora_rank)
    cache = _Cache(DeepseekV4CSACache(cfg))
    with torch.no_grad():
        ref(hidden[:, :S1], q_res[:, :S1], torch.arange(S1).unsqueeze(0), cache, 0)
        all_entries, _ = ref(hidden[:, S1:], q_res[:, S1:], torch.arange(S1, S1 + S2).unsqueeze(0), cache, 0)
    e1, prior = _mirror_entries(ref, hidden[:, :S1], C.empty_prior(cfg.head_dim), 0)
    e2, _ = _mirror_entries(ref, hidden[:, S1:], prior, S1)
    torch.testing.assert_close(e1, all_entries[0, 0, : S1 // 4], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(e2, all_entries[0, 0, S1 // 4 :], rtol=1e-4, atol=1e-4)
    # the prior is this chunk's last 32 Ca rows, gate already biased
    assert prior[0].shape == (32, cfg.head_dim) and prior[1].shape == (32, cfg.head_dim)


def test_indexer_scores_and_selection_match_reference_block_bias():
    cfg = _cfg()
    cfg.index_topk = 64
    ref = _ref(cfg)
    S = 512
    torch.manual_seed(5)
    hidden = torch.randn(1, S, cfg.hidden_size)
    q_res = torch.randn(1, S, cfg.q_lora_rank)
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        entries_ref, block_bias = ref(hidden, q_res, pos, None, 0)  # block_bias [1, 1, S, T]
        idx = ref.indexer
        keys, _ = C.compress_chunk(
            idx.kv_proj(hidden)[0],
            idx.gate_proj(hidden)[0],
            idx.position_bias,
            idx.kv_norm.weight,
            idx.kv_norm.variance_epsilon,
            idx.rotary_emb,
            C.empty_prior(idx.head_dim),
            0,
        )
        cos, sin = idx.rotary_emb(hidden, position_ids=pos, layer_type="compress")
        q = idx.q_b_proj(q_res).view(1, S, -1, idx.head_dim).transpose(1, 2)  # [1, H, S, Dh]
        q = apply_rotary_pos_emb(q, cos, sin)[0]
        weights = idx.scorer.weights_proj(hidden)[0]  # [S, H]
    scores = C.indexer_scores(q, keys, weights, idx.head_dim, cfg.index_n_heads)
    T = keys.shape[0]
    # the causal cut: entry w visible to token p iff w < (p + 1) // 4
    causal = torch.log((torch.arange(T).view(1, T) < ((torch.arange(S) + 1) // 4).view(S, 1)).float())
    sel = C.selection_mask(scores + causal, min(cfg.index_topk, T)) + causal
    ref_mask = block_bias[0, 0]
    same = ((sel == 0) == (ref_mask == 0)).float().mean().item()
    assert same >= 0.999, f"selection agrees with the reference block bias on {same:.5f} of the entries"
    # rows with fewer than k valid entries select exactly their valid entries
    row = 40  # (40+1)//4 = 10 valid entries < k
    assert int((sel[row] == 0).sum()) == 10 and int((ref_mask[row] == 0).sum()) == 10
