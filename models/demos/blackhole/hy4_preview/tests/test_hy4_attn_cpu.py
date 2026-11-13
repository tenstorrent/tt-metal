# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S0 (CPU) tests of the Hy4 attention layer: sharding plan vs Hy4Config, rope tables, absorbed-MLA dataflow == naive formulation, KV row geometry."""
import dataclasses
import os

import pytest
import torch

from models.demos.blackhole.hy4_preview.config import DEFAULT_CKPT, Hy4Config
from models.demos.blackhole.hy4_preview.tt import hy4_kv_cache as KV
from models.demos.blackhole.hy4_preview.tt.hy4_attn_ref import attention_dataflow, attention_naive
from models.demos.blackhole.hy4_preview.tt.hy4_attn_weights import build_attn_host_weights, column_shard
from models.demos.blackhole.hy4_preview.tt.hy4_rope import apply_rope_host, rope_tables, rotation_matrix

CKPT = os.environ.get("HY4_CKPT", DEFAULT_CKPT)
needs_ckpt = pytest.mark.skipif(not os.path.exists(os.path.join(CKPT, "config.json")), reason="checkpoint not mounted")


def meta_sd(cfg, indexer):
    H, D = cfg.num_attention_heads, cfg.hidden_size
    s = {
        "self_attn.q_a_proj.weight": (cfg.q_lora_rank, D),
        "self_attn.q_a_layernorm.weight": (cfg.q_lora_rank,),
        "self_attn.q_b_proj.weight": (cfg.q_b_out, cfg.q_lora_rank),
        "self_attn.kv_a_proj_with_mqa.weight": (cfg.kv_a_out, D),
        "self_attn.kv_a_layernorm.weight": (cfg.kv_lora_rank,),
        "self_attn.kv_b_proj.weight": (cfg.kv_b_out, cfg.kv_lora_rank),
        "self_attn.linear_gate.weight": (cfg.attn_out_dim, D),
        "self_attn.o_proj.weight": (D, cfg.attn_out_dim),
        "self_attn.learnable_sink_param": (H,),
    }
    if indexer:
        s.update(
            {
                "self_attn.indexer.wq_b.weight": (cfg.indexer_q_out, cfg.q_lora_rank),
                "self_attn.indexer.wk.weight": (cfg.index_head_dim, D),
                "self_attn.indexer.weights_proj.weight": (cfg.index_n_heads, D),
                "self_attn.indexer.k_norm.weight": (cfg.index_head_dim,),
                "self_attn.indexer.k_norm.bias": (cfg.index_head_dim,),
            }
        )
    return {k: torch.empty(v, device="meta") for k, v in s.items()}


@needs_ckpt
@pytest.mark.parametrize("indexer", [True, False])
def test_shard_plan_real_dims(indexer):
    cfg = Hy4Config.from_checkpoint(CKPT)
    hw = build_attn_host_weights(cfg, meta_sd(cfg, indexer))
    hc = cfg.heads_per_col
    assert hc == 8
    exp = {  # per-column shape
        "q_a": (1, 1, 6144, 256),
        "q_b": (1, 1, 2048, 8 * 256),
        "kv_a": (1, 1, 6144, 576),
        "w_uk": (1, 8, 192, 512),
        "w_uv_t": (1, 8, 512, 256),
        "gate": (1, 1, 6144, 2048),
        "o_proj": (1, 1, 2048, 6144),
        "sink_over_scale": (1, 1, 1, 8),
        "sink_dec": (1, 1, 1, 32),
        "q_a_norm": (1, 1, 64, 32),
        "kv_a_norm": (1, 1, 16, 32),
    }
    if indexer:
        exp.update(
            {
                "idx_wq_b": (1, 1, 2048, 4096),
                "idx_wk": (1, 1, 6144, 128),
                "idx_wp": (1, 1, 6144, 32),
                "idx_k_norm_w": (1, 1, 4, 32),
                "idx_k_norm_b": (1, 1, 4, 32),
            }
        )
    assert set(hw) == set(exp)
    for k, shp in exp.items():
        w = hw[k]
        got = tuple(w.tensor.shape)
        if w.shard_dim is not None:
            assert got[w.shard_dim] % 8 == 0
        assert tuple(column_shard(w, 3).shape) == shp, k
    # every sharded dim in tiles
    for k in ("q_a", "q_b", "gate"):
        assert hw[k].tensor.shape[3] // 8 % 32 == 0
    assert hw["o_proj"].tensor.shape[2] // 8 % 32 == 0


@needs_ckpt
def test_shard_columns_cover_heads():
    cfg = Hy4Config.from_checkpoint(CKPT)
    sd = {k: v for k, v in meta_sd(cfg, False).items()}
    hw = build_attn_host_weights(cfg, sd)
    # column c's q_b slice covers query heads 8c..8c+7: slice width = 8 heads x 256
    assert hw["q_b"].tensor.shape[3] == 64 * 256 and column_shard(hw["q_b"], 0).shape[3] == 8 * 256
    # kv_b split: w_uk + w_uv_t carry all 64 x (192 + 256) x 512 values
    assert hw["w_uk"].tensor.numel() + hw["w_uv_t"].tensor.numel() == cfg.kv_b_out * cfg.kv_lora_rank


@pytest.mark.parametrize("interleave", [True, False])
def test_rope_tables_match_complex_rotation(interleave):
    pos = torch.tensor([0, 1, 7, 1000, 524288])
    cos, sin = rope_tables(pos, 64, 1e7, interleave)
    x = torch.randn(3, 5, 64)
    got = apply_rope_host(x, cos, sin, interleave)
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2).float() / 64))
    ang = torch.outer(pos.float(), inv)
    z = torch.polar(torch.ones_like(ang), ang)
    if interleave:
        xc = torch.view_as_complex(x.reshape(3, 5, 32, 2).contiguous())
        ref = torch.view_as_real(xc * z).reshape(3, 5, 64)
    else:
        xc = torch.complex(x[..., :32], x[..., 32:])
        r = xc * z
        ref = torch.cat([r.real, r.imag], -1)
    assert torch.allclose(got, ref, atol=1e-5)
    P = rotation_matrix(64, interleave)
    assert torch.equal(P @ P, -torch.eye(64))  # a rotation by 90 degrees squared = -1


def small_cfg():
    cfg = Hy4Config()
    return dataclasses.replace(cfg, hidden_size=128, q_lora_rank=64, num_attention_heads=16, index_n_heads=4)


def rand_sd(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.05
    H, D = cfg.num_attention_heads, cfg.hidden_size
    return {
        "self_attn.q_a_proj.weight": r(cfg.q_lora_rank, D) * 4,
        "self_attn.q_a_layernorm.weight": 1 + r(cfg.q_lora_rank),
        "self_attn.q_b_proj.weight": r(cfg.q_b_out, cfg.q_lora_rank) * 2,
        "self_attn.kv_a_proj_with_mqa.weight": r(cfg.kv_a_out, D) * 4,
        "self_attn.kv_a_layernorm.weight": 1 + r(cfg.kv_lora_rank),
        "self_attn.kv_b_proj.weight": r(cfg.kv_b_out, cfg.kv_lora_rank),
        "self_attn.linear_gate.weight": r(cfg.attn_out_dim, D),
        "self_attn.o_proj.weight": r(D, cfg.attn_out_dim),
        "self_attn.learnable_sink_param": r(H) * 20,
    }


@pytest.mark.parametrize("interleave", [True, False])
@pytest.mark.parametrize("sparse", [False, True])
def test_absorbed_dataflow_equals_naive(interleave, sparse):
    cfg = small_cfg()
    S = 48
    sd = rand_sd(cfg)
    h = torch.randn(S, cfg.hidden_size)
    allowed = torch.ones(S, S, dtype=torch.bool).tril()
    if sparse:  # keep a random 20 of the causal keys of each row (always the diagonal)
        g = torch.Generator().manual_seed(3)
        keep = torch.rand(S, S, generator=g).argsort(-1)[:, :20]
        sel = torch.zeros(S, S, dtype=torch.bool).scatter_(1, keep, True) | torch.eye(S, dtype=torch.bool)
        allowed = allowed & sel
    ref, _ = attention_naive(cfg, sd, h, allowed, interleave)
    got, inter = attention_dataflow(cfg, sd, h, allowed, interleave)
    assert (ref - got).abs().max() < 1e-3 * ref.abs().max(), (ref - got).abs().max()
    assert torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1] > 0.999999


def test_sink_changes_output_and_domain():
    cfg = small_cfg()
    S = 16
    sd = rand_sd(cfg)
    h = torch.randn(S, cfg.hidden_size)
    allowed = torch.ones(S, S, dtype=torch.bool).tril()
    a, _ = attention_naive(cfg, sd, h, allowed)
    sd2 = dict(sd)
    sd2["self_attn.learnable_sink_param"] = torch.full_like(
        sd["self_attn.learnable_sink_param"], -30.0
    )  # sink ~ absent
    b, _ = attention_naive(cfg, sd2, h, allowed)
    assert (a - b).abs().max() > 1e-4


def test_kv_row_geometry():
    assert KV.row_bytes("bf16") == 1152
    assert KV.row_bytes("scaled_fp8", aligned=False) == 656
    assert KV.row_bytes("scaled_fp8") == 704  # kv-design finding: 64 B DRAM alignment
    kv = torch.randn(40, 576).to(torch.bfloat16)
    packed, deq = KV.pack_scaled_fp8_host(kv)
    assert packed.shape == (40, 656) and deq.shape == (40, 576)
    assert torch.equal(deq[:, 512:], kv[:, 512:])  # rope part is stored in bf16
    rel = (deq[:, :512].float() - kv[:, :512].float()).abs().max() / kv.float().abs().max()
    assert rel < 0.07
