# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of Hy4Config against the real checkpoint config.json (skipped when the checkpoint is not mounted) and synthetic edits."""

import copy
import json
import os

import pytest

from models.demos.blackhole.hy4_preview.config import DEFAULT_CKPT, DENSE, FULL, MOE, SHARED, Hy4Config, Hy4ConfigError

CKPT = os.environ.get("HY4_CKPT", DEFAULT_CKPT)
needs_ckpt = pytest.mark.skipif(not os.path.exists(os.path.join(CKPT, "config.json")), reason="checkpoint not mounted")


@pytest.fixture(scope="module")
def raw():
    with open(os.path.join(CKPT, "config.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cfg():
    return Hy4Config.from_checkpoint(CKPT)


@needs_ckpt
def test_core_dims(cfg):
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size) == (6144, 78, 120832)
    assert (cfg.intermediate_size, cfg.moe_intermediate_size) == (18432, 2048)
    assert (cfg.num_attention_heads, cfg.q_lora_rank, cfg.kv_lora_rank) == (64, 2048, 512)
    assert (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.qk_head_dim, cfg.v_head_dim) == (192, 64, 256, 256)
    assert (cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk) == (32, 128, 2048)
    assert cfg.hc_mult == 4 and cfg.hc_eps == 1e-6 and cfg.hc_magnitude == 2.0
    assert (cfg.n_routed_experts, cfg.n_shared_experts, cfg.num_experts_per_tok) == (256, 1, 8)
    assert cfg.routed_scaling_factor == 2.827 and cfg.swiglu_limit == 10.0 and cfg.norm_topk_prob
    assert cfg.rope_theta == 1e7 and cfg.rope_type == "default" and cfg.max_position_embeddings == 1048576
    assert cfg.num_nextn_predict_layers == 1 and cfg.enable_lm_head_fp32 and not cfg.tie_word_embeddings
    assert cfg.architecture == "HYV4ForCausalLM" and cfg.model_type == "hy_v4"
    assert (cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id) == (120000, 120025, 120002)


@needs_ckpt
def test_derived_tensor_dims_match_checkpoint_shapes(cfg):
    # shapes taken from the checkpoint headers (ref-oracle/shape_table.txt)
    assert cfg.q_b_out == 16384 and cfg.kv_a_out == 576 and cfg.kv_b_out == 28672
    assert cfg.attn_out_dim == 16384  # linear_gate [16384, 6144], o_proj [6144, 16384]
    assert cfg.indexer_q_out == 4096  # indexer.wq_b [4096, 2048]
    assert cfg.hc_fn_rows == 8 and cfg.hc_flat_dim == 24576  # hc_fn [8, 24576]


@needs_ckpt
def test_layer_tables(cfg):
    assert len(cfg.main_layers) == 78 and len(cfg.layer_table) == 79
    assert cfg.dense_layers == (0,)
    assert cfg.moe_layers == tuple(range(1, 78))
    full = cfg.full_indexer_layers
    assert full == (0, 1) + tuple(range(5, 78, 4))
    assert len(full) == 21
    assert cfg.layer(0).indexer == FULL and cfg.layer(0).mlp == DENSE
    assert cfg.layer(1).indexer == FULL and cfg.layer(1).mlp == MOE
    for i in (2, 3, 4):
        assert cfg.layer(i).indexer == SHARED and cfg.indexer_source(i) == 1
    for i in (6, 7, 8):
        assert cfg.indexer_source(i) == 5
    assert cfg.indexer_source(77) == 77
    # every shared layer points at an earlier full layer
    for l in cfg.main_layers:
        assert l.indexer_source <= l.index and cfg.layer(l.indexer_source).indexer == FULL
        assert l.has_hc


@needs_ckpt
def test_mtp_layer(cfg):
    (m,) = cfg.mtp_layers
    assert m.index == 78 and m.is_mtp and m.mlp == MOE and m.indexer == FULL and m.indexer_source == 78
    assert not m.has_hc  # no hc tensors under model.mtp_layers.0
    assert cfg.n_total_layers == 79


@needs_ckpt
def test_placement_and_shards(cfg):
    assert cfg.n_chips == 32 and cfg.experts_per_chip == 8
    owners = [cfg.expert_chip(e) for e in range(256)]
    assert owners[0] == 0 and owners[7] == 0 and owners[8] == 1 and owners[255] == 31
    assert all(owners.count(c) == 8 for c in range(32))
    assert cfg.expert_mesh_coord(8 * 9) == (1, 1)
    assert list(cfg.chip_experts(31)) == list(range(248, 256))
    with pytest.raises(IndexError):  # allow-pytest.raises: CPU tests run without the root conftest
        cfg.expert_chip(256)
    assert cfg.heads_per_col == 8 and cfg.hidden_per_col == 768
    assert cfg.shared_inter_per_col == 256 and cfg.tiles(cfg.shared_inter_per_col) == 8
    assert cfg.dense_inter_per_col == 2304 and cfg.tiles(cfg.dense_inter_per_col) == 72
    assert cfg.vocab_per_col == 15104 and cfg.tiles(cfg.vocab_per_col) == 472
    assert cfg.tiles(cfg.hidden_size) == 192 and cfg.tiles(cfg.moe_intermediate_size) == 64


@needs_ckpt
def test_roundtrip_is_deterministic(raw):
    a = Hy4Config.from_dict(raw)
    b = Hy4Config.from_dict(copy.deepcopy(raw))
    assert a == b and hash(a) == hash(b)


def _mut(raw, **kw):
    d = copy.deepcopy(raw)
    d.update(kw)
    return d


@needs_ckpt
@pytest.mark.parametrize(
    "edit",
    [
        dict(indexer_types=["full"] * 77),  # wrong length
        dict(mlp_layer_types=["dense"] * 77),
        dict(n_routed_experts=250),  # not divisible by 32
        dict(n_group=2),
        dict(gating_type="headwise"),
        dict(hc_mult=2),
        dict(tie_word_embeddings=True),
        dict(num_nextn_predict_layers=2),
        dict(qk_head_dim=128),
        dict(vocab_size=120833),
        dict(rope_parameters={"rope_theta": 1e7, "rope_type": "yarn"}),
    ],
)
def test_validation_rejects(raw, edit):
    with pytest.raises(Hy4ConfigError):  # allow-pytest.raises: CPU tests run without the root conftest
        Hy4Config.from_dict(_mut(raw, **edit))


@needs_ckpt
def test_first_layer_must_be_full(raw):
    t = list(raw["indexer_types"])
    t[0] = "shared"
    with pytest.raises(Hy4ConfigError):  # allow-pytest.raises: CPU tests run without the root conftest
        Hy4Config.from_dict(_mut(raw, indexer_types=t))
