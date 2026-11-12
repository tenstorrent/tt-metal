# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for MuseGlimmerModelArgs (no device)."""

import math
import os

import pytest
import torch

from models.demos.muse_glimmer_30b.tt.model_config import MuseGlimmerModelArgs, checkpoint_dir

CKPT = checkpoint_dir()

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "config.json")), reason="checkpoint not available"
)


def _args(tp, **kw):
    return MuseGlimmerModelArgs.from_checkpoint(CKPT, tp=tp, **kw)


def test_tp4():
    a = _args(4)
    assert (a.q_heads_local, a.kv_heads_local, a.kv_replicated) == (8, 1, True)
    assert [a.kv_head_for_rank(r) for r in range(4)] == [0, 0, 1, 1]
    assert a.qkvg_width_local == 2304
    assert a.intermediate_local == 4992
    assert a.intermediate_local_padded == 5120
    assert a.hidden_local == 1664
    assert a.vocab_local == 50688


def test_tp8():
    a = _args(8)
    assert [a.kv_head_for_rank(r) for r in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert a.qkvg_width_local == 1280
    assert a.intermediate_local == 2496
    assert a.intermediate_local_padded == 2560


def test_tp2_tp1():
    a = _args(2)
    assert a.kv_heads_local == 1 and not a.kv_replicated and a.qkvg_width_local == 4352
    assert a.intermediate_local_padded == 10240
    a = _args(1)
    assert a.kv_heads_local == 2 and a.qkvg_width_local == 8704
    assert a.intermediate_local_padded == 19968  # already a multiple of 512


def test_misc_derived():
    a = _args(4)
    assert a.padded_vocab_size == 202752
    assert a.sdpa_scale == pytest.approx(3.87 / math.sqrt(128))
    for i in (3, 51):
        assert not a.is_sliding(i) and not a.uses_rope(i)
    for i in (0, 1, 2, 50):
        assert a.is_sliding(i) and a.uses_rope(i)
    assert a.eos_token_ids == [200001, 200008]
    assert a.max_num_blocks_per_user == 128 and a.max_num_blocks == 32 * 128
    assert _args(4, num_layers=2).n_layers == 2


def test_reference_config_truncation():
    cfg = _args(4, num_layers=4).reference_config()
    assert cfg.num_hidden_layers == 4 and len(cfg.layer_types) == 4 and len(cfg.layer_rope_theta) == 4


def test_load_weights():
    a = _args(4)
    sd = a.load_layer_state_dict(3)
    expected = {
        "self_attn.q_proj.weight": (4096, 6656),
        "self_attn.k_proj.weight": (256, 6656),
        "self_attn.v_proj.weight": (256, 6656),
        "self_attn.gate_proj.weight": (4096, 6656),
        "self_attn.o_proj.weight": (6656, 4096),
        "mlp.gate_proj.weight": (19968, 6656),
        "mlp.up_proj.weight": (19968, 6656),
        "mlp.down_proj.weight": (6656, 19968),
        "input_layernorm.weight": (6656,),
        "post_attention_layernorm.weight": (6656,),
        "pre_feedforward_layernorm.weight": (6656,),
        "post_feedforward_layernorm.weight": (6656,),
    }
    assert set(sd) == set(expected)
    for k, shp in expected.items():
        assert tuple(sd[k].shape) == shp, k
        assert sd[k].dtype == torch.bfloat16
    assert tuple(a.load_lm_head_weight().shape) == (202048, 6656)
    assert tuple(a.load_final_norm_weight().shape) == (6656,)


def test_weight_cache_dir(tmp_path):
    import ttnn

    a = _args(4, cache_dir=str(tmp_path))
    assert a.weight_cache_dir() == tmp_path / "tp4"
    assert (tmp_path / "tp4").is_dir()
    assert a.cache_file("wqkvg", ttnn.bfloat8_b, tag="v1") == str(tmp_path / "tp4" / "wqkvg_v1_bfp8")
    assert a.cache_file("embed", ttnn.bfloat16) == str(tmp_path / "tp4" / "embed_bf16")
