# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Hy4 prefill iHC wiring (#65891): the ``ihc`` weight category and its cache files."""

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config
from models.demos.deepseek_v3_d_p.tt.hy4 import ihc as hy4_ihc
from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc


def _weights(seed):
    g = torch.Generator().manual_seed(seed)
    site = lambda: {
        "fn": torch.randn(8, 4 * 64, generator=g),
        "base": torch.randn(8, generator=g),
        "scale": torch.randn(2),
    }  # noqa: E731
    return {"attn": site(), "ffn": site()}


def test_cache_round_trip(tmp_path, expect_error):
    w = _weights(0)
    f = hy4_ihc.layer_cache_file(tmp_path, 3)
    assert not hy4_ihc.layer_cache_complete(tmp_path, 3)
    got = hy4_ihc.resolve_weights(w, f, "layer 3")  # writes
    assert hy4_ihc.layer_cache_complete(tmp_path, 3)
    back = hy4_ihc.resolve_weights(None, f, "layer 3")  # reads
    for site in ("attn", "ffn"):
        for k in ("fn", "base", "scale"):
            assert back[site][k].dtype == torch.float32
            assert torch.equal(back[site][k], got[site][k]) and torch.equal(back[site][k], w[site][k].float())
    with expect_error(FileNotFoundError, "ihc"):
        hy4_ihc.resolve_weights(None, hy4_ihc.layer_cache_file(tmp_path, 4), "layer 4")


def _missing(tmp_path, model_cfg, kv_only):
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE

    return wc.missing_trunk_categories(
        tmp_path,
        num_layers=3,
        first_layer_idx=0,
        experts_per_chip=8,
        model_cfg=model_cfg,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        is_first_rank=True,
        is_last_rank=True,
        mtp_levels=0,
        kv_only_last_layer=kv_only,
    )


@pytest.mark.parametrize("kv_only", [True, False], ids=["kv-only-last", "full-last"])
def test_ihc_category(tmp_path, kv_only):
    pytest.importorskip("ttnn")
    m = _missing(tmp_path, Hy4Config, kv_only)
    # Every layer (a kv-only last one too: its KV comes from the collapsed streams); the head only when the last
    # layer produces streams to collapse.
    assert m[wc.IHC] == ([0, 1, 2] if kv_only else [-1, 0, 1, 2])
    hy4_ihc.resolve_weights(_weights(1), hy4_ihc.layer_cache_file(tmp_path, 1), "layer 1")
    hy4_ihc.resolve_weights(
        {"fn": torch.ones(4, 256), "base": torch.ones(4), "scale": torch.ones(1)},
        hy4_ihc.head_cache_file(tmp_path),
        "head",
    )
    assert _missing(tmp_path, Hy4Config, kv_only)[wc.IHC] == [0, 2]
    assert wc.IHC not in _missing(tmp_path, GLM53Config, kv_only)  # models without iHC never ask for it


def test_ihc_config_from_model_config():
    import types

    cfg = hy4_ihc.ihc_config(types.SimpleNamespace(hidden_size=6144, rms_norm_eps=1e-5), Hy4Config)
    assert (cfg.dim, cfg.n, cfg.eps, cfg.norm_eps, cfg.magnitude) == (6144, 4, 1e-6, 1e-5, 2.0)
    assert hy4_ihc.ihc_enabled(Hy4Config) and not hy4_ihc.ihc_enabled(GLM53Config)
    keys = hy4_ihc.layer_ihc_keys(7)
    assert keys["attn"]["fn"] == "model.layers.7.hc_attn_layer.hc_pre.hc_fn"
    assert keys["ffn"]["scale"] == "model.layers.7.hc_mlp_layer.hc_pre.hc_scale"
    assert hy4_ihc.HEAD_KEYS["base"] == "model.hc_head.hc_head_base"


def test_balanced_reorder_of_stream_snapshots():
    """Balanced (zigzag) SP: TtPrefillTransformer restores natural order on its host snapshots. iHC stream snapshots are
    [S, n, C] (sequence on dim -3), so reordering them there must equal reordering each stream on its own sequence."""
    from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reverse_reorder_tensor_chunks

    order = create_balanced_chunk_order(8)
    streams = torch.randn(256, 4, 32)
    got = hy4_ihc.unbalance_streams(streams, order)
    for j in range(4):
        assert torch.equal(got[:, j], reverse_reorder_tensor_chunks(streams[:, j], order, seq_dim=-2))


def test_without_ihc_and_residual_layout():
    import ttnn

    noihc = hy4_ihc.without_ihc(Hy4Config)
    assert not hy4_ihc.ihc_enabled(noihc) and issubclass(noihc, Hy4Config) and hy4_ihc.ihc_enabled(Hy4Config)
    assert hy4_ihc.without_ihc(GLM53Config) is GLM53Config
    assert hy4_ihc.residual_layout(Hy4Config) == (4, ttnn.float32)
    assert hy4_ihc.residual_layout(GLM53Config) == (1, ttnn.bfloat16)


def test_d2d_activation_width():
    from models.demos.common.prefill.runners.runner_utils import d2d_activation_width

    assert d2d_activation_width(6144) == 6144
    assert d2d_activation_width(6144, dflash=True) == 2 * 6144
    assert d2d_activation_width(6144, streams=4) == 4 * 6144
