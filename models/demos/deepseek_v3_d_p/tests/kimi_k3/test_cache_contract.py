# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The weight-cache completeness check has to know Kimi-K3 has an output gate. Host only.

`ttMLA.weight_names` appends `g_proj` only when `has_output_gate` is set, and
`TtPrefillBlock.check_cache_complete` did not pass it. So a K3 MLA cache missing its output gate
reported COMPLETE, and the layer then loaded the `torch.empty` placeholder that
`_convert_and_cache_weights` hands `as_tensor` on a cache hit. The model runs. It is wrong. Nothing
raises.

That is the shape of every hazard in this path — a plausible model rather than an error — which is
why the check is worth a test even though the fix is one argument. The gate is read off `model_cfg`,
so every non-gated model's existing cache stays valid: only K3 sets `USE_OUTPUT_GATE`.
"""

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kda.config import KIMI_K3_PROJECTION_DTYPE
from models.demos.deepseek_v3_d_p.tt.kda.weights import PROJECTION_ARTIFACTS as KDA_PROJECTION_ARTIFACTS
from models.demos.deepseek_v3_d_p.tt.kimi_k3 import weights as kimi_k3_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import (
    layer_is_cached,
    layer_is_stale,
    load_layer_state_dict_cached,
    mark_layer_cached,
)
from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock


@pytest.mark.parametrize(
    "model_cfg, expected",
    [(KimiK3Config, True), (KimiK26Config, False), (None, False)],
    ids=["kimi_k3_gated", "kimi_k2_6_ungated", "no_model_cfg"],
)
def test_block_cache_check_forwards_the_output_gate(monkeypatch, tmp_path, model_cfg, expected):
    """K3 asks for `g_proj`; nothing else does, including a caller that passes no config at all."""
    seen = {}

    def _spy(cache_path, prefix, has_indexer=False, has_output_gate=False):
        seen["has_output_gate"] = has_output_gate
        return False  # short-circuit: this test is about the argument, not the files

    monkeypatch.setattr(ttMLA, "check_cache_complete", staticmethod(_spy))
    monkeypatch.setattr(
        "models.demos.deepseek_v3_d_p.tt.tt_prefill_block.TtDistributedRmsNorm.check_cache_complete",
        staticmethod(lambda *a, **k: True),
    )

    TtPrefillBlock.check_cache_complete(tmp_path, layer_idx=3, is_dense=True, model_cfg=model_cfg)
    assert seen["has_output_gate"] is expected


def test_g_proj_is_only_in_the_gated_name_list():
    """The other half of the contract: the name list is what makes the flag load-bearing."""
    assert "g_proj" not in ttMLA.weight_names(has_output_gate=False)
    assert "g_proj" in ttMLA.weight_names(has_output_gate=True)


# --- KDA projection dtype --------------------------------------------------------------------
#
# A layer marker written before the KDA projections moved to `KIMI_K3_PROJECTION_DTYPE` vouches for
# bfloat16 projection tensorbins that the layer no longer loads. Trusting it fails at the first KDA
# layer with FileNotFoundError; so does an interrupted rebuild that wrote only some of them.

_KDA_LAYER = 0
_MLA_LAYER = KimiK3Config.mla_layer_ids()[0]


def _write_projections(root, layer_idx, dtype, names=KDA_PROJECTION_ARTIFACTS):
    for name in names:
        (
            root / f"layer_{layer_idx}.kda.{name}.v2.0123456789abcdef.mesh8x4.tpaxis1"
            f"_dtype_{dtype.name}_layout_TILE.tensorbin"
        ).touch()


@pytest.mark.parametrize(
    "projection_dtypes, cached",
    [
        ((ttnn.bfloat16,), False),
        ((KIMI_K3_PROJECTION_DTYPE,), True),
        ((ttnn.bfloat16, KIMI_K3_PROJECTION_DTYPE), True),
    ],
    ids=["legacy_bf16", "migrated", "migrated_with_legacy_leftovers"],
)
def test_kda_layer_cache_requires_the_projection_dtype(tmp_path, projection_dtypes, cached):
    mark_layer_cached(tmp_path, _KDA_LAYER)
    for dtype in projection_dtypes:
        _write_projections(tmp_path, _KDA_LAYER, dtype)
    assert layer_is_cached(tmp_path, _KDA_LAYER) is cached
    assert layer_is_stale(tmp_path, _KDA_LAYER) is not cached


@pytest.mark.parametrize("missing", KDA_PROJECTION_ARTIFACTS)
def test_partially_migrated_kda_layer_is_stale(tmp_path, missing):
    mark_layer_cached(tmp_path, _KDA_LAYER)
    _write_projections(
        tmp_path, _KDA_LAYER, KIMI_K3_PROJECTION_DTYPE, [n for n in KDA_PROJECTION_ARTIFACTS if n != missing]
    )
    assert not layer_is_cached(tmp_path, _KDA_LAYER)
    assert layer_is_stale(tmp_path, _KDA_LAYER)


def test_unmarked_layer_is_neither_cached_nor_stale(tmp_path):
    _write_projections(tmp_path, _KDA_LAYER, KIMI_K3_PROJECTION_DTYPE)
    assert not layer_is_cached(tmp_path, _KDA_LAYER)
    assert not layer_is_stale(tmp_path, _KDA_LAYER)


def test_mla_layer_needs_no_kda_projections(tmp_path):
    mark_layer_cached(tmp_path, _MLA_LAYER)
    assert layer_is_cached(tmp_path, _MLA_LAYER)
    assert not layer_is_stale(tmp_path, _MLA_LAYER)


def test_runtime_preflight_rejects_a_legacy_kda_cache(tmp_path):
    """Production passes state_dict={}, so preflight is the only place a stale layer can be caught."""
    for idx in (_KDA_LAYER, _KDA_LAYER + 1):
        mark_layer_cached(tmp_path, idx)
        _write_projections(tmp_path, idx, KIMI_K3_PROJECTION_DTYPE)
    assert TtKimiK3Transformer.check_cache_complete(tmp_path, num_layers=2, model_cfg=KimiK3Config)
    for path in tmp_path.glob(f"layer_{_KDA_LAYER + 1}.kda.output_projection.*"):
        path.unlink()
    assert not TtKimiK3Transformer.check_cache_complete(tmp_path, num_layers=2, model_cfg=KimiK3Config)


def test_stale_layer_loads_only_its_kda_weights(monkeypatch, tmp_path):
    """Rebuild the projections from the attention weights, not the layer's ~59 GB of experts."""
    mark_layer_cached(tmp_path, _KDA_LAYER)
    _write_projections(tmp_path, _KDA_LAYER, ttnn.bfloat16)
    sentinel = {"q_proj": object()}
    monkeypatch.setattr(kimi_k3_weights, "load_kda_layer_state_dict", lambda *a, **k: sentinel)
    monkeypatch.setattr(
        kimi_k3_weights, "load_layer_state_dict", lambda *a, **k: pytest.fail("opened the full layer checkpoint")
    )
    assert load_layer_state_dict_cached(tmp_path, _KDA_LAYER, tmp_path) == {"kda_weights": sentinel}
    _write_projections(tmp_path, _KDA_LAYER, KIMI_K3_PROJECTION_DTYPE)
    assert load_layer_state_dict_cached(tmp_path, _KDA_LAYER, tmp_path) == {}
