# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: TtPrefillRuntime._build_mtp_predictor hands the predictor the trunk's MoE (config.moe_impl).

The trunk and the MTP layer must run the same MoE: a packed forward builds one kind of padding for both
(MoEBlockPacking for the MoE block, PackedPadding for TtMoe). The device tests build their predictors directly, so
this pins the production builder. No device: weights, the MTP config and the predictor are stand-ins.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.demos.deepseek_v3_d_p.tt import tt_prefill_runtime
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import MTP_CACHE_ENV


@pytest.mark.parametrize("moe_impl", ["block", "ttmoe", None])
def test_build_mtp_predictor_passes_the_trunks_moe_impl(monkeypatch, tmp_path, moe_impl):
    built = {}

    class _Predictor:
        def __init__(self, *args, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(tt_prefill_runtime, "TtMTPPredictor", _Predictor)
    mtp_cfg = SimpleNamespace(mtp_layer_idx=78, index_share_for_mtp_iteration=None)
    monkeypatch.setattr(
        tt_prefill_runtime, "MTPConfig", SimpleNamespace(from_hf_config=lambda hf_config, num_levels: mtp_cfg)
    )
    monkeypatch.setattr(
        tt_prefill_runtime.weight_categories, "load_mtp_state_dict", lambda *args: {"mtp": {}, "layer": {}}
    )
    for var in ("GLM53_HF_MODEL", "PREFILL_HF_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(MTP_CACHE_ENV, str(tmp_path / "mtp_cache"))  # (empty: the builder takes the build path)

    runtime = tt_prefill_runtime.TtPrefillRuntime.__new__(tt_prefill_runtime.TtPrefillRuntime)
    runtime.mesh_device = None
    runtime.hf_config = Mock()
    runtime.config = Mock(
        mtp_levels=1,
        mtp_ack_levels=0,
        num_layers=78,
        mesh_shape=(8, 4),
        model_cfg=SimpleNamespace(NUM_ROUTED_EXPERTS=256),
        weight_cache_path=str(tmp_path / "cache" / "glm_5_3_bh_32dev" / "8x4"),
        hf_model_path="unused",
        moe_impl=moe_impl,
    )
    runtime._can_build_missing_cache = lambda: True

    runtime._build_mtp_predictor()

    assert "moe_impl" in built, "TtPrefillRuntime._build_mtp_predictor must pass moe_impl to TtMTPPredictor"
    assert built["moe_impl"] == moe_impl
