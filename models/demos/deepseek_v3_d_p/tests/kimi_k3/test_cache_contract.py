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

from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
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
