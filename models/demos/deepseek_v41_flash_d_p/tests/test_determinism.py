# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Determinism (FIX.23, owner 2026-11-12): identical inputs give identical outputs, bit for bit.

FIX.25: the check itself is the framework's (testing/model_determinism.py: novel and code prompts interleaved, cold
first sequence after the load, warm ones after the other prompt, every chunk's final hidden and every layer's served
state bit for bit), which every model-level gate now runs (core/gate.py with_determinism). This file keeps the gates
that name it (P.49, FIX.25, FIX.26) and P.50's A/B metric named by the COMBINE_SPLIT side.

DS41-233 / DS41-234 / DS41-235: the MoE combine sums the routed terms in packet-arrival order; with sums above 32 the
single snapped fp32 sum depended on that order (fixed by COMBINE_SPLIT=on).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_determinism.py
"""

from __future__ import annotations

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.model_determinism import check
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()


@mesh_parametrize(timeout_s=2 * 3600)
def test_determinism(mesh_device):
    layers = S.layers()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    err, worst = check(S, model, layers, label=f"COMBINE_SPLIT={settings.get('COMBINE_SPLIT')}")
    # (P.50's A/B, owner 2026-11-12: the same metric named by the side, COMBINE_SPLIT off / on)
    metrics.record(f"determinism_max_abs_diff_{settings.get('COMBINE_SPLIT')}", worst)
    assert err is None, err
