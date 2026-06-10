# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
E2E MLA layer test for DeepSeek V3.2.

Reuses the v3 test harness (run_model -> run_mla_inference) and swaps in the
v32 ttMLA via monkeypatch, so the only code under test that differs from v3 is
models/demos/deepseek_v32/tt/mla. While the v32 MLA is a passthrough this must
match v3 PCC exactly; it exists to surface integration gaps as DSA lands.

Uses the deepseek_v3_d_p weight variant until V3.2 checkpoints (indexer
weights) are wired into the conftest.
"""

import pytest
from ttnn.device import is_blackhole

import models.demos.deepseek_v3_d_p.tests.test_mla as v3_test_mla
import ttnn
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from models.demos.deepseek_v32.tt.mla import ttMLA as ttMLAv32


@pytest.fixture(autouse=True)
def use_v32_mla(monkeypatch):
    monkeypatch.setattr(v3_test_mla, "ttMLA", ttMLAv32)


# Single config (vs the full v3 sweep) — enough to exercise the e2e path.
@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        }
    ],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained", [False], ids=["random"])
@pytest.mark.parametrize("scale_down_sl", [True], ids=["scaled_sl"])
@pytest.mark.parametrize("seq_len", [128 * 1024], ids=["seq128k"])
@pytest.mark.parametrize("skip_host_comparison", [False], ids=["check_pcc"])
@pytest.mark.parametrize("is_balanced", [False], ids=["sequential"])
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.timeout(0)
def test_v32_mla(
    use_pretrained,
    request,
    mesh_device,
    seq_len,
    skip_host_comparison,
    scale_down_sl,
    is_balanced,
    is_ci_env,
    is_ci_v2_env,
    device_params,
    variant,
):
    v3_test_mla.run_model(
        variant,
        use_pretrained,
        request,
        mesh_device,
        seq_len,
        skip_host_comparison,
        scale_down_sl,
        is_balanced,
        is_ci_env,
        is_ci_v2_env,
        device_params,
    )
