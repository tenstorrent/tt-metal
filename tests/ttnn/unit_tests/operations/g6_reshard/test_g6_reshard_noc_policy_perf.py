# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B of g6_reshard's knobs over the acceptance placements, for profiling.

Run under `scripts/run_safe_pytest.sh --profile --no-py-zones`; each call is
one GenericOpDeviceOperation row in the ops CSV, in parametrize order. The
output is checked bit-exact so a knob setting that breaks the move fails here.
"""

import pytest
import torch

import ttnn.operations.g6_reshard.g6_reshard_program_descriptor as pd

from tests.ttnn.unit_tests.operations.g6_reshard.test_g6_reshard import _check_call, _configs, _make_input
from ttnn.operations.g6_reshard import g6_reshard

PLACEMENTS = ["P", "I0", "I3", "I5", "O1"]
SETTINGS = {
    "fixed2": {"NOC_POLICY": "fixed", "ISSUERS_PER_CORE": 2},
    "wrapfree2": {"NOC_POLICY": "wrap_free", "ISSUERS_PER_CORE": 2},
    "wrapfree1": {"NOC_POLICY": "wrap_free", "ISSUERS_PER_CORE": 1},
    "balanced2": {"NOC_POLICY": "balanced", "ISSUERS_PER_CORE": 2},
}


@pytest.mark.parametrize("name", PLACEMENTS)
@pytest.mark.parametrize("setting", list(SETTINGS))
def test_knob_setting(device, monkeypatch, setting, name):
    for knob, value in SETTINGS[setting].items():
        monkeypatch.setattr(pd, knob, value)
    torch.manual_seed(42)
    in_cfg, out_cfg = _configs(device, name)
    x, x_host = _make_input(device, in_cfg)
    out = g6_reshard(x, out_cfg)
    _check_call(out, x_host, out_cfg)
