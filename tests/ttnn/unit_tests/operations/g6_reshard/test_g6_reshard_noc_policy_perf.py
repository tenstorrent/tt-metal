# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B of g6_reshard's knobs over the acceptance placements, for profiling.

Run under `scripts/run_safe_pytest.sh --profile --no-py-zones`; each call is
one GenericOpDeviceOperation row in the ops CSV, in parametrize order. The
output is checked bit-exact so a knob setting that breaks the move fails here.
"""

import os
from pathlib import Path

import pytest
import torch

import ttnn.operations.g6_reshard.g6_reshard_program_descriptor as pd

from tests.ttnn.unit_tests.operations.g6_reshard.test_g6_reshard import _check_call, _configs, _make_input
from ttnn.operations.g6_reshard import g6_reshard

_issuer_segments = pd.issuer_segments


def _halved_same_noc(rects, noc, cut):
    """Ablation: every uncut one-rectangle issuer reads its rows as two segments on one NoC (link loads unchanged)."""
    if cut is not None or len(rects) != 1 or rects[0][1] - rects[0][0] < 2:
        return _issuer_segments(rects, noc, cut)
    r0, r1, c0, c1 = rects[0]
    mid = (r0 + r1) // 2
    return [(r0, mid, c0, c1, noc), (mid, r1, c0, c1, noc)]


# Probe kernels: the shipped kernel with extra zones or one stage ablated (perf only).
PROBE_KERNELS = Path(__file__).parent / "probes" / "kernels"

PLACEMENTS = os.environ.get("G6_PERF_PLACEMENTS", "P,I0,I3,I5,O1").split(",")
SETTINGS = {
    "fixed2": {"NOC_POLICY": "fixed", "ISSUERS_PER_CORE": 2},
    "wrapfree2": {"NOC_POLICY": "wrap_free", "ISSUERS_PER_CORE": 2},
    "wrapfree1": {"NOC_POLICY": "wrap_free", "ISSUERS_PER_CORE": 1},
    "balanced2": {"NOC_POLICY": "balanced", "ISSUERS_PER_CORE": 2},
    "split_issuer2": {"NOC_POLICY": "split", "SPLIT_SHAPE": "issuer", "ISSUERS_PER_CORE": 2},
    "split_cut2": {"NOC_POLICY": "split", "SPLIT_SHAPE": "cut", "ISSUERS_PER_CORE": 2},
    "split_issuer2_w2": {
        "NOC_POLICY": "split",
        "SPLIT_SHAPE": "issuer",
        "ISSUERS_PER_CORE": 2,
        "SPLIT_WRAP_WEIGHT": 2.0,
    },
    "split_issuer2_halved": {
        "NOC_POLICY": "split",
        "SPLIT_SHAPE": "issuer",
        "ISSUERS_PER_CORE": 2,
        "issuer_segments": _halved_same_noc,
    },
    "split_cut2_w2": {"NOC_POLICY": "split", "SPLIT_SHAPE": "cut", "ISSUERS_PER_CORE": 2, "SPLIT_WRAP_WEIGHT": 2.0},
    "zoned": {"ISSUER_KERNEL": PROBE_KERNELS / "reader_zoned.cpp"},
    "zoned_stateful": {"ISSUER_KERNEL": PROBE_KERNELS / "reader_zoned.cpp", "STATEFUL_READS": True},
    # Refinement 2 levers on the shipped NoC plan; "default" is the module's own settings.
    "default": {},
    "stateful": {"STATEFUL_READS": True},
    "runs_lag1": {"ISSUER_SPLIT": "runs", "ISSUER_LAG_RUNS": (0.0, 1.0)},
    "runs_lag2": {"ISSUER_SPLIT": "runs", "ISSUER_LAG_RUNS": (0.0, 2.0)},
    "runs_lag3": {"ISSUER_SPLIT": "runs", "ISSUER_LAG_RUNS": (0.0, 3.0)},
    "stagger": {"PIECE_ORDER": "stagger"},
    "local_last": {"PIECE_ORDER": "local_last"},
    "balanced": {"ISSUER_SPLIT": "balanced"},
    "st_balanced": {"STATEFUL_READS": True, "ISSUER_SPLIT": "balanced"},
    "st_stagger": {"STATEFUL_READS": True, "PIECE_ORDER": "stagger"},
    "st_local_last": {"STATEFUL_READS": True, "PIECE_ORDER": "local_last"},
    "st_cut": {"STATEFUL_READS": True, "SPLIT_SHAPE": "cut"},
    "st_bal_stagger": {"STATEFUL_READS": True, "ISSUER_SPLIT": "balanced", "PIECE_ORDER": "stagger"},
}


# Launch skew moves one call's duration by a few percent: repeats give a median.
REPEATS = int(os.environ.get("G6_PERF_REPEATS", "1"))
SETTING_NAMES = os.environ.get("G6_PERF_SETTINGS", ",".join(SETTINGS)).split(",")


@pytest.mark.parametrize("rep", range(REPEATS))
@pytest.mark.parametrize("name", PLACEMENTS)
@pytest.mark.parametrize("setting", SETTING_NAMES)
def test_knob_setting(device, monkeypatch, setting, name, rep):
    for knob, value in SETTINGS[setting].items():
        monkeypatch.setattr(pd, knob, value)
    torch.manual_seed(42)
    in_cfg, out_cfg = _configs(device, name)
    x, x_host = _make_input(device, in_cfg)
    out = g6_reshard(x, out_cfg)
    _check_call(out, x_host, out_cfg)
