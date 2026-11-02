# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Selection-function sweep: forces the head-group size (and the R2 gate) on the small-token rows so the
candidates can be ranked by device ns. Off unless QKV_SWEEP=1; run with scripts/run_safe_pytest.sh --profile."""

import os

import pytest

import ttnn.operations.qkv_proj_heads_rope.qkv_proj_heads_rope_program_descriptor as pd
from ttnn.operations.qkv_proj_heads_rope import qkv_proj_heads_rope

from .test_qkv_proj_heads_rope import check_outputs, device_inputs, make_inputs, reference

pytestmark = pytest.mark.skipif(os.environ.get("QKV_SWEEP") != "1", reason="perf sweep, set QKV_SWEEP=1")

ROWS = [
    tuple(int(v) for v in r.split("x")) for r in os.environ.get("QKV_SWEEP_ROWS", "4x512,8x256,2x512,4x256").split(",")
]
HEADS = [int(h) for h in os.environ.get("QKV_SWEEP_HEADS", "1,2,3,4,6").split(",")]


@pytest.mark.parametrize("heads_per_core", HEADS)
@pytest.mark.parametrize("batch, seq", ROWS, ids=[f"{b}x{s}" for b, s in ROWS])
def test_sweep(device, monkeypatch, batch, seq, heads_per_core):
    monkeypatch.setattr(pd, "HEADS_PER_CORE_CANDIDATES", (heads_per_core,))
    if "QKV_SWEEP_WMIN" in os.environ:
        monkeypatch.setattr(pd, "W_MCAST_MIN_GROUP", int(os.environ["QKV_SWEEP_WMIN"]))
    x, w, b, cos, sin = make_inputs(batch, seq, seq)
    tensors = device_inputs(device, x, w, b, cos, sin)
    outputs = qkv_proj_heads_rope(*tensors)
    check_outputs(outputs, reference(x, w, b, cos, sin))
