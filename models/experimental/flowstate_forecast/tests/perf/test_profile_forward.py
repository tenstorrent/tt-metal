# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tracy profile of one warm FlowState forward reduced to one ordinary layer and the last layer.

Real weights and the real bench shape. Layers 0-4 run the same programs on the same shapes, so the profile runs
layer 0 once and counts it 5 times (``layer_counts.json``); the last layer (cut to the CPM-read rows) once.
Signposts (``tt-perf-report --start-signpost``): MODEL_START / MODEL_END; embed, layer_mid, layer_last, decoder
(``<name>_start`` / ``<name>_end``). The profiled forward runs eagerly; the bench runs the captured trace.
    MODEL_PROFILE_SHAPE      paper (default) | any bench shape
    MODEL_PROFILE_PRECISION  accuracy (default) | performance
    MODEL_PROFILE_OUT        directory for layer_counts.json
"""

import json
import os
from pathlib import Path

import torch
import ttnn

from models.experimental.flowstate_forecast.reference import weights
from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS
from models.experimental.flowstate_forecast.tt.model import FlowStateTT

COUNTS = {"embed": 1, "layer_mid": 5, "layer_last": 1, "decoder": 1}


def test_profile_forward(device):
    from tracy import signpost

    shape = os.environ.get("MODEL_PROFILE_SHAPE", "paper")
    precision = os.environ.get("MODEL_PROFILE_PRECISION", "accuracy")
    g = torch.load(GOLDENS / f"bench_{shape}.pt", weights_only=True)
    tt = FlowStateTT(weights.load(), device, precision)
    tok, meta = tt.prepare(g["inputs"].clone(), g["s"], g["horizon"])
    s, L, Bp, r0, P = g["s"], meta["L"], meta["Bp"], meta["r0"], meta["P"]
    per_layer, (dw, db) = tt.constants(s, L, P)
    p = tt.p

    def forward(mark):
        def zone(name, fn):
            if mark:
                signpost(f"{name}_start")
            out = fn()
            if mark:
                ttnn.synchronize_device(device)
                signpost(f"{name}_end")
            return out

        h = zone(
            "embed",
            lambda: ttnn.linear(tok, tt.embed_w, bias=tt.embed_b, dtype=ttnn.bfloat16, compute_kernel_config=p.embed),
        )
        h = zone("layer_mid", lambda: tt.layer(0, h, per_layer[0], L, Bp, 0))
        h = zone("layer_last", lambda: tt.layer(5, h, per_layer[5], L, Bp, r0))
        return zone("decoder", lambda: ttnn.linear(h, dw, bias=db, dtype=ttnn.float32, compute_kernel_config=p.decoder))

    forward(False)  # compile
    ttnn.synchronize_device(device)
    signpost("MODEL_START")
    forward(True)
    ttnn.synchronize_device(device)
    signpost("MODEL_END")
    ttnn.ReadDeviceProfiler(device)
    out = Path(os.environ.get("MODEL_PROFILE_OUT", "/tmp/flowstate_profile"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "layer_counts.json").write_text(
        json.dumps({"shape": shape, "precision": precision, "counts": COUNTS}, indent=1)
    )
