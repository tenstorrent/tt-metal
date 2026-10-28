# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tracy profile of one warm Tafsut forward reduced to two layers (real weights, the real bench shape).

The reduced forward runs the embedding, one ordinary layer (counted 13 times), the last layer with the read-out cut
(counted once) and the head. The profiler is drained after every section (F7: ~1000 programs; never the full
``long_context`` forward). Signposts (``tt-perf-report --start-signpost``): MODEL_START / MODEL_END around the whole
reduced forward, and ``<section>_start`` / ``<section>_end`` for embed, layer, layer_last, head. Runs eagerly; the
bench runs the captured trace.

    MODEL_PROFILE_SHAPE      paper (default) | long_context | long_horizon | gift_ragged | gift_ragged_odd
    MODEL_PROFILE_PRECISION  accuracy (default) | performance
    MODEL_PROFILE_OUT        directory for layer_counts.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from models.experimental.tafsut_forecast.tests.golden.goldens import load_golden

COUNTS = {"embed": 1, "layer": 13, "layer_last": 1, "head": 1}


@pytest.mark.timeout(1800)
def test_profile_forward(device):
    import ttnn
    from tracy import signpost

    from models.experimental.tafsut_forecast.reference import dataflow as df
    from models.experimental.tafsut_forecast.tt.common import Precision
    from models.experimental.tafsut_forecast.tt.model import TtTafsut

    shape = os.environ.get("MODEL_PROFILE_SHAPE", "paper")
    precision = os.environ.get("MODEL_PROFILE_PRECISION", "accuracy")
    g = load_golden(shape)
    tt = TtTafsut(device, precision=Precision.by_name(precision), n_layers=2)
    prep = tt.prepare_inputs(g["x"])
    c = tt.constants(prep.tokens.shape[0], prep.ctx_len)
    inp = tt.upload_inputs(tt.host_inputs(prep))

    def forward(marked: bool):
        def mark(name):
            if marked:
                ttnn.synchronize_device(device)
                signpost(name)

        mark("embed_start")
        h = tt.embed(inp["vm"], c)
        x = tt.norm(h, tt.layers[0]["ln1"])
        mark("embed_end")
        ttnn.ReadDeviceProfiler(device)
        mark("layer_start")
        h, x = tt.layer(h, x, tt.layers[0], inp, c, last=False, g_next=tt.layers[1]["ln1"])
        mark("layer_end")
        ttnn.ReadDeviceProfiler(device)
        mark("layer_last_start")
        h, x = tt.layer(h, x, tt.layers[1], inp, c, last=True, g_next=tt.final_ln)
        mark("layer_last_end")
        ttnn.ReadDeviceProfiler(device)
        mark("head_start")
        out = tt.head(x)
        mark("head_end")
        ttnn.deallocate(h)
        ttnn.deallocate(x)
        return out

    ttnn.deallocate(forward(marked=False))  # warm-up: compile every program
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    signpost("MODEL_START")
    out = forward(marked=True)
    ttnn.synchronize_device(device)
    signpost("MODEL_END")
    ttnn.ReadDeviceProfiler(device)
    ttnn.deallocate(out)
    tt.deallocate_inputs(inp)

    meta = {
        "shape": shape,
        "precision": precision,
        "test": "forward",
        "window": ["MODEL_START", "MODEL_END"],
        "rows": int(g["x"].shape[0]),
        "bucket_context": prep.ctx_len,
        "s_pad": c["s_pad"],
        "sections": list(COUNTS),
        "counts": COUNTS,
        "layers": df.NL,
    }
    print(f"\n[PROFILE] {json.dumps(meta)}")
    out_dir = os.environ.get("MODEL_PROFILE_OUT")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "layer_counts.json").write_text(json.dumps(meta, indent=2) + "\n")
