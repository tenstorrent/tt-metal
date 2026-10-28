# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced-layer Tracy profile of one eager forward_device at the target shape (paper: 512 rows, both flip halves).

    KAIROS_PROFILE_OUT=<iter_dir> python -m tracy -r -p --no-web-server -o <iter_dir>/tracy \\
        -m pytest models/experimental/kairos_forecast/tests/perf/test_profile_forward.py
Signposts: KAIROS_START / KAIROS_END around the forward, and one per section (embed, enc_block, enc_norm, dec_block,
dec_norm, head). layer_counts.json gives the profiled and the full counts for the estimate (summarize_profile.py).
PROFILE_LAYERS (default 2+2), PROFILE_PRECISION (default accuracy).
"""
import json
import os
from pathlib import Path

from models.experimental.kairos_forecast.reference import dataflow as DF
from models.experimental.kairos_forecast.tests.perf.suite import load
from models.experimental.kairos_forecast.tt.common import Precision
from models.experimental.kairos_forecast.tt.generator import _cat_fronts
from models.experimental.kairos_forecast.tt.host_frontend import front_end
from models.experimental.kairos_forecast.tt.model import TtKairos
from models.experimental.kairos_forecast.tt.router_state import RouterState

LAYERS = tuple(int(x) for x in os.environ.get("PROFILE_LAYERS", "2+2").split("+"))
PRECISION = os.environ.get("PROFILE_PRECISION", "accuracy")
SHAPE = os.environ.get("PROFILE_SHAPE", "paper")


def test_profile_forward(device):
    import ttnn
    from tracy import signpost

    W = DF.load_weights()
    m = TtKairos(device, W, getattr(Precision, PRECISION)(), layers=LAYERS)
    rec = load(SHAPE)
    router = RouterState(rec["bias0"])
    fa, _ = front_end(rec["context"], router, W)
    fb, _ = front_end(-rec["context"], router, W)
    inp = m.prepare(_cat_fronts(fa, fb))
    for _ in range(2):  # warm-up: compile, and drain the profiler (pitfall 7)
        m.forward_device(inp)
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    signpost("KAIROS_START")
    m.forward_device(inp, mark=signpost)
    ttnn.synchronize_device(device)
    signpost("KAIROS_END")
    ttnn.ReadDeviceProfiler(device)
    out = os.environ.get("KAIROS_PROFILE_OUT")
    if out:
        meta = {"shape": SHAPE, "precision": PRECISION, "rows": int(inp["B"]), "layer_counts": {
            "embed": 1, "enc_block": [LAYERS[0], 6], "enc_norm": 1, "dec_block": [LAYERS[1], 6], "dec_norm": 1,
            "head": 1}}
        (Path(out) / "layer_counts.json").write_text(json.dumps(meta, indent=1) + "\n")
