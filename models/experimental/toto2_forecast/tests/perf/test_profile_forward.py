# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tracy profile of one warm Toto 2.0 forward, reduced to one real layer of each kind.

Profiling all 48 layers would flood the device profiler buffers, so the profile keeps the real patch embedding
and head plus time layer 0, the last time layer (cut to the read-out tokens after its attention) and the variate
layer, with real weights at the real benchmark shape. Signposts (``tt-perf-report --start-signpost/--end-signpost``):

    MODEL_START / MODEL_END                    the whole reduced forward
    embed_start / embed_end                    patch embedding
    layer_time_start / layer_time_end          time layer 0 (counts for layers 0 .. L-3)
    last_time_start / last_time_end            the last time layer, cut to the read-out tokens
    layer_variate_start / layer_variate_end    the variate layer
    head_start / head_end                      out_norm and the quantile head

``layer_counts.json`` (to ``$MODEL_PROFILE_OUT``) carries the counts; the full-model device time is estimated as
``sum(counts[s] * device(s))``. Run under ``python -m tracy -r -p -m pytest`` (``perf_iteration.sh``).

    MODEL_PROFILE_SHAPE      paper (default) | long_context | multivariate | paper_313m
    MODEL_PROFILE_PRECISION  accuracy (default) | performance
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from models.experimental.toto2_forecast.tests.golden.goldens import load_golden
from models.experimental.toto2_forecast.tests.perf.test_forward_perf import SHAPES


@pytest.mark.timeout(1800)
def test_profile_forward(device):
    import ttnn
    from tracy import signpost

    from models.experimental.toto2_forecast.tt.common import Toto2Precision
    from models.experimental.toto2_forecast.tt.model import TtToto2
    from models.experimental.toto2_forecast.tt.weights import Toto2Weights

    shape = SHAPES[os.environ.get("MODEL_PROFILE_SHAPE", "paper")]
    precision_name = os.environ.get("MODEL_PROFILE_PRECISION", "accuracy")
    weights = Toto2Weights.from_checkpoint(shape.checkpoint)
    n = weights.config.num_layers
    kinds = ["variate" if lw.is_variate else "time" for lw in weights.layers]
    last_time = max(i for i, k in enumerate(kinds) if k == "time")
    assert kinds[last_time + 1 :] == ["variate"] * (n - last_time - 1) and kinds.count("variate") == 1
    reduced = weights.reduced([0, last_time, n - 1])
    tt = TtToto2(device, reduced, Toto2Precision.named(precision_name))
    assert tt.layer_kinds == ["time", "last_time", "variate"]

    golden = load_golden(shape.checkpoint, shape.golden)
    i = golden["inputs"]
    host = tt.prepare_inputs(i["target"], i["target_mask"], i["series_ids"], i["horizon"])
    inputs = tt.upload_inputs(host)
    sections = ["layer_time", "last_time", "layer_variate"]

    def forward(marked: bool):
        def mark(name):
            if marked:
                ttnn.synchronize_device(device)
                signpost(name)

        mark("embed_start")
        x = tt.embed(inputs)
        mark("embed_end")
        for li, section in enumerate(sections):
            mark(f"{section}_start")
            x = tt.run_layer(li, x, inputs)
            mark(f"{section}_end")
        mark("head_start")
        out = tt.head(x)
        mark("head_end")
        return out

    ttnn.deallocate(forward(marked=False))
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    signpost("MODEL_START")
    out = forward(marked=True)
    ttnn.synchronize_device(device)
    signpost("MODEL_END")
    ttnn.ReadDeviceProfiler(device)

    got = ttnn.to_torch(out).float()
    assert torch.isfinite(got).all()
    ttnn.deallocate(out)
    tt.deallocate_inputs(inputs)

    meta = {
        "shape": shape.name,
        "precision": precision_name,
        "test": "forward",
        "checkpoint": shape.checkpoint,
        "rows": host.rows,
        "seq_len": host.seq_len,
        "window": ["MODEL_START", "MODEL_END"],
        "sections": ["embed", *sections, "head"],
        "counts": {"embed": 1, "layer_time": last_time, "last_time": 1, "layer_variate": 1, "head": 1},
    }
    print(f"\n[PROFILE] {json.dumps(meta)}")
    out_dir = os.environ.get("MODEL_PROFILE_OUT")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "layer_counts.json").write_text(json.dumps(meta, indent=2) + "\n")
