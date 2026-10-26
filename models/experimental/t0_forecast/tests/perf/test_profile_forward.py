# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tracy profile of one warm t0-gamma forward, reduced to one real layer per kind plus the last layer.

Profiling all 24 layers would flood the device profiler buffers, so the profile keeps the real patch encoder and
head plus layer 0 (time), layer 2 (group), layer 22 (the last time layer: its attention output is cut to the
read-out patches before its MLP) and layer 23 (the last layer, on the read-out patches), at the real benchmark shape
and weights. Signposts (``tt-perf-report --start-signpost / --end-signpost``):

    MODEL_START / MODEL_END                   the whole reduced forward
    embed_start / embed_end                   patch encoder
    layer_time_start / layer_time_end         one time layer        (x 15 in the full model)
    layer_group_start / layer_group_end       one group layer       (x 7)
    cut_start / cut_end                       layer 22: attention, read-out cut, MLP on the read-out rows (x 1)
    last_start / last_end                     layer 23 on the read-out rows (x 1)
    head_start / head_end                     out_norm, decoder, monotone head

``layer_counts.json`` goes to ``$MODEL_PROFILE_OUT``. Run under ``python -m tracy -r -p -m pytest`` (``perf_iteration.sh``).

    T0_PROFILE_SHAPE      paper (default) | long_context | multivariate
    T0_PROFILE_PRECISION  accuracy (default) | experimental_performance (retired, D5)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from models.experimental.t0_forecast.tests.helpers import golden, weights
from models.experimental.t0_forecast.tests.perf.test_forward_perf import SHAPES
from models.experimental.t0_forecast.tt.weights import GROUP, TIME

REDUCED = {"layer_time": 0, "layer_group": 2, "cut": 22, "last": 23}


@pytest.mark.timeout(1800)
def test_profile_forward():
    import ttnn
    from tracy import signpost

    from models.experimental.t0_forecast.tt.common import TtT0Precision
    from models.experimental.t0_forecast.tt.model import TtT0

    shape = os.environ.get("T0_PROFILE_SHAPE", "paper")
    precision = os.environ.get("T0_PROFILE_PRECISION", "accuracy")
    file, _, _, _, horizon = SHAPES[shape]
    g = golden(file)
    w = weights()
    kinds = [layer.kind for layer in w.layers]
    assert kinds[REDUCED["layer_time"]] == TIME and kinds[REDUCED["layer_group"]] == GROUP
    cut = max(i for i, k in enumerate(kinds) if k == TIME)
    assert REDUCED["cut"] == cut and REDUCED["last"] == len(kinds) - 1 and kinds[-1] == GROUP
    counts = {
        "embed": 1,
        "layer_time": kinds[:cut].count(TIME),
        "layer_group": kinds[:cut].count(GROUP),
        "cut": 1,
        "last": 1,
        "head": 1,
    }

    device = ttnn.open_device(device_id=0)
    try:
        tt = TtT0(device, w.reduced(list(REDUCED.values())), TtT0Precision.named(precision))
        prepared = tt.prepare_inputs(g["context"], horizon)
        inputs = tt.upload_inputs(prepared)
        sections = list(REDUCED)
        assert tt.cut_index == sections.index("cut")

        def forward(marked: bool):
            def mark(name):
                if marked:
                    ttnn.synchronize_device(device)
                    signpost(name)

            mark("embed_start")
            x = tt.embed(inputs)
            mark("embed_end")
            for index, name in enumerate(sections):
                mark(f"{name}_start")
                x = tt.run_layer(index, x, inputs)
                mark(f"{name}_end")
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
    finally:
        ttnn.close_device(device)

    meta = {
        "shape": shape,
        "precision": precision,
        "test": "forward",
        "rows": prepared.rows,
        "num_patches": prepared.num_patches,
        "reduced_layers": REDUCED,
        "window": ["MODEL_START", "MODEL_END"],
        "sections": list(counts),
        "counts": counts,
    }
    print(f"\n[PROFILE] {json.dumps(meta)}")
    out_dir = os.environ.get("MODEL_PROFILE_OUT")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "layer_counts.json").write_text(json.dumps(meta, indent=2) + "\n")
