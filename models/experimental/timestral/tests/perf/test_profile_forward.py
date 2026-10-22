# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tracy profile of one warm Timestral-209M decode block, reduced to one real layer per layer kind.

Profiling all 24 layers would flood the device profiler buffers, so the profile keeps the real
embedding and head plus the first layer of each kind and the last layer (which cuts to the decoder
slots), at the real benchmark shape and weights.
Signposts (for ``tt-perf-report --start-signpost/--end-signpost``):

    TIMESTRAL_START / TIMESTRAL_END       the whole reduced forward
    embed_start / embed_end               patch embedding and forecast-query layout
    layer_<kind>_start / layer_<kind>_end one transformer block of each kind
    layer_last_start / layer_last_end     the last block, cut to the decoder slots (or stream)
    head_start / head_end                 out_norm, quantile head

The full-model device time is estimated as embed + sum(count[kind] * layer[kind]) + head, with the
last layer counted as its own kind ``last``;
``layer_counts.json`` (written next to the report when ``TIMESTRAL_PROFILE_OUT`` is set)
carries the counts. Run under ``python -m tracy -r -p -m pytest``; see ``perf_iteration.sh``.

    TIMESTRAL_PROFILE_SHAPE      paper (default) | long_context | multivariate
    TIMESTRAL_PROFILE_PRECISION  accuracy (default) | performance
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models.experimental.timestral.tests.golden_helpers import load_checkpoint
from models.experimental.timestral.tests.perf.test_forward_perf import SHAPES, _inputs


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_profile_predict_block(mesh_device):
    import ttnn
    from tracy import signpost

    from models.experimental.timestral.tt.common import TtTimestralPrecision
    from models.experimental.timestral.tt.model import TtTimestral
    from models.experimental.timestral.tt.weights import TimestralWeights

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip model")
    shape = SHAPES[os.environ.get("TIMESTRAL_PROFILE_SHAPE", "paper")]
    precision_name = os.environ.get("TIMESTRAL_PROFILE_PRECISION", "accuracy")
    precision = TtTimestralPrecision.performance() if precision_name == "performance" else TtTimestralPrecision()

    weights = TimestralWeights.from_reference(load_checkpoint("Timestral-209M"))
    schedule = list(weights.config.block_schedule)
    kinds = list(dict.fromkeys(schedule))
    reduced = replace(weights, layers=(*(weights.layers[schedule.index(k)] for k in kinds), weights.layers[-1]))
    tt = TtTimestral(mesh_device, reduced, precision)

    context, observed, group_ids = _inputs(shape)
    prepared = tt.prepare_inputs(context, observed, group_ids, shape.horizon_patches)
    inputs = tt.upload_inputs(prepared)

    def forward(marked: bool):
        def mark(name):
            if marked:
                ttnn.synchronize_device(mesh_device)
                signpost(name)

        mark("embed_start")
        x = tt.embed(inputs)
        mark("embed_end")
        for kind, layer in zip(kinds, tt._layers):
            mark(f"layer_{kind}_start")
            x = tt.run_layer(layer, x, inputs)
            mark(f"layer_{kind}_end")
        mark("layer_last_start")
        x = tt.run_layer(tt._layers[-1], x, inputs, last=True)
        mark("layer_last_end")
        mark("head_start")
        out = tt.head(x)
        mark("head_end")
        return out

    ttnn.deallocate(forward(marked=False))
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    signpost("TIMESTRAL_START")
    out = forward(marked=True)
    ttnn.synchronize_device(mesh_device)
    signpost("TIMESTRAL_END")
    ttnn.ReadDeviceProfiler(mesh_device)

    got = ttnn.to_torch(out).float()
    assert torch.isfinite(got).all()
    ttnn.deallocate(out)
    tt.deallocate_inputs(inputs)

    counts = {kind: schedule[:-1].count(kind) for kind in kinds}
    counts["last"] = 1
    meta = {
        "shape": shape.name,
        "precision": precision_name,
        "rows": shape.rows,
        "seq_len": prepared.seq_len,
        "layer_counts": counts,
        "sections": ["embed", *(f"layer_{k}" for k in counts), "head"],
    }
    print(f"\n[PROFILE] {json.dumps(meta)}")
    out_dir = os.environ.get("TIMESTRAL_PROFILE_OUT")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "layer_counts.json").write_text(json.dumps(meta, indent=2) + "\n")
