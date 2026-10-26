# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tracy profile of one warm TimesFM-3 forward, reduced to one real layer, at the real shape.

All 20 MixingTransformer layers run the same ops, so the reduced model keeps the real tokenizer and head
and layer 0. Signposts (``tt-perf-report --start-signpost/--end-signpost``):

    MODEL_START / MODEL_END     the whole reduced forward
    embed_start / embed_end     tokenizer
    layer_start / layer_end     one MixingTransformer layer
    cut_start / cut_end         the last layer with the read-out cut (R1-5), when the model runs it
    head_start / head_end       read-out slice and output head

The full-model device time is estimated as embed + first layer + 18 x layer + cut + head (R3-4, R4-6; 19 x layer without the cut); ``layer_counts.json`` (written to
``$MODEL_PROFILE_OUT``) carries the counts. Run under ``python -m tracy -r -p -m pytest`` (``perf_iteration.sh``).

    MODEL_PROFILE_SHAPE      paper (default) | long_context | multivariate
    MODEL_PROFILE_PRECISION  accuracy (default) | performance
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
import torch

from models.experimental.timesfm3_forecast.tests.golden_io import load_golden
from models.experimental.timesfm3_forecast.tests.perf.test_forward_perf import SHAPES


@pytest.mark.timeout(1800)
def test_profile_forward(device):
    import ttnn
    from tracy import signpost

    from models.experimental.timesfm3_forecast.common.decomposition import NUM_LAYERS
    from models.experimental.timesfm3_forecast.tt.common import TtPrecision
    from models.experimental.timesfm3_forecast.tt.model import TtTimesFM3

    shape = os.environ.get("MODEL_PROFILE_SHAPE", "paper")
    precision = os.environ.get("MODEL_PROFILE_PRECISION", "accuracy")
    g = load_golden(SHAPES[shape])
    tt = TtTimesFM3.from_checkpoint(device, TtPrecision.by_name(precision), layers=[0])
    prepared = tt.prepare_inputs(list(g["contexts"].numpy()), g["horizon"])
    inputs = tt.upload_inputs(prepared)

    def forward(marked: bool):
        def mark(name):
            if marked:
                ttnn.synchronize_device(device)
                signpost(name)

        mark("embed_start")
        x = tt.embed(inputs)
        mark("embed_end")
        # R4-6: the first layer runs the standalone pre-norm; the later ones take the pending post_add_norm output,
        # so the first layer is its own section and the steady-state layer is profiled second.
        mark("first_start")
        x = tt.layer(0, x, inputs)
        mark("first_end")
        mark("layer_start")
        x = tt.layer(0, x, inputs)
        mark("layer_end")
        head_inputs = inputs
        if tt.cut_last_layer:
            mark("cut_start")
            x = tt.layer(0, x, inputs, cut=True)
            mark("cut_end")
            head_inputs = dataclasses.replace(inputs, padded_seq=inputs.readout_count, readout_start=0)
        tt._drop_pending_norm()
        mark("head_start")
        out = tt.head(x, head_inputs)
        ttnn.deallocate(x)
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
        "test": "forward",
        "shape": shape,
        "precision": precision,
        "rows": prepared.rows,
        "tokens": prepared.num_patches,
        "padded_seq": inputs.padded_seq,
        "window": ["MODEL_START", "MODEL_END"],
        "sections": ["embed", "first", "layer", "cut", "head"]
        if tt.cut_last_layer
        else ["embed", "first", "layer", "head"],
        "counts": (
            {"embed": 1, "first": 1, "layer": NUM_LAYERS - 2, "cut": 1, "head": 1}
            if tt.cut_last_layer
            else {"embed": 1, "first": 1, "layer": NUM_LAYERS - 1, "head": 1}
        ),
    }
    print(f"\n[PROFILE] {json.dumps(meta)}")
    out_dir = os.environ.get("MODEL_PROFILE_OUT")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "layer_counts.json").write_text(json.dumps(meta, indent=2) + "\n")
