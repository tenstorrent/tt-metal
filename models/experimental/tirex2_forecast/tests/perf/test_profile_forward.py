# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tracy profile of one warm TiRex-2 forward reduced to one block of each kind, with scaled sLSTM steps.

Real weights and the real bench shape. The reduced model keeps the embedding and head, block 0 (mLSTM) and
block 1 (sLSTM); block 11 is an ordinary sLSTM block in this graph (no read-out cut yet), so it is counted
as one more sLSTM block. Signposts (``tt-perf-report --start-signpost``):

    MODEL_START / MODEL_END                 the whole reduced forward
    embed_start / embed_end                 residual-block patch embedding
    layer_mlstm_start / _end                block 0: mLSTM time mixer + FFN (+ variate mixer if multivariate)
    slstm_pre_start / _end                  block 1: RMSNorm, transpose to time-major, W_in projection + bias
    slstm_first_start / _end                sLSTM step t = 0 (m = pre_i, n = 1)
    slstm_steps_start / _end                PROFILED_STEPS ordinary sLSTM steps (t = 1 .. PROFILED_STEPS)
    slstm_post_start / _end                 concat, per-head norm, transpose back, residual, FFN (+ mixer)
    head_start / head_end                   final norm, residual-block head, read-out window slice

The op-level loop issues ~29 ops per step, and a full 285-step sLSTM block would need ~8k profiled programs
per pass (a 9 GB Tracy capture for warm-up plus measured pass), so the profile runs ``PROFILED_STEPS`` = 71 of
the 284 ordinary steps (a quarter; every step runs the same programs on the same shapes) and counts them 4 x 6
times (``layer_counts.json``). The remaining steps' outputs are zeros, so values after block 1 are not
meaningful; only timings are read. The device profiler buffers are drained every 20 steps. The profiled forward
runs eagerly (op by op); the bench runs the captured trace.

    MODEL_PROFILE_SHAPE      paper (default) | long_context | multivariate
    MODEL_PROFILE_PRECISION  accuracy (default) | performance
    MODEL_PROFILE_OUT        directory for layer_counts.json
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tests.golden.goldens import have_goldens
from models.experimental.tirex2_forecast.tests.pcc.helpers import host_weights
from models.experimental.tirex2_forecast.tests.perf.test_forward_perf import shape_inputs

SLSTM_BLOCKS = df.NUM_BLOCKS // 2
DRAIN_EVERY = 20


def step_plan(cut: int) -> tuple[int, int]:
    """``(ordinary steps after t = 0, profiled ordinary steps)``: a quarter of them (284 / 71 on the full path,
    92 / 23 with the 192-token pad cut, where t = 0 starts from the cached pad state)."""
    ordinary = df.NUM_TOKENS - cut - 1
    return ordinary, ordinary // 4


def counts(cut: int) -> dict:
    ordinary, profiled = step_plan(cut)
    return {
        "embed": 1,
        "layer_mlstm": df.NUM_BLOCKS // 2,
        "slstm_pre": SLSTM_BLOCKS,
        "slstm_first": SLSTM_BLOCKS,
        "slstm_steps": SLSTM_BLOCKS * ordinary // profiled,
        "slstm_post": SLSTM_BLOCKS,
        "head": 1,
    }


@pytest.mark.skipif(not have_goldens(), reason="goldens not generated")
@pytest.mark.timeout(2700)
def test_profile_forward(device):
    import ttnn
    from tracy import signpost

    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.common import Precision
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2

    shape = os.environ.get("MODEL_PROFILE_SHAPE", "paper")
    precision = os.environ.get("MODEL_PROFILE_PRECISION", "accuracy")
    _, ctx, horizon, gid, tm = shape_inputs(shape)
    host = host_weights()
    reduced = replace(host, blocks=[host.blocks[0], host.blocks[1]])
    tt = TtTiRex2(device, reduced, Precision.by_name(precision), variate=shape == "multivariate")
    mlstm_block, slstm_block = tt.weights.blocks
    prep = tt.prepare_inputs(ctx, horizon, gid, tm)
    inputs = tt.upload_inputs(prep)
    ORDINARY_STEPS, PROFILED_STEPS = step_plan(prep.cut)
    assert ORDINARY_STEPS % PROFILED_STEPS == 0
    COUNTS = counts(prep.cut)
    slstm_init = tt.pad_state(prep.cut, prep.device_rows, slstm_block) if prep.cut else None

    def forward(marked: bool):
        def mark(name):
            if marked:
                ttnn.synchronize_device(device)
                signpost(name)

        def hook(t):
            if t == 0:
                mark("slstm_first_end")
                mark("slstm_steps_start")
            elif t % DRAIN_EVERY == 0:
                ttnn.ReadDeviceProfiler(device)  # also in the warm-up: dropped markers break the Tracy import

        mark("embed_start")
        x = tt.embed(inputs)
        mark("embed_end")
        mark("layer_mlstm_start")
        y = tt.run_block(mlstm_block, x, inputs)
        layers.free(x)
        x = y
        mark("layer_mlstm_end")
        ttnn.ReadDeviceProfiler(device)

        mark("slstm_pre_start")
        WxT = layers.slstm_prepare(x, slstm_block.cell, tt.precision)
        mark("slstm_pre_end")
        mark("slstm_first_start")
        ys, (c, n, m) = layers.slstm_recurrence(
            device,
            WxT,
            slstm_block.cell.r,
            PROFILED_STEPS + 1,
            tt.consts,
            step_hook=hook,
            y_fp32=tt.precision.y_fp32,
            init=slstm_init,
        )
        mark("slstm_steps_end")
        ttnn.ReadDeviceProfiler(device)
        layers.free(WxT, c, n, m)
        mark("slstm_post_start")
        y = layers.slstm_finish(device, x, ys, slstm_block.cell, tt.weights.head_mean, tt.consts)
        out = layers.ffn_residual(y, slstm_block.ffn, tt.precision)
        layers.free(y, x)
        if inputs.vmask is not None:
            mixed = layers.variate_block(device, out, slstm_block.variate, tt.precision, inputs.vmask, inputs.keep)
            layers.free(out)
            out = mixed
        mark("slstm_post_end")
        mark("head_start")
        h = tt.head(out, prep.cut)
        mark("head_end")
        layers.free(out)
        return h

    ttnn.deallocate(forward(marked=False))  # warm-up: compile every program
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    signpost("MODEL_START")
    out = forward(marked=True)
    ttnn.synchronize_device(device)
    signpost("MODEL_END")
    ttnn.ReadDeviceProfiler(device)
    ttnn.deallocate(out)
    tt.deallocate_inputs(inputs)

    meta = {
        "shape": shape,
        "precision": precision,
        "test": "forward",
        "window": ["MODEL_START", "MODEL_END"],
        "rows": prep.rows,
        "pad_cut": prep.cut,
        "device_rows": prep.device_rows,
        "sections": list(COUNTS),
        "counts": COUNTS,
        "blocks": {"layer_mlstm": 0, "slstm_*": 1},
        "slstm_step_scaling": f"{PROFILED_STEPS} of {ORDINARY_STEPS} ordinary steps profiled, counted "
        f"{ORDINARY_STEPS // PROFILED_STEPS} x {SLSTM_BLOCKS} times",
        "profiler_drain_every_steps": DRAIN_EVERY,
    }
    print(f"\n[PROFILE] {json.dumps(meta)}")
    out_dir = os.environ.get("MODEL_PROFILE_OUT")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "layer_counts.json").write_text(json.dumps(meta, indent=2) + "\n")
