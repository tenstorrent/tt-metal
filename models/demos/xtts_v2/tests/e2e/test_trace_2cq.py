# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-3 gate: trace + 2CQ per-stage contract for coqui/XTTS-v2.

Opens a device with a trace region + 2 command queues, builds the Pipeline
object, and runs `trace_capture_selftest`: for each stage it pins the sequence
axis to a fixed capacity, captures ONE host-op-free step in
begin/end_trace_capture, execute_trace, verifies PCC vs the eager reference, and
releases the trace before the next stage. Stages whose graduated stub has
unavoidable host ops are degraded to single-CQ eager with a printed fallback.
"""

from __future__ import annotations

import importlib.util as ilu
import os

import torch

import ttnn
from models.demos.xtts_v2.tt import pipeline as P

HF_MODEL_ID = "coqui/XTTS-v2"


def _load_reference():
    here = os.path.dirname(os.path.abspath(__file__))
    rl = os.path.normpath(os.path.join(here, "..", "pcc", "_reference_loader.py"))
    spec = ilu.spec_from_file_location("_reference_loader", rl)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_reference_model(HF_MODEL_ID)


def test_trace_2cq_selftest():
    torch.manual_seed(0)
    device = ttnn.open_device(
        device_id=0, l1_small_size=24576, trace_region_size=200_000_000, num_command_queues=2
    )
    try:
        model = _load_reference()
        pipe = P.Pipeline(device, model, capacity=64)
        print(f"PIPELINE_STAGES={pipe.PIPELINE_STAGES}")
        # exercise the 2CQ write path on a host-free stage
        pipe.gpt_decode_trace_setup()
        pipe.gpt_decode_write_inputs()      # stages next input on command-queue 1
        ok = pipe.trace_capture_selftest(device)
        print(f"trace_capture_selftest host-free-all={ok}")
        # gate: the compute-dominant transformer stages must trace host-free + match
        for st in P.Pipeline._HOSTFREE_STAGES:
            pipe._trace_setup(st)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = pipe._trace_step(st)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            pcc = P.comp_pcc(pipe._ref[st], ttnn.to_torch(out).float(), 0.95)[1]
            ttnn.release_trace(device, tid)
            print(f"  gate {st}: trace PCC={pcc}")
            assert pcc >= 0.95, f"trace step {st} PCC {pcc} < 0.95"
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_trace_2cq_selftest()
