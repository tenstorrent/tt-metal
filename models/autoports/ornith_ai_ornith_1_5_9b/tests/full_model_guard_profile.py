# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Profile full-pool versus selective guards at the real B32 state geometry.

One real linear layer owns the cache. Only guard work is in signposted traces;
this isolates the masked full-model overhead without the B32 generator warmup.
"""

import argparse
import json
from pathlib import Path

import torch
from tracy import signpost

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.batch_state import BatchState
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
    traces = []
    try:
        model = OrnithModel(mesh, layer_indices=[0])
        cache = model.allocate_cache(batch=32, context=512)
        guards = []
        for selected in (False, True):
            model.selective_inactive_restore = selected
            guard = BatchState(model, cache)
            guard.set_active([i != 1 for i in range(32)])
            guards.append(guard)
        for guard in guards:
            guard.before_layer(0)
            guard.after_layer(0)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
        for guard in guards:
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            guard.before_layer(0)
            guard.after_layer(0)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces.append(trace)
        for name, trace in zip(("GUARD_FULL", "GUARD_SELECTIVE"), traces):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost(name)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            signpost(name + "_END")
            ttnn.ReadDeviceProfiler(mesh)
        args.output.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "batch": 32,
                    "active_rows": 31,
                    "scope": "Guard only, real FP32 native recurrent pool; no model arithmetic or sampling in window",
                    "recurrent_bytes_per_slot_per_device": 524288,
                    "layer_count": 24,
                    "trace_ids": traces,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        for trace in traces:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
