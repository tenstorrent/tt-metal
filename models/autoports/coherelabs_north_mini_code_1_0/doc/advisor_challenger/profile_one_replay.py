# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Collect one bounded replay using the same model hooks as the fixed harness."""

from __future__ import annotations

import json
import os

import ttnn
from tracy import signpost

from models.autoports.coherelabs_north_mini_code_1_0.doc.advisor_challenger import harness


def main():
    frozen = json.load(open(os.environ["CHALLENGER_POLICY_JSON"]))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        state = harness.build(mesh, frozen["shipped_policy"])
        harness.decode(state)
        ttnn.synchronize_device(mesh)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        harness.decode(state)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        signpost(header="PERF_DECODE")
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        signpost(header="PERF_DECODE_END")
        ttnn.release_trace(mesh, trace_id)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
