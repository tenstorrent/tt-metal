# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile-only eager replay for the advisor-challenger winner.

Timing decisions remain exclusively in ``harness_template.py``.  Tracy cannot
associate cached trace-replay device rows with host signposts in this checkout,
so this wrapper brackets one eager execution of the exact same build/decode
hooks solely to produce a bounded op-level CSV.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import ttnn
from tracy import signpost

from models.autoports.google_gemma_4_26b_a4b_it.tests.advisor_challenger_harness import build, decode


def main() -> None:
    incumbent = Path(
        "models/autoports/google_gemma_4_26b_a4b_it/doc/advisor_challenger/incumbent.json"
    )
    policy = json.loads(incumbent.read_text())["shipped_policy"]
    if int(os.environ["CHALLENGER_DECODE_BATCH"]) != 1:
        raise SystemExit("advisor-challenger profile is decode batch 1 only")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        state = build(mesh, policy)
        decode(state)
        ttnn.synchronize_device(mesh)
        signpost(header="PERF_DECODE")
        decode(state)
        ttnn.synchronize_device(mesh)
        signpost(header="PERF_DECODE_END")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
