# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile target: the flow model's per-frame solve at B users, for the op profiler.

Runs the solve once eager (compile), then captures it as a trace and replays it a few times, so
the profiler's op table holds a handful of identical traced frames. Nothing is asserted beyond
"it ran"; the output is the op report.

    python -m tracy -r -p -t 8086 -o <profiler output dir>/flow_b32 \\
        -m pytest models/experimental/voxtral_tts/tests/perf/test_flow_profile.py

Env: PROFILE_BATCH (32), PROFILE_REPLAYS (3), VOXTRAL_DEVICE_ID (0).
"""

import os

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    CFG_ALPHA,
    FM_INPUT_DIM,
    N_ACOUSTIC_CODEBOOK,
    N_DECODING_STEPS,
)
from models.experimental.voxtral_tts.tests.reference_helpers import needs_checkpoint  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import TtVoxtralFlow  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

pytestmark = [pytest.mark.slow, needs_checkpoint]

B = int(os.environ.get("PROFILE_BATCH", "32"))
REPLAYS = int(os.environ.get("PROFILE_REPLAYS", "3"))
DEVICE_ID = int(os.environ.get("VOXTRAL_DEVICE_ID", "0"))


def test_flow_solve_profile():
    dev = open_device(device_id=DEVICE_ID)
    try:
        fl = TtVoxtralFlow(dev)
        dv = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)
        x0 = dv(torch.randn(B, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32)
        pair = dv(torch.randn(2 * B, 1, FM_INPUT_DIM) * 0.02, fl.dtype)
        # eager once: compile every program (profiled too, but distinguishable by order)
        fl._solve(x0, pair, B, N_DECODING_STEPS, CFG_ALPHA)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            out = fl._solve(x0, pair, B, N_DECODING_STEPS, CFG_ALPHA)
        finally:
            ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        for _ in range(REPLAYS):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ttnn.release_trace(dev, tid)
        assert tuple(out.shape)[-1] == N_ACOUSTIC_CODEBOOK
    finally:
        ttnn.close_device(dev)
