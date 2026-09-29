# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cost of building the MiMo model (weights: load / dequant / layout / upload), profiled with cProfile.

    MIMO_INIT_LAYERS=0,1,5 scripts/run_safe_pytest.sh models/demos/mimo_v2_d_p/tests/perf/test_model_init_time.py -s
"""

import cProfile
import io
import os
import pstats
import time

import ttnn  # noqa: F401
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.model import TtMiMoModel

LAYERS = [int(x) for x in os.environ.get("MIMO_INIT_LAYERS", "1").split(",")]


@MESH_PARAMS
def test_model_init_time(mesh_device, device_params):
    cfg = MiMoTextConfig.from_json()
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    model = TtMiMoModel(
        mesh_device,
        cfg,
        lambda i: layer_state(i, cfg),
        fabric_config=device_params["fabric_config"],
        max_seq_len=8192,
        chunk_size=4096,
        layers=LAYERS,
        global_state=global_state,
    )
    pr.disable()
    print(f"INIT layers {LAYERS}: {time.time() - t0:.1f} s")
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(45)
    print("\n".join(l for l in s.getvalue().splitlines() if l.strip()))
    del model
