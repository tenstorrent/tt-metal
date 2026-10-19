# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-profile one decode step on one TP=8 row (the first MIMO_PERF_LAYERS layers, default 6).

All four Galaxy rows run it (one replica each). Run under the profiler and rank ops:

    python -m tracy -p -r -v -m pytest models/experimental/mimo_v2_6_flash/tests/test_decode_perf_tp8.py
    tt-perf-report --signpost measured generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv

Layers 0-5 cover every layer type: 0 global + dense MLP, 1-4 SWA + MoE, 5 global + MoE.
"""

import os

import pytest
import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.common import (
    golden_path,
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    load_config_and_ckpt,
)
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator

try:
    from tracy import signpost
except ImportError:  # not running under the profiler

    def signpost(*args, **kwargs):
        pass


USERS = 4


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_decode_perf_tp8(mesh_device):
    # All four rows run the same work: the profiler wants ops on every device it reads, and a lone 1x8
    # mesh cannot bring up the ring fabric on a Galaxy.
    num_layers = int(os.environ.get("MIMO_PERF_LAYERS", 6))
    cfg, ckpt = load_config_and_ckpt()
    gen = MiMoGalaxyGenerator(
        mesh_device,
        cfg,
        ckpt,
        cache_dir("model_bfp4experts"),
        dp=4,
        users_per_replica=USERS,
        max_seq_len=4096,
        num_layers=num_layers,
    )
    golden = torch.load(golden_path())["token_ids"]
    prompts = {u: golden[u % len(golden)] for u in range(gen.num_users)}

    def drain():  # the device profiler buffer holds ~1000 kernel runs; read it out before it overflows
        for sm in gen.submeshes:
            ttnn.ReadDeviceProfiler(sm)

    drain()
    first = {}
    for slot in range(USERS):  # one slot per replica at a time, so each drain stays small
        batch = {u: p for u, p in prompts.items() if gen.locate(u)[1] == slot}
        first.update(gen.prefill(batch))
        drain()
    tokens = {u: int(first[u].argmax()) for u in prompts}
    positions = {u: prompts[u].numel() for u in prompts}

    for step in range(3):
        if step == 2:
            signpost("measured")
        out = gen.decode(tokens, positions)
        drain()
        for u, lg in out.items():
            tokens[u] = int(lg.argmax())
            positions[u] += 1
    signpost("end")
