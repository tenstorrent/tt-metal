# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One warm chunk of the spec's layers for a Tracy op report (no per-op syncs): compile run, then the measured run
between signposts "start" and "stop", with a signpost "L<i>" before each layer. Profile a few representative layers
(e.g. a spec with layers 2-4: kda_dense, dsa_moe, kda_moe) and scale by the layer counts.

    BRINGUP_SPEC=<spec with layers 2-4> TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONPATH=$PWD \\
    python -m tracy -r -p -v -m pytest models/demos/glm53_flash_d_p_lb/tests/test_tracy_chunk.py -s
    tt-perf-report generated/profiler/reports/<stamp>/ops_perf_results_<stamp>.csv --start-signpost start \\
        --end-signpost stop
"""

import os

import torch
from tracy import signpost

import ttnn
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


@mesh_parametrize
def test_tracy_chunk(mesh_device):
    from models.demos.common.bringup.reference.prompt import tokens as prompt_tokens

    seq, chunk = int(S.get("target.seq")), int(os.environ.get("GLM_PROF_CHUNK", S.get("target.chunk")))
    start = int(os.environ.get("GLM_PROF_START", seq - chunk))
    toks = prompt_tokens(S, start + chunk).to(torch.long)[start : start + chunk]
    layers = S.layers()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)

    def run(marks: bool):
        h = model.embed(toks)
        for i in layers:
            if marks:
                signpost(f"L{i}")
            h2 = model.layer(i, h, start, None)
            model.free(h)
            h = h2
        model.free(h)
        model.sync()

    # flush the device profiler buffers between phases (they hold a bounded number of programs per core)
    ttnn.ReadDeviceProfiler(mesh_device)
    run(False)  # compile
    ttnn.ReadDeviceProfiler(mesh_device)
    signpost("start")
    run(True)
    signpost("stop")
    ttnn.ReadDeviceProfiler(mesh_device)
