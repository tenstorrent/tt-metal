# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Op-level device profile of one warm chunk of the whole model (testing/profile.py:op_profile, no golden needed).

One target.chunk chunk at position GLM_PROF_START (default: the last chunk of target.seq, where the DSA context is
longest) is run once to compile, then twice under the bring-up profiler: op mode (a sync and a device-profiler read
after every outermost ttnn op: exact per-op device time per chip, per layer and step) and timeline mode (no syncs:
device kernel time vs idle gaps vs host dispatch). Rows go to generated/glm53_flash_d_p_lb/profile_ops.json; the
roofline analysis is tests/roofline.py.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
    TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONPATH=$PWD \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_profile_ops.py -s
"""

import json
import os
import time
from pathlib import Path

import torch

from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import op_profile

S = spec()
pytestmark = device_timeout(S)
OUT = Path(__file__).resolve().parents[4] / "generated" / "glm53_flash_d_p_lb" / "profile_ops.json"


@mesh_parametrize
def test_profile_ops(mesh_device):
    from models.demos.common.bringup.reference.prompt import tokens as prompt_tokens

    for k, v in profiler.PROFILER_ENV.items():
        assert os.environ.get(k) == v, f"set {k}={v}"
    seq, chunk = int(S.get("target.seq")), int(os.environ.get("GLM_PROF_CHUNK", S.get("target.chunk")))
    start = int(os.environ.get("GLM_PROF_START", seq - chunk))
    toks = prompt_tokens(S, start + chunk).to(torch.long)[start : start + chunk]
    layers = S.layers()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)

    def run():
        h = model.embed(toks)
        for i in layers:
            profiler.set_layer(i)
            h2 = model.layer(i, h, start, None)
            model.free(h)
            h = h2
        profiler.set_layer(None)
        y = model.final_norm(h)
        model.free(h)
        model.free(y)

    run()  # compile
    model.sync()
    t = time.time()
    run()
    model.sync()
    wall = time.time() - t
    print(f"warm chunk [{start}, {start + chunk}) wall {wall * 1e3:.1f} ms", flush=True)
    ops, timeline = op_profile(mesh_device, run)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "chunk": chunk,
                "start": start,
                "chips": mesh_device.get_num_devices(),
                "grid": [
                    mesh_device.compute_with_storage_grid_size().x,
                    mesh_device.compute_with_storage_grid_size().y,
                ],
                "experts_dtype": str(S.get("device.experts_dtype")),
                "wall_ms": round(wall * 1e3, 1),
                "timeline": timeline,
                "ops": ops,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT}", flush=True)
