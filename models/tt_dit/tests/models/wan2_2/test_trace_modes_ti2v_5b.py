# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-execution modes of the TI2V-5B denoise loop must be bit-identical, and 2CQ must not shrink the grid.

Three ways to execute the captured per-step trace (`WanPipeline.configure_trace_execution`):

  blocking      production default: blocking `execute_trace` on queue 0, inputs updated on queue 0
  nonblocking   non-blocking `execute_trace`; the host runs ahead and enqueues the on-device UniPC
                step and the next step's inputs while the device is still in the trace
  2cq           nonblocking + the per-step host->device uploads (timestep, guidance) on queue 1,
                fenced with events in the Tracer (see `models/tt_dit/utils/tracing.py`)

None of them changes what is computed, so the gate is `torch.equal` on the final latents at a
fixed seed. The test opens the mesh with two command queues (needed for `2cq`) and also prints
`compute_with_storage_grid_size()`: the swept DiT matmul tables are keyed on the 1-queue grid, so
a second queue that costs Tensix dispatch cores would be a perf regression by construction and
this number must match the 1-queue grid the perf logs report (11x10 on the 4x8 BH Galaxy).

    pytest models/tt_dit/tests/models/wan2_2/test_trace_modes_ti2v_5b.py -sv --timeout=0 | grep ^TRACEMODE
"""

import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b import WanTI2V5BPipeline
from models.tt_dit.utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links

_PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
MODES = {
    "blocking": dict(blocking=True, input_cq_id=None),
    "nonblocking": dict(blocking=False, input_cq_id=None),
    "2cq": dict(blocking=False, input_cq_id=1),
}


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    [
        [
            (4, 8),
            (4, 8),
            {"trace_region_size": 150000000, "num_command_queues": 2, **ring_params_req_exact_devices},
            ttnn.Topology.Ring,
        ],
    ],
    ids=["bh_4x8_2cq"],
    indirect=["mesh_device", "device_params"],
)
def test_trace_modes_bit_identical(mesh_device, mesh_shape, topology):
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    grid = mesh_device.compute_with_storage_grid_size()
    print(f"TRACEMODE compute_with_storage_grid_size with 2 command queues: {grid.x}x{grid.y}")
    expected_grid = os.environ.get("WAN5B_EXPECT_GRID", "11x10")
    grid_ok = f"{grid.x}x{grid.y}" == expected_grid

    height = int(os.environ.get("WAN5B_TM_HEIGHT", 704))
    width = int(os.environ.get("WAN5B_TM_WIDTH", 1280))
    num_frames = int(os.environ.get("WAN5B_TM_FRAMES", 81))
    steps = int(os.environ.get("WAN5B_TM_STEPS", 8))

    pipeline = WanTI2V5BPipeline.create_pipeline(
        mesh_device=mesh_device, height=height, width=width, num_frames=num_frames, run_warmup=False
    )

    def _run(n, traced):
        with torch.no_grad():
            out = pipeline(prompts=[_PROMPT], num_inference_steps=n, seed=42, output_type="latent", traced=traced)
        ttnn.synchronize_device(mesh_device)
        return out

    _run(2, traced=False)  # compile (denoise only: latent output, no VAE decode)
    _run(2, traced=True)  # capture the trace once; every mode below executes the same trace

    latents = {}
    walls = {}
    for name, cfg in MODES.items():
        pipeline.configure_trace_execution(**cfg)
        t0 = time.perf_counter()
        latents[name] = _run(steps, traced=True).detach().clone()
        walls[name] = (time.perf_counter() - t0) / steps
        print(
            f"TRACEMODE {name:12s} {walls[name] * 1e3:8.2f} ms/step (denoise only, {steps} steps, wall incl. encoder)"
        )
    pipeline.configure_trace_execution()  # back to the default
    pipeline.release_traces()

    ref = latents["blocking"]
    failures = []
    for name, lat in latents.items():
        same = torch.equal(ref, lat)
        diff = (ref.float() - lat.float()).abs().max().item()
        print(f"TRACEMODE {name:12s} vs blocking: bit-identical={same} max_abs_diff={diff}")
        if not same:
            failures.append(f"{name}: max_abs_diff {diff}")
    for name in ("nonblocking", "2cq"):
        print(f"TRACEMODE {name:12s} wall delta vs blocking: {(walls[name] - walls['blocking']) * 1e3:+.2f} ms/step")

    assert not failures, "; ".join(failures)
    assert grid_ok, (
        f"compute grid with 2 command queues is {grid.x}x{grid.y}, expected {expected_grid}: the second "
        "queue costs worker cores and the swept matmul tables (keyed on the grid) would miss"
    )
