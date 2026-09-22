# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host/device gap per traced denoise step for TI2V-5B.

The production denoise loop (``WanPipeline.__call__`` -> ``_step``) executes one captured
trace per step with ``blocking=True`` and then dispatches the on-device UniPC solver eagerly.
Between the end of one trace and the start of the next the device is idle for however long
the host takes to (a) dispatch the solver, (b) copy the new latent into the trace's input
buffer, cast it, upload the timestep and guidance scalars, and (c) call ``execute_trace``
again. This bench sizes that idle time without Tracy, on the traced path itself:

  * ``ttnn.execute_trace`` is wrapped so its blocking wall time is the device time of the
    trace (plus dispatch latency);
  * ``pipeline._step`` is wrapped for the whole step's wall time;
  * ``solver.step`` is wrapped for its host dispatch time, and in the second pass followed by
    a ``synchronize_device`` whose wall time is the solver's device tail.

Per step, the device is busy for ``trace + solver_device``; everything else is host-only gap:

    gap = wall(pass A) - trace(pass A) - solver_device(pass B)

where ``solver_device ~= solver_host_dispatch + sync_tail`` in pass B (upper bound: the device
starts the first solver op as soon as it is dispatched). Pass A is the untouched production
timeline, so its ``wall`` should reproduce the perf test's ms/step.

    WAN5B_GAP_STEPS=40 pytest \
      models/tt_dit/tests/models/wan2_2/test_step_gap_ti2v_5b.py -sv --timeout=0 | grep ^GAP
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b import WanTI2V5BPipeline
from models.tt_dit.utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links

_PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."


class _StepTimer:
    """Wraps execute_trace, pipeline._step and solver.step; one record per denoise step."""

    def __init__(self, pipeline, mesh_device, *, sync_after_solver: bool):
        self.pipeline = pipeline
        self.mesh_device = mesh_device
        self.sync_after_solver = sync_after_solver
        self.records = []  # dicts: wall, trace, solver_host, solver_sync
        self._cur = None

    def __enter__(self):
        self._orig_execute_trace = ttnn.execute_trace
        self._orig_step = self.pipeline._step
        self._orig_solver_step = self.pipeline._solver.step

        def execute_trace(*args, **kwargs):
            t0 = time.perf_counter()
            out = self._orig_execute_trace(*args, **kwargs)
            if self._cur is not None:
                self._cur["trace"] += time.perf_counter() - t0
                self._cur["trace_calls"] += 1
                self._cur["blocking"] = kwargs.get("blocking", True)
            return out

        def solver_step(**kwargs):
            t0 = time.perf_counter()
            out = self._orig_solver_step(**kwargs)
            t1 = time.perf_counter()
            if self._cur is not None:
                self._cur["solver_host"] += t1 - t0
            if self.sync_after_solver:
                ttnn.synchronize_device(self.mesh_device)
                if self._cur is not None:
                    self._cur["solver_sync"] += time.perf_counter() - t1
            return out

        def step(**kwargs):
            self._cur = {"trace": 0.0, "trace_calls": 0, "solver_host": 0.0, "solver_sync": 0.0, "blocking": None}
            t0 = time.perf_counter()
            out = self._orig_step(**kwargs)
            self._cur["wall"] = time.perf_counter() - t0
            self.records.append(self._cur)
            self._cur = None
            return out

        ttnn.execute_trace = execute_trace
        self.pipeline._step = step
        self.pipeline._solver.step = solver_step
        return self

    def __exit__(self, *exc):
        ttnn.execute_trace = self._orig_execute_trace
        self.pipeline._step = self._orig_step
        self.pipeline._solver.step = self._orig_solver_step
        return False


def _ms(xs):
    return f"mean {statistics.mean(xs) * 1e3:7.2f}  median {statistics.median(xs) * 1e3:7.2f}  min {min(xs) * 1e3:7.2f}  max {max(xs) * 1e3:7.2f} ms"


def _summarise(tag, records):
    # Skip the first two steps: UniPC's order taper makes them structurally different and the
    # very first step also pays one-off host work (rope features, latent upload).
    body = records[2:] if len(records) > 4 else records
    wall = [r["wall"] for r in body]
    trace = [r["trace"] for r in body]
    sh = [r["solver_host"] for r in body]
    ss = [r["solver_sync"] for r in body]
    other = [r["wall"] - r["trace"] - r["solver_host"] - r["solver_sync"] for r in body]
    print(f"GAP [{tag}] steps={len(records)} (stats over {len(body)}), execute_trace blocking={records[0]['blocking']}")
    print(f"GAP [{tag}] wall/step         {_ms(wall)}")
    print(f"GAP [{tag}] execute_trace     {_ms(trace)}")
    print(f"GAP [{tag}] solver host       {_ms(sh)}")
    if any(ss):
        print(f"GAP [{tag}] solver sync tail  {_ms(ss)}")
    print(f"GAP [{tag}] other host        {_ms(other)}   (wall - trace - solver)")
    return {
        "wall": statistics.mean(wall),
        "trace": statistics.mean(trace),
        "solver_host": statistics.mean(sh),
        "solver_sync": statistics.mean(ss),
        "other": statistics.mean(other),
    }


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    [
        [(4, 8), (4, 8), ring_params_req_exact_devices, ttnn.Topology.Ring],
    ],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_step_gap_ti2v_5b(mesh_device, mesh_shape, topology):
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    height = int(os.environ.get("WAN5B_GAP_HEIGHT", 704))
    width = int(os.environ.get("WAN5B_GAP_WIDTH", 1280))
    num_frames = int(os.environ.get("WAN5B_GAP_FRAMES", 81))
    steps = int(os.environ.get("WAN5B_GAP_STEPS", 40))

    # Same construction as the perf test: eager cache warmup, then a 2-step traced run that
    # captures the trace, so the timed passes below execute it exactly as production does.
    pipeline = WanTI2V5BPipeline.create_pipeline(
        mesh_device=mesh_device, height=height, width=width, num_frames=num_frames, run_warmup=True
    )

    def _run():
        with torch.no_grad():
            return pipeline(prompts=[_PROMPT], num_inference_steps=steps, seed=42, output_type="latent", traced=True)

    with torch.no_grad():
        pipeline(prompts=[_PROMPT], num_inference_steps=2, traced=True)
    ttnn.synchronize_device(mesh_device)

    logger.info(f"step gap: {width}x{height}, {num_frames}f, {steps} traced steps")

    with _StepTimer(pipeline, mesh_device, sync_after_solver=False) as ta:
        _run()
    a = _summarise("A production", ta.records)

    with _StepTimer(pipeline, mesh_device, sync_after_solver=True) as tb:
        _run()
    b = _summarise("B sync-after-solver", tb.records)

    solver_device = b["solver_host"] + b["solver_sync"]
    gap = a["wall"] - a["trace"] - solver_device
    print(f"GAP solver device time (pass B host + sync)   {solver_device * 1e3:7.2f} ms/step")
    print(f"GAP device busy (trace A + solver device B)    {(a['trace'] + solver_device) * 1e3:7.2f} ms/step")
    print(
        f"GAP HOST GAP = wall A - trace A - solver dev   {gap * 1e3:7.2f} ms/step  ({gap / a['wall'] * 100:.1f}% of wall)"
    )

    pipeline.release_traces()
    assert len(ta.records) == steps and len(tb.records) == steps
