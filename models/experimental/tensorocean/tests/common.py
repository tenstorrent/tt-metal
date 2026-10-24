# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared by the tests and the demo: the two versions behind one interface, the float64 reference, metrics, timing.

Both versions are timed per model time step with everything that changes per step done on the chip: the per-step
inputs (tracer values, f, mask) start in DRAM in their natural row-major layout and the outputs end as natural
[levels, N/2, N] arrays. Mesh constants are prepared once. The host -> device copy is not timed (as in LANL's
optimized_gpu.py, which reports it separately as data movement).
"""
import statistics
import time

import torch
import ttnn

from models.experimental.tensorocean.reference import optimized_ttnn as ref
from models.experimental.tensorocean.tt import tensorocean as opt
from models.experimental.tensorocean.tt.natural_io import STEP_INPUTS, natural_host, upload_natural

PCC_MIN = 0.99999
RMS_REL_MAX = 1e-6  # fp32: measured ~1e-7


class Baseline:
    """reference/optimized_ttnn.horizontal_flux_ttnn as received (fp32). Per step it tilizes the natural inputs on
    the chip; the per-level copies of the mesh constants are made once, as the reference's to_device_inputs does."""

    traceable = False  # the reference synchronizes the device inside the function

    @staticmethod
    def prepare(host, n, levels, device):
        inputs = ref.to_device_inputs(host, device, ttnn.float32)
        for k in STEP_INPUTS:
            ttnn.deallocate(inputs.pop(k))
        return dict(inputs=inputs, nat=upload_natural(natural_host(host), device), n=n, levels=levels, device=device)

    @staticmethod
    def run(s):
        step = {
            k: ttnn.to_layout(
                ttnn.reshape(t, list(t.shape) + [1]), ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for k, t in s["nat"].items()
        }
        even, odd, _, _ = ref.horizontal_flux_ttnn({**s["inputs"], **step}, s["n"], s["levels"], s["device"])
        for t in step.values():
            ttnn.deallocate(t)
        return tuple(ttnn.reshape(o, [s["levels"], s["n"] // 2, s["n"]]) for o in (even, odd))


class Optimized:
    traceable = True

    @staticmethod
    def prepare(host, n, levels, device):
        return opt.prepare(host, n, levels, device)

    @staticmethod
    def run(s):
        return opt.run(s)


VERSIONS = {"baseline": Baseline, "optimized": Optimized}


def reference_outputs(host, n):
    """The algorithm of LANL's optimized_cpu.py (reference.horizontal_flux_torch) in float64."""
    return ref.horizontal_flux_torch({k: v.double() for k, v in host.items()}, n)


def metrics(truth, got):
    got = got.double().reshape(truth.shape)
    t, g = truth.reshape(-1), got.reshape(-1)
    tc, gc = t - t.mean(), g - g.mean()
    return dict(
        pcc=(tc @ gc / (tc.norm() * gc.norm())).item(),
        rms_rel=((g - t).pow(2).mean().sqrt() / t.pow(2).mean().sqrt()).item(),
        finite=bool(torch.isfinite(g).all()),
    )


def check(version, device, n, levels, seed=0):
    host = ref.make_inputs(n, levels, seed)
    truth = reference_outputs(host, n)
    s = version.prepare(host, n, levels, device)
    outs = version.run(s)
    got = [ttnn.to_torch(o).float() for o in outs]
    return [metrics(t, g) for t, g in zip(truth, got)], s


def time_per_step(version, device, s, reps=20):
    """Median over 3 samples of `reps` back-to-back steps (traced when the version allows it), in seconds."""
    version.run(s)
    ttnn.synchronize_device(device)
    if version.traceable:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        version.run(s)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        call = lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    else:
        call = lambda: version.run(s)
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            call()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) / reps)
    if version.traceable:
        ttnn.release_trace(device, tid)
    return statistics.median(samples)
