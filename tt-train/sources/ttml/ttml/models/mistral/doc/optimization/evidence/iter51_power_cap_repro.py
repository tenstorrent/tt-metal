"""Power-cap repro (hardware characterisation, no model code): the gate_up base matmul of the step
([16384, 5184] x [1024, 5184]^T, bf16, fp32 accumulate, the committed 2D-mcast config) looped on all four chips of the
1x4 mesh through the phases idle / HiFi4 / cool-down / HiFi2 / idle while iter51_telemetry_sampler.py reads the ARC
telemetry at 20 Hz in a child process. Writes <out>.json: the phase boundaries (wall-clock timestamps aligned with the
sampler's), and per batch of matmul calls the timestamp and the achieved TFLOP/s per chip (host-timed over the mesh,
so the slowest chip's rate). HiFi2 is measured only as the lower-energy comparison for the clock: it is not a candidate
for the model (the recipe mandates HiFi4). Usage: repro.py <out-prefix> [seconds per load phase]."""
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch
import ttnn
import ttml
from ttml.modules.frozen_linear import forward_transposed_matmul_kwargs

out_prefix = sys.argv[1]
load_s = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
here = os.path.dirname(os.path.abspath(__file__))
sampler = subprocess.Popen(
    [
        "/home/ttuser/.tenstorrent-venv/bin/python",
        os.path.join(here, "iter51_telemetry_sampler.py"),
        out_prefix + "_telemetry.csv",
        str(2 * load_s + 120),
        "20",
    ]
)
mesh = ttml.Mesh((1, 4), ("dp", "tp"))
ttml.open_device_mesh(mesh)
device = ttml.autograd.AutoContext.get_instance().get_device()
M, K, N = 16384, 5184, 1024


def dev(a):
    return ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )


w = dev(torch.randn(1, 1, M, K) * 0.02)
x = dev(torch.randn(1, 1, N, K))
kw = forward_transposed_matmul_kwargs(K, M, N)
assert kw is not None
flops = 2.0 * M * K * N
phases = []
batches = []


def phase(name, seconds, fidelity=None):
    t0 = time.time()
    phases.append({"name": name, "start": t0})
    if fidelity is None:
        time.sleep(seconds)
    else:
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
        )
        calls = 16
        while time.time() - t0 < seconds:
            tb = time.perf_counter()
            for _ in range(calls):
                y = ttnn.matmul(w, x, transpose_b=True, compute_kernel_config=cfg, **kw)
                ttnn.deallocate(y)
            ttnn.synchronize_device(device)
            dt = (time.perf_counter() - tb) / calls
            batches.append(
                {"t": time.time(), "phase": name, "us_per_call": dt * 1e6, "tflops_per_chip": flops / dt / 1e12}
            )
    phases[-1]["end"] = time.time()
    print(f"{name}: {phases[-1]['end'] - t0:.1f} s", flush=True)


phase("idle", 8.0)
phase("HiFi4", load_s, ttnn.MathFidelity.HiFi4)
phase("cool-down", 25.0)
phase("HiFi2", load_s, ttnn.MathFidelity.HiFi2)
phase("idle (end)", 8.0)
json.dump({"phases": phases, "batches": batches, "shape": [M, K, N]}, open(out_prefix + ".json", "w"), indent=1)
ttml.autograd.AutoContext.get_instance().close_device()
time.sleep(2)
sampler.terminate()
sampler.wait()
for p in ("HiFi4", "HiFi2"):
    rates = [b["tflops_per_chip"] for b in batches if b["phase"] == p]
    print(f"{p}: first batch {rates[0]:.1f} TFLOP/s per chip, last {rates[-1]:.1f}, mean {np.mean(rates):.1f}")
