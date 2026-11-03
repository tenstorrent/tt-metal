"""Clock ladder (hardware characterisation, no model code): the chips through idle, a small matmul on 4 cores, a
medium one on 33 cores and the full gate_up matmul on 110 cores, all HiFi4 bf16 with fp32 accumulate, with the 20 Hz ARC
telemetry in a child process. Shows the clock holding 1350 MHz while the power stays under the 125 W cap and dropping
only when the load reaches it. Output as iter51_power_cap_repro.py's (<out>.json, <out>_telemetry.csv), plotted by
iter51_power_cap_plot.py. Usage: ladder.py <out-prefix> [seconds per load phase]."""
import json
import os
import subprocess
import sys
import time
import torch
import ttnn
import ttml
from ttml.modules.frozen_linear import forward_transposed_matmul_kwargs

out_prefix = sys.argv[1]
load_s = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
here = os.path.dirname(os.path.abspath(__file__))
sampler = subprocess.Popen(
    [
        "/home/ttuser/.tenstorrent-venv/bin/python",
        os.path.join(here, "iter51_telemetry_sampler.py"),
        out_prefix + "_telemetry.csv",
        str(3 * load_s + 90),
        "20",
    ]
)
mesh = ttml.Mesh((1, 4), ("dp", "tp"))
ttml.open_device_mesh(mesh)
device = ttml.autograd.AutoContext.get_instance().get_device()
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)


def dev(a):
    return ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )


M, K, N = 16384, 5184, 1024
w, x = dev(torch.randn(1, 1, M, K) * 0.02), dev(torch.randn(1, 1, N, K))
full_kw = forward_transposed_matmul_kwargs(K, M, N)
w_s, x_s = dev(torch.randn(1, 1, 1024, 1024) * 0.02), dev(torch.randn(1, 1, 1024, 1024))
loads = {
    "4 cores [1024 x 1024 x 1024]": (w_s, x_s, {"core_grid": ttnn.CoreGrid(y=2, x=2)}, 2.0 * 1024**3),
    "33 cores [16384 x 5184 x 1024]": (w, x, {"core_grid": ttnn.CoreGrid(y=3, x=11)}, 2.0 * M * K * N),
    "110 cores [16384 x 5184 x 1024]": (w, x, full_kw, 2.0 * M * K * N),
}
phases, batches = [], []


def phase(name, seconds, load=None):
    t0 = time.time()
    phases.append({"name": name, "start": t0})
    if load is None:
        time.sleep(seconds)
    else:
        a, b, kw, flops = load
        calls = 16
        while time.time() - t0 < seconds:
            tb = time.perf_counter()
            for _ in range(calls):
                y = ttnn.matmul(a, b, transpose_b=True, compute_kernel_config=cfg, **kw)
                ttnn.deallocate(y)
            ttnn.synchronize_device(device)
            dt = (time.perf_counter() - tb) / calls
            batches.append(
                {"t": time.time(), "phase": name, "us_per_call": dt * 1e6, "tflops_per_chip": flops / dt / 1e12}
            )
    phases[-1]["end"] = time.time()
    print(f"{name}: {phases[-1]['end'] - t0:.1f} s", flush=True)


phase("idle", 8.0)
for name, load in loads.items():
    phase(name, load_s, load)
    phase("pause", 6.0)
phase("idle (end)", 4.0)
json.dump(
    {
        "phases": phases,
        "batches": batches,
        "shape": [M, K, N],
        "title": "Clock ladder: HiFi4 bf16 matmuls on 4, 33 and 110 cores per chip on the 4-chip QB2 mesh (p300c, TDP 125 W per chip)",
    },
    open(out_prefix + ".json", "w"),
    indent=1,
)
ttml.autograd.AutoContext.get_instance().close_device()
time.sleep(2)
sampler.terminate()
sampler.wait()
