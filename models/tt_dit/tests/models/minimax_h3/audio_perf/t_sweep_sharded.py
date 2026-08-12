"""Attribute the sharded path's cost: T-sweep at a given t_factor, same method as t_sweep.py.

Item 1 split the single-chip decode into ~260 ms fixed + ~670 ms data-proportional by sweeping T,
which works because op count does not depend on T. The same trick attributes the sharded path's
overhead without relying on absolute profiler totals (which are not calibrated for this stage).

row_model.py predicts factor=8 at ~360 ms; the test records 0.898 s. The ~540 ms gap is either

  fixed  -> sharding *added ops* (halo CCLs, tpad masks, layout conversions). The intercept rises far
            above 260 ms and the fix is to remove ops, not to add chips.
  data   -> the per-row cost got worse under sharding (smaller per-chip tensors, worse occupancy).
            The intercept stays near 260 ms and the slope fails to fall by ``factor``.

Run:  T_FACTOR=4 MESH_AXIS=0 python t_sweep_sharded.py
"""

import json
import os
import statistics
import time

import torch
from safetensors.torch import load_file

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

N = int(os.environ.get("BENCH_N", "3"))
T_FACTOR = int(os.environ.get("T_FACTOR", "4"))
MESH_AXIS = int(os.environ.get("MESH_AXIS", "0"))
# T must stay a multiple of 32*factor or _upload_BCT pads it, which would blur the sweep.
align = 32 * T_FACTOR
default_ts = ",".join(str(m * align) for m in (7, 5, 4, 3, 2, 1)) if T_FACTOR == 1 else None
TS = [int(x) for x in os.environ.get("SWEEP_TS", default_ts or "").split(",") if x]
if not TS:
    TS = [m * align for m in (7, 5, 4, 3, 2, 1)]

wd = os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "audio_vae")
cfg = {k: v for k, v in json.load(open(os.path.join(wd, "config.json"))).items() if not k.startswith("_")}

mesh = ttnn.MeshShape(4, 8)
# open_mesh_device takes no fabric_config; conftest.py:496 sets it separately before opening, and the
# CCL ops need FABRIC_1D. Setting it after open is too late.
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
d = ttnn.open_mesh_device(mesh, l1_small_size=65536)
try:
    pc = None if T_FACTOR <= 1 else ParallelFactor(factor=T_FACTOR, mesh_axis=MESH_AXIS)
    ccl = None if pc is None else CCLManager(d, num_links=1, topology=ttnn.Topology.Linear)
    dec = MiniMaxH3AudioDecoder(
        latent_channels=cfg["latent_channels"],
        latent_dim=cfg["latent_dim"],
        decoder_dim=cfg["decoder_dim"],
        decoder_rates=tuple(cfg["decoder_rates"]),
        decoder_kernel_sizes=tuple(cfg["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(cfg["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(x) for x in cfg["resblock_dilation_sizes"]),
        mesh_device=d,
        parallel_config=pc,
        ccl_manager=ccl,
    )
    dec.load_torch_state_dict(
        convert_minimax_h3_audio_state_dict(load_file(os.path.join(wd, "diffusion_pytorch_model.safetensors"))),
        strict=False,
    )
    torch.manual_seed(2)
    print(f"t_factor={T_FACTOR} mesh_axis={MESH_AXIS} align={align} TS={TS}", flush=True)
    rows = []
    for T in TS:
        lat = torch.randn(2, cfg["latent_channels"], T) * 0.1
        dec(lat)
        ttnn.synchronize_device(d)
        s = []
        for _ in range(N):
            t0 = time.perf_counter()
            dec(lat)
            ttnn.synchronize_device(d)
            s.append(time.perf_counter() - t0)
        m = statistics.median(s)
        rows.append((T, m))
        print(f"SWEEP f={T_FACTOR} T={T:>5}  median {m:.4f}s  min {min(s):.4f}  max {max(s):.4f}", flush=True)

    n = len(rows)
    sx = sum(T for T, _ in rows)
    sy = sum(m for _, m in rows)
    sxx = sum(T * T for T, _ in rows)
    sxy = sum(T * m for T, m in rows)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    print(f"\nfit f={T_FACTOR}:  t(T) = {a * 1e3:.1f} ms + {b * 1e6:.1f} us/latent * T")
    print(f"  at T=224: fixed {a * 1e3:.1f} ms + data {b * 224 * 1e3:.1f} ms = {(a + b * 224) * 1e3:.1f} ms")
    print(f"\n  single-chip reference (item 1): 260 ms fixed + 3372 us/latent")
    print(f"  if sharding were clean, fixed would stay ~260 ms and slope would fall ~{T_FACTOR}x")
finally:
    ttnn.close_mesh_device(d)
    # conftest.py:484 disables fabric after closing; leaving it set can wedge the next job's init.
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
