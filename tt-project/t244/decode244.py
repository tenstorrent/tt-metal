"""t244: DiffVAE NA compute-config A/B in ONE process on t48 @34a571c5f47 (shipped defaults).

The NA op reads DIFFVAE_NA_FIDELITY / DIFFVAE_NA_APPROX_EXP on every call, so the arms share one
loaded decoder: per arm set the env, warm-up decode (compiles that arm's NA kernels), SEEDS timed
device-noise decodes, then HOST_SEEDS host-noise decodes (#214 reference noise) written as
<out>/<arm>/ref_dvx_seed{N}.yuv for cmp241.py. A final def re-time checks for drift.
Derived from t243 decode242.py.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import DiffVAEOptions
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tools.diffvae_bench import heartbeat, loaded_production_decoder, open_mesh

lat_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
host_seeds = [int(s) for s in os.environ.get("HOST_SEEDS", "").split(",") if s]
ARMS = {
    "def": {},
    "approx": {"DIFFVAE_NA_APPROX_EXP": "1"},
    "lofi": {"DIFFVAE_NA_FIDELITY": "lofi"},
}
arms = os.environ.get("ARMS", ",".join(ARMS)).split(",")


def set_arm(arm):
    for k in ("DIFFVAE_NA_APPROX_EXP", "DIFFVAE_NA_FIDELITY"):
        os.environ.pop(k, None)
    os.environ.update(ARMS[arm])


def decode(decoder, latent, seed):
    t0 = time.perf_counter()
    yuv = decoder.decode(latent, seed=seed, output_type="yuv")
    ttnn.synchronize_device(decoder.mesh_device)
    if isinstance(yuv, torch.Tensor):
        yuv = yuv.numpy()
    return yuv, time.perf_counter() - t0


summary = {}
with heartbeat(), open_mesh((4, 8), fabric=ttnn.FabricConfig.FABRIC_1D_RING) as mesh:
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Linear)
    t0 = time.perf_counter()
    decoder, _ = loaded_production_decoder(mesh, DiffVAEOptions.production(), ccl)
    print(f"[t244] decoder {decoder.options} loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    latents = {s: torch.load(lat_dir / f"seed{s}.pt") for s in sorted(set(seeds + host_seeds))}
    orig = decoder.stage5.forward

    def host_noise_forward(context, noise, timestep, grid, **kw):
        assert noise is None
        shape = (1, decoder.out_channels, grid.t, grid.h * decoder.patch_size, grid.w * decoder.patch_size)
        noise = torch.randn(shape, generator=torch.Generator().manual_seed(kw["seed"]))
        return orig(context, noise, timestep, grid, **kw)

    def run_arm(arm, host=True):
        set_arm(arm)
        decoder.stage5.forward = orig
        first = seeds[0]
        _, secs = decode(decoder, latents[first], first)
        print(f"[t244] {arm} warm-up decode seed {first}: {secs:.2f}s env={ARMS[arm]}", flush=True)
        times = {}
        for s in seeds:
            yuv, secs = decode(decoder, latents[s], s)
            times[s] = secs
            print(f"[t244] {arm} DECODE seed {s}: {secs:.3f}s md5={hashlib.md5(yuv.tobytes()).hexdigest()}", flush=True)
        print(f"[t244] {arm} DECODE_MEAN_S {sum(times.values()) / len(times):.3f} over {sorted(times)}", flush=True)
        if not host or not host_seeds:
            return times
        yuv_dir = out_dir / arm
        yuv_dir.mkdir(parents=True, exist_ok=True)
        decoder.stage5.forward = host_noise_forward
        host_times = {}
        for s in host_seeds:
            yuv, secs = decode(decoder, latents[s], s)
            host_times[s] = secs
            tmp = yuv_dir / f"ref_dvx_seed{s}.yuv.tmp"
            yuv.tofile(tmp)
            tmp.rename(yuv_dir / f"ref_dvx_seed{s}.yuv")
            print(f"[t244] {arm} host-noise seed {s}: {secs:.3f}s", flush=True)
        decoder.stage5.forward = orig
        (yuv_dir / "decode_times.json").write_text(json.dumps({str(s): t for s, t in host_times.items()}, indent=1))
        return times

    for arm in arms:
        summary[arm] = run_arm(arm)
        (out_dir / "decode_times_arms.json").write_text(json.dumps(summary, indent=1))
    if "def" in arms and len(arms) > 1:
        summary["def_recheck"] = run_arm("def", host=False)
        (out_dir / "decode_times_arms.json").write_text(json.dumps(summary, indent=1))

for arm, t in summary.items():
    print(f"[t244] SUMMARY {arm} mean {sum(t.values()) / len(t):.3f}s {t}", flush=True)
