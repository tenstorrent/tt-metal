"""t262: DiffVAE 1080p 145f decode on 4x8, ONE arm per process (shipped t48 defaults + ARM_ENV).

ARM=<name>, ARM_ENV="K=V,K=V" (set before the decoder is built). Load, warm-up, SEEDS timed device-noise
decodes, optional PROFILE=1 deep timing tree (stage_tree_<arm>.txt), then HOST_SEEDS host-noise decodes
(#214 reference noise) written as <out>/<arm>/ref_dvx_seed{N}.yuv for cmp241.py. Derived from t240/t246.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

arm = os.environ["ARM"]
for kv in filter(None, os.environ.get("ARM_ENV", "").split(",")):
    k, v = kv.split("=", 1)
    os.environ[k] = v

import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import DiffVAEOptions
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tools.diffvae_bench import heartbeat, loaded_production_decoder, open_mesh
from models.tt_dit.utils import timing_tree

lat_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
host_seeds = [int(s) for s in os.environ.get("HOST_SEEDS", "").split(",") if s]
profile = os.environ.get("PROFILE") == "1"


def decode(decoder, latent, seed):
    t0 = time.perf_counter()
    yuv = decoder.decode(latent, seed=seed, output_type="yuv")
    ttnn.synchronize_device(decoder.mesh_device)
    if isinstance(yuv, torch.Tensor):
        yuv = yuv.numpy()
    return yuv, time.perf_counter() - t0


times, host_times = {}, {}
with heartbeat(), open_mesh((4, 8), fabric=ttnn.FabricConfig.FABRIC_1D_RING) as mesh:
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Linear)
    t0 = time.perf_counter()
    decoder, _ = loaded_production_decoder(mesh, DiffVAEOptions.production(), ccl)
    print(f"[t262] {arm} env={os.environ.get('ARM_ENV', '')} loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    latents = {s: torch.load(lat_dir / f"seed{s}.pt") for s in sorted(set(seeds + host_seeds))}
    first = seeds[0]
    _, secs = decode(decoder, latents[first], first)
    print(f"[t262] {arm} warm-up decode seed {first}: {secs:.2f}s", flush=True)
    for s in seeds:
        yuv, secs = decode(decoder, latents[s], s)
        times[s] = secs
        print(f"[t262] {arm} DECODE seed {s}: {secs:.3f}s md5={hashlib.md5(yuv.tobytes()).hexdigest()}", flush=True)
    print(f"[t262] {arm} DECODE_MEAN_S {sum(times.values()) / len(times):.3f} over {sorted(times)}", flush=True)

    if profile:
        timing_tree.ENABLED = True
        timing_tree.DEEP = True
        with timing_tree.span(mesh, "decode (profiled, deep)", root=True):
            _, psecs = decode(decoder, latents[first], first)
        tree = timing_tree.render(
            timing_tree.roots()[-1], title=f"t262 {arm} seed {first} deep profile", measured_ms=psecs * 1000
        )
        print(tree, flush=True)
        (out_dir / f"stage_tree_{arm}.txt").write_text(tree + "\n")
        timing_tree.ENABLED = False
        timing_tree.DEEP = False

    if host_seeds:
        orig = decoder.stage5.forward

        def host_noise_forward(context, noise, timestep, grid, **kw):
            assert noise is None
            shape = (1, decoder.out_channels, grid.t, grid.h * decoder.patch_size, grid.w * decoder.patch_size)
            noise = torch.randn(shape, generator=torch.Generator().manual_seed(kw["seed"]))
            return orig(context, noise, timestep, grid, **kw)

        decoder.stage5.forward = host_noise_forward
        yuv_dir = out_dir / arm
        yuv_dir.mkdir(parents=True, exist_ok=True)
        for s in host_seeds:
            yuv, secs = decode(decoder, latents[s], s)
            host_times[s] = secs
            tmp = yuv_dir / f"ref_dvx_seed{s}.yuv.tmp"
            yuv.tofile(tmp)
            tmp.rename(yuv_dir / f"ref_dvx_seed{s}.yuv")
            print(f"[t262] {arm} host-noise seed {s}: {secs:.3f}s", flush=True)
        (yuv_dir / "decode_times.json").write_text(json.dumps({str(s): t for s, t in host_times.items()}, indent=1))
    (out_dir / f"device_times_{arm}.json").write_text(json.dumps({str(s): t for s, t in times.items()}, indent=1))
