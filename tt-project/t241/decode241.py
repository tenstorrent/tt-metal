"""t241: DiffVAE decode through the decoder's forward, eager (ARM=def) or traced (ARM=traced, DIFFVAE_TRACED=1).

1. warm-up forward on seed SEEDS[0] (eager in both arms; in the traced arm it marks the shape);
2. traced arm: one capture+run on SEEDS[0], timed apart;
3. SEEDS timed decodes (device noise), seed-0 yuv md5 printed for the eager/traced bit-identity check;
4. HOST_SEEDS host-noise decodes (the #214 reference noise, passed as forward(noise=...), which the
   traced arm embeds outside the trace and feeds in as its input) -> <out>/<arm>/ref_dvx_seed{N}.yuv.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import DiffVAEOptions
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tools.diffvae_bench import heartbeat, loaded_production_decoder, open_mesh

lat_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
arm = os.environ["ARM"]
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
host_seeds = [int(s) for s in os.environ.get("HOST_SEEDS", "").split(",") if s]
yuv_dir = out_dir / arm
yuv_dir.mkdir(parents=True, exist_ok=True)


def run(decoder, latent, seed, noise=None):
    t0 = time.perf_counter()
    yuv = decoder.forward(latent, seed=seed, noise=noise, output_type="yuv")
    ttnn.synchronize_device(decoder.mesh_device)
    secs = time.perf_counter() - t0
    if isinstance(yuv, torch.Tensor):
        yuv = yuv.numpy()
    return np.ascontiguousarray(yuv), secs


times, host_times = {}, {}
with heartbeat(), open_mesh((4, 8), fabric=ttnn.FabricConfig.FABRIC_1D_RING, trace_region_size=400_000_000) as mesh:
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Linear)
    t0 = time.perf_counter()
    decoder, _ = loaded_production_decoder(mesh, DiffVAEOptions.production(), ccl)
    print(
        f"[t241] {arm} DIFFVAE_TRACED={os.environ.get('DIFFVAE_TRACED')} loaded in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    latents = {s: torch.load(lat_dir / f"seed{s}.pt") for s in sorted(set(seeds + host_seeds))}
    first = seeds[0]
    _, secs = run(decoder, latents[first], first)
    print(f"[t241] {arm} warm-up seed {first}: {secs:.2f}s", flush=True)
    if arm == "traced":
        _, secs = run(decoder, latents[first], first)
        print(f"[t241] {arm} capture+run seed {first}: {secs:.2f}s", flush=True)
    for s in seeds:
        yuv, secs = run(decoder, latents[s], s)
        times[s] = secs
        print(f"[t241] {arm} DECODE seed {s}: {secs:.3f}s md5={hashlib.md5(yuv.tobytes()).hexdigest()}", flush=True)
    for s in seeds:  # second pass: replay stability
        _, secs = run(decoder, latents[s], s)
        print(f"[t241] {arm} DECODE2 seed {s}: {secs:.3f}s", flush=True)

    for s in host_seeds:
        g = decoder.stage5_grid(*latents[s].shape[2:])
        shape = (1, decoder.out_channels, g.t, g.h * decoder.patch_size, g.w * decoder.patch_size)
        noise = torch.randn(shape, generator=torch.Generator().manual_seed(s))
        yuv, secs = run(decoder, latents[s], s, noise=noise)
        host_times[s] = secs
        tmp = yuv_dir / f"ref_dvx_seed{s}.yuv.tmp"
        yuv.tofile(tmp)
        tmp.rename(yuv_dir / f"ref_dvx_seed{s}.yuv")
        print(f"[t241] {arm} host-noise seed {s}: {secs:.3f}s", flush=True)
    decoder.release_trace()

(out_dir / f"decode_times_{arm}.json").write_text(json.dumps({str(s): t for s, t in times.items()}, indent=1))
if host_times:
    (yuv_dir / "decode_times.json").write_text(json.dumps({str(s): t for s, t in host_times.items()}, indent=1))
print(f"[t241] {arm} DECODE_MEAN_S {sum(times.values()) / len(times):.3f} over {sorted(times)}", flush=True)
