"""t227 job F: DiffVAE 2-D stage 5 (DIFFVAE_S5_2D=1), one arm per process (ARM env, fidelity via DIFFVAE_NA_FIDELITY).

1. production path (device stage-5 noise): warm-up, then SEEDS timed decodes;
2. one decode with the timing tree AND deep (per-op) spans -> stage_tree_<arm>.txt;
3. if HOST_SEEDS is set: host-noise decodes (the #214 reference noise) written as
   <out>/<arm>/ref_dvx_seed{N}.yuv, for cmpS.py against diffvae/ref.
Derived from t222 decodeE.py / decode222.py.
"""

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
from models.tt_dit.utils import timing_tree

lat_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
arm = os.environ["ARM"]
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
host_seeds = [int(s) for s in os.environ.get("HOST_SEEDS", "").split(",") if s]
yuv_dir = out_dir / arm
yuv_dir.mkdir(parents=True, exist_ok=True)


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
    print(f"[t227] {arm} decoder {decoder.options} loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    latents = {s: torch.load(lat_dir / f"seed{s}.pt") for s in sorted(set(seeds + host_seeds))}
    first = seeds[0]
    warm, secs = decode(decoder, latents[first], first)
    print(f"[t227] {arm} warm-up decode seed {first}: {secs:.2f}s", flush=True)
    for s in seeds:
        yuv, secs = decode(decoder, latents[s], s)
        times[s] = secs
        print(f"[t227] {arm} DECODE seed {s}: {secs:.3f}s", flush=True)

    timing_tree.ENABLED = True
    timing_tree.DEEP = True
    with timing_tree.span(mesh, "decode (profiled, deep)", root=True):
        _, psecs = decode(decoder, latents[first], first)
    tree = timing_tree.render(
        timing_tree.roots()[-1], title=f"t227F {arm} seed {first} deep profile (device noise)", measured_ms=psecs * 1000
    )
    print(tree, flush=True)
    (out_dir / f"stage_tree_{arm}.txt").write_text(tree + "\n")
    timing_tree.ENABLED = False
    timing_tree.DEEP = False

    if host_seeds:
        orig = decoder.stage5.forward

        def forward(context, noise, timestep, grid, **kw):
            assert noise is None
            shape = (1, decoder.out_channels, grid.t, grid.h * decoder.patch_size, grid.w * decoder.patch_size)
            noise = torch.randn(shape, generator=torch.Generator().manual_seed(kw["seed"]))
            return orig(context, noise, timestep, grid, **kw)

        decoder.stage5.forward = forward
        for s in host_seeds:
            yuv, secs = decode(decoder, latents[s], s)
            host_times[s] = secs
            tmp = yuv_dir / f"ref_dvx_seed{s}.yuv.tmp"
            yuv.tofile(tmp)
            tmp.rename(yuv_dir / f"ref_dvx_seed{s}.yuv")
            print(f"[t227] {arm} host-noise seed {s}: {secs:.3f}s", flush=True)

(out_dir / f"decode_times_{arm}.json").write_text(json.dumps({str(s): t for s, t in times.items()}, indent=1))
if host_times:
    (yuv_dir / "decode_times.json").write_text(json.dumps({str(s): t for s, t in host_times.items()}, indent=1))
print(f"[t227] {arm} DECODE_MEAN_S {sum(times.values()) / len(times):.3f} over {sorted(times)}", flush=True)
