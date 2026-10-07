"""t225: production DiffVAE decode (device stage-5 noise) of the #214 seed latents, one arm per process (ARM env:
1d = default, 2d = run with DIFFVAE_S5_2D=1). Derived from t222's decodeE.py: full 4x8 mesh, VAE CCL as the pipeline
builds it (2 links, Linear), one warm-up decode, then timed decodes of SEEDS. Writes {arm}_seed{N}.yuv (yuv420p,
1920x1088, 145 frames) after the timer stops, and decode_times_{arm}.json. Seeds already on disk are skipped.
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

lat_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
arm = os.environ["ARM"]
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
todo = [s for s in seeds if not (out_dir / f"{arm}_seed{s}.yuv").exists()]
print(f"[t225] {arm} seeds todo {todo} DIFFVAE_S5_2D={os.environ.get('DIFFVAE_S5_2D', '0')}", flush=True)
if not todo:
    sys.exit(0)


def decode(decoder, latent, seed):
    t0 = time.perf_counter()
    yuv = decoder.decode(latent, seed=seed, output_type="yuv")
    ttnn.synchronize_device(decoder.mesh_device)
    if isinstance(yuv, torch.Tensor):
        yuv = yuv.numpy()
    return yuv, time.perf_counter() - t0


tpath = out_dir / f"decode_times_{arm}.json"
times = json.loads(tpath.read_text()) if tpath.exists() else {}
with heartbeat(), open_mesh((4, 8), fabric=ttnn.FabricConfig.FABRIC_1D_RING) as mesh:
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Linear)
    t0 = time.perf_counter()
    decoder, _ = loaded_production_decoder(mesh, DiffVAEOptions.production(), ccl)
    print(f"[t225] {arm} decoder {decoder.options} loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    if hasattr(decoder.stage5, "_brick"):
        print(f"[t225] {arm} stage5 brick {decoder.stage5._brick}", flush=True)
    latents = {s: torch.load(lat_dir / f"seed{s}.pt") for s in todo}
    warm, secs = decode(decoder, latents[todo[0]], todo[0])
    print(f"[t225] {arm} warm-up seed {todo[0]}: {secs:.2f}s shape {tuple(warm.shape)} {warm.dtype}", flush=True)
    for s in todo:
        yuv, secs = decode(decoder, latents[s], s)
        if s == todo[0]:
            print(f"[t225] {arm} seed {s} warm-up vs timed identical: {bool((warm == yuv).all())}", flush=True)
        tmp = out_dir / f"{arm}_seed{s}.yuv.tmp"
        yuv.tofile(tmp)
        tmp.rename(out_dir / f"{arm}_seed{s}.yuv")
        times[str(s)] = secs
        tpath.write_text(json.dumps(times, indent=1))
        print(f"[t225] {arm} DECODE seed {s}: {secs:.3f}s {tuple(yuv.shape)}", flush=True)
print(f"[t225] {arm} DECODE_MEAN_S {sum(times.values()) / len(times):.3f} over {sorted(times)}", flush=True)
