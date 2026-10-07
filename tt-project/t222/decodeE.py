"""t222 job E: production-path timing (device stage-5 noise, no host randn), one arm per process (ARM env).
Derived from decode222.py:

Original docstring (job B, the unoptimized DiffVAE reference decodes of the saved seed latents):

t48 5e4e0cd643a, DiffVAEOptions.production(), untraced, full 4x8 mesh, VAE CCL as the pipeline builds it
(2 links, Linear). Stage-5 x_t noise comes from the host, torch.randn(shape, Generator().manual_seed(seed)),
with the shape the decoder's own host-noise path draws, so any later arm can feed the same noise.
Writes ref_dvx_seed{N}.yuv (yuv420p, 1920x1088, 145 frames, the bytes the mp4 export would encode).
Seeds with an output already on disk are skipped, so a rerun resumes.
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
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
todo = seeds
print(f"[t214] seeds todo {todo}", flush=True)
arm = os.environ["ARM"]
tree_path = out_dir / f"stage_tree_{arm}.txt"
if not todo and tree_path.exists():
    sys.exit(0)


def host_noise_forward(decoder):
    orig = decoder.stage5.forward

    def forward(context, noise, timestep, grid, **kw):
        assert noise is None
        shape = (1, decoder.out_channels, grid.t, grid.h * decoder.patch_size, grid.w * decoder.patch_size)
        noise = torch.randn(shape, generator=torch.Generator().manual_seed(kw["seed"]))
        return orig(context, noise, timestep, grid, **kw)

    decoder.stage5.forward = forward


def decode(decoder, latent, seed):
    t0 = time.perf_counter()
    yuv = decoder.decode(latent, seed=seed, output_type="yuv")
    ttnn.synchronize_device(decoder.mesh_device)
    if isinstance(yuv, torch.Tensor):
        yuv = yuv.numpy()
    return yuv, time.perf_counter() - t0


times = {}
with heartbeat(), open_mesh((4, 8), fabric=ttnn.FabricConfig.FABRIC_1D_RING) as mesh:
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Linear)
    t0 = time.perf_counter()
    decoder, _ = loaded_production_decoder(mesh, DiffVAEOptions.production(), ccl)
    print(f"[t214] decoder {decoder.options} loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    latents = {s: torch.load(lat_dir / f"seed{s}.pt") for s in todo or seeds[:1]}
    first = (todo or seeds)[0]
    warm, secs = decode(decoder, latents[first], first)
    print(f"[t214] warm-up decode seed {first}: {secs:.2f}s shape {tuple(warm.shape)} {warm.dtype}", flush=True)
    for s in todo:
        yuv, secs = decode(decoder, latents[s], s)
        times[s] = secs
        if s == todo[0]:
            print(f"[t214] seed {s} warm-up vs timed identical: {bool((warm == yuv).all())}", flush=True)
        print(f"[t222E] {arm} DECODE seed {s}: {secs:.3f}s {tuple(yuv.shape)}", flush=True)
    # Stage breakdown: one more decode with the timing tree on. Its spans sync the device, so its
    # total is not comparable with the timed decodes above.
    timing_tree.ENABLED = True
    with timing_tree.span(mesh, "decode (profiled)", root=True):
        _, psecs = decode(decoder, latents[first], first)
    tree = timing_tree.render(
        timing_tree.roots()[-1], title=f"t222E {arm} seed {first} profiled (device noise)", measured_ms=psecs * 1000
    )
    print(tree, flush=True)
    tree_path.write_text(tree + "\n")

if not times:
    sys.exit(0)
prev = out_dir / f"decode_times_{arm}.json"
all_times = json.loads(prev.read_text()) if prev.exists() else {}
all_times.update({str(s): t for s, t in times.items()})
prev.write_text(json.dumps(all_times, indent=1))
print(f"[t222E] {arm} DECODE_MEAN_S {sum(times.values()) / len(times):.3f} over {sorted(times)}", flush=True)
