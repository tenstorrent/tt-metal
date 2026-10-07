"""#216: DiffVAE stage-5 GNA stride A/B on 4x8, both arms in one process on the t212 port build.

Arms (ARMS, default "1x1x1,2x4x4"): DiffVAEOptions.production() with gna_stride replaced, which is what
DIFFVAE_GNA_STRIDE does on t48. At stride > 1 the stage-5 brick is (2,4,4) (neighborhood_choose_brick).
Stage-5 x_t noise comes from the host exactly as #214's decode_ref.py draws it, so every arm and the #214
reference see the same noise. Per arm: load, one warm-up decode, then timed decodes of seeds 0-4.
stage5.forward is bracketed by device syncs to time it (includes x_t embed and the pixel pull).
Writes s{arm}_seed{N}.yuv (yuv420p 1920x1088, 145 frames) and times.json. Existing outputs are skipped.
"""

import dataclasses
import gc
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
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
arms = os.environ.get("ARMS", "1x1x1,2x4x4").split(",")
tpath = out_dir / "times.json"
times = json.loads(tpath.read_text()) if tpath.exists() else {}


def instrument(decoder, acc):
    orig = decoder.stage5.forward

    def forward(context, noise, timestep, grid, **kw):
        assert noise is None
        shape = (1, decoder.out_channels, grid.t, grid.h * decoder.patch_size, grid.w * decoder.patch_size)
        noise = torch.randn(shape, generator=torch.Generator().manual_seed(kw["seed"]))
        ttnn.synchronize_device(decoder.mesh_device)
        t0 = time.perf_counter()
        out = orig(context, noise, timestep, grid, **kw)
        ttnn.synchronize_device(decoder.mesh_device)
        acc.append(time.perf_counter() - t0)
        return out

    decoder.stage5.forward = forward


def decode(decoder, acc, latent, seed):
    acc.clear()
    t0 = time.perf_counter()
    yuv = decoder.decode(latent, seed=seed, output_type="yuv")
    ttnn.synchronize_device(decoder.mesh_device)
    if isinstance(yuv, torch.Tensor):
        yuv = yuv.numpy()
    return yuv, time.perf_counter() - t0, sum(acc)


latents = {s: torch.load(lat_dir / f"seed{s}.pt") for s in seeds}
with heartbeat(), open_mesh((4, 8), fabric=ttnn.FabricConfig.FABRIC_1D_RING) as mesh:
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Linear)
    for arm in arms:
        todo = [s for s in seeds if not (out_dir / f"s{arm}_seed{s}.yuv").exists()]
        print(f"[t216] arm {arm} seeds todo {todo}", flush=True)
        if not todo:
            continue
        stride = tuple(int(v) for v in arm.split("x"))
        options = dataclasses.replace(DiffVAEOptions.production(), gna_stride=stride)
        t0 = time.perf_counter()
        decoder, _ = loaded_production_decoder(mesh, options, ccl)
        acc = []
        instrument(decoder, acc)
        print(f"[t216] arm {arm} decoder {decoder.options} loaded in {time.perf_counter() - t0:.1f}s", flush=True)
        warm, secs, s5 = decode(decoder, acc, latents[todo[0]], todo[0])
        print(f"[t216] arm {arm} warm-up seed {todo[0]}: total {secs:.2f}s stage5 {s5:.2f}s", flush=True)
        if hasattr(decoder.stage5, "_brick"):
            print(f"[t216] arm {arm} stage5 brick {decoder.stage5._brick}", flush=True)
        for s in todo:
            yuv, secs, s5 = decode(decoder, acc, latents[s], s)
            if s == todo[0]:
                print(f"[t216] arm {arm} seed {s} warm-up vs timed identical: {bool((warm == yuv).all())}", flush=True)
            tmp = out_dir / f"s{arm}_seed{s}.yuv.tmp"
            yuv.tofile(tmp)
            tmp.rename(out_dir / f"s{arm}_seed{s}.yuv")
            times.setdefault(arm, {})[str(s)] = {"total_s": secs, "stage5_s": s5, "stage5_calls": len(acc)}
            tpath.write_text(json.dumps(times, indent=1))
            print(
                f"[t216] DECODE arm {arm} seed {s}: total {secs:.3f}s stage5 {s5:.3f}s ({len(acc)} calls)", flush=True
            )
        del decoder, warm
        gc.collect()
for arm, per in times.items():
    tot = [v["total_s"] for v in per.values()]
    s5 = [v["stage5_s"] for v in per.values()]
    print(
        f"[t216] MEAN arm {arm}: total {sum(tot) / len(tot):.3f}s stage5 {sum(s5) / len(s5):.3f}s over {sorted(per)}",
        flush=True,
    )
