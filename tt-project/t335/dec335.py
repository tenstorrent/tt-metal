"""One arm of the conv VAE fidelity A/B: decode the saved LTX-2.5 latents on the full 4x8 mesh.

The arm is whatever LTX_VAE_CONV_FIDELITY says in this process's env (unset = HiFi4 default). The decoder is
built the way the bh_4x8sp1tp0_ring pipeline builds it: H on mesh axis 0 (4), W on axis 1 (8), VAE CCL Linear
with 2 links, FABRIC_1D_RING, num_frames/height/width given so the production conv3d blockings apply. Weights
come straight from the local safetensors (no cache write). Writes <out>/seed{N}.yuv (yuv420p, 145x1632x1920)
and times.json. Usage: dec335.py <latent_dir> <out_dir>
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.tt_dit.models.vae import vae_ltx
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tools.diffvae_bench import heartbeat, open_mesh

NF, H, W = 145, 1088, 1920
lat_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
reps = int(os.environ.get("REPS", "2"))
ckpt = os.environ["VAE_CKPT"]
arm = os.environ.get("LTX_VAE_CONV_FIDELITY", "") or "default"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[t335] arm={arm} ckpt={ckpt} seeds={seeds} reps={reps}", flush=True)


def vae_config_and_state(path):
    with open(path, "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    meta = json.loads(header.get("__metadata__", {}).get("config", "{}"))
    cfg = meta.get("vae", meta)
    state = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            short = k.removeprefix("vae.")
            if short.startswith("decoder."):
                state[short.removeprefix("decoder.")] = f.get_tensor(k)
            elif short in ("per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means"):
                state[short] = f.get_tensor(k)
    return cfg, state


def decode(dec, mesh, lat):
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    yuv = dec(lat, output_type="yuv")
    ttnn.synchronize_device(mesh)
    return torch.as_tensor(yuv).numpy(), time.perf_counter() - t0


cfg, state = vae_config_and_state(ckpt)
latents = {s: torch.load(lat_dir / f"seed{s}.pt") for s in seeds}
times = {}
with heartbeat(), open_mesh((4, 8), fabric=ttnn.FabricConfig.FABRIC_1D_RING) as mesh:
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=4, mesh_axis=0),
        width_parallel=ParallelFactor(factor=8, mesh_axis=1),
    )
    t0 = time.perf_counter()
    dec = vae_ltx.LTXVideoDecoder(
        decoder_blocks=cfg["decoder_blocks"],
        causal=cfg.get("causal_decoder", False),
        base_channels=cfg.get("decoder_base_channels", 128),
        num_frames=NF,
        height=H,
        width=W,
        mesh_device=mesh,
        parallel_config=pc,
        ccl_manager=CCLManager(mesh, num_links=2, topology=ttnn.Topology.Linear),
    )
    dec.load_torch_state_dict(state)
    print(f"[t335] decoder built+loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    warm, secs = decode(dec, mesh, latents[seeds[0]])
    print(f"[t335] warm-up decode seed {seeds[0]}: {secs:.3f}s shape {warm.shape} {warm.dtype}", flush=True)
    for s in seeds:
        ts = []
        for _ in range(reps):
            yuv, secs = decode(dec, mesh, latents[s])
            ts.append(secs)
        times[s] = ts
        yuv.tofile(out_dir / f"seed{s}.yuv")
        md5 = hashlib.md5(yuv.tobytes()).hexdigest()
        print(f"[t335] DECODE arm={arm} seed={s} s={' '.join(f'{t:.4f}' for t in ts)} md5={md5}", flush=True)
    if (warm != torch.as_tensor(dec(latents[seeds[0]], output_type="yuv")).numpy()).any():
        print("[t335] WARNING warm-up and repeat decode differ", flush=True)

allt = [t for ts in times.values() for t in ts]
(out_dir / "times.json").write_text(json.dumps({"arm": arm, "times": times}, indent=1))
print(
    f"[t335] ARM {arm} decode_ms min={min(allt) * 1e3:.1f} median={sorted(allt)[len(allt) // 2] * 1e3:.1f}", flush=True
)
