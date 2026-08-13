"""Median decode time over N runs, building the decoder once.

Timing one call per process, as an earlier script did, is where the session's ~1% run-to-run spread
came from. Build once, warm once, then time N decodes; report median and spread.
"""
import json
import os
import statistics
import sys
import time

import torch
from safetensors.torch import load_file

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder

LABEL = sys.argv[1]
N = int(os.environ.get("BENCH_N", "5"))
OUT = os.environ.get("BENCH_OUT", "/data/rshirvani/bench_out")
# `MINIMAX_H3_MODEL_PATH` is what the test suite uses (`common.weights_subdir`); the older
# `MINIMAX_H3_DIFFUSERS_DIR` is still accepted so existing shells keep working.
_root = os.environ.get("MINIMAX_H3_MODEL_PATH") or os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "")
if not _root:
    raise SystemExit("set MINIMAX_H3_MODEL_PATH to a MiniMax-H3 diffusers snapshot")
wd = os.path.join(_root, "audio_vae")
cfg = {k: v for k, v in json.load(open(os.path.join(wd, "config.json"))).items() if not k.startswith("_")}

d = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
try:
    dec = MiniMaxH3AudioDecoder(
        latent_channels=cfg["latent_channels"],
        latent_dim=cfg["latent_dim"],
        decoder_dim=cfg["decoder_dim"],
        decoder_rates=tuple(cfg["decoder_rates"]),
        decoder_kernel_sizes=tuple(cfg["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(cfg["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(x) for x in cfg["resblock_dilation_sizes"]),
        mesh_device=d,
    )
    dec.load_torch_state_dict(
        convert_minimax_h3_audio_state_dict(load_file(os.path.join(wd, "diffusion_pytorch_model.safetensors"))),
        strict=False,
    )
    torch.manual_seed(2)
    lat = torch.randn(2, cfg["latent_channels"], 207) * 0.1
    out = dec(lat)
    ttnn.synchronize_device(d)
    s = []
    for _ in range(N):
        t0 = time.perf_counter()
        out = dec(lat)
        ttnn.synchronize_device(d)
        s.append(time.perf_counter() - t0)
    os.makedirs(OUT, exist_ok=True)
    torch.save({"wav": torch.as_tensor(out).float().cpu(), "samples": s}, os.path.join(OUT, f"{LABEL}.pt"))
    print(
        f"RESULT {LABEL}: median {statistics.median(s):.4f}s  min {min(s):.4f}  max {max(s):.4f}  n={N}  "
        f"[{', '.join(f'{v:.3f}' for v in s)}]",
        flush=True,
    )
finally:
    ttnn.close_mesh_device(d)
