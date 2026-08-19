# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Median decode time over N runs, building the decoder once.

Build once, warm once, then time N decodes and report median plus spread. Timing a single call per
process is where an earlier script's run-to-run noise came from.

Usage -- the label names the saved waveform:

    export MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3-diffusers
    python models/tt_dit/tests/models/minimax_h3/decode_bench.py baseline

Env: BENCH_N (default 5), BENCH_OUT (default `generated/minimax_h3_audio/bench`).

For latency *with* PSNR against a CPU reference, and for mesh/sharding/trace configurations, use
`cpu_vs_device.py` instead; this script is single-device and reports timing only.
"""
import json
import os
import re
import statistics
import sys
import time

import torch
from safetensors.torch import load_file

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder

# Reaches a filename below, so keep it to one path-safe component.
LABEL = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(sys.argv[1])) or "bench"
N = int(os.environ.get("BENCH_N", "5"))
OUT = os.environ.get("BENCH_OUT") or os.path.join(
    os.environ.get("TT_METAL_HOME", os.getcwd()), "generated", "minimax_h3_audio", "bench"
)
# `MINIMAX_H3_MODEL_PATH` matches the test suite; the older var is still accepted.
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
    # Both halves come from outside the process (BENCH_OUT, argv[1]), so contain the result here
    # rather than relying on the sanitising above.
    _out_dir = os.path.realpath(OUT)
    _dest = os.path.realpath(os.path.join(_out_dir, LABEL + ".pt"))
    if os.path.commonpath([_out_dir, _dest]) != _out_dir:
        raise SystemExit(f"refusing to write outside {_out_dir}: {_dest}")
    torch.save({"wav": torch.as_tensor(out).float().cpu(), "samples": s}, _dest)
    print(
        f"RESULT {LABEL}: median {statistics.median(s):.4f}s  min {min(s):.4f}  max {max(s):.4f}  n={N}  "
        f"[{', '.join(f'{v:.3f}' for v in s)}]",
        flush=True,
    )
finally:
    ttnn.close_mesh_device(d)
