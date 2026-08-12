"""Split the decode into fixed per-op cost and data-proportional device work, without the profiler.

Item 1 asks whether the decode is host-bound or kernel-bound. The profiler cannot answer it: the same
stage has a recorded 224 ms device total (amendment 66 calls it a ~6x undercount) and a recorded
1401 ms device total (PROFILE_2026_08_06.txt) against ~1.1 s of wall. Absolute Tracy totals for this
stage are not calibrated, so any conclusion resting on them inherits that.

This measures the same question with wall clock only. Every loop in ``Vocoder._forward_device`` is over
``num_upsamples`` / ``num_kernels`` / ``num_branches`` -- none over T -- so the op count is the same
6955 at any sequence length, while the data each op touches scales with T. Fit

    t(T) = a + b * T

  a  is what the graph costs when there is no data in it: per-op dispatch, launch, host round-trip.
  b*207  is the part that scales with the audio, i.e. actual kernel work on rows.

a >> b*207 means the wall is op count and item 1 branches to host-bound: trace/multi-CQ is the game.
b*207 >> a means the wall is kernel work and items 2 and 3 are the right work.
"""

import json
import os
import statistics
import time

import torch
from safetensors.torch import load_file

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder

N = int(os.environ.get("BENCH_N", "5"))
TS = [int(x) for x in os.environ.get("SWEEP_TS", "207,160,104,52,26,13").split(",")]
wd = os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "audio_vae")
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
    rows = []
    for T in TS:
        lat = torch.randn(2, cfg["latent_channels"], T) * 0.1
        # Warm per T: a new sequence length is a new set of shapes, so the first call compiles.
        dec(lat)
        ttnn.synchronize_device(d)
        s = []
        for _ in range(N):
            t0 = time.perf_counter()
            dec(lat)
            ttnn.synchronize_device(d)
            s.append(time.perf_counter() - t0)
        m = statistics.median(s)
        rows.append((T, m))
        print(f"SWEEP T={T:>4}  median {m:.4f}s  min {min(s):.4f}  max {max(s):.4f}", flush=True)

    # Least squares on t = a + b*T.
    n = len(rows)
    sx = sum(T for T, _ in rows)
    sy = sum(m for _, m in rows)
    sxx = sum(T * T for T, _ in rows)
    sxy = sum(T * m for T, m in rows)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    at207 = a + b * 207
    print(f"\nfit  t(T) = {a * 1e3:.1f} ms + {b * 1e6:.1f} us/latent * T")
    print(f"  at T=207: fixed {a * 1e3:.1f} ms + data {b * 207 * 1e3:.1f} ms = {at207 * 1e3:.1f} ms")
    print(f"  fixed share of the 207-latent decode: {100 * a / at207:.0f}%")
    print("\n  >70% fixed  -> host/dispatch-bound: op count is the wall, trace/multi-CQ is the lever")
    print("  >70% data   -> kernel-bound: items 2 and 3 are the right work")
finally:
    ttnn.close_mesh_device(d)
