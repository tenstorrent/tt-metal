# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Split the mesh stage timings into WEIGHT UPLOAD vs FORWARD, per block.

`phase_tt_mesh.py` reports one number per stage, but its block helpers call
`preprocess_*_parameters()` (host weight load + replicated host->device upload) *inside* the
timed region. On a 1xN mesh that upload is paid N times, so those numbers are dominated by
setup and cannot be compared against a single-chip warm run. This script times the two halves
separately, and times a second forward with the program cache hot, so the steady-state
per-batch cost is visible.

    XTTS_CKPT=/var/tmp/xtts_ref/model.pth python profile_mesh_stages.py --replicas 32
"""

import argparse
import time

import torch
import ttnn

from models.experimental.xtts_v2.pipeline.phase_tt_mesh import AR_COMP, HOP, ISR, OSR
from models.experimental.xtts_v2.tt.ttnn_xtts_cond import (
    LATENTS,
    TTNNConditioningEncoder,
    TTNNPerceiver,
    preprocess_encoder_parameters,
    preprocess_perceiver_parameters,
)
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTTracedDecoder
from models.experimental.xtts_v2.tt.ttnn_xtts_hifigan import (
    TTNNHifiganGenerator,
    preprocess_hifigan_parameters,
)
from models.experimental.xtts_v2.tt.ttnn_xtts_speaker import TTNNSpeakerEncoder, preprocess_speaker_parameters

G = "models/experimental/xtts_v2/golden"


def timed(label, fn):
    t = time.time()
    r = fn()
    dt = time.time() - t
    print(f"  {label:<34} {dt:8.2f}s", flush=True)
    return r, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=32)
    args = ap.parse_args()
    N = args.replicas

    logmel = torch.load(f"{G}/speaker/logmel.pt").float()
    mel = torch.load(f"{G}/cond/mel_in.pt").float()

    print(f"opening 1x{N} mesh", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), l1_small_size=65536, trace_region_size=60_000_000)
    try:
        mesh.enable_program_cache()
        shard = ttnn.shard_tensor_to_mesh_mapper(mesh, 0) if N > 1 else None
        comp = ttnn.concat_mesh_to_tensor_composer(mesh, 0) if N > 1 else None
        tot_up = tot_fwd = 0.0

        # ---- Block 2 ----
        print(f"Block 2 (speaker encoder)", flush=True)
        p2, dt = timed("weight upload (x%d)" % N, lambda: preprocess_speaker_parameters(mesh))
        tot_up += dt
        m2 = TTNNSpeakerEncoder(mesh, p2)
        lm = torch.cat([logmel] * N, 0)
        lm_tt = ttnn.from_torch(lm, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)

        def f2():
            o = m2(lm_tt)
            o = o[0] if isinstance(o, tuple) else o
            return ttnn.to_torch(o, mesh_composer=comp) if comp else ttnn.to_torch(o)

        spk, dt = timed("forward (cold, compiles)", f2)
        _, dt2 = timed("forward (program cache hot)", f2)
        tot_fwd += dt2

        # ---- Block 1 ----
        print(f"Block 1 (conditioning + Perceiver)", flush=True)
        T = mel.shape[2]
        S = ((T + 31) // 32) * 32
        mel_f = torch.nn.functional.pad(mel.permute(0, 2, 1).contiguous(), (0, 0, 0, S - T))
        mel_b = torch.cat([mel_f] * N, 0)
        (pe, pp), dt = timed(
            "weight upload (x%d)" % N,
            lambda: (
                preprocess_encoder_parameters(mesh, dtype=ttnn.float32),
                preprocess_perceiver_parameters(mesh, dtype=ttnn.float32),
            ),
        )
        tot_up += dt
        enc = TTNNConditioningEncoder(mesh, pe, t_real=T, s_pad=S)
        perc = TTNNPerceiver(mesh, pp)
        km = torch.zeros(1, 1, 1, LATENTS + S)
        km[:, :, :, LATENTS + T :] = -1e9
        km_tt = ttnn.from_torch(
            km, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=enc.mesh_mapper
        )
        mel_tt = ttnn.from_torch(mel_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)

        def f1():
            o = perc(enc(mel_tt), km_tt)
            return ttnn.to_torch(o, mesh_composer=comp) if comp else ttnn.to_torch(o)

        _, dt = timed("forward (cold, compiles)", f1)
        _, dt2 = timed("forward (program cache hot)", f1)
        tot_fwd += dt2

        # ---- Block 3 ----
        print(f"Block 3 (GPT)", flush=True)
        pg, dt = timed("weight upload (x%d)" % N, lambda: preprocess_gpt_parameters(mesh, dtype=ttnn.bfloat16))
        tot_up += dt
        P = 75
        prefix = torch.randn(N, P, 1024) * 0.1
        dec = TTNNGPTTracedDecoder(mesh, pg, max_seq=P + 1 + 605, batch=N, data_mapper=shard)
        _, dt = timed("reset_caches", dec.reset_caches)
        _, dtp = timed("prefill (P=%d)" % P, lambda: dec.prefill(prefix.contiguous()))
        _, dt = timed("trace capture", dec.capture)
        emb = torch.randn(N, 1, 1024) * 0.1

        def step64():
            for i in range(64):
                e = ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
                lat = dec.step_device(e, [P + i] * N)
                ttnn.to_torch(lat, mesh_composer=comp) if comp else ttnn.to_torch(lat)

        _, dt = timed("64 decode steps", step64)
        print(f"  {'-> per step':<34} {1000 * dt / 64:8.2f}ms   ({N} requests/step)")
        tot_fwd += dtp + dt

        # ---- Block 4 ----
        print(f"Block 4 (HiFi-GAN)", flush=True)
        p4, dt = timed("weight upload (x%d)" % N, lambda: preprocess_hifigan_parameters(mesh))
        tot_up += dt
        voc = TTNNHifiganGenerator(mesh, p4)
        lat = torch.randn(N, 94, 1024) * 0.1
        z = torch.nn.functional.interpolate(lat.transpose(1, 2), scale_factor=AR_COMP / HOP, mode="linear")
        z = torch.nn.functional.interpolate(z, scale_factor=OSR / ISR, mode="linear")
        L = z.shape[-1]
        z_t = ttnn.from_torch(
            z.permute(0, 2, 1).reshape(N, 1, L, 1024),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=shard,
        )
        g_t = ttnn.from_torch(
            spk.to(torch.float32).reshape(N, 512),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=shard,
        )

        def f4():
            o = voc(z_t, g_t)
            return ttnn.to_torch(o, mesh_composer=comp) if comp else ttnn.to_torch(o)

        _, dt = timed("forward (cold, compiles)", f4)
        _, dt2 = timed("forward (program cache hot)", f4)
        tot_fwd += dt2

        print()
        print(f"TOTAL weight upload (one-time, x{N} chips): {tot_up:8.2f}s")
        print(f"TOTAL steady-state forward for {N} requests: {tot_fwd:8.2f}s")
    finally:
        mesh.quiesce_devices()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
