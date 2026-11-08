# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""G2 gate for the Hunyuan3D-2.1 ShapeVAE decoder on device vs the CPU fp32 goldens (dump_goldens.py):
transformer output on the saved latents, then SDF logits on the dense grid (a slice, or the whole grid with --full).
usage: python run_vae_vs_golden.py GOLDEN_DIR VAE_CKPT [--full] [--res 256] [--chunk 32768]"""
import argparse
import json
import time

import torch

import ttnn
from models.experimental.hunyuan3d_21.tt.vae import SCALE_FACTOR, TtShapeVAEDecoder, dense_grid

ap = argparse.ArgumentParser()
ap.add_argument("gdir")
ap.add_argument("ckpt")
ap.add_argument("--full", action="store_true")
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--chunk", type=int, default=32768)
ap.add_argument("--slice", type=int, default=4)
a = ap.parse_args()
pcc = lambda x, y: float(torch.corrcoef(torch.stack([x.flatten().double(), y.flatten().double()]))[0, 1])
rel = lambda x, y: float((x.double() - y.double()).norm() / y.double().norm())
sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
sd = {k: v.float() for k, v in (sd["vae"] if "vae" in sd else sd).items()}
g = torch.load(f"{a.gdir}/vae_dec.pt")
grid_ref = torch.load(f"{a.gdir}/grid_logits.pt")[0]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    m = TtShapeVAEDecoder(mesh_device=mesh)
    m.load_torch_state_dict(sd)
    res = {}
    for k in range(2):
        s = time.time()
        lat = m.decode_latents(g["in"] * SCALE_FACTOR)  # golden "in" is already divided by scale_factor
        out = ttnn.to_torch(lat).float()[:, 0]
        res[f"transformer_s_{k}"] = round(time.time() - s, 3)
    res["transformer_pcc"], res["transformer_rel"] = round(pcc(out, g["out"]), 6), round(rel(out, g["out"]), 5)
    print(json.dumps(res), flush=True)
    kv = m.geo.prepare_kv(lat)
    pts = dense_grid(a.res)
    n = pts.shape[0] if a.full else a.slice * a.chunk
    from models.experimental.hunyuan3d_21.tt.vae import QUERY_DIM, QUERY_PAD, fourier_embed
    from models.tt_dit.utils.tensor import bf16_tensor

    vals, t0 = [], time.time()
    for s0 in range(0, n, a.chunk):
        p = pts[s0 : min(s0 + a.chunk, n)]
        emb = torch.zeros(1, 1, a.chunk, QUERY_PAD)
        emb[0, 0, : p.shape[0], :QUERY_DIM] = fourier_embed(p)
        o = m.geo(bf16_tensor(emb, device=mesh), kv)
        vals.append(ttnn.to_torch(o)[0, 0, : p.shape[0], 0].float())
        if s0 == 0:
            t_first = time.time() - t0
    t_all = time.time() - t0
    v = torch.cat(vals)
    ref = grid_ref.reshape(-1)[:n]
    sign_flips = int(((v > 0) != (ref > 0)).sum())
    near = ref.abs() < 0.05
    res.update(
        points=n,
        chunks=len(vals),
        grid_pcc=round(pcc(v, ref), 6),
        grid_rel=round(rel(v, ref), 5),
        sign_flips=sign_flips,
        near_surface=int(near.sum()),
        grid_s=round(t_all, 2),
        first_chunk_s=round(t_first, 2),
        s_per_chunk_warm=round((t_all - t_first) / max(1, len(vals) - 1), 4),
    )
    if a.full:
        torch.save(v.reshape(a.res + 1, a.res + 1, a.res + 1), f"{a.gdir}/grid_logits_tt.pt")
    print("VAE_G2", json.dumps(res), flush=True)
finally:
    ttnn.close_mesh_device(mesh)
