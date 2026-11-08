# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Hunyuan3D-2.1 image -> mesh with the shape DiT and the ShapeVAE decoder on one Blackhole chip.

The upstream pipeline object (Tencent-Hunyuan/Hunyuan3D-2.1 hy3dshape, on sys.path) does preprocessing, the DINOv2-L
conditioner (host, for now), the flow-matching Euler sampler, marching cubes and export; its DiT forward, VAE forward
and dense volume decoder are replaced by the TT modules. The latents stay on device between the VAE transformer and
the grid decoder. Prints per-stage times as JSON.

usage: python image_to_shape.py --repo HY3DSHAPE --weights HY3D_WEIGHTS --image IMG --out DIR [--steps 30] [--octree 256]
       [--seed 0] [--runs 1] [--extra-path P ...]
"""

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--octree", type=int, default=256)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--dit-cache", default=None)
    ap.add_argument("--extra-path", action="append", default=[])
    a = ap.parse_args()
    for p in a.extra_path + [a.repo]:
        sys.path.insert(0, p)
    import torch

    import ttnn
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from models.experimental.hunyuan3d_21.reference import dit_host as host
    from models.experimental.hunyuan3d_21.tt.dit import TtHunyuanDiT
    from models.experimental.hunyuan3d_21.tt.vae import TtShapeVAEDecoder

    os.makedirs(a.out, exist_ok=True)
    T = {"config": vars(a)}
    t0 = time.time()
    d = os.path.join(a.weights, "hunyuan3d-dit-v2-1")
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        f"{d}/model.fp16.ckpt", f"{d}/config.yaml", device="cpu", dtype=torch.float32
    )
    T["host_load_s"] = round(time.time() - t0, 1)
    mesh_dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        t0 = time.time()
        dit = TtHunyuanDiT(mesh_device=mesh_dev)
        dit.load_torch_state_dict(
            host.load_dit_state(f"{d}/model.fp16.ckpt", a.dit_cache or f"{a.out}/dit_model.safetensors")
        )
        vae_sd = torch.load(
            os.path.join(a.weights, "hunyuan3d-vae-v2-1", "model.fp16.ckpt"), map_location="cpu", weights_only=True
        )
        vae = TtShapeVAEDecoder(mesh_device=mesh_dev)
        vae.load_torch_state_dict({k: v.float() for k, v in vae_sd.items()})
        T["tt_build_s"] = round(time.time() - t0, 1)

        st = {}
        cond_fwd = pipe.conditioner.forward

        def cond_hook(*args, **kw):
            s = time.time()
            r = cond_fwd(*args, **kw)
            st["conditioner_s"] = time.time() - s
            return r

        def dit_hook(x, t, contexts, **kw):
            s = time.time()
            r = dit(x, t, contexts["main"])
            st.setdefault("dit_calls_s", []).append(time.time() - s)
            return r

        def vae_hook(lat):
            s = time.time()
            st["latents_dev"] = vae.decode_latents(lat * 1.0039506158752403)  # pipeline already divided by scale_factor
            ttnn.synchronize_device(mesh_dev)
            st["vae_transformer_s"] = time.time() - s
            return lat  # placeholder: the grid decoder below reads the device latents

        def vol_hook(latents, geo_decoder, octree_resolution=None, **kw):
            s = time.time()
            g = vae.grid_logits(st["latents_dev"], octree_resolution, chunk=a.chunk)
            st["volume_decode_s"] = time.time() - s
            return g.unsqueeze(0)

        pipe.conditioner.forward = cond_hook
        pipe.model.forward = dit_hook
        pipe.vae.forward = vae_hook
        pipe.vae.volume_decoder = vol_hook
        se = pipe.vae.surface_extractor if hasattr(pipe.vae, "surface_extractor") else None
        runs = []
        for r in range(a.runs):
            st.clear()
            gen = torch.Generator().manual_seed(a.seed)
            s = time.time()
            m = pipe(
                image=a.image,
                num_inference_steps=a.steps,
                guidance_scale=a.guidance,
                octree_resolution=a.octree,
                num_chunks=a.chunk,
                generator=gen,
                enable_pbar=False,
            )[0]
            total = time.time() - s
            calls = st.get("dit_calls_s", [])
            rec = {
                "run": r,
                "total_s": round(total, 2),
                "conditioner_s": round(st.get("conditioner_s", 0), 2),
                "dit_calls": len(calls),
                "dit_total_s": round(sum(calls), 2),
                "dit_first_s": round(calls[0], 2) if calls else None,
                "dit_warm_mean_s": round(sum(calls[1:]) / max(1, len(calls) - 1), 3),
                "vae_transformer_s": round(st.get("vae_transformer_s", 0), 3),
                "volume_decode_s": round(st.get("volume_decode_s", 0), 2),
            }
            rec["host_other_s"] = round(
                total - rec["conditioner_s"] - rec["dit_total_s"] - rec["vae_transformer_s"] - rec["volume_decode_s"], 2
            )
            rec.update(vertices=len(m.vertices), faces=len(m.faces))
            runs.append(rec)
            print(json.dumps(rec), flush=True)
        m.export(os.path.join(a.out, "mesh.glb"))
        T["runs"] = runs
        json.dump(T, open(os.path.join(a.out, "timings.json"), "w"), indent=1)
        print("HY3D_E2E", json.dumps(runs[-1]))
    finally:
        ttnn.close_mesh_device(mesh_dev)


if __name__ == "__main__":
    main()
