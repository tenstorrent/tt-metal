# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full text-to-video behavioral demo: prompt -> UMT5 embeds -> capped DiT denoise loop ->
VAE decode -> RGB video latent, chained over the graduated TTNN stubs on the 1x4 mesh.

  ./python_env/bin/python -m models.demos.hf_eager.longcat_video.demo.demo_t2v \
      --prompt "A cat playing piano in a sunny room" --steps 4 --frames 1 --size 32
"""
from __future__ import annotations

import argparse

import torch

import ttnn
from models.demos.hf_eager.longcat_video.tt.pipeline import ALL_GRADUATED, build_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A cat playing piano in a sunny room")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--out", default="/tmp/longcat_t2v_out.pt")
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    try:
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    except Exception:
        print("[demo] single-chip fallback")
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        pipe = build_pipeline(dev)
        video = pipe.run_t2v(args.prompt, num_frames=args.frames, height=args.size, width=args.size, steps=args.steps)
        torch.save(video, args.out)
        print(f"[demo] prompt={args.prompt!r} steps={args.steps}")
        print(f"[demo] decoded video shape={tuple(video.shape)} -> saved {args.out}")
        print(f"[demo] video stats: min={video.min():.3f} max={video.max():.3f} mean={video.mean():.3f}")
        print(f"[demo] graduated stubs invoked on the real forward: {sorted(pipe.invoked)}")
        print(
            f"[demo] ({len(pipe.invoked)} of {len(ALL_GRADUATED)} graduated stubs; see README for the VAE sub-block note)"
        )
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
