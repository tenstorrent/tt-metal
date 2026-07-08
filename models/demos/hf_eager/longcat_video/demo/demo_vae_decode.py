# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Call 3 demo: VAE reconstruction (video -> latent -> video) on the 1x4 mesh.

  ./python_env/bin/python -m models.demos.hf_eager.longcat_video.demo.demo_vae_decode
"""
from __future__ import annotations

import argparse

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.longcat_video.tt.pipeline import build_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    try:
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    except Exception:
        print("[demo] single-chip fallback")
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        pipe = build_pipeline(dev)
        video = torch.randn(1, 3, args.frames, args.size, args.size, dtype=torch.float32)
        recon = pipe.run_vae(video)  # composite encode+decode
        _ = pipe.run_vae_encode(video)  # wan_encoder3d real half
        golden = pipe._hf_reference_vae(video).to(torch.float32)
        _, pcc = comp_pcc(golden, recon.to(torch.float32), 0.95)
        print(f"[demo] input video shape={tuple(video.shape)} -> recon shape={tuple(recon.shape)}")
        print(f"[demo] VAE stubs invoked: {sorted(pipe.invoked)}")
        print(f"e2e PCC={pcc}")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
