# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Call 2 demo: one DiT denoise step (latent + timestep + real text embeds -> noise pred).

  ./python_env/bin/python -m models.demos.hf_eager.longcat_video.demo.demo_denoise \
      --prompt "A cat playing piano" --size 32
"""
from __future__ import annotations

import argparse

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.longcat_video.tt.pipeline import DIT_STUBS, build_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A cat playing piano in a sunny room")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--timestep", type=float, default=500.0)
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
        C = pipe._dit_torch_model().config.in_channels
        latent = torch.randn(1, C, args.frames, args.size, args.size, dtype=torch.float32)
        timestep = torch.tensor([args.timestep])
        ids = pipe.encode_prompt(args.prompt, max_length=64)
        embeds = pipe.run_text_encode(ids)  # real TT embeds feed the DiT
        out = pipe.run_denoise(latent, timestep, embeds)
        golden = pipe._hf_reference_denoise(latent, timestep, embeds).to(torch.float32)
        _, pcc = comp_pcc(golden, out.to(torch.float32), 0.95)
        print(f"[demo] latent {tuple(latent.shape)} + t={args.timestep} -> noise pred {tuple(out.shape)}")
        print(
            f"[demo] DiT stubs invoked ({len(pipe.invoked & set(DIT_STUBS))}/10): {sorted(pipe.invoked & set(DIT_STUBS))}"
        )
        print(f"e2e PCC={pcc}")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
