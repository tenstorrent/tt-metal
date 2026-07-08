# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Call 1 demo: prompt -> UMT5-xxl text embeds on the 1x4 mesh (real tokenizer input).

  ./python_env/bin/python -m models.demos.hf_eager.longcat_video.demo.demo_text_encode \
      --prompt "A cat playing piano in a sunny room"
"""
from __future__ import annotations

import argparse

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.longcat_video.tt.pipeline import UMT5_STUBS, _replicated_to_torch, build_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A cat playing piano in a sunny room")
    ap.add_argument("--max-length", type=int, default=32)
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    try:
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    except Exception:
        print("[demo] single-chip fallback")
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        pipe = build_pipeline(dev)
        ids = pipe.encode_prompt(args.prompt, max_length=args.max_length)
        embeds = pipe.run_text_encode(ids)
        out = _replicated_to_torch(embeds, dev).to(torch.float32)
        golden = pipe._hf_reference_text_encode(ids).to(torch.float32)
        _, pcc = comp_pcc(golden, out, 0.99)
        print(f"[demo] prompt={args.prompt!r}")
        print(f"[demo] embeds shape={tuple(out.shape)}")
        print(f"[demo] UMT5 stubs invoked: {sorted(pipe.invoked & set(UMT5_STUBS))}")
        print(f"e2e PCC={pcc}")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
