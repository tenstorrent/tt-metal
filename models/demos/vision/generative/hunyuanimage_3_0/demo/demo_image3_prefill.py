# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable demo for `tencent/HunyuanImage-3.0` — Call-1:
`hunyuan_image3_transformer_prefill`.

Tokenizes a real prompt with the HF tokenizer, runs the SHARED TTNN pipeline
(`tt/pipeline.py` — the exact code the e2e test asserts on), and prints the
real transformer last_hidden_state produced on device.

    ./python_env/bin/python -m \
        models.demos.vision.generative.hunyuanimage_3_0.demo.demo_image3_prefill \
        --prompt "A serene mountain lake at sunrise" --num-layers 1
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0.tt import pipeline as pl


def main():
    ap = argparse.ArgumentParser(description="HunyuanImage-3.0 TTNN transformer prefill demo")
    ap.add_argument("--prompt", default="A serene mountain lake at sunrise, photorealistic.")
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--compare", action="store_true", help="also run the HF reference forward and print PCC")
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"[demo] loading HF reference {pl.HF_MODEL_ID} (this loads the 80B checkpoint) ...")
    model = pl.load_reference_model()

    # Shard-graduated (TP=8) stubs run tensor-parallel on a mesh; this 6U
    # Blackhole Galaxy only brings FABRIC_1D up on the FULL physical mesh.
    # Fabric is enabled via set_fabric_config BEFORE opening the mesh
    # (open_mesh_device takes no fabric_config kwarg).
    pl.enable_fabric_1d()
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*pl._full_mesh_shape()),
        l1_small_size=24576,
    )
    try:
        pipe = pl.build_pipeline(device, model, num_layers=args.num_layers, seq_len=args.seq_len)
        print(f"[demo] prompt: {args.prompt!r}")
        inputs = pipe.make_inputs(args.prompt)
        print(f"[demo] input_ids[0,:16]={inputs['input_ids'][0, :16].tolist()}")

        if args.compare:
            result = pipe.run_and_compare(args.prompt, pcc_target=0.95)
            hidden = result["hidden_tt"]
            print(f"[demo] e2e PCC={result['pcc']}")
            print(f"[demo] l_aux_tt={result['l_aux_tt']:.6f}  l_aux_ref={result['l_aux_ref']:.6f}")
            print(f"[demo] graduated invocations={result['invocations']}")
        else:
            hidden_tt, l_aux_tt = pipe.run_prefill(inputs)
            hidden = pl._mesh_to_torch(hidden_tt, device).to(torch.float32)
            if hidden.dim() == 4:
                hidden = hidden.reshape(hidden.shape[0], hidden.shape[-2], hidden.shape[-1])
            print(f"[demo] graduated invocations={pipe.graduated_invocations()}")

        print(
            f"[demo] last_hidden_state shape={tuple(hidden.shape)} "
            f"mean={hidden.mean().item():.5f} std={hidden.std().item():.5f}"
        )
        print(f"[demo] last_hidden_state[0,0,:8]={hidden[0, 0, :8].tolist()}")
        print("[demo] done — real transformer activations produced on device.")
    finally:
        pl._close_device(device)


if __name__ == "__main__":
    main()
