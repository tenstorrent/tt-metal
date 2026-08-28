# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 2 — decode: a latent becomes an image on a 1 x TP Tenstorrent mesh.

    python -m models.demos.flux_2_klein_9b.vae.demo.demo_decode [--latent PATH] [--tp 8]

The latent is a REAL stage input: by default the HF-captured golden latent
``_captured/decoder/args.pt[0]`` (``[1,32,28,28]``), or ``--latent`` to consume the
``latent.pt`` that ``demo_encode`` just wrote. Producing an image from a latent is the
whole task head, so the stage input is allowed to be a reference tensor — nothing is
injected at a joint INSIDE the chain. The chain is ``tt/pipeline.py``'s ``run_decode``.
"""
from __future__ import annotations

import sys

import torch

from models.demos.flux_2_klein_9b.vae.demo import (
    TAG,
    build_arg_parser,
    close_mesh,
    describe_latent,
    ensure_output_dir,
    open_mesh,
    print_ledger,
    psnr,
    report_pcc,
    save_image,
)
from models.demos.flux_2_klein_9b.vae.tt import reference as R
from models.demos.flux_2_klein_9b.vae.tt.pipeline import build_pipeline


def main(argv=None) -> int:
    parser = build_arg_parser("decode")
    parser.add_argument(
        "--latent",
        type=str,
        default=None,
        help="Path to a saved latent .pt (e.g. demo_encode's latent.pt). Default: the captured golden latent.",
    )
    args = parser.parse_args(argv)
    out = ensure_output_dir(args.output_dir)

    # 1. real stage input
    if args.latent:
        latent = torch.load(args.latent, map_location="cpu").float()
        source = args.latent
    else:
        latent = R.captured_tensor("decoder", which="args", index=0).float()
        source = "_captured/decoder/args.pt[0]"
    print(f"{TAG} input latent from {source}", flush=True)
    describe_latent(latent)

    mesh_device = open_mesh(args.tp)
    try:
        print(f"{TAG} mesh 1x{args.tp} open; building the resident pipeline (layers={args.layers})", flush=True)
        pipeline = build_pipeline(mesh_device, layers=args.layers)

        # 2. THE chain — the only implementation of it lives in tt/pipeline.py
        tt_image = pipeline.run_decode(latent)

        ledger = print_ledger(pipeline)
    finally:
        close_mesh(mesh_device)

    # 3. task output artifacts
    golden = R.hf_reference_decode(latent)
    save_image(tt_image, out / "tt_decode.png", R)
    save_image(golden, out / "hf_decode.png", R)

    # 4. parity + behaviour
    pcc = report_pcc("decode", golden, tt_image)
    print(f"{TAG} decode: psnr_tt_vs_hf={psnr(tt_image.float(), golden.float()):.2f} dB", flush=True)
    if args.image:
        original = R.preprocess_image(R.load_input_image(args.image, size=args.size), size=args.size)
        if tuple(original.shape) == tuple(tt_image.shape):
            print(f"{TAG} decode: psnr_tt_vs_original={psnr(tt_image.float(), original.float()):.2f} dB", flush=True)
        else:
            print(
                f"{TAG} decode: psnr_tt_vs_original=n/a (shape {tuple(original.shape)} != {tuple(tt_image.shape)})",
                flush=True,
            )
    else:
        print(
            f"{TAG} decode: psnr_tt_vs_original=n/a (decode has no source image; pass --image to compare)", flush=True
        )
    print(f"{TAG} decode done — {len(ledger)} graduated modules invoked, PCC={pcc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
