# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 — encode: a real image becomes a latent on a 1 x TP Tenstorrent mesh.

    python -m models.demos.flux_2_klein_9b.vae.demo.demo_encode [--image PATH] [--tp 8]

Consumes a PIL image via ``VaeImageProcessor.preprocess`` -> float32 ``[1,3,C,C]`` in
``[-1,1]``; produces the posterior MODE latent ``[1,32,C/8,C/8]``. The chain itself is
``tt/pipeline.py``'s ``run_encode`` — this file only feeds it and reports.
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
    report_pcc,
    save_image,
    save_latent_preview,
)
from models.demos.flux_2_klein_9b.vae.tt import reference as R
from models.demos.flux_2_klein_9b.vae.tt.pipeline import build_pipeline


def main(argv=None) -> int:
    args = build_arg_parser("encode").parse_args(argv)
    out = ensure_output_dir(args.output_dir)

    # 1. real input, built exactly the way the HF side builds it
    image = R.load_input_image(args.image, size=args.size)
    pixel_values = R.preprocess_image(image, size=args.size)
    print(
        f"{TAG} input: {tuple(pixel_values.shape)} from {args.image or 'reference.load_input_image() default'}",
        flush=True,
    )

    mesh_device = open_mesh(args.tp)
    try:
        print(f"{TAG} mesh 1x{args.tp} open; building the resident pipeline (layers={args.layers})", flush=True)
        pipeline = build_pipeline(mesh_device, layers=args.layers)

        # 2. THE chain — the only implementation of it lives in tt/pipeline.py
        tt_latent = pipeline.run_encode(pixel_values)

        ledger = print_ledger(pipeline)
    finally:
        close_mesh(mesh_device)

    # 3. task output artifacts
    save_image(pixel_values, out / "input.png", R)
    torch.save(tt_latent.detach().float(), out / "latent.pt")
    print(f"{TAG} wrote {out / 'latent.pt'}", flush=True)
    save_latent_preview(tt_latent, out / "latent_preview.png", args.size)
    describe_latent(tt_latent)

    # 4. parity against the HF golden for the SAME input
    golden = R.hf_reference_encode(pixel_values)
    pcc = report_pcc("encode", golden, tt_latent)
    print(f"{TAG} encode done — {len(ledger)} graduated modules invoked, PCC={pcc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
