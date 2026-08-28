# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 3 — reconstruct: the headline demo. An image survives the round trip.

    python -m models.demos.flux_2_klein_9b.vae.demo.demo_reconstruct [--image PATH] [--tp 8]

This is the model's own forward (``AutoencoderKLFlux2(x).sample``, ``sample_posterior=False``):
image -> encoder_stack -> quant_conv -> posterior mode -> post_quant_conv -> decoder_head -> image.
The intermediate latent is the TT one; no reference tensor is injected at the joint.

Writes ``input.png``, ``tt_reconstruction.png`` and ``hf_reconstruction.png`` side by side —
a mis-wired joint shows up as a visibly wrong PNG even when every shape agrees.
The chain is ``tt/pipeline.py``'s ``run_reconstruct``; this file only feeds it and reports.
"""
from __future__ import annotations

import sys

from models.demos.flux_2_klein_9b.vae.demo import (
    TAG,
    build_arg_parser,
    close_mesh,
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
    args = build_arg_parser("reconstruct").parse_args(argv)
    out = ensure_output_dir(args.output_dir)

    # 1. real input
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
        tt_image = pipeline.run_reconstruct(pixel_values)

        ledger = print_ledger(pipeline)
    finally:
        close_mesh(mesh_device)

    # 3. task output artifacts — the behavioural proof
    golden = R.hf_reference_reconstruct(pixel_values)
    save_image(pixel_values, out / "input.png", R)
    save_image(tt_image, out / "tt_reconstruction.png", R)
    save_image(golden, out / "hf_reconstruction.png", R)

    # 4. parity + behaviour
    pcc = report_pcc("reconstruct", golden, tt_image)
    print(f"{TAG} reconstruct: psnr_tt_vs_hf={psnr(tt_image.float(), golden.float()):.2f} dB", flush=True)
    print(f"{TAG} reconstruct: psnr_tt_vs_original={psnr(tt_image.float(), pixel_values.float()):.2f} dB", flush=True)
    print(f"{TAG} reconstruct: psnr_hf_vs_original={psnr(golden.float(), pixel_values.float()):.2f} dB", flush=True)
    print(f"{TAG} reconstruct done — {len(ledger)} graduated modules invoked, PCC={pcc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
