# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CALL 2 demo -- the flow-match Euler loop on the T3K mesh.

    python -m models.demos.flux_2_klein_9b.transformer.demo.demo_denoise_latents \
        --height 256 --width 256 --txt-len 64 --num-steps 4 --check-pcc

This is the real task: N denoise forwards with the latents kept RESIDENT on
device between steps, ending in the final latents a VAE decoder would turn into
an image. Both the packed `[1, S_img, 128]` latents and the unpacked
`[1, 128, h, w]` grid are written out.

`--num-steps` changes BOTH sides: the TT loop and the `--check-pcc` golden
consume the same sigma list, computed once by
`tt/inputs.py::sigma_schedule` from Source A's own `Flux2Pipeline` recipe.
The default of 4 is small because this is the DISTILLED Klein variant, which is
built for few-step sampling; there is no stop token on a denoise loop, so the
`[1, 50]` clamp is the only bound.
"""

from __future__ import annotations

import sys
import time

import torch

from models.demos.flux_2_klein_9b.transformer.demo import _demo_common as common
from models.demos.flux_2_klein_9b.transformer.tt import inputs as tt_inputs
from models.demos.flux_2_klein_9b.transformer.tt import pipeline as tt_pipeline
from models.demos.flux_2_klein_9b.transformer.tt import reference as tt_reference


def main(argv=None) -> int:
    args = common.build_parser(__doc__.splitlines()[0], steps=True).parse_args(argv)

    model = common.load_model(args)
    inputs = common.make_inputs(args)

    device = common.open_mesh(args)
    try:
        pipe = common.build(args, device, model)
        pipe.reset_invocations()

        started = time.time()
        result = tt_pipeline.run_denoise_latents(pipe, inputs, num_steps=args.num_steps)
        latents = tt_pipeline.to_torch(result["latents"], device).to(torch.float32)
        elapsed = time.time() - started
        steps = len(result["sigmas"]) - 1

        print(
            f"[flux2-demo] {steps} Euler steps in {elapsed:.2f}s "
            f"({elapsed / steps:.2f}s/step) -> latents {tuple(latents.shape)}",
            flush=True,
        )
        print(f"[flux2-demo] sigmas {[round(s, 6) for s in result['sigmas']]}", flush=True)
        print(f"[flux2-demo] graduated stubs invoked: {dict(sorted(pipe.invocations.items()))}", flush=True)

        grid = tt_inputs.unpack_latents(latents, inputs["img_ids"])
        print(f"[flux2-demo] unpacked latent grid {tuple(grid.shape)} (what the VAE decoder consumes)", flush=True)

        achieved_pcc = None
        per_step_pcc = None
        if args.check_pcc:
            depth = pipe.depth()
            golden = tt_reference._hf_reference_denoise_latents(
                model,
                inputs,
                num_inference_steps=args.num_steps,
                dual_layers=depth["dual_layers"],
                single_layers=depth["single_layers"],
            )
            per_step_pcc = [
                common.report_pcc(hf_step, tt_pipeline.to_torch(tt_step, device), label=f"step {index}")
                for index, (tt_step, hf_step) in enumerate(zip(result["per_step"], golden["per_step"]))
            ]
            achieved_pcc = common.report_pcc(golden["latents"], latents)

        common.write_outputs(
            args,
            "denoise_latents",
            {"latents": latents, "grid": grid},
            {
                "call": "denoise_latents",
                "depth": pipe.depth(),
                "full_depth": {"dual": pipe.full_dual_layers, "single": pipe.full_single_layers},
                "steps": steps,
                "sigmas": result["sigmas"],
                "shape": list(latents.shape),
                "grid_shape": list(grid.shape),
                "seconds": elapsed,
                "invocations": pipe.invocations,
                "meta": inputs["meta"],
                "e2e_pcc": achieved_pcc,
                "per_step_pcc": per_step_pcc,
                "tp": args.tp,
            },
        )
    finally:
        common.close_mesh(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
