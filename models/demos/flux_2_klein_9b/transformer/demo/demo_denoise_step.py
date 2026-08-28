# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CALL 1 demo -- one denoise forward on the T3K mesh.

    python -m models.demos.flux_2_klein_9b.transformer.demo.demo_denoise_step \
        --height 256 --width 256 --txt-len 64 --check-pcc

Prints the velocity prediction's shape and statistics, writes the tensor out,
and with `--check-pcc` also runs the HF golden and prints the same `e2e PCC=`
line `tests/e2e/test_e2e_denoise_step.py` prints -- because both call the same
`tt/pipeline.py::run_denoise_step`.

What comes out is this checkpoint's whole output: the velocity / noise
prediction `[1, S_img, 128]` that the flow-match sampler integrates. Use
`demo_denoise_latents.py` for the full loop.
"""

from __future__ import annotations

import sys
import time

import torch

from models.demos.flux_2_klein_9b.transformer.demo import _demo_common as common
from models.demos.flux_2_klein_9b.transformer.tt import pipeline as tt_pipeline
from models.demos.flux_2_klein_9b.transformer.tt import reference as tt_reference


def main(argv=None) -> int:
    args = common.build_parser(__doc__.splitlines()[0], steps=False).parse_args(argv)

    model = common.load_model(args)
    inputs = common.make_inputs(args)

    device = common.open_mesh(args)
    try:
        pipe = common.build(args, device, model)
        pipe.reset_invocations()

        started = time.time()
        sample = tt_pipeline.run_denoise_step(pipe, inputs)
        host = tt_pipeline.to_torch(sample, device).to(torch.float32)
        elapsed = time.time() - started

        print(
            f"[flux2-demo] velocity prediction {tuple(host.shape)} in {elapsed:.2f}s "
            f"(mean |v| {float(host.abs().mean()):.4f}, std {float(host.std()):.4f})",
            flush=True,
        )
        print(f"[flux2-demo] graduated stubs invoked: {dict(sorted(pipe.invocations.items()))}", flush=True)

        achieved_pcc = None
        if args.check_pcc:
            depth = pipe.depth()
            golden = tt_reference._hf_reference_denoise_step(
                model, inputs, dual_layers=depth["dual_layers"], single_layers=depth["single_layers"]
            )
            achieved_pcc = common.report_pcc(golden, host)

        common.write_outputs(
            args,
            "denoise_step",
            {"sample": host},
            {
                "call": "denoise_step",
                "depth": pipe.depth(),
                "full_depth": {"dual": pipe.full_dual_layers, "single": pipe.full_single_layers},
                "shape": list(host.shape),
                "seconds": elapsed,
                "invocations": pipe.invocations,
                "meta": inputs["meta"],
                "e2e_pcc": achieved_pcc,
                "tp": args.tp,
            },
        )
    finally:
        common.close_mesh(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
