# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only shared-16 RNE residual ablations; no device allocations or timing.

Imports the existing LoFi operand model unchanged. All arithmetic after input
encoding is FP64; these are representation floors, not fused-kernel results.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("residual_v2_numerics", HERE / "numerics.py")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
MODEL.VARIANTS.update(
    {
        "rne44_qp7": ("e7", ("b4", "b4"), "e7", ("b4", "b4")),
        "rne444_qp7": ("e7", ("b4", "b4", "b4"), "e7", ("b4", "b4", "b4")),
        "rne48_qp7": ("e7", ("b4", "e5_b8"), "e7", ("b4", "e5_b8")),
        "rne444_konly_qp7": ("e7", ("b4", "b4", "b4"), "e7", ("exact",)),
        "rne444_vonly_qp7": ("e7", ("exact",), "e7", ("b4", "b4", "b4")),
    }
)
DEFAULT_VARIANTS = ["rne44_qp7", "rne444_qp7", "rne48_qp7", "pervalue_floor", "pervalue_rne"]


def representation_details(inputs, preprocessing, seed, variant, route):
    q, k, v, correction, v0 = MODEL.V1_NUMERICS.preprocess(inputs, preprocessing.removesuffix("_vmatch"), seed)
    _, kformats, _, vformats = MODEL.VARIANTS[variant]
    details = {}
    for name, values, formats in (("k", k, kformats), ("v", v, vformats)):
        parts = MODEL.components(values, formats, route)
        represented = torch.zeros_like(values)
        errors = []
        for component in parts:
            represented = represented + component
            delta = represented - values
            errors.append(
                dict(
                    l2_pct=MODEL.metrics(represented, values)["l2_pct"],
                    max_abs=float(delta.abs().max()),
                    column_error_mean_rms=float(delta.mean(0).square().mean().sqrt()),
                    zero_fraction=float((component == 0).double().mean()),
                )
            )
        details[name + "_residual_levels"] = errors
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1242])
    parser.add_argument("--variants", nargs="+", choices=MODEL.VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["normal", "outliers", "scaled_qk", "scaled_down", "common_q", "common_k", "common_v"],
    )
    parser.add_argument("--preprocessing", nargs="+", default=["none"])
    parser.add_argument("--route", choices=["host", "rne", "device"], default="host")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    MODEL.self_test()
    sources = [Path(__file__), HERE / "numerics.py", MODEL.V1 / "probe.py", MODEL.V1 / "numerics.py"]
    started = time.monotonic()
    with (HERE / (args.label + ".jsonl")).open("x") as output:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                args=vars(args),
                hostname=platform.node(),
                contract="CPU FP64 arithmetic; native shared-16 input encoding and LoFi operand-bit model; NOT device accuracy or performance",
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            )
        )
        for length in args.lengths:
            assert length % 512 == 0
            for seed in args.seeds:
                for distribution in args.distributions:
                    inputs = MODEL.V1_NUMERICS.FRONTIER.inputs_for(length, seed, distribution)
                    q, k, v = [x.squeeze().double() for x in inputs]
                    reference = torch.softmax(q @ k.T / math.sqrt(128), -1) @ v
                    for preprocessing in args.preprocessing:
                        for variant in args.variants:
                            details = representation_details(inputs, preprocessing, seed, variant, args.route)
                            for result in MODEL.evaluate(inputs, variant, preprocessing, args.route, seed, reference):
                                emit(
                                    dict(
                                        kind="attention_model",
                                        length=length,
                                        seed=seed,
                                        distribution=distribution,
                                        preprocessing=preprocessing,
                                        variant=variant,
                                        **result,
                                        **details,
                                    )
                                )
        emit(dict(kind="completed", seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
