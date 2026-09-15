# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Create an explicitly synthetic fixture for the captured-input evaluator.

This is an interface smoke test, not a model-derived activation capture.
Writes only the explicitly selected fresh output path; never overwrites.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import captured_inputs as C


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1240)
    args = parser.parse_args()
    C.require(
        args.heads > 0 and args.length > 0 and args.length % 512 == 0,
        "Positive heads and positive length divisible by512 required",
    )
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(args.seed)
    artifact = dict(
        schema=C.SCHEMA,
        metadata=dict(
            causal=False,
            mask=None,
            scale=C.SCALE,
            provenance=dict(
                source_kind="synthetic",
                model_id="synthetic-random-not-a-model",
                layer_id="0",
                capture_stage="interface-smoke-only",
                seed=args.seed,
                torch_version=str(torch.__version__),
                fixture_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
        ),
    )
    for name in ("q", "k", "v"):
        artifact[name] = torch.randn(1, args.heads, args.length, 128, generator=generator).bfloat16()
    expected = C.validate_artifact(artifact)
    with args.output.open("xb") as stream:
        torch.save(artifact, stream)
    _, loaded = C.load_capture(args.output)
    C.require(expected["input_sha256"] == loaded["input_sha256"], "Fixture roundtrip changed inputs")
    print(json.dumps(loaded, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
