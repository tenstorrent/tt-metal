# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run inherited public-contract checks with the optimized default decoder."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt import fused_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--contract",
        required=True,
        choices=("long_context", "batched", "prefix_continuation", "request_reuse", "run_decoder"),
    )
    parser.add_argument("--prefill-down-bfp4", action="store_true", help="Validate the reduced prefill-down candidate")
    parser.add_argument("--default-overrides", type=json.loads, default={})
    args, rest = parser.parse_known_args()
    module = importlib.import_module(f"models.autoports.google_gemma_4_26b_a4b_it.tests.{args.contract}")
    output = Path(rest[rest.index("--output") + 1])
    runtime_hash = hashlib.sha256(
        Path(__file__).parents[1].joinpath("tt/optimized_decoder.py").read_bytes()
    ).hexdigest()
    factory = OptimizedDecoder.from_state_dict.__func__
    runtime_policy = {}

    def create(cls, *a, **kw):
        for key, value in args.default_overrides.items():
            kw[key] = (
                getattr(ttnn, value)
                if key.endswith("_dtype")
                else getattr(ttnn.MathFidelity, value)
                if key.endswith("_fidelity")
                else value
            )
        if args.prefill_down_bfp4:
            kw["prefill_down_dtype"] = ttnn.bfloat4_b
        decoder = factory(cls, *a, **kw)
        runtime_policy.update(decoder.precision_policy)
        return decoder

    def forbid_functional(*args, **kwargs):
        raise AssertionError("Optimized contract dispatched the functional fallback")

    sys.argv = [sys.argv[0], "--decoder", "fused", *rest]
    with (
        patch.object(fused_decoder, "FusedDecoder", OptimizedDecoder),
        patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)),
        patch.object(FunctionalDecoder, "_forward", forbid_functional),
    ):
        try:
            module.main()
        finally:
            if output.exists():
                result = json.loads(output.read_text())
                result.update(
                    decoder="optimized",
                    runtime_sha256=runtime_hash,
                    contract=args.contract,
                    functional_fallback="forbidden",
                    precision_policy=runtime_policy,
                )
                if args.prefill_down_bfp4:
                    result["candidate"] = {"prefill_down_dtype": "bfloat4_b"}
                if args.default_overrides:
                    result.setdefault("candidate", {}).update(args.default_overrides)
                output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
