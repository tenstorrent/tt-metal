# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate decode BFP8 activation policies through the audited parity harness.

Expert/shared flags change the input to their gate/up projection only. Their
intermediate activations, down projections, weights, routing and output dtypes
retain the selected decoder policy. The attention flag rounds only the precise
attention query through BFP8 and restores FP32 before attention arithmetic.
"""

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedExperts,
    OptimizedSharedMLP,
)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--expert-activation-bfp8", action="store_true")
    parser.add_argument("--shared-activation-bfp8", action="store_true")
    parser.add_argument("--attention-query-bfp8", "--precise-bfp8-query", action="store_true")
    args, rest = parser.parse_known_args()
    output = Path(rest[rest.index("--output") + 1])
    factory = OptimizedDecoder.from_state_dict.__func__
    expert_chunk = OptimizedExperts._chunk
    shared_call = OptimizedSharedMLP.__call__
    observed = {}

    def record(boundary, original, lowered, quantized):
        # Read tensor descriptors only; runtime audits still forbid host data.
        if boundary not in observed:
            observed[boundary] = {
                "phase": "decode",
                "logical_shape": list(original.shape),
                "input_dtype": str(original.dtype),
                "quantized_dtype": str(quantized.dtype),
                "consumer_dtype": str(lowered.dtype),
                "consumer_layout": str(lowered.layout),
                "consumer_memory_config": str(lowered.memory_config()),
            }

    def chunk(self, x, routing, decode):
        if decode:
            quantized = ttnn.typecast(x, ttnn.bfloat8_b)
            record("expert_gate_up_input", x, quantized, quantized)
            x = quantized
        return expert_chunk(self, x, routing, decode)

    def shared(self, x):
        if x.shape[-2] == 1:
            quantized = ttnn.typecast(x, ttnn.bfloat8_b)
            record("shared_gate_up_input", x, quantized, quantized)
            x = quantized
        return shared_call(self, x)

    def create(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        if args.shared_activation_bfp8 and not isinstance(decoder.layer.shared_mlp, OptimizedSharedMLP):
            raise ValueError("Shared activation probe requires the optimized shared MLP (--defaults or --shared-dtype)")
        if args.attention_query_bfp8:
            attention = decoder.layer.self_attn
            precise = attention.decode_sdpa

            def rounded_query(q, *a, **kw):
                quantized = ttnn.typecast(q, ttnn.bfloat8_b)
                lowered = ttnn.typecast(quantized, ttnn.float32)
                record("precise_attention_query", q, lowered, quantized)
                return precise(lowered, *a, **kw)

            attention.decode_sdpa = rounded_query
        return decoder

    original_argv = sys.argv
    sys.argv = [sys.argv[0], *rest]
    failure = None
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)))
            if args.expert_activation_bfp8:
                stack.enter_context(patch.object(OptimizedExperts, "_chunk", chunk))
            if args.shared_activation_bfp8:
                stack.enter_context(patch.object(OptimizedSharedMLP, "__call__", shared))
            run_optimized_decoder.main()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        sys.argv = original_argv
        report = json.loads(output.read_text()) if output.exists() else {"decoder": "optimized"}
        report["activation_probe"] = vars(args)
        report["activation_probe_runtime"] = observed
        report["activation_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if failure is not None:
            report["activation_probe_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
