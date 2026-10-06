# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tune paged prefill chunks without changing the first BF16-K/V attention call."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_contract
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--prefill-q-chunk", type=int, required=True)
    parser.add_argument("--prefill-k-chunk", type=int, required=True)
    args, rest = parser.parse_known_args()
    for value in (args.prefill_q_chunk, args.prefill_k_chunk):
        if value < 32 or value > 1024 or value % 32 or value & (value - 1):
            parser.error("Chunk sizes must be powers of two from32 through1024")
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(output)
    factory = OptimizedDecoder.from_state_dict.__func__
    observed = {}

    def create(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        helper = decoder.layer.self_attn.configured_chunked_prefill
        if helper is None:
            raise ValueError("Configured paged prefill backend is required")
        grid = helper.program.compute_with_storage_grid_size
        helper.program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=args.prefill_q_chunk,
            k_chunk_size=args.prefill_k_chunk,
            exp_approx_mode=False,
        )
        observed.update(
            paged_program=str(helper.program), compute=str(helper.compute), first_chunk="unchanged BF16 K/V program"
        )
        decoder.precision_policy["paged_prefill_probe"] = observed
        return decoder

    original = sys.argv
    sys.argv = [sys.argv[0], *rest]
    error = None
    try:
        with patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)):
            run_optimized_contract.main()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        sys.argv = original
        report = json.loads(output.read_text()) if output.exists() else {}
        report.update(
            paged_prefill_probe=observed, probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        if error:
            report["probe_error"] = error
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
