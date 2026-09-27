# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate fused gather-QKV's K block without changing the model runtime.

Run with the regular multichip harness arguments. The default control sets
K22; --probe-k-block=44 reproduces the optimized-default failing geometry.
Device execution must be serialized by the invoking agent.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tests import run_multichip_decoder as harness


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-k-block", type=int, choices=(11, 22, 44), default=22)
    args, harness_args = parser.parse_known_args()
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *harness_args, "--output", str(args.output)]
    original = harness.MultichipDecoder.from_state_dict
    policies = []

    def factory(cls, state, **kwargs):
        decoder = original(state, **kwargs)
        if not decoder.fused_agmm or not decoder.sharded_residual:
            raise ValueError("K-block control requires fused gather-QKV and carried sharded residuals")
        projection = decoder.layer.self_attn.source.weights.wqkv
        if projection.program is not projection.projection.program:
            raise ValueError("Expected the fused wrapper to share its underlying projection configuration")
        before = projection.program.in0_block_w
        projection.program.in0_block_w = args.probe_k_block
        policy = dict(
            layer=decoder.layer_idx,
            in0_block_w_before=before,
            in0_block_w_after=projection.program.in0_block_w,
            local_hidden=projection.local_hidden,
            gather_slice_k_tiles=projection.local_hidden // 32,
            qkv_weight_dtype=str(projection.weight.dtype),
            attention_ccl_dtype=str(decoder.attention_ccl_dtype),
            semaphore_core_grid=str(decoder.ccl.ccl_cores),
        )
        policies.append(policy)
        print("SHARDED_KBLOCK_CONTROL", json.dumps(policy), flush=True)
        return decoder

    harness.MultichipDecoder.from_state_dict = classmethod(factory)
    old_mtime = args.output.stat().st_mtime_ns if args.output.exists() else None
    try:
        harness.main()
    finally:
        if args.output.exists() and args.output.stat().st_mtime_ns != old_mtime:
            report = json.loads(args.output.read_text())
            report["sharded_kblock_probe"] = dict(
                command=original_argv,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                actual_tp4_policy=policies,
                original_pcc_gate=0.995,
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
