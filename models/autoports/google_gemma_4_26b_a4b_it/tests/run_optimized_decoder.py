# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run optimization candidates through the original HF and traced parity gates."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt import fused_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--defaults", action="store_true", help="Use the delivered OptimizedDecoder defaults")
    parser.add_argument(
        "--default-overrides", type=json.loads, default={}, help="One-variable controls of delivered defaults"
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--expert-gate-dtype", default="bfloat16")
    parser.add_argument("--expert-down-dtype", default="bfloat16")
    parser.add_argument("--expert-block-w", type=int, default=1)
    parser.add_argument("--expert-fidelity", default="LoFi")
    parser.add_argument("--prefill-grid", type=int, nargs=2)
    parser.add_argument("--prefill-block-w", type=int, default=1)
    parser.add_argument("--active-prefill", action="store_true")
    parser.add_argument("--expert-grid", type=int, nargs=2)
    parser.add_argument("--down-grid", type=int, nargs=2)
    parser.add_argument("--prefill-tokens", type=int)
    parser.add_argument("--compensated-qkv", action="store_true")
    parser.add_argument("--qkv-lanes", type=int, default=0)
    parser.add_argument("--qkv-terms", type=int, default=3)
    parser.add_argument("--qkv-grid", type=int, nargs=2)
    parser.add_argument("--qkv-block-w", type=int, default=1)
    parser.add_argument("--qkv-subblock-w", type=int, default=1)
    parser.add_argument("--qkv-separate", action="store_true")
    parser.add_argument("--qkv-fidelity", default="HiFi4")
    parser.add_argument("--qkv-dram", action="store_true")
    parser.add_argument("--qkv-dram-readers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--qkv-dram-block", type=int, default=1)
    parser.add_argument("--dense-prefill-2d", action="store_true")
    parser.add_argument("--dense-prefill-block-w", type=int, default=4)
    parser.add_argument("--shared-dtype")
    parser.add_argument("--shared-down-dtype")
    parser.add_argument("--shared-fidelity", default="LoFi")
    parser.add_argument("--shared-dram", action="store_true")
    parser.add_argument("--shared-readers", type=int, default=1)
    parser.add_argument("--shared-split", action="store_true")
    parser.add_argument("--prefill-dtype", default="bfloat16")
    parser.add_argument("--prefill-down-dtype")
    parser.add_argument("--prefill-fidelity", default="HiFi4")
    parser.add_argument("--prefill-l1", action="store_true")
    parser.add_argument("--sharded-norms", action="store_true")
    parser.add_argument("--shared-block", type=int, default=11)
    parser.add_argument("--expert-split", action="store_true")
    parser.add_argument("--residual-l1", action="store_true")
    parser.add_argument("--residual-sharded", action="store_true")
    parser.add_argument(
        "--sharded-norm-site", default="all", choices=["all", "input", "post", "common", "input_common", "post_common"]
    )
    args, rest = parser.parse_known_args()
    if args.qkv_dram and not args.defaults and not args.qkv_lanes:
        parser.error("--qkv-dram requires --qkv-lanes")
    runtime_hash = hashlib.sha256(
        Path(__file__).parents[1].joinpath("tt/optimized_decoder.py").read_bytes()
    ).hexdigest()
    factory = OptimizedDecoder.from_state_dict.__func__

    def build(cls, *a, **kw):
        if args.defaults:
            overrides = dict(args.default_overrides)
            for key, value in overrides.items():
                if key.endswith("_dtype"):
                    overrides[key] = getattr(ttnn, value)
                elif key.endswith("_fidelity"):
                    overrides[key] = getattr(ttnn.MathFidelity, value)
            return factory(cls, *a, **kw, **overrides)
        return factory(
            cls,
            *a,
            chunk_size=args.chunk_size,
            expert_gate_dtype=getattr(ttnn, args.expert_gate_dtype),
            expert_down_dtype=getattr(ttnn, args.expert_down_dtype),
            expert_block_w=args.expert_block_w,
            expert_fidelity=getattr(ttnn.MathFidelity, args.expert_fidelity),
            compensated_qkv=args.compensated_qkv,
            dense_prefill_2d=args.dense_prefill_2d,
            dense_prefill_block_w=args.dense_prefill_block_w,
            **(
                dict(
                    qkv_lanes=args.qkv_lanes,
                    qkv_terms=args.qkv_terms,
                    qkv_grid=args.qkv_grid,
                    qkv_block_w=args.qkv_block_w,
                    qkv_subblock_w=args.qkv_subblock_w,
                    qkv_separate=args.qkv_separate,
                    qkv_fidelity=getattr(ttnn.MathFidelity, args.qkv_fidelity),
                    qkv_dram=args.qkv_dram,
                    qkv_dram_readers=args.qkv_dram_readers,
                    qkv_dram_block=args.qkv_dram_block,
                )
                if args.qkv_lanes
                else {"qkv_lanes": 0}
            ),
            expert_grid=args.expert_grid,
            down_grid=args.down_grid,
            prefill_tokens=args.prefill_tokens,
            active_prefill=args.active_prefill,
            prefill_grid=args.prefill_grid,
            prefill_block_w=args.prefill_block_w,
            prefill_dtype=getattr(ttnn, args.prefill_dtype),
            prefill_down_dtype=getattr(ttnn, args.prefill_down_dtype or args.prefill_dtype),
            prefill_fidelity=getattr(ttnn.MathFidelity, args.prefill_fidelity),
            prefill_l1=args.prefill_l1,
            expert_split=args.expert_split,
            residual_l1=args.residual_l1,
            residual_sharded=args.residual_sharded,
            sharded_norms=args.sharded_norms,
            sharded_norm_site=args.sharded_norm_site,
            shared_dtype=getattr(ttnn, args.shared_dtype) if args.shared_dtype else None,
            shared_down_dtype=getattr(ttnn, args.shared_down_dtype) if args.shared_down_dtype else None,
            shared_fidelity=getattr(ttnn.MathFidelity, args.shared_fidelity),
            shared_dram=args.shared_dram,
            shared_readers=args.shared_readers,
            shared_split=args.shared_split,
            shared_block=args.shared_block,
            **kw,
        )

    runtime_policy = {}

    def create(cls, *a, **kw):
        decoder = build(cls, *a, **kw)
        runtime_policy.update(decoder.precision_policy)
        return decoder

    sys.argv = [sys.argv[0], "--decoder", "fused", *rest]
    with (
        patch.object(fused_decoder, "FusedDecoder", OptimizedDecoder),
        patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)),
    ):
        try:
            run_decoder.main()
        finally:
            output = Path(rest[rest.index("--output") + 1])
            if output.exists():
                report = json.loads(output.read_text())
                report["decoder"] = "optimized"
                report["candidate"] = (
                    {"defaults": True, "overrides": args.default_overrides} if args.defaults else vars(args)
                )
                report["runtime_sha256"] = runtime_hash
                report["precision_policy"] = runtime_policy
                output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
