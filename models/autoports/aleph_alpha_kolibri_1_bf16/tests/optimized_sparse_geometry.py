# SPDX-License-Identifier: Apache-2.0
"""Precision-locked geometry search on real recorded post-attention MoE input."""
import argparse
import gc
import itertools
import json
import time

import torch

import ttnn

from .optimized_coverage import ROOT, Harness
from .reference import norm
from .run_decoder import pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--larger-blocks", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(8)
    h = Harness(args.layer, allocation_tracking=False)
    try:
        x = torch.load(ROOT / f"doc/optimized_decoder/recorded_inputs/layer_{args.layer}.pt", weights_only=True)[
            :, :129
        ]
        attn = h.ref.attention(norm(x, h.weights["input_layernorm.weight"]), 0)
        residual = x + norm(attn, h.weights["post_attn_norm.weight"])
        moe_input = norm(residual[:, -1:], h.weights["post_attention_layernorm.weight"])
        expected = h.ref.moe(moe_input)
        inp = h.tt(moe_input[None])
        policies = dict(gate=dict(grid=(8, 2), k=4, block=1, sub=1), down=dict(grid=(8, 2), k=4, block=1, sub=1))

        def config(m, n):
            c = policies["gate" if n == 1024 else "down"]
            cores = c["grid"][0] * c["grid"][1]
            per_n = (n // 32 + cores - 1) // cores
            per_n = ((per_n + c["block"] - 1) // c["block"]) * c["block"]
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=c["grid"],
                in0_block_w=c["k"],
                out_subblock_h=1,
                out_subblock_w=c["sub"],
                out_block_h=1,
                out_block_w=c["block"],
                per_core_M=(m + 31) // 32,
                per_core_N=per_n,
                fuse_batch=False,
                mcast_in0=True,
            )

        h.model._sparse_config = config
        result = dict(provenance=h.provenance, input_source="recorded post-attention activation", rows=[])
        output = (
            ROOT / f'doc/optimized_decoder/sparse_geometry_{args.layer}{"_large" if args.larger_blocks else ""}.json'
        )
        if args.output:
            output = ROOT / "doc/optimized_decoder" / args.output
        if args.larger_blocks:
            policies["gate"] = dict(grid=(8, 4), k=20, block=1, sub=1)
            policies["down"] = dict(grid=(10, 8), k=16, block=1, sub=1)
        for role in ["gate"] if args.larger_blocks else ["gate", "down"]:
            best = None
            for grid, k in itertools.product(
                [(4, 1), (4, 2), (8, 1), (8, 2), (8, 4), (10, 4), (10, 8)],
                ([20, 40, 80] if args.larger_blocks else [1, 2, 4, 5, 8, 10, 16, 20]),
            ):
                kt = 80 if role == "gate" else 16
                if kt % k:
                    continue
                n = 32 if role == "gate" else 80
                per_n = (n + grid[0] * grid[1] - 1) // (grid[0] * grid[1])
                for block in sorted({1, per_n}):
                    sub = max(d for d in ([1, 2, 4] if h.model.policy.expert_fp32 else [1, 2, 4, 8]) if block % d == 0)
                    c = dict(grid=grid, k=k, block=block, sub=sub)
                    policies[role] = c
                    row = dict(role=role, config=c, other_role=dict(policies["down" if role == "gate" else "gate"]))
                    tid = None
                    try:
                        out = h.model._moe(inp)
                        row["pcc"] = pcc(expected, ttnn.to_torch(out))
                        del out
                        gc.collect()
                        ttnn.synchronize_device(h.mesh)
                        tid = ttnn.begin_trace_capture(h.mesh, cq_id=0)
                        out = h.model._moe(inp)
                        ttnn.end_trace_capture(h.mesh, tid, cq_id=0)
                        ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                        start = time.perf_counter()
                        for _ in range(100):
                            ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=False)
                        ttnn.synchronize_device(h.mesh)
                        row["traced_moe_ms"] = (time.perf_counter() - start) * 10
                        row["replay_pcc"] = pcc(expected, ttnn.to_torch(out))
                        del out
                        if (
                            row["pcc"] >= 0.995
                            and row["replay_pcc"] >= 0.995
                            and (best is None or row["traced_moe_ms"] < best["traced_moe_ms"])
                        ):
                            best = row.copy()
                    except RuntimeError as error:
                        row["error"] = str(error).split("backtrace:")[0]
                    finally:
                        if tid is not None:
                            ttnn.release_trace(h.mesh, tid)
                    result["rows"].append(row)
                    print(json.dumps(row), flush=True)
                    output.write_text(json.dumps(result, indent=2) + "\n")
            if best:
                policies[role] = best["config"]
                result.setdefault("best", {})[role] = best
            output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        h.close()


if __name__ == "__main__":
    main()
