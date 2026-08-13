# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape-faithful TP4 sweep for Qwen3.6 linear recurrent decode matmuls.

Each TP rank owns 12 value heads, so both recurrent matmuls have the local
batch shape ``[32, 12, 1, 128] @ [32, 12, 128, 128]``.  This probe changes
only the matmul program or operand residency and reports kernel-family latency
and PCC.  It intentionally does not instantiate or modify the decoder.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc

BATCH = 32
LOCAL_VALUE_HEADS = 12
HEAD_DIM = 128
MESH_SHAPE = ttnn.MeshShape(1, 4)


def program(*, grid, block_w, per_core_n, subblock_w):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=block_w,
        out_subblock_h=1,
        out_subblock_w=subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def candidates():
    result = {"auto": None}
    for block_w in (1, 2, 4):
        result[f"grid4x1_w{block_w}_n1_s1"] = program(grid=(4, 1), block_w=block_w, per_core_n=1, subblock_w=1)
        for subblock_w in (1, 2):
            result[f"grid2x1_w{block_w}_n2_s{subblock_w}"] = program(
                grid=(2, 1), block_w=block_w, per_core_n=2, subblock_w=subblock_w
            )
        for subblock_w in (1, 2, 4):
            result[f"grid1x1_w{block_w}_n4_s{subblock_w}"] = program(
                grid=(1, 1), block_w=block_w, per_core_n=4, subblock_w=subblock_w
            )
        result[f"grid2x2_w{block_w}_n1_s1"] = program(grid=(2, 2), block_w=block_w, per_core_n=1, subblock_w=1)
    return result


def upload(value, mesh, memory_config):
    return ttnn.from_torch(
        value,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--residency",
        choices=("dram", "l1_preloaded", "dram_to_l1"),
        default="dram",
        help="dram_to_l1 includes both input conversions in every timed call",
    )
    parser.add_argument("--candidate", action="append", help="run only named candidate(s)")
    parser.add_argument("--minimum-pcc", type=float, default=0.999)
    parser.add_argument("--result-json")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 3:
        raise ValueError("use at least one warmup and three timed iterations")

    available = candidates()
    selected = args.candidate or list(available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown candidates {unknown}; choices={sorted(available)}")

    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)
    left_host = (torch.randn(BATCH, LOCAL_VALUE_HEADS, 1, HEAD_DIM) * 0.2).bfloat16()
    right_host = (torch.randn(BATCH, LOCAL_VALUE_HEADS, HEAD_DIM, HEAD_DIM) * 0.2).bfloat16()
    expected = torch.matmul(left_host.float(), right_host.float())
    compute = ttnn.init_device_compute_kernel_config(
        ttnn.Arch.BLACKHOLE,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    mesh = ttnn.open_mesh_device(MESH_SHAPE, trace_region_size=0)
    results = []
    try:
        source_memory = ttnn.L1_MEMORY_CONFIG if args.residency == "l1_preloaded" else ttnn.DRAM_MEMORY_CONFIG
        left = upload(left_host, mesh, source_memory)
        right = upload(right_host, mesh, source_memory)

        for name in selected:
            kwargs = {
                "dtype": ttnn.bfloat16,
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "compute_kernel_config": compute,
            }
            if available[name] is not None:
                kwargs["program_config"] = available[name]

            def run():
                in0, in1 = left, right
                if args.residency == "dram_to_l1":
                    in0 = ttnn.to_memory_config(in0, ttnn.L1_MEMORY_CONFIG)
                    in1 = ttnn.to_memory_config(in1, ttnn.L1_MEMORY_CONFIG)
                return ttnn.matmul(in0, in1, **kwargs)

            try:
                for _ in range(args.warmup):
                    run()
                ttnn.synchronize_device(mesh)
                samples = []
                output = None
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    output = run()
                    ttnn.synchronize_device(mesh)
                    samples.append((time.perf_counter() - start) * 1000.0)
                rank_outputs = [ttnn.to_torch(value).float() for value in ttnn.get_device_tensors(output)]
                pcc = [comp_pcc(expected, value, args.minimum_pcc) for value in rank_outputs]
                passed = all(item[0] for item in pcc)
                record = {
                    "candidate": name,
                    "status": "pass" if passed else "pcc_fail",
                    "median_ms": statistics.median(samples),
                    "minimum_ms": min(samples),
                    "pcc": [item[1] for item in pcc],
                }
            except Exception as error:
                record = {"candidate": name, "status": "unsupported", "error": repr(error)}
            results.append(record)
            print(json.dumps(record, sort_keys=True))
    finally:
        ttnn.close_mesh_device(mesh)

    passing = [item for item in results if item["status"] == "pass"]
    summary = {
        "left_shape_per_device": [BATCH, LOCAL_VALUE_HEADS, 1, HEAD_DIM],
        "right_shape_per_device": [BATCH, LOCAL_VALUE_HEADS, HEAD_DIM, HEAD_DIM],
        "mesh": [1, 4],
        "dtype": "BF16",
        "fidelity": "HiFi2",
        "residency": args.residency,
        "iterations": args.iterations,
        "minimum_pcc": args.minimum_pcc,
        "winner": min(passing, key=lambda item: item["median_ms"])["candidate"] if passing else None,
        "results": results,
    }
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
    if not passing:
        raise AssertionError("no correct recurrent-matmul candidate")


if __name__ == "__main__":
    main()
