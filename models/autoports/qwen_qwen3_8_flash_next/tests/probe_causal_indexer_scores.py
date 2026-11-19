# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Causal native scorer versus the selected SFPU scorer on real captured operands."""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from ttnn.tools import trace_allocation_tracker as tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tt.causal_indexer_scores import causal_indexer_scores
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import fused_indexer_scores, fused_selected_token_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="1,3,32")
    parser.add_argument("--starts", default="0,127,2047,2048,32767,205504,262112,3")
    parser.add_argument("--capture", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_score_windows_205504.pt"))
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--packed-decode", action="store_true", help="Use packed Q1 head rows; Q>1 keeps original storage"
    )
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(8)
    source = torch.load(args.capture, weights_only=True, mmap=True)
    # Replicated uploads cross the pinning threshold. Materialize mmap-backed
    # fixture pages before TTNN borrows/pins their storage for DMA.
    keys = source["keys"].bfloat16().clone()
    rows = [int(x) for x in args.rows.split(",")]
    starts = [int(x) for x in args.starts.split(",")]
    report = dict(
        passed=False,
        scope="real captured operands, four-chip causal-score and ordered-selection comparison",
        capture=str(args.capture),
        tracking=tracker.TRACE_ALLOC_TRACKING,
        packed_decode=args.packed_decode,
        regime="serial",
        cases=[],
        environment={
            k: os.environ.get(k)
            for k in ("TT_VISIBLE_DEVICES", "TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE")
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
    traces = []
    try:

        def host(x, dtype):
            return ttnn.from_torch(x.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)

        def device(x, dtype):
            return ttnn.from_torch(
                x.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        owner = SimpleNamespace(
            compute=ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
                math_approx_mode=False,
            )
        )
        print("INPUT_UPLOAD_BEGIN", flush=True)
        key_tt = device(keys, ttnn.bfloat16)
        print("KEY_UPLOAD_DONE", flush=True)
        for count in rows:
            print(f"ROW_INPUT_BEGIN rows={count}", flush=True)
            query = source["query"][:, :, :count].bfloat16().clone()
            query_tt = device(query, ttnn.bfloat16)
            pos_tt = device(torch.arange(count).float().reshape(1, 1, count, 1), ttnn.float32)

            # Prepare both complete graphs before any capture for this component lifetime.
            def old():
                score = fused_indexer_scores(owner, query_tt, key_tt, merge_heads=True, accurate_sum=True)
                return score, fused_selected_token_ids(score, pos_tt)

            def new():
                score = causal_indexer_scores(owner, query_tt, key_tt, pos_tt, packed_decode=args.packed_decode)
                return score, fused_selected_token_ids(score, pos_tt)

            for fn in (old, new):
                print(f"WARM_BEGIN rows={count} function={fn.__name__}", flush=True)
                warm = fn()
                ttnn.synchronize_device(mesh)
                del warm
                print(f"WARM_DONE rows={count} function={fn.__name__}", flush=True)
            outputs = []
            for fn in (old, new):
                print(f"CAPTURE_BEGIN rows={count} function={fn.__name__}", flush=True)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                with runtime_audit():
                    output = fn()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                for tensor in output:
                    tracker.acknowledge_corruptible(tensor)
                traces.append(tid)
                outputs.append(output)
            # Outputs may alias other trace scratch: read each before replaying the other.
            for start in starts:
                if start + count > 262144:
                    continue
                positions = torch.arange(start, start + count).float().reshape(1, 1, count, 1)
                ttnn.copy_host_to_device_tensor(host(positions, ttnn.float32), pos_tt)
                values = []
                times = []
                for tid, output in zip(traces, outputs, strict=True):
                    with runtime_audit():
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    values.append(
                        tuple(
                            [ttnn.to_torch(rank).clone() for rank in ttnn.get_device_tensors(tensor)]
                            for tensor in output
                        )
                    )
                    if not tracker.TRACE_ALLOC_TRACKING:
                        ttnn.synchronize_device(mesh)
                        begin = time.perf_counter()
                        for _ in range(args.iterations):
                            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        times.append((time.perf_counter() - begin) * 1000 / args.iterations)
                max_abs = 0.0
                exact = True
                ids_equal = True
                for rank in range(4):
                    for row in range(count):
                        complete = (start + row + 1) // 4
                        a, b = values[0][0][rank][0, 0, row, :complete], values[1][0][rank][0, 0, row, :complete]
                        if complete:
                            max_abs = max(max_abs, float((a - b).abs().max()))
                            exact = exact and torch.equal(a, b)
                    ids_equal = ids_equal and torch.equal(values[0][1][rank], values[1][1][rank])
                item = dict(
                    rows=count,
                    start=start,
                    max_abs=max_abs,
                    live_scores_exact=exact,
                    ordered_ids_equal=ids_equal,
                    baseline_ms=times[0] if times else None,
                    candidate_ms=times[1] if times else None,
                    passed=exact and ids_equal,
                )
                report["cases"].append(item)
                save()
                print(json.dumps(item), flush=True)
                assert item["passed"], item
            for tid in traces:
                ttnn.release_trace(mesh, tid)
            traces = []
            del outputs, query_tt, pos_tt, output
        report["passed"] = True
    finally:
        save()
        for tid in traces:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
