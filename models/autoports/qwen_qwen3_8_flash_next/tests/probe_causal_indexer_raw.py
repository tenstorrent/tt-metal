# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded externally: locate raw causal scorer failures before baseline work.

Example (run only when the complete mesh is available)::

    timeout 300 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_causal_indexer_raw \
        --output doc/raw_q1.json --rows 1 --capacity 128 --num-cores 1

A phase is persisted and flushed before every device-facing stage. The first
native execution has no old-scorer work ahead of it. This is an eager correctness
and hang-localization probe, not trace or performance acceptance evidence.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.causal_indexer_scores import causal_indexer_dots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=128, choices=(128, 65536))
    parser.add_argument("--num-cores", type=int, default=1)
    parser.add_argument("--rows", default="1,3")
    parser.add_argument("--starts", default="0,3,127")
    parser.add_argument("--capture", type=Path, help="Optional real fixture containing BF16-convertible query and keys")
    parser.add_argument("--selection", action="store_true", help="Also compare exact ordered selected token IDs")
    args = parser.parse_args()
    rows = [int(value) for value in args.rows.split(",")]
    starts = [int(value) for value in args.starts.split(",")]
    if not rows or any(not 1 <= count <= 32 for count in rows):
        parser.error("--rows must be a nonempty comma-separated subset of 1..32")
    if not starts or min(starts) < 0 or max(starts) + max(rows) > args.capacity * 4:
        parser.error("--starts must keep every row within the capacity*4 token range")
    if args.num_cores < 1:
        parser.error("--num-cores must be positive")
    if args.output.exists():
        parser.error("--output already exists; use a fresh evidence path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        passed=False,
        phase="created",
        capacity=args.capacity,
        num_cores=args.num_cores,
        rows=rows,
        starts=starts,
        fixture=str(args.capture) if args.capture else "deterministic normal seed 7319",
        mode="eager raw-op localization; no trace or timing claim",
        cases=[],
        phases=[],
        environment={
            key: os.environ.get(key)
            for key in (
                "TT_VISIBLE_DEVICES",
                "TT_METAL_WATCHER",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_MESH_GRAPH_DESC_PATH",
            )
        },
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def phase(name, **details):
        item = dict(phase=name, wall_time=time.time(), **details)
        report["phase"] = name
        report["phases"].append(item)
        save()
        print(json.dumps(item), flush=True)

    torch.set_num_threads(8)
    phase("host_operands_begin")
    if args.capture:
        source = torch.load(args.capture, weights_only=True, mmap=True)
        query_host = source["query"][:, :, : max(rows)].bfloat16().clone()
        keys_host = source["keys"][:, :, : args.capacity].bfloat16().clone()
    else:
        rng = torch.Generator().manual_seed(7319)
        query_host = torch.randn((1, 4, max(rows), 128), generator=rng).bfloat16()
        keys_host = torch.randn((1, 1, args.capacity, 128), generator=rng).bfloat16()
    if tuple(query_host.shape) != (1, 4, max(rows), 128) or tuple(keys_host.shape) != (1, 1, args.capacity, 128):
        raise ValueError("Fixture does not contain the requested query/key shapes")
    phase("host_operands_done")
    mesh = None
    try:
        phase("mesh_open_begin")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        phase("mesh_open_returned")
        ttnn.synchronize_device(mesh)
        phase("mesh_open_synced")

        def upload(value, dtype):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def read(value):
            return [ttnn.to_torch(rank).clone() for rank in ttnn.get_device_tensors(value)]

        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        phase("keys_upload_begin")
        keys = upload(keys_host, ttnn.bfloat16)
        ttnn.synchronize_device(mesh)
        phase("keys_upload_done")
        for count in rows:
            phase("query_positions_upload_begin", rows=count)
            query = upload(query_host[:, :, :count], ttnn.bfloat16)
            positions = upload(torch.arange(count).float().reshape(1, 1, count, 1), ttnn.float32)
            ttnn.synchronize_device(mesh)
            phase("query_positions_upload_done", rows=count)

            # Crucially, raw native runs before ANY reference graph is invoked.
            phase("raw_zero_warm_begin", rows=count)
            warm = causal_indexer_dots(query, keys, positions, num_cores=args.num_cores)
            phase("raw_zero_warm_submitted", rows=count)
            ttnn.synchronize_device(mesh)
            phase("raw_zero_warm_synced", rows=count)
            warm_values = read(warm)
            phase("raw_zero_warm_read_done", rows=count)
            if count <= 3:
                assert all(torch.count_nonzero(value).item() == 0 for value in warm_values)
            del warm_values, warm

            phase("reference_begin", rows=count)
            q32, k32 = ttnn.typecast(query, ttnn.float32), ttnn.typecast(keys, ttnn.float32)
            expanded_q = ttnn.reshape(q32, (4, count, 1, 128))
            block_size = 65536 if count == 1 else 2048
            blocks = []
            for start in range(0, args.capacity, block_size):
                stop = min(start + block_size, args.capacity)
                products = ttnn.multiply(expanded_q, k32[:, :, start:stop])
                reduced = ttnn.sum(
                    products,
                    dim=-1,
                    keepdim=True,
                    compute_kernel_config=compute,
                    fast_and_approximate_mode=False,
                )
                blocks.append(ttnn.reshape(reduced, (1, 4, count, stop - start)))
            reference = ttnn.concat(blocks, dim=-1) if len(blocks) > 1 else blocks[0]
            ttnn.synchronize_device(mesh)
            phase("reference_synced", rows=count)
            expected = read(reference)
            phase("reference_read_done", rows=count)

            for start in starts:
                phase("positions_refresh_begin", rows=count, start=start)
                host_positions = ttnn.from_torch(
                    torch.arange(start, start + count).float().reshape(1, 1, count, 1),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                )
                ttnn.copy_host_to_device_tensor(host_positions, positions)
                ttnn.synchronize_device(mesh)
                phase("positions_refresh_done", rows=count, start=start)
                phase("raw_case_begin", rows=count, start=start)
                result = causal_indexer_dots(query, keys, positions, num_cores=args.num_cores)
                phase("raw_case_submitted", rows=count, start=start)
                ttnn.synchronize_device(mesh)
                phase("raw_case_synced", rows=count, start=start)
                actual = read(result)
                phase("raw_case_read_done", rows=count, start=start)
                max_abs, max_rel, exact, invalid_zero, finite = 0.0, 0.0, True, True, True
                for rank in range(len(actual)):
                    finite &= bool(torch.isfinite(actual[rank]).all())
                    for row in range(count):
                        live = (start + row + 1) // 4
                        a = expected[rank][0, :, row, :live]
                        b = actual[rank][0, :, row, :live]
                        if live:
                            delta = (a - b).abs()
                            max_abs = max(max_abs, float(delta.max()))
                            max_rel = max(max_rel, float((delta / a.abs().clamp_min(1e-8)).max()))
                            exact &= torch.equal(a, b)
                        invalid_zero &= torch.count_nonzero(actual[rank][0, :, row, live:]).item() == 0
                item = dict(
                    rows=count,
                    start=start,
                    ranks=len(actual),
                    valid_dots_exact=exact,
                    max_abs=max_abs,
                    max_relative=max_rel,
                    invalid_columns_zero=invalid_zero,
                    finite=finite,
                    passed=finite and invalid_zero and exact,
                )
                if args.selection:
                    from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import fused_selected_token_ids

                    phase("selection_begin", rows=count, start=start)
                    selections = []
                    for dots in (reference, result):
                        scores = ttnn.multiply(ttnn.sum(ttnn.relu(dots), dim=1, keepdim=True), 128**-0.5)
                        selections.append(read(fused_selected_token_ids(scores, positions)))
                    item["ordered_ids_equal"] = all(torch.equal(a, b) for a, b in zip(*selections, strict=True))
                    item["passed"] &= item["ordered_ids_equal"]
                    phase("selection_done", rows=count, start=start)
                report["cases"].append(item)
                save()
                print(json.dumps(item), flush=True)
                assert item["passed"], item
                del result, actual
            del query, positions, reference, expected, q32, k32, expanded_q, blocks, products, reduced
        report["passed"] = True
        phase("all_checks_passed")
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        save()
        raise
    finally:
        if mesh is not None:
            phase("mesh_close_begin")
            ttnn.close_mesh_device(mesh)
            phase("mesh_close_done")
        save()


if __name__ == "__main__":
    main()
