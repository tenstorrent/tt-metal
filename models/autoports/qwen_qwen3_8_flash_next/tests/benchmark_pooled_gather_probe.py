# SPDX-License-Identifier: Apache-2.0
"""Isolated pooled-key gather scheduling probe; not live device profiling.

Requires exclusive TP4 ownership, TT_METAL_TRACE_ALLOC_TRACKING=1 and
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 before import. This standalone probe
uses explicit completion synchronization; production phase collection does not.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_indexer_prefix_probe import copy_host, download, upload

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "1c75c43cd2241cfe8b540c9dd26882d2a38f0494"
BASELINE_PATH = "models/autoports/qwen_qwen3_8_flash_next/tt/kernels/qsa_pooled_paged.cpp"
BASELINE_SHA256 = "6aeb1cd74d6d4f97cb8ed0ce4ce742f79e05793cf904c712e10285aab18d1436"


def load_baseline(path=None):
    """Load immutable pre-adoption source, even after raw evidence is archived."""
    identity = str(path.resolve()) if path is not None else f"git:{BASELINE_COMMIT}:{BASELINE_PATH}"
    raw = (
        path.read_bytes()
        if path is not None
        else subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:{BASELINE_PATH}"], cwd=ROOT)
    )
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA256:
        raise RuntimeError("Pinned pre-adoption baseline source hash mismatch")
    return raw.decode(), identity


def cyclic_source(source):
    original = "for (uint32_t job = begin; job < begin + count; ++job) {"
    if source.count(original) != 1:
        raise ValueError("Expected exactly one native gather job loop")
    return source.replace(
        original,
        """const uint32_t stride = get_arg_val<uint32_t>(6);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t job = begin + item * stride;""",
    )


def _probe_transfer(attention, source, output, positions, page_table, *, seq, lane, cyclic):
    """Pinned native gather body with original or cyclic job ownership."""
    work = attention.blocks // 32 * 4
    grid = attention.device.compute_with_storage_grid_size()
    _, cores, group1, _, work1, work2 = ttnn.split_work_to_cores(grid, work, True)
    coords = ttnn.corerange_to_cores(cores, row_wise=True)
    workers = len(coords)
    tensors = [source, output, positions, page_table]
    args = [1, attention.batch_size, seq, attention.max_context, attention.pool_tokens // 32, lane]
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    runtime, offset = [], 0
    for i, core in enumerate(coords):
        if cyclic:
            partition = [i, (work + workers - 1 - i) // workers, workers]
        else:
            count = work1 if i < group1.num_cores() else work2
            partition = [offset, count]
            offset += count
        runtime.append(((core.x, core.y), [*[t.buffer_address() for t in tensors], *partition]))
    scratch = 2048 + 128 + 128 + attention.max_context // 32 * 4
    cb = ttnn.CBDescriptor(
        total_size=scratch,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=cyclic_source(attention.probe_baseline_source) if cyclic else attention.probe_baseline_source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=runtime,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))


def transfer(attention, source, output, positions, page_table, *, seq, lane):
    return _probe_transfer(attention, source, output, positions, page_table, seq=seq, lane=lane, cyclic=False)


def cyclic_transfer(attention, source, output, positions, page_table, *, seq, lane):
    return _probe_transfer(attention, source, output, positions, page_table, seq=seq, lane=lane, cyclic=True)


def check_output(actual, expected, complete, sentinel):
    padded = (complete + 31) // 32 * 32
    metrics = []
    for tensor in actual:
        value = tensor.reshape(65536, 128)
        valid = torch.equal(value[:complete], expected)
        partial = bool((value[complete:padded] == 0).all())
        suffix = bool((value[padded:] == sentinel).all())
        metrics.append(
            dict(
                valid_prefix_exact=valid,
                partial_tile_zero=partial,
                untouched_suffix_exact=suffix,
                valid_rows=complete,
                zero_rows=padded - complete,
                untouched_rows=65536 - padded,
            )
        )
        if not (valid and partial and suffix):
            raise AssertionError(metrics[-1])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-source", type=Path, help="Pinned baseline.cpp override; default reads the recorded git commit"
    )
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.replays < 1 or args.samples < 1:
        parser.error("Positive replays and samples required")
    tracking = dict(
        native_enabled=bool(ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()),
        python_enabled=bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
        skip_program_cache_environment=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        exemptions=[],
    )
    if (
        not tracking["native_enabled"]
        or not tracking["python_enabled"]
        or tracking["skip_program_cache_environment"] != "0"
    ):
        raise RuntimeError("Require actual native/Python trace tracking and no cache exemption")
    for key in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_WATCHER_DUMP_ALL"):
        if os.environ.get(key, "0") not in ("", "0"):
            raise RuntimeError(f"Disable {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    native_source, baseline_identity = load_baseline(args.baseline_source)
    (args.output / "baseline.cpp").write_text(native_source)
    (args.output / "candidate.cpp").write_text(cyclic_source(native_source))
    report = dict(
        status="started",
        trace_tracking=tracking,
        scope="isolated pooled-key gather only; no model/sampling work",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            baseline_identity: BASELINE_SHA256,
            **{str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT / "tt/paged_qsa.py")},
        },
        geometry=dict(batch=32, context=262144, pooled_rows=65536, key_width=128, page_tokens=32),
        timing="host asynchronous execute_trace through standalone completion synchronize; reset/copy/readback excluded",
        results=[],
    )

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    rng = torch.Generator().manual_seed(2026101609)
    # Distinct physical ownership: lane0 has full capacity, all31 other lanes
    # own4096 tokens each. No shared zero page or unallocated tested rows.
    page_counts = [8192] + [128] * 31
    total_pages = sum(page_counts)
    permutation = torch.randperm(total_pages, generator=rng).to(torch.int32)
    pages_host = torch.full((32, 8192), -1, dtype=torch.int32)
    offset = 0
    for lane, count in enumerate(page_counts):
        pages_host[lane, :count] = permutation[offset : offset + count]
        offset += count
    keys_host = torch.randn(1, 1, total_pages * 8, 128, generator=rng).bfloat16()
    positions_host = torch.zeros(32, 32, dtype=torch.int32)
    positions_host[:, 0] = torch.arange(32) * 64 + 127
    sentinel = -8192.0
    output_host = torch.full((1, 1, 65536, 128), sentinel, dtype=torch.bfloat16)
    torch.save(dict(keys=keys_host, page_table=pages_host, positions=positions_host), args.output / "inputs.pt")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=67108864)
    try:
        attention = SimpleNamespace(
            device=mesh,
            batch_size=32,
            blocks=65536,
            max_context=262144,
            pool_tokens=total_pages * 32,
            probe_baseline_source=native_source,
        )
        grid = mesh.compute_with_storage_grid_size()
        report["grid"] = [grid.x, grid.y]
        report["device_ids"] = list(mesh.get_device_ids())
        work = attention.blocks // 32 * 4
        _, cores, group1, _, work1, work2 = ttnn.split_work_to_cores(grid, work, True)
        coords = ttnn.corerange_to_cores(cores, row_wise=True)
        counts = [work1 if i < group1.num_cores() else work2 for i in range(len(coords))]
        starts = []
        offset = 0
        for count in counts:
            starts.append(offset)
            offset += count
        report["assignment"] = dict(
            work=work, workers=len(coords), native_counts=counts, native_starts=starts, cyclic_stride=len(coords)
        )
        report[
            "traffic_caveat"
        ] = "Key-row reads and output-tile writes unchanged; cyclic scheduling repeats page-table-prefix DMA on more live workers. Positions are read once per worker in both variants."
        keys = upload(keys_host, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)
        positions32 = upload(positions_host, mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pages32 = upload(pages_host, mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        positions1 = upload(positions_host[:1], mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pages1 = upload(pages_host[:1], mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        output = upload(output_host, mesh)
        reset = upload(output_host, mesh)
        # Returning to small bounds after fullcontext proves old scratch cannot
        # contaminate completed-prefix/partial-tile/untouched-suffix semantics.
        lane_cases = [
            (32, 0, [1, 4, 128, 132, 512, 1024, 2048, 4096, 262144, 128]),
            (32, 7, [128, 1024, 4096, 132]),
            (32, 31, [512, 2048, 4096, 1]),
            (1, 0, [128, 132, 512, 1024, 2048, 4096, 262144, 1]),
        ]
        for batch, lane, ends in lane_cases:
            attention.batch_size = batch
            positions = positions32 if batch == 32 else positions1
            pages = pages32 if batch == 32 else pages1

            def setup(end):
                positions_host[lane, 0] = end - 1
                copy_host(positions_host[:batch], positions, mesh)
                ttnn.copy(reset, output)

            def expected(end):
                complete = end // 4
                rows = torch.arange(complete)
                physical = pages_host[lane, rows // 8].long() * 8 + rows % 8
                return keys_host.reshape(-1, 128)[physical]

            # Warm both native programs before either capture, with persistent
            # input/output/reset allocations already established.
            for fn in (transfer, cyclic_transfer):
                setup(ends[0])
                fn(attention, keys, output, positions, pages, seq=1, lane=lane)
                ttnn.synchronize_device(mesh)
            for label, fn in (("baseline", transfer), ("candidate", cyclic_transfer)):
                trace = None
                try:
                    setup(ends[0])
                    ttnn.synchronize_device(mesh)
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    fn(attention, keys, output, positions, pages, seq=1, lane=lane)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    for index, end in enumerate(ends):
                        setup(end)
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        actual = download(output)
                        ref = expected(end)
                        metrics = check_output(actual, ref, end // 4, sentinel)
                        raw = args.output / f"b{batch}_lane{lane}_{index}_{end}_{label}.pt"
                        padded = (end // 4 + 31) // 32 * 32
                        torch.save(
                            dict(
                                expected_prefix=ref,
                                actual_written_rows=[x.reshape(65536, 128)[:padded].clone() for x in actual],
                            ),
                            raw,
                        )
                        del actual, ref
                        samples = []
                        for _ in range(args.samples):
                            begin = time.perf_counter()
                            for _ in range(args.replays):
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            samples.append((time.perf_counter() - begin) * 1000 / args.replays)
                        row = dict(
                            variant=label,
                            batch=batch,
                            lane=lane,
                            case_index=index,
                            total_tokens=end,
                            complete_pooled_rows=end // 4,
                            metrics=metrics,
                            wall_ms=samples,
                            median_ms=statistics.median(samples),
                            raw=str(raw),
                        )
                        live_jobs = (end // 4 + 31) // 32 * 4
                        live_workers = (
                            sum(start < live_jobs for start in starts)
                            if label == "baseline"
                            else min(len(coords), live_jobs)
                        )
                        table_bytes = (((end + 31) // 32 * 4 + 31) // 32) * 32
                        row["source_derived_payload"] = dict(
                            live_workers=live_workers,
                            positions_read_bytes=128 * len(coords),
                            page_table_read_bytes=table_bytes * live_workers,
                            key_row_read_bytes=(end // 4) * 256,
                            output_tile_write_bytes=live_jobs * 2048,
                            per_asic=True,
                        )
                        report["results"].append(row)
                        save()
                        print("POOLED_GATHER", json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True)
                finally:
                    if trace is not None:
                        ttnn.release_trace(mesh, trace)
        baseline = {
            (r["batch"], r["lane"], r["case_index"]): r for r in report["results"] if r["variant"] == "baseline"
        }
        report["comparisons"] = [
            dict(
                batch=r["batch"],
                lane=r["lane"],
                total_tokens=r["total_tokens"],
                case_index=r["case_index"],
                speedup=baseline[r["batch"], r["lane"], r["case_index"]]["median_ms"] / r["median_ms"],
            )
            for r in report["results"]
            if r["variant"] == "candidate"
        ]
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
