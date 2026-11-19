# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact native batched BFP8 K/V fill versus per-user loop; CPU check is device-free."""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tt.batched_cache_fill_candidate import batched_fill_pair

ROOT = Path(__file__).resolve().parents[1]


def native_source_evidence():
    repo = ROOT.parents[2]
    base = repo / "ttnn/cpp/ttnn/operations/experimental/paged_cache"
    paths = [
        base / "paged_cache_nanobind.cpp",
        base / "device/fill_cache/paged_fill_cache_device_operation.cpp",
        base / "device/fill_cache/paged_fill_cache_program_factory.cpp",
        base / "device/kernels/dataflow/reader_fill_cache_interleaved.cpp",
        base / "device/kernels/dataflow/writer_fill_cache_interleaved.cpp",
    ]
    text = [p.read_text() for p in paths]
    assert 'nb::arg("batch_idx_tensor")' in text[0]
    assert "input_tensor.dtype() == cache_tensor.dtype()" in text[1]
    assert "input_batch * num_blocks_of_work_per_batch" in text[2]
    assert "batch_idx_arr[batched_fill ? cur_batch : 0]" in text[4]
    assert "cur_batch = row_id / num_blocks_per_batch" in text[4]
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def cpu_check():
    native_source_evidence()
    cases = 0
    for batch in (1, 2, 3, 17, 31, 32):
        for seq in (1, 31, 32, 33, 64, 513):
            tiles = (seq + 31) // 32
            for start in (0, 32, 512):
                page_count = start // 32 + tiles + 2
                table = [list(range(user * page_count, (user + 1) * page_count)) for user in range(batch)]
                for indices in (list(range(batch)), list(reversed(range(batch)))):
                    bulk, baseline = {}, {}
                    for source in range(batch):
                        for tile in range(tiles):
                            page = table[indices[source]][start // 32 + tile]
                            for column in range(8):
                                baseline[page * 8 + column] = (source, tile, column)
                    for row in range(batch * tiles):
                        source, tile = divmod(row, tiles)
                        page = table[indices[source]][start // 32 + tile]
                        for column in range(8):
                            key = page * 8 + column
                            assert key not in bulk
                            bulk[key] = (source, tile, column)
                    assert baseline == bulk and len(bulk) == batch * tiles * 8
                    cases += 1
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    print(
        json.dumps(
            {
                "status": "pass",
                "cases": cases,
                "scope": "CPU row/page mapping only",
                "unaligned_start": "excluded; existing scheduler fallback retained",
            }
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 3, 17, 32])
    ap.add_argument("--lengths", nargs="+", type=int, default=[33, 513])
    ap.add_argument("--starts", nargs="+", type=int, default=[0, 32, 512])
    ap.add_argument("--reverse-users", action="store_true")
    ap.add_argument("--skip-page", action="store_true")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timing-samples", type=int, default=5)
    ap.add_argument("--replays-per-sample", type=int, default=5)
    ap.add_argument("--physical-device-ids", nargs=4, type=int)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    assert all(1 <= b <= 32 for b in args.batches)
    assert all(s > 0 for s in args.lengths) and all(s >= 0 and s % 32 == 0 for s in args.starts)
    assert args.steps >= 2 and min(args.warmup, args.timing_samples, args.replays_per_sample) >= 1
    assert not args.output.exists(), "Use a fresh output path"
    import torch
    import ttnn

    torch.set_num_threads(8)
    report = {
        "status": "running",
        "arguments": vars(args) | {"output": str(args.output)},
        "cases": [],
        "native_source_sha256": native_source_evidence(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), ROOT / "tt/batched_cache_fill_candidate.py")
        },
        "comparison": "exact complete decoded cache values; native raw-copy source establishes no repack",
        "trace_lifetime": "inputs, caches, tables, indices and copy/program warmups precede all captures",
        "environment": {
            k: os.environ.get(k)
            for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def upload(values, dtype=ttnn.bfloat8_b):
        return ttnn.from_torch(
            torch.cat(values, -1).contiguous(),
            device=mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )

    def metadata(value):
        return ttnn.from_torch(
            value,
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def read(tensor):
        return [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(tensor)]

    def check(actual, expected, label, entry):
        checks = []
        for rank, (a, b) in enumerate(zip(actual, expected)):
            assert a.shape == b.shape and a.dtype == b.dtype
            checks.append(
                {
                    "rank": rank,
                    "mismatches": int((a != b).sum()),
                    "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                }
            )
        entry[label] = checks
        save()
        assert len(checks) == 4 and all(x["finite"] and x["mismatches"] == 0 for x in checks), label

    def run_case(batch, length, start):
        seq = ((length + 31) // 32) * 32
        pages_per_user = start // 32 + seq // 32 + 2
        blocks = batch * pages_per_user
        indices = list(reversed(range(batch))) if args.reverse_users else list(range(batch))
        entry = {"batch": batch, "length": length, "padded_seq": seq, "start_pos": start, "eager": [], "trace": []}
        report["cases"].append(entry)
        save()
        print(f"CACHE begin B{batch} S{length} start{start}", flush=True)
        gens = [torch.Generator().manual_seed(20261113 + batch * 97 + rank) for rank in range(4)]
        initial = [
            upload(
                [
                    (torch.randn(blocks, 1, 32, 256, generator=g) * 0.125 + rank * 0.03125).bfloat16()
                    for rank, g in enumerate(gens)
                ]
            )
            for _ in range(2)
        ]
        caches = [[ttnn.empty_like(t) for t in initial] for _ in range(2)]
        bank, table_bank, host_tables = [], [], []
        for step in range(args.steps):
            pair = []
            for which in range(2):
                values = []
                for rank, g in enumerate(gens):
                    value = (
                        torch.randn(batch, 1, seq, 256, generator=g) * 0.25
                        + rank * 0.0625
                        + step * 0.125
                        + which * 0.03125
                    ).bfloat16()
                    value[:, :, length:, :] = 0
                    values.append(value)
                pair.append(upload(values))
            bank.append(pair)
            table = torch.randperm(blocks, generator=gens[0]).reshape(batch, pages_per_user).to(torch.int32)
            if args.skip_page:
                table[indices[-1], start // 32] = -1
            host_tables.append(table)
            table_bank.append(metadata(table))
        active = [ttnn.empty_like(t) for t in bank[0]]
        table = ttnn.empty_like(table_bank[0])
        batch_idx = metadata(torch.tensor(indices, dtype=torch.int32))
        allocations = [
            *initial,
            *(t for pair in caches for t in pair),
            *(t for pair in bank for t in pair),
            *table_bank,
            *active,
            table,
            batch_idx,
        ]
        traces = []

        def reset():
            for pair in caches:
                for src, dst in zip(initial, pair):
                    ttnn.copy(src, dst)

        def set_input(step):
            for src, dst in zip(bank[step], active):
                ttnn.copy(src, dst)
            ttnn.copy(table_bank[step], table)

        def invoke(index):
            if index:
                batched_fill_pair(*caches[1], *active, table, batch_idx, start_pos=start, disjoint_write_pages=True)
            else:
                shifted = table[:, start // 32 :]
                for row, destination in enumerate(indices):
                    for cache, value in zip(caches[0], active):
                        ttnn.experimental.paged_fill_cache(cache, value[row : row + 1], shifted, batch_idx=destination)

        try:
            reset()
            eager = []
            # Independent value-level page oracle includes untouched prefixes,
            # suffixes, old contents and zero padding, not just the native loop.
            oracle = [read(t) for t in initial]
            for step in range(args.steps):
                set_input(step)
                invoke(0)
                invoke(1)
                record = {"step": step}
                entry["eager"].append(record)
                for which in range(2):
                    inputs_host = read(bank[step][which])
                    for rank in range(4):
                        for source, destination in enumerate(indices):
                            for page in range(seq // 32):
                                physical = int(host_tables[step][destination, start // 32 + page])
                                if physical >= 0:
                                    oracle[which][rank][physical, 0] = inputs_host[rank][
                                        source, 0, page * 32 : (page + 1) * 32
                                    ]
                    check(read(caches[0][which]), oracle[which], f"baseline_{which}_oracle", record)
                    check(read(caches[1][which]), oracle[which], f"candidate_{which}_oracle", record)
                eager.append([[t.clone() for t in side] for side in oracle])
            reset()
            set_input(0)
            for index in range(2):
                for _ in range(args.warmup):
                    invoke(index)
            ttnn.synchronize_device(mesh)
            warmed = mesh.num_program_cache_entries()
            for index in range(2):
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                invoke(index)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                traces.append(tid)
            assert mesh.num_program_cache_entries() == warmed
            reset()
            for step in range(args.steps):
                set_input(step)
                for tid in traces:
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                record = {"step": step}
                entry["trace"].append(record)
                for index in range(2):
                    for which in range(2):
                        check(read(caches[index][which]), eager[step][which], f"impl{index}_cache{which}", record)
            times = [[], []]
            reset()
            set_input(0)
            for sample in range(args.timing_samples):
                for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter_ns()
                    for _ in range(args.replays_per_sample):
                        ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times[index].append((time.perf_counter_ns() - begin) / 1000 / args.replays_per_sample)
            entry["trace_us"] = {
                name: {"samples": samples, "median": statistics.median(samples)}
                for name, samples in zip(("baseline", "candidate"), times)
            }
            for index in range(2):
                for which in range(2):
                    check(read(caches[index][which]), eager[0][which], f"timed_impl{index}_cache{which}", entry)
            assert mesh.num_program_cache_entries() == warmed
            # Prefill is normally eager. Measure dispatch-inclusive region only
            # after release; that permits ordinary temporary allocations safely.
            for tid in traces:
                ttnn.release_trace(mesh, tid)
            traces.clear()
            eager_times = [[], []]
            for sample in range(args.timing_samples):
                for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter_ns()
                    invoke(index)
                    ttnn.synchronize_device(mesh)
                    eager_times[index].append((time.perf_counter_ns() - begin) / 1000)
            entry["eager_us"] = {
                name: {"samples": samples, "median": statistics.median(samples)}
                for name, samples in zip(("baseline", "candidate"), eager_times)
            }
            for index in range(2):
                for which in range(2):
                    check(read(caches[index][which]), eager[0][which], f"eager_timed_impl{index}_cache{which}", entry)
            entry["pass"] = True
            save()
            print(f"CACHE pass B{batch} S{length} start{start}", flush=True)
        finally:
            for tid in traces:
                ttnn.release_trace(mesh, tid)
            for tensor in allocations:
                ttnn.deallocate(tensor)

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        kwargs = {"trace_region_size": 16 * 1024 * 1024}
        if args.physical_device_ids:
            kwargs["physical_device_ids"] = args.physical_device_ids
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), **kwargs)
        for batch in args.batches:
            for length in args.lengths:
                for start in args.starts:
                    run_case(batch, length, start)
        report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    if sys.argv[1:] == ["--cpu-check"]:
        cpu_check()
    else:
        main()
