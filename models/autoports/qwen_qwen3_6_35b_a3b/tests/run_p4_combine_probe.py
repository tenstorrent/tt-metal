# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact local combine component gate; run baseline then candidate with --reference.

No weights, dispatch, GEMMs, reduction, profiler, or serving are involved. CPU
constructs the dispatch ABI, including poisoned unused rows. Run under timeout.
Only main imports torch/TTNN, so --help and Python compilation are host-only.
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
E, K, H, RANKS = 256, 8, 2048, 4
PATTERNS = (
    ("zero", 0),
    ("one", 0),
    ("count31", 0),
    ("count32", 0),
    ("sparse_congruent", 0),
    ("many64_once", 0),
    ("many64", 0),
    ("many64", 1),
    ("many64_repeat", 1),
    ("sparse_congruent", 1),
    ("zero", 1),
    ("count_seq", 1),
)


def routes(seq, rank, pattern, generation):
    """Unique (token, top-k) destinations and distinct experts within each row."""
    result = []
    if pattern == "zero":
        return result
    if pattern == "one":
        result = [(63, seq - 1, 7)]
    elif pattern in ("count31", "count32", "count_seq"):
        count = {"count31": 31, "count32": 32, "count_seq": seq}[pattern]
        result = [(7, token, (token + rank) % K) for token in range(count)]
    elif pattern == "sparse_congruent":
        for slot, (expert, count) in enumerate(zip((0, 4, 12, 28, 44, 60), (1, 31, 32, 1, 31, 32))):
            result.extend((expert, token, slot) for token in range(count))
    elif pattern == "many64_once":
        result = [(expert, expert // K, expert % K) for expert in range(64)]
    elif pattern in ("many64", "many64_repeat"):
        result = [((token * K + slot) % 64, token, slot) for token in range(seq) for slot in range(K)]
    else:
        raise ValueError(pattern)
    # Rotate destinations and expert IDs between generations, keeping the same
    # count histogram while refreshing metadata as well as source bytes.
    result = [
        (rank * 64 + (expert + 4 * generation) % 64, (token + 3 * generation) % seq, slot)
        for expert, token, slot in result
    ]
    assert len({(token, slot) for _, token, slot in result}) == len(result)
    for token in range(seq):
        ids = [expert for expert, row, _ in result if row == token]
        assert len(ids) == len(set(ids))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--variant", default="unspecified", help="Evidence label, e.g. baseline or candidate")
    parser.add_argument("--fallback-shapes", action="store_true", help="Also exercise seq33 and seq64")
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--trace-region-size", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path; existing evidence is preserved")
    if not 1 <= args.timing_repeats <= 100:
        parser.error("--timing-repeats must be between 1 and 100")
    for key, value in (("TT_METAL_TRACE_ALLOC_TRACKING", "1"), ("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")):
        os.environ.setdefault(key, value)
        if os.environ[key] != value:
            parser.error(f"{key} must be {value}")
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") not in ("", "0"):
        parser.error("Device profiling must be disabled for this probe")
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference is not None and reference.get("status") != "PASS":
        parser.error("--reference must name a passing run")

    import torch

    import ttnn

    torch.set_num_threads(8)
    sequences = [32, 33, 64] if args.fallback_shapes else [32]
    native = REPO / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device"
    source_paths = [Path(__file__), native / "combine_program_factory.cpp"]
    source_paths += [
        native / "kernels" / name
        for name in (
            "dataflow/reader_untilize.cpp",
            "compute/untilize_combine.cpp",
            "dataflow/writer_untilize.cpp",
            "batch_assignment.hpp",
        )
    ]
    result = dict(
        schema=1,
        status="RUNNING",
        variant=args.variant,
        command=sys.argv,
        contract=dict(
            mesh=[1, 4],
            experts=E,
            owned_per_rank=64,
            topk=K,
            hidden=H,
            seq_lens=sequences,
            patterns=[list(pair) for pair in PATTERNS],
            seed="closed-form-v1",
            init_zeros=True,
            cluster_axis=0,
            dispatch_group_size=1,
        ),
        trace_tracking=True,
        skip_program_cache=False,
        profiler=False,
        source_paths=[str(p.relative_to(REPO)) for p in source_paths if p.exists()],
        exact_encoding="Each output row is all +0 if base==0; otherwise BF16 word[col]=base+(col%8). All 2048 words checked before encoding.",
        limits=[
            "Synthetic combine ABI; no dispatch/GEMM or serving/model quality claim.",
            "Timing includes trace allocation tracking and synchronization; component microbenchmark only.",
            "All signatures prewarmed; each optional shape trace released after its complete workload.",
            "Source paths do not identify the installed factory binary; --variant is bound to the owner build/install record.",
        ],
        cases=[],
        timings=[],
        captures=0,
        replays=0,
        releases=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")

    def exact(actual, expected, label):
        assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
        if expected.dtype == torch.bfloat16:
            actual = actual.contiguous().view(torch.int16)
            expected = expected.contiguous().view(torch.int16)
        else:
            actual, expected = actual.to(torch.int64), expected.to(torch.int64)
        different = actual != expected
        if bool(different.any()):
            coords = different.nonzero()[:8]
            samples = [(coord.tolist(), int(actual[tuple(coord)]), int(expected[tuple(coord)])) for coord in coords]
            raise AssertionError(f"{label}: unequal_words={int(different.sum())}, samples={samples}")

    def fixture(seq, pattern, generation):
        capacity = ((seq * K + 31 * 64 + 31) // 32) * 32
        poison = 0x6000 | (generation << 15)
        source = torch.full((RANKS, 1, capacity, H), poison, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        metadata = torch.zeros((RANKS, 1, capacity, 3), dtype=torch.int32)
        counts = torch.zeros((RANKS, 1, E), dtype=torch.int32)
        offsets = torch.zeros_like(counts)
        expected_bits = torch.zeros((RANKS, seq, K, H), dtype=torch.int32)
        metadata_records = []
        for rank in range(RANKS):
            # Poison metadata targets an in-range output: overreading padding
            # corrupts output instead of generating an out-of-range DMA/hang.
            metadata[rank, 0, :, 0] = rank
            metadata[rank, 0, :, 1] = seq - 1
            metadata[rank, 0, :, 2] = K - 1
            selected = routes(seq, rank, pattern, generation)
            for expert, _, _ in selected:
                counts[rank, 0, expert] += 1
            aligned = ((counts[rank, 0] + 31) // 32) * 32
            offsets[rank, 0] = aligned.cumsum(0) - aligned
            assert int(aligned.sum()) <= capacity
            consumed = [0] * E
            for expert, token, slot in selected:
                row = int(offsets[rank, 0, expert]) + consumed[expert]
                consumed[expert] += 1
                base = 0x1000 + ((rank * seq + token) * K + slot) * 8 + (generation << 15)
                bits = base + torch.arange(H, dtype=torch.int32) % 8
                source[rank, 0, row] = bits.to(torch.int16).view(torch.bfloat16)
                metadata[rank, 0, row] = torch.tensor([rank, token, slot], dtype=torch.int32)
                expected_bits[rank, token, slot] = bits
                metadata_records.append([rank, expert, row, token, slot])
        return dict(
            source=source,
            metadata=metadata,
            counts=counts,
            offsets=offsets,
            expected=expected_bits.to(torch.int16).view(torch.bfloat16),
            active_metadata=metadata_records,
            capacity=capacity,
            poison_u16=poison,
        )

    mesh, live_trace = None, None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        types = {
            "source": (ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "metadata": (ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            "counts": (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "offsets": (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        }

        def host_tensor(value, name):
            dtype, layout = types[name]
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout, mesh_mapper=mapper)

        def refresh(v, f):
            for name in types:
                ttnn.copy_host_to_device_tensor(host_tensor(f[name], name), v[name])

        def read(tensor, shape):
            return torch.stack(
                [ttnn.to_torch(local).reshape(shape).clone() for local in ttnn.get_device_tensors(tensor)]
            )

        def check_inputs(v, f, label):
            for name in types:
                actual = read(v[name], tuple(f[name].shape[1:]))
                exact(actual, f[name], f"{label}/{name}")

        def combine(v):
            return ttnn.experimental.deepseek_prefill.combine(
                v["source"],
                v["metadata"],
                v["counts"],
                v["offsets"],
                dispatch_group_size=1,
                experts_per_chip=E,
                num_experts_per_tok=K,
                seq_len_per_chip=v["seq"],
                cluster_axis=0,
                init_zeros=True,
            )

        def output_record(v, f, output, label):
            actual = read(output, (v["seq"], K, H))
            exact(actual, f["expected"], label)
            words = actual.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
            bases = words[..., 0]
            reconstructed = torch.where(bases[..., None] == 0, 0, bases[..., None] + torch.arange(H) % 8)
            exact(words, reconstructed, label + "/lossless_encoding")
            return dict(
                output_row_base_u16=bases.tolist(),
                counts=f["counts"].reshape(RANKS, E).tolist(),
                offsets=f["offsets"].reshape(RANKS, E).tolist(),
                active_metadata=f["active_metadata"],
                padding_metadata=["rank", v["seq"] - 1, K - 1],
                padding_source_u16=f["poison_u16"],
            )

        # Allocate every persistent input and compile every signature before
        # the first capture. Warm/copy/read paths are exercised here as well.
        variants = []
        eager_records = {}
        for seq in sequences:
            f = fixture(seq, "many64", 0)
            v = dict(seq=seq, capacity=f["capacity"])
            for name in types:
                v[name] = ttnn.to_device(host_tensor(f[name], name), mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            variants.append(v)
            for index, (pattern, generation) in enumerate(PATTERNS):
                f = fixture(seq, pattern, generation)
                refresh(v, f)
                output = combine(v)
                ttnn.synchronize_device(mesh)
                check_inputs(v, f, f"eager/{seq}/{index}")
                eager_records[(seq, index)] = output_record(v, f, output, f"eager/{seq}/{index}")
                ttnn.deallocate(output)
                del output
            refresh(v, fixture(seq, "zero", 0))
        ttnn.synchronize_device(mesh)
        entries = int(mesh.num_program_cache_entries())
        result["prepared_program_entries"] = entries
        result["prewarmed_seq_lens"] = sequences

        for v in variants:
            seq = v["seq"]
            live_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            result["captures"] += 1
            try:
                output = combine(v)
            finally:
                ttnn.end_trace_capture(mesh, live_trace, cq_id=0)
            # Positive control: a newly allocated live buffer must be rejected
            # before submitting work. No allocation exemptions are used.
            unsafe = ttnn.empty_like(v["source"])
            try:
                try:
                    ttnn.execute_trace(mesh, live_trace, cq_id=0, blocking=False)
                except RuntimeError as error:
                    if "device buffer(s) still alive before" not in str(error):
                        raise
                    result.setdefault("tracker_positive_controls", []).append(dict(seq=seq, rejected=True))
                else:
                    raise AssertionError("Trace tracker failed to reject the positive control")
            finally:
                ttnn.deallocate(unsafe)
                del unsafe
            previous_record = None
            for index, (pattern, generation) in enumerate(PATTERNS):
                f = fixture(seq, pattern, generation)
                changed = pattern != "many64_repeat"
                if changed:
                    refresh(v, f)
                ttnn.execute_trace(mesh, live_trace, cq_id=0, blocking=False)
                result["replays"] += 1
                ttnn.synchronize_device(mesh)
                check_inputs(v, f, f"trace/{seq}/{index}")
                record = output_record(v, f, output, f"trace/{seq}/{index}")
                assert record == eager_records[(seq, index)], "Eager/trace exact results differ"
                if not changed:
                    assert record == previous_record, "Unchanged repeat differs"
                previous_record = record
                assert int(mesh.num_program_cache_entries()) == entries, "A signature was not prewarmed"
                case = dict(
                    seq=seq,
                    capacity=v["capacity"],
                    index=index,
                    pattern=pattern,
                    generation=generation,
                    input_refreshed=changed,
                    exact_all_ranks=True,
                    exact_eager_match=True,
                    **record,
                )
                result["cases"].append(case)
                if reference is not None:
                    match = next(c for c in reference["cases"] if c["seq"] == seq and c["index"] == index)
                    assert case == match, f"Baseline/candidate differs at seq={seq} case={index}"
                # Isolate synchronized trace elapsed time from uploads/readback.
                # Do not use this tracking-enabled component number as serving performance.
                if pattern in ("sparse_congruent", "many64") and generation == 0:
                    elapsed = []
                    for _ in range(args.timing_repeats):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter_ns()
                        ttnn.execute_trace(mesh, live_trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        elapsed.append((time.perf_counter_ns() - start) / 1e6)
                        result["replays"] += 1
                    output_record(v, f, output, f"timing_repeat/{seq}/{index}")
                    result["timings"].append(
                        dict(
                            seq=seq,
                            pattern=pattern,
                            synchronized_trace_ms=elapsed,
                            median_ms=statistics.median(elapsed),
                            blocking=False,
                        )
                    )
                save()
            ttnn.release_trace(mesh, live_trace)
            live_trace = None
            result["releases"] += 1
            ttnn.deallocate(output)
            del output
        if reference is not None:
            assert result["contract"] == reference["contract"], "Reference fixture contracts differ"
            assert len(result["cases"]) == len(reference["cases"])
            result["reference_exact_match"] = True
            result["reference_path"] = str(args.reference)
        result.update(status="PASS", case_count=len(result["cases"]), allocation_exemptions=0)
    except BaseException as error:
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if mesh is not None:
                if live_trace is not None:
                    ttnn.release_trace(mesh, live_trace)
                    result["releases"] += 1
                ttnn.close_mesh_device(mesh)
                result["mesh_closed"] = True
        except BaseException as error:
            result.update(status="FAIL", cleanup_error=repr(error), cleanup_traceback=traceback.format_exc())
            raise
        finally:
            save()
    print(
        f"P4_COMBINE_PROBE_PASS cases={len(result['cases'])} bytes={args.output.stat().st_size} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
