# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact GR-down BF16 compaction control; production gather and wiring unchanged."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker as tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tt.compact_gr_down import compact_gr_down

MISMATCH_DIAGNOSTICS = []


def fixtures(count):
    """Rank-distinct finite and raw-bit inputs, with poisoned padding columns."""
    families = []
    for stress in ("random_finite", "raw_finite", "raw_nonfinite"):
        ranks = []
        for rank in range(4):
            rng = torch.Generator().manual_seed(777 + 1000 * count + rank)
            if stress != "random_finite":
                bits = torch.randint(0, 65536, (1, 1, count, 384), generator=rng).to(torch.int16)
                special = torch.tensor([0, 32768, 32640, 65408, 32705, 65473, 1, 32769], dtype=torch.int32).to(
                    torch.int16
                )
                bits[..., :8] = special.roll(rank)
                if stress == "raw_finite":
                    # Replace only exponent255 patterns; retain every finite
                    # subnormal/signed-zero bit pattern, including extremes.
                    nonfinite = (bits.to(torch.int32) & 0x7F80) == 0x7F80
                    bits[nonfinite] = (bits[nonfinite].to(torch.int32) ^ 0x0080).to(torch.int16)
                value = bits.view(torch.bfloat16)
            else:
                value = (torch.randn((1, 1, count, 384), generator=rng) + rank).bfloat16()
            for segment in range(4):
                value[..., segment * 96 + 80 : (segment + 1) * 96] = 1000 + rank * 10 + segment
            ranks.append((value,))
        families.append(ranks)
    return families


def baseline(value):
    return ttnn.concat([value[..., r * 96 : r * 96 + 80] for r in range(4)], dim=-1)


def canonicalize_host(value):
    bits = value.view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = bits & 0x7F80
    result = torch.where(exponent == 0, 0, torch.where(exponent == 0x7F80, bits & 0xFF80, bits))
    return result.to(torch.int16).view(torch.bfloat16)


def validate_canonicalization(path):
    evidence = json.loads(path.read_text())
    assert evidence["passed"] and evidence["unique_patterns_observed"] == 65536
    assert not evidence["inconsistent_mappings"] and evidence["q1_unique_controls_per_rank"] == 518
    observed = torch.arange(65536, dtype=torch.int32)
    for item in evidence["exceptions"]:
        observed[int(item["input"], 16)] = int(item["output"], 16)
    assert hashlib.sha256(observed.numpy().tobytes()).hexdigest() == evidence["mapping_sha256_int32_le"]
    actual = canonicalize_host(torch.arange(65536).to(torch.int16).view(torch.bfloat16))
    assert torch.equal(actual.view(torch.int16).to(torch.int32) & 0xFFFF, observed)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "all_65536_host_rules_match_measured_lookup": True,
        "exceptions": len(evidence["exceptions"]),
    }


def exhaustive_fixtures(count):
    """Two Q32 families cover all bits; two Q1 families cover518 specials/rank."""
    if count not in (1, 32):
        return []
    special = sorted(
        {i for i in range(65536) if (i & 0x7F80) in (0, 0x7F80)} | {0x0080, 0x8080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80}
    )
    families = []
    for batch in range(2):
        family = []
        for rank in range(4):
            bits = (
                (torch.arange(32 * 320, dtype=torch.int32) + (batch * 4 + rank) * 32 * 320) % 65536
                if count == 32
                else torch.tensor(
                    [special[(batch * 320 + i) % len(special)] for i in range(320)], dtype=torch.int32
                ).roll(rank)
            )
            live = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, count, 320)
            source = torch.full((1, 1, count, 384), 123 + rank, dtype=torch.bfloat16)
            for segment in range(4):
                source[..., segment * 96 : segment * 96 + 80] = live[..., segment * 80 : segment * 80 + 80]
            family.append((source,))
        families.append(family)
    return families


def read_ranks(value, physical=False):
    result = []
    for tensor in ttnn.get_device_tensors(value):
        host = ttnn.from_device(tensor)
        if physical:
            host = host.reshape(host.padded_shape)
        result.append(ttnn.to_torch(host))
    return result


def upload(family, mesh):
    # Host tilization preserves BF16 NaN payloads; normal device tilization may
    # canonicalize them before either implementation sees the stress fixture.
    return ttnn.from_torch(
        torch.cat([rank[0] for rank in family], dim=3).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        preserve_nan_values=True,
    )


def zero_padding(value, count):
    result = []
    for rank, physical in enumerate(read_ranks(value, physical=True)):
        assert tuple(physical.shape) == (1, 1, 32, 320), physical.shape
        nonzero = int(torch.count_nonzero(physical[..., count:, :].view(torch.int16)))
        result.append({"rank": rank, "padding_values": (32 - count) * 320, "nonzero_bits": nonzero})
    assert all(item["nonzero_bits"] == 0 for item in result), result
    return result


def classify(bits):
    exponent, mantissa = bits & 0x7F80, bits & 0x007F
    sign = "negative" if bits & 0x8000 else "positive"
    kind = (
        "nan"
        if exponent == 0x7F80 and mantissa
        else (
            "infinity"
            if exponent == 0x7F80
            else "subnormal"
            if exponent == 0 and mantissa
            else "zero"
            if exponent == 0
            else "normal"
        )
    )
    return f"{sign}_{kind}"


def exact(actual, expected, *, context=""):
    assert len(actual) == len(expected) == 4
    result = []
    for rank, (a, b) in enumerate(zip(actual, expected, strict=True)):
        assert a.shape == b.shape
        mismatch = a.view(torch.int16) != b.view(torch.int16)
        both_nan = torch.isnan(a) & torch.isnan(b)
        item = {
            "rank": rank,
            "exact": not bool(mismatch.any()),
            "accepted": not bool(mismatch.any()),
            "mismatched_bits": int(mismatch.sum()),
            "nan_payload_mismatches": int((mismatch & both_nan).sum()),
            "finite_mismatches": int((mismatch & torch.isfinite(b)).sum()),
            "expected_subnormals": int(
                (
                    ((b.view(torch.int16).to(torch.int32) & 0x7F80) == 0)
                    & ((b.view(torch.int16).to(torch.int32) & 0x7F) != 0)
                ).sum()
            ),
        }
        if bool(mismatch.any()):
            samples = []
            for coord in mismatch.nonzero()[:64].tolist():
                coord = tuple(coord)
                observed = int(a.view(torch.int16)[coord]) & 0xFFFF
                wanted = int(b.view(torch.int16)[coord]) & 0xFFFF
                samples.append(
                    {
                        "output_coordinate": list(coord),
                        "input_column": 96 * (coord[-1] // 80) + coord[-1] % 80 if a.shape[-1] == 320 else coord[-1],
                        "expected_bits": f"0x{wanted:04x}",
                        "actual_bits": f"0x{observed:04x}",
                        "expected_class": classify(wanted),
                        "actual_class": classify(observed),
                        "expected_float": str(float(b[coord])),
                        "actual_float": str(float(a[coord])),
                    }
                )
            # Persist before asserting: a failed finite/subnormal control must
            # retain the exact coordinates and patterns needed for diagnosis.
            MISMATCH_DIAGNOSTICS.append(
                {
                    "context": context,
                    **item,
                    "samples": samples,
                    "samples_truncated": int(mismatch.sum()) > len(samples),
                }
            )
        result.append(item)
    assert all(item["accepted"] for item in result), result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-timing", action="store_true")
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--all-rows", action="store_true", help="Exercise all Q1..32 instead of Q1/3/17/32")
    parser.add_argument(
        "--exhaustive", action="store_true", help="Add all65536 BF16 patterns at Q32 and518 Q1 specials/rank"
    )
    args = parser.parse_args()
    assert not args.output.exists(), f"Refusing to overwrite evidence: {args.output}"
    assert args.repeats > 0 and args.rounds > 0
    assert bool(tracker.TRACE_ALLOC_TRACKING) != args.clean_timing
    assert not args.clean_timing or not os.environ.get("TT_METAL_WATCHER")
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("", "0")
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    canonicalization = validate_canonicalization(
        root / "doc/optimized_full_model/gr_compaction_canonicalization_v1.json"
    )
    sources = [
        Path(__file__),
        root / "tt/compact_gr_down.py",
        root / "tt/multichip_decoder.py",
        root / "tt/fused_decoder.py",
    ]
    sources += sorted((root / "tt/kernels").glob("compact_gr_down_*.cpp"))
    repo = root.parents[2]
    sources += [
        repo / "ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp",
        repo / "ttnn/core/tensor/py_to_tt_tensor.cpp",
    ]
    report = {
        "passed": False,
        "rows": [],
        "mesh": [1, 4],
        "scope": "Synthetic actual-shape component control, not full-model acceptance or speedup",
        "accepted_domain": "ALL BF16 patterns. Baseline and native must exactly match independently canonicalized host output, without exemptions: exponent0 -> positive zero; exponent255 clears mantissa preserving sign; all other bits unchanged.",
        "canonicalization_evidence": canonicalization,
        "exhaustive": args.exhaustive,
        "mismatch_diagnostics": MISMATCH_DIAGNOSTICS,
        "tracking": bool(tracker.TRACE_ALLOC_TRACKING),
        "clean_timing": args.clean_timing,
        "watcher": os.environ.get("TT_METAL_WATCHER"),
        "program_cache_allocations_included": os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0",
        "logical_rows": list(range(1, 33)) if args.all_rows else [1, 3, 17, 32],
        "padding_check_scope": "Native physical output padding is read back and checked for positive-zero storage bits",
        "input_upload": "preserve_nan_values=True; all loaded logical BF16 bits verified against host fixtures",
        "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "boundary": "Preloaded persistent inputs; nonblocking replay loop, one synchronization at window end; no per-replay read/copy/sync",
        "preserved_math": "output[c] = legacy_canonicalize(input[96*(c//80)+c%80]); no CCL change",
        "equality": "Baseline/native/canonical-host strict logical BF16 bit equality for every family, including subnormal/signedzero/NaN/Inf inputs. Native padding must be positive-zero bits.",
    }
    mesh, trace = None, None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def phase(label):
        report["last_phase"] = label
        print(f"COMPACT_GR_PHASE {label}", flush=True)

    try:
        report["runtime_guard_positive_controls"] = []
        for name, operation in (("torch", lambda: torch.empty(1)), ("conversion", lambda: ttnn.to_torch(None))):
            try:
                with runtime_audit():
                    operation()
            except RuntimeError as error:
                assert "Runtime" in str(error) and "forbidden" in str(error)
                report["runtime_guard_positive_controls"].append({"name": name, "rejected": True})
            else:
                raise AssertionError("Runtime guard positive control did not reject")
        phase("open_mesh")
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        for count in range(1, 33) if args.all_rows else (1, 3, 17, 32):
            phase(f"q{count}:setup_and_baseline_eager")
            hosts = fixtures(count)
            labels = ["finite_rank_distinct", "raw_finite_signedzero_subnormal", "raw_all_bits"]
            if args.exhaustive:
                extra = exhaustive_fixtures(count)
                hosts.extend(extra)
                labels.extend(f"exhaustive_q{count}_pass{i}" for i in range(len(extra)))
            inputs = [[upload(family, mesh)] for family in hosts]
            expected = [
                [
                    canonicalize_host(torch.cat([rank[0][..., r * 96 : r * 96 + 80] for r in range(4)], dim=-1))
                    for rank in family
                ]
                for family in hosts
            ]
            row = {"count": count, "input_bits": [], "baseline_host_control": [], "eager": [], "traces": {}}
            report["rows"].append(row)
            for index, values in enumerate(inputs):
                row["input_bits"].append(exact(read_ranks(values[0]), [rank[0] for rank in hosts[index]]))
                with runtime_audit():
                    output = baseline(*values)
                row["baseline_host_control"].append(
                    exact(
                        read_ranks(output),
                        expected[index],
                        context=f"q{count}:baseline_host:family{index}",
                    )
                )
                del output
            for index, values in enumerate(inputs):
                phase(f"q{count}:native_eager:family{index}")
                with runtime_audit():
                    output = compact_gr_down(*values)
                row["eager"].append(
                    {
                        "family": labels[index],
                        "ranks": exact(read_ranks(output), expected[index], context=f"q{count}:native:{labels[index]}"),
                        "physical_padding": zero_padding(output, count),
                    }
                )
                del output
            # All immutable input sets and persistent replay inputs predate capture.
            live = [ttnn.empty_like(t) for t in inputs[0]]
            for name, function in (("five_op_baseline", baseline), ("native_compact", compact_gr_down)):
                phase(f"q{count}:{name}:warm")
                for src, dst in zip(inputs[0], live, strict=True):
                    ttnn.copy(src, dst)
                with runtime_audit():
                    warm = function(*live)
                del warm
                gc.collect()
                warmed_cache_entries = mesh.num_program_cache_entries()
                phase(f"q{count}:{name}:capture")
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    with runtime_audit():
                        output = function(*live)
                finally:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                item = {
                    "changed_input_replays": [],
                    "windows_ms_per_replay": [],
                    "runtime_audit": "passed_eager_warm_capture",
                }
                row["traces"][name] = item
                if tracker.TRACE_ALLOC_TRACKING:
                    phase(f"q{count}:{name}:allocation_guard_positive_control")
                    # This trace owns and overwrites its sole retained capture output.
                    tracker.acknowledge_corruptible(output)
                    sentinel = ttnn.empty_like(live[0])
                    try:
                        try:
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        except RuntimeError as error:
                            assert "still alive before trace replay" in str(error), str(error)
                            item["allocation_guard_positive_control"] = {"rejected": True, "error": str(error)}
                        else:
                            raise AssertionError("Allocation guard did not reject retained post-capture sentinel")
                    finally:
                        ttnn.deallocate(sentinel)
                        del sentinel
                phase(f"q{count}:{name}:changed_input_replay")
                for repeat in range(max(12, 2 * len(inputs))):
                    family = repeat % len(inputs)
                    for src, dst in zip(inputs[family], live, strict=True):
                        ttnn.copy(src, dst)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    item["changed_input_replays"].append(
                        {
                            "family": family,
                            "ranks": exact(
                                read_ranks(output),
                                expected[family],
                                context=f"q{count}:{name}:replay{repeat}:family{family}",
                            ),
                        }
                    )
                    if name == "native_compact":
                        item["changed_input_replays"][-1]["physical_padding"] = zero_padding(output, count)
                # Timing uses finite raw bits, retaining signed-zero/subnormal
                # stress. All nonfinite input families also require exact output.
                for src, dst in zip(inputs[1], live, strict=True):
                    ttnn.copy(src, dst)
                item["timing_family"] = "raw_finite_signedzero_subnormal"
                phase(f"q{count}:{name}:timing")
                for _ in range(args.rounds):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.repeats):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    item["windows_ms_per_replay"].append((time.perf_counter() - start) * 1000 / args.repeats)
                    exact(read_ranks(output), expected[1])
                    if name == "native_compact":
                        item["physical_padding_after_timing"] = zero_padding(output, count)
                item["program_cache_entries_before_capture"] = warmed_cache_entries
                item["program_cache_entries_after_replay"] = mesh.num_program_cache_entries()
                assert item["program_cache_entries_after_replay"] == warmed_cache_entries
                item["timing_eligible"] = args.clean_timing
                item["median_ms"] = statistics.median(item["windows_ms_per_replay"])
                item["repeats_per_window"] = args.repeats
                ttnn.release_trace(mesh, trace)
                trace = None
                del output
            row["speedup"] = (
                row["traces"]["five_op_baseline"]["median_ms"] / row["traces"]["native_compact"]["median_ms"]
            )
            print(json.dumps({"count": count, "speedup": row["speedup"]}), flush=True)
            del live, inputs
        report["passed"] = True
        phase("passed")
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
