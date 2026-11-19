# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only reconstruction diagnostic; excludes TTNN conversion and device reads."""

import argparse
import hashlib
import itertools
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch


def baseline(shards, slots, active_rows, vocab_size):
    values = torch.cat(shards, dim=3).reshape(-1, vocab_size).float()
    result = torch.zeros(len(slots), 1, vocab_size)
    for row in active_rows:
        result[row, 0] = values[slots[row]]
    return result


def selected_first(shards, slots, active_rows, vocab_size):
    result = torch.zeros(len(slots), 1, vocab_size)
    if active_rows:
        indices = torch.tensor([slots[row] for row in active_rows], dtype=torch.long)
        selected = [shard.reshape(-1, shard.shape[-1]).index_select(0, indices) for shard in shards]
        values = torch.cat(selected, dim=-1).float()
        for selected_row, row in enumerate(active_rows):
            result[row, 0] = values[selected_row]
    return result


def exact_check(shards, full_values, slots, active_rows, vocab_size):
    expected = torch.zeros(len(slots), 1, vocab_size)
    for row in active_rows:
        expected[row, 0] = full_values[slots[row]]
    actual = selected_first(shards, slots, active_rows, vocab_size)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == torch.float32
    # Include signed zero, NaN payloads and infinities in the exact oracle.
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), (slots, active_rows)
    return 1


def measure(shards, slots, active_rows, vocab_size, repeats):
    samples = {"baseline": [], "selected_first": []}
    functions = {"baseline": baseline, "selected_first": selected_first}
    for _ in range(5):
        for fn in functions.values():
            fn(shards, slots, active_rows, vocab_size)
    for repeat in range(repeats):
        order = ("baseline", "selected_first") if repeat % 2 == 0 else ("selected_first", "baseline")
        for name in order:
            start = time.perf_counter_ns()
            value = functions[name](shards, slots, active_rows, vocab_size)
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
            del value
    summary = {
        name: {"median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)}
        for name, values in samples.items()
    }
    summary["median_saved_ms"] = summary["baseline"]["median_ms"] - summary["selected_first"]["median_ms"]
    summary["raw_ms"] = samples
    summary["slots"] = slots
    summary["active_rows"] = active_rows
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous experiment evidence; choose a fresh output")
    assert not any(name == "ttnn" or name.startswith("ttnn.") for name in sys.modules)
    root = Path(__file__).resolve().parents[1]
    config_path = root / "doc/functional_decoder/target_text_config.json"
    config = json.loads(config_path.read_text())
    vocab_size = config["vocab_size"]
    assert vocab_size == 248320 and vocab_size % 4 == 0
    report = dict(
        passed=False,
        scope="CPU torch reconstruction only; excludes TTNN conversion, untilization, device read and synchronization",
        config=str(config_path.relative_to(root)),
        vocab_size=vocab_size,
        physical_rows=32,
        ranks=4,
        shard_shape=[1, 1, 32, vocab_size // 4],
        shard_dtype="bfloat16",
        output_dtype="float32",
        source_sha256=hashlib.sha256((root / "tt/generator.py").read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        torch_default_dtype=str(torch.get_default_dtype()),
        timings={},
        equivalence={},
        repeat_count=args.repeats,
        limitations="Synthetic already-converted CPU BF16 shards; allocation/cache/CPU contention affect timing. No inference about TTNN conversion/read speed or P2 pass. The running server can contend for CPU resources.",
    )
    started = time.monotonic()
    try:
        assert torch.get_default_dtype() == torch.float32
        torch.set_num_threads(8)
        rng = torch.Generator().manual_seed(42017)
        shards = [torch.randn(report["shard_shape"], dtype=torch.bfloat16, generator=rng) for _ in range(4)]
        for rank, shard in enumerate(shards):
            for slot in range(32):
                shard[0, 0, slot, :6] = torch.tensor(
                    [0.0, -0.0, float("inf"), -float("inf"), float("nan"), slot * 4 + rank], dtype=torch.bfloat16
                )
        original_bits = [shard.view(torch.int16).clone() for shard in shards]
        with torch.inference_mode():
            full_values = torch.cat(shards, dim=3).reshape(-1, vocab_size).float()
            for threads in (8, 1):
                torch.set_num_threads(threads)
                checked = 0
                for slot in range(32):
                    checked += exact_check(shards, full_values, [slot], [0], vocab_size)
                for slots in itertools.permutations(range(32), 2):
                    # All 992 ordered pairs, including every noncontiguous/reversed pair.
                    checked += exact_check(shards, full_values, list(slots), [0, 1], vocab_size)
                for slots in ([31, 7, 0, 19], [19, 0, 7, 31]):
                    for count in range(5):
                        for rows in itertools.permutations(range(4), count):
                            checked += exact_check(shards, full_values, slots, list(rows), vocab_size)
                for slots in (list(reversed(range(32))), list(range(7, 32)) + list(range(7))):
                    for rows in ([], [31, 0], list(range(0, 32, 2)), list(reversed(range(32)))):
                        checked += exact_check(shards, full_values, slots, rows, vocab_size)
                report["equivalence"][str(threads)] = dict(
                    passed=True,
                    cases=checked,
                    exhaustive_single_slots=32,
                    exhaustive_ordered_pairs=992,
                    all_four_row_active_subset_permutations_per_slot_order=65,
                    inactive_zeros_and_nan_inf_signed_zero_bits_exact=True,
                )
                report["timings"][str(threads)] = {
                    name: measure(shards, slots, active, vocab_size, args.repeats)
                    for name, slots, active in (
                        ("B1_slot31", [31], [0]),
                        ("B2_reversed_noncontiguous", [31, 0], [0, 1]),
                        ("B4_two_active_reordered", [31, 7, 0, 19], [3, 1]),
                        ("B2_all_inactive", [31, 0], []),
                    )
                }
            assert all(torch.equal(shard.view(torch.int16), bits) for shard, bits in zip(shards, original_bits))
        report["materialized_concat_plus_float_bytes"] = {
            "baseline_32_rows": 32 * vocab_size * (2 + 4),
            "selected_B1": vocab_size * (2 + 4),
            "selected_B2": 2 * vocab_size * (2 + 4),
            "extra_selected_shard_copy_B1": vocab_size * 2,
            "extra_selected_shard_copy_B2": 2 * vocab_size * 2,
            "excludes_common_result_and_original_shards": True,
        }
        report["recommendation"] = {
            "hypothesis": "Avoidable CPU padded-row concatenation/casting is confirmed in this isolated reconstruction model.",
            "minimal_change": "Preserve device read/host handle lifecycle. Convert each existing rank shard to CPU as before, select output.slots[row] for active_rows in submitted order, concatenate selected shards in the same rank/vocabulary order, then float-cast only those rows and scatter into the existing zero-filled [B,1,V] output. Keep token-output branch unchanged.",
            "required_integration_check": "Verify rank order and host shard views against the existing TTNN composer on actual output handles; preserve immutable handle metadata, inactive zeros and dtype. This CPU experiment does not verify those TTNN operations.",
            "p2_disposition": "Observed CPU savings do not establish that P2 ratio1.205 reaches1.15. Device read/TTNN conversion/sampling and end-to-end impact are excluded; rerun the actual gate after a minimal validated implementation.",
        }
        report["inputs_unchanged"] = True
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["ttnn_imported"] = any(name == "ttnn" or name.startswith("ttnn.") for name in sys.modules)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps({"passed": report["passed"], "seconds": report["seconds"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
