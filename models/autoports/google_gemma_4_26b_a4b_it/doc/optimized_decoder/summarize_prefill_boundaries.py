# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the completed matched boundary campaign from saved artifacts only."""

import hashlib
import json
import math
import statistics
from functools import lru_cache
from pathlib import Path

DOC = Path(__file__).resolve().parent
MODEL = DOC.parent.parent
RUNTIME = "169c0d97d7d0e9f35d97633f133305f1088b987ef625693d3faa100d25c3e67b"


@lru_cache
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    return dict(path=str(path), sha256=digest(path))


def read(path):
    return json.loads(path.read_text())


def option(command, key):
    return command[command.index(key) + 1]


def checked(row):
    assert row["passed"] is True and math.isfinite(row["pcc"]) and 0.995 <= row["pcc"] <= 1.000000001


def main():
    assert digest(MODEL / "tt/optimized_decoder.py") == RUNTIME
    journal_path = DOC / "prefill_boundary_commands.json"
    journal = read(journal_path)
    assert len(journal) == 16 and all(entry["returncode"] == 0 for entry in journal)
    results = {}
    for entry in journal:
        command = entry["command"]
        layer, length = int(option(command, "--layer")), int(option(command, "--length"))
        module = option(command, "-m")
        optimized = module.endswith(".run_optimized_contract")
        decoder = "optimized" if optimized else "fused"
        assert (
            module
            == f"models.autoports.google_gemma_4_26b_a4b_it.tests.{'run_optimized_contract' if optimized else 'run_decoder'}"
        )
        assert option(command, "--contract" if optimized else "--decoder") == ("run_decoder" if optimized else "fused")
        assert layer in (0, 5) and length in (65, 1023, 1024, 1025)
        assert option(command, "--steps") == "1"
        assert all(flag in command for flag in ("--real", "--decode", "--prefill-timing", "--verify-program-cache"))
        assert "--profile" not in command
        path, fixture = Path(option(command, "--output")), Path(option(command, "--input-fixture"))
        report = read(path)
        assert report["decoder"] == decoder and report["length"] == length and report["real_weights"] is True
        assert report["layer_type"] == ("sliding_attention" if layer == 0 else "full_attention")
        assert report["prefix_length"] == 0 and report["runtime_prefill_audit"] == "clean"
        assert report["program_cache_miss_guard"] is True and report["prefill_cache_entries"] > 0
        checked(report)
        decode = report["decode"]
        assert decode["steps"] == 1 and decode["positions"] == [length, length]
        assert decode["traced"] and decode["repeated_equal"] and decode["runtime_decode_audit"] == "clean"
        assert [row["position"] for row in decode["checks"]] == [length, length]
        for row in decode["checks"]:
            checked(row)
        checked(decode)
        assert decode["min_pcc"] == min(row["pcc"] for row in decode["checks"])
        assert Path(report["input_fixture"]).resolve() == fixture.resolve()
        assert report["input_fixture_sha256"] == digest(fixture)
        metadata = report["input_source"]
        assert metadata["source"]["kind"] == "recorded_real_text_hf_layer_inputs"
        assert metadata["layer"] == layer and metadata["length"] == length and metadata["steps"] == 1
        assert metadata["slice_source_sha256"] == digest(Path(metadata["slice_source"]))
        if optimized:
            assert report["runtime_sha256"] == RUNTIME and report["functional_fallback"] == "forbidden"
            assert report["contract"] == "run_decoder" and "candidate" not in report
            qkv = report["precision_policy"]["prefill_qkv_projection"]
            assert qkv["backend"] == "minimal_matmul" and qkv["k_block"] == (8 if layer == 0 else 16)
        physical = []
        for start in range(0, length, 1024):
            valid = min(1024, length - start)
            physical.append(1024 if layer == 0 and start > 0 and valid < 1024 else (valid + 31) // 32 * 32)
        assert entry["logical_length"] == length and entry["chunk_size"] == 1024
        assert entry["chunk_count"] == len(physical) and entry["physical_rows"] == physical
        samples = report["warmed_prefill_host_us"]
        assert len(samples) == 3 and all(math.isfinite(value) and value > 0 for value in samples)
        key = (layer, length, decoder)
        assert key not in results
        results[key] = dict(
            decoder=decoder,
            layer=layer,
            logical_length=length,
            physical_rows=physical,
            samples_host_us=samples,
            median_host_us=statistics.median(samples),
            minimum_host_us=min(samples),
            maximum_host_us=max(samples),
            prefill_hf_pcc=report["pcc"],
            minimum_decode_hf_pcc=decode["min_pcc"],
            program_cache_miss_guard=True,
            prefill_cache_entries=report["prefill_cache_entries"],
            runtime_sha256=report.get("runtime_sha256"),
            report=source(path),
            fixture=source(fixture),
            recorded_slice_source=source(Path(metadata["slice_source"])),
        )
    pairs = []
    for layer in (0, 5):
        for length in (65, 1023, 1024, 1025):
            fused, optimized = (results[(layer, length, decoder)] for decoder in ("fused", "optimized"))
            assert fused["fixture"] == optimized["fixture"] and fused["physical_rows"] == optimized["physical_rows"]
            pairs.append(
                dict(
                    layer=layer,
                    layer_type="sliding_attention" if layer == 0 else "full_attention",
                    logical_length=length,
                    chunk_count=len(optimized["physical_rows"]),
                    physical_rows=optimized["physical_rows"],
                    fused=fused,
                    optimized=optimized,
                    fused_to_optimized_host_speedup=fused["median_host_us"] / optimized["median_host_us"],
                    host_time_saved_us=fused["median_host_us"] - optimized["median_host_us"],
                )
            )
    result = dict(
        status="passed",
        runtime_sha256=RUNTIME,
        command_count=16,
        pairs=pairs,
        sources=[
            source(path)
            for path in (
                journal_path,
                DOC / "run_prefill_boundaries.py",
                Path(__file__),
                MODEL / "tt/optimized_decoder.py",
                MODEL / "tt/fused_decoder.py",
                MODEL / "tt/functional_decoder.py",
                MODEL / "tests/run_decoder.py",
            )
        ],
        timing_scope="Median of three warmed synchronous whole-prefill host calls; host dispatch and final synchronization included; output deallocation and initial synchronization precede each timed interval. No device profiling or decode latency is inferred.",
        cache_scope="The warmup populated program cache and a preceding whole-prefill rerun disallowed misses. Timed samples follow that check; their misses are not separately blocked or counted.",
        physical_scope="Source-derived layer-call physical rows, cross-checked with the command journal; not measured native trace shapes. Both decoders tile-pad each chunk; a short noninitial sliding chunk expands to 1024 rows.",
        comparison_scope="Whole selected optimized decoder versus fused baseline on identical real input bytes and logical/physical geometry; not an isolated minimal-matmul speedup or full-model result.",
        provenance_limit="Optimized reports record the frozen runtime hash. Fused reports do not record a per-run source hash; the current fused/functional source hashes above are audit-time provenance only.",
        correctness_scope="Every prefill aggregate and traced one-step HF check passes PCC .995 with repeat equality and clean device-only audits. No direct fused-to-optimized tensor comparison was recorded in this boundary campaign.",
    )
    (DOC / "prefill_boundary_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    text = [
        "All 16 matched boundary commands passed at PCC ≥ .995 for real recorded layer inputs. The optimized reports pin runtime `"
        + RUNTIME
        + "`. Both decoder kinds pass prefill, traced one-step decode, repeated-output equality, device-only audits, and the program-cache guard. [Raw summary and hashes](prefill_boundary_summary.json) retain all timing samples, report/input hashes, and cache-entry counts.",
        "",
        "| Attention | Logical rows | Physical rows per chunk | Fused host median, µs | Optimized host median, µs | Ratio | Optimized prefill PCC | Optimized decode PCC |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in pairs:
        fused, optimized = pair["fused"], pair["optimized"]
        text.append(
            f"| {'Sliding' if pair['layer'] == 0 else 'Full'} | {pair['logical_length']} | {pair['physical_rows']} | {fused['median_host_us']:,.3f} | {optimized['median_host_us']:,.3f} | {pair['fused_to_optimized_host_speedup']:.3f}× | {optimized['prefill_hf_pcc']:.9f} | {optimized['minimum_decode_hf_pcc']:.9f} |"
        )
    text += [
        "",
        result["timing_scope"]
        + " Each median uses all three recorded samples; no sample was discarded. See [runner](../../tests/run_decoder.py), lines 157–199. The correctness readback follows the final timed call.",
        "",
        result["cache_scope"],
        "",
        "The chunk boundary explains different work at length 1025: sliding runs `[1024,1024]`, while full runs `[1024,32]`. A 65-token input is one 96-row physical chunk; 1023 and 1024 are each one 1024-row chunk. These are source-derived layer-call shapes cross-checked with journal annotations, not new device-profile measurements. Both [optimized prefill](../../tt/optimized_decoder.py), lines 684–705, and [inherited fused prefill](../../tt/functional_decoder.py), lines 158–185, apply this policy. The padded tail is processed and then trimmed to valid output rows.",
        "",
        result["comparison_scope"]
        + " The optimized implementation is faster in every measured pair, but this campaign does not assign the total change to one kernel. With three samples per case, the medians describe these runs rather than a statistical confidence bound.",
        "",
        "The [boundary driver](run_prefill_boundaries.py) slices the first logical rows from each recorded 4096-token fixture and uses the next recorded prefill activation for the single decode step. All eight boundary fixture files and both original fixture files were rehashed, and paired reports reference identical bytes. "
        + result["provenance_limit"],
        "",
        result["correctness_scope"]
        + " Broader final-runtime gates remain in [the v5 validation summary](validated_v5_validation_summary.json).",
        "",
        "Reproduce this CPU-only summary from the checkout root:",
        "",
        "```sh",
        "python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/summarize_prefill_boundaries.py",
        "```",
    ]
    (DOC / "prefill_boundary_summary.md").write_text("\n".join(text) + "\n")
    print(json.dumps(dict(status="passed", commands=16, pairs=8), indent=2))


if __name__ == "__main__":
    main()
