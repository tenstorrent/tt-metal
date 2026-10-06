# SPDX-License-Identifier: Apache-2.0
"""Require current optimized-path correctness, capability and performance evidence."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_decoder"
SOURCE = ROOT / "tt/optimized_decoder.py"
HASH = hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def read(name, current=True):
    data = json.loads((DOC / name).read_text())
    if current:
        assert data["provenance"]["source_sha256"]["tt/optimized_decoder.py"] == HASH, name
        runner = Path(data["provenance"]["argv"][0]).name
        runner_path = ROOT / "tests" / runner
        if runner_path.is_file():
            key = "tests/" + runner
            assert (
                data["provenance"]["source_sha256"][key] == hashlib.sha256(runner_path.read_bytes()).hexdigest()
            ), name
    return data


def rows(data):
    values = data.get("rows", data.get("pcc", []))
    assert values and min(row["pcc"] for row in values) >= 0.995
    return min(row["pcc"] for row in values)


def main():
    tree = ast.parse(SOURCE.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OptimizedDecoder")
    assert ast.unparse(klass.bases[0]) == "LightweightModule"
    assert "functional_decoder" not in SOURCE.read_text() and "fused_decoder" not in SOURCE.read_text()
    contract = json.loads((ROOT / "doc/context_contract.json").read_text())["optimized_decoder_validation"]
    assert contract["status"] == "passed" and contract["supported_context"] == 1048576
    assert contract["capability_reduction"] is None
    summaries = []
    for layer in (0, 4):
        watcher = read(f"watcher_{layer}.json")
        minimum = rows(watcher)
        assert watcher["runtime_audit"] == "clean" and watcher["deterministic"] and watcher["real_weights"]
        assert watcher["provenance"]["decoder_class"].endswith("tt.optimized_decoder.OptimizedDecoder")
        env = watcher["provenance"]["environment"]
        assert env["TT_METAL_WATCHER"] == "10" and not env["TT_METAL_DEVICE_PROFILER"]
        assert env["OPT_REAL_INPUT"] == "1"
        trace = watcher["trace"]
        assert trace["allocation_tracking"] and not trace["skip_program_cache"]
        assert (trace["captures"], trace["replays"], trace["request_releases"]) == (1, 36, 0)
        assert len(watcher["cache_ownership_checks"]) == 4
        assert all(row["exact"] for row in watcher["cache_ownership_checks"])
        for batch in (3, 17, 32):
            data = read(f"batch_{batch}_{layer}.json")
            metrics = [data[key] for key in ("prefill_pcc", "decode_pcc", "eager_decode_pcc", "changed_batch_pcc")]
            for key in (
                "prefill_per_row_pcc",
                "decode_per_row_pcc",
                "eager_decode_per_row_pcc",
                "changed_batch_per_row_pcc",
            ):
                assert len(data[key]) == batch
                metrics.extend(data[key])
            assert min(metrics) >= 0.995 and data["deterministic"] and data["real_weights"]
            assert data["allocation_tracking"] and not data["skip_program_cache"]
            minimum = min(minimum, *metrics)
        bf16 = read(f"bf16_cache_{layer}.json")
        assert min(bf16["prefill_pcc"], bf16["decode_pcc"], bf16["changed_batch_pcc"]) >= 0.995
        context = read(f"context_edges_{layer}.json")
        assert (
            context["capacity"] == context["tested_decode_context"] == context["tested_prefill_end_position"] == 1048576
        )
        assert context["allocation_tracking"] and not context["skip_program_cache"]
        minimum = min(minimum, rows(context))
        assert {row["logical_length"] for row in context["rows"] if row["case"] == "public_precision_boundary"} == {
            65536,
            65664,
        }
        bf16_context = read(f"bf16_context_edges_{layer}.json")
        assert bf16_context["tested_decode_context"] == bf16_context["tested_prefill_end_position"] == 1048576
        assert bf16_context["allocation_tracking"] and not bf16_context["skip_program_cache"]
        minimum = min(minimum, rows(bf16_context))
        final = read(f"final/profile_{layer}.json")
        assert final["provenance"]["decoder_class"].endswith("tt.optimized_decoder.OptimizedDecoder")
        env = final["provenance"]["environment"]
        assert not any(env[key] for key in ("OPT_PROJECTIONS", "OPT_LAYOUT", "OPT_PREFILL", "OPT_SPLIT", "OPT_RUNTIME"))
        assert env["OPT_POLICY"] in (None, "{}") and final["input_source"] == "recorded_checkpoint"
        before = read(f"baseline/real_fused/profile_{layer}.json", current=False)
        assert final["host_elapsed_ms"]["decode"] < before["host_elapsed_ms"]["decode"]
        rows(final)
        split = read(f"final_split_packed/profile_{layer}.json", current=False)
        assert final["host_elapsed_ms"]["decode"] <= split["host_elapsed_ms"]["decode"] * 1.02
        for candidate in DOC.glob(f"final_matched_split_*/profile_{layer}.json"):
            measured = json.loads(candidate.read_text())
            assert final["host_elapsed_ms"]["decode"] < measured["host_elapsed_ms"]["decode"], candidate
        packed = read(f"delivery_packed/profile_{layer}.json")
        rows(packed)
        assert final["host_elapsed_ms"]["decode"] <= packed["host_elapsed_ms"]["decode"] * 1.02
        candidates = list(DOC.glob(f"delivery_final_matched_split_*/profile_{layer}.json"))
        assert len(candidates) == 3
        for candidate in candidates:
            measured = read(str(candidate.relative_to(DOC)))
            rows(measured)
            assert measured["provenance"]["cache_dtype"] == packed["provenance"]["cache_dtype"]
            assert packed["host_elapsed_ms"]["decode"] < measured["host_elapsed_ms"]["decode"], candidate
            assert final["host_elapsed_ms"]["decode"] < measured["host_elapsed_ms"]["decode"], candidate
        for length in (17, 129, 777, 8193):
            public = read(f"public_final_{length}/profile_{layer}.json")
            assert public["public_api"] and public["prefill_tokens"] == length
            rows(public)
        profiled = read(f"tracy_final_{layer}/profile_{layer}.json")
        assert profiled["input_source"] == "recorded_checkpoint"
        rows(profiled)
        long_profiled = read(f"tracy_long_{layer}/profile_{layer}.json")
        assert long_profiled["public_api"] and long_profiled["prefill_tokens"] == 8193
        rows(long_profiled)
        for mode in ("prefill", "decode"):
            for suffix in ("csv", "txt"):
                path = DOC / f"tracy/layer_{layer}/{mode}_perf_report.{suffix}"
                assert path.is_file() or Path(str(path) + ".gz").is_file(), path
        summaries.append(
            dict(
                layer=layer,
                minimum_pcc=minimum,
                final_ms=final["host_elapsed_ms"],
                baseline_ms=before["host_elapsed_ms"],
            )
        )
    result = dict(status="pass", decoder_sha256=HASH, layers=summaries)
    (DOC / "evidence_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
