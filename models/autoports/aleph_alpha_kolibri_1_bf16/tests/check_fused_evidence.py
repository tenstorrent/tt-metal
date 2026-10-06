# SPDX-License-Identifier: Apache-2.0
"""Check stage-owned evidence and its relationship to the delivered decoder."""

import ast
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"
DECODER_HASH = hashlib.sha256((ROOT / "tt/fused_decoder.py").read_bytes()).hexdigest()


def read(name, current=True):
    data = json.loads((DOC / name).read_text())
    if current:
        assert data["provenance"]["source_sha256"]["tt/fused_decoder.py"] == DECODER_HASH, name
    return data


def check_rows(data):
    rows = data.get("rows", data.get("pcc", []))
    assert rows
    assert all(row["pcc"] >= 0.995 for row in rows)
    return min(row["pcc"] for row in rows)


def main():
    tree = ast.parse((ROOT / "tt/fused_decoder.py").read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert klass.name == "FusedDecoder" and ast.unparse(klass.bases[0]) == "LightweightModule"
    assert "functional_decoder" not in (ROOT / "tt/fused_decoder.py").read_text()
    # A snapshot must remain exact even if hooks later inspect evidence files.
    for snapshot in (DOC / "source_snapshots").glob("*.py.txt"):
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == snapshot.name.removesuffix(".py.txt")
    if (DOC / "artifact_manifest.json").exists():
        for row in json.loads((DOC / "artifact_manifest.json").read_text()):
            data = gzip.decompress((DOC / row["archive"]).read_bytes())
            assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
            if row["path"].startswith("source_snapshots/"):
                assert row["sha256"] == Path(row["path"]).name.removesuffix(".py.txt")
    contract = json.loads((ROOT / "doc/context_contract.json").read_text())
    assert contract["supported_context"] == 1048576 and contract["capability_reduction"] is None
    comparison = json.loads((DOC / "candidate_comparison.json").read_text())
    assert comparison["status"] == "pass"
    summaries = []
    for layer in (0, 4):
        watcher = read(f"watcher_{layer}.json")
        assert watcher["runtime_audit"] == "clean" and watcher["deterministic"]
        assert watcher["provenance"]["decoder_class"].endswith("tt.fused_decoder.FusedDecoder")
        env = watcher["provenance"]["environment"]
        assert env["TT_METAL_WATCHER"] and not env["TT_METAL_DEVICE_PROFILER"]
        trace = watcher["trace"]
        assert trace["allocation_tracking"] and not trace["skip_program_cache"]
        assert trace["captures"] == 1 and trace["replays"] == 33 and trace["request_releases"] == 0
        minimum = check_rows(watcher)
        synthetic = read(f"synthetic_{layer}.json")
        minimum = min(minimum, check_rows(synthetic))
        assert not synthetic["real_weights"] and synthetic["deterministic"]
        for batch in (3, 17, 32):
            data = read(f"batch_{batch}_{layer}.json")
            vals = [data["prefill_pcc"], data["decode_pcc"], data["changed_batch_pcc"]]
            vals += data["changed_batch_per_row_pcc"]
            assert min(vals) >= 0.995 and data["deterministic"] and data["real_weights"]
            assert data["allocation_tracking"] and not data["skip_program_cache"]
            minimum = min(minimum, *vals)
        context = read(f"context_edges_{layer}.json")
        assert (
            context["capacity"] == context["tested_decode_context"] == context["tested_prefill_end_position"] == 1048576
        )
        assert context["trace_captures"] == 1 and context["trace_request_releases"] == 0
        assert context["allocation_tracking"] and not context["skip_program_cache"]
        minimum = min(minimum, check_rows(context))
        final = read(f"final/profile_{layer}.json")
        assert not final["fusions"]
        assert final["provenance"]["environment"]["FUSION_IMPL"] is None
        assert final["provenance"]["decoder_class"].endswith("tt.fused_decoder.FusedDecoder")
        compared = next(row for row in comparison["comparisons"] if row["layer"] == layer)
        assert compared["final_ms"] == final["host_elapsed_ms"]["decode"] < compared["candidate_ms"]
        check_rows(final)
        assert min(final["unfused_pcc"].values()) >= 0.995
        before = read(f"baseline/reference/profile_{layer}.json")
        check_rows(before)
        assert final["host_elapsed_ms"]["decode"] < before["host_elapsed_ms"]["decode"]
        for variant in ("", "baseline/"):
            for mode in ("prefill", "decode"):
                for suffix in ("csv", "txt"):
                    path = DOC / f"{variant}tracy/layer_{layer}/{mode}_perf_report.{suffix}"
                    assert path.exists() or Path(str(path) + ".gz").exists(), path
        summaries.append(
            dict(
                layer=layer,
                minimum_pcc=minimum,
                final_ms=final["host_elapsed_ms"],
                baseline_ms=before["host_elapsed_ms"],
            )
        )
    result = dict(status="pass", decoder_sha256=DECODER_HASH, layers=summaries)
    (DOC / "evidence_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
