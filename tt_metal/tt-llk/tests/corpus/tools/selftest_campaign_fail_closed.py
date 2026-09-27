#!/usr/bin/env python3
"""Host-only regression tests for exhaustive-campaign fail-closed gates."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import binary_stream_sweep as binary  # noqa: E402
import fp32_stream_sweep as fp32  # noqa: E402
import galaxy_combine  # noqa: E402
import prove_all  # noqa: E402
import stream_resume  # noqa: E402


def test_resume_provenance() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = root / "streamer.py"
        script.write_text("# pinned streamer\n")
        idmap = root / "idmap.tsv"
        idmap.write_text("op\tsem\tsha1\thand\tsha2\n")
        args = SimpleNamespace(
            idmap=str(idmap),
            farm=str(root),
            venv=sys.executable,
            runner_temp=str(root),
            llk_home=str(root),
            chip="0",
            op="op",
            sem_node="sem-node",
            hand_node="hand-node",
            golden="op",
            tile_dim="256,256",
            band_bits=20,
            idmap_source="source.cpp",
        )
        record = stream_resume.cache_record(script, args, "sem-node", 0, 16, "sem")
        output = root / "band.txt"
        metadata = root / "band.txt.provenance.json"
        output.write_text("output_sha256=" + "a" * 64 + "\n")
        try:
            stream_resume.require_matching_cache(output, metadata, record)
            raise AssertionError("legacy cache was accepted")
        except RuntimeError as error:
            assert "legacy cache" in str(error)
        stream_resume.write_cache_record(metadata, record)
        assert stream_resume.require_matching_cache(output, metadata, record)
        changed = dict(record, count=17)
        try:
            stream_resume.require_matching_cache(output, metadata, changed)
            raise AssertionError("mismatched cache was accepted")
        except RuntimeError as error:
            assert "mismatch" in str(error)


def test_partial_tolerance_coverage_refuses() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        fp_legs = {"sem": fp32._new_leg(), "hand": fp32._new_leg()}
        fp_legs["sem"].update(checked=True, patterns=10)
        fp_legs["hand"].update(checked=True, patterns=9)
        assert not fp32.write_correctness_ledger(
            out, "fp", "BIT-EXACT-ALL-INPUTS", fp_legs, 10
        )
        text = (out / "fp-CORRECTNESS-VERDICT.txt").read_text()
        assert "NUMERIC_GATE=FAIL" in text and "NOT-ULP-CERTIFIED" in text

        binary_legs = {"sem": binary._new_leg(), "hand": binary._new_leg()}
        binary_legs["sem"].update(checked=True, joints=10)
        binary_legs["hand"].update(checked=True, joints=9)
        assert not binary.write_correctness_ledger(
            out, "binary", "BIT-EXACT-ALL-INPUTS", binary_legs, 10
        )


def _write_slice(root: Path, chip: int, op: str, verdict: str, numeric=True) -> None:
    out = root / f"slice-{chip}"
    out.mkdir(exist_ok=True)
    (out / f"{op}-VERDICT.txt").write_text(
        f"OP={op} VERDICT={verdict} covered=5 witness_bands=[]\n"
    )
    if numeric:
        (out / f"{op}-CORRECTNESS-VERDICT.txt").write_text(
            f"OP={op} NUMERIC_GATE=PASS CONTRACT=TOLERANCE-ONLY-NOT-ULP-CERTIFIED\n"
        )


def test_galaxy_combiner_refuses_nonpass() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        _write_slice(out, 0, "op", "BIT-EXACT-ALL-INPUTS")
        _write_slice(out, 1, "op", "BIT-EXACT-ALL-INPUTS")
        _, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert passed
        (out / "slice-1/op-CORRECTNESS-VERDICT.txt").unlink()
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert not passed and "numeric_gate=FAIL" in summary
        _write = out / "slice-1/op-CORRECTNESS-VERDICT.txt"
        _write.write_text("NUMERIC_GATE=PASS\n")
        (out / "slice-1/op-VERDICT.txt").write_text(
            "OP=op VERDICT=DIVERGENT covered=5 witness_bands=[1]\n"
        )
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert not passed and "VERDICT=DIVERGENT" in summary
        (out / "slice-1/op-VERDICT.txt").unlink()
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, True)
        assert not passed and "VERDICT=INCOMPLETE" in summary

        # Total coverage alone is insufficient: duplicated/overlapping slice
        # claims must not add up to a false full-space verdict.
        _write_slice(out, 1, "op", "BIT-EXACT-ALL-INPUTS")
        (out / "slice-0/op-VERDICT.txt").write_text(
            "OP=op VERDICT=BIT-EXACT-ALL-INPUTS covered=4 witness_bands=[]\n"
        )
        (out / "slice-1/op-VERDICT.txt").write_text(
            "OP=op VERDICT=BIT-EXACT-ALL-INPUTS covered=6 witness_bands=[]\n"
        )
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert not passed and "VERDICT=INCOMPLETE" in summary


def test_identity_refusal_exits_nonzero() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "test_node.py").write_text("# harness provenance\n")
        idmap = root / "idmap.tsv"
        idmap.write_text("different\tsem\tsha\thand\tsha\n")
        common = [
            "--op",
            "op",
            "--sem-node",
            "sem",
            "--hand-node",
            "hand",
            "--farm",
            str(root),
            "--venv",
            sys.executable,
            "--llk-home",
            str(root),
            "--runner-temp",
            str(root),
            "--out",
            str(root / "out"),
            "--idmap",
            str(idmap),
        ]
        for script in ("fp32_stream_sweep.py", "binary_stream_sweep.py"):
            result = subprocess.run([sys.executable, str(HERE / script), *common])
            assert result.returncode == 2, (script, result.returncode)


def test_failed_dispatch_output_is_not_accepted() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "test_node.py").write_text("# harness provenance\n")
        idmap = root / "idmap.tsv"
        idmap.write_text("op\tsem\tsha1\thand\tsha2\n")
        output = root / "band.txt"
        args = SimpleNamespace(
            idmap=str(idmap),
            op="op",
            sem_node="sem",
            hand_node="hand",
            golden="",
            tile_dim="256,256",
            band_bits=20,
            idmap_source="source.cpp",
            llk_home=str(root),
            runner_temp=str(root),
            chip="0",
            venv=sys.executable,
            farm=str(root),
            timeout=1,
        )
        original_run = fp32.subprocess.run
        calls = 0

        def fake_run(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            # Only the failed first process writes a superficially valid result.
            if calls == 1:
                output.write_text("output_sha256=" + "a" * 64 + "\n")
            return SimpleNamespace(returncode=1 if calls == 1 else 0)

        fp32.subprocess.run = fake_run
        try:
            try:
                fp32.run_band_leg(args, "sem", 0, 16, output, root / "band.log")
                raise AssertionError("failed-dispatch output was accepted")
            except RuntimeError as error:
                assert "produced no SHA" in str(error)
        finally:
            fp32.subprocess.run = original_run


def test_proof_cache_is_provenance_bound() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        prov = {"pin": "pin", "shas": {"compiler": "a"}}
        row = {"engine": "formal_equiv", "sem_node": "s", "hand_node": "h"}
        key = prove_all.verdict_cache_key(prov, "-mflags", "op", row)
        prove_all.save_verdict(
            out,
            "op",
            {"class": "SMT-PROVEN-ALL-INPUTS", "engine": "formal_equiv"},
            key,
        )
        assert prove_all.valid_cached(out, "op", key) is not None
        assert prove_all.valid_cached(out, "op", "different") is None
        prove_all.save_verdict(
            out,
            "failed",
            {"class": "UNSWEPT", "engine": "formal_equiv", "verdict": "PROVER-FAILED"},
            key,
        )
        assert prove_all.valid_cached(out, "failed", key) is None
        assert "failed" in prove_all.operational_failures(
            {"failed": {"class": "UNSWEPT", "verdict": "PROVER-FAILED"}}
        )
        path = prove_all.verdict_path(out, "legacy")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"class": "SMT-PROVEN-ALL-INPUTS", "engine": "formal_equiv"}))
        assert prove_all.valid_cached(out, "legacy", key) is None


if __name__ == "__main__":
    test_resume_provenance()
    test_partial_tolerance_coverage_refuses()
    test_galaxy_combiner_refuses_nonpass()
    test_identity_refusal_exits_nonzero()
    test_failed_dispatch_output_is_not_accepted()
    test_proof_cache_is_provenance_bound()
    print("SELFTEST: campaign fail-closed gates PASS")
