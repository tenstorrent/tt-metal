#!/usr/bin/env python3
"""Host-only regression tests for exhaustive-campaign fail-closed gates."""

from __future__ import annotations

import json
import os
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
import ulp_admission  # noqa: E402


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
        selected_record = stream_resume.cache_record(
            script, args, "sem-node", 0, 16, "selected",
            compiler_options="-mselected", golden="",
            runner_temp=str(root / "selected-host-path"), staged_variant="c" * 64,
        )
        baseline_record = stream_resume.cache_record(
            script, args, "sem-node", 0, 16, "sem",
            compiler_options="-mbaseline", golden="op",
        )
        assert selected_record["compiler_options"] == "-mselected"
        assert selected_record["staged_variant"] == "c" * 64
        assert selected_record["runner_temp"].endswith("selected-host-path")
        assert baseline_record["compiler_options"] == "-mbaseline"
        assert selected_record != baseline_record
        output = root / "band.txt"
        metadata = root / "band.txt.provenance.json"
        output.write_text("output_sha256=" + "a" * 64 + "\n")
        corr = root / "band.txt.corr"
        corr.write_text("SFPU_CORRECTNESS,fixture=1\n")
        try:
            stream_resume.require_matching_cache(output, metadata, record)
            raise AssertionError("legacy cache was accepted")
        except RuntimeError as error:
            assert "legacy cache" in str(error)
        stream_resume.write_cache_record(metadata, record, output)
        assert stream_resume.require_matching_cache(output, metadata, record)
        output.write_text("output_sha256=" + "b" * 64 + "\n")
        try:
            stream_resume.require_matching_cache(output, metadata, record)
            raise AssertionError("tampered cached output was accepted")
        except RuntimeError as error:
            assert "output digest mismatch" in str(error)
        output.write_text("output_sha256=" + "a" * 64 + "\n")
        corr.write_text("SFPU_CORRECTNESS,tampered=1\n")
        try:
            stream_resume.require_matching_cache(output, metadata, record)
            raise AssertionError("tampered cached correctness was accepted")
        except RuntimeError as error:
            assert "correctness digest mismatch" in str(error)
        corr.write_text("SFPU_CORRECTNESS,fixture=1\n")
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
        assert "LOCAL_SEM_ABSOLUTE=PASS" in text
        assert "LOCAL_HAND_ORACLE_COMPLETE=FAIL" in text
        assert "NOT-ABSOLUTE-ULP-CERTIFIED" in text

        binary_legs = {"sem": binary._new_leg(), "hand": binary._new_leg()}
        binary_legs["sem"].update(checked=True, joints=10)
        binary_legs["hand"].update(checked=True, joints=9)
        assert not binary.write_correctness_ledger(
            out, "binary", "BIT-EXACT-ALL-INPUTS", binary_legs, 10
        )


def test_per_class_ulp_admission() -> None:
    candidate = {"ordinary": (100, 5.0), "special": (2, 5.0)}
    hand = {"ordinary": (100, 5.0), "special": (2, 100.0)}
    assert ulp_admission.candidate_not_worse(candidate, hand)[0]

    # A global max comparison would pass 5 <= 100, but the candidate is worse
    # in the ordinary class and must therefore be refused.
    regressed = {"ordinary": (100, 6.0), "special": (2, 5.0)}
    ok, reason = ulp_admission.candidate_not_worse(regressed, hand)
    assert not ok and reason == "candidate-ulp-regression-ordinary"
    special_regression = {
        "in_domain_finite_normal": (100, 5.0),
        "pos_zero_input": (1, 1.0),
    }
    special_hand = {
        "in_domain_finite_normal": (100, 5.0),
        "pos_zero_input": (1, 0.0),
    }
    ok, reason = ulp_admission.candidate_not_worse(special_regression, special_hand)
    assert not ok and reason == "candidate-ulp-regression-pos_zero_input"
    assert not ulp_admission.candidate_not_worse(
        candidate, {"ordinary": (99, 5.0), "special": (2, 100.0)}
    )[0]
    assert not ulp_admission.candidate_not_worse({}, hand)[0]
    for invalid in (float("nan"), float("inf"), -1.0, 1.5):
        ok, reason = ulp_admission.candidate_not_worse(
            {"ordinary": (100, invalid)}, {"ordinary": (100, 5.0)}
        )
        assert not ok and reason == "invalid-class-metric-ordinary"
    encoded = ulp_admission.format_class_ulp(
        {"ordinary": (100, 65535.0), "special": (2, 0.0)}
    )
    assert ulp_admission.parse_class_ulp(encoded) == {
        "ordinary": (100, 65535.0),
        "special": (2, 0.0),
    }
    for malformed in ("ordinary:1:1.5", "ordinary:0:1", "ordinary:1:nan"):
        try:
            ulp_admission.parse_class_ulp(malformed)
            raise AssertionError(f"malformed class ULP accepted: {malformed}")
        except ValueError:
            pass

    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        legs = {"sem": fp32._new_leg(), "hand": fp32._new_leg()}
        for leg in legs.values():
            leg.update(checked=True, patterns=102, n_out=0)
        legs["sem"]["class_ulp"] = candidate
        legs["hand"]["class_ulp"] = hand
        assert fp32.write_correctness_ledger(
            out, "op", "DIVERGENT", legs, 102
        )
        verdict = (out / "op-CORRECTNESS-VERDICT.txt").read_text()
        assert "LOCAL_SEM_ABSOLUTE=PASS" in verdict
        assert "LOCAL_ULP_COMPARISON=PASS" in verdict

        legs["sem"]["class_ulp"] = regressed
        assert fp32.write_correctness_ledger(
            out, "op", "DIVERGENT", legs, 102
        )
        verdict = (out / "op-CORRECTNESS-VERDICT.txt").read_text()
        assert "LOCAL_SEM_ABSOLUTE=PASS" in verdict
        assert "LOCAL_ULP_COMPARISON=FAIL" in verdict


def test_correctness_sidecar_identity_and_class_data() -> None:
    args = SimpleNamespace(golden="exp")
    good = {
        "op": "exp",
        "leg": "sem",
        "patterns": "16",
        "max_bf16_ulp": "2",
        "n_out_of_tol": "0",
        "within_contract": "True",
        "class_ulp": "in_domain_finite_normal:16:2",
    }
    fp32.validate_corr(good, args, "sem", 16)
    unchecked = {
        "op": "exp", "leg": "sem", "status": "UNCHECKED",
        "patterns": "16", "reason": "no registered oracle",
    }
    fp32.validate_corr(unchecked, args, "sem", 16)
    for changed, message in (
        (dict(good, op="other"), "identity mismatch"),
        (dict(good, patterns="15"), "coverage mismatch"),
        (
            dict(good, class_ulp="in_domain_finite_normal:15:2"),
            "class coverage mismatch",
        ),
        (dict(good, class_ulp=""), "invalid golden sidecar"),
        (
            {key: value for key, value in good.items() if key != "n_out_of_tol"},
            "missing n_out_of_tol",
        ),
        (dict(good, n_out_of_tol="-1"), "invalid golden sidecar counts"),
        (dict(good, n_out_of_tol="17", within_contract="False"), "invalid golden sidecar counts"),
        (dict(good, max_bf16_ulp="nan"), "invalid golden sidecar max_bf16_ulp"),
        (dict(good, max_bf16_ulp="65536"), "invalid golden sidecar max_bf16_ulp"),
        (dict(good, max_bf16_ulp="1"), "max/class mismatch"),
        (dict(good, class_ulp="bogus:16:2"), "class vocabulary"),
        (dict(good, within_contract="False"), "invalid golden sidecar within_contract"),
    ):
        try:
            fp32.validate_corr(changed, args, "sem", 16)
            raise AssertionError(f"bad sidecar accepted: {changed}")
        except RuntimeError as error:
            assert message in str(error)
    for changed in (
        dict(unchecked, status="SKIPPED"),
        {key: value for key, value in unchecked.items() if key != "reason"},
        dict(unchecked, reason="   "),
        dict(unchecked, patterns="15"),
        dict(unchecked, n_out_of_tol="0"),
    ):
        try:
            fp32.validate_corr(changed, args, "sem", 16)
            raise AssertionError(f"bad unchecked sidecar accepted: {changed}")
        except RuntimeError:
            pass

    binary_args = SimpleNamespace(golden="binarypow")
    binary_good = {
        "op": "binarypow",
        "leg": "hand",
        "joints": "16",
        "max_bf16_ulp": "4",
        "n_out_of_tol": "1",
        "within_contract": "False",
        "class_ulp": "pos_normal_base_normal_exp:15:4|base_nan:1:0",
    }
    binary.validate_corr(binary_good, binary_args, "hand", 16)
    binary_unchecked = {
        "op": "binarypow", "leg": "hand", "status": "UNCHECKED",
        "joints": "16", "reason": "no registered oracle",
    }
    binary.validate_corr(binary_unchecked, binary_args, "hand", 16)
    for changed in (
        {key: value for key, value in binary_good.items() if key != "n_out_of_tol"},
        dict(binary_good, n_out_of_tol="-1"),
        dict(binary_good, n_out_of_tol="17"),
        dict(binary_good, max_bf16_ulp="inf"),
        dict(binary_good, max_bf16_ulp="3"),
        dict(binary_good, class_ulp="bogus:16:4"),
    ):
        try:
            binary.validate_corr(changed, binary_args, "hand", 16)
            raise AssertionError(f"bad binary sidecar accepted: {changed}")
        except RuntimeError:
            pass
    for changed in (
        dict(binary_unchecked, status="SKIPPED"),
        {key: value for key, value in binary_unchecked.items() if key != "reason"},
        dict(binary_unchecked, reason=""),
        dict(binary_unchecked, joints="15"),
        dict(binary_unchecked, class_ulp="base_nan:16:0"),
    ):
        try:
            binary.validate_corr(changed, binary_args, "hand", 16)
            raise AssertionError(f"bad unchecked binary sidecar accepted: {changed}")
        except RuntimeError:
            pass

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for parser in (fp32.parse_corr, binary.parse_corr):
            duplicate = root / f"duplicate-{parser.__module__}.corr"
            duplicate.write_text(
                "SFPU_CORRECTNESS,op=exp,leg=sem,n_out_of_tol=0,n_out_of_tol=1\n"
            )
            malformed = root / f"malformed-{parser.__module__}.corr"
            malformed.write_text("SFPU_CORRECTNESS,op=exp,broken\n")
            multiline = root / f"multiline-{parser.__module__}.corr"
            multiline.write_text("SFPU_CORRECTNESS,op=exp\nextra=record\n")
            for path in (duplicate, malformed, multiline):
                try:
                    parser(path)
                    raise AssertionError(f"malformed sidecar accepted: {path}")
                except RuntimeError:
                    pass


def _write_slice(
    root: Path,
    chip: int,
    op: str,
    verdict: str,
    numeric=True,
    start: int | None = None,
    total: int = 5,
) -> None:
    out = root / f"slice-{chip}"
    out.mkdir(exist_ok=True)
    if start is None:
        start = chip * total
    (out / f"{op}-VERDICT.txt").write_text(
        f"OP={op} VERDICT={verdict} start={start} total={total} "
        f"covered={total} witness_bands=[]\n"
    )
    if numeric:
        (out / f"{op}-CORRECTNESS-VERDICT.txt").write_text(
            f"OP={op} LOCAL_SEM_ABSOLUTE=PASS "
            "LOCAL_HAND_ORACLE_COMPLETE=PASS "
            "CAMPAIGN_ADMISSION=DEFERRED_GLOBAL\n"
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
        assert not passed and "numeric_gate=LOCAL_SEM_FAIL" in summary
        _write = out / "slice-1/op-CORRECTNESS-VERDICT.txt"
        _write.write_text(
            "OP=op LOCAL_SEM_ABSOLUTE=PASSIVE LOCAL_HAND_ORACLE_COMPLETE=PASS\n"
        )
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert not passed and "numeric_gate=LOCAL_SEM_FAIL" in summary
        _write.write_text(
            "OP=op LOCAL_SEM_ABSOLUTE=PASS LOCAL_HAND_ORACLE_COMPLETE=PASS\n"
        )
        (out / "slice-1/op-VERDICT.txt").write_text(
            "OP=op VERDICT=DIVERGENT start=5 total=5 covered=5 witness_bands=[1]\n"
        )
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert not passed and "VERDICT=DIVERGENT" in summary
        (out / "slice-1/op-VERDICT.txt").unlink()
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, True)
        assert not passed and "VERDICT=INCOMPLETE" in summary

        # Pre-range legacy verdicts are not resumable campaign evidence.
        (out / "slice-1/op-VERDICT.txt").write_text(
            "OP=op VERDICT=BIT-EXACT-ALL-INPUTS covered=5 witness_bands=[]\n"
        )
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert not passed and "VERDICT=INCOMPLETE" in summary

        # Equal-sized duplicate ranges have the right total count but overlap;
        # exact start/total checking must refuse them.
        _write_slice(out, 0, "op", "BIT-EXACT-ALL-INPUTS", start=0)
        _write_slice(out, 1, "op", "BIT-EXACT-ALL-INPUTS", start=0)
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert not passed and "VERDICT=INCOMPLETE" in summary

        # A verdict file whose payload names another operation is not evidence
        # for the requested op, even when its filename and ranges look right.
        (out / "slice-1/op-VERDICT.txt").write_text(
            "OP=other VERDICT=BIT-EXACT-ALL-INPUTS start=5 total=5 "
            "covered=5 witness_bands=[]\n"
        )
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert not passed and "VERDICT=INCOMPLETE" in summary

        # Token parsing is exact: PASSIVE must not satisfy the local semantic gate.
        _write_slice(out, 1, "op", "BIT-EXACT-ALL-INPUTS")
        (out / "slice-1/op-CORRECTNESS-VERDICT.txt").write_text(
            "OP=op LOCAL_SEM_ABSOLUTE=PASSIVE LOCAL_HAND_ORACLE_COMPLETE=PASS\n"
        )
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert not passed and "numeric_gate=LOCAL_SEM_FAIL" in summary

        summary, passed = galaxy_combine.combine(out, 2, 10, "op", False, False)
        assert passed and "numeric_gate=NOT_REQUESTED" in summary


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


def test_formal_refuses_identical_compiled_objects() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        identities = iter(
            (
                (out / "sem.trace", {"path": "sem.elf", "text_sha256": "a" * 64}, None),
                (out / "hand.trace", {"path": "hand.elf", "text_sha256": "a" * 64}, None),
            )
        )
        original = prove_all._run_leg
        prove_all._run_leg = lambda *_args, **_kwargs: next(identities)
        try:
            result = prove_all.run_formal(
                "op",
                {"sem_node": "sem", "hand_node": "hand", "reason": "fixture"},
                out,
                "-mfixture",
                1,
            )
        finally:
            prove_all._run_leg = original
        assert result["class"] == "UNSWEPT"
        assert result["verdict"] == "REFUSED-IDENTITY"
        assert result["sem_elf"]["text_sha256"] == result["hand_elf"]["text_sha256"]


def test_formal_row_wrapper_is_thin() -> None:
    wrapper = HERE / "formal_equiv_row.sh"
    source = wrapper.read_text()
    assert "formal_campaign.py" in source
    assert "EXPECTED_SIM_SHA" not in source
    assert "compgen -e" not in source
    assert "prove_all" not in source


if __name__ == "__main__":
    test_resume_provenance()
    test_partial_tolerance_coverage_refuses()
    test_per_class_ulp_admission()
    test_correctness_sidecar_identity_and_class_data()
    test_galaxy_combiner_refuses_nonpass()
    test_identity_refusal_exits_nonzero()
    test_failed_dispatch_output_is_not_accepted()
    test_proof_cache_is_provenance_bound()
    test_formal_refuses_identical_compiled_objects()
    test_formal_row_wrapper_is_thin()
    print("SELFTEST: campaign fail-closed gates PASS")
