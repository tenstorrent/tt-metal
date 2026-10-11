# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Only qualify measured gains from identical sources, inputs and precision."""

import copy
import hashlib
import json
from argparse import Namespace

import pytest

from models.demos.qwen38_27b_qb2.demo import run_compact_followup as queue
from models.demos.qwen38_27b_qb2.demo.run_compact_gdn import compare_sweeps
from models.demos.qwen38_27b_qb2.tests.compact_gdn import COMBINED_GDN_POLICY, PADDING_GDN_POLICY, policy_pair
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, summarize
from models.demos.qwen38_27b_qb2.tests.unit.test_compact_gdn_queue import arms


def evidence(tmp_path, gain=1.2, *, combined=False, padding=False):
    baseline, candidate = policy_pair(combined=combined, padding=padding)
    source = tmp_path / "source"
    manifest = {}
    for name in (
        "tt/model.py",
        "config/precision.json",
        *(f"config/precision_{r}_bfp8_all.json" for r in (baseline, candidate)),
    ):
        path = source / queue.MODEL_PREFIX / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        manifest[queue.MODEL_PREFIX + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = arms()
    for name, recurrence in (("before", baseline), ("compact", candidate), ("after", baseline)):
        report = make_plan(batches=(16,), input_lengths=(32768, 16384))
        report.update(
            state="completed",
            cleanup_completed=True,
            source_sha256={
                k.removeprefix(queue.MODEL_PREFIX): v
                for k, v in manifest.items()
                if "/tt/" in k or k.endswith("config/precision.json")
            },
            precision={"decode_recurrence": recurrence, "kv_cache_dtype": "bfloat8_b"},
            configuration={
                "environment": {"QWEN_PRECISION_CONFIG": recurrence, "QWEN_PREFILL_MAX_BATCH_TOKENS": "32768"}
            },
        )
        report["source_sha256"]["effective_precision_override"] = manifest[
            queue.MODEL_PREFIX + f"config/precision_{recurrence}_bfp8_all.json"
        ]
        for cell, measured in zip(report["cells"], data[name]["cells"]):
            cell.update(measured, prompt_sha256="same")
            for sample in cell["samples"]:
                sample["trace_captures"] = 0
                if name == "compact":
                    sample["decode_s"] /= gain
            cell["warmup"] = copy.deepcopy(cell["samples"][0])
            cell["summary"] = summarize(
                cell["samples"], concurrency=16, input_tokens=cell["input_tokens"], output_tokens=128
            )
        data[name] = report
        directory = tmp_path / "compact" / name
        directory.mkdir(parents=True)
        (directory / "sweep.json").write_text(json.dumps(report))
    receipt = dict(state="completed", cleanup_completed=True, comparisons=compare_sweeps(data))
    (tmp_path / "compact/queue.json").write_text(json.dumps(receipt))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return (
        Namespace(
            source=source,
            compact_results=tmp_path / "compact",
            manifest=path,
            compact_manifest=path,
            output=tmp_path / "output",
            profile_output=tmp_path / "profiles",
            task=tmp_path / "task",
            weights=tmp_path / "weights",
            after_unit="compact.service",
            after_invocation="original",
            combined=combined,
            padding=padding,
        ),
        receipt,
        manifest,
    )


@pytest.mark.parametrize("gain,winning", [(1.0, False), (1.02, False), (1.07, False), (1.1, True), (1.2, True)])
def test_requires_measured_primary_workload_gain(tmp_path, gain, winning):
    args, receipt, manifest = evidence(tmp_path, gain)
    assert queue.measured_win(args.compact_results, receipt, manifest)[0] is winning


@pytest.mark.parametrize(
    "defect", ["source", "precision", "policy", "runtime", "prompt", "accounting", "tokens", "drift", "claim"]
)
def test_rejects_unmatched_or_corrupted_evidence(tmp_path, defect, expect_error):
    args, receipt, manifest = evidence(tmp_path)
    path = args.compact_results / ("after" if defect == "drift" else "compact") / "sweep.json"
    report = json.loads(path.read_text())
    if defect == "source":
        report["source_sha256"]["tt/model.py"] = "changed"
    elif defect == "precision":
        report["precision"]["kv_cache_dtype"] = "bfloat4_b"
    elif defect == "policy":
        report["precision"]["decode_recurrence"] = queue.BASELINE
    elif defect == "runtime":
        report["configuration"]["environment"]["QWEN_PREFILL_MAX_BATCH_TOKENS"] = "65536"
    elif defect == "prompt":
        report["cells"][0]["prompt_sha256"] = "changed"
    elif defect == "accounting":
        report["cells"][0]["summary"]["tpot_ms"] = 1
    elif defect == "claim":
        receipt["comparisons"][0]["speedup"] = 4
    else:
        cell = report["cells"][0]
        for sample in [cell["warmup"], *cell["samples"]]:
            if defect == "tokens":
                sample["output_sha256_per_replica"] = ["b" * 64]
            else:
                sample["decode_s"] /= 2
        cell["summary"] = summarize(cell["samples"], concurrency=16, input_tokens=32768, output_tokens=128)
    path.write_text(json.dumps(report))
    with expect_error(ValueError, ".+"):
        queue.measured_win(args.compact_results, receipt, manifest)


@pytest.mark.parametrize("change", ["edit", "delete", "add"])
def test_model_identity_includes_added_and_deleted_files(tmp_path, change, expect_error):
    args, _, manifest = evidence(tmp_path)
    queue.verify_model_source(args.source, manifest)
    path = args.source / queue.MODEL_PREFIX / "tt/model.py"
    if change == "edit":
        path.write_text("changed")
    elif change == "delete":
        path.unlink()
    else:
        path.with_name("extra.py").write_text("new")
    with expect_error(ValueError, "differs from measured"):
        queue.verify_model_source(args.source, manifest)


@pytest.mark.parametrize("gain,score_passes", [(1.0, False), (1.02, False), (1.2, False), (1.2, True)])
@pytest.mark.parametrize("padding", [False, True])
def test_controller_orders_qualification_and_matching_profiles(tmp_path, monkeypatch, gain, score_passes, padding):
    args, _, _ = evidence(tmp_path, gain, padding=padding)
    candidate = PADDING_GDN_POLICY if padding else queue.CANDIDATE
    calls = []
    monkeypatch.setattr(queue.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        queue.subprocess,
        "check_output",
        lambda *args, **kwargs: "MainPID=0\nActiveState=inactive\nLoadState=loaded\nResult=success\nInvocationID=original\n",
    )
    monkeypatch.setattr(queue, "environment", lambda *args: {})

    def qualify(options):
        calls.append("qualify")
        assert options.control_precision == f"precision_{candidate}_bfp8_all.json"
        assert options.accuracy_only and options.native_control_only
        options.results.mkdir()
        (options.results / "queue.json").write_text(
            json.dumps(
                dict(
                    state="completed",
                    passed=score_passes,
                    **{
                        "native-control": {
                            "owned_processes_stopped": True,
                            "gpqa": {"completed_samples": 198, "passed": score_passes},
                        }
                    },
                )
            )
        )

    def capture(command, *, env, root, **kwargs):
        assert calls[0] == "qualify"
        calls.append(root.name)
        assert env["QWEN_PROFILE_RECURRENCE"] == candidate
        assert env["QWEN_PROFILE_BATCH"] == "16" and env["QWEN_PROFILE_CONTEXT"] == "32768"
        assert ("--profile-ops" in command) == (root.name == "profiled")
        (root / "hardware.xml").write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>'
        )
        (root / "profile.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    cleanup_completed=True,
                    precision={"decode_recurrence": candidate},
                    output_hashes=["same"],
                    token_hashes=["same"],
                    operand_hashes={"input": "same"},
                )
            )
        )

    monkeypatch.setattr(queue, "qualify", qualify)
    monkeypatch.setattr(queue, "run_capture", capture)
    monkeypatch.setattr(queue, "collect", lambda root: dict(comparisons=[], full_trace_reconciliation_passed=True))
    queue.run(args)
    result = json.loads((args.output / "queue.json").read_text())
    assert result["state"] == "completed" and result["cleanup_completed"]
    assert result["promoted_to_serving"] is False
    if gain <= 1.02:
        assert calls == [] and result["hardware_started"] is False
        assert result["qualification_deferred_for_batch"] is True
    else:
        assert calls == ["qualify", "unprofiled", "profiled"]
        assert result["accuracy_passed"] is score_passes


@pytest.mark.parametrize("candidate", [COMBINED_GDN_POLICY, PADDING_GDN_POLICY])
def test_combined_measurement_cannot_be_mistaken_for_the_old_policy_pair(tmp_path, expect_error, candidate):
    args, receipt, manifest = evidence(
        tmp_path, combined=candidate == COMBINED_GDN_POLICY, padding=candidate == PADDING_GDN_POLICY
    )
    with expect_error(ValueError, ".*"):
        queue.measured_win(args.compact_results, receipt, manifest)
    winning, comparisons = queue.measured_win(
        args.compact_results, receipt, manifest, baseline=queue.CANDIDATE, candidate=candidate
    )
    assert winning and len(comparisons) == 2
    assert all(not r["gpqa_qualified"] for r in comparisons)
