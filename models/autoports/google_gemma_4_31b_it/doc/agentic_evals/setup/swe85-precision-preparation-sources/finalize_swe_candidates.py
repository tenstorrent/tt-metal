import datetime
import json
import pathlib
import subprocess

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
O = D / "anomalies/swe85-below-band/correct-actions"
jobs = []
rows = []
for p in sorted(O.glob("*/manifest.json")):
    j = json.loads(p.read_text())
    d = p.parent
    if not j.get("action_case"):
        continue
    if j["task"] == "scikit-learn__scikit-learn-10908":
        for e in j["audited_prior_tracked_edits"]:
            e["files"] = [
                "sklearn/utils/__init__.py",
                "sklearn/feature_extraction/text.py",
                "sklearn/__check_build/_check_build.py",
            ]
        j["image_absent_paths"] = ["sklearn/__check_build/_check_build.py"]
        j["prior_generated_files"] = ["sklearn/__check_build/_check_build.py"]
    if j["task"] == "astropy__astropy-13398":
        j["image_absent_paths"] = ["astropy/coordinates/builtin_frames/itrs_observed_transforms.py"]
    patch = d / "state-relative.patch.private"
    r = subprocess.run(
        ["patch", "--batch", "--dry-run", "-p1", "-i", str(patch.resolve())],
        cwd=d / "precall",
        capture_output=True,
        text=True,
    )
    s = subprocess.run(["bash", "-n", str(d / "candidate-action.private.sh")], capture_output=True, text=True)
    j["state_relative_patch_validation"] = {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    j["action_shell_syntax"] = {"returncode": s.returncode, "stderr": s.stderr}
    assert r.returncode == s.returncode == 0, (p, r.stdout, r.stderr)
    p.write_text(json.dumps(j, indent=2) + "\n")
    rows.append({"task": j["task"], "call": j["call"], "manifest": str(p), "semantic": "pending"})
    jobs.append(
        {"name": j["task"] + ".call" + str(j["call"]), "case": j["action_case"], "branches": ["chosen", "correct"]}
    )
(O / "all-candidate-action-cases.json").write_text(json.dumps(jobs, indent=2) + "\n")
wrong = []
reviewed = list((D / "current/swe-extended85-head-bfp8/live-tasks").glob("*/coordinator.json"))
for p in reviewed:
    j = json.loads(p.read_text())
    if j.get("cause") == "wrong":
        wrong.append(j["task"])
ready = {r["task"] for r in rows}
summary = {
    "status": "CPU source reconstruction/application/syntax only; all semantic execution pending",
    "snapshot_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "reviewed_coordinators": len(reviewed),
    "coordinator_wrong_tasks": len(wrong),
    "wrong_tasks_with_candidate": sum(t in ready for t in wrong),
    "candidate_cases": len(rows),
    "prepared_task_calls": rows,
    "explicit_gaps": [t for t in wrong if t not in ready],
    "gap_resolution": {
        "psf__requests-1142": "Native initial ls proves build/lib exists at entry. Stopped-image build/lib models.py recovered size20789 matches native pre-copy listing and hash equals image tracked source. Restore52 and compat edit56 reproduced; no inferred unknown build step. Native logs do not include cryptographic source hash.",
        "mwaskom__seaborn-3069": "Replayed exact audited sed/whitespace source operations only. Failed message74 has no file write. Literal assignments92 extracted with AST, not arbitrary agent Python execution. Native numbered source slices95/99 match exactly.",
        "scikit-learn__scikit-learn-10908": "Actual pre34 erroneous text.py ABC import retained in precall. Candidate now repairs concrete defaultdict/namedtuple imports in text.py and all three call34 target files while using ABC module only for abstract classes. Task reference fix additional; no import/test execution claim.",
    },
    "cpu_validation": {"state_relative_patch_dry_runs": len(rows), "shell_syntax": len(rows), "semantic_tests_run": 0},
    "limitations": [
        "Source state file-scoped; original dependency installations and untracked reproducer/config state not fully reconstructed.",
        "Canonical reply token identity unproven; native IDs null and count gaps retained.",
        "Counterfactual actions condition on original possibly faulty reasoning.",
        "References are offline forced targets, never scored model inputs.",
    ],
}
(D / "setup/swe85-correct-action-roster-coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    {
        k: summary[k]
        for k in [
            "reviewed_coordinators",
            "coordinator_wrong_tasks",
            "wrong_tasks_with_candidate",
            "candidate_cases",
            "explicit_gaps",
        ]
    }
)
