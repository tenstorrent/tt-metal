"""Owned goal-12 dispatch ledger; no implicit retry or unowned cancellation."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/agentic_evals"
REPO = "tenstorrent/tt-agentic-bringup-qb2"
WORKFLOW = "manual-tt-shield-dispatch.yml"


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["dispatch", "register", "poll", "collect"])
    parser.add_argument("arm")
    args = parser.parse_args()
    path = DOC / "runs" / args.arm / "manifest.json"
    manifest = json.loads(path.read_text())
    ledger_path = DOC / "compute.json"
    ledger = json.loads(ledger_path.read_text())
    if args.action == "dispatch":
        assert manifest.get("prediction") and manifest.get("estimated_hours"), "Prediction required"
        assert not manifest.get("dispatched_at"), "Never redispatch a manifest"
        assert manifest["estimated_hours"][1] < 14
        active = [r for r in ledger["runs"] if r.get("status") != "completed"]
        assert len(active) < ledger["max_parallel"]
        assert not any(r.get("run_id") is None for r in active), "Register prior dispatch first"
        inputs = manifest["inputs"]
        for key in ("tt-metal-git-ref", "inference-server-git-ref", "vllm-git-ref"):
            assert len(inputs[key]) == 40, key
        assert inputs["workflow"] == "agentic" and inputs["runner-label"] == "bh-qb-ge"
        manifest["dispatched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["status"] = "dispatch_pending"
        save(path, manifest)
        ledger["runs"].append({"arm": args.arm, "status": "dispatch_pending", "run_id": None})
        save(ledger_path, ledger)
        command = ["workflow", "run", WORKFLOW, "-R", REPO, "--ref", manifest["workflow_ref"]]
        for key, value in inputs.items():
            command += ["-f", f"{key}={str(value).lower() if isinstance(value, bool) else value}"]
        print(gh(*command))
    elif args.action == "register":
        runs = json.loads(gh("api", f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs?event=workflow_dispatch&per_page=50"))["workflow_runs"]
        created = dt.datetime.fromisoformat(manifest["dispatched_at"])
        matches = [r for r in runs if dt.datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) >= created - dt.timedelta(seconds=2)
                   and r["head_branch"] == manifest["workflow_ref"]
                   and manifest["inputs"]["tt-metal-git-ref"] in r["display_title"]
                   and manifest["inputs"]["inference-server-git-ref"] in r["display_title"]]
        assert len(matches) == 1, f"Need unique run, found {len(matches)}"
        manifest.update(run_id=matches[0]["id"], url=matches[0]["html_url"], status=matches[0]["status"])
        save(path, manifest)
        entry = next(r for r in ledger["runs"] if r["arm"] == args.arm)
        entry.update(run_id=manifest["run_id"], status=manifest["status"], url=manifest["url"])
        ledger["peak_runs_in_flight"] = max(ledger["peak_runs_in_flight"], sum(r["status"] != "completed" for r in ledger["runs"]))
        save(ledger_path, ledger)
        print(json.dumps(entry))
    else:
        run_id = manifest["run_id"]
        run = json.loads(gh("api", f"repos/{REPO}/actions/runs/{run_id}"))
        jobs = json.loads(gh("api", f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100"))
        save(path.parent / "run.json", run)
        save(path.parent / "jobs.json", jobs)
        manifest.update(status=run["status"], conclusion=run["conclusion"])
        save(path, manifest)
        entry = next(r for r in ledger["runs"] if r["run_id"] == run_id)
        entry.update(status=run["status"], conclusion=run["conclusion"])
        save(ledger_path, ledger)
        print(json.dumps({"run_id": run_id, "status": run["status"], "conclusion": run["conclusion"],
                          "jobs": [{k: j.get(k) for k in ("name", "status", "conclusion", "labels", "runner_name", "started_at")} for j in jobs["jobs"]]}, indent=2))
        if args.action == "collect":
            assert run["status"] == "completed", "Collect after completion"
            print(gh("run", "download", str(run_id), "-R", REPO, "-D", str(path.parent / "artifacts")))


if __name__ == "__main__":
    main()
