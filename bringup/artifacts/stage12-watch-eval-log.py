"""Private observer only: no dispatch, cancellation, scoring or harness mutation."""
import argparse, datetime, json, pathlib, re, subprocess, time

p = argparse.ArgumentParser()
p.add_argument("run_id", type=int)
p.add_argument("--clients", type=int, required=True)
p.add_argument("--hours", type=float, default=12)
a = p.parse_args()
root = pathlib.Path.cwd()
out = root / "bringup/artifacts/stage12-eval-observation" / str(a.run_id)
out.mkdir(parents=True, exist_ok=True)
deadline = time.monotonic() + a.hours * 3600
while time.monotonic() < deadline:
    status = root / f"bringup/artifacts/stage12-ci-monitor/{a.run_id}.json"
    d = json.loads(status.read_text()) if status.exists() else {}
    job = next((j for j in d.get("jobs", []) if "/ run-tests /" in j["name"]), None)
    if job:
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp = out / "next.log"
        with tmp.open("wb") as f:
            r = subprocess.run(
                [
                    "timeout",
                    "90",
                    "gh",
                    "api",
                    f"repos/tenstorrent/tt-agentic-bringup-qb2/actions/jobs/{job['id']}/logs",
                ],
                stdout=f,
                stderr=subprocess.PIPE,
            )
        end = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta = {
            "run_id": a.run_id,
            "job_id": job["id"],
            "fetch_started_at": start,
            "observed_at": end,
            "exit_code": r.returncode,
        }
        if r.returncode == 0:
            tmp.replace(out / "current.log")
            s = (out / "current.log").read_text(errors="replace")
            ts = re.findall(r"^([0-9]{4}-[0-9T:.Z-]+)", s, re.M)
            rows = re.findall(r'active_rows["\s]*[:=]\s*([0-9]+)', s)
            meta.update(
                bytes=(out / "current.log").stat().st_size,
                last_event=ts[-1] if ts else None,
                last_sampled_active_rows=int(rows[-1]) if rows else None,
                requesting_counters=re.findall(r"Requesting API[^\r\n]{0,180}", s)[-2:],
            )
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            result = subprocess.run(
                [
                    "python3",
                    str(root / "bringup/artifacts/gpqa-first-agent-observer/observe.py"),
                    "--snapshot",
                    str(out / "current.log"),
                    "--state",
                    str(out / "observer-state.json"),
                    "--output",
                    str(out / f"observer-{stamp}.json"),
                    "--boundary-marker",
                    "Running eval task=r1_gpqa_diamond",
                    "--fetch-start",
                    start,
                    "--fetch-end",
                    end,
                    "--clients",
                    str(a.clients),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            meta["observer_exit_code"] = result.returncode
            if result.returncode:
                meta["observer_error"] = result.stderr[-1000:]
            else:
                meta["observer_output"] = json.loads(result.stdout)
        else:
            meta["error"] = "Fetch failed; previous successful snapshot preserved"
        (out / "observation.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta), flush=True)
    if d.get("status") == "completed":
        break
    time.sleep(60)
