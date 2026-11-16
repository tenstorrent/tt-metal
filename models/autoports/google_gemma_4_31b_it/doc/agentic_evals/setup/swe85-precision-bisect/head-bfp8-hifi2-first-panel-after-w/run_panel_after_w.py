#!/usr/bin/env python3
"""Wait for W terminal success, verify frozen sources and safe preflight, then run P HiFi2 panel."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[8]
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/swe85-precision-bisect/head-bfp8-hifi2-first-panel-after-w"
PROPOSAL = OUT / "proposal.json"
ADMISSION = OUT / "admission.json"
STOP = OUT / "STOP_BEFORE_LAUNCH"
W = D / "setup/swe85-adaptive-req5414-cap2048-after-v-candidate"
V = D / "setup/swe85-adaptive-suffix4096-candidate"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def check_sources(proposal):
    for item in proposal["source_hashes"]:
        assert sha(ROOT / item["path"]) == item["sha256"], f"source hash changed: {item['path']}"


def active_hf_processes():
    hits = []
    proc = Path("/proc")
    for ent in proc.iterdir():
        if not ent.name.isdigit() or int(ent.name) == os.getpid():
            continue
        try:
            args = (ent / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if "precision_repro_hf.py" in args:
            hits.append({"pid": int(ent.name), "cmdline": args[:400]})
    return hits


def wait_w(proposal):
    deadline = time.monotonic() + 43200
    while True:
        if STOP.exists():
            raise RuntimeError("stop file set before W closure")
        if W.joinpath("queue.json").exists() and W.joinpath("full-completion.json").exists():
            break
        if time.monotonic() > deadline:
            raise TimeoutError("W did not reach terminal success within12h")
        time.sleep(10)
    q = read(W / "queue.json")
    s = read(W / "served-completion.json")
    f = read(W / "full-completion.json")
    assert (
        q.get("status") == "all_req5414_cap2048_extensions_completed" and q.get("exit_code") == 0
    ), "W queue not successful terminal"
    assert s.get("name") == "served" and s.get("exit_code") == 0, "W served completion failed"
    assert f.get("name") == "full" and f.get("exit_code") == 0, "W full completion failed"
    assert sha(W / "admission.json") == proposal["predecessors"]["W"]["admission_sha256"]
    assert sha(W / "proposal.json") == proposal["predecessors"]["W"]["proposal_sha256"]
    assert sha(W / "run_after_v.py") == proposal["predecessors"]["W"]["driver_sha256"]
    return {
        "queue_sha256": sha(W / "queue.json"),
        "served_completion_sha256": sha(W / "served-completion.json"),
        "full_completion_sha256": sha(W / "full-completion.json"),
    }


def main():
    admission = read(ADMISSION)
    assert admission.get("status") == "admitted", "root admission required"
    proposal = read(PROPOSAL)
    ah = sha(ADMISSION)
    assert sha(PROPOSAL) == admission["proposal_sha256"]
    assert sha(Path(__file__)) == admission["driver_sha256"]
    check_sources(proposal)
    state = {
        "status": "waiting_for_W_terminal_success",
        "admission_sha256": ah,
        "stop_file": str(STOP),
        "queued_unix": time.time(),
    }
    (OUT / "queue.json").write_text(json.dumps(state, indent=2) + "\n")
    w_hashes = wait_w(proposal)
    vq = read(V / "queue.json")
    vf = read(V / "full-completion.json")
    assert vq.get("status") == "all_suffix4096_jobs_completed" and vq.get("exit_code") == 0
    assert vf.get("name") == "full" and vf.get("exit_code") == 0
    pytest = read(V / "full-coordinator-pytest-cap4096.json")
    assert (
        pytest.get("verdict") == "normal"
        and pytest.get("generated_tokens") == 2380
        and pytest.get("cap") == 4096
        and pytest.get("prefix2048_exact") is True
        and pytest.get("score_changed") is False
    )
    assert sha(V / "admission.json") == proposal["predecessors"]["V"]["admission_sha256"]
    state.update(
        status="predecessors_succeeded",
        w_hashes=w_hashes,
        v_full_completion_sha256=sha(V / "full-completion.json"),
        v_queue_sha256=sha(V / "queue.json"),
    )
    (OUT / "queue.json").write_text(json.dumps(state, indent=2) + "\n")
    if STOP.exists():
        state.update(status="stopped_before_panel")
        (OUT / "queue.json").write_text(json.dumps(state, indent=2) + "\n")
        return
    check_sources(proposal)
    assert sha(ADMISSION) == ah and sha(PROPOSAL) == admission["proposal_sha256"]
    hits = active_hf_processes()
    assert not hits, f"active HF reference process overlaps: {hits}"
    sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server

    owners = benchmark_server.device_owners()
    assert not owners, f"device has owners: {owners}"
    assert (
        sha(W / "queue.json") == w_hashes["queue_sha256"]
        and sha(W / "served-completion.json") == w_hashes["served_completion_sha256"]
        and sha(W / "full-completion.json") == w_hashes["full_completion_sha256"]
    ), "W terminal evidence changed before launch"
    assert (
        sha(V / "full-completion.json") == state["v_full_completion_sha256"]
        and sha(V / "queue.json") == state["v_queue_sha256"]
    ), "V terminal evidence changed before launch"
    assert not (OUT / "launch.json").exists(), "refuse duplicate launch"
    command = proposal["execution"]["command"]
    launch = {
        "command": command,
        "start_unix": time.time(),
        "admission_sha256": ah,
        "w_hashes": w_hashes,
        "v_full_completion_sha256": state["v_full_completion_sha256"],
        "owner_preflight": owners,
        "active_hf_processes": hits,
        "forecast": proposal["execution"],
    }
    (OUT / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    state.update(status="panel_started", start_unix=launch["start_unix"])
    (OUT / "queue.json").write_text(json.dumps(state, indent=2) + "\n")
    with (OUT / "panel.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    done = {
        **launch,
        "end_unix": time.time(),
        "exit_code": result.returncode,
        "elapsed_seconds": time.time() - launch["start_unix"],
    }
    (OUT / "completion.json").write_text(json.dumps(done, indent=2) + "\n")
    state.update(
        status="completed" if result.returncode == 0 else "failed", exit_code=result.returncode, end_unix=time.time()
    )
    (OUT / "queue.json").write_text(json.dumps(state, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError("precision panel failed; no automatic retry")


if __name__ == "__main__":
    main()
