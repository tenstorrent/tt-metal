"""Own the benchmark server processes and attest their observed configuration."""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
BENCH = ROOT / "doc/benchmark"
STATE = BENCH / "owned_server.json"
MODEL = "IFM/K2-Horizon-MoVA-36B-A4B"
REVISION = "e9ce717642b2074082875be233c02c729ba47960"


def process_identity(pid):
    """Use boot and start ticks to distinguish a process from a recycled PID."""
    proc = Path(f"/proc/{pid}")
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "start_ticks": int(fields[19]),
            "process_group": int(fields[2]),
        }
    except FileNotFoundError:
        return None


def absent_process(state):
    try:
        os.killpg(state["pid"], 0)
    except ProcessLookupError:
        return False
    raise RuntimeError("Server leader is absent but its group survives; retaining ownership evidence for investigation")


def owned_process(state):
    observed = process_identity(state["pid"])
    if observed is None:
        return absent_process(state)
    if observed != state.get("process_identity") or observed["process_group"] != state["pid"]:
        raise RuntimeError(
            "Recorded server PID is live but ownership cannot be verified; refusing to signal or reuse it"
        )
    try:
        command = Path(f"/proc/{state['pid']}/cmdline").read_bytes().decode().strip("\0").split("\0")
    except FileNotFoundError:
        return absent_process(state)
    # A zombie leader can still have a surviving engine in its owned group.
    if command == [""]:
        return True
    if command != state["argv"]:
        raise RuntimeError("Owned server command changed; refusing to signal or reuse it")
    return True


def stop(state=None):
    previous_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        _stop(state)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _stop(state=None):
    if state is None and not STATE.exists():
        return
    if state is None:
        state = json.loads(STATE.read_text())
    pid = state["pid"]
    if owned_process(state):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # The runner allows ten seconds for a terminated hook to clean up.
        # Escalate within that grace period, including a surviving engine child.
        for _ in range(8):
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(1)
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if STATE.exists():
        STATE.rename(BENCH / f"stopped-server-{pid}.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-num-seqs", type=int)
    p.add_argument("--base-url")
    p.add_argument("--output", type=Path)
    p.add_argument("--stop", action="store_true")
    a = p.parse_args()
    if a.stop:
        stop()
        return
    if STATE.exists():
        state = json.loads(STATE.read_text())
        proc = Path(f"/proc/{state['pid']}/cmdline")
        prior = Path(state["log"]).parent
        prior_identity = prior / f"perf-b{a.max_num_seqs}-server.json"
        if owned_process(state) and prior_identity.exists():
            recorded = json.loads(prior_identity.read_text())
            observed = proc.read_bytes().decode().strip("\0").split("\0")
            if recorded["server_command"] == observed and requests.get(a.base_url + "/health", timeout=5).ok:
                # Preserve the original, still-live event sink. Only a matching
                # owned server can be reused, and its launch evidence is retained.
                for suffix in (".jsonl", ".runtime.json"):
                    source = prior / f"phase-b{a.max_num_seqs}{suffix}"
                    target = a.output.parent / source.name
                    if source != target:
                        target.symlink_to(source.resolve())
                evidence = prior / recorded["configuration_evidence"]
                target = a.output.parent / evidence.name
                if evidence != target:
                    shutil.copyfile(evidence, target)
                    shutil.copyfile(Path(state["log"]), a.output.parent / "reused-server-startup.log")
                recorded["reused_owned_server"] = True
                recorded["prior_startup_evidence"] = str(evidence)
                a.output.write_text(json.dumps(recorded, indent=2) + "\n")
                if a.max_num_seqs == 32:
                    (BENCH / "identity.json").write_text(json.dumps(recorded, indent=2) + "\n")
                print(f"Reused owned server PID {state['pid']}, slots {a.max_num_seqs}")
                return
    stop()
    out = a.output.parent
    capacity = a.max_num_seqs
    log = out / f"server-b{capacity}.log"
    timing = out / f"phase-b{capacity}.jsonl"
    argv = [
        str(REPO / "python_env/bin/python"),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--block-size",
        "64",
        "--max-num-seqs",
        str(capacity),
        "--max-model-len",
        "524288",
        "--port",
        "8000",
        "--trust-remote-code",
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--max-num-batched-tokens",
        "256",
        "--no-enable-prefix-caching",
        "--reasoning-parser-plugin",
        str(ROOT / "tests/benchmark_reasoning.py"),
        "--reasoning-parser",
        "k2_horizon",
        "--additional-config",
        json.dumps(
            {
                "tt": {
                    "trace_region_size": 0,
                    "fabric_config": "FABRIC_1D_RING",
                    "trace_mode": "all",
                    "sample_on_device_mode": "all",
                }
            }
        ),
    ]
    env = dict(
        os.environ,
        MESH_DEVICE="P300x2",
        HF_MODEL=MODEL,
        TT_MODEL_CLASS_OVERRIDES="TTK2HorizonForCausalLM=models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.generator_vllm:K2HorizonForCausalLM",
        K2_VLLM_HOST_COMPAT="1",
        K2_BENCHMARK_TIMING=str(timing),
        TT_METAL_TRACE_ALLOC_TRACKING="0",
    )
    for key in list(env):
        if "PROFILER" in key or key in ("TT_METAL_WATCHER", "K2_VLLM_EVENTS", "K2_VLLM_LAYERS"):
            env.pop(key)
    began = time.time()
    launch_state = None
    try:
        # Defer runner cancellation until the newly detached server is recorded.
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            with log.open("w") as stream:
                process = subprocess.Popen(
                    argv,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    # This standalone helper is single-threaded. Do not leave
                    # the server with its parent's temporary SIGTERM mask.
                    preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask),
                )
            launch_state = dict(
                pid=process.pid, argv=argv, log=str(log), started=began, process_identity=process_identity(process.pid)
            )
            temporary_state = STATE.with_suffix(".tmp")
            temporary_state.write_text(json.dumps(launch_state, indent=2) + "\n")
            temporary_state.replace(STATE)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        for _ in range(900):
            if process.poll() is not None:
                raise RuntimeError(f"server exited {process.returncode}: {log}")
            try:
                if requests.get(a.base_url + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("server readiness timeout")
        observed = Path(f"/proc/{process.pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
        init = json.loads(timing.read_text().splitlines()[0])
        if init["layers"] != list(range(48)) or init["max_num_seqs"] != capacity or init["max_model_len"] != 524288:
            raise ValueError("actual model initialization differs from requested profile")
        snapshot = out / f"configuration-b{capacity}.json"
        evidence = dict(
            argv=observed,
            initialization=init,
            startup_log=log.name,
            startup_seconds=time.time() - began,
            server_pid=process.pid,
            source_hashes={
                str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    ROOT / "tt/generator.py",
                    ROOT / "tt/generator_vllm.py",
                    ROOT / "tests/benchmark_timing.py",
                ]
            },
            prefix_caching_flag="--no-enable-prefix-caching" in observed,
        )
        snapshot.write_text(json.dumps(evidence, indent=2) + "\n")
        commits = {
            "tt-metal": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "vllm-tt-plugin": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO.parent / "vllm-tt-plugin", text=True
            ).strip(),
            "vllm": "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        }
        identity = dict(
            model=MODEL,
            implementation=str(ROOT.relative_to(REPO)),
            generator_module="models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.generator_vllm",
            model_revision=REVISION,
            tokenizer_revision=REVISION,
            precision=init["precision"],
            layer_count=48,
            configured_layer_count=48,
            source_commits=commits,
            hardware="P300x2; four Blackhole p300c ASICs; TP/EP mesh 1x4; devices 0,1,2,3",
            server_command=observed,
            prefix_caching=False,
            max_num_seqs=capacity,
            max_model_len=524288,
            base_url=a.base_url,
            configuration_evidence=snapshot.name,
            configuration_evidence_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        )
        a.output.write_text(json.dumps(identity, indent=2) + "\n")
        if capacity == 32:
            (BENCH / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        print(json.dumps(evidence))
    except BaseException:
        stop(launch_state)
        raise


def entrypoint():
    def interrupted(signum, frame):
        raise InterruptedError(f"server-selection hook interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException:
        # Reuse checks and the initial profile-switch cleanup can be cancelled
        # before the launch-specific guard is entered.
        stop()
        raise


if __name__ == "__main__":
    entrypoint()
