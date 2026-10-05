"""Own and identify the Stage 11 vLLM servers; never touch unrelated processes."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO = MODEL_DIR.parents[2]
ROOT = REPO.parent
EVIDENCE = MODEL_DIR / "doc/benchmark"
STATE = EVIDENCE / "server-state.json"
REVISION = "036114ce8d46c32b24c15423211069abb9c5d25e"
MODEL = "IFM/K2-Horizon-7B"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def alive(state):
    try:
        stat = Path(f"/proc/{state['pid']}/stat").read_text().split()
        return stat[2] != "Z" and stat[21] == state["process_start_ticks"]
    except FileNotFoundError:
        return False


def stop():
    if not STATE.exists():
        return
    state = json.loads(STATE.read_text())

    def members():
        live_members = []
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().split()
                if int(fields[4]) == state["pid"] and fields[2] != "Z":
                    live_members.append(fields[0])
            except (FileNotFoundError, ProcessLookupError):
                continue
        return live_members

    if alive(state):
        if os.getpgid(state["pid"]) != state["pid"]:
            raise RuntimeError("Owned server process group changed")
    else:
        # A failed API leader can leave its owned engine alive. Establish
        # ownership from the unique inherited phase path before cleanup.
        expected = f"K2_BENCHMARK_PHASE_PATH={state['phase_path']}".encode()
        for pid in members():
            if expected not in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                raise RuntimeError(f"Cannot prove ownership of process {pid}")
    try:
        os.killpg(state["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(600):
        if not members():
            state["stopped_at"] = time.time()
            STATE.write_text(json.dumps(state, indent=2) + "\n")
            return
        time.sleep(0.1)
    raise RuntimeError("Owned server group did not finish shutdown; preserve and inspect before reset")


def ready(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    def interrupted(signum, frame):
        raise InterruptedError(f"Server hook interrupted by {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-seqs", type=int, choices=(1, 32))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    if args.stop:
        stop()
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not (state and alive(state) and state["max_num_seqs"] == args.max_num_seqs and ready(args.base_url)):
        if ready(args.base_url) and not (state and alive(state)):
            raise RuntimeError("Endpoint is owned by an unrecorded process")
        stop()
        stamp = time.time_ns()
        log = EVIDENCE / "setup" / f"server-b{args.max_num_seqs}-{stamp}.log"
        phase = log.with_suffix(".phases.jsonl")
        runtime = log.with_suffix(".runtime.json")
        transport = log.with_suffix(".transport.json")
        cmd = [
            str(REPO / "python_env/bin/python"),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL,
            "--block-size",
            "32",
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--port",
            "8000",
            "--max-model-len",
            "524288",
            "--no-enable-prefix-caching",
            "--additional-config",
            json.dumps(
                {
                    "tt": {
                        "sample_on_device_mode": "all",
                        "trace_region_size": 200000000,
                        "fabric_config": "FABRIC_1D_RING",
                        "trace_mode": "all",
                    }
                }
            ),
            "--trust-remote-code",
            "--async-scheduling",
            "--max-logprobs",
            "-1",
            "--revision",
            REVISION,
            "--code-revision",
            REVISION,
            "--tokenizer-revision",
            REVISION,
            "--served-model-name",
            MODEL,
            "--reasoning-parser-plugin",
            str(MODEL_DIR / "tests/benchmark_reasoning_parser.py"),
            "--reasoning-parser",
            "k2_horizon_benchmark",
            "--middleware",
            "models.demos.k2_horizon_7b_qb2.tests.benchmark_chat_middleware.K2BenchmarkChatMiddleware",
        ]
        env = dict(os.environ)
        env.update(
            MESH_DEVICE="P300x2",
            HF_MODEL=MODEL,
            HF_HUB_OFFLINE="1",
            # Keep the validated Stage10 single-user setting. The accuracy
            # server uses eight physical CPU cores for full-vocabulary work.
            OMP_NUM_THREADS="16" if args.max_num_seqs == 1 else "8",
            TT_METAL_TRACE_ALLOC_TRACKING="1",
            K2_VLLM_ALLOW_HOST_SAMPLING="1",
            K2_VLLM_FORCE_HOST_SAMPLING="0",
            K2_BENCHMARK_PHASE_PATH=str(phase),
            K2_BENCHMARK_IDENTITY_PATH=str(runtime),
            K2_BENCHMARK_CHAT_TRANSPORT_PATH=str(transport),
        )
        for key in list(env):
            if key.startswith("TT_METAL_PROFILER") or key in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER"):
                env.pop(key)
        started = time.time()
        with log.open("w") as stream:
            stream.write(json.dumps({"argv": cmd, "started_at": started}) + "\n")
            stream.flush()
            proc = subprocess.Popen(
                cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
        state = dict(
            pid=proc.pid,
            process_start_ticks=Path(f"/proc/{proc.pid}/stat").read_text().split()[21],
            max_num_seqs=args.max_num_seqs,
            command=cmd,
            log=str(log),
            phase_path=str(phase),
            runtime_identity=str(runtime),
            chat_transport_identity=str(transport),
            started_at=started,
        )
        STATE.write_text(json.dumps(state, indent=2) + "\n")
        try:
            while not ready(args.base_url):
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited {proc.returncode}; inspect {log}")
                if time.time() - started > 600:
                    raise TimeoutError(f"Server did not become ready; inspect {log}")
                time.sleep(1)
        except BaseException:
            stop()
            raise
        state.update(ready_at=time.time(), startup_seconds=time.time() - started)
        STATE.write_text(json.dumps(state, indent=2) + "\n")
    runtime = json.loads(Path(state["runtime_identity"]).read_text())
    transport = json.loads(Path(state["chat_transport_identity"]).read_text())
    transport_source = (MODEL_DIR / "tests/benchmark_chat_middleware.py").resolve()
    if (
        transport.get("file") != str(transport_source)
        or transport.get("sha256") != sha(transport_source)
        or transport.get("model") != MODEL
        or transport.get("template_rendered_by_middleware") is not False
    ):
        raise RuntimeError("Actual imported chat transport differs from the recorded model-local adapter")
    observed_argv = Path(f"/proc/{state['pid']}/cmdline").read_bytes().decode().strip("\0").split("\0")
    if observed_argv != state["command"]:
        raise RuntimeError("Observed server argv differs from the recorded launch")
    process_environment = dict(
        item.split("=", 1)
        for item in Path(f"/proc/{state['pid']}/environ").read_bytes().decode().strip("\0").split("\0")
        if "=" in item
    )
    observed_environment = {
        key: process_environment.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "MESH_DEVICE",
            "HF_MODEL",
            "HF_HUB_OFFLINE",
            "K2_VLLM_ALLOW_HOST_SAMPLING",
            "K2_VLLM_FORCE_HOST_SAMPLING",
            "K2_BENCHMARK_CHAT_TRANSPORT_PATH",
            "TT_METAL_TRACE_ALLOC_TRACKING",
            "TT_METAL_HOME",
            "TT_METAL_CACHE_DIR",
            "PYTHONPATH",
        )
    }
    expected_threads = 16 if args.max_num_seqs == 1 else 8
    if int(observed_environment["OMP_NUM_THREADS"]) != expected_threads:
        raise RuntimeError("Running server has the wrong CPU thread configuration")
    # OMP_NUM_THREADS is the validated Stage10 launch setting, not a
    # guarantee of Torch's effective pool size. This MKL build initializes
    # eight physical-core threads even with OMP_NUM_THREADS=16. Compare to
    # the same interpreter/environment initialization instead of silently
    # forcing a different, unvalidated single-user configuration.
    thread_probe = json.loads(
        subprocess.check_output(
            [
                str(REPO / "python_env/bin/python"),
                "-c",
                (
                    "import json, os, torch; print(json.dumps({"
                    "'torch_intraop': torch.get_num_threads(), "
                    "'torch_interop': torch.get_num_interop_threads(), "
                    "'cpu_affinity': sorted(os.sched_getaffinity(0)), "
                    "'omp_environment': os.environ.get('OMP_NUM_THREADS'), "
                    "'parallel_info': torch.__config__.parallel_info()}))"
                ),
            ],
            env=process_environment,
            text=True,
            timeout=30,
        )
    )
    for key in ("torch_intraop", "torch_interop", "cpu_affinity", "omp_environment"):
        if runtime["cpu_threads"][key] != thread_probe[key]:
            raise RuntimeError(f"Actual engine CPU thread initialization differs: {key}")
    if runtime["max_num_seqs"] != args.max_num_seqs or runtime["layer_count"] != 36:
        raise RuntimeError("Imported model capacity/layers do not match the server")
    log_text = Path(state["log"]).read_text(errors="replace")
    if "max_model_len=524288" not in log_text or "enable_prefix_caching=False" not in log_text:
        raise RuntimeError("Effective vLLM configuration not found in startup log")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.parent / f"configuration-b{args.max_num_seqs}.log"
    snapshot.write_text(
        json.dumps(
            {"process": state, "runtime": runtime, "environment": observed_environment, "chat_transport": transport},
            indent=2,
        )
        + "\n"
        + log_text
    )
    identity = dict(
        model=MODEL,
        implementation="models/demos/k2_horizon_7b_qb2",
        generator_module="models.demos.k2_horizon_7b_qb2.tt.generator_vllm",
        model_revision=REVISION,
        tokenizer_revision=REVISION,
        precision=json.loads((MODEL_DIR / "doc/datatype_sweep/selected_precision_config.json").read_text()),
        layer_count=runtime["layer_count"],
        configured_layer_count=36,
        source_commits={
            name: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
            for name, path in (("tt-metal", REPO), ("vllm", ROOT / "vllm"))
        },
        hardware="4 Blackhole p300c chips / P300x2 / MeshShape(1,4), TP4 DP1, FABRIC_1D_RING",
        server_command=observed_argv,
        server_environment=observed_environment,
        prefix_caching=False,
        max_num_seqs=args.max_num_seqs,
        max_model_len=524288,
        base_url=args.base_url,
        configuration_evidence=snapshot.name,
        configuration_evidence_sha256=sha(snapshot),
        runtime=runtime,
        cpu_thread_initialization=thread_probe,
        process=state,
        chat_transport=transport,
    )
    args.output.write_text(json.dumps(identity, indent=2) + "\n")
    if args.max_num_seqs == 32 and args.output.parent.resolve() == (EVIDENCE / "run").resolve():
        # This is the observed baseline for the actual timed attempt. A prior
        # attempt's identity must not stand in for a newly launched process.
        (EVIDENCE / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(
        json.dumps(
            {"pid": state["pid"], "max_num_seqs": args.max_num_seqs, "startup_seconds": state["startup_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
