# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One CI job, one T3K, sealed main/PR Python sources and identical native bytes.

Diagnostic evidence only. No device reset, cache purge, threshold change or
required-check replacement. Incomplete visibility or a native hang stops the pair.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import shutil
import stat
import subprocess
import tarfile
import time
import urllib.request


BASELINE = "21754c005c015161fe5de2d1cdb10bf64a6fd5a1"
CANDIDATE = "ee072dd216c2d19c6b228be26f695757d8ec1500"
PLUGIN = "c1c85eb6bfe14a3e47afc450e25f7a8ddc81d974"
IMAGE = "ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64@sha256:3643bc059bf70bd14d3031f5f53f8853abc564cd2a5d9a3dcff1b54c7003e25b"
HANG = re.compile(r"Native Timeout|Timeout detected|device hang|Segmentation fault|Fatal @|terminate called", re.I)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def quiet(path):
    """Only PID/start time/comm/device rdev; no args, environ or FD paths."""
    devices = {
        os.stat(node).st_rdev for node in Path("/dev/tenstorrent").iterdir() if stat.S_ISCHR(node.stat().st_mode)
    }
    assert len(devices) == 8, "The assigned runner must expose exactly one T3K"
    holders, unknown = [], []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    descriptor = fd.stat()
                except FileNotFoundError:
                    continue
                if stat.S_ISCHR(descriptor.st_mode) and descriptor.st_rdev in devices:
                    start = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
                    holders.append(
                        {
                            "pid": int(proc.name),
                            "start_ticks": start,
                            "comm": (proc / "comm").read_text().strip(),
                            "rdev": descriptor.st_rdev,
                        }
                    )
        except FileNotFoundError:
            continue  # A process/FD disappeared during this read-only scan.
        except PermissionError:
            unknown.append(int(proc.name))
    verdict = {
        "quiet": not holders and not unknown,
        "device_rdevs": sorted(devices),
        "holders": holders,
        "incomplete_pid_visibility": sorted(set(unknown)),
        "telemetry_exclusions": [],
        "runner_name": os.environ["RUNNER_NAME"],
        "run_id": os.environ["GITHUB_RUN_ID"],
    }
    write(path, verdict)
    assert verdict["quiet"], "Busy or incompletely visible driver; do not open any device"


def run(command, log, env, cwd, seconds):
    with log.open("w") as stream:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError("Phase deadline: preserve and stop the pair")


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "ppid": int(fields[1]),
            "session": int(fields[3]),
            "start_ticks": int(fields[19]),
            "state": fields[0],
        }
    except FileNotFoundError:
        return None


def track_server(server, original, tracked):
    current = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        identity = process_identity(int(proc.name))
        if identity:
            current.append(identity)
    changed = True
    while changed:
        changed = False
        for identity in current:
            if (
                identity["pid"] not in tracked
                and identity["start_ticks"] >= original["start_ticks"]
                and (identity["session"] == server.pid or identity["ppid"] in tracked)
            ):
                tracked[identity["pid"]] = identity
                changed = True


def close_server(server, original, tracked, path):
    track_server(server, original, tracked)
    now = process_identity(server.pid)
    if now and now["start_ticks"] == original["start_ticks"]:
        server.terminate()
    deadline = time.monotonic() + 45
    remaining = []
    while time.monotonic() < deadline:
        server.poll()  # Reap the launcher once it has exited.
        track_server(server, original, tracked)
        remaining = [
            pid
            for pid, saved in tracked.items()
            if (now := process_identity(pid)) and now["start_ticks"] == saved["start_ticks"] and now["state"] != "Z"
        ]
        if not remaining:
            break
        time.sleep(1)
    escalated = bool(remaining)
    if escalated:
        # Escalation is cleanup failure and cannot admit the next source leg.
        # Signal only captured identities that still have the same start time.
        for pid in remaining:
            now = process_identity(pid)
            if now and now["start_ticks"] == tracked[pid]["start_ticks"]:
                os.kill(pid, signal.SIGKILL)
        server.wait(timeout=10)
    write(
        path,
        {
            "ordinary_close_seconds": 45,
            "escalated": escalated,
            "tracked": [{"pid": p, "start_ticks": value["start_ticks"]} for p, value in tracked.items()],
        },
    )
    return not escalated


def validate_tar(native):
    process = subprocess.Popen(["zstd", "-dc", str(native)], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            path = Path(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert path.parts[0] in {"build", "runtime", "tt_metal", "ttnn"}
            assert (
                member.isfile() or member.isdir() or member.issym() or member.islnk()
            ), "Unexpected archive member type"
            if member.issym() or member.islnk():
                target = Path(member.linkname)
                assert not target.is_absolute(), "Archive must not create an external link"
                resolved = (
                    os.path.normpath(str(path.parent / target)) if member.issym() else os.path.normpath(str(target))
                )
                assert not resolved.startswith("../"), "Archive link escapes the sealed source tree"
    assert process.wait(timeout=30) == 0


def seed_caches(source, workspace):
    """Copy one preexisting T3K seed into independent writable leg directories."""
    assert source.is_dir() and not source.is_symlink(), "Missing regular preexisting T3K seed"
    files = sorted(source.rglob("*"))
    assert files and any(p.suffix == ".tensorbin" for p in files), "No preexisting tensor cache seed"
    identities = {}
    for file in files:
        info = file.lstat()
        assert stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), "Unsealed cache link or special file"
        if file.is_file():
            identities[str(file.relative_to(source))] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    total = sum(value[2] for value in identities.values())
    free = shutil.disk_usage(workspace).free
    assert free >= 2 * total + 32 * 1024**3, "Insufficient space for two seeds and bounded cache growth"
    destinations = [
        workspace / "task-cache" / leg / "meta-llama--Llama-3.1-8B-Instruct/T3K" for leg in ["baseline", "candidate"]
    ]
    for directory in destinations:
        directory.mkdir(parents=True, exist_ok=False)
    records = []
    for name, identity in identities.items():
        file = source / name
        targets = [directory / name for directory in destinations]
        expected = sha(file)
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
            assert sha(target) == expected, "Seed copy differs from preexisting bytes"
        info = file.stat()
        assert (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) == identity, "Seed changed during copy"
        records.append({"path": name, "bytes": identity[2], "sha256": expected})
    current = {
        str(p.relative_to(source)): (p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
        for p in source.rglob("*")
        if p.is_file()
    }
    assert current == identities, "Preexisting seed changed during preparation"
    write(
        workspace / "evidence/cache-seed.json",
        {
            "source": str(source),
            "runner_name": os.environ["RUNNER_NAME"],
            "shared_mount_read_only": True,
            "two_independent_copies": True,
            "initial_free_bytes": free,
            "seed_bytes": total,
            "files": records,
        },
    )


def completed_assertions(code, output, proof):
    receipt = json.loads((output / "assertion-provenance.json").read_text())
    assert receipt["sealed_payload_proof"] == proof
    assert receipt["collected"] == proof["test_names"]
    assert receipt["exit_status"] == code and code in [0, 1]
    outcomes = receipt["outcomes"]
    assert set(outcomes) == set(proof["test_names"])
    assert all(
        row["setup"] == "passed" and row["teardown"] == "passed" and row["call"] in ["passed", "failed"]
        for row in outcomes.values()
    ), "Incomplete original assertion outcomes"
    assert (code == 1) == any(row["call"] == "failed" for row in outcomes.values())


def leg(protocol, name):
    started = time.monotonic()
    root = Path("/work")
    output = Path("/pair/evidence") / name
    output.mkdir(parents=True, exist_ok=True)
    quiet(output / "initial-quiet.json")
    manifest = json.loads(Path("/pair/control/.github/scripts/utils/formatter_pair_manifest.json").read_text())
    row = manifest["protocols"][protocol]
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        == {"baseline": BASELINE, "candidate": CANDIDATE}[name]
    )
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/pair/plugin", text=True).strip() == PLUGIN
    for artifact, member in manifest["native_members"].items():
        file = Path("/pair/artifacts") / artifact / member["path"]
        assert sha(file) == member["sha256"]
    native = Path("/pair/artifacts/native") / manifest["native_members"]["native"]["path"]
    wheel = Path("/pair/artifacts/wheel") / manifest["native_members"]["wheel"]["path"]
    validate_tar(native)
    subprocess.run(["tar", "--zstd", "-xf", str(native)], cwd=root, check=True, timeout=120)
    env = os.environ.copy()
    env.update(
        {
            "TT_METAL_HOME": "/work",
            "PYTHONPATH": "/work:/pair/plugin/src",
            "LD_LIBRARY_PATH": "/work/build/lib",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_CACHE": "/mnt/MLPerf/huggingface/hub",
            "TT_CACHE_HOME": "/task-cache",
            "MESH_DEVICE": "T3K",
            "TT_LLAMA_TEXT_VER": "tt_transformers",
            "HF_MODEL": row["model"],
            "TT_CACHE_PATH": "/task-cache/meta-llama--Llama-3.1-8B-Instruct",
            "VLLM_RPC_TIMEOUT": "300000",
            "LOGURU_LEVEL": "INFO",
            "TT_METAL_OPERATION_TIMEOUT_SECONDS": "5",
            "FORMATTER_PAIR_LEG": name,
            "FORMATTER_STRUCTURED_SOURCE_SHA256": manifest["structured_source_sha256"],
        }
    )
    assert not any(
        key.startswith("TT_METAL_WATCHER") for key in env
    ), "Preserve the original serving watcher environment"
    env["FORMATTER_PLAIN_SOURCE_SHA256"] = manifest["plain_source_sha256"]
    ref = Path(env["HF_HUB_CACHE"]) / "models--meta-llama--Llama-3.1-8B-Instruct/refs/main"
    assert ref.read_text().strip() == "0e9e39f249a16976918f6564b8830bc894c89659"
    setup = f"set -euo pipefail\nuv pip install {shlex.quote(str(wheel))}\ncd /pair/plugin\nsource docs/install-vllm-tt.sh\n"
    assert run(["bash", "-c", setup], output / "install.log", env, root, 1200) == 0
    packages = subprocess.check_output(["uv", "pip", "freeze"], env=env, text=True)
    (output / "packages.txt").write_text(packages)
    assert (
        subprocess.check_output(
            ["python3", "-c", "import importlib.metadata; print(importlib.metadata.version('vllm'))"],
            env=env,
            text=True,
        ).strip()
        == "0.26.0+empty"
    )
    if name == "candidate":
        assert (
            packages == Path("/pair/evidence/baseline/packages.txt").read_text()
        ), "Dependency drift invalidates a paired control"
    native_identity = subprocess.check_output(
        [
            "python3",
            "-c",
            "import hashlib, pathlib, importlib.metadata, ttnn; p=pathlib.Path(ttnn._ttnn.__file__); print(importlib.metadata.version('ttnn')); print(hashlib.sha256(p.read_bytes()).hexdigest())",
        ],
        cwd=root,
        env=env,
        text=True,
    )
    (output / "binding.txt").write_text(native_identity)
    if name == "candidate":
        assert native_identity == Path("/pair/evidence/baseline/binding.txt").read_text()
    quiet(output / "preflight.json")
    (root / "output").symlink_to(output)
    # Preserve a filesystem sentinel on the original five-second native timeout.
    # No imported setup/triage/reset path executes in this diagnostic.
    sentinel = output / "native-hang-sentinel"
    env["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"] = f"touch {sentinel}"
    server_log = output / "server.log"
    args = [
        "python3",
        "/pair/plugin/examples/server_example_tt.py",
        "--model",
        row["model"],
        *shlex.split(row["additional-server-args"]),
        "--additional-config",
        json.dumps({"tt": json.loads(row["tt-config"])}),
    ]
    write(
        output / "provenance.json",
        {
            "python_source": BASELINE if name == "baseline" else CANDIDATE,
            "native_source": CANDIDATE,
            "same_native_wheel": True,
            "server_args": args,
            "operation_timeout_seconds": 5,
            "phase_limits_minutes": manifest["limits"][protocol],
            "diagnostic_only": True,
        },
    )
    failed = False
    with server_log.open("w") as stream:
        server = subprocess.Popen(
            args, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        original = process_identity(server.pid)
        assert original and original["session"] == server.pid
        tracked = {server.pid: original}
        try:
            deadline = min(time.monotonic() + row["server-timeout"] * 60, started + 45 * 60)
            while True:
                track_server(server, original, tracked)
                assert server.poll() is None, "Server exited before readiness"
                assert not HANG.search(server_log.read_text(errors="replace")), "Native failure: stop pair"
                assert time.monotonic() < deadline, "Server readiness deadline: stop pair"
                try:
                    with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(20)
            phases = manifest["commands"][protocol]
            for index, phase in enumerate(phases):
                track_server(server, original, tracked)
                assert time.monotonic() - started < 45 * 60, "Original per-leg deadline"
                env["FORMATTER_PAIR_FIXTURE"] = "/pair/evidence/fixtures/" + protocol + "-" + phase["fixture"] + ".json"
                env["FILE_SERVER_LOG"] = str(server_log)
                code = run(
                    phase["argv"],
                    output / f"phase-{index}.log",
                    env,
                    root,
                    min(phase["seconds"], 45 * 60 - (time.monotonic() - started)),
                )
                write(output / f"phase-{index}.json", {"exit_code": code, "original_assertions": True})
                if phase["fixture"] == "assertions":
                    completed_assertions(code, output, manifest["assertion_payload_proof"])
                assert not sentinel.exists(), "Native timeout observer fired: stop pair"
                assert not HANG.search(server_log.read_text(errors="replace")), "Native failure: stop pair"
                assert server.poll() is None, "Server exited: stop pair"
                if code:
                    assert phase["kind"] == "correctness", "Incomplete original benchmark: stop pair"
                    assert code == 1, "Observer/collection error is not a completed correctness comparison"
                    failed = True
                    break  # Keep stock assertion ordering; do not run later tests.
        finally:
            # Only this job's launched process group; never reset devices or touch
            # another task's server. Archive output before ordinary owned cleanup.
            closed = close_server(server, original, tracked, output / "server-cleanup.json")
            jit = Path.home() / ".cache/tt-metal-cache"
            copied = []
            if jit.exists():
                for file in jit.rglob("*"):
                    if file.is_file() and file.suffix in {".elf", ".map", ".ld"}:
                        target = output / "jit" / file.relative_to(jit)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(file, target)
                        copied.append({"file": str(target.relative_to(output)), "sha256": sha(target)})
            write(output / "jit-files.json", copied)
            quiet(output / "post-cleanup.json")
            assert closed, "Escalated server cleanup: candidate is prohibited"
    return 1 if failed else 0


def host_inner(protocol):
    assert protocol == "chunked", "Only the original chunked 8+3 pair is currently reviewed for launch"
    workspace = Path.cwd()
    evidence = workspace / "evidence"
    (evidence / "fixtures").mkdir(parents=True, exist_ok=False)
    control = workspace / "control/.github/scripts/utils/formatter_serial_control.py"
    assert (
        subprocess.check_output(
            ["git", "diff", "--name-only", BASELINE, CANDIDATE], cwd=workspace / "candidate", text=True
        ).splitlines()
        == json.loads((control.parent / "formatter_pair_manifest.json").read_text())["source_diff"]
    )
    tree_proof = {}
    for directory in ["tt_metal", "ttnn"]:
        trees = [
            subprocess.check_output(
                ["git", "rev-parse", f"{revision}:{directory}"], cwd=workspace / "candidate", text=True
            ).strip()
            for revision in [BASELINE, CANDIDATE]
        ]
        assert trees[0] == trees[1]
        tree_proof[directory] = trees[0]
    write(
        evidence / "source-boundary.json",
        {
            "baseline": BASELINE,
            "candidate": CANDIDATE,
            "identical_native_trees": tree_proof,
            "ordered_device_state_limit": "Baseline precedes candidate; native device state is not reset. A difference is a diagnostic observation, not sole proof of formatter causality or a timing result.",
        },
    )
    results = {}
    assignment = json.loads((evidence / "ci-assignment-and-artifacts.json").read_text())
    deadline = datetime.fromisoformat(assignment["started_at"].replace("Z", "+00:00")).timestamp() + 110 * 60
    seed_caches(Path("/mnt/MLPerf/huggingface/tt_cache/meta-llama--Llama-3.1-8B-Instruct/T3K"), workspace)
    assert (
        deadline - time.time() >= 97 * 60
    ), "Preparation consumed the paired budget; preserve ten-minute upload margin"
    containers = []
    for name in ["baseline", "candidate"]:
        command = [
            "docker",
            "create",
            "--name",
            f"formatter-pair-{os.environ['GITHUB_RUN_ID']}-{name}",
            "--label",
            f"codex.formatter-pair.run={os.environ['GITHUB_RUN_ID']}",
            "--label",
            f"codex.formatter-pair.leg={name}",
            "--label",
            f"codex.formatter-pair.job={assignment['job_id']}",
            "--pid=host",
            "--cap-add=SYS_PTRACE",
            "--device=/dev/tenstorrent",
            "-v",
            "/dev/hugepages-1G:/dev/hugepages-1G",
            "-v",
            "/mnt/MLPerf:/mnt/MLPerf:ro",
            "-v",
            f"{workspace}:/pair",
            "-v",
            f"{workspace / name}:/work",
            "-v",
            f"{workspace / 'task-cache' / name}:/task-cache",
            "-w",
            "/work",
            "-e",
            f"RUNNER_NAME={os.environ['RUNNER_NAME']}",
            "-e",
            f"GITHUB_RUN_ID={os.environ['GITHUB_RUN_ID']}",
            IMAGE,
            "python3",
            "/pair/control/.github/scripts/utils/formatter_serial_control.py",
            "leg",
            protocol,
            name,
        ]
        containers.append({"id": None, "name": command[command.index("--name") + 1], "leg": name})
        write(evidence / "created-containers.json", containers)
        container = subprocess.check_output(command, text=True, timeout=60).strip()
        assert re.fullmatch(r"[0-9a-f]{64}", container)
        containers[-1]["id"] = container
        write(evidence / "created-containers.json", containers)
        try:
            result = subprocess.run(
                ["docker", "start", "--attach", container], timeout=min(47 * 60, deadline - time.time())
            )
            state = json.loads(
                subprocess.check_output(
                    ["docker", "inspect", "--format", "{{json .State}}", container], text=True, timeout=15
                )
            )
            assert (
                state["Status"] == "exited" and not state["OOMKilled"] and state["ExitCode"] in [0, 1]
            ), "Unsafe or incomplete container exit"
            assert result.returncode == state["ExitCode"]
            results[name] = state["ExitCode"]
        finally:
            cleanup_containers(workspace)
        write(evidence / "pair-results.json", results)
        assert result.returncode in [0, 1], "Incomplete/unsafe diagnostic leg: no candidate or retry"
    return max(results.values())


def host(protocol):
    try:
        return host_inner(protocol)
    finally:
        cleanup_containers(Path.cwd())


def cleanup_containers(workspace):
    path = workspace / "evidence/created-containers.json"
    if not path.exists():
        return
    assignment = json.loads((workspace / "evidence/ci-assignment-and-artifacts.json").read_text())
    for item in json.loads(path.read_text()):
        assert item["name"] == f"formatter-pair-{os.environ['GITHUB_RUN_ID']}-{item['leg']}"
        container = item["id"] or item["name"]
        assert item["id"] is None or re.fullmatch(r"[0-9a-f]{64}", container)
        format_string = '{{.Id}} {{index .Config.Labels "codex.formatter-pair.run"}} {{index .Config.Labels "codex.formatter-pair.leg"}} {{index .Config.Labels "codex.formatter-pair.job"}} {{.State.Running}}'
        receipt = subprocess.run(
            ["docker", "inspect", "--format", format_string, container], capture_output=True, text=True, timeout=15
        )
        if receipt.returncode:
            assert "No such" in receipt.stderr, "Cannot prove exact container identity"
            continue
        identity, run_id, leg_name, job_id, running = receipt.stdout.strip().split()
        assert (
            (item["id"] is None or identity == container)
            and run_id == os.environ["GITHUB_RUN_ID"]
            and leg_name == item["leg"]
            and job_id == str(assignment["job_id"])
        )
        container = identity
        if running == "true":
            subprocess.run(["docker", "stop", "--time", "45", container], check=True, timeout=60)
        subprocess.run(["docker", "rm", container], check=True, timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["host", "leg", "cleanup"])
    parser.add_argument("protocol", nargs="?", choices=["chunked", "structured"])
    parser.add_argument("leg", nargs="?", choices=["baseline", "candidate"])
    arguments = parser.parse_args()

    def interrupted(signum, frame):
        raise RuntimeError("CI cancellation: preserve and clean only exact owned identities")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if arguments.mode == "cleanup":
            cleanup_containers(Path.cwd())
            raise SystemExit(0)
        raise SystemExit(
            host(arguments.protocol) if arguments.mode == "host" else leg(arguments.protocol, arguments.leg)
        )
    except Exception as error:
        # All exception messages in this controller are public-source diagnostics.
        print(f"Stopped diagnostic: {type(error).__name__}: {error}")
        raise SystemExit(2)
