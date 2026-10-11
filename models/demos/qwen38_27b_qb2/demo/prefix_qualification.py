# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Freeze, seal, and explicitly launch bounded complete-prefix hardware gates.

freeze/seal never open devices or start services. Native and HTTP phases are
separate explicit commands; generated systemd recipes are not automatically run.
No reset, recovery, native installation, weight download or AgentX is performed.
"""

import argparse
import ast
import fcntl
import hashlib
import json
import os
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

MODEL = Path("models/demos/qwen38_27b_qb2")
MODEL_REVISION = "d1019c0dc125913a99ba82938d06d5d662e17f7d"
PLUGIN_REVISION = "13b9777876dc08b268dfc2f627571496484d0f5a"
WEIGHT_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
NATIVE_REVISION = "a08819ddbe23077f8037d3802303939064868ff6"
POLICY = MODEL / "config/precision_single_step_compact_gdn_bfp8_all.json"


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=30).strip()


def verify_files(root, files):
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Frozen file escapes the task bundle")
        if sha(path) != expected:
            raise ValueError("Frozen file changed: " + name)


def freeze(args):
    metal, plugin, output = args.metal, args.plugin, args.bundle
    if git(plugin, "rev-parse", "HEAD") != PLUGIN_REVISION:
        raise ValueError("Require the paired pinned prefix plugin")
    for root in (metal, plugin):
        if git(root, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Freeze only clean committed worktrees")
    git(metal, "merge-base", "--is-ancestor", MODEL_REVISION, "HEAD")
    if git(metal, "diff", MODEL_REVISION, "HEAD", "--", str(MODEL / "tt"), str(MODEL / "config")):
        raise ValueError("Qualification preparation must retain d1019c0d model/precision bytes")
    output.mkdir(parents=False, exist_ok=False)
    manifest = {}
    for root, dest, paths in (
        (
            metal,
            "source",
            [str(MODEL / p) for p in ("tt", "config", "tests", "demo")]
            + [str(MODEL / "__init__.py"), str(MODEL / "vllm_metadata.json"), "conftest.py", "pytest.ini"],
        ),
        (plugin, "plugin", ["src", "pyproject.toml"]),
    ):
        for name in git(root, "ls-files", "--", *paths).splitlines():
            if Path(name).suffix not in (".py", ".cpp", ".h", ".hpp", ".json", ".ini", ".toml"):
                continue
            target = output / dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
            manifest[str(target.relative_to(output))] = sha(target)
    plan = dict(
        state="frozen_not_launched",
        model_revision=MODEL_REVISION,
        preparation_revision=git(metal, "rev-parse", "HEAD"),
        plugin_revision=PLUGIN_REVISION,
        native_revision=NATIVE_REVISION,
        weight_revision=WEIGHT_REVISION,
        task=str(args.task),
        weights=str(args.weights),
        destination=str(args.destination),
        mode="batched",
        device_ids=[0, 4, 12, 8],
        precision_sha256=sha(output / "source" / POLICY),
        files=manifest,
        created_at=time.time(),
        agentx_launched=False,
    )
    write(output / "bundle.json", plan)
    print(json.dumps({"bundle": str(output), "files": len(manifest), "launched": False}))


def validate_bundle(bundle, *, sealed=True):
    plan = json.loads((bundle / "bundle.json").read_text())
    if (
        plan["mode"] != "batched"
        or plan["model_revision"] != MODEL_REVISION
        or plan["plugin_revision"] != PLUGIN_REVISION
    ):
        raise ValueError("Wrong model/plugin/transfer-mode bundle")
    if plan["native_revision"] != NATIVE_REVISION or plan["weight_revision"] != WEIGHT_REVISION:
        raise ValueError("Wrong native runtime or checkpoint revision")
    if plan["device_ids"] != [0, 4, 12, 8]:
        raise ValueError("Wrong TP4 allocation")
    verify_files(bundle, plan["files"])
    if sealed:
        seal = json.loads((bundle / "seal.json").read_text())
        if seal["bundle_sha256"] != sha(bundle / "bundle.json") or seal["prompts_sha256"] != sha(
            bundle / "prompts.json"
        ):
            raise ValueError("Bundle or prompts changed after host sealing")
        for name, expected in seal["native_libraries"].items():
            if sha(Path(plan["task"]) / name) != expected:
                raise ValueError("Native library changed after host sealing: " + name)
        sys.path.insert(0, str(bundle / "source"))
        from models.demos.qwen38_27b_qb2.tt.prefix_backend import weight_metadata

        if weight_metadata(plan["weights"]) != seal["weight_metadata"]:
            raise ValueError("Pinned checkpoint metadata changed after host sealing")
    return plan


def seal(args):
    bundle = args.bundle.resolve()
    plan = validate_bundle(bundle, sealed=False)
    if bundle != Path(plan["destination"]) or (bundle / "seal.json").exists():
        raise ValueError("Seal a fresh bundle only at its explicit host destination")
    task = Path(plan["task"])
    if git(task / "metal", "rev-parse", "HEAD") != NATIVE_REVISION:
        raise ValueError("Unexpected native runtime source revision")
    sys.path.insert(0, str(bundle / "source"))
    from models.demos.qwen38_27b_qb2.demo.prefix_http_probe import make_prompts
    from models.demos.qwen38_27b_qb2.tt.prefix_backend import weight_metadata

    libraries = sorted((task / "metal-install/lib").glob("*.so"))
    if not any(path.name == "libtt_metal.so" for path in libraries):
        raise ValueError("Native Metal shared library is missing")
    write(bundle / "prompts.json", make_prompts(plan["weights"]))
    metadata = dict(
        bundle_sha256=sha(bundle / "bundle.json"),
        prompts_sha256=sha(bundle / "prompts.json"),
        weight_metadata=weight_metadata(plan["weights"]),
        native_libraries={str(path.relative_to(task)): sha(path) for path in libraries},
        sealed_at=time.time(),
        hardware_launched=False,
    )
    write(bundle / "seal.json", metadata)
    controller = bundle / "source" / MODEL / "demo/prefix_qualification.py"
    import shlex

    for phase, limit in (("native", 7200), ("http", 9000)):
        unit = f"qwen38-prefix-{phase}-{bundle.name}"
        command = [
            "systemd-run",
            "--user",
            "--unit=" + unit,
            "--property=RuntimeMaxSec=" + str(limit),
            "--property=TimeoutStopSec=180",
            "--property=KillMode=control-group",
            "--property=MemoryHigh=192G",
            "--property=MemoryMax=256G",
            "--property=LimitFSIZE=2G",
            "--property=StandardOutput=append:" + str(bundle / (phase + ".log")),
            "--property=StandardError=append:" + str(bundle / (phase + ".log")),
            str(task / "serving_env/bin/python"),
            str(controller),
            phase,
            "--bundle",
            str(bundle),
        ]
        recipe = bundle / ("start-" + phase + ".sh")
        recipe.write_text("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n")
        recipe.chmod(0o700)
    print(json.dumps({"sealed": str(bundle), "hardware_launched": False}))


@contextmanager
def device_lease():
    with Path("/tmp/tt-device.lock").open("a") as lock:
        # An explicitly started job waits at most ten minutes. It never declares
        # the device free just because another unit's status/log is quiet.
        deadline = time.monotonic() + 600
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Device lock still owned after ten minutes")
                time.sleep(2)
        if Path("/tmp/tt-device.dirty").exists():
            raise RuntimeError("Device is marked dirty; qualification does not reset hardware")
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop(process):
    """Reap the parent and every member of its exclusive process group."""
    if process is None:
        return True
    process.poll()
    if group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        process.poll()
        if not group_alive(process.pid):
            process.wait(timeout=1)
            return True
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=30)
    return False


def logged_environment(env):
    # Never serialize inherited credentials or arbitrary host environment.
    explicit = {
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "MODEL_WEIGHTS_DIR",
        "ARCH_NAME",
        "OMP_NUM_THREADS",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "MESH_DEVICE",
        "EXTRA_MODELS_DIR",
        "HF_HUB_OFFLINE",
        "VLLM_NO_USAGE_STATS",
        "MPLCONFIGDIR",
        "TT_VISIBLE_DEVICES",
    }
    return {key: value for key, value in env.items() if key in explicit or key.startswith(("QWEN_", "TT_METAL_"))}


def check_budget(bundle):
    if shutil.disk_usage(bundle).free < 12 * 2**30:
        raise OSError("Qualification requires at least 12 GiB of free host disk")
    # Follow no directory symlinks; only this task's source/log/cache files count.
    if sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file()) > 8 * 2**30:
        raise OSError("Qualification artifacts exceeded the 8 GiB task budget")


def limit_files():
    # Applied only by explicitly launched native/http phases and inherited by
    # their children. A pathological log cannot fill the host disk.
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    bound = 2 * 2**30 if hard == resource.RLIM_INFINITY else min(hard, 2 * 2**30)
    resource.setrlimit(resource.RLIMIT_FSIZE, (bound, hard))


def run_child(command, *, env, cwd, log, timeout, bundle):
    with log.open("w") as output:
        child = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + timeout
            while child.poll() is None:
                check_budget(bundle)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native stage exceeded its explicit time budget")
                time.sleep(2)
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
        finally:
            if not stop(child):
                raise RuntimeError("Child process group required forced termination")


def runtime_env(bundle, plan):
    from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment

    task, source = Path(plan["task"]), bundle / "source"
    env = environment(task, source, Path(plan["weights"]))
    for key in ("TT_VISIBLE_DEVICES", "TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_METAL_DISABLE_SFPLOADMACRO"):
        env.pop(key, None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        QWEN_DECODE_BUCKETS="1",
        QWEN_PREFIX_BATCHED_TRANSFER="1",
        QWEN_PRECISION_CONFIG=str(source / POLICY),
        TT_METAL_OPERATION_TIMEOUT_SECONDS="15",
    )
    return env


def native_stage(bundle, plan, output, layers=None):
    output.mkdir()
    env = runtime_env(bundle, plan)
    source = bundle / "source"
    receipt = output / "result.json"
    if layers is None:
        env.update(QWEN_PREFIX_TRANSFER="1", QWEN_PREFIX_TRANSFER_RECEIPT=str(receipt))
        test, timeout = "test_prefix_transfer.py", 900
    else:
        env.update(
            QWEN_PREFIX_CONTINUATION="1", QWEN_PREFIX_LAYERS=str(layers), QWEN_PREFIX_CONTINUATION_RECEIPT=str(receipt)
        )
        test, timeout = "test_prefix_continuation.py", 1200 if layers == 4 else 3600
    command = [
        str(Path(plan["task"]) / "python_env/bin/python"),
        "-m",
        "pytest",
        str(MODEL / "tests" / test),
        "-q",
        "-s",
        "--timeout=" + str(timeout - 120),
        "--junitxml=" + str(output / "hardware.xml"),
    ]
    write(
        output / "invocation.json",
        dict(command=command, environment=logged_environment(env), bundle_sha256=sha(bundle / "bundle.json")),
    )
    run_child(command, env=env, cwd=source, log=output / "run.log", timeout=timeout, bundle=bundle)
    measured = json.loads(receipt.read_text())
    validate_native_result(measured, layers, plan, bundle)
    return dict(path=str(receipt.relative_to(bundle)), sha256=sha(receipt), layers=layers)


def validate_native_result(measured, layers, plan, bundle):
    if not all(measured.get(key) is True for key in ("passed", "cleanup_completed", "batched_transfer")):
        raise ValueError("Native receipt lacks a clean successful BATCHED gate; serial receipts cannot authorize it")
    if measured.get("device_ids") != plan["device_ids"]:
        raise ValueError("Native receipt used a different physical TP4")
    if layers is None:
        if not all(
            measured.get(key) is True
            for key in ("all_rank_bytes_exact", "neighbours_unchanged", "corrupt_restore_unpublished")
        ):
            raise ValueError("Packed-transfer receipt lacks complete byte/isolation evidence")
    else:
        from models.demos.qwen38_27b_qb2.demo.run_prefix_validation import validate_stage

        validate_stage(measured, layers, batched=True)
        if measured["precision"] != json.loads((bundle / "source" / POLICY).read_text()):
            raise ValueError("Native continuation precision differs from the frozen BFP8/FP32 policy")
        source = bundle / "source" / MODEL
        expected = {
            str(path.relative_to(source)): sha(path)
            for path in (source / "tt").rglob("*")
            if path.suffix in (".py", ".cpp", ".h", ".hpp")
        }
        expected["config/precision.json"] = sha(source / "config/precision.json")
        expected["effective_precision_override"] = sha(bundle / "source" / POLICY)
        if measured["source_sha256"] != expected:
            raise ValueError("Native continuation used different or incomplete source/precision identity")


def native(args):
    bundle = args.bundle.resolve()
    plan = validate_bundle(bundle)
    output = bundle / "native"
    output.mkdir()
    limit_files()
    check_budget(bundle)
    report = dict(passed=False, state="waiting_for_lock", bundle_sha256=sha(bundle / "bundle.json"), stages=[])
    write(output / "queue.json", report)
    acquired = False
    try:
        with device_lease():
            acquired = True
            for name, layers in (("transfer", None), ("layers-4", 4), ("layers-64", 64)):
                validate_bundle(bundle)
                report.update(state=name)
                write(output / "queue.json", report)
                report["stages"].append(native_stage(bundle, plan, output / name, layers))
            report.update(state="completed", passed=True, cleanup_completed=True)
    except BaseException as error:
        report.update(state="failed", detail=repr(error), cleanup_completed=False)
        if acquired:
            Path("/tmp/tt-device.dirty").touch()
        raise
    finally:
        write(output / "queue.json", report)


def require_native(bundle, plan):
    report = json.loads((bundle / "native/queue.json").read_text())
    if (
        not report.get("passed")
        or not report.get("cleanup_completed")
        or report["bundle_sha256"] != sha(bundle / "bundle.json")
    ):
        raise ValueError("HTTP serving requires the same frozen bundle's clean native qualification")
    if [stage["layers"] for stage in report["stages"]] != [None, 4, 64]:
        raise ValueError("Native qualification must contain fresh transfer, four-layer and full64 gates")
    for stage in report["stages"]:
        expected_path = (
            "native/" + ("transfer" if stage["layers"] is None else "layers-" + str(stage["layers"])) + "/result.json"
        )
        if stage["path"] != expected_path:
            raise ValueError("Native gate must use its fresh task-owned receipt")
        path = bundle / stage["path"]
        if sha(path) != stage["sha256"]:
            raise ValueError("Native receipt changed after qualification")
        validate_native_result(json.loads(path.read_text()), stage["layers"], plan, bundle)


def validate_serving_precision(log, expected):
    policies = [ast.literal_eval(value) for value in re.findall(r"Qwen3\.8 vLLM precision: (\{[^\n]*\})", log)]
    if not policies or any(policy != expected for policy in policies):
        raise ValueError("Serving startup did not confirm the frozen BFP8/FP32 precision policy")
    return dict(precision_confirmations=len(policies), policy=expected)


def http(args):
    bundle = args.bundle.resolve()
    plan = validate_bundle(bundle)
    require_native(bundle, plan)
    output = bundle / "http"
    output.mkdir()
    limit_files()
    check_budget(bundle)
    from models.demos.qwen38_27b_qb2.demo.prefix_http_probe import exercise, restart_probe, wait_ready
    from models.demos.qwen38_27b_qb2.tt.prefix_storage import AtomicDirectoryStore

    precision = json.loads((bundle / "source" / POLICY).read_text())
    cache = output / "checkpoint-store"
    AtomicDirectoryStore(cache, max_bytes=2**31, create=True)
    settings = dict(
        root=str(cache),
        max_bytes=2**31,
        namespace="isolated-prefix-qualification",
        weights_path=plan["weights"],
        weights_revision=plan["weight_revision"],
        precision=precision,
        capture_interval=4096,
        batched_transfer=True,
    )
    transfer = dict(
        kv_connector="TTCompletePrefixConnector",
        kv_connector_module_path="vllm_tt_plugin.hybrid_prefix",
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={"backend_config": settings},
    )
    write(output / "kv-transfer-config.json", transfer)
    env = runtime_env(bundle, plan)
    env.update(
        PYTHONPATH=str(bundle / "plugin/src") + ":" + env["PYTHONPATH"],
        MESH_DEVICE="(1,4)",
        TT_VISIBLE_DEVICES=",".join(map(str, plan["device_ids"])),
        EXTRA_MODELS_DIR=str(bundle / "source" / MODEL.parent),
        QWEN_COMPLETE_PREFIX_EXPERIMENT="1",
        QWEN_VLLM_HOST_COMPATIBILITY="all",
        HF_HUB_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES="0",
    )
    port = 18086
    command = [
        str(Path(plan["task"]) / "serving_env/bin/python"),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        plan["weights"],
        "--served-model-name",
        "prefix-qualification",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--hf-overrides",
        json.dumps({"architectures": ["TTQwen38ForCausalLM"]}),
        "--block-size",
        "32",
        "--max-num-seqs",
        "16",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "4096",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--long-prefill-token-threshold",
        "4096",
        "--disable-hybrid-kv-cache-manager",
        "--no-enable-log-requests",
        "--no-enable-log-outputs",
        "--kv-transfer-config",
        json.dumps(transfer),
        "--additional-config",
        json.dumps(
            {
                "tt": {
                    "fabric_config": "FABRIC_1D",
                    "fabric_max_packet_payload_size_bytes": 8192,
                    "l1_small_size": 24576,
                    "trace_mode": "decode_only",
                    "trace_region_size": 200000000,
                }
            }
        ),
    ]
    write(
        output / "launch.json",
        dict(command=command, environment=logged_environment(env), bundle_sha256=sha(bundle / "bundle.json")),
    )
    report = dict(state="waiting_for_lock", passed=False, bundle_sha256=sha(bundle / "bundle.json"), phase_reports=[])
    write(output / "queue.json", report)
    endpoint = "http://127.0.0.1:" + str(port)
    acquired = False
    try:
        with device_lease():
            acquired = True
            for phase in ("first", "restart"):
                validate_bundle(bundle)
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", port))
                with (output / (phase + "-server.log")).open("w") as log:
                    server = subprocess.Popen(
                        command,
                        cwd=bundle / "source",
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        report.update(state=phase + "_loading")
                        write(output / "queue.json", report)
                        wait_ready(endpoint, server, timeout=1800, check=lambda: check_budget(bundle))
                        check_budget(bundle)
                        confirmed = validate_serving_precision(
                            (output / (phase + "-server.log")).read_text(), precision
                        )
                        write(output / (phase + "-precision.json"), confirmed)
                        result = (exercise if phase == "first" else restart_probe)(endpoint, bundle, output)
                        check_budget(bundle)
                        report["phase_reports"].append(result)
                    finally:
                        if not stop(server):
                            raise RuntimeError("Serving required forced termination; cleanup is not qualified")
            # Opening the same TP4 again verifies post-serving device health;
            # this is a fresh physical test, never a reused pre-serving receipt.
            report["cleanup_probe"] = native_stage(bundle, plan, output / "cleanup-transfer")
            report.update(state="completed", passed=True, cleanup_completed=True)
    except BaseException as error:
        report.update(state="failed", detail=repr(error), cleanup_completed=False)
        if acquired:
            Path("/tmp/tt-device.dirty").touch()
        raise
    finally:
        write(output / "queue.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("freeze", "seal", "native", "http"):
        child = subparsers.add_parser(phase)
        child.add_argument("--bundle", type=Path, required=True)
        if phase == "freeze":
            for name in ("metal", "plugin", "task", "weights", "destination"):
                child.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()

    def interrupted(signum, frame):
        raise InterruptedError(f"Qualification received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    globals()[arguments.phase](arguments)
