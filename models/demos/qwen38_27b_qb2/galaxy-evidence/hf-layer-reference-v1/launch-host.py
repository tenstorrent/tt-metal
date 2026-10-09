import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

art = Path("/home/ttuser/qwen38-artifacts-20261007")
task = Path("/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006")
source = art / "hf-layer-source-v1"
control = art / "hf-layer-control-v1"
results = art / "hf-layer-v1"
model = source / "models/demos/qwen38_27b_qb2"
unit = "qwen38-hf-layer-v1-20261009"
predecessor = "qwen38-post-head-cpu-v1-20261009.service"
invocation = "dc7524a132964cc0a36cdeecfe8cd82c"
assert source.is_dir() and control.is_dir() and not results.exists()
assert not (control / "launch.json").exists()
props = dict(
    line.split("=", 1)
    for line in subprocess.check_output(
        ["systemctl", "--user", "show", predecessor, "-p", "InvocationID", "-p", "MainPID", "-p", "ActiveState"],
        text=True,
    ).splitlines()
    if "=" in line
)
assert props.get("InvocationID") == invocation, props
env = dict(
    os.environ,
    PYTHONPATH=f"{source}:{task}/metal:{task}/metal/tools",
    TT_METAL_HOME=str(task / "metal"),
    LD_LIBRARY_PATH=f"{task}/metal-install/lib:{task}/metal-build/lib",
    ARCH_NAME="blackhole",
    OMP_NUM_THREADS="1",
    TT_METAL_INSPECTOR_RPC="0",
)
python = str(task / "python_env/bin/python")
if (control / "preflight.log").exists():
    assert "missing 1 required positional argument: 'message'" in (control / "preflight.log").read_text()
    assert not (control / "preflight-fixture-failed.log").exists()
    (control / "preflight.log").rename(control / "preflight-fixture-failed.log")
    (control / "unit.xml").rename(control / "unit-fixture-failed.xml")
tests = [
    python,
    "-m",
    "pytest",
    str(model / "tests/unit/test_reference_comparison.py"),
    str(model / "tests/unit/test_hf_layer_followup.py"),
    str(model / "tests/test_hf_layer_reference.py"),
    "--rootdir=" + str(source),
    "-c",
    str(source / "pytest.ini"),
    "-o",
    "addopts=",
    "-q",
    "--junitxml=" + str(control / "unit.xml"),
]
with (control / "preflight.log").open("w") as log:
    subprocess.run(tests, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
    subprocess.run(
        [
            python,
            "-c",
            'import torch, ttnn; from models.demos.qwen38_27b_qb2.tt.model import Qwen38Model; print("Native model imports passed; no device opened")',
        ],
        cwd=source,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=90,
    )
print((control / "preflight.log").read_text()[-2500:], flush=True)
manifest = {
    str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in source.rglob("*")
    if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".hpp", ".json", ".sh", ".ini")
}
(control / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
command = [
    "systemd-run",
    "--user",
    "--unit=" + unit,
    "--property=RuntimeMaxSec=22h",
    "--property=TimeoutStopSec=210",
    "--property=KillMode=control-group",
    "--property=MemoryMax=160G",
    "--property=CPUQuota=800%",
    f"--property=StandardOutput=append:{control}/run.log",
    f"--property=StandardError=append:{control}/run.log",
    "--working-directory=" + str(source),
    f"--setenv=PATH={task}/python_env/bin:/usr/local/bin:/usr/bin:/bin",
    f"--setenv=PYTHONPATH={source}:{task}/metal:{task}/metal/tools",
    "--setenv=PYTHONUNBUFFERED=1",
    "--setenv=OMP_NUM_THREADS=8",
    "--setenv=TT_METAL_INSPECTOR_RPC=0",
    python,
    "-m",
    "models.demos.qwen38_27b_qb2.demo.run_hf_layer_followup",
    "--task",
    str(task),
    "--source",
    str(source),
    "--weights",
    str(art / "checkpoint-pinned-1d4bf0f2"),
    "--results",
    str(results),
    "--manifest",
    str(control / "source-manifest.json"),
    "--qualification",
    str(art / "accuracy-head-v1/native-g0/receipts/full-model.json"),
    "--reference",
    str(art / "hf-head-reference-v1"),
    "--predecessor-receipt",
    str(art / "post-head-cpu-v1/status.json"),
    "--gpqa-summary",
    str(art / "accuracy-head-v1/native-control/evaluation/gpqa/summary.json"),
    "--predecessor-unit",
    predecessor,
    "--predecessor-invocation",
    invocation,
    "--wait-timeout",
    "72000",
]
launch = dict(
    created_at=time.time(),
    command=command,
    predecessor=props,
    source_manifest_sha256=hashlib.sha256((control / "source-manifest.json").read_bytes()).hexdigest(),
    condition="Only run after CPU reference completion and head-control full GPQA below 177/198",
    scope="B1 eager TP4 diagnostic; not a score, throughput claim or long-context qualification",
    preflight_command=tests,
    survives_disconnect=True,
    resumes_after_reboot=False,
)
(control / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
subprocess.run(command, check=True)
print(json.dumps(launch), flush=True)
