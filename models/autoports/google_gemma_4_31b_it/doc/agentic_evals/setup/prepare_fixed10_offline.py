import copy
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess

root = pathlib.Path.cwd()
setup = root / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup"
out = setup / "fixed10-offline-verifier"
out.mkdir(exist_ok=True)
src = setup / "swe-tasks"
dest = setup / "swe-tasks-offline-verifier"
dest.mkdir(exist_ok=True)
manifest = json.loads((setup / "swe10_tasks.json").read_text())
template = (setup / "offline-verifier-build/test.sh.template").read_text()
parser = template.split("cat > parser.py <<EOF\n", 1)[1].split("\nEOF", 1)[0]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
records = []
for row in manifest["tasks"]:
    task = row["task"]
    base = "v5-swe-prepared:" + task
    tag = "v5-swe-offline-verifier:" + task
    log = out / (task + ".build.log")
    cmd = [
        "docker",
        "build",
        "--network",
        "none",
        "--pull=false",
        "--build-arg",
        "BASE_IMAGE=" + base,
        "-t",
        tag,
        str(root / "bringup/artifacts/swe-verifier-image-context"),
    ]
    if task == "pylint-dev__pylint-6903":
        cmd = ["docker", "tag", "v5-swe-offline-verifier-test:" + task, tag]
    with log.open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=240)
    images = json.loads(subprocess.check_output(["docker", "image", "inspect", base, tag]))
    assert images[1]["RootFS"]["Layers"][: len(images[0]["RootFS"]["Layers"])] == images[0]["RootFS"]["Layers"]
    assert not (dest / task).exists()
    shutil.copytree(src / task, dest / task)
    original = (src / task / "tests/test.sh").read_text()
    prefix, rest = original.split("cat > parser.py <<EOF\n", 1)
    oldparser, suffix = rest.split("\nEOF", 1)
    fixed = prefix + "cat > parser.py <<EOF\n" + parser + "\nEOF" + suffix
    (dest / task / "tests/test.sh").write_text(fixed)
    toml = (src / task / "task.toml").read_text()
    assert toml.count(base) == 1
    (dest / task / "task.toml").write_text(toml.replace(base, tag))
    # Source/generated test execution outside the parser is exactly preserved.
    assert fixed.split("cat > parser.py <<EOF\n", 1)[0] == prefix
    assert fixed.split("\nEOF", 1)[1] == suffix
    changes = []
    for originalfile in (src / task).rglob("*"):
        if originalfile.is_file():
            rel = originalfile.relative_to(src / task)
            if sha(originalfile) != sha(dest / task / rel):
                changes.append(str(rel))
    assert set(changes) == {"task.toml", "tests/test.sh"}
    record = {
        "task": task,
        "base_tag": base,
        "base_image": images[0]["Id"],
        "derived_tag": tag,
        "derived_image": images[1]["Id"],
        "base_layers_preserved": True,
        "changed_task_files": changes,
        "original_task_path": str(src / task),
        "derived_task_path": str(dest / task),
        "original_test_sh_sha256": sha(src / task / "tests/test.sh"),
        "fixed_test_sh_sha256": sha(dest / task / "tests/test.sh"),
        "config_sha256": sha(src / task / "tests/config.json"),
        "instruction_sha256": sha(src / task / "instruction.md"),
        "build_command": cmd,
        "build_log": str(log),
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generic_fix_sha": "e98973d132ec705baa4f59ca62680587ef5281a9",
        "all_other_task_files_identical": True,
        "test_execution_prefix_and_suffix_identical": True,
    }
    records.append(record)
    (out / "images-and-task-copies.json").write_text(json.dumps(records, indent=2) + "\n")
    print(task, "prepared", flush=True)
config_path = setup / "swe-jobs/swe-phase2-ready_harbor_config.json"
config = json.loads(config_path.read_text())
new = copy.deepcopy(config)
new["job_name"] = "swe-fixed10-offline-verifier-PREPARED-NOT-LAUNCHED"
new["datasets"][0]["path"] = str(dest)
new["agents"][0]["env"]["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
(out / "harbor-config-template.json").write_text(json.dumps(new, indent=2) + "\n")
check = copy.deepcopy(new)
check["job_name"] = config["job_name"]
check["datasets"][0]["path"] = config["datasets"][0]["path"]
del check["agents"][0]["env"]["LITELLM_LOCAL_MODEL_COST_MAP"]
assert check == config
(out / "configuration-equivalence.json").write_text(
    json.dumps(
        {
            "source_config": str(config_path),
            "source_sha256": sha(config_path),
            "changes": [
                "job_name placeholder for fresh run",
                "datasets[0].path isolated copied tasks",
                "agents[0].env.LITELLM_LOCAL_MODEL_COST_MAP=True",
            ],
            "all_reference_limits_sampling_resources_network_preserved": True,
            "runtime_harbor_sha": "e76c762fca5267dfb36e1f900f0a7724ed72e735",
            "verifier_template_fix_sha": "e98973d132ec705baa4f59ca62680587ef5281a9",
            "status": "prepared_not_launched; select fresh job name and fresh server-log check before any future launch",
        },
        indent=2,
    )
    + "\n"
)
