import collections
import datetime
import glob
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys

import pyarrow.parquet as pq

root = pathlib.Path("/home/mvasiljevic/tt-metal-gemma4-31b-pipeline")
setup = root / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup"
out = setup / "swe65-candidate"
out.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
seed = "stage12-swe-extended-v1"
key = lambda s: hashlib.sha256((seed + "|" + s).encode()).hexdigest()
parquet = pathlib.Path(
    glob.glob("/home/mvasiljevic/.cache/huggingface/hub/datasets--princeton*/snapshots/*/data/*.parquet")[0]
)
rows = pq.read_table(parquet, columns=["instance_id", "repo", "difficulty"]).to_pylist()
byid = {r["instance_id"]: r for r in rows}
fixed = [r["task"] for r in json.loads((setup / "swe10_tasks.json").read_text())["tasks"]]
order = list(fixed)
repo_count = collections.Counter(byid[i]["repo"] for i in order)
difficulty_count = collections.Counter((byid[i]["repo"], byid[i]["difficulty"]) for i in order)
remaining = set(byid) - set(order)
while remaining:
    i = min(
        remaining,
        key=lambda i: (
            repo_count[byid[i]["repo"]],
            difficulty_count[(byid[i]["repo"], byid[i]["difficulty"])],
            key(byid[i]["repo"]),
            key(i),
        ),
    )
    order.append(i)
    remaining.remove(i)
    repo_count[byid[i]["repo"]] += 1
    difficulty_count[(byid[i]["repo"], byid[i]["difficulty"])] += 1
ids = order[:65]
tags = set(
    subprocess.check_output(["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"], text=True).splitlines()
)
entries = []
for i in ids:
    tag = "swebench/sweb.eval.x86_64." + i.replace("__", "_1776_") + ":latest"
    entries.append(
        {
            "task": i,
            **{k: byid[i][k] for k in ["repo", "difficulty"]},
            "official_image": tag,
            "image_cached": tag in tags,
            "prepared_image_cached": "v5-swe-offline-verifier:" + i in tags,
        }
    )
manifest = {
    "status": "candidate_frozen_not_admitted_not_launched",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "count": 65,
    "tasks": entries,
    "full_nested_order": order,
    "seed": seed,
    "selection_rule": "Preserve fixed10 order. Repeatedly choose remaining task minimizing current repo representation, then current within-repo difficulty representation, then SHA256(seed|repo), then SHA256(seed|instance_id). Only official instance ID/repo/difficulty metadata used; no scores, trajectories, or patch content used. Future lists are prefixes of same500 ordering.",
    "dataset_revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    "cached_parquet_sha256": sha(parquet),
    "fixed10_manifest_sha256": sha(setup / "swe10_tasks.json"),
    "projection": "Conditional existing head_bfp8 baseline:65 tasks9.862198h with25% margin; not validated for future precision policy. Agentic pass and current server/rates review still required.",
}
(out / "tasks.json").write_text(json.dumps(manifest, indent=2) + "\n")


# Deny any network before loading/generating task definitions.
def denied(*args, **kwargs):
    raise RuntimeError("Network prohibited during candidate generation")


socket.socket.connect = denied
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.chdir(setup / "dataset-loader")
sys.path.insert(0, str(root / "bringup/artifacts/harbor-offline-log-grading/adapters/swebench/src"))
from swebench.harness.test_spec.test_spec import TestSpec
from swebench_adapter.adapter import SWEBenchAdapter


# Official image key only depends on these four fields. Keep adapter generation unchanged.
def image_map(self):
    mapping = {}
    for datum in self.loader.all_records():
        spec = object.__new__(TestSpec)
        spec.instance_id = datum["instance_id"]
        spec.namespace = "swebench"
        spec.arch = "x86_64"
        spec.instance_image_tag = "latest"
        mapping[datum["instance_id"]] = spec.instance_image_key
    return mapping


SWEBenchAdapter._build_image_map = image_map
baseline = out / "tasks-original"
candidate = out / "tasks-offline-verifier"
adapter = SWEBenchAdapter(
    output_dir=baseline,
    task_ids=ids,
    template_dir=root / "bringup/artifacts/parser-protocol-harbor/adapters/swebench/src/swebench_adapter/task-template",
)
adapter.run()
adapter2 = SWEBenchAdapter(
    output_dir=candidate,
    task_ids=ids,
    template_dir=root
    / "bringup/artifacts/harbor-offline-log-grading/adapters/swebench/src/swebench_adapter/task-template",
)
adapter2.run()
checks = []
for i in ids:
    a = baseline / i
    b = candidate / i
    toml = b / "task.toml"
    text = toml.read_text()
    text = text.replace("[environment]", '[environment]\ndocker_image = "v5-swe-offline-verifier:' + i + '"')
    toml.write_text(text)
    changes = [str(p.relative_to(a)) for p in a.rglob("*") if p.is_file() and sha(p) != sha(b / p.relative_to(a))]
    assert set(changes) == {"task.toml", "tests/test.sh"}
    sa = (a / "tests/test.sh").read_text()
    sb = (b / "tests/test.sh").read_text()
    assert sa.split("cat > parser.py <<EOF\n")[0] == sb.split("cat > parser.py <<EOF\n")[0]
    assert sa.split("\nEOF", 1)[1] == sb.split("\nEOF", 1)[1]
    check = {
        "task": i,
        "changes": changes,
        "instruction_sha256": sha(a / "instruction.md"),
        "config_sha256": sha(a / "tests/config.json"),
        "test_commands_identical": True,
    }
    if i in fixed:
        check["fixed10_original_instruction_equal"] = sha(a / "instruction.md") == sha(
            setup / "swe-tasks" / i / "instruction.md"
        )
        check["fixed10_original_config_equal"] = sha(a / "tests/config.json") == sha(
            setup / "swe-tasks" / i / "tests/config.json"
        )
        check["fixed10_original_test_equal"] = sha(a / "tests/test.sh") == sha(
            setup / "swe-tasks" / i / "tests/test.sh"
        )
        assert all(check[k] for k in check if k.startswith("fixed10_"))
    checks.append(check)
(out / "definition-equivalence.json").write_text(json.dumps(checks, indent=2) + "\n")
config = json.loads((setup / "fixed10-offline-verifier/harbor-config-template.json").read_text())
config["job_name"] = "swe-extended65-CANDIDATE-NOT-ADMITTED"
config["n_concurrent_trials"] = 12
config["datasets"][0]["path"] = str(candidate)
config["datasets"][0]["task_names"] = ids
(out / "harbor-config-candidate.json").write_text(json.dumps(config, indent=2) + "\n")
print(
    json.dumps(
        {
            "tasks": len(ids),
            "repos": dict(collections.Counter(byid[i]["repo"] for i in ids)),
            "official_cached": sum(r["image_cached"] for r in entries),
            "prepared_cached": sum(r["prepared_image_cached"] for r in entries),
            "definitions_generated": True,
        }
    )
)
