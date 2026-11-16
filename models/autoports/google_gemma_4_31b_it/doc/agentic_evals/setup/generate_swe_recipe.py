import copy
import glob
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path.cwd()
out = root / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup"
sys.path.insert(0, str(root / "bringup/artifacts/v5-main-harness"))
ids = [
    "django__django-11299",
    "astropy__astropy-14096",
    "matplotlib__matplotlib-25332",
    "sympy__sympy-13551",
    "scikit-learn__scikit-learn-14629",
    "pydata__xarray-3993",
    "sphinx-doc__sphinx-9673",
    "pytest-dev__pytest-5840",
    "pylint-dev__pylint-6903",
    "psf__requests-1142",
]
data = {
    r["instance_id"]: r
    for r in pq.read_table(
        glob.glob("/home/mvasiljevic/.cache/huggingface/hub/datasets--princeton*/snapshots/*/data/*.parquet")[0]
    ).to_pylist()
}
rows = []
for id in ids:
    r = data[id]
    image = "swebench/sweb.eval.x86_64." + id.replace("__", "_1776_") + ":latest"
    p = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"], capture_output=True, text=True
    )
    rows.append(
        {
            "task": id,
            "repo": r["repo"],
            "difficulty": r["difficulty"],
            "patch_lines": len(r["patch"].splitlines()),
            "image": image,
            "image_present": p.returncode == 0,
            "digests": json.loads(p.stdout) if p.returncode == 0 else [],
        }
    )
(out / "swe10_tasks.json").write_text(
    json.dumps(
        {
            "suite": "swe_bench_verified",
            "dataset_revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
            "tasks": rows,
            "list_rule": "Main Qwen3.8 nightly five plus five distinct repositories represented by locally cached official task images. Frozen before v5 scores; ten repositories and multiple difficulty/patch-size classes. Patch contents not exposed, only line counts. No prior results reused.",
        },
        indent=2,
    )
    + "\n"
)
from llm_module.agentic.harbor import HarborRunConfig, _write_harbor_config
from reference_config.evals.eval_config import ALL_EVAL_CONFIGS

ref = next(
    t for t in ALL_EVAL_CONFIGS["Qwen/Qwen3.8-27B"].tasks if t.task_name == "swe_bench_verified"
).agentic_eval_config
kwargs = copy.deepcopy(ref.agent_kwargs)
kwargs.setdefault("config", {}).setdefault("model", {}).setdefault("model_kwargs", {}).setdefault("extra_body", {})[
    "chat_template_kwargs"
] = {"enable_thinking": True}
c = HarborRunConfig(
    task_name="swe-phase2",
    dataset=ref.dataset,
    agent=ref.agent,
    model_name="openai/google/gemma-4-31B-it",
    jobs_dir=out / "swe-jobs",
    api_base="http://127.0.0.1:8000/v1",
    n_concurrent_trials=10,
    n_attempts=1,
    environment_type="docker",
    agent_kwargs=kwargs,
    n_tasks=None,
    override_cpus=None,
    override_memory_mb=None,
    timeout_multiplier=None,
    agent_timeout_sec=ref.agent_timeout_sec,
    llm_timeout_sec=ref.llm_timeout_sec,
    task_names=ids,
    dataset_path=out / "swe-tasks",
    allow_agent_endpoint=True,
)
p = _write_harbor_config(c)
(out / "swe_command.json").write_text(
    json.dumps(
        {
            "argv": [str(root / "bringup/artifacts/agentic-v5-swe-env/bin/harbor"), "run", "--config", str(p)],
            "config": str(p),
            "reference_model": "Qwen/Qwen3.8-27B",
            "limits": {
                "agent_timeout_sec": ref.agent_timeout_sec,
                "llm_timeout_sec": ref.llm_timeout_sec,
                "output_cap_tokens": ref.agent_kwargs["max_tokens"],
            },
            "blockers": [],
            "environment_ready": True,
            "source_checkout": "bringup/artifacts/parser-protocol-harbor",
            "source_sha": "e76c762fca5267dfb36e1f900f0a7724ed72e735",
            "score_reference": 0.648,
            "source": "Operator inputs, not Qwen provisional published score",
            "no_network_sources_fetched": True,
        },
        indent=2,
    )
    + "\n"
)
print("wrote SWE metadata/config recipe", len(rows), "images present", sum(r["image_present"] for r in rows))
