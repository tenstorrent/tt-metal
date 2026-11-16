import copy
import glob
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset

root = Path.cwd()
out = root / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup"
main = root / "bringup/artifacts/v5-main-harness"
sys.path.insert(0, str(main))
x = Dataset.from_file(
    glob.glob("/home/mvasiljevic/.cache/huggingface/datasets/Idavidrein___gpqa/gpqa_diamond/*/*/gpqa-train.arrow")[0]
)
ids = []
for domain, n in [("Physics", 4), ("Chemistry", 3), ("Biology", 3)]:
    a = [i for i, row in enumerate(x) if row["High-level domain"] == domain]
    ids += [a[round(j * (len(a) - 1) / (n - 1))] for j in range(n)]
ids.sort()
rows = [
    {
        "doc_id": i,
        "record_id": x[i]["Record ID"],
        "domain": x[i]["High-level domain"],
        "question_sha256": hashlib.sha256(x[i]["Question"].encode()).hexdigest(),
    }
    for i in ids
]
(out / "gpqa10_tasks.json").write_text(
    json.dumps(
        {
            "suite": "r1_gpqa_diamond",
            "tasks": rows,
            "list_rule": "Official dataset row order; evenly spaced endpoints and quantiles within each domain: Physics4/Chemistry3/Biology3; sorted doc IDs avoid main lm_eval unsorted sample attribution defect. Chosen from metadata before any v5 scores.",
            "prior_results_used": False,
        },
        indent=2,
    )
    + "\n"
)
import llm_module.eval_command as ec
from reference_config.evals.eval_config import ALL_EVAL_CONFIGS

client = root / "bringup/artifacts/benchmark-client"
ec.get_venv_provisioner = lambda: SimpleNamespace(venv_path=lambda _: client)
t = copy.deepcopy(next(t for t in ALL_EVAL_CONFIGS["google/gemma-4-31B-it"].tasks if t.task_name == "r1_gpqa_diamond"))
object.__setattr__(t, "doc_ids", ids)
spec = SimpleNamespace(
    hf_model_repo="google/gemma-4-31B-it",
    model_id="gemma4-v5",
    model_name="gemma4-v5",
    device_model_spec=SimpleNamespace(max_concurrency=10, max_context=262144),
)
cmd = ec.build_eval_command(
    t,
    spec,
    None,
    out.parent / "current/gpqa-phase1",
    8000,
    tokenizer_path="/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475",
)
(out / "gpqa_command.json").write_text(
    json.dumps(
        {
            "argv": cmd,
            "main_sha": "7a51c2f20f51e7508d8ea365d09092d00bbd5fcb",
            "reference_limits": {"request_timeout_seconds": 14400, "output_cap_tokens": 126976},
            "agents": 10,
            "note": "Concurrency must be lowered if pool-first launch evidence selects fewer agents; request protocol unchanged main builder. Start at reference thinking=true, not learned fastest thinking-off.",
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(cmd))
print(ids)
