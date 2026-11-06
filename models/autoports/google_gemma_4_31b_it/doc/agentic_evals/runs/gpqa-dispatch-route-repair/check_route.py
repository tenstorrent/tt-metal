"""Host-only route check; invoke from the immutable TTI candidate worktree."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))
from llm_module.drivers.agentic import resolve_n_tasks
from llm_module.eval_command import build_eval_command
from llm_module.eval_configs import get_llm_eval_tasks
from reference_config.evals.eval_config import _eval_config_map
from test_module.llm_tests.agentic_eval_tests import _select_agentic_tasks

count, clients = map(int, sys.argv[1:])
os.environ.pop("TT_EVAL_CLIENT_CONCURRENCY", None)
config = _eval_config_map["google/gemma-4-31B-it"]
runtime = SimpleNamespace(limit_samples_mode="ci-nightly", agentic_benchmark=None)
spec = SimpleNamespace(
    model_id="gemma-4-31B-it",
    model_name="gemma-4-31B-it",
    hf_model_repo="google/gemma-4-31B-it",
    metadata={},
    supported_modalities=["text"],
    device_model_spec=SimpleNamespace(max_context=262144, max_concurrency=8, eval_max_retries=0),
)
wrong = _select_agentic_tasks(SimpleNamespace(all_params=config, runtime_config=runtime))
assert [x.task_name for x in wrong] == ["swe_bench_verified", "terminal_bench_2_1"]
assert [resolve_n_tasks(x, runtime) for x in wrong] == [0, 0]
correct = get_llm_eval_tasks(spec, runtime)
assert [x.task_name for x in correct] == ["r1_gpqa_diamond"]
with tempfile.TemporaryDirectory() as temporary:
    command = build_eval_command(correct[0], spec, "P300x2", Path(temporary), 8000, runtime_config=runtime)
ids = json.loads(command[command.index("--samples") + 1])["r1_gpqa_diamond"]
expected = [152, 189, 46, 87, 52, 181, 31, 185, 50, 187, 153, 135, 81, 93, 7, 42, 24, 168, 20, 28][:count]
assert ids == expected
args = command[command.index("--model_args") + 1]
assert f"num_concurrent={clients}" in args and "timeout=14400" in args
assert "revision=842da3794eaa0b77d5f08bae87a17459d91ff475" in args
assert "local-chat-completions" in command
print(
    json.dumps(
        {
            "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "incorrect_workflow": "agentic",
            "incorrect_selected_peer_sizes": [0, 0],
            "correct_workflow": "evals",
            "selected": "r1_gpqa_diamond",
            "doc_ids": ids,
            "planned_clients": clients,
            "route": "local-chat-completions",
            "host_only": True,
        },
        indent=2,
    )
)
