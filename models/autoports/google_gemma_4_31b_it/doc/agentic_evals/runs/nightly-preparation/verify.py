# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only stock-route check for one prepared nightly registration."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))
from llm_module.agentic.harbor import _write_harbor_config
from llm_module.config import DriverContext, ServerConnection
from llm_module.drivers.agentic import build_harbor_config
from llm_module.eval_command import build_eval_command
from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode

suite = sys.argv[1]
cohorts = json.loads((Path(__file__).resolve().parents[2] / "cohorts.json").read_text())
model = "google/gemma-4-31B-it"
tasks = _eval_config_map[model].tasks
assert {t.task_name for t in tasks} == set(cohorts)
rc = SimpleNamespace(limit_samples_mode="ci-nightly")
with tempfile.TemporaryDirectory() as output:
    for task in tasks:
        expected = cohorts[task.task_name]["rungs"]
        assert set(expected["smoke"]).issubset(expected["nightly"])
        assert task.priority == "should"
        assert task.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == (
            len(expected["nightly"]) if task.task_name == suite else 0
        )
        if task.task_name != suite:
            continue
        if suite == "r1_gpqa_diamond":
            spec = SimpleNamespace(
                model_id="gemma-4-31B-it",
                model_name="gemma-4-31B-it",
                hf_model_repo=model,
                device_model_spec=SimpleNamespace(max_context=262144, max_concurrency=2, eval_max_retries=0),
            )
            command = build_eval_command(task, spec, "P300X2", output, 8000, runtime_config=rc)
            assert json.loads(command[command.index("--samples") + 1]) == {suite: expected["nightly"]}
            assert task.max_concurrent == 2
            assert task.model_kwargs["timeout"] == 14400
            assert task.gen_kwargs["max_gen_toks"] == 126976
        else:
            config = build_harbor_config(
                task,
                ServerConnection(base_url="http://localhost", service_port=8000, model=model),
                DriverContext(output_dir=Path(output)),
                runtime_config=rc,
                n_tasks=len(expected["nightly"]),
            )
            generated = json.loads(_write_harbor_config(config).read_text())
            assert generated["datasets"][0]["task_names"] == expected["nightly"]
            assert generated["datasets"][0]["n_tasks"] == len(expected["nightly"])
            assert generated["n_concurrent_trials"] == 2
            assert generated["agents"][0]["override_timeout_sec"] == 21600
            assert "override_cpus" not in generated["environment"]
            assert "override_memory_mb" not in generated["environment"]
print(suite, "PASS: exact nested nightly cohort; peer suites disabled; planned_clients=2")
