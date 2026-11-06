"""Host-only AST equivalence and stock Harbor kwargs resolution check."""
import ast
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path("/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-terminal-native-history")
SOURCE = "reference_config/evals/eval_config.py"
BASE = "4b94cf3ba86d30a6816149872e59cc34438748ae"
old = ast.parse(subprocess.check_output(["git", "show", BASE + ":" + SOURCE], cwd=ROOT, text=True))
new = ast.parse((ROOT / SOURCE).read_text())


def kw(node, name):
    return next(k for k in node.keywords if k.arg == name)


def model(tree):
    return next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and any(
            k.arg == "hf_model_repo" and isinstance(k.value, ast.Constant) and k.value.value == "google/gemma-4-31B-it"
            for k in n.keywords
        )
    )


def tasks(tree):
    return {ast.literal_eval(kw(n, "task_name").value): n for n in kw(model(tree), "tasks").value.elts}


a, b = tasks(old), tasks(new)
terminal = kw(b["terminal_bench_2_1"], "agentic_eval_config").value
options = ast.literal_eval(kw(terminal, "agent_kwargs").value)
assert options.pop("store_all_messages") is True
original_terminal = kw(a["terminal_bench_2_1"], "agentic_eval_config").value
assert options == ast.literal_eval(kw(original_terminal, "agent_kwargs").value)
mode = SimpleNamespace(SMOKE_TEST="smoke", CI_NIGHTLY="nightly")


def resolve(node):
    return eval(compile(ast.Expression(node), "<config>", "eval"), {"EvalLimitMode": mode})


cohorts = json.loads(Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals/cohorts.json").read_text())[
    "terminal_bench_2_1"
]["rungs"]
assert resolve(kw(terminal, "task_names_map").value) == {k: cohorts[k] for k in ("smoke", "nightly")}
for name, expected in [("r1_gpqa_diamond", 0), ("swe_bench_verified", 0), ("terminal_bench_2_1", 10)]:
    assert resolve(kw(b[name], "limit_samples_map").value) == {"smoke": 2, "nightly": expected}
# Undo only authorized differences; entire file AST must then equal baseline.
for name in ("swe_bench_verified", "terminal_bench_2_1"):
    kw(b[name], "limit_samples_map").value = kw(a[name], "limit_samples_map").value
for name in ("task_names_map", "agent_kwargs"):
    kw(terminal, name).value = kw(original_terminal, name).value
assert ast.dump(new) == ast.dump(old)
source = ROOT / "llm_module/agentic/harbor.py"
node = next(
    n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_get_agent_kwargs"
)
ns = {"copy": copy, "Any": Any, "HarborRunConfig": object, "_MINI_SWE_AGENT": "mini-swe-agent"}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
config = SimpleNamespace(
    agent="terminus-2", agent_import_path=None, agent_kwargs={**options, "store_all_messages": True}
)
resolved = ns["_get_agent_kwargs"](config)
assert resolved == config.agent_kwargs and resolved["store_all_messages"] is True
assert resolved is not config.agent_kwargs
print("PASS: frozen cohort, peers 0, C2 and all other configuration unchanged; stock Harbor kwargs retention true")
