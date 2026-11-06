import ast
import json
import subprocess
from pathlib import Path

import yaml
from workflows.model_spec import load_templates_from_yaml, resolve_model_spec

root = Path.cwd()
base = "6efe6739e6513d0edbf49a8ef39527b80a23784d"
name = "workflows/model_specs/dev/llm.yaml"
before = yaml.safe_load(subprocess.check_output(["git", "show", base + ":" + name], text=True))
after = yaml.safe_load((root / name).read_text())
changes = []


def diff(a, b, path=""):
    assert type(a) == type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert not (a.keys() - b.keys())
        for k in b.keys() - a.keys():
            changes.append((path + "." + k, b[k]))
        for k in a:
            diff(a[k], b[k], path + "." + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, path + f"[{i}]")
    else:
        assert a == b, (path, a, b)


diff(before, after)
assert sorted((p.rsplit(".", 1)[1], v) for p, v in changes) == [
    ("VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "true"),
    ("enable_log_requests", True),
    ("max_log_len", 0),
]
templates = load_templates_from_yaml(root / name)
spec = resolve_model_spec(
    [s for t in templates for s in t.expand_to_specs()],
    model="google/gemma-4-31B-it",
    device="p300x2",
    impl="gemma4-31b-autoport-v3",
    catalog_name="dev",
)
data = spec.get_serialized_dict()
if isinstance(data, str):
    data = json.loads(data)
assert data["env_vars"]["VLLM_DEBUG_LOG_API_SERVER_RESPONSE"] == "true"
args = data["device_model_spec"]["vllm_args"]
source = ast.parse((root / "vllm-tt-metal/src/run_vllm_api_server.py").read_text())
fn = next(x for x in source.body if isinstance(x, ast.FunctionDef) and x.name == "_append_vllm_arg")
namespace = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<stock argument transport>", "exec"), namespace)
argv = []
for k in ("enable_log_requests", "max_log_len"):
    namespace["_append_vllm_arg"](argv, "--" + k.replace("_", "-"), args[k], set())
assert argv == ["--enable-log-requests", "--max-log-len", "0"]
print(
    json.dumps(
        {
            "exact_added_fields": changes,
            "resolved_serialized_env": "true",
            "stock_cli_argv": argv,
            "all_other_yaml_fields_equal": True,
            "cohort_source_unchanged": subprocess.check_output(
                ["git", "diff", base, "--", "reference_config/evals/eval_config.py"], text=True
            )
            == "",
        },
        indent=2,
    )
)
