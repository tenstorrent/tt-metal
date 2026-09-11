"""Host-only checks: execute production function ASTs without TTNN imports or devices."""
import ast
from collections import defaultdict
import math
import os
from pathlib import Path
import tempfile
from unittest.mock import Mock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


scope = dict(os=os, pd=pd, pytest=pytest, logger=Mock(), math=math, defaultdict=defaultdict)
function("models/tt_transformers/tests/test_utils.py", "merge_device_rows", scope)
run = function("models/demos/deepseek_v3_d_p/utils/perf_utils.py", "run_model_device_perf_test_with_merge", scope)
metric = "AVG DEVICE KERNEL DURATION [ns]"
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "ops.csv"
    scope["get_latest_ops_log_filename"] = lambda _: str(path)

    def execute(expected, values=(10.0, 20.0), markers=True):
        rows = []

        def row(op, kind, duration, device=0):
            return {"OP CODE": op, "OP TYPE": kind, "DEVICE KERNEL DURATION [ns]": duration, "DEVICE ID": device}

        rows += [row("Setup", "tt_dnn_device", 999.0, d) for d in range(2)]
        if markers:
            rows.append(row("MLA_START", "signpost", None))
        rows += [row("Matmul", "tt_dnn_device", v, d) for d, v in enumerate(values)]
        if markers:
            rows.append(row("MLA_END", "signpost", None))
        pd.DataFrame(rows).to_csv(path, index=False)
        scope["run_device_perf"] = Mock(return_value={metric: 9999.0})
        scope["check_device_perf"] = Mock(return_value={"threshold": 123.0})
        scope["prep_device_perf_report"] = Mock()
        run("worker", expected, "test", "mistral4", between_signposts=("MLA_START", "MLA_END"))

    execute(None)
    scope["check_device_perf"].assert_not_called()
    report = scope["prep_device_perf_report"].call_args.kwargs
    assert report["post_processed_results"][metric] == 20.0  # excludes setup; real device merge
    assert report["expected_results"] == {}
    execute(20.0)
    scope["check_device_perf"].assert_called_once()
    assert scope["check_device_perf"].call_args.kwargs["assert_on_fail"] is True
    assert scope["prep_device_perf_report"].call_args.kwargs["expected_results"] == {"threshold": 123.0}
    for values in ((float("nan"), 20.0), (-1.0, 20.0), (float("inf"), 20.0), (0.0, 0.0), ()):
        with pytest.raises(pytest.fail.Exception):
            execute(None, values)
    with pytest.raises(pytest.fail.Exception):
        execute(None, markers=False)

# Exercise the actual new worker and wrapper bodies; only their hardware callees are mocked.
worker_scope = dict(_run_chunked_prefill=Mock(), per_axis_topology=lambda fabric: ("ring", "linear"))
worker = function(
    "models/demos/deepseek_v3_d_p/tests/test_mla.py", "test_mistral4_mla_chunked_prefill_loudbox", worker_scope
)
worker("request", "mesh", {"fabric_config": "torus_y"}, "mistral_small_4")
args = worker_scope["_run_chunked_prefill"].call_args.kwargs
assert args["iters_isl"] == [5120] and args["prefill_len"] == 51200
assert args["reference"] is None and args["use_metadata_tensor"] is False and args["determinism_check"] is False
wrapper_scope = dict(
    os=os,
    pytest=pytest,
    _is_galaxy_env=lambda: False,
    run_model_device_perf_test_with_merge=Mock(),
    _CMD_MISTRAL4_CHUNKED_8X1="worker",
)
wrapper = function(
    "models/demos/deepseek_v3_d_p/tests/perf/test_mla_perf.py", "test_mistral4_mla_chunked_perf_loudbox", wrapper_scope
)
wrapper()
args = wrapper_scope["run_model_device_perf_test_with_merge"].call_args.kwargs
assert args["expected_device_perf_ns_per_iteration"] is None
assert args["between_signposts"] == ("MLA_START", "MLA_END") and args["num_iterations"] == 1
wrapper_scope["_is_galaxy_env"] = lambda: True
os.environ.pop("TT_VISIBLE_DEVICES", None)
with pytest.raises(pytest.skip.Exception):
    wrapper()
print(
    "PASS: record-only reporting, retained gate, invalid/missing duration rejection, signpost filtering, single-forward worker/wrapper, Galaxy skip"
)

os.environ["TT_VISIBLE_DEVICES"] = "0,1,2,3,11,10,9,8"
wrapper()
assert "glx_column" in wrapper_scope["run_model_device_perf_test_with_merge"].call_args.kwargs["model_name"]
print("PASS: filtered Galaxy column accepted and labeled separately from LoudBox")
