# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only check of review D13 (bench/packaging/DRAM_PREFILL_FIX_PLAN_v2_ADDENDUM_4.md), by AST (no ttnn import):
the `single-user` PB2 fire handler releases both traces and runs the L1 guard, and the PB adapter binds the fire
handler as the generator's pb_on_taint, so the override is what a fire calls."""

import ast
from pathlib import Path

TT = Path(__file__).resolve().parents[1] / "tt"


def _method(path, cls, name):
    for node in ast.walk(ast.parse((TT / path).read_text())):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return f
    raise KeyError((path, cls, name))


def _calls(fn):
    return [n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]


def test_pb2_fire_handler_runs_the_l1_guard():
    fn = _method("generator_resident_spec_vllm_pb2.py", "Gemma4ResidentSpecPB2ForCausalLM", "_pb_release_traces")
    calls = _calls(fn)
    assert calls == ["_release_sampling_trace", "_release_draft_trace", "_l1_guard"], calls
    guard = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_l1_guard")
    assert [a.value for a in guard.args] == ["pb guard fire"]


def test_pb_adapter_binds_the_fire_handler():
    init = _method("generator_resident_spec_vllm_pb.py", "Gemma4ResidentSpecPBForCausalLM", "__init__")
    binds = [
        n for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "pb_on_taint" for t in n.targets)
    ]
    assert len(binds) == 1
    value = binds[0].value
    assert isinstance(value, ast.Attribute) and value.attr == "_pb_release_traces"
    assert isinstance(value.value, ast.Name) and value.value.id == "self"


def test_pb2_subclasses_the_pb_adapter():
    tree = ast.parse((TT / "generator_resident_spec_vllm_pb2.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert [b.id for b in cls.bases] == ["Gemma4ResidentSpecPBForCausalLM"]
