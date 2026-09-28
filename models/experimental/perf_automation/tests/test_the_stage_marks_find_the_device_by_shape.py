"""The per-stage pass finds the device the way it finds the pipeline: by shape, never by name.

2026-09-27, a generated perf test whose device is its `mesh_device` fixture: the injected pass handed
over a variable literally named `device`, so every profiled run printed
STAGE_MARKS_SKIPPED=NameError("name 'device' is not defined") and no stage was marked. Blocks written
by that template were then kept forever as "already injected". These pin the lookup, the template
call, and the in-place refresh of a block that predates it.
"""

import ast

from agent import stage_marks as sm


class _Mesh:
    def get_num_devices(self):
        return 32


class _Pipe:
    PIPELINE_STAGES = ()


def test_the_device_is_found_under_any_name():
    mesh = _Mesh()
    assert sm.find_device_in_scope({"cfg": 1, "whatever_it_is_called": mesh}) is mesh


def test_the_pipelines_own_device_is_the_fallback():
    pipe = _Pipe()
    pipe.device = _Mesh()
    assert sm.find_device_in_scope({"pipe": pipe}, pipe) is pipe.device
    assert sm.find_device_in_scope({}, None) is None
    assert sm.find_device_in_scope(None) is None


def test_the_found_device_is_the_one_the_marks_use(monkeypatch):
    mesh, seen = _Mesh(), []
    monkeypatch.setattr(sm, "find_pipeline_in_scope", lambda scope: _Pipe())
    monkeypatch.setattr(sm, "mark_stages_for", lambda pipe, device: seen.append(device) or 1)
    assert sm.mark_stages_in_scope({"m": mesh}) == 1
    assert seen == [mesh]
    explicit = object()
    sm.mark_stages_in_scope({"m": mesh}, explicit)
    assert seen[-1] is explicit, "a caller that passes a device positionally still wins"


def test_the_template_names_nothing_but_the_scope():
    block = sm._MARK_PASS_TEMPLATE.format(i="", bind="")
    call = next(ln for ln in block.splitlines() if sm._MARK_PASS_CALL_KEY in ln)
    names = {n.id for n in ast.walk(ast.parse(call.strip())) if isinstance(n, ast.Name)}
    assert names <= {"print", "_tt_sm2", "locals"}, names


def _old_block(indent, bind):
    fresh = sm._MARK_PASS_TEMPLATE.format(i=indent, bind=bind)
    return fresh.replace("mark_stages_in_scope(locals()", "mark_stages_in_scope(locals(), device")


def test_a_block_from_the_old_template_is_refreshed_in_place():
    text = "def f():\n    pipe = 1\n" + _old_block("    ", ", bind=_prep") + "    return pipe\n"
    out, refreshed = sm._refresh_mark_pass(text)
    assert refreshed
    assert "mark_stages_in_scope(locals(), bind=_prep)" in out and "locals(), device" not in out
    assert out.startswith("def f():\n    pipe = 1\n") and out.endswith("    return pipe\n")
    ast.parse(out)
    assert sm._refresh_mark_pass(out) == (out, False), "idempotent"


def test_the_refresh_is_a_success_for_the_run():
    text = "def f():\n    pipe = 1\n" + _old_block("    ", "") + "    return pipe\n"
    out, why = sm._relocate_mark_pass(text)
    assert why == sm._REFRESHED and sm.marks_ok(why), why
    again, why2 = sm._relocate_mark_pass(out)
    assert again == out and why2 == "already injected"


def test_a_current_block_is_left_alone():
    text = "def f():\n    pipe = 1\n" + sm._MARK_PASS_TEMPLATE.format(i="    ", bind="") + "    return pipe\n"
    assert sm._refresh_mark_pass(text) == (text, False)
