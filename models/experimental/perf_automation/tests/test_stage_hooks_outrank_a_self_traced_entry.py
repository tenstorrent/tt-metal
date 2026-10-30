# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A pipeline that self-traces OUTSIDE its stage hooks still gets the standard (stage adapter) perf test.

Self-recording was decided by "anything the demo calls begins a capture", so a pipeline that traced
its decode inside generate() -- while keeping every `<stage>_trace_step` hook capture-free -- got the
time-only test: no eager pass, so no stage marks, and discovery refused the run
(Mistral-Small-4-119B-2603, 2026-10-14). The adapter only ever captures around a stage step hook, so
only a capture reachable from one of those can nest. Every placement below is checked both ways: what
_stage_steps_trace_free answers, and which test generate_perf_test then builds or refuses.

Stage names are invented on purpose: the check reads them from the model's own PIPELINE_STAGES.
"""

import textwrap

from agent import stage_seams as _seams
from agent.perf_test_gen import _self_tracing_fns, _stage_steps_trace_free, generate_perf_test

_STAGES = ("alpha", "beta")
_CAPTURE = "    tid = ttnn.begin_trace_capture(dev, cq_id=0)\n    ttnn.end_trace_capture(dev, tid, cq_id=0)\n"


def _step(stage):
    return _seams.hook(stage, _seams.STEP)


def _method(name, args, body):
    """A method of the fixture's class: `body` is written at function level (4 spaces) like _CAPTURE."""
    return "\n    def %s(self%s):\n%s" % (name, args, textwrap.indent(body, "    "))


def _pipeline(tmp_path, step_body="    return self.run()\n", entry_traces=True, extra=""):
    (tmp_path / "tt").mkdir(parents=True, exist_ok=True)
    src = "import ttnn\nPIPELINE_STAGES = %r\n\n\nclass Pipe:" % list(_STAGES)
    src += _method("generate", ", x", (_CAPTURE if entry_traces else "") + "    return x\n")
    src += _method("run", "", "    return 1\n")
    for s in _STAGES:
        src += _method(_seams.hook(s, _seams.SETUP), ", inputs", "    pass\n")
        src += _method(_step(s), "", step_body)
    (tmp_path / "tt" / "pipeline.py").write_text(src + extra)
    d = tmp_path / "demo"
    d.mkdir(exist_ok=True)
    (d / "demo_fast.py").write_text("from tt.pipeline import Pipe\nout = Pipe().generate(x)\n")
    return "demo/demo_fast.py"


# A standard draft: an eager pass that calls the pipeline's own entry, and the adapter's capture.
_STANDARD = (
    "import ttnn\ndef test_fast_perf(device):\n"
    "    def _eager_forward():\n"
    "        pipe.generate(x)\n"
    "    _eager_forward()\n"
    "    measure_adapter(adapter, device)\n"
    "    print('TRACE_REPLAY_PATH=trace+1cq')\n"
)
# A time-only draft: the self-recording entry timed directly.
_TIME_ONLY = (
    "import ttnn\ndef test_fast_perf(device):\n"
    "    pipe.generate(x)\n"
    "    print('TRACE_PER_TOKEN_MS=1.0'); print('TRACE_REPLAY_PATH=trace+1cq')\n"
)


def _generate(tmp_path, demo_rel, draft):
    prompts = []

    def runner(prompt):
        prompts.append(prompt)
        return draft

    node = generate_perf_test(tmp_path, "fast", demo_rel, runner=runner, force=True)
    return node, prompts


def test_trace_inside_the_entry_point_keeps_the_stage_adapter(tmp_path):
    demo = _pipeline(tmp_path)
    assert "generate" in _self_tracing_fns(tmp_path)
    assert _stage_steps_trace_free(tmp_path)
    node, prompts = _generate(tmp_path, demo, _STANDARD)
    assert node is not None, "the standard test was refused as a nested capture"
    assert "SELF-RECORDING PIPELINE" not in prompts[0]


def test_trace_outside_the_pipeline_class_keeps_the_stage_adapter(tmp_path):
    helper = "\n\ndef fast_loop(pipe, dev):\n" + _CAPTURE
    demo = _pipeline(tmp_path, entry_traces=False, extra=helper)
    (tmp_path / "demo" / "demo_fast.py").write_text("from tt.pipeline import Pipe, fast_loop\nfast_loop(Pipe(), d)\n")
    assert "fast_loop" in _self_tracing_fns(tmp_path)
    assert _stage_steps_trace_free(tmp_path)
    draft = _STANDARD.replace("pipe.generate(x)", "fast_loop(pipe, device)")
    node, prompts = _generate(tmp_path, demo, draft)
    assert node is not None
    assert "SELF-RECORDING PIPELINE" not in prompts[0]


def test_trace_inside_a_stage_step_keeps_self_recording(tmp_path):
    demo = _pipeline(tmp_path, step_body="    dev = self.dev\n" + _CAPTURE)
    assert not _stage_steps_trace_free(tmp_path)
    node, prompts = _generate(tmp_path, demo, _STANDARD)
    assert node is None, "a capture around a self-tracing stage step nests and hangs the device"
    assert "SELF-RECORDING PIPELINE" in prompts[0]
    node, _ = _generate(tmp_path, demo, _TIME_ONLY)
    assert node is not None


def test_trace_reached_from_a_stage_step_through_another_file_keeps_self_recording(tmp_path):
    demo = _pipeline(tmp_path, step_body="    return replay_step(self.dev)\n")
    (tmp_path / "tt" / "replay.py").write_text("import ttnn\n\n\ndef replay_step(dev):\n" + _CAPTURE)
    assert not _stage_steps_trace_free(tmp_path)
    node, _ = _generate(tmp_path, demo, _STANDARD)
    assert node is None


def test_no_trace_anywhere_is_not_self_recording(tmp_path):
    _pipeline(tmp_path, entry_traces=False)
    assert _self_tracing_fns(tmp_path) == set()
    assert _stage_steps_trace_free(tmp_path)


def test_unproven_models_keep_self_recording(tmp_path):
    """Each thing the proof needs, missing on its own, answers False."""
    no_stages = tmp_path / "no_stages"
    _pipeline(no_stages)
    p = no_stages / "tt" / "pipeline.py"
    p.write_text(p.read_text().replace("PIPELINE_STAGES", "OTHER_NAME"))
    assert not _stage_steps_trace_free(no_stages)

    missing_hook = tmp_path / "missing_hook"
    _pipeline(missing_hook)
    p = missing_hook / "tt" / "pipeline.py"
    p.write_text(p.read_text().replace("def %s(" % _step(_STAGES[-1]), "def not_a_hook("))
    assert not _stage_steps_trace_free(missing_hook)

    unparsed = tmp_path / "unparsed"
    _pipeline(unparsed)
    (unparsed / "tt" / "broken.py").write_text("def (:\n")
    assert not _stage_steps_trace_free(unparsed)
