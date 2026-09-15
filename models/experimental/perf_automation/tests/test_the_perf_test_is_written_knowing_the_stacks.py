# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The perf test is generated knowing how many block stacks the model has.

A FEATURE WITH TESTS THAT NEVER RAN. generate_perf_test accepts `stacks`, documents it, and carries a
multi-stack branch behind it: given more than one stack it replaces the single `_pl` / `PERF_LAYERS`
lines with one env var per stack, so each stack's depth can be capped separately. `stacks` defaults
to None and NOT ONE production caller passed it -- before_loop's --pcc-test path and both
model_files paths all omitted it -- so the branch only ever executed inside
test_multi_stack_skeleton.py. Every perf test this tool has generated was written as if the model had
exactly one stack.

THE COST, measured on Voxtral-Mini-3B 2026-08-13. The generated test read only TT_PERF_LAYERS. The
depth bridge later discovered two stacks and set TT_PERF_STACK0_LAYERS / TT_PERF_STACK1_LAYERS --
names the already-written test does not read. One depth therefore reached every stack, and it had to
be max(stack0=2, stack2=32, stack3=3) = 32. The audio encoder IS 32 deep, so capping to 32 changed no
work, and the run concluded the depth knob never reached the builder -- discarding a correct window
and refusing, on a model whose knobs were wired and working.

WHY IT CAN BE ANSWERED BEFORE GENERATION. The walk needs a built model, and discovery used to get one
by running the GENERATED perf test -- so the test had to exist before the walk, and the walk could not
inform the test. The PCC gate breaks that circle: it is supplied by the operator, exists before
anything is written, and builds the model.

These tests pin the WIRING, because the branch itself was already covered and still never ran.
"""

import ast
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _calls_to(src: str, fname: str) -> list:
    """Every call to `fname` in `src`, as AST nodes."""
    return [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == fname
    ]


def test_every_production_caller_passes_the_stacks():
    """THE BUG WAS AN OMITTED ARGUMENT, so this asserts on the call sites themselves.

    A generator that silently accepts "I don't know" produces a single-stack test and nothing
    downstream can tell that from a genuine single-stack model.
    """
    for rel in ("agent/before_loop.py", "agent/model_files.py"):
        src = (_PA / rel).read_text()
        calls = _calls_to(src, "generate_perf_test")
        assert calls, "%s no longer generates a perf test" % rel
        for call in calls:
            kw = {k.arg for k in call.keywords}
            assert "stacks" in kw, "%s:%d generates a perf test without telling it the stacks" % (rel, call.lineno)


def test_the_walk_runs_before_generation_not_after():
    """Ordering is the point: a survey taken after the test is written cannot inform it."""
    src = (_PA / "agent" / "before_loop.py").read_text()
    survey = src.index("_survey_stacks(")
    gen = src.index("perf_node = generate_perf_test(")
    assert survey < gen, "the stack survey runs after the perf test is generated"


def test_the_survey_actually_reaches_the_node_it_is_given():
    """BEHAVIOUR, NOT SPELLING. The first version of these tests asserted that the source contained
    the right identifiers -- and every one of them passed while the survey was handed a
    MODEL-root-relative path and run from the REPO root, so pytest never found the file. A string
    check cannot fail when the string is right and the VALUE is wrong.

    This runs the real function against a real file and asserts on what it did.
    """
    import tempfile

    from agent.stack_survey import LAST_REASON, survey

    repo = Path(tempfile.mkdtemp())
    probe_dir = repo / "models" / "experimental" / "perf_automation" / "cc_optimize"
    probe_dir.mkdir(parents=True)
    # A probe stand-in that proves the node reached it: it echoes a census naming the node it got.
    probe_dir.joinpath("_op_sig_probe.py").write_text(
        "import sys, json\n"
        "rows = [{'kind': 'device', 'path': sys.argv[1], 'blocks': 7, 'cls': 'B'}]\n"
        "print('PERF_STACK_CENSUS=' + json.dumps(rows))\n"
    )
    node = repo / "models" / "demo" / "tests" / "test_e2e.py"
    node.parent.mkdir(parents=True)
    node.write_text("def test_e2e():\n    pass\n")

    # the form before_loop passes: ABSOLUTE
    got = survey(repo, "%s::test_e2e" % node)
    assert [(s.path, s.count) for s in got] == [(str(node), 7)], LAST_REASON[0]

    # the form that silently produced nothing in production
    assert survey(repo, "tests/test_e2e.py::test_e2e") == []
    assert "not found" in LAST_REASON[0]


def test_only_device_stacks_are_offered_to_the_generator():
    """A depth variable aimed at the HF reference caps nothing -- those torch modules are held for
    weight loading and never dispatch a ttnn op. Sizing from one asked Voxtral for depth 32, its own
    full depth, so the cap changed no work."""
    from agent.stack_survey import stacks_from_census

    rows = [
        {"kind": "reference", "path": "hf.model.audio_tower.layers", "blocks": 32, "cls": "VoxtralEncoderLayer"},
        {"kind": "reference", "path": "hf.model.language_model.layers", "blocks": 30, "cls": "LlamaDecoderLayer"},
        {"kind": "device", "path": "enc_a._inner.layers", "blocks": 32, "cls": "TtEncoderLayer"},
        {"kind": "device", "path": "lm_layers", "blocks": 3, "cls": "_LmBlock"},
    ]
    got = stacks_from_census(rows)
    assert [(s.path, s.count) for s in got] == [("enc_a._inner.layers", 32), ("lm_layers", 3)]


def test_a_one_element_list_is_not_offered_as_a_stack():
    """A capped build shrinks every stack, and the walk correctly rejects a list of one -- so a
    survey taken against a capped model reports structure the model does not have."""
    from agent.stack_survey import stacks_from_census

    assert stacks_from_census([{"kind": "device", "path": "lm_layers", "blocks": 1}]) == []


def test_the_survey_walks_at_full_depth():
    """Checked by RUNNING it: a capped build shrinks every stack, and a one-element list is not a
    stack -- so a survey inheriting a cap reports structure the model does not have."""
    import os
    import tempfile

    from agent.stack_survey import survey

    repo = Path(tempfile.mkdtemp())
    probe_dir = repo / "models" / "experimental" / "perf_automation" / "cc_optimize"
    probe_dir.mkdir(parents=True)
    probe_dir.joinpath("_op_sig_probe.py").write_text(
        "import os, json\n"
        "d = os.environ.get('TT_PERF_LAYERS', 'ABSENT')\n"
        "rows = [{'kind': 'device', 'path': 'depth=' + d, 'blocks': 4, 'cls': 'B'}]\n"
        "print('PERF_STACK_CENSUS=' + json.dumps(rows))\n"
    )
    node = repo / "t.py"
    node.write_text("def test_x():\n    pass\n")

    os.environ["TT_PERF_LAYERS"] = "2"
    try:
        got = survey(repo, str(node))
    finally:
        os.environ.pop("TT_PERF_LAYERS", None)
    assert got and got[0].path == "depth=ABSENT", "the survey inherited a depth cap: %s" % got[0].path


def test_an_unwalkable_model_degrades_to_todays_behaviour():
    """No answer must mean "generate as before", never a manufactured stack list."""
    from agent.stack_survey import stacks_from_census, survey

    assert stacks_from_census([]) == []
    assert survey(_PA, "") == []
    assert survey("/nonexistent/repo", "tests/x.py::test_y") == []


def test_the_survey_uses_the_repos_interpreter():
    """The system python has no ttnn, and a probe that dies importing it still exits 0 printing
    PERF_OP_SIGS=[] -- indistinguishable from a model with no stacks. That exact mistake cost a probe
    run on 2026-08-13."""
    src = (_PA / "agent" / "stack_survey.py").read_text()
    assert '"python_env" / "bin" / "python"' in src, "the survey can fall back to a python without ttnn"


def test_the_node_is_resolved_from_the_directory_the_probe_runs_in():
    """THE BUG THAT MADE THE FIRST LIVE ATTEMPT USELESS.

    resolve_pcc_node returns a MODEL-ROOT-relative node ("tests/e2e/test_e2e_pipeline.py::..."), and
    the probe runs pytest from the REPO root. Handing it the relative form gives pytest a path that
    does not exist: collection fails, no census is printed, and the empty result is indistinguishable
    from a model that genuinely has no stacks. Measured 2026-08-13 -- the survey reported "no block
    stacks discovered" while the walk three steps later found two on the same model.
    """
    import tempfile

    from agent.stack_survey import LAST_REASON, survey

    repo = Path(tempfile.mkdtemp())
    (repo / "models" / "experimental" / "perf_automation" / "cc_optimize").mkdir(parents=True)
    (repo / "models" / "experimental" / "perf_automation" / "cc_optimize" / "_op_sig_probe.py").write_text("")

    assert survey(repo, "tests/e2e/nope.py::test_x") == []
    assert "not found" in LAST_REASON[0], LAST_REASON[0]
    assert str(repo) in LAST_REASON[0], "the reason does not say which path was tried"


def test_an_empty_survey_always_records_why():
    """A survey that cannot walk and a model with no stacks both return []. Only one is a defect, and
    silence is what let a wrong path read as a finding."""
    from agent.stack_survey import LAST_REASON, describe, survey

    survey("/nonexistent/repo", "x.py::y")
    assert LAST_REASON[0], "an empty survey recorded no reason"
    assert LAST_REASON[0] in describe([]), "the reason is not surfaced in the run log"


def test_both_call_sites_pass_an_absolute_node():
    for rel, marker in (("agent/before_loop.py", "pcc_abs"), ("agent/model_files.py", "_abs")):
        src = (_PA / rel).read_text()
        i = src.index("_survey_stacks(")
        window = src[max(0, i - 700) : i + 200]
        assert marker in window, "%s still hands the survey a relative node" % rel


def test_the_survey_builds_the_model_rather_than_borrowing_a_tests_build():
    """THE DEPENDENCY THAT MADE IT RETURN NOTHING.

    Counting stacks needs a built model. Running a test and waiting for it to call build_pipeline
    works only for tests that build it that way -- the correctness gate does not, so the hook never
    fired and a two-stack model surveyed as zero. The contract guarantees the factory; call it.
    """
    src = (_PA / "agent" / "before_loop.py").read_text()
    i = src.index("_survey = _survey_build(")
    assert i > 0, "the survey still depends on a test to build the model"
    assert src.index("_survey_build(") < src.index("perf_node = generate_perf_test("), "survey runs too late"
    # Checked on the CODE, not the prose: the docstring says "no pytest", and a naive substring
    # match on the whole file is satisfied by that sentence alone.
    probe_src = (_PA / "cc_optimize" / "_stack_probe.py").read_text()
    tree = ast.parse(probe_src)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and ast.get_docstring(n):
            n.body = n.body[1:]
    code = ast.unparse(tree)
    assert "build_pipeline" in code, "the stack probe never calls the factory"
    assert "pytest" not in code, "the stack probe still goes through pytest"


def test_the_baseline_is_capped_at_the_same_depth_as_the_candidates():
    """A DICT WHERE A NUMBER WAS EXPECTED MEANT 'NO CAP', SILENTLY.

    _coverage_layers returns a per-stack dict. str() of it wrote "{'stack3': 2, 'stack2': 2}" into
    TT_PERF_LAYERS, which fails .isdigit() in every generated perf test and yields None -- ALL
    LAYERS. So the baseline was measured on the FULL model while every candidate after it ran capped,
    and any gain computed from that pair compared two different models. Measured on Voxtral
    2026-08-13: a 3977 ms baseline over ~32700 device ops against a capped model of 2965.
    """
    src = (_PA / "agent" / "before_loop.py").read_text()
    assert 'os.environ["TT_PERF_LAYERS"] = str(_bl_cov)' not in src, "a dict is still written to the depth knob"
    i = src.index('os.environ["TT_PERF_LAYERS"] = str(_bl_scalar)')
    window = src[max(0, i - 500) : i]
    assert "max(int(v) for v in _bl_cov.values())" in window, "the per-stack dict is not reduced to a scalar"


def test_the_baseline_records_the_depth_it_was_measured_at():
    """An unstamped number cannot be checked against anything -- and _record_baseline_anchor already
    reads profile["perf_layers"], so it had been recording "all" no matter what actually ran."""
    src = (_PA / "agent" / "before_loop.py").read_text()
    i = src.index('(Path(run.profiles_dir) / "baseline_profile.json").write_text')
    window = src[max(0, i - 900) : i]
    assert '"perf_layers"' in window, "the baseline profile is written without the depth it used"


def test_the_weights_answer_before_anything_is_built():
    """YOU CANNOT AFFORD TO BUILD THE MODEL JUST TO COUNT ITS STACKS.

    The survey runs BEFORE the knob repair, so the factory has no depth argument yet and a
    `layers=2` is swallowed by **kwargs -- meaning a "shallow" build is a FULL build. On Voxtral that
    is 30+ minutes (capping every stack took the same build to 7.1 seconds) spent to learn a number
    the checkpoint states in milliseconds: a repeated block prints its index into every key it owns.

    Building stays as the fallback for models whose weights are not readable that way.
    """
    import json
    import struct
    import tempfile

    from agent.stack_survey import survey_model

    d = Path(tempfile.mkdtemp())
    (d / "tt").mkdir()
    (d / "tt" / "pipeline.py").write_text("def build_pipeline(device, **kw):\n    return None\n")
    keys = ["audio_tower.layers.%d.attn.weight" % i for i in range(32)]
    keys += ["language_model.layers.%d.mlp.weight" % i for i in range(30)]
    head = {k: {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]} for k in keys}
    blob = json.dumps(head).encode()
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 32)

    got = survey_model("/nonexistent/repo", d)  # repo path unusable: proves no probe was spawned
    assert [(s.path, s.count) for s in got] == [("audio_tower.layers", 32), ("language_model.layers", 30)], got
