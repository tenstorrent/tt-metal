"""A model directory reached through a link must still be named inside the tree it is run in.

An optimize run works in a git worktree, which by construction holds only committed files. Anything
still untracked -- which a model is, on the first run after its demo is emitted -- is staged into
that worktree as a link back to the source checkout. ``Path.resolve()`` follows links, so naming the
model through it answers with the source checkout, and every path derived from that answer walks out
of the worktree:

    ../../home/<user>/<repo>/models/<...>/tests/e2e/test_main_perf.py     written by before_loop
                            models/<...>/tests/e2e/test_main_perf.py     looked up by the optimizer

Those are two spellings of one file, so the depth cap the bridge proved was filed under a key nothing
later asks for. The profiler then ran unbounded, overflowed the device marker buffers and produced no
ops csv -- which reads exactly like a device fault, and was misread as one five times.

The second test is the floor under all of it: however a key comes to diverge, a profiled run that
arrives with no cap at all caps itself rather than capturing unbounded.

No stage name and no variable spelling is written here. Both are asked of the model.
"""

import os
from pathlib import Path

from agent import layer_depth as ld


def _model_with_stages(tmp_path: Path, stages) -> Path:
    """A model directory that declares its own stages, the way the tool discovers them."""
    root = tmp_path / "src" / "a_model"
    (root / "tt").mkdir(parents=True)
    (root / "tt" / "pipeline.py").write_text("PIPELINE_STAGES = %r\n" % (list(stages),))
    return root


def test_a_linked_model_is_named_inside_the_worktree(tmp_path):
    """The naming must not follow the link; resolving it is what escaped the tree."""
    src = tmp_path / "src" / "a_model"
    src.mkdir(parents=True)
    tree = tmp_path / "worktree"
    (tree / "models").mkdir(parents=True)
    staged = tree / "models" / "a_model"
    staged.symlink_to(src)

    sub = "tests/e2e/test_main_perf.py"
    followed = os.path.relpath(Path(staged).resolve() / sub, tree)
    in_tree = os.path.relpath(Path(os.path.abspath(staged)) / sub, tree)

    assert followed.startswith(".."), "the link no longer escapes; this test has stopped proving anything"
    assert in_tree == f"models/a_model/{sub}"
    assert not in_tree.startswith(".."), "a staged model dir was named outside the tree it runs in"


def test_a_real_model_dir_is_named_identically(tmp_path):
    """The same naming on a committed model must be a no-op, or this trades one divergence for another."""
    tree = tmp_path / "worktree"
    real = tree / "models" / "a_model"
    real.mkdir(parents=True)
    sub = "tests/e2e/test_main_perf.py"

    followed = os.path.relpath(Path(real).resolve() / sub, tree)
    in_tree = os.path.relpath(Path(os.path.abspath(real)) / sub, tree)
    assert followed == in_tree == f"models/a_model/{sub}"


def test_before_loop_names_the_perf_test_without_following_the_link():
    """The call site itself: model_root stays resolved for git, the perf-test path does not."""
    import agent.before_loop as bl

    body = Path(bl.__file__).read_text()  # read, not executed -- the caller needs a device run

    assert "model_root_in_tree = Path(os.path.abspath(config[" in body, "the non-following spelling is gone"
    assert 'model_root = Path(config["model_root"]).resolve()' in body, "model_root must stay resolved for gitio"
    line = next(ln for ln in body.splitlines() if ln.strip().startswith("perf_rel = "))
    nxt = body.splitlines()[body.splitlines().index(line) + 1]
    assert "model_root_in_tree" in line + nxt, "perf_rel still names the perf test through the resolved root"


def test_an_uncapped_profile_caps_itself(tmp_path):
    """No cap arriving is a reachable state, and it must not mean 'capture everything'."""
    root = _model_with_stages(tmp_path, ["alpha", "beta"])
    caps = ld.fallback_depth_caps(environ={}, model_root=root)

    assert caps, "a profiled run with no cap would have captured unbounded"
    assert set(caps) == set(ld.depth_var_names(model_root=root))
    assert {int(v) for v in caps.values()} == {ld.MIN_COVERAGE_DEPTH}
    for stage in ("alpha", "beta"):
        assert ld.stage_layers_var(stage) in caps, "a stage the model declares was left uncapped"


def test_the_floor_never_overrides_a_cap_that_arrived(tmp_path):
    """It is a floor, not a policy: anything already proven wins, on any spelling."""
    root = _model_with_stages(tmp_path, ["alpha"])
    assert ld.fallback_depth_caps(environ={ld.ENV: "4"}, model_root=root) == {}
    assert ld.fallback_depth_caps(environ={ld.stage_layers_var("alpha"): "3"}, model_root=root) == {}


def test_the_floor_stands_down_when_full_depth_was_asked_for(tmp_path):
    """The whole-model gates clear the cap on purpose; refilling it would grade a two-block model."""
    root = _model_with_stages(tmp_path, ["alpha"])
    assert ld.fallback_depth_caps(environ={ld.FORCE_ALL: "1"}, model_root=root) == {}


def test_the_floor_asks_the_model_and_not_a_pattern(tmp_path):
    """A model that renames its stages must still be capped on the names it actually reports."""
    renamed = _model_with_stages(tmp_path / "x", ["widget", "gadget"])
    caps = ld.fallback_depth_caps(environ={}, model_root=renamed)
    assert ld.stage_layers_var("widget") in caps and ld.stage_layers_var("gadget") in caps
