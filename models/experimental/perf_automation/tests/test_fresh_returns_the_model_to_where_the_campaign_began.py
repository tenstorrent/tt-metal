"""--fresh cleared the run's memory and left the model optimized.

The wins are committed to the model tree and survive a restart; the baseline and the ceiling they
are measured against live in the state --fresh deletes. Keeping the first while resetting the other
two is the combination that lies -- the run re-derives its ceiling from a model that already carries
the optimizations, so the target moves with the work.

voxtral, measured: a fidelity lever took the pinned peak from 175.5 TFLOPS (HiFi4, pre-campaign) to
702.0 (LoFi), and prefill's roofline ceiling from 203.82 ms to 50.95 -- a 4x change in the yardstick
caused by a win rather than by hardware. The report meanwhile printed `baseline 12.87 ms` with no
mention of the 38 commits already in the tree, presenting a mid-campaign checkpoint as the model's
starting point.

The origin is DERIVED, not remembered: --fresh deletes anything it could have stored, so publication
is the only record that survives it. What is on a remote is the model before this campaign; every
commit after that is the campaign's own work."""
import subprocess

import pytest


def _git(*a, cwd):
    return subprocess.run(["git", *a], cwd=str(cwd), capture_output=True, text=True, timeout=120)


@pytest.fixture()
def repos(tmp_path):
    """An 'upstream' with the published model, and a clone that then optimizes it locally."""
    up, work = tmp_path / "up", tmp_path / "work"
    up.mkdir()
    _git("init", "-q", "-b", "main", cwd=up)
    _git("config", "user.email", "t@t", cwd=up)
    _git("config", "user.name", "t", cwd=up)
    m = up / "models" / "demo"
    m.mkdir(parents=True)
    (m / "pipeline.py").write_text("FIDELITY = 'HiFi4'\n")
    _git("add", "-A", cwd=up)
    _git("commit", "-qm", "bring-up", cwd=up)

    subprocess.run(["git", "clone", "-q", str(up), str(work)], check=True, timeout=180)
    _git("config", "user.email", "t@t", cwd=work)
    _git("config", "user.name", "t", cwd=work)
    return up, work, work / "models" / "demo"


def _optimize(work, model, text="FIDELITY = 'LoFi'\n", msg="win: drop to LoFi"):
    (model / "pipeline.py").write_text(text)
    _git("add", "-A", cwd=work)
    _git("commit", "-qm", msg, cwd=work)


def test_the_origin_is_the_published_state_not_the_latest_commit(repos):
    from agent.fresh_start import published_origin

    up, work, model = repos
    published = _git("rev-parse", "HEAD", cwd=work).stdout.strip()
    _optimize(work, model)
    assert published_origin(model) == published, "the campaign's own commits were taken as the origin"


def test_fresh_returns_the_model_to_the_published_state(repos):
    from agent.fresh_start import reset_model_to_published

    up, work, model = repos
    _optimize(work, model)
    assert "LoFi" in (model / "pipeline.py").read_text()
    out = reset_model_to_published(model)
    assert out["changed"] is True
    assert "HiFi4" in (model / "pipeline.py").read_text(), "the win survived a --fresh reset"


def test_a_dry_run_changes_nothing(repos):
    from agent.fresh_start import reset_model_to_published

    up, work, model = repos
    _optimize(work, model)
    out = reset_model_to_published(model, dry_run=True)
    assert out["changed"] is True and "would reset" in out["why"]
    assert "LoFi" in (model / "pipeline.py").read_text(), "a dry run wrote to the tree"


def test_an_unpublished_model_is_refused_rather_than_guessed(tmp_path):
    """No published commit means no origin to return to -- resetting would discard work nobody
    agreed to lose."""
    from agent.fresh_start import reset_model_to_published

    r = tmp_path / "local"
    (r / "models" / "demo").mkdir(parents=True)
    _git("init", "-q", "-b", "main", cwd=r)
    _git("config", "user.email", "t@t", cwd=r)
    _git("config", "user.name", "t", cwd=r)
    (r / "models" / "demo" / "pipeline.py").write_text("x = 1\n")
    _git("add", "-A", cwd=r)
    _git("commit", "-qm", "local only", cwd=r)
    out = reset_model_to_published(r / "models" / "demo")
    assert out["changed"] is False and "no published commit" in out["why"]


def test_an_already_pristine_model_is_left_alone(repos):
    from agent.fresh_start import reset_model_to_published

    up, work, model = repos
    out = reset_model_to_published(model)
    assert out["changed"] is False and "already at the published state" in out["why"]


def test_wipe_still_never_touches_a_tracked_file(repos):
    """The reset is deliberately SEPARATE from wipe(), whose one promise is that it does not."""
    from agent.fresh_start import wipe

    up, work, model = repos
    _optimize(work, model)
    wipe(str(work / "state"), tool_root=None, model_dir=model)
    assert "LoFi" in (model / "pipeline.py").read_text(), "wipe() reverted a tracked file"
