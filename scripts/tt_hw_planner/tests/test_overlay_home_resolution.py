import scripts.tt_hw_planner.overlay_manager as om


def test_resolver_defaults_to_main_repo_overlays():
    om._overlays_dir_cache = None
    r = om._main_repo_overlays_dir()
    assert r.name == "overlays"
    assert r.parent.name == "tt_hw_planner"
    assert r.parent.parent.name == "scripts"


def test_explicit_override_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "ov")
    assert om._main_repo_overlays_dir() == tmp_path / "ov"
    assert om._model_dir("a/b") == tmp_path / "ov" / "a_b"


def test_env_home_used_when_no_override(monkeypatch, tmp_path):
    monkeypatch.setattr(om, "_OVERLAYS_DIR", None)
    monkeypatch.setattr(om, "_overlays_dir_cache", None)
    monkeypatch.setenv(om._OVERLAYS_HOME_ENV, str(tmp_path / "home" / "overlays"))
    r = om._main_repo_overlays_dir()
    assert r == tmp_path / "home" / "overlays"


def test_env_home_normalizes_to_overlays_subdir(monkeypatch, tmp_path):
    monkeypatch.setattr(om, "_OVERLAYS_DIR", None)
    monkeypatch.setattr(om, "_overlays_dir_cache", None)
    root = tmp_path / "repo"
    monkeypatch.setenv(om._OVERLAYS_HOME_ENV, str(root))
    r = om._main_repo_overlays_dir()
    assert r == root / "overlays"
