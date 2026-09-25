import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "summary_relabel_ut",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "summary.py"),
)
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def test_disp_level_relabels_tt_lang_to_ttnn_when_ttl_absent(monkeypatch):
    monkeypatch.setattr(summary, "_ttl_absent", lambda: True)
    assert summary._disp_level("tt-lang") == "ttnn"
    assert summary._disp_level("cpp") == "cpp"
    assert summary._disp_level("grid") == "grid"
    assert summary._disp_level("host") == "host"


def test_disp_level_keeps_tt_lang_when_ttl_present(monkeypatch):
    monkeypatch.setattr(summary, "_ttl_absent", lambda: False)
    assert summary._disp_level("tt-lang") == "tt-lang"
    assert summary._disp_level("cpp") == "cpp"


def test_ttl_absent_is_a_real_find_spec_check():
    assert summary._ttl_absent() == (importlib.util.find_spec("ttl") is None)


def test_render_header_shows_ttnn_when_ttl_absent(monkeypatch):
    monkeypatch.setattr(summary, "_ttl_absent", lambda: True)
    hdr = "op " + " ".join(summary._disp_level(c) for c in summary._LEVEL_COLS)
    assert "ttnn" in hdr and "tt-lang" not in hdr


def test_render_header_shows_tt_lang_when_ttl_present(monkeypatch):
    monkeypatch.setattr(summary, "_ttl_absent", lambda: False)
    hdr = "op " + " ".join(summary._disp_level(c) for c in summary._LEVEL_COLS)
    assert "tt-lang" in hdr and "ttnn" not in hdr
