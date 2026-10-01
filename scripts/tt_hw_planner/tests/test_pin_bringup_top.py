import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "module_optimize_ut",
    str(Path(__file__).resolve().parents[1] / "commands" / "module_optimize.py"),
)
mo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mo)

_M1 = "<!-- BEGIN module:a -->\n## Module: `a` — 1/3\nx\n<!-- END module:a -->"
_M2 = "<!-- BEGIN module:b -->\n## Module: `b` — 2/3\ny\n<!-- END module:b -->"
_BR = "<!-- BEGIN bringup -->\n# Bring-up run report\nz\n<!-- END bringup -->"


def _write(d, txt):
    (d / "RUN_REPORT.md").write_text(txt)


def _read(d):
    return (d / "RUN_REPORT.md").read_text()


def test_bringup_hoisted_above_modules(tmp_path):
    _write(tmp_path, _M1 + "\n\n" + _BR + "\n\n" + _M2 + "\n")
    mo._pin_bringup_top(tmp_path)
    out = _read(tmp_path)
    assert out.index("BEGIN bringup") < out.index("BEGIN module:a") < out.index("BEGIN module:b")


def test_noop_when_bringup_already_top(tmp_path):
    orig = _BR + "\n\n" + _M1 + "\n\n" + _M2 + "\n"
    _write(tmp_path, orig)
    mo._pin_bringup_top(tmp_path)
    out = _read(tmp_path)
    assert out.index("BEGIN bringup") < out.index("BEGIN module:a") < out.index("BEGIN module:b")


def test_noop_when_no_bringup(tmp_path):
    orig = _M1 + "\n\n" + _M2 + "\n"
    _write(tmp_path, orig)
    mo._pin_bringup_top(tmp_path)
    out = _read(tmp_path)
    assert "BEGIN bringup" not in out and out.index("module:a") < out.index("module:b")


def test_no_content_dropped(tmp_path):
    _write(tmp_path, _M1 + "\n\n" + _BR + "\n\n" + _M2 + "\n")
    mo._pin_bringup_top(tmp_path)
    out = _read(tmp_path)
    for marker in ("module:a", "module:b", "bringup", "Bring-up run report"):
        assert marker in out
