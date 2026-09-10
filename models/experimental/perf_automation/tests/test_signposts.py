import tempfile
from pathlib import Path

from agent.probes import resolve_signposts


def _mk(root, rel, content):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_start_stop_detected():
    with tempfile.TemporaryDirectory() as t:
        _mk(t, "tests/pcc/test_model.py", 'from tracy import signpost\nsignpost("start")\nsignpost("stop")\n')
        r = resolve_signposts(Path(t) / "tests")
        assert (r["start_signpost"], r["end_signpost"]) == ("start", "stop")
        assert r["warning"] is None and set(r["found"]) == {"start", "stop"}


def test_none_defaults_not_null():
    with tempfile.TemporaryDirectory() as t:
        _mk(t, "tests/pcc/test_x.py", "def test_x():\n    assert True\n")
        r = resolve_signposts(Path(t) / "tests")
        assert (r["start_signpost"], r["end_signpost"]) == ("start", "stop")  # default, never None
        assert r["found"] == [] and "no tracy signposts" in r["warning"]


def test_custom_names_warn_and_default():
    with tempfile.TemporaryDirectory() as t:
        _mk(t, "tests/perf/test_p.py", 'signpost("prefill")\nsignpost(header="decode")\n')
        r = resolve_signposts(Path(t) / "tests")
        assert r["start_signpost"] == "start" and set(r["found"]) == {"prefill", "decode"}
        assert "custom signposts" in r["warning"]


def test_constants_skipped():
    with tempfile.TemporaryDirectory() as t:
        _mk(t, "tests/test_c.py", 'signpost(WARMUP_SIGNPOST)\nsignpost("start")\nsignpost("stop")\n')
        r = resolve_signposts(Path(t) / "tests")
        assert set(r["found"]) == {"start", "stop"}  # constant not captured, literals are
