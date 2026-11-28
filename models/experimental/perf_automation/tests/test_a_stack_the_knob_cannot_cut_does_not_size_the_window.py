"""A signposted stack sizes the profiling window only if the depth knob can actually cut it.

Qwen-Image-Edit on a WH Galaxy, 2026-09-30. With no signposts visible the measured ladder sized the
window at 2 (906 op types covered). After the VAE wins its 11 blocks became visible to the probe --
the only signposted stack, every block a different shape, run in full whatever TT_PERF_LAYERS says.
The signpost path took them as the window (11), the bridge then saw 11->11 and switched capping off,
and the full-depth capture blew tracy's 32K source-location limit. Same tool, same layers: which path
ran depended on which blocks happened to be visible.

The fix asks the model: one probe at the ladder's lowest rung. A stack whose deepest first-op block
does not move under it ignores the knob and is not used; if none is left, the measured ladder sizes
the window, exactly as on the run that worked. Models whose stacks the knob does cut -- Voxtral, whose
window is genuinely its full 26 layers -- are sized exactly as before.
"""

import importlib.util
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

SP = "PERF_BLOCK_SIGNPOST:"


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_knob_sizes", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _blocks(n, sid=None, last_op=None):
    """n signposted blocks, each with its own op (so coverage needs every one), as the probe emits."""
    out = []
    for b in range(n):
        out.append(f"{SP}{sid}:{b}" if sid else f"{SP}{b}")
        out += ["shared_op", f"op_{sid or 's'}_{b}"]
        if last_op and b == n - 1:
            out.append(last_op)
    return out


def _uniform(n, sid=None, last_op=None):
    """n signposted blocks running the same ops; optionally a last block with one op of its own."""
    out = []
    for b in range(n):
        out.append(f"{SP}{sid}:{b}" if sid else f"{SP}{b}")
        out += ["mm", "ln"]
        if last_op and b == n - 1:
            out.append(last_op)
    return out


# -- the helper on its own -----------------------------------------------------------------------------


def test_a_stack_the_cap_does_not_move_is_dropped():
    m = _run()
    full, _ = m._first_block_map(_blocks(11))
    assert m._stacks_the_knob_sizes(full, lambda k: _blocks(11), [2, 4, 8]) == {}


def test_a_stack_the_cap_cuts_is_kept():
    m = _run()
    full, _ = m._first_block_map(_blocks(26))
    assert m._stacks_the_knob_sizes(full, lambda k: _blocks(k), [2, 4, 8]) == full


def test_only_the_stack_the_cap_cuts_survives():
    m = _run()
    seq = _blocks(11, sid="stack0") + _blocks(20, sid="stack1")
    full, _ = m._first_block_map(seq)
    capped_seq = _blocks(11, sid="stack0") + _blocks(2, sid="stack1")  # stack0 ignores the cap
    kept = m._stacks_the_knob_sizes(full, lambda k: capped_seq, [2, 4])
    assert set(kept) == {"stack1"}


def test_a_stack_too_short_to_cut_and_a_stack_gone_under_the_cap_are_kept():
    m = _run()
    seq = _blocks(2, sid="stack0") + _blocks(9, sid="stack1")
    full, _ = m._first_block_map(seq)
    capped_seq = _blocks(2, sid="stack0")  # stack1 vanished: the cap reached it
    assert set(m._stacks_the_knob_sizes(full, lambda k: capped_seq, [2])) == {"stack0", "stack1"}


def test_no_answer_from_the_probe_changes_nothing():
    m = _run()
    full, _ = m._first_block_map(_blocks(11))

    def _raises(k):
        raise RuntimeError("device")

    for probe in (lambda k: [], lambda k: None, _raises):
        assert m._stacks_the_knob_sizes(full, probe, [2]) == full
    assert m._stacks_the_knob_sizes(full, lambda k: _blocks(11), []) == full, "no ladder: no verdict"
    assert m._stacks_the_knob_sizes({}, lambda k: _blocks(11), [2]) == {}


# -- through _coverage_layers, the decision the run actually makes --------------------------------------


def _size(m, monkeypatch, full_seq, capped_seq_at, ladder_result=(2, [], "measured")):
    calls = {"ladder": 0, "probes": []}

    def _fake_sigs(_repo, env, _dev, _node, _case, k, *a, **kw):
        calls["probes"].append((k, env.get("TT_PERF_LAYERS")))
        seq = full_seq if not k else capped_seq_at(k)
        return ({t for t in seq if not t.startswith(SP)}, "", seq)

    def _fake_measure(*a, **k):
        calls["ladder"] += 1
        return ladder_result

    monkeypatch.setattr(m, "_run_op_sigs", _fake_sigs)
    monkeypatch.setattr(m, "_measure_cov", _fake_measure)
    monkeypatch.setattr(m, "_parse_facts", lambda raw, s: {})
    monkeypatch.setattr(m, "_coverage_cache_get", lambda *a, **k: None)
    monkeypatch.setattr(m, "_coverage_cache_put", lambda *a, **k: None)
    monkeypatch.setattr(m, "_model_root_from_node", lambda *a, **k: Path("/nonexistent"))
    cov, facts = m._coverage_layers(Path("/repo"), {}, "0", "n.py::t", None, depth_knob={"TT_PERF_LAYERS": "2"})
    return cov, calls


def test_the_qwen_shape_is_sized_by_the_ladder_as_on_the_run_that_worked(monkeypatch, capsys):
    """Only the VAE is signposted and it ignores the knob: the ladder sizes the window, not the VAE."""
    m = _run()
    cov, calls = _size(m, monkeypatch, _blocks(11), lambda k: _blocks(11))
    assert calls["ladder"] == 1, "the measured ladder must decide"
    assert cov == 2 or cov == {"stack0": 2} or (isinstance(cov, dict) and set(cov.values()) == {2}), cov
    out = capsys.readouterr().out
    assert "ignores the depth knob" in out


def test_the_voxtral_shape_keeps_its_full_window(monkeypatch):
    """The knob cuts the stack and its last block has an op of its own: the window is the whole stack,
    exactly as before, and the ladder is not run."""
    m = _run()
    cov, calls = _size(m, monkeypatch, _uniform(26, last_op="last_only"), lambda k: _uniform(k))
    assert calls["ladder"] == 0
    window = next(iter(cov.values())) if isinstance(cov, dict) else cov
    assert window == 26


def test_the_check_probes_with_the_runs_own_knob_at_the_lowest_rung(monkeypatch):
    m = _run()
    _cov, calls = _size(m, monkeypatch, _uniform(26, last_op="last_only"), lambda k: _uniform(k))
    capped = [p for p in calls["probes"] if p[0]]
    assert capped and capped[0] == (2, "2"), capped


def test_the_probe_env_is_spelled_once_for_both_paths():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _measure_cov(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "_knob_probe_env(mcp_env, base, d)" in body and "_set_depth(env, d, key=numkey)" not in body
    assert "_knob_probe_env(mcp_env, _kb, k)" in src


def test_the_probe_env_sets_the_cap_as_the_old_inline_code_did():
    m = _run()
    base = {"TT_PERF_LAYERS": "2"}
    env = dict(base)
    m._set_depth(env, 8, key="TT_PERF_LAYERS")
    expected = {"X": "1", **env}
    assert m._knob_probe_env({"X": "1"}, base, 8) == expected
    assert m._knob_probe_env({"X": "1"}, {}, 8) == {"X": "1"}, "no knob: the probe env is the run's own"
