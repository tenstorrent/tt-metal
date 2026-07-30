# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""STRESS for issues 2 + 3: the depth-knob discovery chain.

The chain is: cache -> tool's own convention -> env-read-line prompt -> validate -> cache.
Each hop was individually capable of poisoning every later run, so the properties that matter are
about the chain as a whole, not any one hop.

  s1  scale: trees from 1 file to 400 files / 5 MB. The answer must be found and the agent must
      never be asked when the convention is present, regardless of size or filename ordering.
  s2  400 adversarial agent answers -- none may be cached unless every variable in it is really
      read by the model
  s3  env-read syntax dialects that must be recognised, and look-alikes that must NOT be
  s4  no-poison invariant: after ANY sequence of hostile answers the cache is either empty or
      holds only genuinely-read variables
  s5  the retry loop terminates, is bounded, and never calls the agent when the answer is known
  s6  purity: no env mutation, deterministic, model tree never written to
"""

import importlib.util
import random
import string
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"


def _mod():
    sys.path.insert(0, str(_PA))
    spec = importlib.util.spec_from_file_location("cc_run_chain", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tree(tmp_path, files):
    tt = tmp_path / "tt"
    tt.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (tt / name).write_text(body)
    return tmp_path


def _wire(monkeypatch, answer="{}"):
    m = _mod()
    state = {"calls": 0, "cached": None, "prompts": []}

    def _fake(prompt):
        state["calls"] += 1
        state["prompts"].append(prompt)
        return answer(state["calls"]) if callable(answer) else answer

    monkeypatch.setattr(m, "_claude_text", _fake)
    monkeypatch.setattr(m, "_knob_cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_knob_cache_put", lambda _r, env: state.update(cached=env))
    monkeypatch.setenv("PERF_MCP_KNOB_RETRIES", "3")
    return m, state


_CONV = 'import os\nn = os.environ.get("TT_PERF_LAYERS")\n'


# --------------------------------------------------------------------------- s1
@pytest.mark.parametrize("n_files,pad_lines", [(1, 0), (10, 500), (100, 2000), (400, 1000)])
def test_s1_convention_found_at_any_tree_size(tmp_path, monkeypatch, n_files, pad_lines):
    m, st = _wire(monkeypatch)
    files = {f"f{i:04d}.py": "# pad\n" + ("x = 1\n" * pad_lines) for i in range(n_files)}
    files["zzzz_last.py"] = _CONV  # worst case for an alphabetical, size-truncated scan
    root = _tree(tmp_path, files)
    assert m._llm_depth_env(root, 4) == {"TT_PERF_LAYERS": "4"}
    assert st["calls"] == 0, f"agent called {st['calls']}x for a knob the tool injects itself"


def test_s1_convention_found_regardless_of_position(tmp_path, monkeypatch):
    for pos in ("aaaa", "mmmm", "zzzz"):
        m, st = _wire(monkeypatch)
        root = _tree(
            tmp_path / pos,
            {"bbbb.py": "x = 1\n" * 20000, f"{pos}.py": _CONV, "yyyy.py": "y = 2\n" * 20000},
        )
        assert m._llm_depth_env(root, 2) == {"TT_PERF_LAYERS": "2"}
        assert st["calls"] == 0


def test_s1_cov_value_is_reflected(tmp_path, monkeypatch):
    m, _st = _wire(monkeypatch)
    root = _tree(tmp_path, {"p.py": _CONV})
    for cov in (1, 2, 4, 8, 16, 32, 999):
        assert m._llm_depth_env(root, cov) == {"TT_PERF_LAYERS": str(cov)}


# --------------------------------------------------------------------------- s2
def _rand_name(rng):
    return "".join(rng.choice(string.ascii_uppercase + "_") for _ in range(rng.randint(3, 20)))


def test_s2_400_adversarial_answers_never_poison_the_cache(tmp_path, monkeypatch):
    rng = random.Random(20260730)
    root = _tree(tmp_path, {"p.py": 'import os\nn = os.environ.get("REAL_CAP")\n'})
    accepted = 0
    for i in range(400):
        real = rng.random() < 0.25
        name = "REAL_CAP" if real else _rand_name(rng)
        if name == "REAL_CAP":
            real = True
        m, st = _wire(monkeypatch, '{"%s": "4"}' % name)
        got = m._llm_depth_env(root, 4)
        if real:
            assert got == {"REAL_CAP": "4"}
            accepted += 1
        else:
            assert got == {}, f"answer {i} named {name!r}, which the model never reads, yet was accepted"
            assert st["cached"] is None, f"answer {i} poisoned the knob cache with {name!r}"
    assert accepted > 50, "the fixture never exercised the accept path; the test proves nothing"


@pytest.mark.parametrize(
    "answer",
    [
        "{}",
        "",
        "not json",
        "null",
        "[]",
        '{"": "4"}',
        '{"REAL_CAP": null}',
        "{'REAL_CAP': '4'}",  # single quotes -> not JSON
        '{"REAL_CAP": "4"',  # truncated
        '{"A": {"B": "4"}}',
        '{"REAL_CAP": "4"} trailing junk',
    ],
)
def test_s2_malformed_answers_degrade_without_caching_garbage(tmp_path, monkeypatch, answer):
    m, st = _wire(monkeypatch, answer)
    root = _tree(tmp_path, {"p.py": 'import os\nn = os.environ.get("REAL_CAP")\n'})
    got = m._llm_depth_env(root, 4)
    assert isinstance(got, dict)
    if st["cached"] is not None:
        assert set(st["cached"]) <= {"REAL_CAP"}, f"cached non-readable vars from {answer!r}: {st['cached']}"


# --------------------------------------------------------------------------- s3
@pytest.mark.parametrize(
    "src",
    [
        'os.environ.get("CAP")',
        "os.environ.get('CAP')",
        'os.getenv("CAP")',
        'os.environ["CAP"]',
        'os.environ.get( "CAP" , "8")',
        'x = int(os.environ.get("CAP", "1"))',
    ],
)
def test_s3_env_read_dialects_are_recognised(tmp_path, monkeypatch, src):
    m, st = _wire(monkeypatch, '{"CAP": "4"}')
    root = _tree(tmp_path, {"p.py": f"import os\n{src}\n"})
    assert m._llm_depth_env(root, 4) == {"CAP": "4"}, f"dialect not recognised: {src}"


@pytest.mark.parametrize(
    "src",
    [
        '"CAP"  # just a string, never read from the environment',
        "CAP = 4",
        "# os.environ.get('CAP')  -- commented out is still text, so this may match; see below",
    ],
)
def test_s3_non_reads_do_not_authorise_a_variable(tmp_path, monkeypatch, src):
    """A bare mention must not validate a knob. (A commented-out read is deliberately allowed to
    match -- a line-based scan cannot parse intent, and over-accepting a variable the source at
    least references is far safer than rejecting a real one.)"""
    m, _st = _wire(monkeypatch, '{"CAP": "4"}')
    root = _tree(tmp_path, {"p.py": f"import os\n{src}\n"})
    got = m._llm_depth_env(root, 4)
    if "os.environ" in src or "os.getenv" in src:
        assert got == {"CAP": "4"}
    else:
        assert got == {}, f"a bare mention authorised the knob: {src}"


# --------------------------------------------------------------------------- s4
def test_s4_no_poison_after_a_hostile_sequence(tmp_path, monkeypatch):
    """Rotate through hostile answers within one call's retry budget; the cache must stay clean."""
    seq = ['{"FAKE1": "4"}', '{"FAKE2": "4"}', '{"FAKE3": "4"}']
    m, st = _wire(monkeypatch, lambda n: seq[(n - 1) % len(seq)])
    root = _tree(tmp_path, {"p.py": 'import os\nn = os.environ.get("REAL_CAP")\n'})
    assert m._llm_depth_env(root, 4) == {}
    assert st["cached"] is None
    assert st["calls"] == 3, f"retry budget not honoured (calls={st['calls']})"


def test_s4_recovers_when_a_later_attempt_is_correct(tmp_path, monkeypatch):
    m, st = _wire(monkeypatch, lambda n: '{"FAKE": "4"}' if n < 3 else '{"REAL_CAP": "4"}')
    root = _tree(tmp_path, {"p.py": 'import os\nn = os.environ.get("REAL_CAP")\n'})
    assert m._llm_depth_env(root, 4) == {"REAL_CAP": "4"}
    assert st["cached"] == {"REAL_CAP": "4"}


# --------------------------------------------------------------------------- s5
def test_s5_retry_budget_is_bounded_and_terminates(tmp_path, monkeypatch):
    m, st = _wire(monkeypatch, '{"FAKE": "4"}')
    monkeypatch.setenv("PERF_MCP_KNOB_RETRIES", "5")
    root = _tree(tmp_path, {"p.py": 'import os\nn = os.environ.get("REAL_CAP")\n'})
    assert m._llm_depth_env(root, 4) == {}
    assert st["calls"] == 5


def test_s5_no_env_reads_at_all_asks_nobody(tmp_path, monkeypatch):
    m, st = _wire(monkeypatch, '{"ANYTHING": "4"}')
    root = _tree(tmp_path, {"p.py": "x = 1\n"})
    assert m._llm_depth_env(root, 4) == {}
    assert st["calls"] == 0, "nothing in the tree reads the environment; there is nothing to ask about"


def test_s5_prompt_is_small_even_for_a_huge_tree(tmp_path, monkeypatch):
    m, st = _wire(monkeypatch, "{}")
    files = {f"f{i:04d}.py": "x = 1\n" * 5000 for i in range(100)}  # ~3 MB
    files["answer.py"] = 'import os\nn = os.environ.get("WEIRD_CAP")\n'
    root = _tree(tmp_path, files)
    m._llm_depth_env(root, 4)
    p = st["prompts"][0]
    assert "WEIRD_CAP" in p, "the answer was truncated out of the prompt"
    assert len(p) < 20000, f"prompt is {len(p)} chars; env-read lines should be tiny"


# --------------------------------------------------------------------------- s6
def test_s6_pure(tmp_path, monkeypatch):
    import os

    m, _st = _wire(monkeypatch, '{"REAL_CAP": "4"}')
    root = _tree(tmp_path, {"p.py": 'import os\nn = os.environ.get("REAL_CAP")\n'})
    before_env = dict(os.environ)
    before_src = (root / "tt" / "p.py").read_text()
    a = m._llm_depth_env(root, 4)
    b = m._llm_depth_env(root, 4)
    assert a == b
    assert dict(os.environ) == before_env, "discovery mutated the environment"
    assert (root / "tt" / "p.py").read_text() == before_src, "discovery wrote to the model tree"


def test_s6_none_model_root(monkeypatch):
    m, st = _wire(monkeypatch)
    assert m._llm_depth_env(None, 4) == {}
    assert st["calls"] == 0
