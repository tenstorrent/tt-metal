"""The kv-cache gate labelled its target with a stage word this tool chose.

`op_class="decode"` is not in the op_class vocabulary at all. It survived because recall_knobs sends
unknown values through _integrity.classify -- an LLM round-trip -- to be remapped, so every firing
spent a model call resolving a constant we wrote ourselves, and resolved it to a single op_class,
which cannot reach the KV-cache section anyway (`op_class: attention,datamove` + `regime: decode`).

The stage belongs on the regime axis, which router.VOCABULARY deliberately leaves open so a model
names its own stages."""
import importlib.util
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _mcp():
    spec = importlib.util.spec_from_file_location("_pm_stage", _PA / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_gate_no_longer_invents_a_stage_word():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    assert '"op_class": "decode"' not in src, "the gate still labels its target with a word no model said"
    assert "op_class='decode'" not in src, "the reason text still tells the agent to route by it"


def test_the_gate_emits_an_in_vocabulary_op_class():
    """Out-of-vocabulary values cost an LLM remap on every call."""
    import sys

    sys.path.insert(0, str(_PA))
    from agent.router import VOCABULARY

    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index('"op": "generation_loop",')
    stanza = src[i : i + 600]
    assert '"op_class": "host_fallback"' in stanza
    assert "host_fallback" in VOCABULARY["op_class"]


def test_a_declared_single_stage_is_used_verbatim(monkeypatch):
    """The model's own word, whatever it is -- not a word matched against ours."""
    m = _mcp()
    import agent.stack_knob_repair as skr

    monkeypatch.setattr(skr, "stage_names", lambda _r: ["Transcribe"])
    assert m._token_stage_name() == "transcribe"


def test_nothing_declared_means_unnarrowed_not_a_guess(monkeypatch):
    m = _mcp()
    import agent.stack_knob_repair as skr

    monkeypatch.setattr(skr, "stage_names", lambda _r: [])
    assert m._token_stage_name() == ""


def test_several_stages_refuses_to_guess_by_name(monkeypatch):
    """THE GUESS THIS REPLACES. headline_unit's docstring calls a substring test on stage names 'a
    guess wearing an observation's clothes'. With several stages and no device to read
    <stage>_trace_items, the honest answer is 'I do not know' -- which recall_knobs treats as
    unnarrowed, handing over the whole catalogue with a note."""
    m = _mcp()
    import agent.stack_knob_repair as skr

    monkeypatch.setattr(skr, "stage_names", lambda _r: ["encode", "prefill", "decode"])
    assert m._token_stage_name() == "", "it name-matched 'decode' instead of admitting it cannot tell"


def test_recall_knobs_can_finally_narrow_by_stage():
    """router.DIMENSIONS has carried 'regime' all along and route() filters on it -- no caller ever
    passed one, so a stage-tagged lever was only findable via whatever op_class it also declared."""
    import inspect

    m = _mcp()
    assert "regime" in inspect.signature(m.recall_knobs).parameters, "the stage axis is still unreachable"
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index('q["regime"] = _rg')
    assert 'q["bound"] = b' in src[max(0, i - 600) : i], "regime is not wired into the router query"
