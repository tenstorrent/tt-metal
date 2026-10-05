"""Pin: emit-e2e requires a SIGNAL-rendering pipeline to score its output, and requires it
of nothing else.

A text-to-speech port read PCC >= 0.99 on every stage and on its final waveform while producing
unusable audio: the correctness run ended on its safety cap after a fraction of the real output,
so PCC only ever saw a matched prefix. The gate asks for two scores PCC cannot give -- what
fraction of the requested words survive the output, and how natural it is predicted to sound --
and asks for them ONLY where the pipeline declares an output rate, i.e. only where the output is
a rendered signal. A token or tensor output is not asked for them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.commands import emit_e2e as E
from scripts.tt_hw_planner.commands.emit_e2e import (
    _identifier_mentions,
    _renders_signal,
    _signal_quality_gate,
)

_SIGNAL_PIPELINE = (
    "def run_speech(stubs, inputs):\n"
    "    wave = stubs['codec'](codes)\n"
    "    return {'waveform': wave, 'sampling_rate': int(cfg.output_rate)}\n"
)
_TOKEN_PIPELINE = (
    "def run_text(stubs, inputs):\n"
    "    ids = stubs['decoder'](x)\n"
    "    return {'token_ids': ids, 'logits': logits}\n"
)

_SCORED_TEST = (
    "def test_quality(evidence):\n"
    "    wer = transcribe_and_score(tt['waveform'], prompts)\n"
    "    mos = naturalness(tt['waveform'])\n"
    "    assert wer <= ref_wer + margin\n"
    "    assert mos >= ref_mos - margin\n"
)
_PRINTED_ONLY_TEST = (
    "def test_quality(evidence):\n"
    "    wer = transcribe_and_score(tt['waveform'], prompts)\n"
    "    mos = naturalness(tt['waveform'])\n"
    "    print(f'wer={wer} mos={mos}')\n"
    "    assert pcc >= 0.99\n"
)


def _demo(tmp_path: Path, pipeline_src: str, test_src: str | None = None) -> Path:
    d = tmp_path / "demo"
    (d / "tt").mkdir(parents=True)
    (d / "tt" / "pipeline.py").write_text(pipeline_src)
    if test_src is not None:
        (d / "tests" / "e2e").mkdir(parents=True)
        (d / "tests" / "e2e" / "test_e2e.py").write_text(test_src)
    return d


def test_signal_output_without_scores_fails(tmp_path: Path) -> None:
    r = _signal_quality_gate(_demo(tmp_path, _SIGNAL_PIPELINE, _PRINTED_ONLY_TEST))
    assert r and "signal-quality" in r
    assert "asserts no WER and no MOS." in r


def test_signal_output_with_asserted_scores_passes(tmp_path: Path) -> None:
    assert _signal_quality_gate(_demo(tmp_path, _SIGNAL_PIPELINE, _SCORED_TEST)) is None


def test_partial_scoring_names_only_what_is_missing(tmp_path: Path) -> None:
    only_wer = "def test_q(e):\n    assert wer <= 0.1\n"
    r = _signal_quality_gate(_demo(tmp_path, _SIGNAL_PIPELINE, only_wer))
    assert r and "asserts no MOS." in r and "no WER" not in r


def test_token_output_is_never_asked_for_signal_scores(tmp_path: Path) -> None:
    assert _signal_quality_gate(_demo(tmp_path, _TOKEN_PIPELINE, _PRINTED_ONLY_TEST)) is None


def test_no_tests_dir_is_safe(tmp_path: Path) -> None:
    assert _signal_quality_gate(_demo(tmp_path, _SIGNAL_PIPELINE)) is None


def test_no_tt_dir_is_safe(tmp_path: Path) -> None:
    (tmp_path / "demo").mkdir()
    assert _signal_quality_gate(tmp_path / "demo") is None


def test_unparseable_pipeline_does_not_raise(tmp_path: Path) -> None:
    assert _renders_signal(_demo(tmp_path, "def broken(:\n")) is False


def test_rate_declared_as_a_keyword_counts(tmp_path: Path) -> None:
    src = "def run(s):\n    return Output(wave, sample_rate=cfg.rate)\n"
    assert _renders_signal(_demo(tmp_path, src)) is True


def test_identifier_matching_is_by_part_not_substring() -> None:
    assert _identifier_mentions("wer", "wer")
    assert _identifier_mentions("MIN_WER", "wer")
    assert _identifier_mentions("wer_budget", "wer")
    assert not _identifier_mentions("answer", "wer")
    assert not _identifier_mentions("lower", "wer")
    assert not _identifier_mentions("moses", "mos")


# --------------------------------------------------------------------------------------------
# the scores are required PER ROW, in both places the rule is read, from one constant
# --------------------------------------------------------------------------------------------


def _one_line(text: str) -> str:
    return " ".join(text.split())


def test_the_per_row_rule_names_the_three_ways_a_row_collapses() -> None:
    """A repeat loop runs long, drops words and sounds wrong; each bound is against the golden's own
    row, with the numbers coming from the named constants rather than being retyped."""
    rule = E.PER_ROW_QUALITY_RULE
    assert "PER ROW" in rule and "golden's rendering of the SAME prompt" in rule
    for axis in ("runs long", "drops words", "sounds wrong"):
        assert axis in rule, axis
    assert "never absolute" in rule
    assert ("%.1fx" % E._ROW_LENGTH_RATIO) in rule and ("plus %d frames" % E._ROW_LENGTH_SLACK_FRAMES) in rule
    assert ("%.2f (a loop drops words)" % E._ROW_WER_MARGIN) in rule
    assert ("%.2f below the golden's row MOS" % E._ROW_MOS_MARGIN) in rule


def test_the_builder_s_checklist_carries_the_per_row_rule_verbatim() -> None:
    """Item 5 of the TT-only contract is where a builder learns what to score; the rule is spliced in
    from the one constant (wrapped), and the placeholder it replaces is gone."""
    contract = E._TT_ONLY_CONTRACT
    assert E._PER_ROW_RULE_SLOT not in contract, "the per-row slot was never filled"
    assert _one_line(E.PER_ROW_QUALITY_RULE) in _one_line(contract)
    # it sits inside the signal-scoring item, after the corpus-margin sentence it extends
    item = contract.index("SCORE A RENDERED SIGNAL")
    assert contract.index("inventing absolute numbers.") < contract.index("PER ROW") and item < contract.index(
        "PER ROW"
    )
    assert contract.index("PER ROW") < contract.index("ALLOWED HF USAGE")


def test_the_graduation_message_carries_the_per_row_rule(tmp_path: Path) -> None:
    """The G7 message a signal pipeline without scores is refused with is the same rule."""
    r = _signal_quality_gate(_demo(tmp_path, _SIGNAL_PIPELINE, _PRINTED_ONLY_TEST))
    assert r and _one_line(E.PER_ROW_QUALITY_RULE) in _one_line(r)
    assert "do not merely print them." in r


def test_the_rule_is_written_once() -> None:
    """Two readers, one source: the sentence must not exist as a second copy anywhere in the module."""
    src = Path(E.__file__).read_text()
    assert src.count("Assert them PER ROW as well as over the corpus") == 1


def test_the_per_row_rule_reaches_the_rendered_builder_prompt(tmp_path: Path, monkeypatch) -> None:
    """The checklist is only read through the agent prompt; the rule has to survive the render."""
    monkeypatch.setattr(E, "_required_heads_block", lambda model_id, all_tasks: "")
    prompt = E._build_agent_prompt(model_id="some/model", demo_dir=tmp_path, pcc=0.99)
    assert _one_line(E.PER_ROW_QUALITY_RULE) in _one_line(prompt)
