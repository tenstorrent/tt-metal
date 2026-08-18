# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A stage's ceiling comes from what it processes, never from what it is called.

THE SHAPE OF THE BUG, in four places at once. Voxtral's audio encoder was measured at 12.79 ms and
priced at 0.041 ms, and every input to that number was decided by the string "encode":

  * active_bytes RAISED for any regime but decode|prefill, and its callers wrap it in
    `except Exception: return base` -- so the encoder did not get an error, it got a weights-only
    read set, silently, with no activation term at all;
  * _roofline_stage_share asked active_bytes to price the stage and read the ANSWER as a verdict --
    accepted meant "this stage reads the whole backbone", raised meant "refuse" -- which is that same
    name list, laundered through an exception, in a function whose comment says it is not doing that;
  * the item count that sets the compute ceiling (2 x params x items) was parsed into a map under a
    hardcoded key, `stage_isl["prefill"] = _iv`, so no other stage could ever have one;
  * a synthetic host-overhead bucket was tagged `regime: decode`, a real stage name that a per-stage
    reader matches on.

What actually distinguishes stages is structural and already recorded: how many subtrees the model
has, which subtree each stage runs, and how many items one call retires. This asserts the tool reads
those and not the spelling.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


# --------------------------------------------------------------- the byte model has no name gate


def test_the_byte_model_prices_any_stage_it_is_handed():
    from agent.perf_target import active_bytes

    mf = {"total_params": 1_000_000, "layers": 4, "hidden_size": 512, "kv_heads": 4, "head_dim": 64}
    for name in ("decode", "prefill", "encode", "denoise", "vocode", "", "whatever"):
        assert active_bytes(mf, regime=name, seq_len=128, items=8) > 0, name


def test_two_stages_doing_identical_work_get_identical_bytes():
    """The property that makes the name safe to keep as a label: it moves nothing."""
    from agent.perf_target import active_bytes

    mf = {"total_params": 1_000_000, "layers": 4, "hidden_size": 512, "kv_heads": 4, "head_dim": 64}
    got = {active_bytes(mf, regime=n, seq_len=64, items=16, batch=2) for n in ("prefill", "encode", "zzz")}
    assert len(got) == 1, got


def test_work_is_what_moves_the_byte_count():
    from agent.perf_target import active_bytes

    mf = {"total_params": 1_000_000, "layers": 4, "hidden_size": 512, "kv_heads": 4, "head_dim": 64}
    idle = active_bytes(mf, regime="encode", seq_len=0, items=0)
    busy = active_bytes(mf, regime="encode", seq_len=128, items=128)
    assert busy > idle > 0


def test_no_source_file_asks_the_byte_model_to_reject_a_name():
    """The guard is gone; a future edit must not reintroduce a list of acceptable stages."""
    src = (_PA / "agent" / "perf_target.py").read_text()
    code = "".join(seg for i, seg in enumerate(src.split('"""')) if i % 2 == 0)
    code = "\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith("#"))
    assert 'regime not in ("decode", "prefill")' not in code
    assert "NotImplementedError" not in code.split("def active_bytes", 1)[-1].split("\ndef ", 1)[0]


# ------------------------------------------------------- the share is decided by structure, not names


def _share(mf, stage):
    import cc_optimize.summary as S

    return S._roofline_stage_share(mf, stage)


def test_a_mapped_stage_is_priced_from_its_own_subtree(monkeypatch):
    import cc_optimize.summary as S

    monkeypatch.setattr(S, "_SECTION_BYTES", {"audio_tower": 1000, "language_model": 9000})
    mf = {"stage_roots": {"encode": "audio_tower", "decode": "language_model"}}
    assert abs(_share(mf, "encode") - 0.1) < 1e-9
    assert abs(_share(mf, "decode") - 0.9) < 1e-9


def test_an_unmapped_stage_on_a_multi_tower_model_is_refused_whatever_it_is_called(monkeypatch):
    """The old rule refused `encode` and handed `decode` the whole model. Both are the same error:
    a stage charged for a tower it never reads."""
    import cc_optimize.summary as S

    monkeypatch.setattr(S, "_SECTION_BYTES", {"audio_tower": 1000, "language_model": 9000})
    for stage in ("encode", "decode", "prefill", "anything"):
        assert _share({"blocks": {"a": {}, "b": {}}}, stage) == 0.0, stage


def test_a_single_tower_model_prices_every_stage_at_the_whole_model(monkeypatch):
    import cc_optimize.summary as S

    monkeypatch.setattr(S, "_SECTION_BYTES", {"model": 9000})
    for stage in ("decode", "encode", "denoise"):
        assert _share({"blocks": {"model": {}}}, stage) == 1.0, stage


def test_no_evidence_of_towers_is_not_evidence_of_two(monkeypatch):
    """Refusing on an empty map would strip the ceiling off every model whose census never ran."""
    import cc_optimize.summary as S

    monkeypatch.setattr(S, "_SECTION_BYTES", {})
    assert _share({}, "decode") == 1.0
    assert _share({}, "encode") == 1.0


def test_the_share_does_not_consult_the_byte_model_for_a_verdict():
    """It called active_bytes and used acceptance/refusal as the answer. With the name gate gone that
    returns 1.0 for everything, so the coupling must be gone too, not merely harmless."""
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index("def _roofline_stage_share(")
    body = src[i : src.index("\ndef ", i + 1)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "active_bytes" not in code, "the share is deciding structure by whether the byte model raises"


# ----------------------------------------------------------------- the item count is per stage


def test_a_stage_can_state_how_many_items_it_retires():
    """<stage>_trace_items() -- the same optional seam as _trace_inputs. Only the pipeline knows an
    encoder's frame count; it is not the prompt length and cannot be derived from the byte model."""
    from agent.perf_adapter import _Stage

    assert _Stage("encode", lambda: None, items=1500).items == 1500
    assert _Stage("decode", lambda: None).items == 0, "an unstated count must not read as one"


def test_the_measured_stage_prints_its_item_count_beside_its_time():
    src = (_PA / "agent" / "trace_replay.py").read_text()
    i = src.index('print("TRACE_STAGE_MS[')
    assert "TRACE_STAGE_ITEMS[" in src[i : i + 900], "the count is not emitted with the measurement"


def test_the_parser_records_the_count_for_whatever_stage_stated_it():
    """`stage_isl["prefill"] = _iv` was the only writer, so one stage in any model could have a count
    and the reader's {stage: items} map was a one-key map by construction."""
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "stage_isl[_nm] = _nv" in code, "the count is still keyed by a hardcoded stage name"
    assert 'stage_isl["prefill"] = _iv' not in code, "the hardcoded writer is back"
    assert 'stage_isl.setdefault("prefill"' in code, "the legacy marker no longer feeds prefill"


def test_a_stated_count_wins_over_the_legacy_marker():
    """setdefault, so PERF_ISL_TOKENS remains the answer for a generated test that predates
    _trace_items, and yields the moment the stage states its own."""
    isl = {}
    isl["prefill"] = 4096  # TRACE_STAGE_ITEMS, whichever order the lines arrived in
    isl.setdefault("prefill", 128)  # PERF_ISL_TOKENS
    assert isl["prefill"] == 4096


# ------------------------------------------------------------------- a host row is not a stage


def test_host_overhead_is_not_tagged_with_a_real_stage_name():
    """summary selects buckets by `stage or regime`, so `regime: decode` on a synthetic host row is
    matchable as a decode measurement."""
    src = (_PA / "agent" / "tracy_tool.py").read_text()
    i = src.index('"id": "host_overhead"')
    block = src[i : i + 1200]
    assert '"regime": "decode"' not in block
    assert '"regime": "na"' in block


# --------------------------------------------- a model with no decode is not told to fix its decode


def test_the_kv_cache_gate_asks_whether_the_model_decodes_at_all(monkeypatch):
    """_decode_gate BLOCKS a run until a KV-cache lever lands, from `decode_status ==
    "repeat_prefill"` -- which trace_replay emits whenever its capture was SKIPPED, on any pipeline.
    A classifier exposes no traceable step, skips, and was then ordered to add a cached single-token
    decode step to a model that emits no tokens, with no way to clear it but the attempt cap."""
    import cc_optimize.perf_mcp as PM

    prof = {"decode_status": "repeat_prefill"}
    monkeypatch.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "inference")
    assert PM._decode_gate(prof, []) is None, "a one-pass model was ordered to add a KV-cache"
    monkeypatch.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "step")
    assert PM._decode_gate(prof, []) is None, "a diffusion model was ordered to add a KV-cache"


def test_the_kv_cache_gate_still_fires_for_a_model_that_does_decode(monkeypatch):
    """The gate is the whole reason repeat-prefill pipelines get fixed; it must not go quiet."""
    import cc_optimize.perf_mcp as PM

    monkeypatch.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "token")
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    assert PM._decode_gate({"decode_status": "repeat_prefill"}, []) is not None


def test_an_unrecorded_unit_does_not_silence_the_gate(monkeypatch):
    """A run predating the marker must keep the old behaviour: unknown is not "no decode"."""
    import cc_optimize.perf_mcp as PM

    monkeypatch.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "")
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    monkeypatch.setattr(PM, "_reliable_forward_unit", lambda: "")
    assert PM._decode_gate({"decode_status": "repeat_prefill"}, []) is not None


# ------------------------------------------------- the fallback row is named for the measured unit


def test_a_model_that_declares_nothing_is_not_handed_two_llm_stages():
    """The docstring of test_the_roofline_prices_the_stages_the_model_declares calls this the
    original bug -- "a model with NO decode was still handed a DECODE row" -- and it survived in the
    fallback branch."""
    from cc_optimize.summary import _stage_roofs

    assert list(_stage_roofs(1_000_000, 512.0, 1, "img/s", None, None)) == ["inference"]
    assert list(_stage_roofs(1_000_000, 512.0, 1, "step/s", None, None)) == ["step"]


def test_a_token_model_that_declares_nothing_still_gets_its_pair():
    from cc_optimize.summary import _stage_roofs

    assert list(_stage_roofs(1_000_000, 512.0, 1, "tok/s/u", None, None)) == ["prefill", "decode"]


def test_the_end_to_end_line_does_not_describe_every_model_as_an_llm(monkeypatch):
    from cc_optimize.run import _e2e_shape

    monkeypatch.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "inference")
    assert _e2e_shape() == ", 1 forward pass"
    monkeypatch.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "token")
    assert _e2e_shape() == ", prefill + 1 decode"
    monkeypatch.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "")
    assert _e2e_shape() == "", "an unknown shape is described rather than guessed"


def test_the_target_does_not_announce_a_regime_nobody_set():
    from agent.perf_target import PerfTarget
    import inspect

    src = inspect.getsource(PerfTarget)
    assert 'regime: str = "decode"' not in src, "the label defaults to a stage again"
