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
    # BEHAVIOUR CHANGE: this used to REQUIRE the legacy marker to be filed under _LEGACY_PROMPT_KEY,
    # as the fallback for a stage stating no count of its own. That made a workload fact reachable
    # through one typed name -- one stage per model could be sized, and only if it was called that,
    # while every other stage fell back to a single item. summary._stage_items_observed now reads the
    # count off the matmuls each stage actually ran, so no stage needs to be named to be sized.
    assert (
        "stage_isl_per_request.setdefault(_LEGACY_PROMPT_KEY" not in code
    ), "a stage is being sized by a typed name again"
    _sum = (_PA / "cc_optimize" / "summary.py").read_text()
    assert "def _stage_items_observed(" in _sum, "nothing derives the count the removed writer supplied"
    # The constant survives for ONE reason: reading a doc written before `prompt_tokens` existed. That
    # is an on-disk schema key, not a claim about this model's stages, and it cannot mis-price -- a
    # doc without it simply yields 0.
    assert '_LEGACY_PROMPT_KEY = "prefill"' in code, "the on-disk compat key is no longer named once"
    _reads = [ln for ln in code.splitlines() if "_LEGACY_PROMPT_KEY" in ln and "=" not in ln.split("_LEGACY")[0]]
    assert all("_doc.get" in ln or "def " in ln or "_LEGACY_PROMPT_KEY =" in ln for ln in _reads), _reads


def test_a_stated_count_is_a_total_and_is_not_multiplied_by_the_batch(monkeypatch):
    """TWO UNITS. A count a stage states through <stage>_trace_items() is the total for ONE CALL --
    voxtral's prefill_trace_items returns PREFILL_C * B, and its encode traces at batch 1 whatever
    the pipeline serves. The legacy PERF_ISL_TOKENS marker is the prompt length PER REQUEST.

    Read from one map and multiplied uniformly, prefill was counted at 8x its real work and encode
    was given a batch it does not have -- and the compute ceiling is 2 x params x this."""
    import cc_optimize.summary as S
    import cc_optimize.perf_mcp as PM

    monkeypatch.setattr(S, "_request_batch", lambda: 8)
    monkeypatch.setattr(PM, "read_stage_isl_map", lambda *a, **k: {"prefill": 1024, "encode": 1500})
    monkeypatch.setattr(PM, "read_stage_isl_per_request_map", lambda *a, **k: {})

    assert S._stage_units("prefill", 128) == 1024, "a stated total was multiplied by the batch"
    assert S._stage_units("encode", 128) == 1500, "encode was given a batch it does not run at"


def test_the_legacy_per_request_marker_is_still_multiplied(monkeypatch):
    """A generated test that predates _trace_items prints only PERF_ISL_TOKENS, per request. That
    behaviour must not change: 128 tokens at batch 8 is 1024 items in one unit of work."""
    import cc_optimize.summary as S
    import cc_optimize.perf_mcp as PM

    monkeypatch.setattr(S, "_request_batch", lambda: 8)
    monkeypatch.setattr(PM, "read_stage_isl_map", lambda *a, **k: {})
    monkeypatch.setattr(PM, "read_stage_isl_per_request_map", lambda *a, **k: {"prefill": 128})

    assert S._stage_units("prefill", 128) == 1024


def test_a_stated_total_wins_over_the_legacy_marker(monkeypatch):
    import cc_optimize.summary as S
    import cc_optimize.perf_mcp as PM

    monkeypatch.setattr(S, "_request_batch", lambda: 8)
    monkeypatch.setattr(PM, "read_stage_isl_map", lambda *a, **k: {"prefill": 512})
    monkeypatch.setattr(PM, "read_stage_isl_per_request_map", lambda *a, **k: {"prefill": 128})

    assert S._stage_units("prefill", 128) == 512


def test_a_stage_nobody_counted_retires_one(monkeypatch):
    import cc_optimize.summary as S
    import cc_optimize.perf_mcp as PM

    monkeypatch.setattr(PM, "read_stage_isl_map", lambda *a, **k: {})
    monkeypatch.setattr(PM, "read_stage_isl_per_request_map", lambda *a, **k: {})
    assert S._stage_units("decode", 128) == 1


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


# ------------------------------------- "is this model generative" is a contract, not a substring


def test_a_demo_is_not_generative_because_it_contains_a_for_loop(tmp_path):
    """`for _ in range` was one of six substrings that classed a demo generative, and it appears in
    very nearly every Python file. Almost every model was therefore told its perf test must cap a
    decode loop, and the test was regenerated until it did -- a requirement a classifier cannot
    satisfy honestly, burning correction rounds on it."""
    from agent.perf_test_gen import _pipeline_is_generative

    (tmp_path / "tt").mkdir()
    (tmp_path / "tt" / "pipeline.py").write_text(
        "class P:\n"
        "    def forward(self, x):\n"
        "        for _ in range(4):\n"
        "            x = self.block(x)\n"
        "        return x\n"
    )
    demo = "for _ in range(10):\n    out = pipe.forward(img)\n"
    assert _pipeline_is_generative(tmp_path, demo) is False


def test_a_pipeline_keeping_the_decode_contract_is_generative(tmp_path):
    from agent.perf_test_gen import _pipeline_is_generative

    (tmp_path / "tt").mkdir()
    (tmp_path / "tt" / "pipeline.py").write_text(
        "class P:\n    def decode_prefill(self, ids): ...\n    def decode_step(self, state): ...\n"
    )
    assert _pipeline_is_generative(tmp_path, "") is True


def test_an_unreadable_source_falls_back_to_generation_apis_only(tmp_path):
    """The fallback keeps the two markers that ARE generation APIs and drops the four that are
    incidental words."""
    from agent.perf_test_gen import _pipeline_is_generative

    assert _pipeline_is_generative(tmp_path / "nope", "out = m.generate(x, max_new_tokens=32)") is True
    assert _pipeline_is_generative(tmp_path / "nope", "for _ in range(3): y = net(x)") is False
    assert _pipeline_is_generative(tmp_path / "nope", "next_token = argmax(logits)") is False


# ------------------------------------- the recurring stage reports itself; nothing matches its name


def test_a_stage_that_retires_one_item_is_the_recurring_one():
    """Derived, not matched. `recurring` is what the headline is per, and it used to be found by
    `"decode" in name.lower()`: a loop called `generate` read as one-pass, and any stage called
    decode read as autoregressive whether it looped or not."""
    from agent.perf_adapter import _Stage

    assert _Stage("generate", lambda: None, items=1).recurring is True
    assert _Stage("encode", lambda: None, items=1500).recurring is False
    assert _Stage("decode", lambda: None).recurring is False, "a name must not make a stage recurring"


def test_the_legacy_decode_contract_states_its_own_count():
    """decode_step(state) retires one token per call by definition, so the legacy path feeds the same
    derived machinery the declared path does instead of relying on a fallback."""
    from agent.trace_replay import _LegacyStage

    st = _LegacyStage(type("A", (), {"step": staticmethod(lambda: None)})())
    assert st.items == 1 and st.recurring is True


def test_the_headline_stage_is_chosen_with_no_name_read_at_all():
    """The flag used to merely run FIRST, with the name match kept underneath it as a fallback. A
    fallback that guesses is still a guess -- it just waits its turn -- so there is now no name test
    on any path: the stage comes from `recurring`, which the pipeline reports, and the unit from
    headline_unit, which derives it."""
    src = (_PA / "agent" / "trace_replay.py").read_text()
    body = src[src.index("_rec = {st.name for st in stages") :]
    body = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    for guess in ('"decode" in', '"denoise"', '"diffus"', ".lower() for k in"):
        assert guess not in body, "a name guess survives in the headline selection: %s" % guess


def test_a_pipeline_may_declare_its_own_unit():
    """PIPELINE_UNIT is the model stating its unit the way PIPELINE_STAGES states its stages -- the
    only source that cannot be wrong about a model nobody anticipated."""
    from agent.perf_adapter import headline_unit

    class _P:
        PIPELINE_UNIT = "step"

        def decode_step(self, state):  # would otherwise force "token"
            return state

    assert headline_unit(["decode"], _P()) == "step"


def test_the_decode_contract_still_answers_when_nothing_is_declared():
    from agent.perf_adapter import headline_unit

    class _P:
        def decode_step(self, state):
            return state

    assert headline_unit(["whatever"], _P()) == "token"
    assert headline_unit(["classify"], None) == "inference"


# ------------------------------------------- the lever catalogue can name a stage the model has


def test_a_lever_can_be_tagged_for_a_stage_the_model_actually_declares():
    """The regime axis was frozenset({"prefill", "decode", "na"}), so a lever written for an audio
    encoder could not be TAGGED -- `regime: encode` failed validation and the lever was flagged
    out-of-vocabulary. A model with no decode at all was still routed by a vocabulary that knows
    only decode."""
    from agent import router

    router._DECLARED_STAGES.clear()
    router.declare_stages(["encode", "prefill", "decode"])
    for st in ("encode", "prefill", "decode", "na"):
        router._validate_query({"regime": st})  # must not raise
    try:
        router._validate_query({"regime": "vocode"})
    except ValueError:
        pass
    else:
        raise AssertionError("an undeclared stage was accepted as a regime")
    router._DECLARED_STAGES.clear()


def test_a_model_with_no_decode_does_not_have_decode_in_its_vocabulary():
    """The point: the axis follows the model. Asking for `decode` on a classifier is now a rejected
    query rather than a silently valid one."""
    from agent import router

    router._DECLARED_STAGES.clear()
    router.declare_stages(["classify"])
    router._validate_query({"regime": "classify"})
    try:
        router._validate_query({"regime": "decode"})
    except ValueError:
        pass
    else:
        raise AssertionError("decode is still valid for a model that has no decode")
    router._DECLARED_STAGES.clear()


def test_an_undeclared_axis_stays_open_but_still_refuses_junk():
    """Before any model has spoken -- indexing the catalogue, a unit test -- any identifier is
    accepted, because not knowing the stages is not grounds to reject them."""
    from agent import router

    router._DECLARED_STAGES.clear()
    router._validate_query({"regime": "anything_at_all"})
    try:
        router._validate_query({"regime": "NOT AN IDENT!"})
    except ValueError:
        pass
    else:
        raise AssertionError("junk was accepted on the open axis")


def test_the_stage_list_has_one_reader():
    """Three places walked the AST for PIPELINE_STAGES independently; anything else that wanted the
    answer either copied the walk or guessed."""
    from agent.model_contract import declared_stage_names

    assert declared_stage_names("/nonexistent") == []


def test_the_trace_knob_is_derived_per_stage():
    """One TT_PERF_PREFILL_TRACE was set for every model, so a pipeline whose stages are
    encode/vocode had no way to be told to trace them -- the flag names the stage."""
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert 'env["TT_PERF_%s_TRACE" % str(_st).upper()] = "1"' in code
    assert 'env["TT_PERF_PREFILL_TRACE"] = "1"' in code, "an older generated test lost its trace"


def test_an_encoder_lever_is_reachable_from_an_encoder_stage():
    """A lever written for the VIT PATTERN -- an image encoder -- was tagged `regime: prefill,na`,
    so a stage called `encode` could never be routed to it. Not because anyone decided encoders
    should not have it: those were the only two values a lever could carry.

    The lever's real condition is in its own body -- the activation fits L1, per_core_M is the
    SEQUENCE rather than the batch -- which is a property of the work. The shape filters the
    recurring stage out; the name never should have."""
    from agent import router

    idx = router.build_index()
    assert not router.index_warnings(idx), router.index_warnings(idx)[:3]

    router._DECLARED_STAGES.clear()
    router.declare_stages(["encode", "prefill", "decode"])
    seen = {
        st: {h["id"] for h in router.route(idx, {"op_class": "matmul", "regime": st})}
        for st in ("encode", "prefill", "decode")
    }
    router._DECLARED_STAGES.clear()

    assert "mlp-program-config" in seen["encode"], "the ViT lever is still invisible to an encoder"
    # and the genuinely stage-specific ones stay narrow
    assert "decode-host-comm" in seen["decode"] and "decode-host-comm" not in seen["encode"]


# ------------------------------------------------------------- the seam must reach its PRODUCERS


def test_the_generator_asks_for_every_seam_the_engine_binds():
    """THE HALF-BUILT FIX, and the test that would have caught it.

    `_trace_items` was added in August to the adapter, the marker, the parser and the renderer --
    every CONSUMER -- and to nothing that makes a model. The emit-e2e prompt lists the seams a
    pipeline must expose, and it was not updated, so no model has ever emitted the marker: the
    reader worked perfectly on a value nobody ever sent. Voxtral's encoder was therefore priced at
    one item instead of 1500 and reported memory-bound while being compute-bound, and the suite
    stayed green because the unit test constructs `_Stage(..., items=1500)` by hand -- proving the
    consumer, never the chain.

    This asserts the two ends agree, which is the only thing that fails when a seam is half-added.
    """
    from agent import stage_seams

    prompt = (_PA.parent.parent.parent / "scripts" / "tt_hw_planner" / "commands" / "emit_e2e.py").read_text()
    missing = [s for s in stage_seams.ALL if s not in prompt]
    assert not missing, "emit-e2e never tells a model to write %s, so no model will ever have it" % missing


def test_the_contract_asks_for_every_seam_the_engine_binds():
    """The other producer-side end: a model already emitted can only learn of a new seam here."""
    from agent import stage_seams

    src = (_PA / "agent" / "model_contract.py").read_text()
    missing = [s for s in stage_seams.ALL if s not in src and ("_seams" not in src)]
    assert not missing, "the contract never asks for %s, so an existing model is never told" % missing


def test_an_unstated_item_count_is_reported_and_never_blocks(tmp_path):
    """Reported so a placeholder ceiling cannot pass for a measurement; porting so the direct path
    -- hand-written models that never went through emit-e2e -- is never refused for it."""
    from agent.model_contract import check

    root = tmp_path / "m"
    (root / "tt").mkdir(parents=True)
    # The stage name is the MODEL'S, read back from what it declares; nothing here is a known name.
    (root / "tt" / "pipeline.py").write_text(
        "PIPELINE_STAGES = ['whatever_this_model_calls_it']\n"
        "def build_pipeline(device, model=None, layers=None, **kwargs): return object()\n"
        "def whatever_this_model_calls_it_trace_setup(i): ...\n"
        "def whatever_this_model_calls_it_trace_step(): ...\n"
    )
    found = [f for f in check(root) if f.clause == "stage-items"]
    assert found, "an unstated item count is silent again"
    assert "whatever_this_model_calls_it" in found[0].detail, "the clause did not use the model's own name"
    assert not any(f.blocking for f in found), "a missing optional seam must not refuse the direct path"


# ------------------------------------------------- the item count comes off what the stage RAN


def _prof(**stages):
    return {"stage_buckets": {k: [{"top_ops": v}] for k, v in stages.items()}}


def test_a_stage_states_its_item_count_through_the_matmuls_it_ran():
    """THE COUNT WITHOUT A NAME. A stage that states no <stage>_trace_items() was priced at ONE item
    unless a workload marker happened to be filed under its name -- so exactly one stage per model
    could be sized, and only if it was called the name the tool typed. A matmul's M is the rows the
    stage pushed through, which is the same quantity the seam states."""
    from cc_optimize.summary import _stage_items_observed as obs

    p = _prof(
        zzz_tower=[{"shape": "1500x1280 @ 1280x5120", "count": 32}],
        zzz_prompt=[{"shape": "4096x3072 @ 3072x8192", "count": 30}],
    )
    assert obs("zzz_tower", p) == 1500
    assert obs("zzz_prompt", p) == 4096
    # nothing typed: stages named anything at all are sized the same way
    assert obs("zzz_absent", p) == 0


def test_the_padded_vocab_head_does_not_set_the_item_count():
    """THE REGRESSION THE FLOP RANKING WOULD HAVE SHIPPED. Ranking by FLOPs picks the widest single
    matmul, and on a decode step that is the vocab head -- running at a TILE-PADDED 32 rows for a
    batch of 8. The stage would have read as retiring 32 items and been labelled a request-rate
    stage instead of one token per user. The modal row count carries the true figure."""
    from cc_optimize.summary import _stage_items_observed as obs

    p = _prof(
        zzz_step=[
            {"shape": "8x3072 @ 3072x8192", "count": 30},
            {"shape": "32x3072 @ 3072x131072", "count": 1},  # the padded head, widest by far
        ]
    )
    assert obs("zzz_step", p) == 8


def test_an_unparseable_stage_gets_no_count_rather_than_a_wrong_one():
    """A wrong divisor is worse than a missing one -- the rule stage_roots already follows when two
    sections have equal depth. 0 means the caller keeps its own fallback."""
    from cc_optimize.summary import _stage_items_observed as obs

    assert obs("zzz", _prof(zzz=[{"shape": "?", "count": 3}])) == 0
    assert obs("zzz", None) == 0
    assert obs("zzz", {}) == 0
    assert obs(None, _prof(zzz=[{"shape": "8x8 @ 8x8", "count": 1}])) == 0


def test_a_tile_padded_row_count_is_not_the_item_count():
    """WHAT TRACY ACTUALLY WRITES. The shape fingerprint carries the PADDED dim -- _op_shape builds it
    from _pad(), which keeps the kernel's computed size and drops the logical one. A step retiring one
    row per user pads 8 rows to a 32-row tile, so EVERY matmul in it reads 32 and the stage would look
    like it retires 32 items: not one per user, so not a per-user rate. The logical count rides beside
    the fingerprint as `rows`, and that is what an item count means."""
    from cc_optimize.summary import _stage_items_observed as obs

    padded = _prof(
        zzz_step=[
            {"shape": "32x3072 @ 3072x8192", "rows": 8, "count": 30},
            {"shape": "32x3072 @ 3072x131072", "rows": 8, "count": 1},
        ]
    )
    assert obs("zzz_step", padded) == 8, "the padded tile height became the item count"
    # a tower whose rows are not a tile multiple: 1500 asked for, 1504 computed
    assert obs("zzz_tower", _prof(zzz_tower=[{"shape": "1504x1280 @ 1280x5120", "rows": 1500, "count": 32}])) == 1500
    # a profile written before `rows` existed still parses, from the fingerprint
    assert obs("zzz_old", _prof(zzz_old=[{"shape": "8x3072 @ 3072x8192", "count": 30}])) == 8


def test_the_logical_dim_is_carried_out_of_the_raw_row():
    """_pad and _logical split '32[8]' the two ways it is needed: the kernel's size for bytes and the
    fingerprint, the asked-for size for counting items."""
    from agent.tracy_tool import _logical, _pad

    assert (_pad("32[8]"), _logical("32[8]")) == ("32", "8")
    assert (_pad("4096"), _logical("4096")) == ("4096", "4096")
    assert _logical("") == "?" and _logical(None) == "?"
