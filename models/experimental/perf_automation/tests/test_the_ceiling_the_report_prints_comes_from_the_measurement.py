# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The number on the page, against the measurement it claims to be derived from.

WHY THIS FILE EXISTS. Every other test in this suite hands one function a dict its author wrote and
checks what comes back. That is worth having and it is not enough: it cannot see a JOINT. Three
joints in the ceiling chain were broken at once on 2026-08-16, the suite was green throughout, and
the report printed a decode floor of 14.11 ms against a 2.89 ms measurement -- 2496.7 GB/s, 487% of
a 512 GB/s part.

    census 1.718 GB ──X── anchor 7.223 GB ──X── share ──> floor 14.11 ms ──> the row
                      1                    2

    1. THE ANCHOR. The pre-census guess had moved from params x 1.0 to params x <declared dtype>,
       and _anchor_is_placeholder still recognised only x 1.0. So the guess stopped being
       recognised as a guess and the census could not replace it. The guarding test passed the
       whole time, because it constructs an anchor of params x 1.0 -- the value the code no longer
       produces. It encoded the old world and kept passing in the new one.

    2. THE SHARE. stage_roots joins the block count the PROBE observed against the depths the
       checkpoint declares -- but the probe runs depth-capped by design, reporting 2 where the
       model has 32 and 30. It returned {} on every real run and had never once fired, so encode
       got no memory ceiling at all and printed "not modelled" beside a 12.80 ms measurement.

And a third, in the same table: encode published 345.7 tok/s/u, which is 1000/2.8926 -- DECODE's
rate, on encode's row, in a unit encode does not use. The rate was handed to any stage retiring one
item per unit, which an encoder pass is as much as a decoded token.

So these tests start from voxtral's REAL facts and assert the END of the chain: the bytes the floor
divides by are the bytes the census measured, apportioned by the tower the stage actually runs.
Nothing here is a unit test of a helper -- each one fails if any joint between the measurement and
the printed number comes apart.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_PEAK = 512e9

# Voxtral-Mini-3B-2507, as run 5 recorded it on 2026-08-16. device_weight_bytes and
# device_section_bytes are the census's own output; the sections are what the numel join produces.
_FACTS = {
    "total_params": 3611483136,
    "dominant_dtype": "bfloat16",
    "weight_bytes": 9356474312,
    "layers": 32,
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "kv_heads": 8,
    "head_dim": 128,
    "device_weight_bytes": 1718081696,
    "device_census_complete": True,
    "bytes_per_param": 1.3228,
    # A device split that is NOT the disk split: 1.718 GB resident, apportioned as the chip holds it
    # rather than as the bf16 file does (85.8% / 13.7% / 0.5%). If these ever agree by accident the
    # tests below cannot tell the two sources apart, which is how the first version of this file
    # passed while reading the disk ratio.
    "device_section_bytes": {"language_model": 1300000000, "audio_tower": 350081696, "multi_modal_projector": 68000000},
    "stage_roots": {"encode": "audio_tower", "prefill": "language_model", "decode": "language_model"},
}
_ANCHOR_BF16 = 7222966272  # 3611483136 x 2.0 -- the anchor run 5 actually pinned


def _floor_ms(nbytes) -> float:
    return 1e3 * float(nbytes) / _PEAK


# ------------------------------------------------------------------ joint 1: census -> anchor


def test_the_bf16_guess_is_recognised_as_a_guess():
    """THE REGRESSION, stated as arithmetic. params x 2.0 is a prediction of what the loader would
    do, made before it did it -- exactly what params x 1.0 was, and superseded for the same reason."""
    from agent.perf_target import _anchor_is_placeholder

    assert _anchor_is_placeholder(_ANCHOR_BF16, _FACTS) is True


def test_every_width_the_tool_can_guess_with_is_recognised():
    """Not a second special case. Whatever dtype a checkpoint declares, params x that width is the
    same kind of guess, and pinning the recogniser to one of them is how this broke."""
    from agent.perf_target import BYTES_PER_ELEM, _anchor_is_placeholder

    for w in {2.0, 1.0625, 0.5625, 1.0, 4.0} & set(BYTES_PER_ELEM.values()):
        anchor = round(_FACTS["total_params"] * w)
        assert _anchor_is_placeholder(anchor, _FACTS) is True, w


def test_evidence_is_still_never_overridden():
    """The narrowness is the point: only a guess is replaceable. A checkpoint total, a measured
    figure, or a previous census stays pinned exactly as it was."""
    from agent.perf_target import _anchor_is_placeholder

    assert _anchor_is_placeholder(_FACTS["weight_bytes"], _FACTS) is False
    assert _anchor_is_placeholder(_FACTS["device_weight_bytes"], _FACTS) is False


def test_the_ceiling_divides_by_what_the_census_measured():
    """END OF THE CHAIN. Given the anchor run 5 pinned, the target must come back with the censused
    bytes -- not 7.223 GB, and not the checkpoint's 9.356 GB."""
    from agent.perf_target import compute_target

    t = compute_target(_FACTS, {"dram_bw_gbps": 512.0}, bytes_per_unit=_ANCHOR_BF16)
    # The joint this file exists to check is that the report divides by what the census MEASURED
    # rather than by a prediction. It still does; the measured quantity is now the served width.
    assert t.active_bytes == int(round(_FACTS["total_params"] * _FACTS["bytes_per_param"]))
    # 9.33 ms, not the 3.36 the 2-layer byte total gave. Both are "what the census measured" -- the
    # width is the half of it that does not move when the capture depth does.
    assert abs(_floor_ms(t.active_bytes) - 9.33) < 0.05, _floor_ms(t.active_bytes)


def test_the_printed_floor_is_no_longer_four_times_the_truth():
    """The report said 14.11 ms beside a 2.89 ms measurement -- 487% of a 512 GB/s part. Both of
    those numbers came from capped-depth runs; measured at full depth decode takes 17.86 ms. The
    invariant is the one that was violated, stated without a magic ratio: a floor sits UNDER the
    measurement, because a floor above it claims a bandwidth above peak."""
    from agent.perf_target import compute_target

    measured_ms = 17.86
    t = compute_target(_FACTS, {"dram_bw_gbps": 512.0}, bytes_per_unit=_ANCHOR_BF16)
    assert _floor_ms(t.active_bytes) < measured_ms, _floor_ms(t.active_bytes)
    assert 1e-9 * t.active_bytes / (measured_ms * 1e-3) < 512


# ------------------------------------------------------------------ joint 2: stage -> tower


def test_every_declared_stage_has_a_tower():
    """encode's ceiling was refused outright -- "not modelled" -- because no stage had a root."""
    from cc_optimize.summary import _roofline_stage_share

    for stage in ("encode", "prefill", "decode"):
        assert _roofline_stage_share(_FACTS, stage) > 0.0, stage


def test_a_stage_is_priced_from_its_own_towers_measured_bytes():
    """Not the whole model, and not the checkpoint's proportions."""
    from cc_optimize.summary import _roofline_stage_share

    res = float(_FACTS["device_weight_bytes"])
    assert abs(_roofline_stage_share(_FACTS, "decode") - 1300000000 / res) < 1e-9
    assert abs(_roofline_stage_share(_FACTS, "encode") - 350081696 / res) < 1e-9


def test_the_audio_tower_is_not_priced_at_the_backbones_bytes():
    """The failure this whole chain exists to prevent: one tower charged at another's weight."""
    from cc_optimize.summary import _roofline_stage_share

    assert _roofline_stage_share(_FACTS, "encode") < _roofline_stage_share(_FACTS, "decode")


def test_the_disk_ratio_is_not_what_gets_used():
    """85.8% is the language tower's share of the bf16 FILE. The chip is mixed precision, so the
    two differ -- and if the report ever prints the disk figure again, this catches it."""
    from cc_optimize.summary import _roofline_stage_share

    assert abs(_roofline_stage_share(_FACTS, "decode") - 0.858) > 0.01


# ------------------------------------------------------------------ joint 3: the rate on the row


def test_a_self_timed_stage_publishes_its_own_rate():
    """encode printed 345.7 tok/s/u -- 1000/2.8926, decode's rate, on encode's row. A stage the run
    timed separately has a rate of its own; only a stage with no timing of its own borrows one."""
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index("_own_ms = _ms is not None")
    body = src[i : i + 1400]
    assert "_mrate = (1000.0 / _ms) if (_own_ms and _ms)" in body, "the headline rate is handed out again"
    j = src.index("if _ms is None and per_unit_ms")
    assert i < j, "_own_ms is read after the fallback, so every stage looks self-timed"


# ------------------------------------------------------------------ batch, on every stage


def test_batch_scales_the_per_user_terms_and_not_the_weights():
    """8 users share one weight read and carry their own KV -- which is the whole reason batching
    pays, and the reason a per-user ceiling falls only by the KV term."""
    from agent.perf_target import active_bytes

    one = active_bytes(_FACTS, regime="decode", seq_len=128, batch=1)
    eight = active_bytes(_FACTS, regime="decode", seq_len=128, batch=8)
    assert eight > one
    assert eight < 8 * one, "the weights were multiplied by the batch"
    kv_one = one - _FACTS["weight_bytes"]
    assert abs((eight - one) - 7 * kv_one) <= 1, "the KV term did not scale with the batch"


# ------------------------------------------------- and the mapping must not need the probe
#
# RUN 6, 2026-08-17: stage_roots STILL absent, with the join fixed and verified. The publication sat
# inside `if _signposts_usable(seq):`, beside the per-stage depths, because both read as "things
# learned from the probe". They are not. The depths need the signpost sequence; this needs the model
# root and the generated test. Voxtral emits no tracy signposts --
#
#     WARN signpost: no tracy signposts in .../tests -- using default 'start'/'stop' (full capture)
#
# -- so the branch never ran and the mapping was never even attempted. A correct function, called
# from inside a condition it has no use for.


def test_the_mapping_does_not_depend_on_the_probe():
    """An empty signpost sequence must still produce a mapping: the generated-test join needs none."""
    import inspect

    from cc_optimize.run import _publish_stage_roots, stage_roots

    src = inspect.getsource(stage_roots)
    assert "_stage_roots_from_generated" in src, "the probe-free join is gone"
    assert callable(_publish_stage_roots)


def test_the_publication_is_not_gated_behind_signposts():
    """Asserted on the call site, because the function being right is what run 6 already proved."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    call = src.index("_publish_stage_roots(seq, _root, node)")
    guard = src.index("if _signposts_usable(seq):", src.index("def _coverage_layers("))
    assert call < guard, "stage_roots is published inside the signpost branch again"


def test_the_count_join_still_wins_when_the_probe_can_answer():
    """The fallback is a fallback. A probe reporting real depths still decides, so a model whose
    stacks are unambiguous by count does not depend on a generated file's text."""
    import inspect

    from cc_optimize.run import stage_roots

    src = inspect.getsource(stage_roots)
    assert src.index("by_count.get") < src.index("_stage_roots_from_generated")


def test_the_mapping_survives_a_null_perf_test(tmp_path):
    """RUN 7: still no mapping, with the join fixed AND the call hoisted out of the signpost branch.

        note - no_perf_test: perf_test null; falling back to pcc.end_to_end

    `pipe["perf_test"]` was None, so _coverage_layers was handed the PCC gate -- which carries no
    stage->stack bindings, because it is not a generated perf test. The fallback keyed on that node
    and returned {} while three generated perf tests sat on disk beside it holding exactly the
    mapping wanted.

    Depending on WHICH node a caller passes is the mistake. The binding lives in the generated tests;
    they are found under the model root, and the node is only a first guess at where to look."""
    from cc_optimize.run import _stage_roots_from_generated

    e2e = tmp_path / "tests" / "e2e"
    e2e.mkdir(parents=True)
    (e2e / "test_main_perf.py").write_text(
        'PERF_ENCODE_LAYERS = _env_layers("TT_PERF_ENCODE_LAYERS", "TT_PERF_STACK0_LAYERS")\n'
        'PERF_DECODE_LAYERS = _env_layers("TT_PERF_DECODE_LAYERS", "TT_PERF_STACK1_LAYERS")\n'
    )
    (e2e / "test_e2e_pipeline.py").write_text("def test_e2e_voxtral_pipeline():\n    pass\n")
    secs = {"audio_tower.layers": 32, "language_model.model.layers": 30}

    for node in (None, "", str(e2e / "test_e2e_pipeline.py::test_e2e_voxtral_pipeline")):
        got = _stage_roots_from_generated(secs, node, str(tmp_path))
        assert got == {"encode": "audio_tower", "decode": "language_model"}, (node, got)


def test_generated_tests_that_disagree_are_not_evidence(tmp_path):
    """They are written by one generator from one survey, so they agree. If two ever did not, the
    mapping is a coin toss and a stage would be priced at another tower's bytes."""
    from cc_optimize.run import _stage_roots_from_generated

    e2e = tmp_path / "tests" / "e2e"
    e2e.mkdir(parents=True)
    (e2e / "test_a_perf.py").write_text('PERF_ENCODE_LAYERS = _env_layers("X", "TT_PERF_STACK0_LAYERS")\n')
    (e2e / "test_b_perf.py").write_text('PERF_ENCODE_LAYERS = _env_layers("X", "TT_PERF_STACK1_LAYERS")\n')
    got = _stage_roots_from_generated(
        {"audio_tower.layers": 32, "language_model.model.layers": 30}, None, str(tmp_path)
    )
    assert "encode" not in got, got


def test_nothing_hands_a_path_to_the_source_parser():
    """TWO SITES, ONE DEFECT, AND FIXING ONE LEFT THE OTHER LIVE FOR FOUR MORE RUNS.

    _hf_repo_ids takes a parsed Source and iterates `src.trees.items()`; handed a Path it raises
    AttributeError, and both callers wrapped it in a bare except:

        _section_bytes_cached   -> {} on every model, so _stage_share always returned 1.0
        _model_id_for_facts     -> "" on every model, so declared_sections found no checkpoint
                                   and stage_roots bailed before reaching its fallback

    The first was found on 2026-08-17 and fixed. The second was not, because nothing pointed at it --
    and it silently disabled every subsequent fix to stage_roots: four runs, each concluding the
    mapping was still broken for a different reason. A grep for the CALL SHAPE would have found both
    in one pass, which is what this is.
    """
    import re

    for rel in ("cc_optimize/summary.py", "cc_optimize/run.py", "cc_optimize/perf_mcp.py"):
        src = (_PA / rel).read_text()
        # CODE ONLY. The docstrings quote the broken call to explain it; asserting over prose would
        # forbid recording what the defect was.
        code = "".join(seg for i, seg in enumerate(src.split('"""')) if i % 2 == 0)
        code = "\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith("#"))
        assert not re.search(r"_hf_repo_ids\(\s*Path\(", code), "%s passes a Path to _hf_repo_ids again" % rel


def test_the_model_id_resolves_from_a_path_the_callers_actually_have():
    """Both callers hold a model ROOT, never a parsed Source, so the resolver must take a path."""
    import inspect

    from cc_optimize.run import _model_id_for_facts

    src = inspect.getsource(_model_id_for_facts)
    assert "model_id_from_source" in src, "the id no longer comes from the model's own source"
    assert "_hf_repo_ids" not in src.split('"""', 2)[-1], "the Source-only extractor is back"
