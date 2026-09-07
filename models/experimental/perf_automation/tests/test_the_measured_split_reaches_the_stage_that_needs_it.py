# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Three defects found in RUN 10's own artifacts, while it was still running.

The run produced, for the first time ever, both things the roofline had been waiting for -- and
neither was usable:

    stage_roots           {'encode': 'audio_tower'}
    device_section_bytes  {'embed':…, 'lm_layers':…, 'enc_a':…, 'kv':…, 'mlp':…, 'attn':…}

1. stage_roots named ONE stage of three. The count join resolved encode and nothing else, and
   `return out or _stage_roots_from_generated(...)` falls back only on a COMPLETELY empty result --
   so a partial answer suppressed a complete one. The generated test names all three unambiguously.
   prefill and decode were left unmapped on a two-tower model, which is refused rather than guessed,
   so the two heaviest stages would have lost their memory ceiling entirely.

2. device_section_bytes carried no `audio_tower` and no `language_model`. census() records a
   subtree's bytes under two names -- the attribute it was reached through, and the checkpoint
   section it came from -- and only the second can be looked up by a stage_roots entry. The
   checkpoint argument was never passed by the caller, so only the first existed. The measured split
   was present, correct, and unusable.

3. And passing it was not enough: the checkpoint readers glob `<arg>/*.safetensors`, so a HUB ID --
   a relative path that does not exist -- found nothing and returned empty. Not an error: no
   tensors, indistinguishable from a checkpoint with none.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


# THE JOIN DOES NOT CARE WHICH MODEL THIS IS. The two cases below exercise how two sources of
# stage->root are MERGED; the id is only an argument on the way through. Passing a real one made them
# resolve it against the local HF cache, so they passed on a machine that had voxtral downloaded and
# failed everywhere else -- while their two siblings in this file already skip when the cache is
# empty. A skip would have been the pattern; it also stops the join being tested at all, on every
# machine without the weights. Stubbing the one call that reaches the cache keeps them running.
_ANY_ID = "org/some-model"

# NAMES THIS TEST INVENTED. The merge is about MAPPING, not about which model is being mapped, and
# stage/root names typed in a case are the same defect as stage names typed in the tool: a join that
# quietly special-cased a real tower would pass a case written in that tower's vocabulary. Nothing
# here appears in any model.
_TOWER_A, _TOWER_B = "tower_alpha", "tower_beta"
_STAGE_1, _STAGE_2, _STAGE_3 = "stage_one", "stage_two", "stage_three"
_BLOCKS_A, _BLOCKS_B = 32, 30


def _a_cached_model():
    """SOME model the local cache holds, whichever it is, or "" when it holds none.

    The bugs below are not voxtral's. A reader that globs <arg>/*.safetensors returns empty for ANY
    hub id, and a census that cannot find the checkpoint from a pipeline fails for ANY model -- so
    naming one here narrowed a universal check to a machine that had downloaded that specific model,
    and told the next model nothing. Discovered from the cache instead: the case runs wherever there
    is anything to read, and skips only on a machine with an empty cache.
    """
    import re as _re

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for _d in sorted(hub.glob("models--*--*")) if hub.is_dir() else []:
        if not any(_d.glob("snapshots/*/*.safetensors")):
            continue
        _m = _re.match(r"models--(.+?)--(.+)$", _d.name)
        if _m:
            return "%s/%s" % (_m.group(1), _m.group(2))
    return ""


def _trace_replay():
    """agent.trace_replay imports ttnn at module scope, so it cannot be imported without a device
    build. The function under test touches no device; a stand-in module is enough to reach it."""
    import sys as _sys
    import types

    _sys.modules.setdefault("ttnn", types.ModuleType("ttnn"))
    from agent import trace_replay as TR

    return TR


def test_a_partial_count_join_does_not_suppress_the_generated_one():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def stage_roots(")
    body = src[i : src.index("\ndef ", i + 1)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "out or _stage_roots_from_generated" not in code, "a partial mapping wins again"
    assert "setdefault" in code, "the two joins are not merged per stage"


def _stub_sections(monkeypatch, R):
    """The only call in stage_roots that reads the checkpoint, so the join can be tested without one.

    declared_sections(model_root, model_id) resolves the id against the HF cache. Everything else
    these two cases touch is already stubbed; this was the one path left reaching outside the test.

    Patched on its OWN module, not on run: stage_roots imports it inside the function body, so the
    name never exists as an attribute of run to replace.
    """
    from agent import checkpoint_sections

    # {path: block count}, shaped so the COUNT JOIN can actually answer: the stack below carries
    # _BLOCKS_A, and exactly one section has that count, so it resolves to _TOWER_A. A stub whose
    # counts are not unique leaves the count join empty and the cases stop testing the merge they
    # exist for -- which is what a first attempt here did.
    monkeypatch.setattr(
        checkpoint_sections,
        "declared_sections",
        lambda root, model_id="": {"%s.layers" % _TOWER_A: _BLOCKS_A, "%s.layers" % _TOWER_B: _BLOCKS_B},
    )


def test_the_two_joins_merge_per_stage(monkeypatch):
    """Run 10 exactly: the count join reaches encode only; the generated test names all three."""
    import cc_optimize.run as R

    _stub_sections(monkeypatch, R)
    monkeypatch.setattr(R, "stacks_by_stage", lambda seq: {_STAGE_1: ["s0"]})
    monkeypatch.setattr(R, "_stack_paths", lambda seq: [("s0", _BLOCKS_A, "k")])
    monkeypatch.setattr(
        R,
        "_stage_roots_from_generated",
        lambda secs, perf_test, model_root=None: {
            _STAGE_1: _TOWER_A,
            _STAGE_2: _TOWER_B,
            _STAGE_3: _TOWER_B,
        },
    )
    got = R.stage_roots(None, "/nonexistent", _ANY_ID, None)
    assert got == {_STAGE_1: _TOWER_A, _STAGE_2: _TOWER_B, _STAGE_3: _TOWER_B}


def test_the_count_join_keeps_its_answer_where_it_has_one(monkeypatch):
    """Merged, not overwritten: a stage the count join established is not re-decided."""
    import cc_optimize.run as R

    _stub_sections(monkeypatch, R)
    monkeypatch.setattr(R, "stacks_by_stage", lambda seq: {_STAGE_1: ["s0"]})
    monkeypatch.setattr(R, "_stack_paths", lambda seq: [("s0", _BLOCKS_A, "k")])
    monkeypatch.setattr(
        R,
        "_stage_roots_from_generated",
        lambda secs, perf_test, model_root=None: {_STAGE_1: "SOMETHING_ELSE", _STAGE_3: _TOWER_B},
    )
    got = R.stage_roots(None, "/nonexistent", _ANY_ID, None)
    assert got[_STAGE_1] == _TOWER_A, "the count join's answer was overwritten by the generated one"
    assert got[_STAGE_3] == _TOWER_B, "a stage the count join could not reach must take the fallback"


# ------------------------------------------------------------- the checkpoint readers take an id


def test_the_checkpoint_readers_accept_a_hub_id_not_only_a_directory():
    """They glob <arg>/*.safetensors. A hub id is a relative path that does not exist, so they
    returned EMPTY -- no tensors, indistinguishable from a checkpoint with none."""
    from agent.weight_census import checkpoint_numels, checkpoint_section_numels
    from agent.checkpoint_sections import hf_cache_dir

    mid = _a_cached_model()
    if not mid or not hf_cache_dir(mid):
        import pytest

        pytest.skip("no model with weights in the local HF cache")

    by_id = checkpoint_section_numels(mid)
    by_dir = checkpoint_section_numels(str(hf_cache_dir(mid)))
    assert by_id == by_dir and by_id, "an id and its cache directory must read the same"
    # WHAT the sections are called is the model's business. That they were FOUND is the bug: a hub id
    # used to read as a directory that does not exist, so this came back empty and was
    # indistinguishable from a checkpoint with no tensors.
    assert all(str(v).strip() for v in by_id.values()), "every tensor must land in a named section"
    assert len(checkpoint_numels(mid)) == len(checkpoint_numels(str(hf_cache_dir(mid))))


def test_an_unresolvable_name_is_still_empty_rather_than_an_error():
    from agent.weight_census import checkpoint_section_numels

    assert checkpoint_section_numels("no-such-org/no-such-model") == {}
    assert checkpoint_section_numels("") == {}


# ------------------------------------------------- the census is told where the checkpoint is


def test_the_census_is_called_with_a_checkpoint():
    src = (_PA / "agent" / "trace_replay.py").read_text()
    i = src.index("_census(\n") if "_census(\n" in src else src.index('scope="pipeline"')
    assert "checkpoint=" in src[max(0, i - 200) : i + 300], "the census records attribute names only again"


def test_the_checkpoint_is_found_from_the_pipeline_itself(monkeypatch, tmp_path):
    """No env var names the model root -- checked against a live run's whole process tree. The
    object being measured knows where it lives: its class's module file sits inside the model dir."""
    import sys as _sys
    import types

    TR = _trace_replay()

    monkeypatch.delenv("PERF_MCP_MODEL_ROOT", raising=False)
    monkeypatch.delenv("TT_PERF_MODEL_ROOT", raising=False)

    # A DEMO BUILT HERE, not one borrowed from the tree. This walked up from the real voxtral demo
    # and asserted the id that demo happens to name, so it skipped wherever that model was absent and
    # proved nothing about any other. The behaviour under test is the WALK -- pipeline module ->
    # its file -> up to the directory whose source names a hub repo -- and that is the same walk for
    # every model. So the fixture states its own id and the case runs everywhere.
    mid = _a_cached_model()
    if not mid:
        import pytest

        pytest.skip("no model with weights in the local HF cache")

    # The id has to be one the cache really holds: model_id_from_source deliberately returns only an
    # id with weights behind it, because a pipeline commonly names several repos and the weights are
    # the one that matters. So the fixture names whatever this machine has -- any model exercises the
    # same walk.
    demo = tmp_path / "some_demo"
    (demo / "tt").mkdir(parents=True)
    (demo / "tt" / "pipeline.py").write_text("class Pipeline:\n    pass\n")
    (demo / "model.py").write_text('MODEL_ID = "%s"\n' % mid)

    mod_name = "a_demo_somewhere.tt.pipeline"
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(demo / "tt" / "pipeline.py")
    monkeypatch.setitem(_sys.modules, mod_name, mod)

    class _Pipe:
        pass

    _Pipe.__module__ = mod_name
    assert TR._checkpoint_for_census(_Pipe()) == mid


def test_no_pipeline_and_no_env_is_none_not_a_crash():
    assert _trace_replay()._checkpoint_for_census(None) is None


def test_the_walk_stops_at_a_repository_boundary():
    """Without the .git guard this could climb out of the model and scan a monorepo."""
    src = (_PA / "agent" / "trace_replay.py").read_text()
    i = src.index("def _checkpoint_for_census(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert '".git"' in body and "break" in body
