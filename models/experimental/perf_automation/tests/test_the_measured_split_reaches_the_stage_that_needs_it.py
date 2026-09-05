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

_MID = "mistralai/Voxtral-Mini-3B-2507"


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


def test_the_two_joins_merge_per_stage(monkeypatch):
    """Run 10 exactly: the count join reaches encode only; the generated test names all three."""
    import cc_optimize.run as R

    monkeypatch.setattr(R, "stacks_by_stage", lambda seq: {"encode": ["s0"]})
    monkeypatch.setattr(R, "_stack_paths", lambda seq: [("s0", 32, "k")])
    monkeypatch.setattr(
        R,
        "_stage_roots_from_generated",
        lambda secs, perf_test, model_root=None: {
            "encode": "audio_tower",
            "prefill": "language_model",
            "decode": "language_model",
        },
    )
    got = R.stage_roots(None, "/nonexistent", _MID, None)
    assert got == {"encode": "audio_tower", "prefill": "language_model", "decode": "language_model"}


def test_the_count_join_keeps_its_answer_where_it_has_one(monkeypatch):
    """Merged, not overwritten: a stage the count join established is not re-decided."""
    import cc_optimize.run as R

    monkeypatch.setattr(R, "stacks_by_stage", lambda seq: {"encode": ["s0"]})
    monkeypatch.setattr(R, "_stack_paths", lambda seq: [("s0", 32, "k")])
    monkeypatch.setattr(
        R,
        "_stage_roots_from_generated",
        lambda secs, perf_test, model_root=None: {"encode": "SOMETHING_ELSE", "decode": "language_model"},
    )
    got = R.stage_roots(None, "/nonexistent", _MID, None)
    assert got["encode"] != "SOMETHING_ELSE" or got["encode"] == "audio_tower"
    assert got["decode"] == "language_model"


# ------------------------------------------------------------- the checkpoint readers take an id


def test_the_checkpoint_readers_accept_a_hub_id_not_only_a_directory():
    """They glob <arg>/*.safetensors. A hub id is a relative path that does not exist, so they
    returned EMPTY -- no tensors, indistinguishable from a checkpoint with none."""
    from agent.weight_census import checkpoint_numels, checkpoint_section_numels
    from agent.checkpoint_sections import hf_cache_dir

    if not hf_cache_dir(_MID):
        import pytest

        pytest.skip("voxtral not in the local HF cache")

    by_id = checkpoint_section_numels(_MID)
    by_dir = checkpoint_section_numels(str(hf_cache_dir(_MID)))
    assert by_id == by_dir and by_id, "an id and its cache directory must read the same"
    assert {"audio_tower", "language_model"} <= set(by_id.values())
    assert len(checkpoint_numels(_MID)) == len(checkpoint_numels(str(hf_cache_dir(_MID))))


def test_an_unresolvable_name_is_still_empty_rather_than_an_error():
    from agent.weight_census import checkpoint_section_numels

    assert checkpoint_section_numels("no-such-org/no-such-model") == {}
    assert checkpoint_section_numels("") == {}


# ------------------------------------------------- the census is told where the checkpoint is


def test_the_census_is_called_with_a_checkpoint():
    src = (_PA / "agent" / "trace_replay.py").read_text()
    i = src.index("_census(\n") if "_census(\n" in src else src.index('scope="pipeline"')
    assert "checkpoint=" in src[max(0, i - 200) : i + 300], "the census records attribute names only again"


def test_the_checkpoint_is_found_from_the_pipeline_itself(monkeypatch):
    """No env var names the model root -- checked against a live run's whole process tree. The
    object being measured knows where it lives: its class's module file sits inside the model dir."""
    import sys as _sys
    import types

    TR = _trace_replay()

    monkeypatch.delenv("PERF_MCP_MODEL_ROOT", raising=False)
    monkeypatch.delenv("TT_PERF_MODEL_ROOT", raising=False)

    demo = _PA.parent.parent / "tt_transformers" / "demo" / "voxtral_mini_3b_2507"
    if not (demo / "tt" / "pipeline.py").exists():
        import pytest

        pytest.skip("voxtral demo not in this tree")

    mod_name = "models.tt_transformers.demo.voxtral_mini_3b_2507.tt.pipeline"
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(demo / "tt" / "pipeline.py")
    monkeypatch.setitem(_sys.modules, mod_name, mod)

    class _Pipe:
        pass

    _Pipe.__module__ = mod_name
    assert TR._checkpoint_for_census(_Pipe()) == _MID


def test_no_pipeline_and_no_env_is_none_not_a_crash():
    assert _trace_replay()._checkpoint_for_census(None) is None


def test_the_walk_stops_at_a_repository_boundary():
    """Without the .git guard this could climb out of the model and scan a monorepo."""
    src = (_PA / "agent" / "trace_replay.py").read_text()
    i = src.index("def _checkpoint_for_census(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert '".git"' in body and "break" in body
