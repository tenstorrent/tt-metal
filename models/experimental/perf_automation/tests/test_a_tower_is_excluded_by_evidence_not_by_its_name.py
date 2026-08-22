# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which weights a token streams was decided by matching tensor NAMES against a list.

    _TOWER_ONLY = re.compile(r"(^|\\.)(vision_tower|vision_model|vision_encoder|visual|image_encoder"
                             r"|image_tower|audio_tower|audio_encoder|speech_encoder"
                             r"|multi_modal_projector|mm_projector)\\.", re.I)

An encoder runs once per image or clip, never per generated token, so its parameters must not be
charged to a decode step -- gemma-3-12b-it carries 437 such tensors, 0.411 B params. The list is a
good list. It is still a list of the names encoders USUALLY have, and it fails SILENTLY: a model
naming its encoder something else has that encoder counted in every token's read set, which inflates
the divisor and so the ceiling. Too high a ceiling is the direction that ends a run early believing
it is at the wall.

Everything else in this chain was already moved off names. Gathered tensors are found by watching
which gathers the device RAN. stage_roots binds a stage to a section using the stack depths the tool
measured and the indices its own generated test emitted. This was the last place a number came from
someone's naming convention, and summary already states the physics it should have used:

    "the model-level figure divides by the WHOLE resident model, including towers the recurring
     stage never reads ... the difference being an audio encoder a decoded token does not touch."

So weight_bytes now takes the sections a unit actually reads. The name list stays as the fallback:
the map arrives from discovery, which does not always precede the byte walk -- the same ordering that
once filed the byte anchor under a placeholder key -- so an absent map must degrade to the old
behaviour rather than to a wrong number.
"""
import struct
import sys
import json
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from agent import model_bytes as MB  # noqa: E402


def _ckpt(tmp_path, tensors):
    """A one-shard safetensors file: {name: n_params}."""
    hdr, off = {}, 0
    for name, numel in tensors.items():
        hdr[name] = {"dtype": "BF16", "shape": [numel], "data_offsets": [off, off + numel * 2]}
        off += numel * 2
    raw = json.dumps(hdr).encode()
    f = tmp_path / "model.safetensors"
    with open(f, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        fh.write(b"\0" * off)
    return tmp_path


# An encoder called something the list has never heard of, beside a normal backbone.
_ODD = {"perception_stack.layers.0.weight": 1_000_000, "language_model.layers.0.weight": 4_000_000}
# The conventional spelling, which the list does catch.
_USUAL = {"audio_tower.layers.0.weight": 1_000_000, "language_model.layers.0.weight": 4_000_000}


def test_an_unconventionally_named_encoder_was_charged_to_every_token(tmp_path):
    """THE DEFECT, with no map supplied: the name list cannot see this encoder, so its 1M params land
    in the per-token read set."""
    d = _ckpt(tmp_path, _ODD)
    got = MB.weight_bytes(d, unit="token")
    assert got["params"] == 5_000_000, got["params"]


def test_the_derived_map_excludes_it(tmp_path):
    """With stage_roots supplied, the encoder is excluded because it is not what a token reads --
    whatever it is called."""
    d = _ckpt(tmp_path, _ODD)
    got = MB.weight_bytes(d, unit="token", streamed_sections={"language_model"})
    assert got["params"] == 4_000_000, got["params"]


def test_the_map_and_the_name_list_agree_on_a_conventional_model(tmp_path):
    """The change must not move models the list already handled -- that is most of them."""
    d = _ckpt(tmp_path, _USUAL)
    by_name = MB.weight_bytes(d, unit="token")
    by_map = MB.weight_bytes(d, unit="token", streamed_sections={"language_model"})
    assert by_name["params"] == by_map["params"] == 4_000_000


def test_no_map_falls_back_to_the_name_list(tmp_path):
    """An absent map degrades to the previous behaviour, never to a wrong number."""
    d = _ckpt(tmp_path, _USUAL)
    for empty in (None, set(), ()):
        assert MB.weight_bytes(d, unit="token", streamed_sections=empty)["params"] == 4_000_000


def test_a_non_token_unit_reads_everything(tmp_path):
    """A denoise step or one classifier pass runs the whole pipeline per unit, so nothing is
    excluded -- and the caller returns None for those rather than a set."""
    d = _ckpt(tmp_path, _USUAL)
    assert MB.weight_bytes(d, unit="step")["params"] == 5_000_000


def test_the_caller_refuses_to_guess():
    """None whenever the map is not genuinely available: no mirror, no facts file, or a unit whose
    recurring stage is not a token step."""
    import cc_optimize.run as R

    assert R._streamed_sections_for_unit("/nonexistent", "step") is None
    assert R._streamed_sections_for_unit("/nonexistent", "token") is None
    assert R._streamed_sections_for_unit("/nonexistent", "") is None


def test_the_caller_reads_the_map_from_the_facts_file(tmp_path):
    """stage_roots lives in two places; the facts file is the one that survives without --persist."""
    import cc_optimize.run as R

    (tmp_path / "perf_target_inputs.json").write_text(
        json.dumps({"stage_roots": {"encode": "perception_stack", "prefill": "lm", "decode": "lm"}})
    )
    assert R._streamed_sections_for_unit(tmp_path, "token") == {"lm"}


def test_prefill_and_decode_share_the_subtree(tmp_path):
    """Both read the backbone, so the set is the same either way and the encoder is out of both."""
    import cc_optimize.run as R

    (tmp_path / "perf_target_inputs.json").write_text(
        json.dumps({"stage_roots": {"encode": "audio_tower", "prefill": "language_model", "decode": "language_model"}})
    )
    assert R._streamed_sections_for_unit(tmp_path, "token") == {"language_model"}


def test_a_flat_checkpoint_is_not_emptied_by_the_map(tmp_path):
    """A tensor with no dotted section ("w") belongs to no section, and _checkpoint_tensor_sections --
    where the map's sections come from -- skips such names, so no map can ever list them. Excluding
    them would drop real streamed weights, RAISING the ceiling. Keep them."""
    d = _ckpt(tmp_path, {"w": 1_000_000})
    assert MB.weight_bytes(d, unit="token", streamed_sections={"language_model"})["params"] == 1_000_000


def test_a_mixed_checkpoint_keeps_the_flat_tensor_and_drops_the_foreign_section(tmp_path):
    d = _ckpt(tmp_path, {"w": 500_000, "perception_stack.l.weight": 1_000_000, "lm.l.weight": 2_000_000})
    assert MB.weight_bytes(d, unit="token", streamed_sections={"lm"})["params"] == 2_500_000


# --- a bug in this block used to be indistinguishable from an unreadable checkpoint ---------------


def _snap_with(tmp_path, tensors):
    """Built ONCE and reused. _hf_snapshots is called twice on this path, so a factory that mkdir()s
    raises FileExistsError on the second call -- an OSError, which the generic handler swallows, so
    the test silently exercised the wrong branch."""
    d = tmp_path / "snap"
    d.mkdir(exist_ok=True)
    return _ckpt(d, tensors)


def test_a_programming_error_in_the_byte_walk_is_reported(tmp_path, monkeypatch, capsys):
    """WHAT THIS COST, live. Wiring the section map in referenced `model_root` inside
    _perf_target_inputs(demo_dir, ...) -- no such name. The NameError hit `except Exception: pass`,
    total_params vanished for the whole run, and the ceiling silently fell back to the checkpoint's
    FILE SIZE: the stored dtype, 2.4x the served divisor on Llama-3.1-8B. No warning, no traceback,
    and the number is plausible enough to read past.

    `except Exception: pass` is correct for what it was written for -- a truncated shard, a dtype
    with no width. It is wrong for a defect in the tool, because the fall-through hides it perfectly.
    """
    import cc_optimize.run as R

    monkeypatch.setattr(R, "_model_weight_bytes", lambda d, h=None, *_a, **_k: 4_000_000)
    monkeypatch.setattr(R, "_resolve_model_id", lambda d, h=None, *_a, **_k: "org/m")
    monkeypatch.setattr(R, "_hf_cache_dims", lambda mid: {"hidden_size": 1000})
    _snap = _snap_with(tmp_path, {"lm.w": 1_000_000})
    monkeypatch.setattr(R, "_hf_snapshots", lambda mid: [_snap])
    # the exact shape of the mistake: a name that does not exist in this scope
    monkeypatch.setattr(
        R, "_streamed_sections_for_unit", lambda *a, **k: (_ for _ in ()).throw(NameError("model_root"))
    )

    R._perf_target_inputs(tmp_path, None, {})
    err = capsys.readouterr().err
    assert "BUG in the analytic byte walk" in err, err
    assert "NameError" in err and "model_root" in err, err


def test_an_unreadable_checkpoint_stays_quiet(tmp_path, monkeypatch, capsys):
    """The environmental case keeps the documented silent fall-through -- this adds a signal for
    defects, it does not make every unreadable shard shout."""
    import cc_optimize.run as R

    monkeypatch.setattr(R, "_model_weight_bytes", lambda d, h=None, *_a, **_k: 4_000_000)
    monkeypatch.setattr(R, "_resolve_model_id", lambda d, h=None, *_a, **_k: "org/m")
    monkeypatch.setattr(R, "_hf_cache_dims", lambda mid: {"hidden_size": 1000})
    _snap = _snap_with(tmp_path, {"lm.w": 1_000_000})
    monkeypatch.setattr(R, "_hf_snapshots", lambda mid: [_snap])
    monkeypatch.setattr(R, "_streamed_sections_for_unit", lambda *a, **k: (_ for _ in ()).throw(OSError("truncated")))

    R._perf_target_inputs(tmp_path, None, {})
    assert "BUG in the analytic byte walk" not in capsys.readouterr().err
