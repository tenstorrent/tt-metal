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
