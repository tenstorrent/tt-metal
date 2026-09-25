# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which weights a token streams is decided by a list of names, and the list can be wrong quietly.

    _TOWER_ONLY = re.compile(r"(^|\\.)(vision_tower|vision_model|...|audio_tower|...)\\.", re.I)

An encoder runs once per clip, never per generated token, so its parameters must stay out of a decode
step's read set -- gemma-3-12b-it carries 437 such tensors, 0.411 B params. Right physics. But it is a
list of the names encoders USUALLY have, and a model naming its encoder off-list has that encoder
charged to every token: divisor too large, ceiling too low, at-floor reads high, and the run stops
early believing it is at the wall.

WHAT WAS TRIED AND REVERTED. Substituting stage_roots' sections for the name list. It is worse:
stage_roots names the section holding a stage's BLOCK STACK, and on any untied model `lm_head` is a
SIBLING of that stack -- streamed every token, named by no stage_roots entry -- so excluding
everything outside the map silently drops it. Measured on a plain-llama shape: 8.0B -> 7.0B, 12.5% of
the read set gone, erring toward a HIGHER ceiling. One silent error traded for a bigger one.

WHY NEITHER IS THE REAL FIX. There is no measurement of what a token read. _stage_roofs says so:
"_top_ops keys on (op_code, shape, memory) and records nothing about which phase an op ran in", and
the per-stage byte reader that once stood in summary returned 0 on every real profile and is now
deleted -- buckets carry no `bytes` key and their regime tag is "na" for all of them. Until the
profiler tags a phase, every route is a guess, and the least-bad guess is the one whose failure mode
is known and bounded.

SO THIS REPORTS THE DISAGREEMENT INSTEAD. stage_roots is reliable for what it IS -- derived from the
checkpoint's own key structure, it says a non-recurring stage runs out of section X. _TOWER_ONLY says
whether X was excluded. Both are evidence; when they conflict the name list missed a tower, and a
human can see it rather than the divisor moving quietly.
"""
import json
import struct
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from agent import model_bytes as MB  # noqa: E402

_VOX = {"encode": "audio_tower", "prefill": "language_model", "decode": "language_model"}
_ODD = {"encode": "perception_stack", "prefill": "language_model", "decode": "language_model"}


def _ckpt(tmp_path, sized):
    """{name: numel}. SIZES MATTER: the backbone is told from an encoder by being much larger
    (voxtral, 4.014 B against 0.637 B), so a fixture of equal-sized tensors tests a tie, not a model."""
    hdr, off = {}, 0
    for n, ne in sized.items():
        hdr[n] = {"dtype": "BF16", "shape": [ne], "data_offsets": [off, off + ne * 2]}
        off += ne * 2
    raw = json.dumps(hdr).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * off)
    return tmp_path


def test_a_conventional_tower_raises_nothing(tmp_path):
    """voxtral: audio_tower IS on the list, so there is nothing to report -- and this must stay quiet,
    or the warning is noise on every run."""
    d = _ckpt(tmp_path, {"audio_tower.l.w": 637, "language_model.l.w": 4014})
    assert MB.untowered_sections(d, _VOX) == []


def test_an_off_list_tower_is_named(tmp_path):
    """THE CASE THE LIST CANNOT SEE. stage_roots knows a separate stage runs out of perception_stack;
    the name list does not exclude it; so it is in every token's read set."""
    d = _ckpt(tmp_path, {"perception_stack.l.w": 637, "language_model.l.w": 4014})
    assert MB.untowered_sections(d, _ODD) == ["perception_stack"]


def test_a_plain_llm_raises_nothing(tmp_path):
    """No non-recurring stage at all -- most models. Silence."""
    d = _ckpt(tmp_path, {"model.layers.0.w": 4014, "lm_head.w": 400})
    assert MB.untowered_sections(d, {"prefill": "model", "decode": "model"}) == []


def test_a_section_not_in_this_checkpoint_is_not_reported(tmp_path):
    """stage_roots can name a section a given snapshot does not contain; that is not a missed tower."""
    d = _ckpt(tmp_path, {"language_model.l.w": 4014})
    assert MB.untowered_sections(d, _ODD) == []


def test_no_stage_roots_reports_nothing(tmp_path):
    """Before discovery there is no map, and a detector must not invent one."""
    d = _ckpt(tmp_path, {"perception_stack.l.w": 637})
    assert MB.untowered_sections(d, {}) == []
    assert MB.untowered_sections(d, None) == []


def test_the_recurring_stage_is_never_reported_as_a_tower(tmp_path):
    """The backbone IS streamed per token. Reporting it would be exactly backwards."""
    d = _ckpt(tmp_path, {"language_model.l.w": 4014})
    assert "language_model" not in MB.untowered_sections(d, _VOX)


def test_a_bad_snapshot_path_is_silent(tmp_path):
    """It must never cost a ceiling."""
    assert MB.untowered_sections("/nonexistent", _ODD) == []


def test_it_does_not_change_any_byte_count(tmp_path):
    """DETECTION ONLY. The reverted attempt changed the number; this must not, or it is the same
    mistake wearing a different name."""
    d = _ckpt(tmp_path, {"perception_stack.l.w": 637, "language_model.l.w": 4014})
    before = MB.weight_bytes(d, unit="token")
    MB.untowered_sections(d, _ODD)
    assert MB.weight_bytes(d, unit="token") == before


def test_weight_bytes_takes_no_section_map_any_more(tmp_path):
    """The reverted parameter must stay gone -- it drops lm_head on untied models."""
    import inspect

    assert "streamed_sections" not in inspect.signature(MB.weight_bytes).parameters


def test_two_subtrees_of_similar_size_report_nothing(tmp_path):
    """A TIE CARRIES NO SIGNAL. Size tells a backbone from an encoder only when they differ by a lot;
    picking the larger by a hair would name the BACKBONE as an unexcluded tower, which is the most
    misleading thing this could say. Silence costs a warning, never a number."""
    d = _ckpt(tmp_path, {"tower_a.l.w": 1000, "tower_b.l.w": 1100})
    assert MB.untowered_sections(d, {"encode": "tower_a", "decode": "tower_b"}) == []
