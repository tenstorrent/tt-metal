# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A stage's share of the resident weights, measured on the chip instead of read off the disk.

WHAT THE REPORT PRINTED, 2026-08-16:

    DECODE — per token │ THEORETICAL │ MEASURED
      memory ← binds   │  3.15 ms    │  2.89 ms
                       │ 512.0 GB/s  │ 557.8 GB/s

557.8 GB/s on a 512 GB/s part -- 109% of peak, so the row is flagged suspect and the one number a
reader would act on cannot be acted on. Every other input had been chased down and measured: the
timing is trace-replay wall time, the peak is the board's, the census walks the built model at the
widths the loader chose (1.72 GB resident, 0.476 B/param). One input was still a guess.

THE SPLIT. A decode token reads the language tower and never the audio tower, so the ceiling needs
that tower's bytes -- and the census returned ONE total for the whole model. `_stage_share` bridged
the gap with the checkpoint: language_model is 8.028 GB of a 9.356 GB file, so 85.8%, applied to the
1.72 GB resident. But the file is bf16 throughout and the device is not: the loader picks a width per
tensor, and nothing requires it to pick the same ones for an audio encoder as for a language
backbone. 85.8% of the FILE is not 85.8% of the CHIP unless it happens to be, and a 9% error in that
one ratio is the whole overshoot.

WHAT THE WALK ALREADY KNEW. It reached every tensor by following a named attribute and discarded the
name. Keeping it costs a tuple per queue entry and answers the question directly: bytes credited to
every name on the path in, so a caller asks by the name the model declared -- at whatever depth the
tower sits, without matching architecture strings or knowing what a tower is called.

    _stage_share(decode)   0.858 (of the file)  ->  measured (of the chip)

AND IT STAYS PUT. Written inside the same first-complete-census-wins guard as the total, for the same
reason: a dtype rung that halves one tower moves the proportions, and a ceiling that moves during a
run scores the run against two different targets.

NOT DECODE, AND NOT VOXTRAL. Nothing here names a stage or a tower. Any block the model declares a
root for is priced from its own resident bytes; a single-tower model is untouched, because its one
root is the whole model and the share was already 1.0.
"""

import json
import struct
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


class _T:
    """A device tensor as the census recognises one: resident, with a shape and a real dtype."""

    def __init__(self, numel, dtype="bfloat16"):
        self.shape = (numel,)
        self.dtype = dtype

    def storage_type(self):
        return "DEVICE"


class _M:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ------------------------------------------------------------------ the walk records where it went


def test_each_tensor_is_credited_to_the_attribute_it_was_reached_through():
    from agent.weight_census import census

    c = census(_M(audio_tower=_M(w=_T(1000)), language_model=_M(w=_T(4000))))
    assert c["sections"]["audio_tower"] == 2000
    assert c["sections"]["language_model"] == 8000
    assert c["weight_bytes"] == 10000


def test_a_tower_nested_deep_is_still_found_by_its_own_name():
    """THE REASON IT IS EVERY NAME AND NOT THE FIRST. A pipeline wraps the model; keying on the
    top-level attribute would credit both towers to the wrapper and answer 1.0 for everything."""
    from agent.weight_census import census

    c = census(_M(model=_M(inner=_M(language_model=_M(layers=[_M(w=_T(4000))])))))
    assert c["sections"]["language_model"] == 8000
    assert c["sections"]["model"] == 8000  # the wrapper is on the path too, and covers everything


def test_list_indices_do_not_become_names():
    """A 32-layer stack must not produce 32 groups; only named attributes identify a subtree."""
    from agent.weight_census import census

    c = census(_M(tower=_M(layers=[_M(w=_T(100)) for _ in range(32)])))
    assert c["sections"]["tower"] == 6400
    assert all(not k.isdigit() for k in c["sections"])


def test_a_shared_tensor_is_counted_once():
    """Tied embeddings are reachable from two towers; counting them twice inflates a share."""
    from agent.weight_census import census

    shared = _T(1000)
    c = census(_M(a=_M(w=shared), b=_M(w=shared)))
    assert c["weight_bytes"] == 2000
    assert c["sections"].get("a", 0) + c["sections"].get("b", 0) == 2000


def test_mixed_precision_towers_do_not_share_a_ratio():
    """THE BUG, as arithmetic. Equal parameter counts, different served widths: the checkpoint says
    50/50 and the chip says otherwise. This is the case the disk ratio cannot see."""
    from agent.weight_census import census

    c = census(_M(audio_tower=_M(w=_T(1000, "bfloat16")), language_model=_M(w=_T(1000, "bfloat4_b"))))
    assert c["sections"]["audio_tower"] == 2000
    assert c["sections"]["language_model"] == 562  # 1000 x 0.5625, not half of the total
    assert c["sections"]["language_model"] / c["weight_bytes"] < 0.5


def test_a_single_tower_model_reports_its_one_root():
    from agent.weight_census import census

    c = census(_M(decoder=_M(w=_T(500))))
    assert c["sections"] == {"decoder": 1000}


# ------------------------------------------------------------------ across the process boundary


def test_the_marker_round_trips():
    from cc_optimize.perf_mcp import _parse_census_sections
    from agent.weight_census import census, sections_marker

    c = census(_M(audio_tower=_M(w=_T(1000)), language_model=_M(w=_T(4000))))
    got = _parse_census_sections(sections_marker(c) + "\n")
    assert got["audio_tower"] == 2000 and got["language_model"] == 8000


def test_no_sections_prints_no_line():
    """An empty marker rather than an empty field: a parser must not have to tell them apart."""
    from agent.weight_census import sections_marker

    assert sections_marker({"weight_bytes": 10}) == ""


def test_the_marker_is_bounded():
    """A deep model has hundreds of attribute names and the tail is LayerNorms. One line stays one
    line; what survives is sorted by bytes, so nothing big enough to be a tower falls off."""
    from agent.weight_census import _SECTIONS_IN_MARKER, sections_marker

    big = {"n%03d" % i: (i + 1) * 1000 for i in range(400)}
    out = sections_marker({"sections": big})
    pairs = out.split("=", 1)[1].split(",")
    assert len(pairs) == _SECTIONS_IN_MARKER
    assert "n399:400000" in pairs, "the largest section was dropped"


def test_a_malformed_pair_does_not_discard_the_line():
    from cc_optimize.perf_mcp import _parse_census_sections

    got = _parse_census_sections("TRACE_WEIGHT_SECTIONS=audio_tower:2000,broken,lm:-4\n")
    assert got == {"audio_tower": 2000}


def test_an_absent_line_is_not_an_error():
    from cc_optimize.perf_mcp import _parse_census_sections

    assert _parse_census_sections("TRACE_WEIGHT_BYTES=10 complete=1\n") == {}


# ------------------------------------------------------------------ and it is pinned, like the total


def _facts_dir(tmp_path, monkeypatch):
    from cc_optimize import perf_mcp

    box = tmp_path / "box"
    box.mkdir(exist_ok=True)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(box))
    monkeypatch.setenv("PERF_MCP_BOARD_STATE_DIR", str(box))
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT_STATED", True)
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path)
    p = tmp_path / "perf_target_inputs.json"
    p.write_text(json.dumps({"total_params": 3611483136}))
    return perf_mcp, p


def test_the_split_is_recorded_with_the_total(tmp_path, monkeypatch):
    m, p = _facts_dir(tmp_path, monkeypatch)
    m._persist_device_weight_bytes(1718081696, True, 1.3228, {"audio_tower": 244000000, "language_model": 1474081696})
    doc = json.loads(p.read_text())
    assert doc["device_section_bytes"]["language_model"] == 1474081696


def test_the_split_does_not_move_once_pinned(tmp_path, monkeypatch):
    """A dtype rung halves the language tower on a later iteration. The ceiling must not follow it
    down -- the same guarantee the total already has, and for the same reason."""
    m, p = _facts_dir(tmp_path, monkeypatch)
    m._persist_device_weight_bytes(1718081696, True, 1.3228, {"language_model": 1474081696})
    m._persist_device_weight_bytes(859040848, True, 0.6614, {"language_model": 737040848})
    doc = json.loads(p.read_text())
    assert doc["device_section_bytes"]["language_model"] == 1474081696, "the split moved mid-run"
    assert doc["device_weight_bytes"] == 1718081696


def test_a_split_arriving_after_a_bare_total_is_still_recorded(tmp_path, monkeypatch):
    """THE UPGRADE CASE, and it is not a drift. Every model already optimised has a facts file pinned
    by a tool version with no split to record -- complete, so nothing may replace it. Refusing here
    means the split can never arrive for exactly the models that have already run, and the ceiling
    goes on apportioning by the checkpoint with no sign that it did. The total is unchanged, so this
    is the same census described further."""
    m, p = _facts_dir(tmp_path, monkeypatch)
    m._persist_device_weight_bytes(1718081696, True, 1.3228)
    m._persist_device_weight_bytes(1718081696, True, 1.3228, {"language_model": 1474081696})
    assert json.loads(p.read_text())["device_section_bytes"]["language_model"] == 1474081696


def test_a_split_carrying_a_different_total_is_refused(tmp_path, monkeypatch):
    """The guarantee that survives: a different total is a different census, whatever it brings with
    it. This is the dtype-rung case, and letting it in is how the ceiling starts chasing the run."""
    m, p = _facts_dir(tmp_path, monkeypatch)
    m._persist_device_weight_bytes(1718081696, True, 1.3228)
    m._persist_device_weight_bytes(859040848, True, 0.6614, {"language_model": 737040848})
    doc = json.loads(p.read_text())
    assert "device_section_bytes" not in doc
    assert doc["device_weight_bytes"] == 1718081696


# ------------------------------------------------------------------ what the roofline does with it


def _share(mf, stage="decode"):
    """_stage_share is a closure over the facts, so reach it the way the report does."""
    from cc_optimize import summary

    return summary._roofline_stage_share(mf, stage)


def test_the_measured_split_is_preferred_over_the_checkpoint_ratio():
    """THE FIX. 85.8% is the file's proportion; this model put 70% of its resident bytes there."""
    mf = {
        "stage_roots": {"decode": "language_model"},
        "device_weight_bytes": 1718081696,
        "device_section_bytes": {"language_model": 1202657187, "audio_tower": 515424509},
    }
    assert abs(_share(mf) - 0.70) < 0.01


def test_every_declared_block_is_priced_from_its_own_bytes():
    """Not decode, and not two stages. Whatever the model declares a root for gets its own share."""
    mf = {
        "stage_roots": {"decode": "language_model", "encode": "audio_tower", "project": "adapter"},
        "device_weight_bytes": 2000,
        "device_section_bytes": {"language_model": 1400, "audio_tower": 500, "adapter": 100},
    }
    assert abs(_share(mf, "decode") - 0.70) < 1e-9
    assert abs(_share(mf, "encode") - 0.25) < 1e-9
    assert abs(_share(mf, "project") - 0.05) < 1e-9


def test_a_root_the_census_never_saw_falls_back_to_the_checkpoint():
    """The census may not reach a subtree the checkpoint names. Falling back is the old behaviour,
    which is an approximation -- refusing outright would drop a ceiling that used to print."""
    mf = {
        "stage_roots": {"decode": "language_model"},
        "device_weight_bytes": 1718081696,
        "device_section_bytes": {"audio_tower": 515424509},
    }
    assert _share(mf) > 0.0


def test_a_share_above_one_is_refused():
    """Impossible: it means the total and the split came from different walks. A wrong divisor is
    worse than the fallback, which is the rule this file's whole chain already follows."""
    mf = {
        "stage_roots": {"decode": "language_model"},
        "device_weight_bytes": 1000,
        "device_section_bytes": {"language_model": 4000},
    }
    assert _share(mf) <= 1.0


def test_a_run_with_no_census_is_unchanged():
    """Every model that never produced a census keeps exactly the ceiling it had before."""
    mf = {"stage_roots": {"decode": "language_model"}}
    assert _share(mf) > 0.0


# ------------------------------------------------- and it must answer in the CALLER's vocabulary
#
# THE FIRST VERSION MEASURED THE RIGHT THING AND WAS NEVER READ. Run 5, 2026-08-16: the census
# recorded 19 subtrees of the built voxtral model --
#
#     _inner 93.8%  embed 46.9%  lm_layers 13.2%  lm_head 13.2%  enc_a 5.4%  enc_b 5.4%  ...
#
# -- and every consumer asks by CHECKPOINT SECTION, because that is what stage_roots establishes:
# `language_model`, `audio_tower`. Not one name in common. `_stage_share` looked up `language_model`,
# found nothing, and fell back to the disk ratio it was written to replace. The whole change was
# inert on the run that was supposed to prove it.
#
# Translating names afterwards is not available. A tower is renamed, re-nested and wrapped on its way
# into a TT model -- and worse, `encode` and `prefill` DO exist as device attributes while being 1 MB
# state objects, so a name match would have priced the audio stage at a megabyte.
#
# An element count survives all of it: transposed, tile-padded, sharded, re-quantised, a tensor keeps
# the numel the file recorded. That is already how checkpoint_numels tells a weight from a runtime
# buffer, so the join reuses a property the module already trusts.


def _write_ckpt(d, tensors):
    """A minimal safetensors header: 8-byte little-endian length, then JSON. No tensor data."""
    hdr = {n: {"dtype": "BF16", "shape": list(s), "data_offsets": [0, 0]} for n, s in tensors.items()}
    raw = json.dumps(hdr).encode()
    with open(Path(d) / "model.safetensors", "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
    return str(d)


_VOXTRAL_SHAPED = {
    "audio_tower.layers.0.w": (1000,),
    "audio_tower.layers.1.w": (1001,),
    "language_model.embed.w": (5000,),
    "language_model.layers.0.w": (5001,),
}


def _built_under_other_names():
    """The same four tensors as the TT model actually exposes them -- no name in common."""
    return _M(_inner=_M(enc_a=_M(w=_T(1000)), enc_b=_M(w=_T(1001)), embed=_M(w=_T(5000)), lm_layers=_M(w=_T(5001))))


def test_the_split_is_reported_by_checkpoint_section(tmp_path):
    """THE FIX. The caller asks `language_model`; the built model has never heard the word."""
    from agent.weight_census import census

    c = census(_built_under_other_names(), checkpoint=_write_ckpt(tmp_path, _VOXTRAL_SHAPED))
    assert c["sections"]["language_model"] == 20002  # (5000 + 5001) x 2.0
    assert c["sections"]["audio_tower"] == 4002  # (1000 + 1001) x 2.0


def test_the_attribute_view_is_kept_alongside_it():
    """It is the only view when there is no checkpoint to match against, and it costs nothing."""
    from agent.weight_census import census

    c = census(_built_under_other_names())
    assert c["sections"]["enc_a"] == 2000 and c["sections"]["lm_layers"] == 10002
    assert "language_model" not in c["sections"], "a section was invented with no checkpoint to read"


def test_a_size_used_by_two_towers_is_not_attributed(tmp_path):
    """AMBIGUITY IS DROPPED, NOT GUESSED. Picking one would move bytes between towers silently, and
    a share is exactly the thing nobody can check by eye."""
    from agent.weight_census import census, checkpoint_section_numels

    ckpt = _write_ckpt(tmp_path, {"audio_tower.a": (4096,), "language_model.b": (4096,)})
    assert 4096 not in checkpoint_section_numels(ckpt)

    c = census(_M(x=_M(w=_T(4096))), checkpoint=ckpt)
    assert c["sections"].get("unmatched") == 8192
    assert "language_model" not in c["sections"] and "audio_tower" not in c["sections"]


def test_unmatched_bytes_are_reported_rather_than_dropped(tmp_path):
    """A reader comparing a stage against the total has to be able to see what is unaccounted for.

    Unmatched means AMBIGUOUS, not absent: a tensor the checkpoint never had is a runtime buffer and
    is already excluded from the byte total, so it is not a tower with an unknown name."""
    from agent.weight_census import census

    ckpt = _write_ckpt(tmp_path, {"language_model.w": (5000,), "audio_tower.x": (77,), "language_model.y": (77,)})
    c = census(_M(a=_M(w=_T(5000)), b=_M(w=_T(77))), checkpoint=ckpt)
    assert c["sections"]["language_model"] == 10000
    assert c["sections"]["unmatched"] == 154
    assert c["sections"]["language_model"] + c["sections"]["unmatched"] == c["weight_bytes"]


def test_a_checkpoint_section_wins_a_name_collision(tmp_path):
    """Both vocabularies can say `layers`. Adding the two rules into one key yields a number that is
    neither -- so they are accumulated apart and the checkpoint's answer is the one published."""
    from agent.weight_census import census

    ckpt = _write_ckpt(tmp_path, {"layers.w": (700,)})
    c = census(_M(layers=_M(w=_T(700))), checkpoint=ckpt)
    assert c["sections"]["layers"] == 1400, c["sections"]


def test_a_tensor_the_checkpoint_never_had_is_not_a_tower(tmp_path):
    """Runtime buffers are already excluded from the byte total; they must not appear as a section
    either, or a stage's share would be measured against a denominator that includes scratch."""
    from agent.weight_census import census

    ckpt = _write_ckpt(tmp_path, {"language_model.w": (5000,)})
    c = census(_M(w=_M(x=_T(5000)), kv=_M(cache=_T(999999))), checkpoint=ckpt)
    assert c["weight_bytes"] == 10000, "a runtime buffer was counted as a weight"
    assert c["sections"]["language_model"] == 10000
    assert 999999 * 2 not in c["sections"].values()


def test_an_unreadable_checkpoint_leaves_the_attribute_view_intact(tmp_path):
    """The join is best-effort; losing it must cost the section names, not the census."""
    from agent.weight_census import census

    (tmp_path / "model.safetensors").write_bytes(b"not a safetensors file at all")
    c = census(_M(tower=_M(w=_T(100))), checkpoint=str(tmp_path))
    assert c["weight_bytes"] == 200 and c["sections"]["tower"] == 200
