# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bytes-per-unit-of-work computed from the checkpoint, and the unit it belongs to.

The ceiling is peak_BW / bytes-per-unit. Two earlier numerators were wrong in opposite directions:

    checkpoint FILE SIZE   counts the stored dtype -- 15.0 GB of bf16 for Llama-3.1-8B where the
                           device streams 6.09 GB as bfp4/bfp8, understating the ceiling 2.4x.
    profile per-op bytes   has no reliable per-unit divisor: one window's call counts implied 51
                           tokens from the FFN matmuls, 25 from QKV and 376 from the LM head.

These cover the analytic replacement -- per-tensor shape and dtype from safetensors headers, device
widths applied by name pattern -- and the unit map, which decides whether a ceiling may be published
at all. A tag with no unit must yield NO ceiling: a wrong one reads as a target and can stop a run.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _mb():
    from agent import model_bytes

    return model_bytes


def _shard(path: Path, tensors: dict):
    """Write a safetensors file with only a header (no tensor data needed to read shapes/dtypes)."""
    hdr = {name: {"dtype": dt, "shape": list(shape), "data_offsets": [0, 0]} for name, (dt, shape) in tensors.items()}
    blob = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


@pytest.fixture
def ckpt(tmp_path):
    """A miniature two-shard checkpoint shaped like a decoder-only LLM."""
    _shard(
        tmp_path / "model-00001-of-00002.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", [1000, 64]),
            "model.layers.0.self_attn.q_proj.weight": ("BF16", [64, 64]),
            "model.layers.0.mlp.gate_proj.weight": ("BF16", [256, 64]),
        },
    )
    _shard(
        tmp_path / "model-00002-of-00002.safetensors",
        {
            "model.layers.1.self_attn.q_proj.weight": ("BF16", [64, 64]),
            "model.layers.1.mlp.gate_proj.weight": ("BF16", [256, 64]),
            "lm_head.weight": ("BF16", [1000, 64]),
        },
    )
    return tmp_path


def test_bytes_are_summed_per_tensor_from_the_headers(ckpt):
    mb = _mb()
    r = mb.weight_bytes(ckpt, unit="token")
    # 2 q_proj (4096) + 2 gate_proj (16384) + lm_head (64000) elements, all bf16
    assert r["bytes"] == (2 * 64 * 64 + 2 * 256 * 64 + 1000 * 64) * 2
    assert r["tensors"] == 6 and r["shards"] == 2


def test_the_embedding_table_is_excluded_for_a_token_unit(ckpt):
    """A token reads ONE embedding row, not the table. The output projection IS read in full."""
    mb = _mb()
    r = mb.weight_bytes(ckpt, unit="token")
    assert r["skipped_lookup_bytes"] == 1000 * 64 * 2
    assert r["bytes"] > 1000 * 64 * 2  # lm_head, same shape, still counted


def test_the_embedding_table_is_included_for_a_single_pass_unit(ckpt):
    mb = _mb()
    tok = mb.weight_bytes(ckpt, unit="token")
    inf = mb.weight_bytes(ckpt, unit="inference")
    assert inf["bytes"] == tok["bytes"] + tok["skipped_lookup_bytes"]
    assert inf["skipped_lookup_bytes"] == 0


def test_device_widths_override_the_stored_dtype_per_pattern(ckpt):
    """The point of the whole module: a bf16 checkpoint served as bfp4 streams a quarter of the bytes."""
    mb = _mb()
    stored = mb.weight_bytes(ckpt, unit="token")
    ov = mb.weight_bytes(ckpt, unit="token", overrides=mb.parse_overrides("gate_proj=bfloat4_b"))
    gate = 2 * 256 * 64
    assert ov["bytes"] == stored["bytes"] - gate * 2 + gate * (4 + 8 / 16) / 8
    assert any("gate_proj" in k for k in ov["by_pattern"])


def test_block_float_widths_carry_the_shared_exponent():
    """bfp8 is 8.5 bits per element, not 8: one shared exponent per 16 elements."""
    mb = _mb()
    assert mb.device_width("bfloat8_b") == 1.0625
    assert mb.device_width("bfloat4_b") == 0.5625
    assert mb.device_width("bfloat16") == 2.0
    assert mb.device_width("nonsense") is None


def test_an_unmatched_tensor_keeps_its_stored_width(ckpt):
    """Never assume a narrower dtype than is known -- that inflates the ceiling."""
    mb = _mb()
    r = mb.weight_bytes(ckpt, unit="token", overrides=mb.parse_overrides("q_proj=bfloat4_b"))
    assert any(k.startswith("stored:") for k in r["by_pattern"])


def test_a_default_device_dtype_applies_where_no_pattern_matches(ckpt):
    mb = _mb()
    r = mb.weight_bytes(ckpt, unit="token", default_device_dtype="bfloat8_b")
    stored = mb.weight_bytes(ckpt, unit="token")
    assert r["bytes"] < stored["bytes"]


def test_a_directory_with_no_checkpoint_yields_nothing(tmp_path):
    mb = _mb()
    assert mb.weight_bytes(tmp_path, unit="token") == {}
    assert mb.weight_bytes("/does/not/exist", unit="token") == {}


def test_a_corrupt_shard_is_skipped_not_fatal(ckpt):
    mb = _mb()
    (ckpt / "model-00003-of-00003.safetensors").write_bytes(b"\x00" * 4)
    assert mb.weight_bytes(ckpt, unit="token")["bytes"] > 0


# --- the unit map: what may be published at all -----------------------------------------------------


@pytest.mark.parametrize(
    "tag,unit",
    [
        ("text-generation", "token"),
        ("image-text-to-text", "token"),  # VLM text stage
        ("video-text-to-text", "token"),
        ("audio-text-to-text", "token"),
        ("summarization", "token"),
        ("translation", "token"),
        ("automatic-speech-recognition", "token"),
        ("text-to-speech", "token"),
        ("text-to-image", "step"),  # diffusion
        ("unconditional-image-generation", "step"),
        ("text-to-video", "step"),
        ("image-to-3d", "step"),
        ("feature-extraction", "inference"),
        ("image-classification", "inference"),
        ("text-classification", "inference"),
        ("visual-document-retrieval", "inference"),
    ],
)
def test_units_cover_the_hf_tags_the_planner_was_missing(tag, unit):
    """The planner's category map covers 33 of HF's 47 tags; 19 fell through to a keyword guess.
    Keying on the unit needs 4 answers rather than a taxonomy that goes stale as HF adds tags."""
    assert _mb().unit_for_tag(tag) == unit


@pytest.mark.parametrize(
    "tag", ["reinforcement-learning", "tabular-regression", "tabular-classification", "", "not-a-tag"]
)
def test_a_tag_with_no_well_defined_unit_gets_no_ceiling(tag):
    """The safe direction: no unit -> no rate ceiling -> the caller falls back to the per-op floor.
    Publishing a decode ceiling for one of these would read as a target and could stop a run early."""
    assert _mb().unit_for_tag(tag) == ""


def test_unit_labels_are_what_the_report_prints():
    mb = _mb()
    assert mb.unit_label("token") == "tok/s/u"
    assert mb.unit_label("step") == "steps/s"
    assert mb.unit_label("inference") == "inferences/s"
    assert mb.unit_label("") == ""


@pytest.mark.parametrize(
    "spec,expect",
    [
        ("gate_proj=bfloat4_b", 1),
        ("a=bfloat4_b,b=bfloat8_b", 2),
        ("a=nonsense_dtype", 0),  # unknown width -> refused, not guessed
        ("=bfloat4_b", 0),  # no pattern
        ("gate_proj", 0),  # no dtype
        ("[unclosed=bfloat4_b", 0),  # invalid regex
        ("", 0),
    ],
)
def test_override_parsing_refuses_what_it_cannot_verify(spec, expect):
    assert len(_mb().parse_overrides(spec)) == expect


# --- the class-name fallback is GONE; the unit is observed, not derived from a config ----------------


def test_the_config_derived_unit_is_deleted():
    """These tested unit_from_config / unit_for_architectures -- a second answer to "what does this
    model retire per call", derived from a class name.

    Both are removed because neither input can answer it. A tag names the TASK and a head names the
    HEAD; neither states whether the model LOOPS. `text-to-speech` covers XTTS, which emits tokens,
    and Kokoro-82M, which is StyleTTS2 and produces a whole waveform in one pass -- one tag, two
    units, so the table had to pick and was wrong for the other. HunyuanImage-3.0 is the same failure
    inverted: tagged text-to-image, so read as a denoise loop, when HunyuanImage3ForCausalMM is
    autoregressive.

    A wrong unit does not merely degrade a ceiling -- it puts it in the wrong currency, and the band,
    the at-floor verdict and the headline rate all inherit that. The unit now comes from the built
    pipeline (perf_adapter.headline_unit: a callable decode_step retires one token per call, by
    definition), and no observation means NO unit ceiling rather than a guessed one -- the rule
    _anchored_ceiling_facts already stated: "No recoverable unit means no ceiling, which lands on the
    floor fallback: weaker, but not wrong."
    """
    mb = _mb()
    assert not hasattr(mb, "unit_from_config")
    assert not hasattr(mb, "unit_for_architectures")
    assert not hasattr(mb, "_UNIT_BY_ARCH_SUFFIX")


def test_the_tag_lookup_survives_for_the_lookup_only_exclusion():
    """Its one remaining job, in the param-count walk: a token unit reads its embedding table by
    INDEX -- one row per token -- so counting the whole table as streamed bytes overstates what a
    decode step moves. It no longer feeds the ceiling unit, and no longer picks conditions."""
    mb = _mb()
    assert mb.unit_for_tag("text-generation") == "token"
    assert mb.unit_for_tag("nonsense-tag") == ""


# --- measurement conditions come from TT_PERF_* and the config, not a table -------------------------


def test_default_conditions_is_deleted():
    """default_conditions / conditions_label / _DEFAULT_CONDITIONS produced ISL 128, OSL 128, 50
    steps, seq_len 384, resolutions and batch -- and had NO production caller. A run takes ISL/OSL
    from TT_PERF_ISL_TOKENS / TT_PERF_OSL_TOKENS, seq_len from TT_PERF_SEQ_LEN (which defaults to 128,
    not the 384 this table claimed), and batch from perf_adapter.resolve_batch, which asks the
    pipeline. A second, unused source of the same facts that disagreed with the live one is exactly
    what this suite's single-source rule exists to prevent."""
    mb = _mb()
    for gone in ("default_conditions", "conditions_label", "_DEFAULT_CONDITIONS"):
        assert not hasattr(mb, gone), gone


@pytest.mark.parametrize(
    "cfg,expect",
    [
        ({"vision_config": {"image_size": 384}}, 384),
        ({"image_size": 224}, 224),
        ({"sample_size": 64, "vae_scale_factor": 8}, 512),
        ({"sample_size": 128, "vae_scale_factor": 8}, 1024),
        ({"sample_size": 64}, 512),  # SD-family VAE scale is 8
        ({"vision_config": {"image_size": 336}, "image_size": 224}, 336),  # tower wins
    ],
)
def test_resolution_is_read_from_the_config(cfg, expect):
    """It IS the work for a step or vision unit -- a denoise step at 1024 is ~4x the step at 512 --
    and emit-e2e already reads it for the PCC input while the perf side had no notion of it."""
    assert _mb().resolution_from_config(cfg) == expect


@pytest.mark.parametrize("cfg", [{}, None, {"image_size": 0}, {"sample_size": None}, {"foo": 1}])
def test_a_model_with_no_resolution_reports_none(cfg):
    """None must stay None. A resolution printed for a text model states a condition that never
    existed."""
    assert _mb().resolution_from_config(cfg) is None
