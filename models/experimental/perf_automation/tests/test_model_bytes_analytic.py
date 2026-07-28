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


# --- the unit must come from the CONFIG, since config.json has no pipeline_tag -----------------------


def test_the_unit_comes_from_the_architecture_when_there_is_no_pipeline_tag():
    """THE GAP: HF config.json carries `architectures` for every model but usually NOT a pipeline_tag
    (that lives on the model card). Keying only on the tag meant the analytic path never fired for a
    local checkpoint and silently fell back to the file size -- a 2.4x wrong ceiling for Llama."""
    mb = _mb()
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
    assert cfg.get("pipeline_tag") is None
    assert mb.unit_from_config(cfg) == "token"


@pytest.mark.parametrize(
    "arch,unit",
    [
        ("LlamaForCausalLM", "token"),
        ("MistralForCausalLM", "token"),
        ("T5ForConditionalGeneration", "token"),
        ("WhisperForConditionalGeneration", "token"),
        ("GPT2LMHeadModel", "token"),
        ("BertForSequenceClassification", "inference"),
        ("BertForMaskedLM", "inference"),
        ("ViTForImageClassification", "inference"),
        ("DetrForObjectDetection", "inference"),
        ("SegformerForSemanticSegmentation", "inference"),
        ("UNet2DConditionModel", "step"),
        ("Transformer2DModel", "step"),
        ("FluxTransformer2DModel", "step"),
    ],
)
def test_architecture_heads_imply_the_unit(arch, unit):
    """A class name states the head, and the head states what one unit of work is."""
    assert _mb().unit_for_architectures([arch]) == unit


@pytest.mark.parametrize("arch", ["SomeNewThing", "MysteryModel", "", "ForSomethingElse"])
def test_an_unrecognised_head_publishes_no_unit(arch):
    assert _mb().unit_for_architectures([arch]) == ""


def test_a_pipeline_tag_still_wins_when_present():
    """The tag is the more specific signal, so it is consulted first."""
    mb = _mb()
    cfg = {"pipeline_tag": "text-to-image", "architectures": ["SomethingForCausalLM"]}
    assert mb.unit_from_config(cfg) == "step"


def test_an_empty_config_yields_no_unit():
    mb = _mb()
    for cfg in ({}, None, {"architectures": []}, {"architectures": None}):
        assert mb.unit_from_config(cfg) == ""


# --- default measurement CONDITIONS, keyed on the unit of work ---------------------------------------


@pytest.mark.parametrize(
    "unit,cfg,expect",
    [
        ("token", {}, "ISL 128, OSL 128, batch 1"),
        ("step", {}, "50 steps, batch 1"),
        ("step", {"sample_size": 128, "vae_scale_factor": 8}, "50 steps, 1024px, batch 1"),
        ("step", {"sample_size": 64}, "50 steps, latent 64, batch 1"),
        ("inference", {}, "seq_len 384, batch 1"),
        ("inference", {"image_size": 224}, "224px, batch 1"),
        ("", {}, ""),
        ("nonsense", {}, ""),
    ],
)
def test_default_conditions_are_keyed_on_the_unit_not_the_model_family(unit, cfg, expect):
    """ISL/OSL only mean something for an autoregressive unit. A diffusion model's condition is steps
    and resolution; a classifier's is one forward at batch 1. One default for all three would describe
    a workload that does not exist for two of them."""
    mb = _mb()
    assert mb.conditions_label(mb.default_conditions(unit, cfg)) == expect


def test_isl_osl_default_to_the_standard_short_context_point():
    """THE GAP THIS CLOSES: ISL and OSL appear in no config.json, so whoever writes the perf test picks
    them -- and an LLM asked to fill in a prompt picked six tokens, which nothing then recorded."""
    c = _mb().default_conditions("token")
    assert c["isl"] == 128 and c["osl"] == 128 and c["batch"] == 1


def test_a_config_stated_limit_wins_over_the_default():
    """A condition the model states is read, never defaulted: asking for 128 tokens of context from a
    model whose max_position_embeddings is 64 would simply fail."""
    mb = _mb()
    assert mb.default_conditions("token", {"max_position_embeddings": 64})["isl"] == 64
    assert mb.default_conditions("inference", {"max_position_embeddings": 32})["seq_len"] == 32


def test_resolution_and_seq_len_are_mutually_exclusive():
    """seq_len is the TEXT fallback; a model stating an image size is not a text model."""
    c = _mb().default_conditions("inference", {"image_size": 384})
    assert c["resolution"] == 384 and "seq_len" not in c


def test_a_unit_less_model_gets_no_conditions_and_no_ceiling():
    """Same safe direction as the ceiling: no unit of work -> publish nothing rather than invent it."""
    mb = _mb()
    for tag in ("reinforcement-learning", "tabular-regression", "not-a-tag", ""):
        assert mb.unit_for_tag(tag) == ""
        assert mb.default_conditions(mb.unit_for_tag(tag)) == {}


def test_the_step_count_is_the_diffusers_documented_default():
    """50, not a round number someone liked: diffusers documents
    `StableDiffusionPipeline.__call__(num_inference_steps: int = 50)`, so the default is citable."""
    assert _mb().default_conditions("step")["steps"] == 50


def test_a_unets_sample_size_is_latent_not_pixels():
    """THE TRAP: diffusers documents height as `unet.config.sample_size * vae_scale_factor`, so
    SD-1.5's sample_size=64 is a 512px image. Reporting 64px understates the workload 8x per side."""
    mb = _mb()
    assert mb.default_conditions("step", {"sample_size": 64, "vae_scale_factor": 8})["resolution"] == 512
    assert mb.default_conditions("step", {"sample_size": 128, "vae_scale_factor": 8})["resolution"] == 1024


def test_an_unknown_scale_factor_reports_latent_rather_than_guessing_pixels():
    """8 is the SD-family VAE factor, not a law. A guessed multiplier is how a plausible-looking wrong
    number reaches a report, so say what is known instead."""
    c = _mb().default_conditions("step", {"sample_size": 64})
    assert c.get("latent") == 64 and "resolution" not in c


def test_an_image_processor_size_is_already_pixels():
    """`image_size` on a vision config is pixels, so it must NOT be scaled."""
    assert _mb().default_conditions("inference", {"image_size": 224})["resolution"] == 224


def test_every_hf_tag_with_a_definable_unit_of_work_has_one():
    """Coverage is checkable, not a matter of opinion: HF publishes the tag list. Two were genuinely
    missing -- audio-to-audio (a speech enhancer IS one forward pass over a segment) and
    table-question-answering (TAPAS is an encoder pass) -- so those models got no ceiling at all."""
    mb = _mb()
    import sys as _s
    from pathlib import Path as _P

    _s.path.insert(0, str(_P(__file__).resolve().parents[3] / "../scripts/tt_hw_planner/tests"))
    from test_pipeline_category_coverage import HF_TAGS  # the pinned HF /api/tasks snapshot

    unitless = sorted(t for t in HF_TAGS if not mb.unit_for_tag(t))
    assert unitless == sorted(mb.NO_UNIT_TAGS), (
        "a tag lost or gained a unit without the deliberate list being updated: %s" % unitless
    )


def test_the_unitless_set_is_deliberate_and_small():
    """Three tags have no fixed weight-read-per-unit: an RL rollout's length is not a model property,
    and tabular models are usually trees rather than nets. Pinned so the set cannot grow silently."""
    mb = _mb()
    assert len(mb.NO_UNIT_TAGS) == 3
    for tag in mb.NO_UNIT_TAGS:
        assert mb.unit_for_tag(tag) == "" and mb.default_conditions(mb.unit_for_tag(tag)) == {}


def test_an_audio_workload_is_a_duration_not_a_token_count():
    """ "seq_len 128" describes nothing for a speech enhancer. Whisper carries chunk_length=30."""
    mb = _mb()
    c = mb.default_conditions("inference", {"chunk_length": 30})
    assert c["seconds"] == 30 and "seq_len" not in c
    assert mb.conditions_label(c) == "30s audio, batch 1"


def test_no_audio_duration_is_invented_when_the_config_is_silent():
    """A segment length is a preprocessing choice; guessing one puts a fabricated condition in a report."""
    assert "seconds" not in _mb().default_conditions("inference", {})


def test_the_text_encoder_sequence_length_is_the_mlperf_reference():
    """The one condition with no HF number to inherit: max_position_embeddings is a CAP and HF
    pipelines pad to the batch. 384 is MLPerf's BERT inference sequence length, so the default cites a
    published reference instead of a figure chosen for internal consistency."""
    assert _mb().default_conditions("inference")["seq_len"] == 384
