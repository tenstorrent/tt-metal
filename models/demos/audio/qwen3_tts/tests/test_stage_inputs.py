# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""A vLLM-Omni request's `additional_information` -> the same prompt the pipeline builds.

Host only: the encoders are skipped by giving the request an encoded `voice_clone_prompt`,
which is one of the two ways vLLM-Omni sends a clip. What is tested is the reading of the
payload, which builder it selects, and the fit to the scheduler's prompt length.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/test_stage_inputs.py
"""

import pytest
import torch

from models.demos.audio.qwen3_tts import stage_inputs
from models.demos.audio.qwen3_tts.tests.checkpoints import hidden_width
from models.demos.audio.qwen3_tts.tests.reference_helpers import codebook_size
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import (
    HostEmbeddings,
    build_streaming_clone_prefill,
    build_streaming_x_vector_prefill,
    build_voice_clone_prefill,
    build_x_vector_prefill,
)

TEXT = "This voice was cloned on Tenstorrent hardware."
REFERENCE_TEXT = "Good morning. This sentence is being spoken by a language model."
LANGUAGE = "English"
REFERENCE_FRAMES = 25


@pytest.fixture(scope="module")
def tables():
    return HostEmbeddings()


@pytest.fixture(scope="module")
def encoded():
    """An encoded clip as vLLM-Omni carries it: codes [frames, 16] and a voice vector."""
    generator = torch.Generator().manual_seed(0)
    codes = torch.randint(0, codebook_size(), (REFERENCE_FRAMES, 16), generator=generator)
    voice = torch.randn(hidden_width(), generator=generator)
    return codes, voice


def _info(encoded, **fields):
    codes, voice = encoded
    prompt = {"ref_code": codes.tolist(), "ref_spk_embedding": voice.tolist(), "icl_mode": True}
    prompt.update(fields.pop("prompt", {}))
    info = {
        "task_type": ["Base"],
        "text": [TEXT],
        "language": [LANGUAGE],
        "ref_text": [REFERENCE_TEXT],
        "voice_clone_prompt": [prompt],
    }
    info.update({key: [value] for key, value in fields.items()})
    return info


@pytest.mark.parametrize(
    "fields, builder, streams",
    [
        ({}, build_streaming_clone_prefill, True),
        ({"non_streaming_mode": True}, build_voice_clone_prefill, False),
        ({"prompt": {"icl_mode": False}}, build_streaming_x_vector_prefill, True),
        ({"prompt": {"icl_mode": False}, "non_streaming_mode": True}, build_x_vector_prefill, False),
    ],
)
def test_the_payload_selects_the_builder_upstream_would(tables, encoded, fields, builder, streams):
    """Base streams the text unless told not to; `icl_mode` decides in context or voice only."""
    info = _info(encoded, **fields)
    reference = stage_inputs.clone_reference(info)
    expected = builder(TEXT, reference, LANGUAGE, tables)
    expected = expected[0] if isinstance(expected, tuple) else expected
    length = expected.shape[1]

    embeddings, feed = stage_inputs.build_stage_prefill(info, reference, length, tables)

    assert torch.equal(embeddings, expected)
    assert (feed is not None) == streams


def test_an_encoded_clip_keeps_its_frames_in_order(encoded):
    codes, voice = encoded
    reference = stage_inputs.clone_reference(_info(encoded))

    assert torch.equal(reference.codes, codes.t()), "[frames, 16] must transpose, not reshape"
    assert torch.equal(reference.speaker_embedding, voice.reshape(1, -1))
    assert reference.text == REFERENCE_TEXT


def test_a_voice_only_clip_carries_no_codes(encoded):
    reference = stage_inputs.clone_reference(_info(encoded, prompt={"icl_mode": False}))

    assert reference.codes is None


def test_a_batch_of_one_ref_code_is_unwrapped(encoded):
    codes, _ = encoded
    reference = stage_inputs.clone_reference(_info(encoded, prompt={"ref_code": [codes.tolist()]}))

    assert torch.equal(reference.codes, codes.t())


def test_ref_code_in_the_wrong_layout_is_refused(encoded, expect_error):
    codes, _ = encoded
    with expect_error(ValueError, "ref_code must be"):
        stage_inputs.clone_reference(_info(encoded, prompt={"ref_code": codes.t().tolist()}))


@pytest.mark.parametrize("extra", [0, 3])
def test_the_prompt_is_fit_to_the_schedulers_length(tables, encoded, extra):
    """Short of the estimate: `tts_pad` fills the rest. Past it: cut, as vLLM-Omni does."""
    info = _info(encoded)
    reference = stage_inputs.clone_reference(info)
    built, _ = build_streaming_clone_prefill(TEXT, reference, LANGUAGE, tables)
    length = built.shape[1]

    longer, _ = stage_inputs.build_stage_prefill(info, reference, length + extra, tables)
    shorter, _ = stage_inputs.build_stage_prefill(info, reference, length - extra, tables)

    assert longer.shape[1] == length + extra and shorter.shape[1] == length - extra
    assert torch.equal(longer[:, :length], built)
    assert torch.equal(longer[:, length:], tables.tts_pad.expand(1, extra, -1))
    assert torch.equal(shorter, built[:, : length - extra])


@pytest.mark.parametrize("task_type", [None, "CustomVoice", "VoiceDesign"])
def test_only_base_requests_are_built(tables, encoded, task_type, expect_error):
    info = _info(encoded)
    if task_type is None:
        del info["task_type"]  # vLLM-Omni's default is CustomVoice, not Base
    else:
        info["task_type"] = [task_type]
    reference = stage_inputs.clone_reference(info)

    with expect_error(ValueError, "only Base"):
        stage_inputs.build_stage_prefill(info, reference, 64, tables)


def test_in_context_cloning_refuses_an_instruction(tables, encoded, expect_error):
    info = _info(encoded, instruct="speak slowly")

    with expect_error(ValueError, "does not accept an instruction"):
        stage_inputs.build_stage_prefill(info, stage_inputs.clone_reference(info), 64, tables)


def test_a_clip_to_encode_needs_the_device(expect_error):
    info = {"task_type": ["Base"], "ref_text": [REFERENCE_TEXT], "ref_audio": [[[0.0] * 2400, 24000]]}

    with expect_error(ValueError, "needs the device"):
        stage_inputs.clone_reference(info)


def test_a_request_without_any_clip_is_refused(expect_error):
    with expect_error(ValueError, "needs ref_audio"):
        stage_inputs.clone_reference({"task_type": ["Base"], "ref_text": [REFERENCE_TEXT]})
