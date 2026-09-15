# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The end-to-end CustomVoice pipeline: dual-track prompt assembly and the decode loop.

**This file needs the CustomVoice checkpoint**, which the rest of the suite does not. The
two releases are complementary: Base carries `speaker_encoder` but leaves `spk_id` empty,
CustomVoice carries the nine speakers but no speaker encoder. So the speaker-encoder tests
want Base and these want CustomVoice, and the checkpoint is resolved here by repo id rather
than from the ambient `$QWEN3_TTS_CKPT`.

What is checked, and what cannot be:

  * The prefill's shape and composition, position by position, against the same tables
    upstream builds it from. Bit-exactness against upstream's own assembled prefill was
    verified separately (max absolute difference 0.0 at all 14 positions for a 3-token
    utterance) but needs a second venv running transformers 4.57.3, so it cannot live here.
  * That the loop's steps agree with the CPU reference when both see the same prefix. Free
    running greedy decode does diverge: one near-tie flip changes the input to every later
    step. Measured 13 of 14 steps matching with a forced prefix, the single miss having a
    logit gap of 0.197.
"""

import os

import pytest
import torch

from models.demos.audio.qwen3_tts import frontend, weights
from models.demos.audio.qwen3_tts.reference.qwen3_talker_ref import TalkerReference, default_position_ids
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import (
    ROLE_IDS,
    TAIL_IDS,
    HostEmbeddings,
    Qwen3TTSPipeline,
    build_custom_voice_prefill,
)

CUSTOM_VOICE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TEXT = "Hello there."
SPEAKER = "ryan"
LANGUAGE = "English"

# How far the reference may prefer its own pick over the device's before a disagreement
# stops looking like rounding. Measured worst: 0.197.
MAX_PREFERENCE_GAP = 0.25
STEPS = 4


def _clear_caches():
    """Drop every cached view of the checkpoint, so switching does not leak across tests.

    `frontend` caches too, and forgetting it let this module's checkpoint switch change what
    a later tokenizer test saw: CustomVoice's `codec_language_id` carries 12 entries, the ten
    languages plus `beijing_dialect` and `sichuan_dialect` for its dialect speakers.
    """
    for cached in (
        weights.checkpoint_dir,
        weights._model_config_json,
        weights.codec_dir,
        weights._codec_config_json,
        frontend.tokenizer,
        frontend.special_tokens,
        frontend.language_ids,
    ):
        cached.cache_clear()


@pytest.fixture(scope="module", autouse=True)
def custom_voice_checkpoint():
    """Point the whole module at CustomVoice, whatever the ambient setting is."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(CUSTOM_VOICE_REPO)
    previous = os.environ.get("QWEN3_TTS_CKPT")
    os.environ["QWEN3_TTS_CKPT"] = path
    _clear_caches()
    yield path
    if previous is None:
        os.environ.pop("QWEN3_TTS_CKPT", None)
    else:
        os.environ["QWEN3_TTS_CKPT"] = previous
    _clear_caches()


@pytest.fixture(scope="module")
def tables(custom_voice_checkpoint):
    return HostEmbeddings()


# ── the prompt ──────────────────────────────────────────────────────────────


def test_this_checkpoint_has_speakers_and_no_speaker_encoder():
    """The premise of this file: the two releases carry different halves."""
    assert weights.talker_config()["spk_id"], "CustomVoice must define speakers"
    assert "speaker_encoder_config" not in weights.model_config(), "and must not carry a speaker encoder"


def test_prefill_length_is_text_plus_eleven(tables):
    """3 role + 6 think/speaker/pad + (n_text + 1) text and eos + 1 codec_bos."""
    embeddings, prompt_ids = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    n_text = len(prompt_ids) - ROLE_IDS - TAIL_IDS

    assert embeddings.shape == (1, n_text + 11, 2048)


def test_prefill_composition_position_by_position(tables):
    """Each position is a text-track embedding plus a codec-track one; check every seam."""
    config = weights.talker_config()
    embeddings, prompt_ids = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]

    codec = lambda ids: tables.codec(ids).reshape(-1, 2048)
    think = codec(
        [
            config["codec_think_id"],
            config["codec_think_bos_id"],
            frontend.language_id(LANGUAGE),
            config["codec_think_eos_id"],
        ]
    )
    pad = tables.tts_pad.reshape(-1)

    # 0-2: role tokens, text track only.
    assert torch.allclose(embeddings[0, :ROLE_IDS], tables.text(prompt_ids[:ROLE_IDS]).reshape(-1, 2048))
    # 3-6: the think block against tts_pad.
    assert torch.allclose(embeddings[0, 3:7], think + pad)
    # 7: the speaker id against tts_pad.
    assert torch.allclose(embeddings[0, 7], codec([config["spk_id"][SPEAKER]])[0] + pad)
    # 8: codec_pad against tts_bos.
    assert torch.allclose(embeddings[0, 8], codec([config["codec_pad_id"]])[0] + tables.tts_bos.reshape(-1))
    # 9 onward: the text then tts_eos, each against codec_pad.
    body = torch.cat([tables.text(text_ids), tables.tts_eos], dim=1).reshape(-1, 2048)
    assert torch.allclose(embeddings[0, 9 : 9 + len(text_ids) + 1], body + codec([config["codec_pad_id"]])[0])
    # last: codec_bos against tts_pad.
    assert torch.allclose(embeddings[0, -1], codec([config["codec_bos_id"]])[0] + pad)


def test_language_and_speaker_never_enter_the_text_stream(tables):
    """Both are single codec-vocabulary ids, so the text ids must not mention them."""
    _, prompt_ids = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    assert frontend.language_id(LANGUAGE) not in prompt_ids
    assert weights.talker_config()["spk_id"][SPEAKER] not in prompt_ids


def test_an_unknown_speaker_is_refused(tables, expect_error):
    with expect_error(ValueError, "unknown speaker"):
        build_custom_voice_prefill(TEXT, "nobody", LANGUAGE, tables)


# ── the loop ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_loop_steps_agree_with_the_reference(device, tables):
    """Each step judged on the same prefix, so an earlier flip cannot cascade into it."""
    pipeline = Qwen3TTSPipeline(device)
    reference = TalkerReference(dtype=torch.float32)
    head = tables.codec_head

    embeddings, _ = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    exact, wide = 0, []
    for step in range(STEPS):
        hidden, device_logits = pipeline._talker_step(embeddings)
        reference_hidden = reference(embeddings, position_ids=default_position_ids(embeddings.shape[1]))[:, -1, :]
        reference_logits = (reference_hidden @ head.T).reshape(-1)

        device_pick, reference_pick = int(device_logits.argmax()), int(reference_logits.argmax())
        if device_pick == reference_pick:
            exact += 1
        else:
            gap = float(reference_logits[reference_pick] - reference_logits[device_pick])
            print(f"  step {step}: device {device_pick}, reference {reference_pick}, preferred by {gap:.4f}")
            if gap > MAX_PREFERENCE_GAP:
                wide.append(f"step {step} gap {gap:.4f}")

        # Advance on the reference's own choice so the prefix stays shared.
        rest = pipeline.predictor.generate(hidden, reference_pick)
        embeddings = torch.cat([embeddings, pipeline._frame_embedding([reference_pick] + list(rest))], dim=1)

    print(f"steps matching exactly {exact}/{STEPS}")
    assert not wide, "steps the reference feels strongly about: " + "; ".join(wide)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_generate_produces_audio_of_the_right_length(device):
    """A short utterance end to end: frames in, 1920 samples per frame out."""
    pipeline = Qwen3TTSPipeline(device)
    waveform, codes = pipeline.generate(TEXT, speaker=SPEAKER, language=LANGUAGE, max_frames=24)

    assert codes.shape[1] == 16, "every frame carries 16 codebooks"
    assert codes.shape[0] >= 1
    assert waveform.shape == (1, codes.shape[0] * 1920)
    assert torch.isfinite(waveform).all()
    assert waveform.abs().max() <= 1.0
    assert int(codes.max()) < weights.codec_decoder_config()["codebook_size"], "no control id may reach the codec"
    print(f"generated {codes.shape[0]} frames -> {waveform.shape[1] / 24000:.2f} s")
