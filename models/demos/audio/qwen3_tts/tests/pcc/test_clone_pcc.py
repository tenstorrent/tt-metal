# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Voice cloning: a reference clip becomes a prompt, and the prompt becomes speech.

**This file needs the Base checkpoint**, which is the ambient one. It is the complement of
`test_pipeline.py`: Base carries `speaker_encoder` and leaves `spk_id` empty, CustomVoice
carries the nine speakers and no encoder, so cloning is a Base-only path and the nine
speakers are a CustomVoice-only path.

A clone prompt joins four things that are each tested elsewhere: the codec encoder's codes,
the speaker encoder's vector (as wide as the talker), the tokenizer's ids for two pieces of
text, and the talker's own embedding tables. What is new here is the arrangement, which is
`generate_icl_prompt` with `non_streaming_mode=True`:

    3           role, text track only
    4           think / think_bos / language / think_eos, against tts_pad
    1           the speaker embedding, against tts_pad
    1           codec_pad, against tts_bos
    n_ref +     the reference transcript, then the text to speak, then tts_eos,
    n_text + 1  each against codec_pad
    1           codec_bos, against tts_pad
    T_ref       the reference clip, one summed frame per position, against tts_pad

**Bit-exactness against upstream was measured, not assumed.** The same prompt through
upstream's own `generate_icl_prompt`, captured at the talker's door under transformers
4.57.3 in a separate venv: max absolute difference 0.0 across all 72 positions, for both
`language="English"` (72 positions) and `language="Auto"` (71, no language tag). That check
needs two transformers versions in one comparison and cannot live in the suite, so the tests
here pin the composition against the tables instead.
"""

import pytest
import torch

from models.demos.audio.qwen3_tts import frontend, weights
from models.demos.audio.qwen3_tts.tests.checkpoints import hidden_width
from models.demos.audio.qwen3_tts.tests.reference_helpers import codebook_size, synthetic_voiced_clip
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import (
    REFERENCE_TAIL_IDS,
    ROLE_IDS,
    TAIL_IDS,
    CloneReference,
    HostEmbeddings,
    Qwen3TTSPipeline,
    build_clone_reference,
    build_voice_clone_prefill,
    build_x_vector_prefill,
)

TEXT = "This voice was cloned on Tenstorrent hardware."
REFERENCE_TEXT = "Good morning. This sentence is being spoken by a language model."
LANGUAGE = "English"

# The clip the device tests clone from. 2 s is 25 frames, enough for a prompt and short
# enough that the encoders' convolutions do not dominate the run.
CLIP_SECONDS = 2.0
REFERENCE_FRAMES = 25

# The pipeline's decoders run from captured traces, which need a trace region.
DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]


@pytest.fixture(scope="module")
def tables():
    return HostEmbeddings()


@pytest.fixture(scope="module")
def reference():
    """A reference built from valid codes and an arbitrary voice vector.

    The prompt assembly does not care where either came from, so the host tests here skip
    both encoders; `test_a_reference_clip_becomes_codes_and_a_voice` covers the real ones.
    """
    generator = torch.Generator().manual_seed(0)
    codes = torch.randint(0, codebook_size(), (16, REFERENCE_FRAMES), generator=generator)
    voice = torch.randn(1, hidden_width(), generator=generator)
    return CloneReference(codes, voice, REFERENCE_TEXT)


# ── host ────────────────────────────────────────────────────────────────────


def test_this_checkpoint_has_a_speaker_encoder_and_no_speakers():
    """The premise of this file, and the mirror of `test_pipeline.py`'s."""
    assert "speaker_encoder_config" in weights.model_config(), "Base must carry a speaker encoder"
    assert not weights.talker_config()["spk_id"], "and must leave the speaker table empty"


def test_the_clone_prompt_has_the_length_the_two_tracks_imply(tables, reference):
    """9 head + (n_ref + n_text + 1) text + (1 + T_ref) codec."""
    embeddings, prompt_ids = build_voice_clone_prefill(TEXT, reference, LANGUAGE, tables)
    n_text = len(prompt_ids) - ROLE_IDS - TAIL_IDS
    n_reference = len(frontend.reference_text_ids(REFERENCE_TEXT)) - ROLE_IDS - REFERENCE_TAIL_IDS

    assert embeddings.shape == (1, 9 + (n_reference + n_text + 1) + (1 + reference.frames), hidden_width())


def test_the_clone_prompt_position_by_position(tables, reference):
    """Every seam, against the same tables upstream builds the prompt from."""
    config = weights.talker_config()
    embeddings, prompt_ids = build_voice_clone_prefill(TEXT, reference, LANGUAGE, tables)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]
    reference_ids = frontend.reference_text_ids(REFERENCE_TEXT)[ROLE_IDS:-REFERENCE_TAIL_IDS]

    codec = lambda ids: tables.codec(ids).reshape(-1, hidden_width())
    pad = tables.tts_pad.reshape(-1)
    think = codec(
        [
            config["codec_think_id"],
            config["codec_think_bos_id"],
            frontend.language_id(LANGUAGE),
            config["codec_think_eos_id"],
        ]
    )

    # 0-2 role, 3-6 the think block, 7 the voice, 8 codec_pad against tts_bos.
    assert torch.allclose(embeddings[0, :ROLE_IDS], tables.text(prompt_ids[:ROLE_IDS]).reshape(-1, hidden_width()))
    assert torch.allclose(embeddings[0, 3:7], think + pad)
    assert torch.allclose(embeddings[0, 7], reference.speaker_embedding.reshape(-1) + pad)
    assert torch.allclose(embeddings[0, 8], codec([config["codec_pad_id"]])[0] + tables.tts_bos.reshape(-1))

    # The reference transcript comes before the text to speak, which is the whole point of
    # ICL: the model is shown what the clip said as well as how it sounded.
    spoken = torch.cat([tables.text(list(reference_ids) + list(text_ids)), tables.tts_eos], dim=1)
    spoken = spoken.reshape(-1, hidden_width()) + codec([config["codec_pad_id"]])[0]
    assert torch.allclose(embeddings[0, 9 : 9 + spoken.shape[0]], spoken)

    # Then codec_bos and the clip itself, one summed frame per position.
    voice = (
        torch.cat([codec([config["codec_bos_id"]]), tables.frames(reference.codes).reshape(-1, hidden_width())]) + pad
    )
    assert torch.allclose(embeddings[0, 9 + spoken.shape[0] :], voice)


def test_the_language_tag_swaps_the_think_id(tables, reference):
    """`Auto` means no tag, and upstream marks that with `codec_nothink_id`, not `think`.

    Regression: using `codec_think_id` for both left the prompt one embedding row wrong in
    a position no length check would notice, on what is upstream's default language.
    """
    config = weights.talker_config()
    tagged, _ = build_voice_clone_prefill(TEXT, reference, "English", tables)
    untagged, _ = build_voice_clone_prefill(TEXT, reference, "Auto", tables)

    assert untagged.shape[1] == tagged.shape[1] - 1, "dropping the tag drops a position"
    codec = lambda name: tables.codec([config[name]]).reshape(-1)
    pad = tables.tts_pad.reshape(-1)
    assert torch.allclose(untagged[0, 3], codec("codec_nothink_id") + pad)
    assert torch.allclose(tagged[0, 3], codec("codec_think_id") + pad)
    assert not torch.allclose(untagged[0, 3], tagged[0, 3])


def test_the_reference_frames_are_the_sum_of_sixteen_codebooks(tables, reference):
    """Codebook 0 reads the talker's table, 1 to 15 the code predictor's own."""
    frames = tables.frames(reference.codes)
    assert frames.shape == (1, reference.frames, hidden_width())

    expected = tables.codec_table[reference.codes[0]]
    for index, table in enumerate(tables.predictor_tables):
        expected = expected + table[reference.codes[index + 1]]
    assert torch.allclose(frames.reshape(-1, hidden_width()), expected)


def test_a_reference_without_a_transcript_is_refused(expect_error):
    """ICL conditions on the transcript as well as the codes, so it is not optional."""
    with expect_error(ValueError, "transcript"):
        CloneReference(torch.zeros(16, 4, dtype=torch.long), torch.zeros(1, hidden_width()), "")


# ── cloning from the voice alone: upstream's x_vector_only_mode ─────────────


def test_the_x_vector_prompt_is_the_custom_voice_prompt_with_a_measured_voice(tables, reference):
    """Upstream's other cloning mode, and the shape it takes.

    `x_vector_only_mode` puts the speaker vector where a named speaker would sit and uses
    neither the codes nor the transcript, so the prompt is the CustomVoice one. Diffed
    against upstream at 0.0 in both regimes.
    """
    voice_only = CloneReference(None, reference.speaker_embedding)
    prompt, _ = build_x_vector_prefill(TEXT, voice_only, "Auto", tables)
    icl, _ = build_voice_clone_prefill(TEXT, reference, "Auto", tables)

    n_text = len(frontend.text_ids(TEXT)) - ROLE_IDS - TAIL_IDS
    assert prompt.shape[1] == n_text + 10, "the clip's length must not reach this prompt"
    assert prompt.shape[1] < icl.shape[1]
    # Position 6 with `Auto`: three role, three think, then the speaker, on a pad as always.
    voice_position = tables.tts_pad + reference.speaker_embedding.reshape(1, 1, -1)
    assert torch.equal(prompt[:, 6:7], voice_position), "the voice sits in one position, against tts_pad"
    print(f"x-vector {prompt.shape[1]} positions against ICL's {icl.shape[1]} for the same clip and text")


def test_a_voice_only_reference_needs_no_transcript():
    """The point of the mode: most clips come without one."""
    voice_only = CloneReference(None, torch.zeros(1, hidden_width()))
    assert voice_only.frames == 0
    assert "voice only" in repr(voice_only)


def test_an_icl_prompt_refuses_a_voice_only_reference(tables, expect_error):
    """A reference with no codes cannot fill the codec track ICL needs."""
    voice_only = CloneReference(None, torch.zeros(1, hidden_width()))
    with expect_error(ValueError, "only a speaker vector"):
        build_voice_clone_prefill(TEXT, voice_only, "Auto", tables)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_a_voice_only_reference_skips_the_codec_encoder(device):
    """Half the work of a full reference: 3.38 s against 4.39 on a 7.28 s clip."""
    clip = synthetic_voiced_clip(seconds=CLIP_SECONDS, voice="low")
    voice_only = build_clone_reference(device, clip, x_vector_only=True)
    full = build_clone_reference(device, clip, REFERENCE_TEXT)

    assert voice_only.codes is None and voice_only.frames == 0
    assert voice_only.speaker_embedding.shape == (1, hidden_width())
    assert torch.allclose(voice_only.speaker_embedding, full.speaker_embedding), "same clip, same voice"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_x_vector_cloning_produces_audio_of_the_right_length(device):
    """No reference frames in the prompt, so none to decode alongside and none to cut."""
    clip = synthetic_voiced_clip(seconds=CLIP_SECONDS, voice="low")
    voice_only = build_clone_reference(device, clip, x_vector_only=True)
    pipeline = Qwen3TTSPipeline(device, max_frames=32, seed=0)
    waveform, codes = pipeline.generate_clone(TEXT, voice_only, x_vector_only=True)

    assert waveform.shape == (1, codes.shape[0] * 1920), "no reference frames to trim"
    assert torch.isfinite(waveform).all() and waveform.abs().max() <= 1.0
    print(f"{codes.shape[0]} frames -> {waveform.shape[1] / 24000:.2f} s from a voice with no clip behind it")


def test_codes_in_the_wrong_layout_are_refused(expect_error):
    """Upstream's layout is [T, 16] and this directory's is [16, T].

    Reshaping one into the other reinterprets the memory rather than transposing it, so a
    caller who hands over upstream's layout would get a scrambled voice and no error. 16
    frames would even pass a shape check, which is why the first dimension is what is
    tested.
    """
    with expect_error(ValueError, "16, frames"):
        CloneReference(torch.zeros(20, 16, dtype=torch.long), torch.zeros(1, hidden_width()), REFERENCE_TEXT)
    with expect_error(ValueError, "16, frames"):
        CloneReference(torch.zeros(16, dtype=torch.long), torch.zeros(1, hidden_width()), REFERENCE_TEXT)


# ── device ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_a_reference_clip_becomes_codes_and_a_voice(device):
    """Both encoders, joined: a clip in, a usable `CloneReference` out.

    The codes and the embedding are each measured against their own reference elsewhere.
    What this adds is that the two run together off one clip, at the sample rate they
    share, and that the result says something about the voice: two different voices must
    not produce the same vector.
    """
    low = build_clone_reference(device, synthetic_voiced_clip(seconds=CLIP_SECONDS, voice="low"), REFERENCE_TEXT)
    high = build_clone_reference(device, synthetic_voiced_clip(seconds=CLIP_SECONDS, voice="high"), REFERENCE_TEXT)

    assert low.frames == REFERENCE_FRAMES, f"a {CLIP_SECONDS} s clip is {REFERENCE_FRAMES} frames, got {low.frames}"
    assert low.codes.shape == (16, REFERENCE_FRAMES)
    assert int(low.codes.min()) >= 0 and int(low.codes.max()) < codebook_size()
    assert low.speaker_embedding.shape == (1, hidden_width())
    assert torch.isfinite(low.speaker_embedding).all()

    similarity = torch.nn.functional.cosine_similarity(low.speaker_embedding, high.speaker_embedding).item()
    print(f"low against high: cosine {similarity:.4f}, {int((low.codes != high.codes).sum())} codes differ")
    assert similarity < 0.99, f"two different voices gave the same vector: cosine {similarity:.4f}"
    assert not torch.equal(low.codes, high.codes), "and the same codes"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_generate_clone_produces_audio_of_the_right_length(device):
    """The whole clone path: clip in, speech out, with the reference trimmed off the front.

    The reference frames are decoded together with the generated ones because the codec
    decoder is causal, then cut off the front. The cut is exact: every frame is 1920 samples.
    """
    frames = 16
    reference = build_clone_reference(device, synthetic_voiced_clip(seconds=CLIP_SECONDS), REFERENCE_TEXT)
    pipeline = Qwen3TTSPipeline(device, max_frames=frames, seed=0)

    waveform, codes = pipeline.generate_clone(TEXT, reference, language=LANGUAGE, max_frames=frames)

    assert codes.shape[1] == 16, f"16 codebooks per frame, got {codes.shape[1]}"
    assert 0 < codes.shape[0] <= frames
    assert waveform.shape == (1, codes.shape[0] * 1920), f"got {tuple(waveform.shape)}"
    assert torch.isfinite(waveform).all()
    assert waveform.abs().max() <= 1.0
    print(f"{codes.shape[0]} frames, {waveform.shape[1] / 24000:.2f} s, peak {waveform.abs().max():.4f}")
