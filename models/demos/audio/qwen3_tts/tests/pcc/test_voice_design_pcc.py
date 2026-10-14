# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""VoiceDesign: a voice described in a sentence of English, with no clip and no speaker id.

**This file needs the VoiceDesign checkpoint**, the third release. All three carry the same
architecture and differ in `tts_model_type` and their weights, so every ported block runs on
it unchanged; what differs is the prompt. The checkpoint is resolved here at the ambient
size, the way `test_pipeline.py` resolves CustomVoice. There is no 0.6B VoiceDesign, so at
0.6B the whole module skips.

The prompt, which is upstream's `generate` with `instruct_ids` set and
`non_streaming_mode=True`:

    n_instruct  the instruction, text-projected, on the text track alone
    3           role, text track only
    4           think / think_bos / language / think_eos, against tts_pad
    1           codec_pad, against tts_bos
    n_text + 1  the text to speak, then tts_eos, each against codec_pad
    1           codec_bos, against tts_pad

Two things separate it from the CustomVoice prompt. There is **no speaker position**, since
VoiceDesign defines no speakers, which shortens the head from nine positions to eight. And
the instruction is placed **whole**: the text to speak has its role tokens sliced off, the
instruction keeps its `<|im_start|>user` opener and `<|im_end|>` closer.

**Bit-exactness against upstream was measured**, by capturing its assembled prompt at the
talker's door under transformers 4.57.3: max absolute difference 0.0 at 37 positions with a
language tag, 33 with `Auto`, and 20 with an empty instruction. Two transformers versions
cannot share an environment, so the tests here pin the composition against the tables.

What the instruction is worth, measured with the speaker encoder from the Base checkpoint
on two designs of the same sentence: cosine 0.8923 between "a calm older man speaking
slowly" and "a bright young woman, cheerful and quick". One speaker measures about 0.99 and
an unrelated pair about 0.82, so the instruction moves the voice most of the way.
"""

import pytest
import torch

from models.demos.audio.qwen3_tts import frontend, weights
from models.demos.audio.qwen3_tts.tests.checkpoints import hidden_width, use_release
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import (
    ROLE_IDS,
    TAIL_IDS,
    HostEmbeddings,
    Qwen3TTSPipeline,
    build_voice_design_prefill,
    is_voice_design_checkpoint,
)

TEXT = "This voice was designed from a sentence of English."
INSTRUCTION = "A calm older man speaking slowly, with a slight rasp."
OTHER_INSTRUCTION = "A bright young woman, cheerful and quick, with a high clear voice."
LANGUAGE = "English"

# The head loses the speaker position CustomVoice has, so it is 8 rather than 9, and the
# body is the text, tts_eos and codec_bos.
HEAD = 8

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]


@pytest.fixture(scope="module", autouse=True)
def voice_design_checkpoint():
    """Point the whole module at VoiceDesign at the ambient size, or skip where none exists."""
    yield from use_release("voice_design")


@pytest.fixture(scope="module")
def tables(voice_design_checkpoint):
    return HostEmbeddings()


# ── host ────────────────────────────────────────────────────────────────────


def test_this_checkpoint_is_the_voice_design_release():
    """The premise of this file, and what `generate_design` checks before it runs."""
    assert is_voice_design_checkpoint()
    assert weights.model_config()["tts_model_type"] == "voice_design"
    assert not weights.talker_config()["spk_id"], "VoiceDesign names no speakers"
    assert "speaker_encoder_config" not in weights.model_config(), "and carries no speaker encoder"


def test_the_architecture_is_the_base_checkpoint_s():
    """Which is why every ported block runs on this release without a change.

    If a future release moved a dimension, this fails here rather than as a shape error
    somewhere inside the talker.
    """
    import json

    from huggingface_hub import hf_hub_download

    repo, revision = weights.sibling_repo("base")
    base = json.load(open(hf_hub_download(repo, "config.json", revision=revision)))
    assert weights.talker_config() == base["talker_config"]


def test_the_prompt_length_is_the_instruction_plus_the_text_plus_ten(tables):
    """8 head + (n_text + 1) text and eos + 1 codec_bos, behind the whole instruction."""
    embeddings, prompt_ids = build_voice_design_prefill(TEXT, INSTRUCTION, LANGUAGE, tables)
    n_text = len(prompt_ids) - ROLE_IDS - TAIL_IDS
    n_instruction = len(frontend.instruction_ids(INSTRUCTION))

    assert embeddings.shape == (1, n_instruction + HEAD + n_text + 2, hidden_width())


def test_the_prompt_position_by_position(tables):
    """Every seam, against the same tables upstream builds the prompt from."""
    config = weights.talker_config()
    embeddings, prompt_ids = build_voice_design_prefill(TEXT, INSTRUCTION, LANGUAGE, tables)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]
    instruction_ids = frontend.instruction_ids(INSTRUCTION)
    start = len(instruction_ids)

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

    # The instruction, text track only, with nothing from the codec track added.
    assert torch.allclose(embeddings[0, :start], tables.text(instruction_ids).reshape(-1, hidden_width()))
    # Then the role tokens, also text only.
    assert torch.allclose(
        embeddings[0, start : start + ROLE_IDS], tables.text(prompt_ids[:ROLE_IDS]).reshape(-1, hidden_width())
    )
    # The think block against tts_pad, and no speaker position after it.
    assert torch.allclose(embeddings[0, start + 3 : start + 7], think + pad)
    assert torch.allclose(embeddings[0, start + 7], codec([config["codec_pad_id"]])[0] + tables.tts_bos.reshape(-1))
    # The text, then tts_eos, each against codec_pad; then codec_bos against tts_pad.
    body = torch.cat([tables.text(text_ids), tables.tts_eos], dim=1).reshape(-1, hidden_width())
    assert torch.allclose(embeddings[0, start + HEAD : -1], body + codec([config["codec_pad_id"]])[0])
    assert torch.allclose(embeddings[0, -1], codec([config["codec_bos_id"]])[0] + pad)


def test_there_is_no_speaker_position(tables):
    """CustomVoice puts a speaker id at position 7; VoiceDesign has nothing to put there.

    Measured as a length: the head is 8 positions rather than 9, which is also why the
    checkpoint leaves `spk_id` empty.
    """
    embeddings, prompt_ids = build_voice_design_prefill(TEXT, "", LANGUAGE, tables)
    n_text = len(prompt_ids) - ROLE_IDS - TAIL_IDS

    assert embeddings.shape[1] == HEAD + n_text + 2
    # Position 7 is codec_pad against tts_bos, which in CustomVoice sits at 8.
    config = weights.talker_config()
    assert torch.allclose(
        embeddings[0, 7], tables.codec([config["codec_pad_id"]]).reshape(-1) + tables.tts_bos.reshape(-1)
    )


def test_an_empty_instruction_drops_its_block(tables):
    """Upstream treats an empty instruction as no instruction, leaving the model to invent."""
    with_it, _ = build_voice_design_prefill(TEXT, INSTRUCTION, LANGUAGE, tables)
    without, _ = build_voice_design_prefill(TEXT, "", LANGUAGE, tables)
    whitespace, _ = build_voice_design_prefill(TEXT, "   ", LANGUAGE, tables)

    assert without.shape[1] == with_it.shape[1] - len(frontend.instruction_ids(INSTRUCTION))
    assert torch.equal(without, whitespace), "whitespace is not an instruction either"
    # What is left is the CustomVoice prompt minus its speaker position.
    assert torch.equal(without, with_it[:, len(frontend.instruction_ids(INSTRUCTION)) :])


def test_the_instruction_keeps_its_role_tokens(tables):
    """The text to speak is sliced to its own tokens; the instruction is not.

    Easy to get wrong by symmetry, and invisible in a length check that counts the
    instruction's ids either way.
    """
    instruction_ids = frontend.instruction_ids(INSTRUCTION)
    special = frontend.special_tokens()

    assert instruction_ids[0] == special["im_start"], "the instruction opens its own turn"
    assert special["im_end"] in instruction_ids, "and closes it"

    embeddings, _ = build_voice_design_prefill(TEXT, INSTRUCTION, LANGUAGE, tables)
    assert torch.allclose(embeddings[0, 0], tables.text([instruction_ids[0]]).reshape(-1))


def test_the_other_releases_refuse_an_instruction(tables, monkeypatch, expect_error):
    """Base and CustomVoice were never shown an instruction, so asking is an error.

    Without the guard they would speak in some voice the description had no part in
    choosing, which is worse than failing. Checked here rather than on a device, since the
    prompt builder holds the check and needs no card.
    """
    monkeypatch.setattr(weights, "model_config", lambda *args, **kwargs: {"tts_model_type": "custom_voice"})
    assert not is_voice_design_checkpoint()

    with expect_error(ValueError, "VoiceDesign"):
        build_voice_design_prefill(TEXT, INSTRUCTION, LANGUAGE, tables)


def test_a_different_instruction_gives_a_different_prompt(tables):
    """The instruction has to reach the model, not sit beside it unused."""
    one, _ = build_voice_design_prefill(TEXT, INSTRUCTION, LANGUAGE, tables)
    two, _ = build_voice_design_prefill(TEXT, OTHER_INSTRUCTION, LANGUAGE, tables)

    shared = min(one.shape[1], two.shape[1])
    assert not torch.allclose(one[:, :shared], two[:, :shared])


# ── device ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_generate_design_speaks_and_the_instruction_changes_the_codes(device):
    """The whole path: an instruction in, speech out, and two instructions that differ.

    Comparing codes rather than voices keeps this on one checkpoint. Whether the voices
    differ *to the ear* is measured with the Base checkpoint's speaker encoder, which
    VoiceDesign does not carry; the module docstring records that number.
    """
    frames = 24
    pipeline = Qwen3TTSPipeline(device, max_frames=frames, seed=0)

    waveform, codes = pipeline.generate_design(TEXT, INSTRUCTION, language=LANGUAGE, max_frames=frames)
    assert codes.shape[1] == 16
    assert 0 < codes.shape[0] <= frames
    assert waveform.shape == (1, codes.shape[0] * 1920)
    assert torch.isfinite(waveform).all()
    assert waveform.abs().max() <= 1.0
    print(f"{codes.shape[0]} frames, {waveform.shape[1] / 24000:.2f} s, peak {waveform.abs().max():.4f}")

    pipeline.reseed(0)
    _, other = pipeline.generate_design(TEXT, OTHER_INSTRUCTION, language=LANGUAGE, max_frames=frames)
    shared = min(codes.shape[0], other.shape[0])
    differing = int((codes[:shared] != other[:shared]).sum())
    print(f"same seed, different instruction: {differing} of {shared * 16} codes differ")
    assert differing > 0, "the same seed and a different instruction must not give the same codes"
