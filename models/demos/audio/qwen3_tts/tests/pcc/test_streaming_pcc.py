# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming text input: the second regime this model was trained in.

Upstream's `non_streaming_mode=False`, and its default. The difference is when each text
token reaches the model: the prompt carries the first and every frame brings the next, so
a CustomVoice prompt is ten positions whatever the text against `n_text + 11`.

Prompts and text tracks were diffed against upstream under transformers 4.57.3 at 0.0,
twelve comparisons, which needs two transformers versions and cannot live here. It caught
a second `tts_eos` after the prompt had closed the text track, and a projection split
worth 7.5e-08; both are pinned below.
"""

import pytest
import torch

from models.demos.audio.qwen3_tts import frontend, weights
from models.demos.audio.qwen3_tts.tests.checkpoints import hidden_width, use_release
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import (
    ROLE_IDS,
    TAIL_IDS,
    CloneReference,
    HostEmbeddings,
    Qwen3TTSPipeline,
    StreamingText,
    build_custom_voice_prefill,
    build_streaming_clone_prefill,
    build_streaming_prefill,
)

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]

SPEAKER = "ryan"
LANGUAGE = "English"
TEXT = "The kettle is on."
LONGER = "The kettle is on, and the rain has not let up since yesterday morning."

# Role, think block, speaker, codec_pad, first text token. One fewer with `Auto`.
STREAMING_PROMPT = 10
STREAMING_PROMPT_AUTO = 9


@pytest.fixture(scope="module", autouse=True)
def custom_voice_checkpoint():
    yield from use_release("custom_voice")


@pytest.fixture(scope="module")
def tables(custom_voice_checkpoint):
    return HostEmbeddings()


# ── the prompt ──────────────────────────────────────────────────────────────


def test_the_prompt_length_does_not_depend_on_the_text(tables):
    """The point of the regime: a sentence and a paragraph prefill the same ten positions."""
    short, _ = build_streaming_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    long, _ = build_streaming_prefill(LONGER * 4, SPEAKER, LANGUAGE, tables)
    assert short.shape == (1, STREAMING_PROMPT, hidden_width())
    assert long.shape == (1, STREAMING_PROMPT, hidden_width())

    plain_short, _ = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    plain_long, _ = build_custom_voice_prefill(LONGER * 4, SPEAKER, LANGUAGE, tables)
    assert plain_short.shape[1] < plain_long.shape[1], "non-streaming grows with the text"
    print(
        f"streaming {short.shape[1]} positions for both; non-streaming {plain_short.shape[1]} and {plain_long.shape[1]}"
    )


def test_auto_drops_a_position(tables):
    """No language tag means no language id in the think block, in either regime."""
    tagged, _ = build_streaming_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    auto, _ = build_streaming_prefill(TEXT, SPEAKER, "Auto", tables)
    assert tagged.shape[1] == STREAMING_PROMPT
    assert auto.shape[1] == STREAMING_PROMPT_AUTO


def test_the_prompt_is_the_non_streaming_head_plus_one_text_token(tables):
    """Everything up to the speaker is shared; the tenth position is where they part."""
    streaming, feed = build_streaming_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    plain, prompt_ids = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    shared = STREAMING_PROMPT - 1

    assert torch.equal(streaming[:, :shared], plain[:, :shared]), "the head must be the same prompt"

    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]
    codec_bos = weights.talker_config()["codec_bos_id"]
    want = tables.text([text_ids[0]]) + tables.codec([codec_bos])
    assert torch.equal(streaming[:, shared:], want), "the last position is the first text token on codec_bos"

    rest = torch.cat([feed.next() for _ in range(len(text_ids) - 1)], dim=1)
    assert torch.equal(rest, tables.text(text_ids[1:])), "the feed carries the rest of the text in order"


def test_the_feed_sends_eos_once_and_then_pads(tables):
    """Tokens, then one `tts_eos`, then pads. A second `tts_eos` is the bug this pins."""
    _, feed = build_streaming_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    n_text = len(frontend.text_ids(TEXT)) - ROLE_IDS - TAIL_IDS

    # The prompt took the first token, so the feed has n_text - 1 left, then eos at that index.
    positions = [feed.next() for _ in range(n_text + 3)]
    assert torch.equal(positions[n_text - 1], tables.tts_eos), "eos follows the last text token"
    for position in positions[n_text:]:
        assert torch.equal(position, tables.tts_pad), "everything after eos is a pad"
    assert feed.spent


def test_text_in_pieces_matches_the_same_text_whole(tables):
    """Feeding "a b" as ["a ", "b"] must give the same positions, when the split is clean.

    Same to fp32 rounding, not bit for bit: the feed projects whatever ids are waiting in one
    call, so pieces project in different batch sizes, and CPU matmuls round differently by batch
    size on some builds (torch 2.11 measured 1.2e-7 at worst). A different tokenisation, the
    failure this guards against, differs at order 1.
    """
    same = lambda a, b: torch.allclose(a, b, rtol=0, atol=1e-6)
    whole, whole_feed = build_streaming_prefill(
        "The kettle is on, and the rain has not let up.", SPEAKER, LANGUAGE, tables
    )
    pieces, piece_feed = build_streaming_prefill(
        iter(["The kettle is on,", " and the rain has not let up."]), SPEAKER, LANGUAGE, tables
    )
    assert same(whole, pieces), "the prompt must not depend on how the text arrived"

    for index in range(12):
        assert same(whole_feed.next(), piece_feed.next()), f"position {index} differed"


def test_a_split_inside_a_word_is_a_different_tokenisation(tables):
    """The caveat, pinned rather than hidden: tokenisation is greedy over what it is given."""
    clean = frontend.encode("kettle")
    split = frontend.encode("ket") + frontend.encode("tle")
    assert clean != split, "if these ever agree, this test has stopped measuring anything"
    print(f"'kettle' is {clean} whole and {split} split after three letters")


def test_streaming_needs_a_first_token(tables, expect_error):
    """An empty text cannot open the prompt, since the prompt carries a text token."""
    with expect_error(ValueError, "at least one token"):
        build_streaming_prefill("", SPEAKER, LANGUAGE, tables)


def test_the_feed_refuses_to_run_dry_mid_prompt(tables, expect_error):
    """A generator that yields nothing is the same failure as an empty string."""
    with expect_error(ValueError, "at least one token"):
        StreamingText(tables, iter([]))


# ── the clone prompt, which sums the two tracks instead ──────────────────────


def _reference(tables, frames=24, words=6):
    """A `CloneReference` with plausible codes and a transcript of known length."""
    generator = torch.Generator().manual_seed(0)
    codebook = weights.codec_decoder_config()["codebook_size"]
    codes = torch.randint(0, codebook, (16, frames), generator=generator)
    return CloneReference(
        codes=codes,
        speaker_embedding=torch.zeros(1, hidden_width()),
        text=" ".join(["word"] * words),
    )


def test_a_clone_prompt_sums_the_tracks_and_pads_the_shorter(tables):
    """Clip longer than the text: the text track pads out and nothing is left to stream."""
    reference = _reference(tables, frames=40)
    prompt, feed = build_streaming_clone_prefill(TEXT, reference, "Auto", tables)

    head = STREAMING_PROMPT_AUTO - 1  # the clone head has no text token of its own
    assert prompt.shape[1] == head + reference.frames + 1, "prompt is the head plus codec_bos and the clip"
    assert torch.equal(feed.next(), tables.tts_pad), "a spent text track feeds pads"
    assert feed.spent


def test_a_clone_prompt_leaves_a_long_text_to_stream(tables):
    """Text longer than the clip: the surplus goes to the feed rather than the prompt."""
    reference = _reference(tables, frames=8, words=2)
    long_text = LONGER * 2
    prompt, feed = build_streaming_clone_prefill(long_text, reference, "Auto", tables)

    head = STREAMING_PROMPT_AUTO - 1
    assert prompt.shape[1] == head + reference.frames + 1, "the codec track decides the length"

    spare = 0
    while not feed.spent:
        feed.next()
        spare += 1
        assert spare < 500, "the feed never finished"
    assert spare > 1, "a text this long must leave positions over"
    print(f"{reference.frames} reference frames, {spare} text positions left to stream")


def test_the_clone_prompt_does_not_repeat_eos(tables):
    """The prompt closes the text track, so the feed must not send `tts_eos` again."""
    reference = _reference(tables, frames=40)
    _, feed = build_streaming_clone_prefill(TEXT, reference, "Auto", tables)
    for _ in range(4):
        assert torch.equal(feed.next(), tables.tts_pad), "eos was sent twice"


# ── on device ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_streaming_generates_audio_of_the_right_length(device):
    """End to end in the streaming regime: frames in, 1920 samples a frame out."""
    pipeline = Qwen3TTSPipeline(device, max_frames=48, seed=0)
    waveform, codes = pipeline.generate(TEXT, speaker=SPEAKER, language=LANGUAGE, streaming=True)

    assert codes.shape[1] == 16
    assert waveform.shape == (1, codes.shape[0] * 1920)
    assert torch.isfinite(waveform).all()
    assert pipeline.last_timings["prompt"] == STREAMING_PROMPT
    print(f"{STREAMING_PROMPT}-position prompt -> {codes.shape[0]} frames, {waveform.shape[1] / 24000:.2f} s")


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_pieces_and_a_whole_string_decode_the_same(device):
    """The property that makes the regime usable: when text arrives cannot change the speech."""
    pipeline = Qwen3TTSPipeline(device, max_frames=48, seed=7)
    text = "The kettle is on, and the rain has not let up."

    pipeline.reseed(7)
    whole = pipeline.codes(text, speaker=SPEAKER, language=LANGUAGE, streaming=True)
    pipeline.reseed(7)
    pieces = pipeline.codes(
        iter(["The kettle is on,", " and the rain has not let up."]), speaker=SPEAKER, language=LANGUAGE, streaming=True
    )

    assert torch.equal(whole, pieces), "the same text in two pieces gave different speech"
    print(f"{whole.shape[0]} frames either way")
