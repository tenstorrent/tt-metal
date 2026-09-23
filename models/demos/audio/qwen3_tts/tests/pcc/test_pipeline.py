# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The end-to-end CustomVoice pipeline: dual-track prompt assembly and the decode loop.

**This file needs the CustomVoice checkpoint**, which the rest of the suite does not. The
two releases are complementary: Base carries `speaker_encoder` but leaves `spk_id` empty,
CustomVoice carries the nine speakers but no speaker encoder. So the speaker-encoder tests
want Base and these want CustomVoice, the one at the ambient checkpoint's size, rather than
the ambient `$QWEN3_TTS_CKPT` itself.

What is checked, and what cannot be:

  * The prefill's shape and composition, position by position, against the same tables
    upstream builds it from. Bit-exactness against upstream's own assembled prefill was
    verified separately, by capturing it at the talker's door under transformers 4.57.3:
    max absolute difference 0.0 for `ryan` with a language tag and without one, and for the
    dialect speaker `dylan` under Chinese, Auto and English. That needs two transformers
    versions in one comparison, so it cannot live here.
  * That the talker's steps agree with the CPU reference when both see the same prefix.
    Free running decode does diverge: one near-tie flip changes the input to every later
    step. Measured 13 of 14 steps matching with a forced prefix, the single miss having a
    logit gap of 0.197. That test drives the uncached graph, which `test_decode_pcc.py`
    then holds the pipeline's cached one to.
  * That a seeded run is reproducible and a full utterance comes out the right length.
    Whether the speech is *good* is not something a test settles; the pipeline samples with
    the checkpoint's own settings, and `sampling` records what greedy does instead.
"""

import pytest
import torch

from models.demos.audio.qwen3_tts import frontend, weights
from models.demos.audio.qwen3_tts.reference.qwen3_talker_ref import TalkerReference, default_position_ids
from models.demos.audio.qwen3_tts.tests.checkpoints import hidden_width, use_release
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor import (
    TtCodePredictor,
    preprocess_code_predictor_parameters,
)
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import (
    CONTROL_ID_COUNT,
    MIN_FRAMES,
    ROLE_IDS,
    TAIL_IDS,
    HostEmbeddings,
    Qwen3TTSPipeline,
    build_custom_voice_prefill,
)
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import TtTalker, preprocess_talker_parameters

TEXT = "Hello there."
LONGER_TEXT = "The kettle is on, and the rain has not let up."
INSTRUCTION = "Say it in a very angry tone."
SPEAKER = "ryan"
LANGUAGE = "English"

# How far the reference may prefer its own pick over the device's before a disagreement
# stops looking like rounding. Measured worst: 0.197.
MAX_PREFERENCE_GAP = 0.25
STEPS = 4

# The pipeline's decoders run from captured traces, which need a trace region.
DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]


@pytest.fixture(scope="module", autouse=True)
def custom_voice_checkpoint():
    """Point the whole module at CustomVoice, at the ambient checkpoint's size."""
    yield from use_release("custom_voice")


# Upstream drops an instruction at 0.6B and the pipeline refuses one there instead.
needs_instructions = pytest.mark.skipif(
    "weights.model_size() == '0b6'", reason="the 0.6B checkpoints take no instruction"
)


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

    assert embeddings.shape == (1, n_text + 11, hidden_width())


def test_prefill_composition_position_by_position(tables):
    """Each position is a text-track embedding plus a codec-track one; check every seam."""
    config = weights.talker_config()
    embeddings, prompt_ids = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]

    codec = lambda ids: tables.codec(ids).reshape(-1, hidden_width())
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
    assert torch.allclose(embeddings[0, :ROLE_IDS], tables.text(prompt_ids[:ROLE_IDS]).reshape(-1, hidden_width()))
    # 3-6: the think block against tts_pad.
    assert torch.allclose(embeddings[0, 3:7], think + pad)
    # 7: the speaker id against tts_pad.
    assert torch.allclose(embeddings[0, 7], codec([config["spk_id"][SPEAKER]])[0] + pad)
    # 8: codec_pad against tts_bos.
    assert torch.allclose(embeddings[0, 8], codec([config["codec_pad_id"]])[0] + tables.tts_bos.reshape(-1))
    # 9 onward: the text then tts_eos, each against codec_pad.
    body = torch.cat([tables.text(text_ids), tables.tts_eos], dim=1).reshape(-1, hidden_width())
    assert torch.allclose(embeddings[0, 9 : 9 + len(text_ids) + 1], body + codec([config["codec_pad_id"]])[0])
    # last: codec_bos against tts_pad.
    assert torch.allclose(embeddings[0, -1], codec([config["codec_bos_id"]])[0] + pad)


def test_language_and_speaker_never_enter_the_text_stream(tables):
    """Both are single codec-vocabulary ids, so the text ids must not mention them."""
    _, prompt_ids = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    assert frontend.language_id(LANGUAGE) not in prompt_ids
    assert weights.talker_config()["spk_id"][SPEAKER] not in prompt_ids


def test_a_dialect_speaker_overrides_the_language_tag(tables):
    """`eric` speaks Sichuanese and `dylan` Beijing, and the tag follows the speaker.

    Upstream replaces the language id with the dialect's whenever the language is Chinese or
    `Auto` and the speaker is one of those two. For them `Auto` therefore stops meaning "no
    tag": the prompt gains the tagged think block a plain `Auto` would not have.
    """
    config = weights.talker_config()
    assert config["spk_is_dialect"]["dylan"] == "beijing_dialect", "the premise of this test"
    assert config["spk_is_dialect"][SPEAKER] is False

    dialect_id = config["codec_language_id"]["beijing_dialect"]
    for language in ("Chinese", "Auto"):
        embeddings, _ = build_custom_voice_prefill(TEXT, "dylan", language, tables)
        assert torch.allclose(
            embeddings[0, 5], tables.codec([dialect_id]).reshape(-1) + tables.tts_pad.reshape(-1)
        ), f"{language} with a dialect speaker must tag the dialect"

    # A speaker who is not a dialect speaker keeps the plain behaviour, tag and all.
    plain, _ = build_custom_voice_prefill(TEXT, SPEAKER, "Auto", tables)
    tagged, _ = build_custom_voice_prefill(TEXT, "dylan", "Auto", tables)
    assert plain.shape[1] == tagged.shape[1] - 1, "Auto leaves the tag off for everyone else"

    # And an explicit non-Chinese language is left alone even for a dialect speaker.
    english, _ = build_custom_voice_prefill(TEXT, "dylan", LANGUAGE, tables)
    assert torch.allclose(
        english[0, 5], tables.codec([frontend.language_id(LANGUAGE)]).reshape(-1) + tables.tts_pad.reshape(-1)
    )


def test_an_unknown_speaker_is_refused(tables, expect_error):
    with expect_error(ValueError, "unknown speaker"):
        build_custom_voice_prefill(TEXT, "nobody", LANGUAGE, tables)


# ── the loop ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_talker_steps_on_the_real_prompt_agree_with_the_reference(device, tables):
    """Each step judged on the same prefix, so an earlier flip cannot cascade into it.

    Drives the uncached talker and the uncached predictor, which are the graphs the CPU
    reference is compared against everywhere else in this suite. The cached ones the
    pipeline actually runs are held to these in `test_decode_pcc.py`, on device, rather
    than against a second CPU pass here.
    """
    import ttnn

    talker = TtTalker(device, preprocess_talker_parameters(device))
    predictor = TtCodePredictor(device, preprocess_code_predictor_parameters(device))
    reference = TalkerReference(dtype=torch.float32)
    head = tables.codec_head

    embeddings, _ = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    exact, wide = 0, []
    for step in range(STEPS):
        length = embeddings.shape[1]
        cos, sin, mask = talker.host_inputs(length)
        to_device = lambda tensor: ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        hidden = talker(*(to_device(tensor) for tensor in (embeddings, cos, sin, mask)))
        row = ttnn.slice(hidden, [0, length - 1, 0], [1, length, hidden.shape[2]])
        device_hidden = ttnn.to_torch(row).float().reshape(1, 1, -1)
        device_logits = (device_hidden.reshape(-1) @ head.T).reshape(-1)
        ttnn.deallocate(hidden)

        reference_hidden = reference(embeddings, position_ids=default_position_ids(length))[:, -1, :]
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
        rest = predictor.generate(device_hidden, reference_pick)
        embeddings = torch.cat([embeddings, _frame_embedding(tables, [reference_pick] + list(rest))], dim=1)

    print(f"steps matching exactly {exact}/{STEPS}")
    assert not wide, "steps the reference feels strongly about: " + "; ".join(wide)


def _frame_embedding(tables, frame):
    """The 16 codebooks of one frame, summed with `tts_pad`: one prompt position."""
    total = tables.codec([frame[0]])
    for index, code in enumerate(frame[1:]):
        total = total + tables.predictor_tables[index][code].reshape(1, 1, -1)
    return total + tables.tts_pad


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_generate_produces_audio_of_the_right_length(device):
    """A short utterance end to end: frames in, 1920 samples per frame out."""
    pipeline = Qwen3TTSPipeline(device, max_frames=24, seed=0)
    waveform, codes = pipeline.generate(TEXT, speaker=SPEAKER, language=LANGUAGE)

    assert codes.shape[1] == 16, "every frame carries 16 codebooks"
    assert codes.shape[0] >= 1
    assert waveform.shape == (1, codes.shape[0] * 1920)
    assert torch.isfinite(waveform).all()
    assert waveform.abs().max() <= 1.0
    assert int(codes.max()) < weights.codec_decoder_config()["codebook_size"], "no control id may reach the codec"
    print(f"generated {codes.shape[0]} frames -> {waveform.shape[1] / 24000:.2f} s")


@needs_instructions
def test_an_instruction_joins_a_named_speaker(tables):
    """Upstream's `generate_custom_voice(..., instruct=...)`: a speaker and a delivery.

    Diffed against upstream at 0.0 in both regimes. Checked here: the instruction leads the
    prompt on the text track alone and changes nothing after it.
    """
    plain, _ = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    instructed, _ = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables, INSTRUCTION)

    block = tables.text(frontend.instruction_ids(INSTRUCTION))
    assert instructed.shape[1] == plain.shape[1] + block.shape[1]
    assert torch.equal(instructed[:, : block.shape[1]], block), "the instruction leads the prompt"
    assert torch.equal(instructed[:, block.shape[1] :], plain), "and changes nothing after it"
    print(f"{plain.shape[1]} positions without an instruction, {instructed.shape[1]} with one")


def test_an_empty_instruction_is_no_instruction(tables):
    """Whitespace and None behave like the plain prompt, as upstream treats them."""
    plain, _ = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    for empty in (None, "", "   "):
        got, _ = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables, empty)
        assert torch.equal(got, plain), f"{empty!r} changed the prompt"


@pytest.mark.skipif("weights.model_size() != '0b6'", reason="only the 0.6B checkpoints refuse an instruction")
def test_the_small_checkpoints_refuse_an_instruction(tables, expect_error):
    """Upstream silently drops `instruct` at 0.6B; speaking without it would ignore the caller."""
    with expect_error(ValueError, "do not take an instruction"):
        build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables, INSTRUCTION)
    plain, _ = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables, "  ")
    assert plain.shape[1] > 0, "an empty instruction is still no instruction, and allowed"


@needs_instructions
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_an_instruction_changes_what_a_named_speaker_does(device):
    """The instruction must reach the speech, not just the prompt."""
    pipeline = Qwen3TTSPipeline(device, max_frames=160, seed=4)
    counts = {}
    for label, instruct in (("plain", None), ("angry", "Say it in a very angry tone.")):
        pipeline.reseed(4)
        codes = pipeline.codes(LONGER_TEXT, speaker=SPEAKER, language=LANGUAGE, instruct=instruct)
        counts[label] = codes.shape[0]
    print(f"frames: {counts}")
    assert counts["plain"] != counts["angry"], "the instruction did not reach the decode loop"


# One sentence per language, with a matching speaker where the nine offer one.
LANGUAGE_CASES = (
    ("Chinese", "水壶已经烧开了，雨一直没有停。", "vivian"),
    ("Japanese", "やかんが沸いていて、雨はまだ止んでいません。", "ono_anna"),
    ("Korean", "주전자가 끓고 있고 비는 아직 그치지 않았습니다.", "sohee"),
    ("German", "Der Kessel kocht, und der Regen hat nicht aufgehört.", "ryan"),
    ("French", "La bouilloire est en marche et la pluie n'a pas cessé.", "ryan"),
    ("Spanish", "La tetera está puesta y la lluvia no ha parado.", "ryan"),
    ("Italian", "Il bollitore è acceso e la pioggia non è cessata.", "ryan"),
    ("Portuguese", "A chaleira está ligada e a chuva não parou.", "ryan"),
    ("Russian", "Чайник поставлен, и дождь всё ещё не прекратился.", "ryan"),
)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_every_language_decodes_and_stops(device):
    """Nine languages besides English, through the frame loop.

    A language is one codec-vocabulary id and nothing else about the prompt changes, which
    the prompt tests pin. This adds that each one decodes: stops on its own, codes inside the
    codebook, a length that is speech. Whether the speech is right was measured with Whisper
    instead, and the README has that table. Codes only, since ten frame counts would compile
    ten sets of convolution programs.
    """
    pipeline = Qwen3TTSPipeline(device, max_frames=200, seed=1)
    codebook = weights.codec_decoder_config()["codebook_size"]
    rows = []
    for language, text, speaker in LANGUAGE_CASES:
        pipeline.reseed(1)
        codes = pipeline.codes(text, speaker=speaker, language=language)
        frames = codes.shape[0]
        rows.append(f"{language} {frames}")
        assert frames >= MIN_FRAMES, f"{language}: {frames} frames is not an utterance"
        assert frames < 200, f"{language}: ran to the frame budget instead of stopping"
        assert int(codes.min()) >= 0 and int(codes.max()) < codebook, f"{language}: a control id reached the codec"
        assert codes.shape[1] == 16
    print("frames: " + ", ".join(rows))


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_no_utterance_is_shorter_than_two_frames(device):
    """`min_new_tokens=2` upstream: one frame is 80 ms, and the sampler can reach eos at once."""
    pipeline = Qwen3TTSPipeline(device, max_frames=48)
    shortest = 10**6
    for seed in range(10):
        pipeline.reseed(seed)
        shortest = min(shortest, pipeline.codes(TEXT, speaker=SPEAKER, language=LANGUAGE).shape[0])
    print(f"shortest utterance over ten seeds: {shortest} frames")
    assert shortest >= MIN_FRAMES


def test_control_ids_are_suppressed_and_end_of_speech_only_at_first(tables):
    """What the talker may draw: real codes, plus end-of-speech once two frames exist."""
    talker = weights.talker_config()
    vocab, eos = talker["vocab_size"], talker["codec_eos_token_id"]
    codebook = weights.codec_decoder_config()["codebook_size"]

    pipeline = Qwen3TTSPipeline.__new__(Qwen3TTSPipeline)  # the id sets need no device
    pipeline.talker_config, pipeline.eos = talker, eos
    control = [index for index in range(vocab - CONTROL_ID_COUNT, vocab) if index != eos]
    pipeline._control_ids = torch.tensor(control, dtype=torch.long)
    pipeline._control_ids_and_eos = torch.tensor(sorted(control + [eos]), dtype=torch.long)

    assert vocab - CONTROL_ID_COUNT == codebook, "the suppressed range should start where the codebook ends"
    early, late = pipeline._suppressed(0), pipeline._suppressed(MIN_FRAMES)
    assert eos in early.tolist() and eos not in late.tolist()
    assert set(late.tolist()) | {eos} == set(range(codebook, vocab))
    assert not [index for index in late.tolist() if index < codebook], "no real code may be suppressed"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_the_prompt_bucket_changes_nothing_it_keeps(device):
    """Padding the prompt up to a bucket must give the same frames as not padding.

    Attention is causal, so the filler cannot reach a real position and its cache slots are
    the ones decode overwrites before reading. If either claim were wrong these would differ.
    """
    from models.demos.audio.qwen3_tts.tt import ttnn_qwen3_pipeline as module

    pipeline = Qwen3TTSPipeline(device, max_frames=12, seed=5)
    original = module.PROMPT_BUCKET
    try:
        module.PROMPT_BUCKET = 1
        pipeline.reseed(5)
        exact = pipeline.codes(TEXT, speaker=SPEAKER, language=LANGUAGE)
        exact_prompt = pipeline.last_timings["padded_prompt"]

        module.PROMPT_BUCKET = original
        pipeline.reseed(5)
        bucketed = pipeline.codes(TEXT, speaker=SPEAKER, language=LANGUAGE)
        padded_prompt = pipeline.last_timings["padded_prompt"]
    finally:
        module.PROMPT_BUCKET = original

    assert padded_prompt > exact_prompt, "the test needs a prompt the bucket actually pads"
    assert torch.equal(exact, bucketed), "padding the prompt changed the frames"
    print(f"{exact_prompt} positions padded to {padded_prompt}: {exact.shape[0]} frames, identical")


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_a_seeded_run_repeats_itself(device):
    """Sampling costs reproducibility unless the seed is held, so hold it and check.

    Also the only test that runs two utterances through one pipeline, which is where a
    cache that outlived its `generate` would show up.
    """
    pipeline = Qwen3TTSPipeline(device, max_frames=12, seed=11)
    first, first_codes = pipeline.generate(TEXT, speaker=SPEAKER, language=LANGUAGE)

    pipeline.generator = torch.Generator().manual_seed(11)
    again, again_codes = pipeline.generate(TEXT, speaker=SPEAKER, language=LANGUAGE)

    assert torch.equal(first_codes, again_codes), "the same seed gave different codes"
    assert torch.equal(first, again), "the same codes gave a different waveform"
    print(f"{first_codes.shape[0]} frames reproduced exactly")
