# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Qwen3-TTS on device: text in, waveform out, in the voice you ask for.

Joins the device blocks to the two host pieces: dual-track prompt assembly and the
autoregressive loop. Five ways to choose a voice, each in either text regime, and the
README's tables say what they cost and how they were verified against upstream:

    generate(text, speaker=...)                   a named CustomVoice speaker
    generate(text, speaker=..., instruct=...)      that speaker, directed
    generate_design(text, instruction)            VoiceDesign, a voice in words
    generate_clone(text, reference)                Base, a clip in context
    generate_clone(..., x_vector_only=True)        Base, the clip's voice alone

`streaming=True` on any of them moves the text to one token per frame, upstream's
`non_streaming_mode=False`; `StreamingText` has that schedule.

Per frame the loop runs the talker over the prompt, the code predictor over its hidden
state for codebooks 1 to 15, and sums the 16 embeddings into the next position. When
codebook 0 comes back as `codec_eos_token_id` the frames go to the codec decoder.

Prompt layouts, the part that has to be exactly right, are in the builders below; each
was diffed against upstream's own assembly under transformers 4.57.3 at 0.0.

On device: the talker, the code predictor, the codec decoder, `codec_head`, and for a
clone the codec and speaker encoders. On host: tokenising, embedding gathers, the small
`text_projection`, the mel front-end, sampling and the quantizer's lookup.
"""

import time

import torch
import torch.nn.functional as F

import ttnn
from models.demos.audio.qwen3_tts import audio as host_audio
from models.demos.audio.qwen3_tts import frontend, sampling
from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.frontend import AUTO_LANGUAGE
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor_decode import (
    TtCodePredictorCachedDecoder,
    preprocess_cached_predictor_parameters,
)
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec import TtCodecDecoder, preprocess_codec_parameters
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec_encoder import TtCodecEncoder, preprocess_codec_encoder_parameters
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_speaker import speaker_embedding as run_speaker_encoder
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker_decode import (
    TtTalkerCachedDecoder,
    preprocess_cached_talker_parameters,
)

# Upstream's fixed slices into the prompt: three leading ids of role, five trailing. A
# reference transcript closes its turn instead of leaving one open, so it sheds two.
ROLE_IDS = 3
TAIL_IDS = 5
REFERENCE_TAIL_IDS = 2

# Frames the cache is sized for when the caller names no limit. 12.5 frames a second, so
# this is 32 s of speech.
DEFAULT_MAX_FRAMES = 400

# Prompt lengths the prefill rounds up to: each compiles its own programs, 1.41 s cold.
PROMPT_BUCKET = 32

# The gap between the talker's 3072 ids and the codec's 2048, which upstream suppresses.
CONTROL_ID_COUNT = 1024

MIN_FRAMES = 2  # upstream's `min_new_tokens=2`; one frame is 80 ms


class Stopwatch:
    """Wall clock split across the blocks of one utterance.

    `split(name)` charges everything since the last split to `name`. Device work is
    asynchronous, so `strict` syncs at every split to stop a block being charged for
    another's work; the perf test uses it.
    """

    def __init__(self, device=None, strict=False):
        self.device = device
        self.strict = strict and device is not None
        self.totals = {}
        self.mark = time.time()

    def split(self, name):
        if self.strict:
            ttnn.synchronize_device(self.device)
        now = time.time()
        self.totals[name] = self.totals.get(name, 0.0) + now - self.mark
        self.mark = now

    def restart(self):
        if self.strict:
            ttnn.synchronize_device(self.device)
        self.mark = time.time()


class HostEmbeddings:
    """The lookup tables and the small projection that the prompt is built from."""

    def __init__(self, dtype=torch.float32):
        talker = checkpoint.load_prefixed("talker.", dtype=dtype, strip=True)
        self.text_table = talker["model.text_embedding.weight"]
        self.codec_table = talker["model.codec_embedding.weight"]
        self.fc1 = (talker["text_projection.linear_fc1.weight"], talker["text_projection.linear_fc1.bias"])
        self.fc2 = (talker["text_projection.linear_fc2.weight"], talker["text_projection.linear_fc2.bias"])
        self.codec_head = talker["codec_head.weight"]
        # 2048 at 1.7B, 1024 at 0.6B; a wrong width folds positions rather than failing.
        self.width = self.codec_table.shape[1]

        predictor = checkpoint.load_prefixed("talker.code_predictor.model.codec_embedding.", dtype=dtype)
        self.predictor_tables = [predictor[f"{index}.weight"] for index in range(len(predictor))]

        special = frontend.special_tokens()
        # Upstream builds these three together and chunks them in this order.
        stacked = self.project(self.text_table[[special["tts_bos"], special["tts_eos"], special["tts_pad"]]])
        self.tts_bos, self.tts_eos, self.tts_pad = (row.reshape(1, 1, -1) for row in stacked)

    def project(self, embedded):
        """`text_projection`: linear_fc2(silu(linear_fc1(x))). Tiny, so it stays on host."""
        return F.linear(F.silu(F.linear(embedded, *self.fc1)), *self.fc2)

    def text(self, ids):
        """Text ids -> projected embeddings [1, n, width]."""
        return self.project(self.text_table[torch.as_tensor(ids, dtype=torch.long)]).reshape(1, -1, self.width)

    def codec(self, ids):
        """Codec ids -> embeddings [1, n, width], no projection."""
        return self.codec_table[torch.as_tensor(ids, dtype=torch.long)].reshape(1, -1, self.width)

    def frames(self, codes):
        """Codes [16] or [16, T] -> one summed embedding per frame, [1, T, width].

        Codebook 0 reads the talker's table, codebooks 1 to 15 the predictor's own, indexed
        per group. Both the decode loop's next prompt position and a reference clip's codec
        track are this sum; neither adds the `tts_pad` here.
        """
        codes = torch.as_tensor(codes, dtype=torch.long).reshape(len(self.predictor_tables) + 1, -1)
        total = self.codec_table[codes[0]]
        for index, table in enumerate(self.predictor_tables):
            total = total + table[codes[index + 1]]
        return total.reshape(1, -1, self.width)


class CloneReference:
    """What a reference clip contributes to a prompt: its codes, its voice, its transcript.

    `codes` [16, T] from the codec encoder, `speaker_embedding` [1, width] from the speaker
    encoder, `text` the transcript. Upstream calls this a `VoiceClonePromptItem` and builds
    it once per clip, separately from generation, because neither encoder is needed again
    once the prompt exists.

    The codes are checked rather than reshaped into place. Upstream's own layout is [T, 16]
    and this directory's is [16, T], and reshaping one into the other reinterprets the
    memory instead of transposing it, which would scramble a voice silently.
    """

    def __init__(self, codes, speaker_embedding, text=None):
        # In-context cloning carries the transcript beside the codes, so it is required there.
        # A reference with no codes is the `x_vector_only` kind, which has nothing to describe.
        if codes is not None and not str(text or "").strip():
            raise ValueError("a clone reference needs the clip's transcript; upstream calls this ICL mode")

        if codes is not None:
            groups = checkpoint.talker_config()["code_predictor_config"]["num_code_groups"]
            codes = torch.as_tensor(codes, dtype=torch.long)
            if codes.dim() != 2 or codes.shape[0] != groups:
                raise ValueError(f"codes must be [{groups}, frames], got {tuple(codes.shape)}")

        self.codes = codes  # None for a reference built with `x_vector_only`
        self.speaker_embedding = torch.as_tensor(speaker_embedding, dtype=torch.float32).reshape(1, -1)
        self.text = text

    @property
    def frames(self):
        """Reference frames, or 0 for a reference that carries only the voice."""
        return 0 if self.codes is None else self.codes.shape[1]

    def __repr__(self):
        if self.codes is None:
            return "CloneReference(voice only, no codes)"
        return f"CloneReference({self.frames} frames, {self.frames / 12.5:.2f} s, text={self.text!r})"


def build_clone_reference(device, audio, text=None, sample_rate=None, x_vector_only=False):
    """A reference clip -> `CloneReference`, running the encoders it needs on device.

    `x_vector_only=True` skips the codec encoder and needs no transcript. Call this before
    a pipeline captures its traces, or release them first: eager work beside a live trace
    hangs the card.
    """
    audio = torch.as_tensor(audio, dtype=torch.float32).reshape(-1)
    expected = checkpoint.codec_config()["input_sample_rate"]
    if sample_rate is not None and int(sample_rate) != expected:
        raise ValueError(f"a reference clip must be {expected} Hz audio, got {sample_rate}")
    if not x_vector_only and not str(text or "").strip():
        raise ValueError(
            "in-context cloning needs the clip's transcript, since the prompt carries it "
            "beside the codes; pass x_vector_only=True to clone from the voice alone"
        )

    embedding = run_speaker_encoder(device, host_audio.speaker_mel(audio))
    if x_vector_only:
        return CloneReference(None, embedding, text)

    encoder = TtCodecEncoder(device, preprocess_codec_encoder_parameters(device))
    return CloneReference(encoder.encode(audio)[0], embedding, text)


def resolve_language(language, speaker=None):
    """Language name -> codec id, with upstream's dialect override. `None` means no tag.

    Two CustomVoice speakers are dialect speakers, `eric` (sichuan) and `dylan` (beijing).
    Asking either for Chinese, or for `Auto`, gets the dialect's id rather than the plain
    language's, and for them `Auto` stops meaning "no tag" at all. Base defines no speakers,
    so this only ever fires on CustomVoice.
    """
    language_id = frontend.language_id(language)
    if speaker is None:
        return language_id

    dialect = checkpoint.talker_config()["spk_is_dialect"].get(str(speaker).strip().lower())
    if dialect and str(language).strip().lower() in ("chinese", AUTO_LANGUAGE):
        return frontend.language_ids()[dialect]
    return language_id


def _prompt_head(tables, role_ids, language_id, speaker_embed):
    """The positions every prompt opens with, and the `codec_bos` that follows them.

    Returns (head [1, 9, width], codec_bos [1, 1, width]) when a language and a speaker are
    both present: three text-only role positions, then the think block, the speaker and
    `codec_pad`, each against a text-track pad or `tts_bos`. Upstream holds `codec_bos`
    back here and re-adds it after the text, so this returns it rather than placing it.

    `speaker_embed` is a codec-table row for CustomVoice and the speaker encoder's own
    output for a clone, which is the talker's width at either size. Both occupy one position and neither is projected.
    """
    talker_config = checkpoint.talker_config()

    think = [talker_config["codec_think_id"], talker_config["codec_think_bos_id"]]
    think += [] if language_id is None else [language_id]
    think += [talker_config["codec_think_eos_id"]]
    if language_id is None:
        # No language tag, so no `codec_think_id` either: upstream swaps in `codec_nothink_id`.
        think[0] = talker_config["codec_nothink_id"]

    codec_track = [tables.codec(think)]
    if speaker_embed is not None:
        codec_track.append(speaker_embed.reshape(1, 1, -1))
    codec_track.append(tables.codec([talker_config["codec_pad_id"], talker_config["codec_bos_id"]]))
    codec_track = torch.cat(codec_track, dim=1)

    # Text track opposite it: pads under the think block and the speaker, tts_bos under
    # codec_pad.
    text_track = torch.cat([tables.tts_pad.expand(-1, codec_track.shape[1] - 2, -1), tables.tts_bos], dim=1)
    head = torch.cat([tables.text(role_ids), text_track + codec_track[:, :-1]], dim=1)
    return head, codec_track[:, -1:]


def _prompt_body(tables, text_ids, codec_bos):
    """The text to speak, then `tts_eos`, each against `codec_pad`; then `codec_bos`.

    The tail of every prompt here, and the last thing the model sees before it starts
    producing frames. Returns [1, n_text + 2, width].
    """
    codec_pad = checkpoint.talker_config()["codec_pad_id"]
    spoken = torch.cat([tables.text(text_ids), tables.tts_eos], dim=1)
    body = spoken + tables.codec([codec_pad] * spoken.shape[1])
    return torch.cat([body, tables.tts_pad + codec_bos], dim=1)


class StreamingText:
    """The text track during streaming input: one embedding per frame, on demand.

    The prompt carries the first text token and every frame adds the next, then `tts_eos`
    once, then `tts_pad`. `text` may arrive in pieces, which upstream leaves to the caller;
    each piece is tokenised on its own, so feed whole words.
    """

    def __init__(self, tables, text):
        self.tables = tables
        self.chunks = iter([text] if isinstance(text, str) else text)
        self.ids = []
        self.embeddings = []
        self.closed = False
        self.sent_eos = False
        if not self._ids():
            raise ValueError("streaming input needs at least one token of text to open the prompt")

    def _ids(self):
        """Pull chunks until an id is in hand, or the text is over."""
        while not self.ids and not self.closed:
            chunk = next(self.chunks, None)
            if chunk is None:
                self.closed = True
                break
            self.ids.extend(int(one) for one in frontend.encode(str(chunk)))
        return bool(self.ids)

    def _project(self):
        """Project the waiting ids in one call, as upstream does: token by token shows at 1e-7."""
        projected = self.tables.text(self.ids)
        self.embeddings = [projected[:, index : index + 1] for index in range(projected.shape[1])]
        self.ids = []

    def first(self):
        """The one text token the prompt carries, projected on its own as upstream does."""
        return self.tables.text([self.ids.pop(0)])

    def next(self):
        """The text embedding for the next frame: a token, then `tts_eos`, then pads."""
        if not self.embeddings and self._ids():
            self._project()
        if self.embeddings:
            return self.embeddings.pop(0)
        if not self.sent_eos:
            self.sent_eos = True
            return self.tables.tts_eos
        return self.tables.tts_pad

    def take_ids(self, limit):
        """Up to `limit` ids, unprojected, for a caller that projects them itself.

        The clone prompt projects its text alongside the reference transcript, so it needs ids.
        """
        taken = []
        while len(taken) < limit and self._ids():
            taken.append(self.ids.pop(0))
        return taken

    def take_eos(self):
        """The `tts_eos` closing the text track, marked sent so decode does not repeat it."""
        self.sent_eos = True
        return self.tables.tts_eos

    def push_front(self, embeddings):
        """Put positions back, for a prompt that took more of the text than it needed."""
        self.embeddings[:0] = [embeddings[:, index : index + 1] for index in range(embeddings.shape[1])]

    @property
    def spent(self):
        """True once the text and its `tts_eos` have both gone in."""
        return self.closed and not self.ids and not self.embeddings and self.sent_eos


def speaker_row(speaker, tables):
    """A named speaker's codec-table row, or a refusal naming the ones this checkpoint has."""
    speakers = checkpoint.talker_config()["spk_id"]
    if not speakers:
        raise ValueError("this checkpoint defines no speakers; CustomVoice needs the CustomVoice weights")
    key = str(speaker).strip().lower()
    if key not in speakers:
        raise ValueError(f"unknown speaker {speaker!r}; this checkpoint offers {', '.join(sorted(speakers))}")
    return key, tables.codec([speakers[key]])


def instruction_block(instruction, tables):
    """An instruction as prompt positions, or nothing.

    Text track only, ahead of everything. Refused on the 0.6B checkpoints, which upstream
    silently drops it for.
    """
    if not str(instruction or "").strip():
        return []
    size = str(checkpoint.model_config().get("tts_model_size", ""))
    if size.startswith("0b6"):
        raise ValueError(f"the {size} checkpoints do not take an instruction; upstream disables it for them")
    return [tables.text(frontend.instruction_ids(instruction))]


def build_streaming_prefill(text, speaker, language, tables=None, instruction=None):
    """The CustomVoice prompt with streaming text input: ten positions, whatever the text.

    Returns (embeddings, `StreamingText`). Non-streaming spends `n_text + 11`.
    """
    tables = tables or HostEmbeddings()
    key, speaker_embed = speaker_row(speaker, tables)
    feed = StreamingText(tables, text)
    head, codec_bos = _prompt_head(tables, frontend.role_ids(), resolve_language(language, key), speaker_embed)
    parts = instruction_block(instruction, tables) + [head, feed.first() + codec_bos]
    return torch.cat(parts, dim=1), feed


def build_custom_voice_prefill(text, speaker, language, tables=None, instruction=None):
    """The dual-track prompt for CustomVoice, non-streaming: `n_text + 11` positions.

    `instruction` is upstream's `generate_custom_voice(..., instruct=...)`: the voice stays
    the speaker's and the instruction shapes the delivery.
    """
    tables = tables or HostEmbeddings()
    key, speaker_embed = speaker_row(speaker, tables)

    prompt_ids = frontend.text_ids(text)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]
    head, codec_bos = _prompt_head(tables, prompt_ids[:ROLE_IDS], resolve_language(language, key), speaker_embed)
    parts = instruction_block(instruction, tables) + [head, _prompt_body(tables, text_ids, codec_bos)]
    return torch.cat(parts, dim=1), prompt_ids


def build_x_vector_prefill(text, reference, language="Auto", tables=None, instruction=None):
    """The clone prompt carrying only the voice, upstream's `x_vector_only_mode`.

    The speaker vector sits where a named speaker would and the clip's codes and transcript
    go unused, so no transcript is needed. The README compares it with in-context cloning.
    """
    tables = tables or HostEmbeddings()
    prompt_ids = frontend.text_ids(text)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]
    head, codec_bos = _prompt_head(
        tables, prompt_ids[:ROLE_IDS], resolve_language(language), reference.speaker_embedding
    )
    parts = instruction_block(instruction, tables) + [head, _prompt_body(tables, text_ids, codec_bos)]
    return torch.cat(parts, dim=1), prompt_ids


def build_streaming_x_vector_prefill(text, reference, language="Auto", tables=None, instruction=None):
    """`build_x_vector_prefill` with the text on the per-frame schedule."""
    tables = tables or HostEmbeddings()
    feed = StreamingText(tables, text)
    head, codec_bos = _prompt_head(tables, frontend.role_ids(), resolve_language(language), reference.speaker_embedding)
    parts = instruction_block(instruction, tables) + [head, feed.first() + codec_bos]
    return torch.cat(parts, dim=1), feed


def is_voice_design_checkpoint():
    """Whether the resolved checkpoint is the VoiceDesign release.

    Three releases carry the same architecture and answer to different prompts, and
    `tts_model_type` is how they say which: `base` clones from a clip, `custom_voice` speaks
    as one of nine named speakers, `voice_design` takes a sentence describing a voice.
    """
    return checkpoint.model_config().get("tts_model_type") == "voice_design"


def build_voice_design_prefill(text, instruction, language="Auto", tables=None):
    """The dual-track prompt for VoiceDesign, non-streaming: `n_instruction + n_text + 10`.

    The instruction goes first, whole, on the text track alone, with no role tokens sliced
    off and nothing opposite it. Then the head, which has no speaker position here. An empty
    instruction is allowed and leaves the model to invent a voice; the other two releases are
    refused, since they were never shown one and would speak in an unrelated voice instead.
    """
    if not is_voice_design_checkpoint():
        raise ValueError(
            "an instruction needs the VoiceDesign checkpoint (this one is "
            f"{checkpoint.model_config().get('tts_model_type')!r}); point $QWEN3_TTS_CKPT "
            "or $HF_MODEL at Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        )

    tables = tables or HostEmbeddings()
    prompt_ids = frontend.text_ids(text)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]
    head, codec_bos = _prompt_head(tables, prompt_ids[:ROLE_IDS], resolve_language(language), None)
    parts = instruction_block(instruction, tables) + [head, _prompt_body(tables, text_ids, codec_bos)]
    return torch.cat(parts, dim=1), prompt_ids


def build_streaming_design_prefill(text, instruction, language="Auto", tables=None):
    """The VoiceDesign prompt with streaming text input.

    The instruction still goes in whole: it describes the voice rather than being spoken.
    """
    if not is_voice_design_checkpoint():
        raise ValueError(
            "an instruction needs the VoiceDesign checkpoint (this one is "
            f"{checkpoint.model_config().get('tts_model_type')!r}); point $QWEN3_TTS_CKPT "
            "or $HF_MODEL at Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        )

    tables = tables or HostEmbeddings()
    feed = StreamingText(tables, text)
    head, codec_bos = _prompt_head(tables, frontend.role_ids(), resolve_language(language), None)
    parts = instruction_block(instruction, tables) + [head, feed.first() + codec_bos]
    return torch.cat(parts, dim=1), feed


def build_streaming_clone_prefill(text, reference, language="Auto", tables=None):
    """The clone prompt with streaming text input, which sums the two tracks.

    Upstream's `generate_icl_prompt` with `non_streaming_mode=False` adds the text track to
    the codec track position by position and cuts to the shorter, so a clip longer than the
    text leaves nothing to stream and a longer text leaves the surplus to the feed.
    """
    if reference.codes is None:
        raise ValueError(
            "this reference carries only a speaker vector; pass x_vector_only=True to clone "
            "from it, or rebuild it without x_vector_only for in-context cloning"
        )
    tables = tables or HostEmbeddings()
    feed = StreamingText(tables, text)
    reference_ids = frontend.reference_text_ids(reference.text)[ROLE_IDS:-REFERENCE_TAIL_IDS]
    head, codec_bos = _prompt_head(tables, frontend.role_ids(), resolve_language(language), reference.speaker_embedding)

    codec_track = torch.cat([codec_bos, tables.frames(reference.codes)], dim=1)
    codec_lens = codec_track.shape[1]

    # Transcript and text projected in one call, as upstream does: two calls show at 1e-7.
    ids = list(reference_ids) + feed.take_ids(codec_lens)
    text_track = tables.text(ids)
    if not feed._ids():
        # The text is all in hand, so it closes here. With text still waiting, `tts_eos` must
        # follow the last of it instead, and the feed sends it when that comes.
        text_track = torch.cat([text_track, feed.take_eos()], dim=1)

    if text_track.shape[1] > codec_lens:
        # A text longer than the clip: the surplus goes back rather than being projected again.
        feed.push_front(text_track[:, codec_lens:])
        text_track = text_track[:, :codec_lens]
    elif text_track.shape[1] < codec_lens:
        # A clip longer than the text, the usual way: nothing left to stream, so decode pads.
        text_track = torch.cat([text_track, tables.tts_pad.expand(-1, codec_lens - text_track.shape[1], -1)], dim=1)

    return torch.cat([head, text_track + codec_track], dim=1), feed


def build_voice_clone_prefill(text, reference, language="Auto", tables=None):
    """The dual-track prompt for voice cloning in context, non-streaming.

    The text track carries the reference transcript before the text to speak, then the codec
    track carries the clip frame by frame. The transcript is what makes it in-context.
    """
    if reference.codes is None:
        raise ValueError(
            "this reference carries only a speaker vector; pass x_vector_only=True to clone "
            "from it, or rebuild it without x_vector_only for in-context cloning"
        )
    tables = tables or HostEmbeddings()
    talker_config = checkpoint.talker_config()

    prompt_ids = frontend.text_ids(text)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]
    reference_ids = frontend.reference_text_ids(reference.text)[ROLE_IDS:-REFERENCE_TAIL_IDS]
    head, codec_bos = _prompt_head(
        tables, prompt_ids[:ROLE_IDS], resolve_language(language), reference.speaker_embedding
    )

    # Text track: transcript first, which is the point of ICL, then the text and tts_eos.
    spoken = torch.cat([tables.text(list(reference_ids) + list(text_ids)), tables.tts_eos], dim=1)
    body = spoken + tables.codec([talker_config["codec_pad_id"]] * spoken.shape[1])

    # Codec track: codec_bos, then the clip frame by frame, in place of `_prompt_body`'s tail.
    voice = torch.cat([codec_bos, tables.frames(reference.codes)], dim=1) + tables.tts_pad

    return torch.cat([head, body, voice], dim=1), prompt_ids


class Qwen3TTSPipeline:
    """The three device blocks plus the host loop that drives them.

    The talker's cache is sized on construction, so `max_frames` is fixed for the life of
    the instance; ask for what you need up front. Each `generate` prefills eagerly and
    then captures the two step traces, since the two cannot coexist.
    """

    def __init__(self, device, max_frames=DEFAULT_MAX_FRAMES, seed=None, profile=False):
        self.device = device
        self.max_frames = max_frames
        # Charges each block of the frame loop separately, and syncs the device between them
        # so the charge lands on the block that did the work. The syncs cost a little, so
        # this is off unless a caller asks: see `tests/perf/test_perf.py`.
        self.profile = profile
        self.tables = HostEmbeddings()
        self.talker_config = checkpoint.talker_config()
        self.groups = self.talker_config["code_predictor_config"]["num_code_groups"]
        self.eos = self.talker_config["codec_eos_token_id"]

        # A prompt is n_text + 11 positions, so leave room for the longest text the
        # tokenizer will hand back plus every frame plus the warmup slot past both.
        self.prompt_room = self.talker_config.get("max_position_embeddings", 2048)
        room = min(self.prompt_room, 512) + max_frames + 8
        self.talker = TtTalkerCachedDecoder(device, preprocess_cached_talker_parameters(device), max_seq=room)
        self.predictor = TtCodePredictorCachedDecoder(device, preprocess_cached_predictor_parameters(device))
        self.codec = TtCodecDecoder(device, preprocess_codec_parameters(device))
        self.codec_head = ttnn.from_torch(
            self.tables.codec_head.t().contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # What the talker may not emit: control ids always, end-of-speech for the first frames.
        vocab = self.talker_config["vocab_size"]
        control = [index for index in range(vocab - CONTROL_ID_COUNT, vocab) if index != self.eos]
        self._control_ids = torch.tensor(control, dtype=torch.long)
        self._control_ids_and_eos = torch.tensor(sorted(control + [self.eos]), dtype=torch.long)

        self.generation = checkpoint.generation_config()
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)
        # Filled in by every generate, so a caller can see where the time went. The first
        # utterance at any new prompt length or frame count also compiles its kernels, and
        # that lands inside whichever stage triggered it.
        self.last_timings = {}

    def _capture(self):
        """Warm both decoders, then capture both, in that order.

        A trace cannot compile new programs, so each decoder runs its step once eagerly
        first. Those warmup buffers have to exist before the first trace does, and no
        trace may execute until the last one is captured: capturing the predictor's after
        the talker's had already run hung the device until it was reset.
        """
        self.talker.warmup()
        self.predictor.warmup()
        self.talker.capture()
        self.predictor.capture()

    def _suppressed(self, emitted):
        """Codebook-0 ids this step may not draw: control ids always, end-of-speech until
        `MIN_FRAMES` frames exist. Both are upstream's.
        """
        if emitted >= MIN_FRAMES:
            return self._control_ids
        return self._control_ids_and_eos

    def _pick(self, logits, seen=(), penalty=1.0, emitted=MIN_FRAMES):
        """One id from a row of logits: the checkpoint's sampler, or argmax if told to."""
        row = logits.detach().float().reshape(-1)
        suppress = self._suppressed(emitted)
        if not self.generation.get("do_sample", True):
            masked = row.clone()
            masked[suppress] = -float("inf")
            return int(masked.argmax())
        return sampling.sample(
            row,
            seen=seen,
            temperature=self.generation.get("temperature", 1.0),
            top_k=self.generation.get("top_k", 0),
            top_p=self.generation.get("top_p", 1.0),
            penalty=penalty,
            generator=self.generator,
            suppress=suppress,
        )

    def _inner_pick(self):
        """The code predictor's sampler, which reads the `subtalker_*` settings."""
        if not self.generation.get("subtalker_dosample", self.generation.get("do_sample", True)):
            return None  # the predictor defaults to argmax
        return lambda row: sampling.sample(
            row,
            temperature=self.generation.get("subtalker_temperature", 1.0),
            top_k=self.generation.get("subtalker_top_k", 0),
            top_p=self.generation.get("subtalker_top_p", 1.0),
            generator=self.generator,
        )

    def _frame_embedding(self, frame, feed=None):
        """The 16 codebooks of one frame, summed against the text track's next position.

        `tts_pad` without a feed, the next text embedding with one: that is the whole of what
        streaming input changes during decode.
        """
        text = self.tables.tts_pad if feed is None else feed.next()
        return self.tables.frames(frame) + text

    def reseed(self, seed=None):
        """Restart the sampler from `seed`, or make it unseeded again.

        Sampling draws from one generator for the whole pipeline, so a caller that wants
        each utterance reproducible on its own has to reset it between them.
        """
        self.generator = None if seed is None else torch.Generator().manual_seed(int(seed))

    def release(self):
        """Drop both captured traces, so eager device work is safe again.

        Anything untraced hangs the card while a trace is live, `build_clone_reference`
        included, so call this before building a second reference clip in a process that
        has already generated something. `generate` and `generate_clone` release on their
        own, since each starts with an eager prefill.
        """
        self.talker.release()
        self.predictor.release()

    def _decode_waveform(self, codes):
        """Codes [1, 16, T] -> waveform, making room for the decode's programs first.

        A length the decoder has not compiled holds L1_SMALL scratch until the program cache is
        dropped, so this drops it, which costs about 0.1 s. Both traces come down first: a trace
        built from dropped programs hangs.
        """
        frames = codes.shape[-1]
        if self.codec.program_room_needed(frames):
            self.release()
            self.device.clear_program_cache()
            self.codec.forget_programs()
            self.last_timings["program_cache_cleared"] = True

        started = time.time()
        waveform = self.codec.decode(codes)
        self.last_timings["codec_s"] = time.time() - started
        self.last_timings["codec_frames"] = frames
        # What the decoder ran, padding included, which is what its time is proportional to.
        self.last_timings["codec_padded_frames"] = self.codec.padded_frames(frames)
        return waveform

    def _decode_frames(self, embeddings, limit, on_frame=None, feed=None):
        """Prefill the prompt, then sample frames until end of speech. Returns codes [T, 16].

        A prefill is eager work, and eager work beside a live trace is what hangs the
        device, so every utterance releases both traces, prefills, then captures again.
        That costs about two seconds and buys back the frame loop.
        """
        prompt = embeddings.shape[1]
        padded = -(-prompt // PROMPT_BUCKET) * PROMPT_BUCKET
        if padded + limit + 1 > self.talker.max_seq:
            raise ValueError(
                f"prompt of {prompt} plus {limit} frames exceeds the cache's {self.talker.max_seq}; "
                "build the pipeline with a larger max_frames"
            )
        if padded > prompt:
            filler = self.tables.tts_pad.expand(1, padded - prompt, -1)
            embeddings = torch.cat([embeddings, filler], dim=1)

        self.release()
        self.talker.reset()
        started = time.time()
        hidden = self.talker.prefill(embeddings)
        last = ttnn.slice(hidden, [0, prompt - 1, 0], [1, prompt, hidden.shape[2]])
        ttnn.deallocate(hidden)
        timings = {"prompt": prompt, "padded_prompt": padded, "prefill_s": time.time() - started}

        started = time.time()
        self._capture()
        timings["capture_s"] = time.time() - started

        inner = self._inner_pick()
        penalty = self.generation.get("repetition_penalty", 1.0)
        frames, seen = [], []
        watch = Stopwatch(self.device, strict=self.profile)
        started = time.time()
        for step in range(limit):
            logits = ttnn.linear(last, self.codec_head)
            row = ttnn.to_torch(logits).float().reshape(-1)
            ttnn.deallocate(logits)  # released before the next trace runs, or it aliases trace memory
            watch.split("codec_head")
            first = self._pick(row, seen=seen, penalty=penalty, emitted=len(frames))
            watch.split("sample")
            if first == self.eos:
                break
            seen.append(first)

            rest = self.predictor.generate(
                ttnn.to_torch(last).float().reshape(1, 1, -1), first, pick=inner, watch=watch
            )
            frame = [first] + list(rest)
            frames.append(frame)
            if on_frame is not None:
                on_frame(step, frame)
            watch.restart()
            embedding = self._frame_embedding(frame, feed)
            watch.split("embed")
            last = self.talker.step(embedding, prompt + step)
            watch.split("talker")

        timings["decode_s"] = time.time() - started
        timings["blocks"] = watch.totals
        timings["frames"] = len(frames)
        timings["ms_per_frame"] = 1000 * timings["decode_s"] / max(len(frames), 1)
        self.last_timings = timings

        if not frames:
            raise RuntimeError("the talker emitted end-of-speech before any frame")
        return torch.tensor(frames, dtype=torch.long)

    def codes(
        self,
        text,
        speaker="ryan",
        language="English",
        max_frames=None,
        on_frame=None,
        streaming=False,
        instruct=None,
    ):
        """The frames for `text` without decoding them, [frames, 16].

        `generate` is this plus the codec, which is the one block that compiles per length.
        """
        limit = min(max_frames or self.max_frames, self.max_frames)
        if streaming:
            embeddings, feed = build_streaming_prefill(text, speaker, language, self.tables, instruct)
        else:
            embeddings = build_custom_voice_prefill(text, speaker, language, self.tables, instruct)[0]
            feed = None
        return self._decode_frames(embeddings, limit, on_frame, feed=feed)

    def generate(
        self,
        text,
        speaker="ryan",
        language="English",
        max_frames=None,
        on_frame=None,
        streaming=False,
        instruct=None,
    ):
        """text -> (waveform [1, N] at 24 kHz, codes [frames, 16]).

        Sampled with the checkpoint's own settings; pass `seed` to the constructor to repeat a
        run. `instruct` directs the named speaker, `streaming=True` feeds the text a token per
        frame and then accepts an iterable of pieces.
        """
        codes = self.codes(text, speaker, language, max_frames, on_frame, streaming, instruct)
        waveform = self._decode_waveform(codes.t().unsqueeze(0)).reshape(1, -1)
        return waveform, codes

    def generate_design(self, text, instruction, language="Auto", max_frames=None, on_frame=None, streaming=False):
        """text spoken in a voice described in words -> (waveform, codes). VoiceDesign only.

        An empty instruction is allowed and leaves the model to invent a voice.
        """
        limit = min(max_frames or self.max_frames, self.max_frames)
        if streaming:
            embeddings, feed = build_streaming_design_prefill(text, instruction, language, self.tables)
        else:
            embeddings, feed = build_voice_design_prefill(text, instruction, language, self.tables)[0], None
        codes = self._decode_frames(embeddings, limit, on_frame, feed=feed)
        waveform = self._decode_waveform(codes.t().unsqueeze(0)).reshape(1, -1)
        return waveform, codes

    def generate_clone(
        self,
        text,
        reference,
        language="Auto",
        max_frames=None,
        on_frame=None,
        streaming=False,
        x_vector_only=False,
        instruct=None,
    ):
        """text spoken in a reference clip's voice -> (waveform, codes). Base only.

        In context by default: the clip's frames decode with the generated ones and are cut off
        the front, which is upstream's arrangement and not cosmetic, since the codec decoder is
        causal and reads them as context. `x_vector_only=True` clones from the voice alone, with
        no transcript and nothing to cut.
        """
        limit = min(max_frames or self.max_frames, self.max_frames)
        if x_vector_only and streaming:
            embeddings, feed = build_streaming_x_vector_prefill(text, reference, language, self.tables, instruct)
        elif x_vector_only:
            embeddings = build_x_vector_prefill(text, reference, language, self.tables, instruct)[0]
            feed = None
        elif streaming:
            embeddings, feed = build_streaming_clone_prefill(text, reference, language, self.tables)
        else:
            embeddings = build_voice_clone_prefill(text, reference, language, self.tables)[0]
            feed = None
        codes = self._decode_frames(embeddings, limit, on_frame, feed=feed)

        if x_vector_only:
            # No reference frames in the prompt, so none to decode alongside and none to cut.
            waveform = self._decode_waveform(codes.t().unsqueeze(0)).reshape(1, -1)
            return waveform, codes

        together = torch.cat([reference.codes.t(), codes], dim=0)
        waveform = self._decode_waveform(together.t().unsqueeze(0)).reshape(1, -1)
        return waveform[:, reference.frames * self.codec.upsample :], codes
