# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Qwen3-TTS on device: text in, waveform out, in a named voice or a cloned one.

Joins the device blocks with the two host pieces that were missing: the dual-track prompt
assembly and the autoregressive loop that drives them.

Two entry points, and which one works depends on the checkpoint:

    generate(text, speaker=...)          CustomVoice: one of nine named speakers
    generate_clone(text, reference)      Base: a voice taken from a reference clip
    generate_design(text, instruction)   VoiceDesign: a voice described in a sentence

`build_clone_reference` turns a clip into the `CloneReference` the second takes, running the
codec encoder and the speaker encoder once each.

Per frame the loop does:

    talker(prompt so far)      -> hidden state, then codec_head -> codebook 0
    code predictor(hidden, cb0) -> codebooks 1..15               (15 inner steps)
    sum the 16 codebook embeddings + tts_pad -> one new prompt position

and when codebook 0 comes back as `codec_eos_token_id` the utterance is over and the
collected frames go to the codec decoder.

**The prefill, which is the part that has to be exactly right.** For CustomVoice in
non-streaming mode it is `n_text + 11` positions, each the sum of a text-track and a
codec-track embedding:

    3          role, `<|im_start|>assistant\\n`, text track only
    4          codec think / think_bos / language / think_eos, against tts_pad
    1          the speaker id, against tts_pad
    1          codec_pad, against tts_bos
    n_text + 1 the text tokens then tts_eos, each against codec_pad
    1          codec_bos, against tts_pad

Language and speaker both enter as single codec-vocabulary tokens, which is why neither
appears in the text stream. In non-streaming mode `trailing_text_hidden` is a constant
`tts_pad_embed`, so after the prefill the model gets no further text signal; that is
upstream's design and the regime where its sampler is known to wander.

**Both regimes are here.** `streaming=True` on any of the three entry points switches to
upstream's `non_streaming_mode=False`: the prompt carries one text token, ten positions
whatever the text, and every frame brings the next token in. `StreamingText` has the
schedule and the README the measurements.

A clone prompt is the same head followed by a different body: the reference transcript goes
into the text track ahead of the text to speak, and the clip itself follows on the codec
track, one summed 16-codebook embedding per frame. `build_voice_clone_prefill` has the
layout. It was checked against upstream's own assembly, max absolute difference 0.0 across
all 72 positions.

**What runs where.** On device: the 28-layer talker, the 5-layer code predictor, the codec
decoder, the `codec_head` projection, and for a clone the codec encoder and speaker encoder
too. On host: tokenisation, the embedding gathers, the small `text_projection` MLP over the
prompt, the mel front-end, sampling, and the quantizer's table lookup. That is 2.03B of the
parameters on device and a few million of lookups on host.

**Sampling, not greedy.** The checkpoint ships `do_sample: true` and this follows it, with
its own temperature, top_k and `repetition_penalty`. Greedy is not the conservative
choice: upstream's package, on CPU, greedy, runs past the end of the text and spends the
rest of its budget on a silence code. Measured 699 frames of a 700 frame budget against
414 frames and a clean stop when sampling. `sampling` carries the details.

**Both decoders carry a KV cache and run from a captured trace.** One 28-layer step and
one 5-layer step, replayed per frame, which puts a sentence at roughly 44 ms per frame, or
half of real time. Two rules the traces impose, each of which cost a board reset to learn:
capture every trace before executing any of them, and allocate nothing new once one is
live. Both are why `_capture` warms both decoders before it captures either, and why the
loop below releases the one buffer it allocates per frame.
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

# Prompt lengths the prefill rounds up to. Every distinct length compiles its own programs,
# and that is the whole of the cold prefill: 1.41 s the first time a length is seen against
# 0.02 s once it is compiled. A server sees a new length per sentence, so without this it
# pays that second and a half on nearly every first utterance of a text.
#
# Padding a prompt is safe because attention is causal. The filler positions sit after the
# last real one, so no real position attends to them, and the hidden state the decode starts
# from is sliced at the true last position. Their keys and values do land in the cache, at
# the slots decode is about to write: slot `prompt` is overwritten by the first frame before
# anything reads it, and each later slot the same way. The filler is `tts_pad`, the
# embedding the model already sees at every codec-track position, rather than zeros.
PROMPT_BUCKET = 32


class Stopwatch:
    """Wall clock split across the blocks of one utterance.

    `split(name)` charges everything since the last split to `name`, so the loop reads as a
    sequence of named blocks. Device work is asynchronous, so a block that ends without a
    read would be charged to whichever later block waits for it; `strict` syncs the device
    at every split to stop that. The loop already reads logits per frame, so the unstrict
    numbers are close, and the perf test uses `strict` to be sure.
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
        """Text ids -> projected embeddings [1, n, 2048]."""
        return self.project(self.text_table[torch.as_tensor(ids, dtype=torch.long)]).reshape(1, -1, 2048)

    def codec(self, ids):
        """Codec ids -> embeddings [1, n, 2048], no projection."""
        return self.codec_table[torch.as_tensor(ids, dtype=torch.long)].reshape(1, -1, 2048)

    def frames(self, codes):
        """Codes [16] or [16, T] -> one summed embedding per frame, [1, T, 2048].

        Codebook 0 reads the talker's table, codebooks 1 to 15 the predictor's own, indexed
        per group. Both the decode loop's next prompt position and a reference clip's codec
        track are this sum; neither adds the `tts_pad` here.
        """
        codes = torch.as_tensor(codes, dtype=torch.long).reshape(len(self.predictor_tables) + 1, -1)
        total = self.codec_table[codes[0]]
        for index, table in enumerate(self.predictor_tables):
            total = total + table[codes[index + 1]]
        return total.reshape(1, -1, 2048)


class CloneReference:
    """What a reference clip contributes to a prompt: its codes, its voice, its transcript.

    `codes` [16, T] from the codec encoder, `speaker_embedding` [1, 2048] from the speaker
    encoder, `text` the transcript. Upstream calls this a `VoiceClonePromptItem` and builds
    it once per clip, separately from generation, because neither encoder is needed again
    once the prompt exists.

    The codes are checked rather than reshaped into place. Upstream's own layout is [T, 16]
    and this directory's is [16, T], and reshaping one into the other reinterprets the
    memory instead of transposing it, which would scramble a voice silently.
    """

    def __init__(self, codes, speaker_embedding, text):
        if not str(text or "").strip():
            raise ValueError("a clone reference needs the clip's transcript; upstream calls this ICL mode")

        groups = checkpoint.talker_config()["code_predictor_config"]["num_code_groups"]
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.dim() != 2 or codes.shape[0] != groups:
            raise ValueError(f"codes must be [{groups}, frames], got {tuple(codes.shape)}")

        self.codes = codes
        self.speaker_embedding = torch.as_tensor(speaker_embedding, dtype=torch.float32).reshape(1, -1)
        self.text = text

    @property
    def frames(self):
        return self.codes.shape[1]

    def __repr__(self):
        return f"CloneReference({self.frames} frames, {self.frames / 12.5:.2f} s, text={self.text!r})"


def build_clone_reference(device, audio, text, sample_rate=None):
    """A reference clip -> `CloneReference`, running both encoders once on device.

    The codec encoder turns the clip into codes and the speaker encoder turns it into one
    2048-wide vector; the prompt carries the first on its codec track and the second in a
    single position. Both sets of weights are loaded here and dropped on return, so a
    server that clones one voice many times should keep the result, not call this again.

    **Call this before a pipeline captures its traces**, or release them first: this is
    eager device work, and eager work beside a live trace hangs the card.
    """
    audio = torch.as_tensor(audio, dtype=torch.float32).reshape(-1)
    expected = checkpoint.codec_config()["input_sample_rate"]
    if sample_rate is not None and int(sample_rate) != expected:
        raise ValueError(f"a reference clip must be {expected} Hz audio, got {sample_rate}")

    encoder = TtCodecEncoder(device, preprocess_codec_encoder_parameters(device))
    codes = encoder.encode(audio)[0]
    embedding = run_speaker_encoder(device, host_audio.speaker_mel(audio))
    return CloneReference(codes, embedding, text)


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

    Returns (head [1, 9, 2048], codec_bos [1, 1, 2048]) when a language and a speaker are
    both present: three text-only role positions, then the think block, the speaker and
    `codec_pad`, each against a text-track pad or `tts_bos`. Upstream holds `codec_bos`
    back here and re-adds it after the text, so this returns it rather than placing it.

    `speaker_embed` is a codec-table row for CustomVoice and the speaker encoder's own
    2048-wide output for a clone. Both occupy one position and neither is projected.
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
    producing frames. Returns [1, n_text + 2, 2048].
    """
    codec_pad = checkpoint.talker_config()["codec_pad_id"]
    spoken = torch.cat([tables.text(text_ids), tables.tts_eos], dim=1)
    body = spoken + tables.codec([codec_pad] * spoken.shape[1])
    return torch.cat([body, tables.tts_pad + codec_bos], dim=1)


class StreamingText:
    """The text track during streaming-input decode: one embedding per frame, on demand.

    In this regime the prompt carries only the **first** text token and every generated
    frame adds the next one, against the codes the model just emitted. Upstream calls it
    `non_streaming_mode=False` and its own docstring is careful about what the flag does:
    it "only simulates streaming text input", since it knows the whole string up front and
    changes nothing but when each token reaches the model.

    The schedule is the whole of the regime, so this also takes text that arrives in
    pieces: any chunk that lands before the frame that needs it is indistinguishable from
    having had the string all along. That is the part upstream leaves to the caller.

    **The caveat with pieces is tokenisation.** Each chunk is tokenised on its own, so a
    merge that would have spanned a boundary does not happen and the ids differ from the
    same text given whole. Feed whole words, and prefer whole clauses.

    After the last text token comes `tts_eos`, once, and then `tts_pad` for as long as the
    model keeps going. That ordering is upstream's and it is what tells the model the text
    has ended, so a caller that never closes the iterator never sends it.
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
        """Project whatever ids are waiting, in one call, and keep them by position.

        One call rather than one per token because that is what upstream does, and the
        projection is a matmul whose accumulation order shows: projecting token by token
        left a difference of 1.2e-7 against upstream's tensor, and projecting the group
        leaves none. A chunk boundary is therefore a boundary in the arithmetic too, which
        is the second reason to prefer whole clauses.
        """
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
        """Up to `limit` text ids, unprojected, for a caller that will project them itself.

        The clone prompt sums the two tracks, so its text track has to be projected
        alongside the reference transcript in one call to match upstream bit for bit. That
        means handing the ids over rather than the embeddings. Pulls chunks until it has
        `limit` of them or the text ends, which is what the prompt needs before it can
        close: the clip's frame count fixes how much text the prompt swallows.
        """
        taken = []
        while len(taken) < limit and self._ids():
            taken.append(self.ids.pop(0))
        return taken

    def take_eos(self):
        """The `tts_eos` that closes the text track, and the note that it has gone.

        A prompt that swallows the whole text closes the track itself, and the feed must
        then not synthesise a second `tts_eos` during decode. Getting that wrong left the
        clone prompt's first decode position 0.068 away from upstream's, which is the size
        of an embedding rather than the size of rounding.
        """
        self.sent_eos = True
        return self.tables.tts_eos

    def push_front(self, embeddings):
        """Put positions back, for a prompt that took more of the text than it needed.

        The clone prompt sums the two tracks and cuts to the shorter, which it cannot do
        until it has seen how long the text is. Handing the surplus back beats projecting
        those tokens twice, and keeps `next` the only place positions come from.
        """
        self.embeddings[:0] = [embeddings[:, index : index + 1] for index in range(embeddings.shape[1])]

    @property
    def spent(self):
        """True once the text and its `tts_eos` have both gone in."""
        return self.closed and not self.ids and not self.embeddings and self.sent_eos


def build_streaming_prefill(text, speaker, language, tables=None):
    """The dual-track prompt for streaming text input: ten positions, whatever the text.

    Returns (embeddings [1, 10, 2048], `StreamingText`). The head is the same nine
    positions every prompt opens with, and the tenth is the first text token against
    `codec_bos`. Everything after that arrives during decode, one token per frame, which
    is why the length does not depend on the text: a sentence and a paragraph both prefill
    ten positions. Non-streaming spends `n_text + 11`.

    `text` is a string, or an iterable of pieces. See `StreamingText`.
    """
    tables = tables or HostEmbeddings()
    talker_config = checkpoint.talker_config()
    speakers = talker_config["spk_id"]
    if not speakers:
        raise ValueError("this checkpoint defines no speakers; CustomVoice needs the CustomVoice weights")
    key = str(speaker).strip().lower()
    if key not in speakers:
        raise ValueError(f"unknown speaker {speaker!r}; this checkpoint offers {', '.join(sorted(speakers))}")

    feed = StreamingText(tables, text)
    head, codec_bos = _prompt_head(
        tables, frontend.role_ids(), resolve_language(language, key), tables.codec([speakers[key]])
    )
    return torch.cat([head, feed.first() + codec_bos], dim=1), feed


def build_custom_voice_prefill(text, speaker, language, tables=None):
    """The dual-track prompt for CustomVoice, non-streaming.

    Returns (embeddings [1, P, 2048], prompt_ids) where P is `n_text + 11`.
    """
    tables = tables or HostEmbeddings()
    talker_config = checkpoint.talker_config()
    speakers = talker_config["spk_id"]
    if not speakers:
        raise ValueError("this checkpoint defines no speakers; CustomVoice needs the CustomVoice weights")
    key = str(speaker).strip().lower()
    if key not in speakers:
        raise ValueError(f"unknown speaker {speaker!r}; this checkpoint offers {', '.join(sorted(speakers))}")

    prompt_ids = frontend.text_ids(text)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]
    head, codec_bos = _prompt_head(
        tables, prompt_ids[:ROLE_IDS], resolve_language(language, key), tables.codec([speakers[key]])
    )
    return torch.cat([head, _prompt_body(tables, text_ids, codec_bos)], dim=1), prompt_ids


def is_voice_design_checkpoint():
    """Whether the resolved checkpoint is the VoiceDesign release.

    Three releases carry the same architecture and answer to different prompts, and
    `tts_model_type` is how they say which: `base` clones from a clip, `custom_voice` speaks
    as one of nine named speakers, `voice_design` takes a sentence describing a voice.
    """
    return checkpoint.model_config().get("tts_model_type") == "voice_design"


def build_voice_design_prefill(text, instruction, language="Auto", tables=None):
    """The dual-track prompt for VoiceDesign, non-streaming.

    Returns (embeddings [1, P, 2048], prompt_ids) where P is
    `n_instruction + n_text + 10`, or two less when the language tag is off and the
    instruction is empty.

    A third checkpoint release, `tts_model_type: voice_design`. Its architecture is Base's
    exactly, so every ported block runs on it unchanged; the voice comes from a sentence of
    English rather than a clip or a speaker id.

    The instruction goes **first, whole, and on the text track alone**: no slicing off its
    role tokens the way the text to speak gets sliced, and nothing added from the codec
    track. Then the usual head, which has no speaker position here, then the text.

    An empty instruction is allowed and upstream treats it as no instruction at all, which
    leaves the model to invent a voice.

    Refuses the other two releases. They were never shown an instruction, so they would
    speak in some voice the description had no part in choosing rather than fail.
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

    parts = [head, _prompt_body(tables, text_ids, codec_bos)]
    if str(instruction or "").strip():
        parts.insert(0, tables.text(frontend.instruction_ids(instruction)))
    return torch.cat(parts, dim=1), prompt_ids


def build_streaming_design_prefill(text, instruction, language="Auto", tables=None):
    """The VoiceDesign prompt with streaming text input.

    The instruction still goes in whole, ahead of everything: it describes the voice rather
    than being spoken, so there is nothing to stream about it. Only the text to speak moves
    to the per-frame schedule, which is why this is the CustomVoice change with the
    instruction block in front and no speaker position.

    Returns (embeddings [1, n_instruction + 9, 2048], `StreamingText`).
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

    parts = [head, feed.first() + codec_bos]
    if str(instruction or "").strip():
        parts.insert(0, tables.text(frontend.instruction_ids(instruction)))
    return torch.cat(parts, dim=1), feed


def build_streaming_clone_prefill(text, reference, language="Auto", tables=None):
    """The voice-clone prompt with streaming text input, which is a different shape again.

    Upstream's `generate_icl_prompt` with `non_streaming_mode=False` does not put the two
    tracks one after the other. It **sums them position by position** and cuts to the
    shorter:

        text track   the reference transcript, the text to speak, then `tts_eos`
        codec track  `codec_bos`, then one summed embedding per reference frame

    Whichever runs out first decides. A clip longer than the text (the usual case, since a
    second of audio is 12.5 frames and a second of speech is a few words) pads the text
    track with `tts_pad` and leaves nothing to stream, so the text track is spent inside
    the prompt and decode adds pads: the same arithmetic as non-streaming, over a shorter
    prompt. A text longer than the clip is the interesting case, and there the leftover
    text becomes the per-frame track.

    Returns (embeddings [1, 9 + max(T_text, T_codec), 2048], `StreamingText`), where the
    feed is already positioned at whatever the prompt did not consume.
    """
    tables = tables or HostEmbeddings()
    feed = StreamingText(tables, text)
    reference_ids = frontend.reference_text_ids(reference.text)[ROLE_IDS:-REFERENCE_TAIL_IDS]
    head, codec_bos = _prompt_head(tables, frontend.role_ids(), resolve_language(language), reference.speaker_embedding)

    codec_track = torch.cat([codec_bos, tables.frames(reference.codes)], dim=1)
    codec_lens = codec_track.shape[1]

    # Reference transcript and text to speak projected in one call, as upstream does, then
    # `tts_eos`. One call rather than two because the projection's accumulation order shows
    # at 1e-7 and the point of this comparison is that nothing shows at all.
    ids = list(reference_ids) + feed.take_ids(codec_lens)
    text_track = torch.cat([tables.text(ids), feed.take_eos()], dim=1)

    if text_track.shape[1] > codec_lens:
        # A text longer than the clip. The prompt takes what the codec track covers and the
        # rest is fed per frame, so the surplus goes back to the feed rather than being
        # projected a second time.
        feed.push_front(text_track[:, codec_lens:])
        text_track = text_track[:, :codec_lens]
    elif text_track.shape[1] < codec_lens:
        # A clip longer than the text, which is the usual way round: the text track pads
        # out to the clip and there is nothing left to stream. Decode then adds `tts_pad`
        # at every frame, exactly as non-streaming does, over a shorter prompt.
        text_track = torch.cat([text_track, tables.tts_pad.expand(-1, codec_lens - text_track.shape[1], -1)], dim=1)

    return torch.cat([head, text_track + codec_track], dim=1), feed


def build_voice_clone_prefill(text, reference, language="Auto", tables=None):
    """The dual-track prompt for voice cloning, non-streaming.

    `reference` is a `CloneReference`: the reference clip's codes, its speaker embedding
    and its transcript. Returns (embeddings [1, P, 2048], prompt_ids) where P is
    `9 + (n_ref + n_text + 1) + (1 + T_ref)`.

    Mirrors `Qwen3TTSForConditionalGeneration.generate_icl_prompt` with
    `non_streaming_mode=True`. Two things differ from CustomVoice beyond the speaker
    embedding's source:

      * the text track carries the reference transcript before the text to speak, so the
        model sees what the reference clip said,
      * a codec block follows the text: `codec_bos` and then one summed embedding per
        reference frame, each against `tts_pad`. That block is what actually carries the
        voice.

    `build_streaming_clone_prefill` is the other regime, which sums the two tracks instead
    of placing one after the other.
    """
    tables = tables or HostEmbeddings()
    talker_config = checkpoint.talker_config()

    prompt_ids = frontend.text_ids(text)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]
    reference_ids = frontend.reference_text_ids(reference.text)[ROLE_IDS:-REFERENCE_TAIL_IDS]
    head, codec_bos = _prompt_head(
        tables, prompt_ids[:ROLE_IDS], resolve_language(language), reference.speaker_embedding
    )

    # Text track: the reference transcript, the text to speak, then tts_eos, all against
    # codec_pad. The reference transcript comes first, which is the whole point of ICL.
    spoken = torch.cat([tables.text(list(reference_ids) + list(text_ids)), tables.tts_eos], dim=1)
    body = spoken + tables.codec([talker_config["codec_pad_id"]] * spoken.shape[1])

    # Codec track: codec_bos, then the reference clip frame by frame, against tts_pad. This
    # replaces `_prompt_body`'s single `codec_bos` tail, since the clip rides behind it.
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

    def _pick(self, logits, seen=(), penalty=1.0):
        """One id from a row of logits: the checkpoint's sampler, or argmax if told to."""
        row = logits.detach().float().reshape(-1)
        if not self.generation.get("do_sample", True):
            return int(row.argmax())
        return sampling.sample(
            row,
            seen=seen,
            temperature=self.generation.get("temperature", 1.0),
            top_k=self.generation.get("top_k", 0),
            top_p=self.generation.get("top_p", 1.0),
            penalty=penalty,
            generator=self.generator,
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

        With no `feed` the text track is a constant `tts_pad`, which is what non-streaming
        mode sends after its prompt. With one, it is the next text embedding: streaming
        input differs from non-streaming in exactly this line and in how much of the text
        the prompt carried.
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

        A length the decoder has not compiled yet needs L1_SMALL scratch that the device
        holds until its program cache is dropped, so a process speaking many lengths runs
        the region out. Dropping the cache costs about a tenth of a second and this is the
        one point in an utterance where it is safe: the frames are in, and both traces come
        down anyway before the next prefill.

        Releasing first is not optional. A trace built from programs that have been dropped
        is a hang, and `clear_program_cache` drops every program on the device.
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
            first = self._pick(row, seen=seen, penalty=penalty)
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

    def codes(self, text, speaker="ryan", language="English", max_frames=None, on_frame=None, streaming=False):
        """The frames for `text` without decoding them, [frames, 16].

        `generate` is this plus the codec. Separate because the codec is the one block that
        compiles per length, so anything measuring the frame loop over many utterances
        wants to skip it: `tests/pcc/test_generation_stops.py` does, and a caller feeding
        the codes to something other than this codec can use it too.
        """
        limit = min(max_frames or self.max_frames, self.max_frames)
        if streaming:
            embeddings, feed = build_streaming_prefill(text, speaker, language, self.tables)
        else:
            embeddings, feed = build_custom_voice_prefill(text, speaker, language, self.tables)[0], None
        return self._decode_frames(embeddings, limit, on_frame, feed=feed)

    def generate(self, text, speaker="ryan", language="English", max_frames=None, on_frame=None, streaming=False):
        """text -> (waveform [1, N] at 24 kHz, codes [frames, 16]).

        Sampled with the checkpoint's own settings. Pass `seed` to the constructor for a
        reproducible run; the same seed and text give the same waveform.

        `streaming=True` switches to the other regime this model was trained in: the prompt
        carries one text token and the rest arrives a token per frame, so `text` may be an
        iterable of pieces rather than a string. `StreamingText` has the schedule.
        """
        codes = self.codes(text, speaker, language, max_frames, on_frame, streaming)
        waveform = self._decode_waveform(codes.t().unsqueeze(0)).reshape(1, -1)
        return waveform, codes

    def generate_design(self, text, instruction, language="Auto", max_frames=None, on_frame=None, streaming=False):
        """text spoken in a voice described in words -> (waveform [1, N], codes [frames, 16]).

        Needs the **VoiceDesign** checkpoint, the third release: `tts_model_type` reads
        `voice_design`, and neither Base nor CustomVoice answers to an instruction. Its
        architecture is Base's, so nothing else about the pipeline changes.

        `instruction` is a sentence of English describing the voice, such as "A calm older
        man speaking slowly, with a slight rasp." An empty one is allowed, and leaves the
        model to invent a voice. `build_voice_design_prefill` refuses the other releases.

        `streaming=True` moves the text to the per-frame schedule. The instruction still
        goes in whole: it describes the voice rather than being spoken.
        """
        limit = min(max_frames or self.max_frames, self.max_frames)
        if streaming:
            embeddings, feed = build_streaming_design_prefill(text, instruction, language, self.tables)
        else:
            embeddings, feed = build_voice_design_prefill(text, instruction, language, self.tables)[0], None
        codes = self._decode_frames(embeddings, limit, on_frame, feed=feed)
        waveform = self._decode_waveform(codes.t().unsqueeze(0)).reshape(1, -1)
        return waveform, codes

    def generate_clone(self, text, reference, language="Auto", max_frames=None, on_frame=None, streaming=False):
        """text spoken in a reference clip's voice -> (waveform [1, N], codes [frames, 16]).

        `reference` comes from `build_clone_reference`, which needs the **Base** checkpoint:
        CustomVoice ships the nine speakers and no speaker encoder, Base the other way round.

        The reference frames are decoded together with the generated ones and then cut off
        the front of the waveform. That is upstream's arrangement and it is not cosmetic:
        the codec decoder is causal, so the first generated frames read the reference's
        codes as context, and decoding them alone gives a different and worse onset. Each
        frame is exactly 1920 samples, so the cut is exact.

        `streaming=True` sums the two tracks instead of placing one after the other, which
        `build_streaming_clone_prefill` explains. It shortens the prompt and it does not
        make the clip optional: the prompt still swallows as much text as the clip has
        frames before it can close.
        """
        limit = min(max_frames or self.max_frames, self.max_frames)
        if streaming:
            embeddings, feed = build_streaming_clone_prefill(text, reference, language, self.tables)
        else:
            embeddings, feed = build_voice_clone_prefill(text, reference, language, self.tables)[0], None
        codes = self._decode_frames(embeddings, limit, on_frame, feed=feed)

        together = torch.cat([reference.codes.t(), codes], dim=0)
        waveform = self._decode_waveform(together.t().unsqueeze(0)).reshape(1, -1)
        return waveform[:, reference.frames * self.codec.upsample :], codes
