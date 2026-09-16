# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Qwen3-TTS on device: text in, waveform out, in a named voice or a cloned one.

Joins the device blocks with the two host pieces that were missing: the dual-track prompt
assembly and the autoregressive loop that drives them.

Two entry points, and which one works depends on the checkpoint:

    generate(text, speaker=...)          CustomVoice: one of nine named speakers
    generate_clone(text, reference)      Base: a voice taken from a reference clip

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

    # The text itself, then tts_eos, each against codec_pad; then codec_bos against tts_pad.
    body = torch.cat([tables.text(text_ids), tables.tts_eos], dim=1) + tables.codec(
        [talker_config["codec_pad_id"]] * (len(text_ids) + 1)
    )
    tail = tables.tts_pad + codec_bos

    return torch.cat([head, body, tail], dim=1), prompt_ids


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

    In streaming mode the two tracks are summed position by position instead, and whatever
    text is left over becomes the `trailing_text_hidden` fed in during decode. This
    directory implements the non-streaming regime only; see the module docstring.
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
    # codec_pad.
    spoken = torch.cat([tables.text(list(reference_ids) + list(text_ids)), tables.tts_eos], dim=1)
    body = spoken + tables.codec([talker_config["codec_pad_id"]] * spoken.shape[1])

    # Codec track: codec_bos, then the reference clip frame by frame, against tts_pad.
    voice = torch.cat([codec_bos, tables.frames(reference.codes)], dim=1) + tables.tts_pad

    return torch.cat([head, body, voice], dim=1), prompt_ids


class Qwen3TTSPipeline:
    """The three device blocks plus the host loop that drives them.

    The talker's cache is sized on construction, so `max_frames` is fixed for the life of
    the instance; ask for what you need up front. Each `generate` prefills eagerly and
    then captures the two step traces, since the two cannot coexist.
    """

    def __init__(self, device, max_frames=DEFAULT_MAX_FRAMES, seed=None):
        self.device = device
        self.max_frames = max_frames
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

    def _frame_embedding(self, frame):
        """The 16 codebooks of one frame, summed against `tts_pad`: the next prompt position."""
        return self.tables.frames(frame) + self.tables.tts_pad

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

    def _decode_frames(self, embeddings, limit, on_frame=None):
        """Prefill the prompt, then sample frames until end of speech. Returns codes [T, 16].

        A prefill is eager work, and eager work beside a live trace is what hangs the
        device, so every utterance releases both traces, prefills, then captures again.
        That costs about two seconds and buys back the frame loop.
        """
        prompt = embeddings.shape[1]
        if prompt + limit + 1 > self.talker.max_seq:
            raise ValueError(
                f"prompt of {prompt} plus {limit} frames exceeds the cache's {self.talker.max_seq}; "
                "build the pipeline with a larger max_frames"
            )

        self.release()
        self.talker.reset()
        started = time.time()
        hidden = self.talker.prefill(embeddings)
        last = ttnn.slice(hidden, [0, prompt - 1, 0], [1, prompt, hidden.shape[2]])
        ttnn.deallocate(hidden)
        timings = {"prompt": prompt, "prefill_s": time.time() - started}

        started = time.time()
        self._capture()
        timings["capture_s"] = time.time() - started

        inner = self._inner_pick()
        penalty = self.generation.get("repetition_penalty", 1.0)
        frames, seen = [], []
        started = time.time()
        for step in range(limit):
            logits = ttnn.linear(last, self.codec_head)
            row = ttnn.to_torch(logits).float().reshape(-1)
            ttnn.deallocate(logits)  # released before the next trace runs, or it aliases trace memory
            first = self._pick(row, seen=seen, penalty=penalty)
            if first == self.eos:
                break
            seen.append(first)

            rest = self.predictor.generate(ttnn.to_torch(last).float().reshape(1, 1, -1), first, pick=inner)
            frame = [first] + list(rest)
            frames.append(frame)
            if on_frame is not None:
                on_frame(step, frame)
            last = self.talker.step(self._frame_embedding(frame), prompt + step)

        timings["decode_s"] = time.time() - started
        timings["frames"] = len(frames)
        timings["ms_per_frame"] = 1000 * timings["decode_s"] / max(len(frames), 1)
        self.last_timings = timings

        if not frames:
            raise RuntimeError("the talker emitted end-of-speech before any frame")
        return torch.tensor(frames, dtype=torch.long)

    def generate(self, text, speaker="ryan", language="English", max_frames=None, on_frame=None):
        """text -> (waveform [1, N] at 24 kHz, codes [frames, 16]).

        Sampled with the checkpoint's own settings. Pass `seed` to the constructor for a
        reproducible run; the same seed and text give the same waveform.
        """
        limit = min(max_frames or self.max_frames, self.max_frames)
        embeddings, _ = build_custom_voice_prefill(text, speaker, language, self.tables)
        codes = self._decode_frames(embeddings, limit, on_frame)
        started = time.time()
        waveform = self.codec.decode(codes.t().unsqueeze(0)).reshape(1, -1)
        self.last_timings["codec_s"] = time.time() - started
        return waveform, codes

    def generate_clone(self, text, reference, language="Auto", max_frames=None, on_frame=None):
        """text spoken in a reference clip's voice -> (waveform [1, N], codes [frames, 16]).

        `reference` comes from `build_clone_reference`, which needs the **Base** checkpoint:
        CustomVoice ships the nine speakers and no speaker encoder, Base the other way round.

        The reference frames are decoded together with the generated ones and then cut off
        the front of the waveform. That is upstream's arrangement and it is not cosmetic:
        the codec decoder is causal, so the first generated frames read the reference's
        codes as context, and decoding them alone gives a different and worse onset. Each
        frame is exactly 1920 samples, so the cut is exact.
        """
        limit = min(max_frames or self.max_frames, self.max_frames)
        embeddings, _ = build_voice_clone_prefill(text, reference, language, self.tables)
        codes = self._decode_frames(embeddings, limit, on_frame)

        together = torch.cat([reference.codes.t(), codes], dim=0)
        started = time.time()
        waveform = self.codec.decode(together.t().unsqueeze(0)).reshape(1, -1)
        self.last_timings["codec_s"] = time.time() - started
        self.last_timings["codec_frames"] = together.shape[0]
        return waveform[:, reference.frames * self.codec.upsample :], codes
