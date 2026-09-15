# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Qwen3-TTS CustomVoice on device: text in, waveform out.

Joins the three device blocks with the two host pieces that were missing: the dual-track
prompt assembly and the autoregressive loop that drives them.

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

**What runs where.** On device: the 28-layer talker, the 5-layer code predictor, the codec
decoder, and the `codec_head` projection. On host: tokenisation, the embedding gathers, the
small `text_projection` MLP over the prompt, sampling, and the quantizer's table lookup.
That is 2.03B of the parameters on device and a few million of lookups on host.

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

import torch
import torch.nn.functional as F

import ttnn
from models.demos.audio.qwen3_tts import frontend, sampling
from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor_decode import (
    TtCodePredictorCachedDecoder,
    preprocess_cached_predictor_parameters,
)
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec import TtCodecDecoder, preprocess_codec_parameters
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker_decode import (
    TtTalkerCachedDecoder,
    preprocess_cached_talker_parameters,
)

# Upstream's fixed slices into the prompt: three leading ids of role, five trailing.
ROLE_IDS = 3
TAIL_IDS = 5

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
    language_id = frontend.language_id(language)

    think = [talker_config["codec_think_id"], talker_config["codec_think_bos_id"]]
    think += [] if language_id is None else [language_id]
    think += [talker_config["codec_think_eos_id"]]

    # Codec track: think block, speaker, then codec_pad and codec_bos.
    codec_track = torch.cat(
        [
            tables.codec(think),
            tables.codec([speakers[key]]),
            tables.codec([talker_config["codec_pad_id"], talker_config["codec_bos_id"]]),
        ],
        dim=1,
    )

    # Text track opposite it: pads under the think block and the speaker, tts_bos under
    # codec_pad. Upstream drops the final codec_bos here and re-adds it after the text.
    text_track = torch.cat([tables.tts_pad.expand(-1, codec_track.shape[1] - 2, -1), tables.tts_bos], dim=1)
    head = torch.cat([tables.text(prompt_ids[:ROLE_IDS]), text_track + codec_track[:, :-1]], dim=1)

    # The text itself, then tts_eos, each against codec_pad; then codec_bos against tts_pad.
    body = torch.cat([tables.text(text_ids), tables.tts_eos], dim=1) + tables.codec(
        [talker_config["codec_pad_id"]] * (len(text_ids) + 1)
    )
    tail = tables.tts_pad + tables.codec([talker_config["codec_bos_id"]])

    return torch.cat([head, body, tail], dim=1), prompt_ids


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
        """The 16 codebooks of one frame, summed, which becomes the next prompt position.

        Codebook 0 reads the talker's table, the rest the predictor's own, indexed per step.
        """
        total = self.tables.codec([frame[0]])
        for index, code in enumerate(frame[1:]):
            total = total + self.tables.predictor_tables[index][code].reshape(1, 1, -1)
        return total + self.tables.tts_pad

    def generate(self, text, speaker="ryan", language="English", max_frames=None, on_frame=None):
        """text -> (waveform [1, N] at 24 kHz, codes [frames, 16]).

        Sampled with the checkpoint's own settings. Pass `seed` to the constructor for a
        reproducible run; the same seed and text give the same waveform.
        """
        limit = min(max_frames or self.max_frames, self.max_frames)
        embeddings, _ = build_custom_voice_prefill(text, speaker, language, self.tables)
        prompt = embeddings.shape[1]
        if prompt + limit + 1 > self.talker.max_seq:
            raise ValueError(
                f"prompt of {prompt} plus {limit} frames exceeds the cache's {self.talker.max_seq}; "
                "build the pipeline with a larger max_frames"
            )

        # A prefill is eager work, and eager work beside a live trace is what hangs the
        # device, so every utterance releases, prefills, then captures again. That costs
        # about two seconds per utterance and buys back the frame loop.
        self.talker.release()
        self.predictor.release()
        self.talker.reset()
        hidden = self.talker.prefill(embeddings)
        last = ttnn.slice(hidden, [0, prompt - 1, 0], [1, prompt, hidden.shape[2]])
        ttnn.deallocate(hidden)
        self._capture()

        inner = self._inner_pick()
        penalty = self.generation.get("repetition_penalty", 1.0)
        frames, seen = [], []
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

        if not frames:
            raise RuntimeError("the talker emitted end-of-speech before any frame")

        codes = torch.tensor(frames, dtype=torch.long)  # [frames, 16]
        waveform = self.codec.decode(codes.t().unsqueeze(0)).reshape(1, -1)
        return waveform, codes
