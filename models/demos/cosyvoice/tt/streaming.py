# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chunked synthesis with cross-chunk continuity.

The bring-up scope names the streaming path explicitly, so the weight-bearing
stages run on device here exactly as they do offline. The
bookkeeping around them -- the fades, the splices, the caches -- runs on device
too, because it is all elementwise work on tensors that are already there and
pulling them back to the host to crossfade would be host residue for no reason.

The algorithm, from `cosyvoice/cli/model.py`:

    hop = 100 tokens, overlap = 20            (2 s and 0.4 s at 50 Hz)
    while at least hop + overlap tokens are available:
        synthesise those hop + overlap tokens
        emit, then drop the first `hop` -- the overlap carries into the next chunk

Three caches keep the chunks from sounding like separate utterances, and each
one exists for a different reason:

**mel overlap** — the last 34 mel frames of a chunk are Hamming-crossfaded with
the first 34 of the next. Chunk boundaries are where the flow decoder's context
changes, so the mel jumps slightly; the fade hides it.

**hift mel cache** — the last 20 mel frames are *prepended* to the next chunk
before vocoding. The vocoder's receptive field is wide (two transposed
convolutions and eight ResBlocks), so a chunk decoded in isolation has wrong
context at its left edge. The prepended frames are then discarded from the
output.

**excitation cache** — the last 5120 samples of the NSF source are spliced into
the front of the next chunk's excitation. This is the important one. The source
is a phase-continuous oscillator, and a fresh chunk restarts its phase
accumulator at zero, so without the splice **every boundary is a phase
discontinuity** -- an audible click, and the thing the f0-into-phase analysis predicts will
be worst. The corresponding speech tail is separately crossfaded with a Hamming
window over the same 5120 samples.

Randomness is injected per chunk, as everywhere else in this port.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import ttnn

TOKEN_RATE = 50
SAMPLE_RATE = 22050
HOP = 256  # mel hop; also the vocoder's total upsample


def hamming(n: int) -> torch.Tensor:
    """`np.hamming(n)` -- symmetric, matching the reference's window exactly.

    numpy's `hamming` is the symmetric form (`0.54 - 0.46 cos(2*pi*i/(n-1))`), not
    the periodic one `scipy.signal.get_window` returns by default. A crossfade
    built from the periodic variant is subtly asymmetric and does not sum to one
    across the seam.
    """
    i = torch.arange(n, dtype=torch.float32)
    return 0.54 - 0.46 * torch.cos(2 * math.pi * i / (n - 1))


@dataclass
class StreamConfig:
    """The reference's constants, derived rather than copied where they are derived."""

    token_hop_len: int = 2 * TOKEN_RATE  # token_min_hop_len
    token_max_hop_len: int = 4 * TOKEN_RATE
    token_overlap_len: int = 20
    mel_cache_len: int = 20
    stream_scale_factor: float = 1.0

    @property
    def mel_overlap_len(self) -> int:
        return int(self.token_overlap_len / TOKEN_RATE * SAMPLE_RATE / HOP)

    @property
    def source_cache_len(self) -> int:
        return self.mel_cache_len * HOP

    def chunk_size(self) -> int:
        return self.token_hop_len + self.token_overlap_len


@dataclass
class StreamState:
    """Everything carried between chunks. All device tensors."""

    mel_overlap: object | None = None  # [1, mel_overlap_len, 80]
    hift_mel: object | None = None  # [1, mel_cache_len, 80]
    hift_source: object | None = None  # [1, source_cache_len, 1]
    hift_speech: object | None = None  # [1, source_cache_len, 1]
    emitted: list = field(default_factory=list)

    def free(self):
        for t in (self.mel_overlap, self.hift_mel, self.hift_source, self.hift_speech):
            if t is not None:
                ttnn.deallocate(t)
        self.mel_overlap = self.hift_mel = self.hift_source = self.hift_speech = None


class TtStreamingSynthesizer:
    """Token chunks in, waveform chunks out. Flow and vocoder both on device."""

    def __init__(self, device, flow, hift, config: StreamConfig | None = None, dtype=ttnn.bfloat16):
        self.device, self.flow, self.hift, self.dtype = device, flow, hift, dtype
        self.cfg = config or StreamConfig()
        # Deliberately does NOT turn on the vocoder's trace cache, even though every
        # middle chunk decodes the same geometry. Capturing a fresh geometry costs about
        # a second against the ~34 ms per chunk a replay saves, so the crossover is near
        # 30 chunks -- a minute of audio. Below that, enabling it makes a stream slower;
        # a 12-chunk stream measured 2.2x worse. Long-form callers can opt in with
        # `hift.enable_trace(True)` or `COSYVOICE_HIFT_TRACE=1`. See
        # TtHiFTGenerator.enable_trace for the numbers.
        n = self.cfg.mel_overlap_len
        # Split windows uploaded once: fade_in_out multiplies the head of the new
        # signal by the window's first half and the tail of the old by its second.
        w = hamming(2 * n)
        self._mel_in = self._dev(w[:n].reshape(1, -1, 1))
        self._mel_out = self._dev(w[n:].reshape(1, -1, 1))
        s = self.cfg.source_cache_len
        ws = hamming(2 * s)
        self._sp_in = self._dev(ws[:s].reshape(1, -1, 1))
        self._sp_out = self._dev(ws[s:].reshape(1, -1, 1))

    def _dev(self, v, dtype=None):
        return ttnn.from_torch(v, dtype=dtype or self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

    # ----------------------------------------------------------------------
    @staticmethod
    def _fade(new, old, w_in, w_out, n):
        """`fade_in_out`: crossfade `new`'s first n frames with `old`'s last n.

        `old` is already exactly n long here (it is a saved tail), so only `new`
        needs slicing. Returns a fresh tensor; the caller owns it.
        """
        head = ttnn.slice(new, [0, 0, 0], [new.shape[0], n, new.shape[2]])
        rest = ttnn.slice(new, [0, n, 0], [new.shape[0], new.shape[1], new.shape[2]])
        mixed = ttnn.add(ttnn.multiply(head, w_in), ttnn.multiply(old, w_out))
        ttnn.deallocate(head)
        out = ttnn.concat([mixed, rest], dim=1)
        ttnn.deallocate(mixed)
        ttnn.deallocate(rest)
        return out

    @staticmethod
    def _tail(x, n):
        return ttnn.slice(x, [0, x.shape[1] - n, 0], [x.shape[0], x.shape[1], x.shape[2]])

    @staticmethod
    def _head(x, n):
        return ttnn.slice(x, [0, 0, 0], [x.shape[0], n, x.shape[2]])

    # ----------------------------------------------------------------------
    def token2wav(self, mel, mel_frames: int, state: StreamState, rng, finalize: bool):
        """One chunk of mel -> one chunk of waveform, updating `state` in place.

        `mel` is consumed. Returns `(waveform, n_samples)`; the caller owns both.
        """
        cfg = self.cfg

        if state.mel_overlap is not None:
            faded = self._fade(mel, state.mel_overlap, self._mel_in, self._mel_out, cfg.mel_overlap_len)
            ttnn.deallocate(mel)
            ttnn.deallocate(state.mel_overlap)
            state.mel_overlap = None
            mel = faded

        # Prepend the vocoder's mel context. Without it the left edge of every
        # chunk is decoded with the wrong receptive-field content.
        if state.hift_mel is not None:
            joined = ttnn.concat([state.hift_mel, mel], dim=1)
            ttnn.deallocate(mel)
            ttnn.deallocate(state.hift_mel)
            state.hift_mel = None
            mel = joined
            mel_frames += cfg.mel_cache_len

        if not finalize:
            # Hold back the last mel_overlap_len frames for the next chunk's fade.
            keep = mel_frames - cfg.mel_overlap_len
            state.mel_overlap = self._tail(mel, cfg.mel_overlap_len)
            trimmed = self._head(mel, keep)
            ttnn.deallocate(mel)
            mel, mel_frames = trimmed, keep

        phase, noise_unit = rng(mel_frames)
        wav, n_samples, source = self.hift.inference(
            mel,
            mel_frames,
            phase_vec=self._dev(phase, ttnn.float32),
            sine_noise_unit=self._dev(noise_unit, ttnn.float32),
            cache_source=state.hift_source,
        )
        if state.hift_source is not None:
            ttnn.deallocate(state.hift_source)
            state.hift_source = None
        # The mel context for the next chunk is the tail of the mel that was just
        # vocoded -- after the overlap trim, not before -- so it has to be taken
        # here, while `mel` is still alive.
        next_mel = None if finalize else self._tail(mel, cfg.mel_cache_len)
        ttnn.deallocate(mel)

        if state.hift_speech is not None:
            faded = self._fade(wav, state.hift_speech, self._sp_in, self._sp_out, cfg.source_cache_len)
            ttnn.deallocate(wav)
            ttnn.deallocate(state.hift_speech)
            state.hift_speech = None
            wav = faded

        if finalize:
            ttnn.deallocate(source)
            return wav, n_samples

        # Carry the tails; emit everything before them. The held-back speech tail is
        # not lost -- it is what the next chunk crossfades into.
        state.hift_mel = next_mel
        state.hift_source = self._tail(source, cfg.source_cache_len)
        ttnn.deallocate(source)
        state.hift_speech = self._tail(wav, cfg.source_cache_len)
        emit = self._head(wav, n_samples - cfg.source_cache_len)
        ttnn.deallocate(wav)
        return emit, n_samples - cfg.source_cache_len

    # ----------------------------------------------------------------------
    def session(self, ctx, rng_for_chunk) -> "StreamSession":
        """An incremental driver: push tokens in, take waveform chunks out.

        This is the form the pipelined path needs -- see `StreamSession`.
        """
        return StreamSession(self, ctx, rng_for_chunk)

    def synthesize(self, tokens, ctx, rng_for_chunk, on_chunk=None):
        """Full streaming run over an already-generated token list.

        Returns the waveform chunks. `ctx` carries the flow's prompt (tokens, mel
        and speaker vector); `rng_for_chunk(mel_frames)` returns
        `(phase, noise_unit)`.

        Implemented by pushing the finished list through `StreamSession` one token
        at a time, so this and the pipelined path share **one** chunk scheduler
        rather than two that have to be kept in agreement. Chunk boundaries are a
        pure function of the token count, so feeding a completed list token by token
        produces exactly the chunks the original loop did -- which is what lets the
        pipelined path inherit `test_device_streamed_matches_non_streamed`'s content
        guarantee instead of re-proving it. `test_incremental_session_cuts_the_same_
        chunks_as_the_batch_loop` checks that equivalence on the host, across
        lengths, rather than once at whatever length the golden happens to be.
        """
        out = []
        with self.session(ctx, rng_for_chunk) as session:
            for token in tokens:
                for wav, n in session.push(token):
                    out.append(wav)
                    if on_chunk:
                        on_chunk(wav, n)
            wav, n = session.finish()
            out.append(wav)
            if on_chunk:
                on_chunk(wav, n)
        return out

    def _one(self, chunk_tokens, ctx, state, rng, finalize):
        """Flow-decode one token chunk, then vocode it with the carried caches."""
        mel, mel_frames = ctx.flow_chunk(chunk_tokens)
        return self.token2wav(mel, mel_frames, state, rng, finalize)


class StreamSession:
    """The chunk scheduler, driven one token at a time.

    **This is the difference between chunked vocoding and streaming.** The batch
    form -- hand it a finished token list and let it cut that list into chunks --
    can only ever measure whether chunked synthesis reconstructs the same audio. It
    cannot start the flow decoder before the LLM has produced its last token,
    because it is given the list only once the list exists.

    Driving the same scheduler from a callback inverts that. `push` is called by the
    AR decode loop as each token is sampled (`TtTransformerLM.generate`'s `on_token`);
    the moment `hop + overlap` tokens have accumulated, the flow decoder and the
    vocoder run on them and hand back audio, and generation resumes. So the first
    waveform chunk exists after roughly `token_hop_len + token_overlap_len` tokens
    rather than after the whole utterance -- **time to first audio stops growing with
    utterance length**, which is the property that makes a TTS pipeline streamable.

    The three stages still run one after another on one device and one command queue.
    There is no overlap of *compute*, and a chunk's flow/vocoder work does pause token
    generation while it runs, so the total gets slightly worse. What changes is when
    the first sample can be handed to a caller, and that is the number the streaming
    requirement is about. `tests/perf/test_streaming_perf.py` measures both.

    The schedule itself is unchanged from the batch loop it replaces, deliberately:
    same `hop`, same growth by `stream_scale_factor`, same overlap, same finalize.
    """

    def __init__(self, synth: TtStreamingSynthesizer, ctx, rng_for_chunk):
        self.synth, self.ctx, self.rng = synth, ctx, rng_for_chunk
        self.state = StreamState()
        self.tokens: list[int] = []
        self.i = 0
        self.hop = synth.cfg.token_hop_len
        self.n_chunks = 0
        self._done = False

    # ------------------------------------------------------------------
    def push(self, token: int):
        """Accept one freshly generated token; yield any chunk it completes.

        A generator rather than an optional return: one token can complete at most
        one chunk, but `extend` may complete several, and a caller written as
        `for wav, n in session.push(t)` works for both.
        """
        self.tokens.append(token)
        yield from self._drain()

    def extend(self, tokens):
        """Accept several tokens at once."""
        self.tokens.extend(tokens)
        yield from self._drain()

    def _drain(self):
        cfg = self.synth.cfg
        while self.i + self.hop + cfg.token_overlap_len <= len(self.tokens):
            chunk = self.tokens[self.i : self.i + self.hop + cfg.token_overlap_len]
            wav, n = self.synth._one(chunk, self.ctx, self.state, self.rng, finalize=False)
            self.i += self.hop
            self.hop = min(cfg.token_max_hop_len, int(self.hop * cfg.stream_scale_factor))
            self.n_chunks += 1
            yield wav, n

    def finish(self):
        """Vocode whatever is left. Call once -- a second call would re-run the tail
        against caches the first call has already consumed."""
        if self._done:
            raise RuntimeError("StreamSession.finish() called twice")
        self._done = True
        wav, n = self.synth._one(self.tokens[self.i :], self.ctx, self.state, self.rng, finalize=True)
        self.n_chunks += 1
        return wav, n

    def close(self):
        """Free whatever the carried caches still hold. Idempotent."""
        self.state.free()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
