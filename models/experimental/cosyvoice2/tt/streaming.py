# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Streaming synthesis: upstream's `CosyVoice2Model.tts(stream=True)` chunk schedule over the flow and HiFT.

Upstream (cosyvoice/cli/model.py at 074ca6dc9e80, :343-374 and `token2wav` :292-326):
- A chunk goes out whenever `hop + 3` tokens past the offset exist. The last 3 are look-ahead context for the flow's
  encoder. The first hop is 25 plus the prompt pad (`ceil(P / 25) * 25 - P` for a P-token prompt), so the first
  chunk ends on a 25-token boundary of prompt + speech, where the flow's chunk-causal mask has one. After each chunk
  the hop doubles, up to 100. Upstream never resets the hop between requests; this port resets it for every utterance
  (notes: D3).
- Each chunk's flow call recomputes the whole prefix (prompt + every token so far) with chunk-causal masks and keeps
  the mel frames from `2 x offset` on. After the LLM ends, the final chunk's flow runs NON-streaming over every token
  (`token2wav` without `stream`: notes D31) and keeps the frames from `2 x offset` on.
- HiFT: each call's mel is the previous call's last 8 frames followed by the new ones; the NSF source's first 3,840
  samples are the previous call's last 3,840 (the sine phase runs on); the first 3,840 output samples are crossfaded
  with the previous call's held-back last 3,840 (a Hamming window of 7,680, as tt/hifigan/chunking.py). A call that
  is not the final one holds its last 8 frames, source and output samples back for the next.

HiFT geometries here (`HiFTStream`): the middle calls run at their exact lengths, `8 + 2 x hop` = 108 and 208 frames.
The first call (`2 x first hop` = 50-98 frames, no cache) is padded to 128 with silence IN FRONT: its last 8 frames
are held back and crossfaded, so they must see upstream's right context (the conv's own zero padding), not silence
frames; front padding is unvoiced, so the sine phase at the first real frame is upstream's. The final call is padded at
the end to 128 or 256 and masked there (`valid_frames`, tt/hifigan/valid_length.py), so it computes upstream's call over
its real frames to the last sample (before 2026-09-30 the padding was silence, and it silenced the last ~25 ms of every
utterance: notes B28).

Offline (stage A, `stream_fixed_tokens`): the tokens are given up front, as if the LLM had finished; no trace is
involved. Live (stage B, `StreamSession`, driven by `CosyVoice2TTNN.synthesize_stream`): the LLM's
`generate(on_token=...)` pushes each token, and a due chunk's flow and HiFT run between two decode steps while the
decode trace is alive, so every streaming geometry must be warmed first (`warmup_streaming`). Both stages run the same
session code. docs/VALIDATION.md, "Streaming", has the gates.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import ttnn

from .hifigan.chunking import HOP, OVERLAP_FRAMES, fade_in_out, speech_window

TOKEN_HOP = 25  # upstream's token_hop_len: the flow's trained chunk size
MAX_TOKEN_HOP = 4 * TOKEN_HOP  # token_max_hop_len
HOP_SCALE = 2  # stream_scale_factor
PRE_LOOKAHEAD = 3  # the flow encoder's pre_lookahead_len
SOURCE_CACHE = OVERLAP_FRAMES * HOP  # 3,840 samples
FIRST_CALL_FRAMES = 128  # the first HiFT call, padded in front
FINAL_CALL_BUCKETS = (128, 256)  # the final HiFT call, padded at the end and masked there


@dataclass(frozen=True)
class StreamChunk:
    offset: int  # generated tokens already emitted before this chunk
    hop: int  # new tokens this chunk emits (the final chunk: every token left)
    final: bool


def prompt_pad(n_prompt: int) -> int:
    """Tokens that bring prompt + first hop to a 25-token boundary: `ceil(P / 25) * 25 - P`."""
    return int(math.ceil(n_prompt / TOKEN_HOP) * TOKEN_HOP - n_prompt)


def stream_schedule(n_tokens: int, n_prompt: int) -> list[StreamChunk]:
    """Upstream's chunks for `n_tokens` generated tokens after an `n_prompt`-token prompt, the hop starting at 25."""
    out, offset, hop, pad = [], 0, TOKEN_HOP, prompt_pad(n_prompt)
    while True:
        this_hop = hop + pad if offset == 0 else hop
        if n_tokens - offset < this_hop + PRE_LOOKAHEAD:
            break
        out.append(StreamChunk(offset, this_hop, False))
        offset += this_hop
        hop = min(MAX_TOKEN_HOP, hop * HOP_SCALE)
    out.append(StreamChunk(offset, n_tokens - offset, True))
    return out


def silence_mel(frames: int, channels: int = 80) -> torch.Tensor:
    from .pipeline import MEL_SILENCE

    return torch.full((1, frames, channels), MEL_SILENCE)


class HiFTStream:
    """HiFT over a stream of mel pieces, with upstream's cache (see the module docstring). Host in, host out; the
    state between calls (8 mel frames, 3,840 source and 3,840 output samples, about 30 KB) stays on the host."""

    def __init__(self, hift, harmonics: int, dtype=ttnn.float32):
        self.hift, self.harmonics, self.dtype = hift, harmonics, dtype
        self.mel_cache = self.source_cache = self.speech_tail = None
        self.window = speech_window()
        self.calls: list[dict] = []  # per call: frames, the padded run length, seconds

    def step(self, new_mel: torch.Tensor, final: bool, noise: torch.Tensor, f0: torch.Tensor | None = None):
        """`new_mel` `[1, F, 80]`: the chunk's new frames. `noise` `[1, (cache + F) x 480, harmonics]` and `f0`
        `[1, cache + F]` (optional, the seam gate injects upstream's) cover this call's whole input, cached frames
        first. Returns the audio this chunk emits (host float32, 1-D)."""
        first = self.mel_cache is None
        mel = new_mel if first else torch.cat([self.mel_cache, new_mel], dim=1)
        frames = int(mel.shape[1])
        assert noise.shape[1] == frames * HOP, (noise.shape, frames)
        if first and not final:
            front, run = FIRST_CALL_FRAMES - frames, FIRST_CALL_FRAMES
            assert front >= 0, f"a first chunk of {frames} frames exceeds {FIRST_CALL_FRAMES}"
        elif final:
            run = next((b for b in FINAL_CALL_BUCKETS if b >= frames), frames)
            front = 0
        else:
            front, run = 0, frames  # 108 or 208, exact
        back = run - front - frames
        assert back == 0 or front == 0, (front, back)  # only the final call is padded at its end, and never in front
        mel_run = torch.cat([silence_mel(front), mel, torch.zeros(1, back, mel.shape[2])], dim=1)  # the end: masked
        noise_run = torch.cat(
            [torch.zeros(1, front * HOP, self.harmonics), noise, torch.zeros(1, back * HOP, self.harmonics)], dim=1
        )
        f0_run = None if f0 is None else torch.cat([torch.zeros(1, front), f0.reshape(1, -1), torch.zeros(1, back)], 1)
        t0 = time.perf_counter()
        mel_dev = ttnn.from_torch(mel_run, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.hift.device)
        # the cache replaces the source's first samples of the REAL frames: only the final call has a cache and
        # padding, and its padding is at the end
        assert front == 0 or self.source_cache is None
        wav_dev, source = self.hift.inference(
            mel_dev,
            run,
            1,
            sine_noise=noise_run,
            f0=f0_run,
            cache_source=self.source_cache,
            return_source=True,
            valid_frames=frames if back else None,
        )
        wav = ttnn.to_torch(wav_dev).float().reshape(-1)[front * HOP : (front + frames) * HOP]
        if back:  # the real call's window normalization on its last samples
            index, gain = self.hift.end_gain(frames, run)
            wav[index] = wav[index] * gain
        ttnn.deallocate(wav_dev)
        ttnn.deallocate(mel_dev)
        source = source.reshape(-1)[front * HOP : (front + frames) * HOP]
        if self.speech_tail is not None:
            wav = fade_in_out(wav, self.speech_tail, self.window)
        self.calls.append({"frames": frames, "run": run, "front": front, "s": time.perf_counter() - t0})
        if final:
            self.mel_cache = self.source_cache = self.speech_tail = None
            return wav.numpy()
        self.mel_cache = mel[:, -OVERLAP_FRAMES:].clone()
        self.source_cache = source[-SOURCE_CACHE:].clone().reshape(1, -1)
        self.speech_tail = wav[-SOURCE_CACHE:].clone()
        return wav[:-SOURCE_CACHE].numpy()


@dataclass
class StreamedChunk:
    chunk: StreamChunk
    mel: torch.Tensor  # the chunk's new frames, [1, 2 x hop, 80]
    audio: np.ndarray  # what the chunk emitted
    timings: dict = field(default_factory=dict)


def flow_chunk(pipeline, ctx, tokens: list[int], chunk: StreamChunk) -> torch.Tensor:
    """The flow for one chunk: its new mel frames, `[1, 2 x chunk.hop, 80]`. A middle chunk runs streaming over the
    prompt and every token up to its look-ahead, at a flow bucket; the final chunk runs the non-streaming flow over
    every token (`tokens_to_mel`), as upstream does."""
    from .pipeline import TOKEN_MEL_RATIO, bucket_for

    if chunk.final:
        mel = pipeline.tokens_to_mel(tokens, ctx)
    else:
        n = chunk.offset + chunk.hop + PRE_LOOKAHEAD
        bucket = bucket_for(ctx.n_prompt_tokens + n, pipeline.config.flow_token_buckets())
        mel = pipeline.flow.inference_streaming(
            torch.tensor([tokens[:n]], dtype=torch.int32),
            ctx.flow_prompt_speech_tokens,
            ctx.prompt_feat,
            ctx.embedding,
            bucket,
            context_len=PRE_LOOKAHEAD,
        )
    return mel[:, TOKEN_MEL_RATIO * chunk.offset : TOKEN_MEL_RATIO * (chunk.offset + chunk.hop)]


class StreamSession:
    """One utterance streamed as its tokens arrive (stage B): `push` each token (it runs a chunk's flow and HiFT when
    one is due), then `finish` once the LLM has ended (the final chunk). The hop restarts at 25 for every session,
    so for every segment (notes: D3). `on_audio(audio)` receives each chunk's audio as soon as it exists; `chunks`
    keeps each one with its times (seconds since `t0`): `ready_s` when its audio was done, plus flow and HiFT."""

    def __init__(self, pipeline, ctx, noise_for, on_audio=None, t0: float | None = None):
        self.pipeline, self.ctx, self.noise_for, self.on_audio = pipeline, ctx, noise_for, on_audio
        self.tokens: list[int] = []
        self.offset, self.hop, self.pad = 0, TOKEN_HOP, prompt_pad(ctx.n_prompt_tokens)
        self.hift = HiFTStream(
            pipeline.hift, pipeline.harmonics, dtype=getattr(ttnn, pipeline.config.hift_source_dtype)
        )
        self.chunks: list[StreamedChunk] = []
        self.t0 = time.perf_counter() if t0 is None else t0

    def push(self, token: int) -> None:
        self.tokens.append(int(token))
        hop = self.hop + (self.pad if self.offset == 0 else 0)
        if len(self.tokens) - self.offset >= hop + PRE_LOOKAHEAD:
            self._emit(StreamChunk(self.offset, hop, False))
            self.offset += hop
            self.hop = min(MAX_TOKEN_HOP, self.hop * HOP_SCALE)

    def finish(self) -> None:
        if len(self.tokens) > self.offset:
            self._emit(StreamChunk(self.offset, len(self.tokens) - self.offset, True))

    def _emit(self, chunk: StreamChunk) -> None:
        device = self.pipeline.device
        clock = getattr(self.pipeline, "_clock", None)  # the pipeline's stage clock times the CFM inside the flow
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        cfm0 = clock.totals.get("flow_cfm", 0.0) if clock is not None else 0.0
        mel = flow_chunk(self.pipeline, self.ctx, self.tokens, chunk)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        cfm = (clock.totals.get("flow_cfm", 0.0) - cfm0) if clock is not None else float("nan")
        k = len(self.chunks)
        frames = mel.shape[1] + (0 if k == 0 else OVERLAP_FRAMES)
        audio = self.hift.step(mel, chunk.final, self.noise_for(k, frames * HOP))
        ttnn.synchronize_device(device)
        t2 = time.perf_counter()
        timings = {"start_s": t0 - self.t0, "flow": t1 - t0, "cfm": cfm, "hift": t2 - t1, "ready_s": t2 - self.t0}
        self.chunks.append(StreamedChunk(chunk, mel, audio, timings))
        if self.on_audio is not None:
            self.on_audio(audio)

    @property
    def audio(self) -> np.ndarray:
        return np.concatenate([c.audio for c in self.chunks]) if self.chunks else np.zeros(0, np.float32)


def stream_fixed_tokens(pipeline, ctx, tokens: list[int], noise_for) -> list[StreamedChunk]:
    """Stage A: a fixed token list through a `StreamSession`, as if the LLM had produced it and ended.
    `noise_for(k, samples)` gives HiFT call k's sine noise, `[1, samples, harmonics]`."""
    session = StreamSession(pipeline, ctx, noise_for)
    for token in tokens:
        session.push(token)
    session.finish()
    return session.chunks
