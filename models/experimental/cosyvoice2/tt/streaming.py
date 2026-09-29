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
the end to 128 or 256 (the tail effect docs/VALIDATION.md measures for bucketing).

Offline (stage A, `stream_fixed_tokens`): the tokens are given up front, as if the LLM had finished; no trace is
involved. docs/VALIDATION.md, "Streaming", has the gate against upstream's own streaming run.
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
FINAL_CALL_BUCKETS = (128, 256)  # the final HiFT call, padded at the end


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
        mel_run = torch.cat([silence_mel(front), mel, silence_mel(back)], dim=1)
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
        )
        wav = ttnn.to_torch(wav_dev).float().reshape(-1)[front * HOP : (front + frames) * HOP]
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


def stream_fixed_tokens(pipeline, ctx, tokens: list[int], noise_for) -> list[StreamedChunk]:
    """Stage A: the chunks of a fixed token list, flow then HiFT per chunk. `noise_for(k, samples)` gives HiFT call
    k's sine noise, `[1, samples, harmonics]`."""
    hift = HiFTStream(pipeline.hift, pipeline.harmonics, dtype=getattr(ttnn, pipeline.config.hift_source_dtype))
    out = []
    for k, chunk in enumerate(stream_schedule(len(tokens), ctx.n_prompt_tokens)):
        ttnn.synchronize_device(pipeline.device)
        t0 = time.perf_counter()
        mel = flow_chunk(pipeline, ctx, tokens, chunk)
        ttnn.synchronize_device(pipeline.device)
        t1 = time.perf_counter()
        frames = mel.shape[1] + (0 if k == 0 else OVERLAP_FRAMES)
        audio = hift.step(mel, chunk.final, noise_for(k, frames * HOP))
        ttnn.synchronize_device(pipeline.device)
        out.append(StreamedChunk(chunk, mel, audio, {"flow": t1 - t0, "hift": time.perf_counter() - t1}))
    return out
