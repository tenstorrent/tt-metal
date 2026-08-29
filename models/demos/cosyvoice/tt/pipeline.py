# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The whole model, wired: text -> semantic tokens -> mel -> waveform.

    TtTransformerLM        text tokens          -> speech tokens
    TtMaskedDiffWithXvec   speech tokens        -> mel
    TtHiFTGenerator        mel                  -> waveform

All three stages run on device; nothing between them returns to the host except
the sampled token IDs, which RAS needs on the host in Stage 1.

**The four modes are prompt construction, not different networks.** Every one of
them runs the same three stages with the same weights; what changes is which parts
of the prefix are populated:

| mode | prompt text | prompt audio | speaker vector |
|---|---|---|---|
| `sft` | -- | -- | from the speaker table |
| `zero_shot` | transcript of the prompt audio | yes | from the prompt audio |
| `cross_lingual` | -- | yes | from the prompt audio |
| `instruct` | the instruction, as a *style description* | -- | from the speaker table |

`instruct` is worth a warning. `frontend_instruct` places the instruction in the
LLM's **`prompt_text` slot**, concatenated in front of the sentence -- so the model
reads it the way it reads any prefix. CosyVoice-1 wants a character or style
*description* ("A cheerful young woman"); CosyVoice-2's `instruct2` directive
phrasing ("Speak cheerfully") makes this model **read the instruction aloud**. That
mistake took zh CER from 9.09% to 42.42% in the WER sweep.

**Randomness is injected, never drawn here.** Three places in this model sample:
the CFM's initial noise `z`, and the vocoder's `phase_vec` and `noise` in `SineGen`
-- the latter two unconditionally, with no `if self.training` guard. Reproducing
the reference bit-for-bit therefore requires its captured draws; generating fresh
audio requires fresh ones. `RandomSources` makes the choice explicit rather than
hiding a `torch.randn` inside a forward pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn

from .flow.model import TtMaskedDiffWithXvec
from .hifigan.generator import TtHiFTGenerator, shape_trace
from .llm.model import TtTransformerLM

MODES = ("sft", "zero_shot", "cross_lingual", "instruct")


@dataclass
class RandomSources:
    """The three stochastic draws in a CosyVoice forward pass.

    Leave a field None to draw it fresh; supply a captured array to reproduce a
    reference run exactly. Seeding is not an alternative -- a device RNG cannot be
    aligned with torch's stream, which is the rule the vocoder's source module and
    the CFM solver both already follow.
    """

    cfm_z: torch.Tensor | None = None
    sine_phase: torch.Tensor | None = None
    sine_noise: torch.Tensor | None = None

    def z_for(self, frames: int, channels: int = 80) -> torch.Tensor:
        return self.cfm_z if self.cfm_z is not None else torch.randn(1, frames, channels)

    def phase_vec_for(self, harmonic_num: int) -> torch.Tensor:
        """`[1, 1, harmonic_num+1]`, broadcast over time by `TtSineGen`.

        Matches `SineGen.forward` exactly: `Uniform(-pi, pi)` per harmonic, with
        the fundamental (index 0) forced to 0. `phase_vec=None` at the TTNN call
        site is a *deterministic* zero offset, not a fresh draw -- see
        `TtHiFTGenerator.inference`'s docstring -- so a genuinely fresh utterance
        has to draw this itself, exactly as `z_for` does for the CFM.
        """
        if self.sine_phase is not None:
            return self.sine_phase
        v = (torch.rand(1, 1, harmonic_num + 1) * 2.0 - 1.0) * math.pi
        v[:, :, 0] = 0.0
        return v

    def sine_noise_kwargs_for(self, audio_len: int, harmonic_num: int) -> dict:
        """One of `sine_noise=` (a captured, already-scaled draw) or
        `sine_noise_unit=` (a fresh, unscaled standard normal) -- never both.

        The reference's noise amplitude depends on `uv` (voiced/unvoiced), which
        only exists once f0 has been computed on device -- so a fresh draw must
        be *unscaled* and let `TtSineGen` apply that amplitude itself. Passing a
        pre-scaled fresh draw as `sine_noise` would double-apply nothing wrong
        arithmetically, but would silently skip the uv-dependent shaping that
        makes unvoiced frames noisier than voiced ones.
        """
        if self.sine_noise is not None:
            return {"sine_noise": self.sine_noise}
        return {"sine_noise_unit": torch.randn(1, audio_len, harmonic_num + 1)}


def _phase_draw(harmonic_num: int, generator: torch.Generator) -> torch.Tensor:
    """`RandomSources.phase_vec_for`'s draw, but from an explicit generator.

    Streaming needs one reproducible phase vector for a whole utterance, drawn
    before any chunk exists; `RandomSources` draws afresh on every call, which would
    give each chunk a different oscillator offset.
    """
    v = (torch.rand(1, 1, harmonic_num + 1, generator=generator) * 2.0 - 1.0) * math.pi
    v[:, :, 0] = 0.0
    return v


@dataclass
class StreamResult:
    """What one interleaved run produced, and when.

    `first_audio_s` is the whole point: wall-clock seconds from the call to the
    first waveform chunk. Compare it against `total_s` -- the batch path's first
    audio arrives at `total_s`, because there is nothing to hand out before the last
    token has been generated and vocoded.
    """

    chunks: list
    tokens: list[int]
    first_audio_s: float | None
    total_s: float
    n_chunks: int

    def free(self):
        for c in self.chunks:
            ttnn.deallocate(c)
        self.chunks = []


@dataclass
class PromptContext:
    """Everything the modes vary, in one place.

    Fields are host tensors; the pipeline uploads them. `text_tokens` already has
    the prompt text (or the instruction) prepended, because that is what the LLM's
    `prompt_text` slot means -- see the note on `instruct` above.

    **LLM and flow each get their own prompt-token and speaker-embedding
    fields, not one shared pair.** The real frontend splits both in ways a
    single field cannot represent:

    * `frontend_cross_lingual` deletes `llm_prompt_speech_token` (the LLM sees
      only the target text) but keeps `flow_prompt_speech_token` (the flow
      decoder still conditions on the prompt audio).
    * `frontend_instruct` deletes `llm_embedding` entirely -- the LLM runs with
      no speaker x-vector at all, "to avoid information leakage" per the
      reference's own comment -- but keeps `flow_embedding`.

    Every other mode happens to set both members of each pair to the same
    tensor, but the split has to exist for `cross_lingual` and `instruct` to be
    representable at all. `TtTransformerLM.build_prefix` already documents the
    corresponding behaviour: "the speaker row is omitted entirely when there is
    no x-vector."
    """

    text_tokens: torch.Tensor  # [1, T_text] int
    n_text: int  # text length excluding the prompt, for the length bounds
    flow_embedding: torch.Tensor  # [1, 1, 192] speaker x-vector; always present
    llm_embedding: torch.Tensor | None = None  # [1, 1, 192]; None in instruct mode
    llm_prompt_speech_tokens: torch.Tensor | None = None  # [1, T_prompt] int, LLM's prefix
    flow_prompt_speech_tokens: torch.Tensor | None = None  # [1, T_prompt] int, flow's prompt
    prompt_feat: torch.Tensor | None = None  # [1, mel_len1, 80]

    @property
    def mel_len1(self) -> int:
        return 0 if self.prompt_feat is None else int(self.prompt_feat.shape[1])

    @property
    def n_prompt_tokens(self) -> int:
        """The flow decoder's prompt-token count -- what feeds `flow.inference`'s
        length arg. Not necessarily the LLM's; see the field-split note above."""
        return 0 if self.flow_prompt_speech_tokens is None else int(self.flow_prompt_speech_tokens.shape[1])

    @classmethod
    def from_npz(cls, path: str) -> tuple["PromptContext", dict]:
        """Load one of `scripts/prepare_inputs.py`'s per-`(mode, lang)` `.npz`
        files and build the `PromptContext` it describes.

        Which keys are present decides the shape of the result -- there is no
        per-mode branch here, because the frontend itself already expresses each
        mode as "start from one dict, add or delete keys" (see `frontend_*` in
        `cosyvoice/cli/frontend.py`). `text_tokens` gets `prompt_text` prepended
        when one exists, exactly as `PromptContext`'s own docstring requires;
        `text_len` is the *tts* text length the frontend recorded before any such
        concatenation, so it is `n_text` unmodified.

        Returns `(ctx, meta)` -- `meta` carries `mode`/`lang`/`text`/`checkpoint`
        for the caller to report, not anything the model consumes.
        """
        import json

        import numpy as np

        from .common import as_torch

        with np.load(path) as data:
            meta = json.loads(bytes(data["__meta__"]).decode())

            def opt(key):
                return as_torch(data[key]) if key in data.files else None

            def emb(key):
                t = opt(key)
                return t.reshape(1, 1, -1) if t is not None else None

            text_tok = opt("text")
            prompt_text_tok = opt("prompt_text")
            text_tokens = torch.cat([prompt_text_tok, text_tok], dim=1) if prompt_text_tok is not None else text_tok

            ctx = cls(
                text_tokens=text_tokens,
                n_text=int(data["text_len"].flatten()[0]),
                flow_embedding=emb("flow_embedding"),
                llm_embedding=emb("llm_embedding"),
                llm_prompt_speech_tokens=opt("llm_prompt_speech_token"),
                flow_prompt_speech_tokens=opt("flow_prompt_speech_token"),
                prompt_feat=opt("prompt_speech_feat"),
            )
        return ctx, meta


class CosyVoiceTTNN:
    """The three stages behind one call."""

    def __init__(self, device, llm_bag, flow_bag, hift_bag, dtype=ttnn.bfloat16):
        self.device, self.dtype = device, dtype
        self.llm = TtTransformerLM(device, llm_bag, llm_bag.meta, dtype)
        self.flow = TtMaskedDiffWithXvec(device, flow_bag, flow_bag.meta, dtype)
        self.hift = TtHiFTGenerator(device, hift_bag, dtype)
        self.input_frame_rate = flow_bag.meta.get("input_frame_rate", 50)
        self.sample_rate = hift_bag.meta.get("sampling_rate", 22050)

    # ----------------------------------------------------------------------
    def _dev(self, v, dtype=None, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(v, dtype=dtype or self.dtype, layout=layout, device=self.device)

    def _ids(self, v):
        return ttnn.from_torch(v.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    # ----------------------------------------------------------------------
    def text_to_tokens(self, ctx: PromptContext, *, sampler="ras", max_tokens=None, seed=None) -> list[int]:
        """Stage 1: the LLM. Returns the generated semantic tokens only -- the
        prompt's tokens are prefix, not output.

        `ctx.llm_embedding is None` in `instruct` mode -- the reference deletes it
        to avoid leaking the speaker table's identity into a style instruction --
        and `build_prefix` already omits the speaker row entirely for that case.
        """
        spk = self.llm.speaker_embedding(self._dev(ctx.llm_embedding)) if ctx.llm_embedding is not None else None
        has_llm_prompt = ctx.llm_prompt_speech_tokens is not None and ctx.llm_prompt_speech_tokens.shape[1] > 0
        prompt = self._ids(ctx.llm_prompt_speech_tokens) if has_llm_prompt else None
        tokens = self.llm.generate(
            self._ids(ctx.text_tokens),
            spk_emb=spk,
            prompt_speech_tokens=prompt,
            text_len=ctx.n_text,
            sampler=sampler,
            max_tokens=max_tokens,
            seed=seed,
        )
        if prompt is not None:
            ttnn.deallocate(prompt)
        return tokens

    def text_to_tokens_batch(self, ctxs: list[PromptContext], **kw) -> list[list[int]]:
        """Stage 1 for several utterances in one decode loop -- see
        `TtTransformerLM.generate_batch` for what batching buys and what it costs."""
        requests, owned = [], []
        for ctx in ctxs:
            spk = self.llm.speaker_embedding(self._dev(ctx.llm_embedding)) if ctx.llm_embedding is not None else None
            has_prompt = ctx.llm_prompt_speech_tokens is not None and ctx.llm_prompt_speech_tokens.shape[1] > 0
            prompt = self._ids(ctx.llm_prompt_speech_tokens) if has_prompt else None
            if prompt is not None:
                owned.append(prompt)
            requests.append(
                {
                    "text_tokens": self._ids(ctx.text_tokens),
                    "spk_emb": spk,
                    "prompt_speech_tokens": prompt,
                    "text_len": ctx.n_text,
                }
            )
        try:
            return self.llm.generate_batch(requests, **kw)
        finally:
            for t in owned:
                ttnn.deallocate(t)

    def synthesize_batch(self, ctxs: list[PromptContext], rngs=None, **kw):
        """Several utterances end to end. Returns `[(waveform, tokens), ...]`.

        **Only Stage 1 is batched, and that is a measurement, not a shortcut.** The
        LLM runs once per *token* and is 76 % of an utterance at default settings;
        the flow decoder and the vocoder each run once per *utterance* and are 21 %
        and 4 %. Batching the LLM therefore reaches almost all of the available win,
        while batching the other two would mean padding every utterance in the batch
        to the longest mel -- the flow decoder's cost is linear in mel length, so the
        padding is paid in full and the batch does not amortise anything the single
        call was not already amortising. The numbers behind that split are in
        `PERF.md`; the split itself is why this method looks asymmetric.
        """
        rngs = rngs or [RandomSources() for _ in ctxs]
        token_lists = self.text_to_tokens_batch(ctxs, **kw)
        out = []
        for ctx, rng, tokens in zip(ctxs, rngs, token_lists):
            mel, mel_len2 = self.tokens_to_mel(tokens, ctx, rng)
            wav = self.mel_to_wav(mel, mel_len2, rng)
            ttnn.deallocate(mel)
            out.append((wav, tokens))
        return out

    def tokens_to_mel(self, tokens: list[int], ctx: PromptContext, rng: RandomSources):
        """Stage 2: the flow decoder. Returns a device `[1, mel_len2, 80]`."""
        prompt_tokens = ctx.flow_prompt_speech_tokens
        all_tokens = (
            torch.cat([prompt_tokens, torch.tensor(tokens, dtype=torch.int32).reshape(1, -1)], dim=1)
            if prompt_tokens is not None
            else torch.tensor(tokens, dtype=torch.int32).reshape(1, -1)
        )
        mel_len1 = ctx.mel_len1
        mel_len2 = TtMaskedDiffWithXvec.mel_len_for(len(tokens), self.input_frame_rate, self.sample_rate)
        prompt_feat = ctx.prompt_feat if ctx.prompt_feat is not None else torch.zeros(1, 0, self.flow.output_size)
        z = rng.z_for(mel_len1 + mel_len2, self.flow.output_size)
        return (
            self.flow.inference(
                self._ids(all_tokens),
                ctx.n_prompt_tokens,
                mel_len1,
                mel_len2,
                self._dev(prompt_feat),
                self._dev(ctx.flow_embedding),
                self._dev(z),
            ),
            mel_len2,
        )

    def mel_to_wav(self, mel, mel_frames: int, rng: RandomSources):
        """Stage 3: the vocoder. `TtHiFTGenerator.inference` builds the NSF
        excitation itself, from this `mel`, via the f0 predictor -- so the only
        thing a caller supplies is `SineGen`'s two draws, exactly as
        `tokens_to_mel` uses the same `rng` for the CFM's `z`."""
        harmonic_num = self.hift.m_source.sine_gen.harmonic_num
        audio_len = self.audio_length_for(mel_frames)
        phase_vec = self._dev(rng.phase_vec_for(harmonic_num), dtype=ttnn.float32)
        noise_kw = {
            k: self._dev(v, dtype=ttnn.float32) for k, v in rng.sine_noise_kwargs_for(audio_len, harmonic_num).items()
        }
        wav, _, source = self.hift.inference(mel, mel_frames, phase_vec=phase_vec, **noise_kw)
        ttnn.deallocate(phase_vec)
        for v in noise_kw.values():
            ttnn.deallocate(v)
        ttnn.deallocate(source)
        return wav

    # ----------------------------------------------------------------------
    def synthesize(self, ctx: PromptContext, rng: RandomSources | None = None, **kw):
        """The full chain. Returns `(waveform, tokens)`.

        All three stochastic draws -- the CFM's `z` and the vocoder's `phase_vec`
        / `sine_noise` -- come from `rng`, exactly as the module docstring
        describes. Leave `rng` at its default to draw fresh, non-reproducing
        audio; pass a `RandomSources` with captured fields to reproduce a
        reference run bit-for-bit.
        """
        rng = rng or RandomSources()
        tokens = self.text_to_tokens(ctx, **kw)
        mel, mel_len2 = self.tokens_to_mel(tokens, ctx, rng)
        wav = self.mel_to_wav(mel, mel_len2, rng)
        ttnn.deallocate(mel)
        return wav, tokens

    # ----------------------------------------------------------------------
    # streaming: audio before the last token
    # ----------------------------------------------------------------------
    def _stream_ctx(self, ctx: PromptContext, generator: torch.Generator | None):
        """The `flow_chunk` / `rng_for_chunk` pair `TtStreamingSynthesizer` drives on.

        Both close over `ctx`'s prompt, which is shared by every chunk: the flow
        decoder conditions each chunk on the *same* prompt tokens and prompt mel, and
        only the generated tokens differ. That is what lets a chunk be decoded before
        the utterance is finished -- the conditioning does not depend on tokens that
        do not exist yet.

        Draws are fresh per chunk and sized to the chunk, so a captured
        `RandomSources` cannot be used here: its arrays are sized for one whole
        utterance. Reproducibility comes from `generator` instead.
        """
        prompt_tokens = ctx.flow_prompt_speech_tokens
        mel_len1 = ctx.mel_len1
        prompt_feat = ctx.prompt_feat if ctx.prompt_feat is not None else torch.zeros(1, 0, self.flow.output_size)
        harmonic_num = self.hift.m_source.sine_gen.harmonic_num
        # One phase vector for the whole utterance, not one per chunk. A fresh phase
        # per chunk would restart the excitation oscillator at a different offset at
        # every seam, which is exactly the discontinuity the source cache exists to
        # prevent -- see `TtStreamingSynthesizer`'s module docstring.
        phase = (
            RandomSources().phase_vec_for(harmonic_num) if generator is None else _phase_draw(harmonic_num, generator)
        )

        def flow_chunk(tokens):
            toks = torch.tensor(tokens, dtype=torch.int32).reshape(1, -1)
            if prompt_tokens is not None:
                toks = torch.cat([prompt_tokens, toks], dim=1)
            mel_len2 = TtMaskedDiffWithXvec.mel_len_for(len(tokens), self.input_frame_rate, self.sample_rate)
            z = torch.randn(1, mel_len1 + mel_len2, self.flow.output_size, generator=generator)
            mel = self.flow.inference(
                self._ids(toks),
                ctx.n_prompt_tokens,
                mel_len1,
                mel_len2,
                self._dev(prompt_feat),
                self._dev(ctx.flow_embedding),
                self._dev(z),
            )
            return mel, mel_len2

        def rng_for_chunk(mel_frames):
            audio_len = self.audio_length_for(mel_frames)
            return phase, torch.randn(1, audio_len, harmonic_num + 1, generator=generator)

        from types import SimpleNamespace

        return SimpleNamespace(flow_chunk=flow_chunk), rng_for_chunk

    def synthesize_streaming(
        self,
        ctx: PromptContext,
        *,
        stream_config=None,
        on_chunk=None,
        seed: int | None = None,
        **kw,
    ) -> "StreamResult":
        """The full chain, **interleaved**: waveform chunks start before the LLM stops.

        `synthesize` above runs the three stages strictly in order -- every token,
        then all the mel, then all the audio -- so the first sample of output exists
        only once the last token does. This method runs the same three stages against
        the same weights, but drives the flow decoder and the vocoder from a callback
        on the AR decode loop: as soon as one chunk's worth of tokens has been
        sampled, that chunk is decoded and vocoded, and generation resumes.

        Nothing overlaps in *compute* -- one device, one queue, and a chunk's flow and
        vocoder work does pause token generation while it runs. What changes is
        **time to first audio**, which stops scaling with the length of the utterance
        and becomes a function of the chunk size instead. For a caller playing audio
        out, that is the latency that matters, and `total_s` in the result says what
        the interleaving costs against the batch path.

        `on_chunk(wav, n_samples)` is called for each chunk as it is produced, which
        is where a real player would hand samples to its output device. The chunks
        are also collected and returned; a caller that only wants the callback should
        deallocate them.

        Chunk boundaries are identical to `TtStreamingSynthesizer.synthesize`'s, so
        the content equivalence that test proves carries over unchanged -- see
        `StreamSession`.
        """
        import time

        from .streaming import TtStreamingSynthesizer

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        synth = TtStreamingSynthesizer(self.device, self.flow, self.hift, stream_config, self.dtype)
        stream_ctx, rng_for_chunk = self._stream_ctx(ctx, generator)

        chunks: list = []
        first_audio_s: float | None = None
        t0 = time.perf_counter()

        with synth.session(stream_ctx, rng_for_chunk) as session:

            def emit(wav, n):
                nonlocal first_audio_s
                if first_audio_s is None:
                    first_audio_s = time.perf_counter() - t0
                chunks.append(wav)
                if on_chunk:
                    on_chunk(wav, n)

            def on_token(token, index):
                for wav, n in session.push(token):
                    emit(wav, n)

            tokens = self.text_to_tokens(ctx, on_token=on_token, **kw)
            emit(*session.finish())

        return StreamResult(
            chunks=chunks,
            tokens=tokens,
            first_audio_s=first_audio_s,
            total_s=time.perf_counter() - t0,
            n_chunks=len(chunks),
        )

    # ----------------------------------------------------------------------
    @staticmethod
    def audio_length_for(mel_frames: int, n_fft: int = 16, hop_len: int = 4) -> int:
        """Samples a mel of `mel_frames` becomes, straight off the vocoder's own
        length graph -- so a caller can size the excitation without a device."""
        return shape_trace(mel_frames, n_fft=n_fft, hop_len=hop_len)["audio_length"]

    @staticmethod
    def describe_mode(mode: str) -> dict:
        """What each mode populates. Used by the demo and by the mode tests, so the
        table in the module docstring cannot drift from the code."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        return {
            "sft": {"prompt_text": False, "prompt_audio": False, "speaker_table": True},
            "zero_shot": {"prompt_text": True, "prompt_audio": True, "speaker_table": False},
            "cross_lingual": {"prompt_text": False, "prompt_audio": True, "speaker_table": False},
            # the instruction goes in the prompt_text slot -- see the module docstring
            "instruct": {"prompt_text": True, "prompt_audio": False, "speaker_table": True},
        }[mode]
