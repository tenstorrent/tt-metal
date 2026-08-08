# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The whole model, wired: text -> semantic tokens -> mel -> waveform.

    TtTransformerLM        text tokens          -> speech tokens   (P4)
    TtMaskedDiffWithXvec   speech tokens        -> mel             (P3)
    TtHiFTGenerator        mel                  -> waveform        (P2)

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
mistake took zh CER from 9.09% to 42.42% in P0's WER sweep.

**Randomness is injected, never drawn here.** Three places in this model sample:
the CFM's initial noise `z`, and the vocoder's `phase_vec` and `noise` in `SineGen`
-- the latter two unconditionally, with no `if self.training` guard. Reproducing
the reference bit-for-bit therefore requires its captured draws; generating fresh
audio requires fresh ones. `RandomSources` makes the choice explicit rather than
hiding a `torch.randn` inside a forward pass.
"""
from __future__ import annotations

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


@dataclass
class PromptContext:
    """Everything the modes vary, in one place.

    Fields are host tensors; the pipeline uploads them. `text_tokens` already has
    the prompt text (or the instruction) prepended, because that is what the LLM's
    `prompt_text` slot means -- see the note on `instruct` above.
    """

    text_tokens: torch.Tensor  # [1, T_text] int
    n_text: int  # text length excluding the prompt, for the length bounds
    embedding: torch.Tensor  # [1, 1, 192] speaker x-vector
    prompt_speech_tokens: torch.Tensor | None = None  # [1, T_prompt] int
    prompt_feat: torch.Tensor | None = None  # [1, mel_len1, 80]

    @property
    def mel_len1(self) -> int:
        return 0 if self.prompt_feat is None else int(self.prompt_feat.shape[1])

    @property
    def n_prompt_tokens(self) -> int:
        return 0 if self.prompt_speech_tokens is None else int(self.prompt_speech_tokens.shape[1])


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
        prompt's tokens are prefix, not output."""
        spk = self.llm.speaker_embedding(self._dev(ctx.embedding))
        prompt = self._ids(ctx.prompt_speech_tokens) if ctx.n_prompt_tokens else None
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

    def tokens_to_mel(self, tokens: list[int], ctx: PromptContext, rng: RandomSources):
        """Stage 2: the flow decoder. Returns a device `[1, mel_len2, 80]`."""
        prompt_tokens = ctx.prompt_speech_tokens
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
                self._dev(ctx.embedding),
                self._dev(z),
            ),
            mel_len2,
        )

    def mel_to_wav(self, mel, mel_frames: int, f0_source):
        """Stage 3: the vocoder. `f0_source` is the NSF excitation at audio rate."""
        return self.hift.decode(mel, f0_source, mel_frames)

    # ----------------------------------------------------------------------
    def synthesize(self, ctx: PromptContext, f0_source, rng: RandomSources | None = None, **kw):
        """The full chain. Returns `(waveform, tokens)`.

        `f0_source` is supplied rather than computed because the f0 predictor and
        `SourceModuleHnNSF` need `SineGen`'s two injected draws; assembling them is
        the caller's decision, exactly as `RandomSources` documents.
        """
        rng = rng or RandomSources()
        tokens = self.text_to_tokens(ctx, **kw)
        mel, mel_len2 = self.tokens_to_mel(tokens, ctx, rng)
        wav = self.mel_to_wav(mel, mel_len2, f0_source)
        ttnn.deallocate(mel)
        return wav, tokens

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
