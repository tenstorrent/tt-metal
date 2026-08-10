# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 4b: end-to-end ACE-Step 1.5 turbo pipeline on TTNN.

    text/lyric hidden states (host, from Qwen3)  +  timbre latents
        -> condition encoder (device)        -> encoder_hidden_states [1, enc_L, 2048]
        -> 8 x DiT + Euler   (device)        -> final latents         [1, T, 64]
        -> Oobleck VAE decoder (device)      -> waveform              [1, 2, T*1920]

Stage boundaries, and the one awkward seam
------------------------------------------
The three blocks were built independently and their interfaces differ, so this module owns the
conversions rather than pushing them into the blocks:

* **cond** takes **host** ``torch`` tensors (its inputs come off Qwen3 on the CPU anyway) and
  returns **device** tensors.
* **solver/DiT** is device-resident throughout, ``[1, 1, T, 64]`` NSC.
* **VAE** ``decode()`` takes a **host** ``torch`` tensor in **NCL** ``[1, 64, T]`` because it owns
  its own chunking and tracing over the time axis.

So there is one device -> host -> device hop between the solver and the VAE, plus an NSC -> NCL
transpose. At S=128 that is a 256x64 fp32 round trip (~64 KB) and irrelevant; at S=768 it is still
under 400 KB. Logged as a **known inefficiency, not a correctness issue** -- collapsing it means
teaching ``OobleckDecoder.decode`` to accept a device tensor, which is a perf-phase change and
would invalidate its existing trace shapes. See ACE_STEP_1_5_BRINGUP.md §5b.

What is deliberately NOT here (deferred, see master doc)
-------------------------------------------------------
* **Qwen3-Embedding-0.6B and the tokenizer** stay on the host and are *inputs* to this pipeline.
  Bringing them onto device is a separate block.
* **``context_latents``** is taken as an argument. For ``task_type="text2music"`` it is
  ``cat([src_latents(64), chunk_masks(64)], -1)`` where both halves are deterministic, but the
  general construction (repaint / extend / audio2audio) is Phase 2, so this module does not
  synthesise it.
* **The VAE encoder** (reference-audio path) is likewise Phase 2; ``timbre_latents`` arrives
  pre-computed.

BATCH-1 ASSUMPTION (for now): every stage here asserts batch 1. Future work will not be batch 1;
the shapes are written ``[1, ...]`` throughout so the assertions fail loudly rather than silently
mis-broadcasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import ttnn
from models.experimental.ace_step_v15.tt.ttnn_ace_step_common import to_host
from models.experimental.ace_step_v15.tt.ttnn_ace_step_solver import (
    TURBO_NUM_STEPS,
    TURBO_SHIFT,
    denoise,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

LATENT_CHANNELS = 64
#: 48000 Hz / 25 Hz latent rate. One latent frame decodes to this many samples per channel.
SAMPLES_PER_LATENT_FRAME = 1920
#: Reference output loudness: peak normalised to -1 dBFS. ``10 ** (-1/20)``.
TARGET_PEAK_DBFS = -1.0


def normalize_audio(audio: torch.Tensor, *, target_db: float = TARGET_PEAK_DBFS) -> torch.Tensor:
    """The reference pipeline's two-stage output normalisation. **Not optional.**

    ``diffusers`` ``pipeline_ace_step.py`` step 11 does, per batch item:

    1. **anti-clip** -- if ``peak > 1``, divide by the peak;
    2. **loudness** -- rescale so ``peak == 10 ** (target_db / 20)``, i.e. -1 dBFS ~= 0.8913.

    Stage 2 is what makes ``golden/pipeline/*/audio.pt`` have peak exactly 0.8913. Skipping this
    leaves the raw decoder output, which for our S=128 case peaks at **1.2217** -- 1.39x too loud
    and hard-clipping when written to a wav.

    Worth knowing how this hides from a PCC gate: PCC is a correlation and so is nearly blind to a
    constant gain. The un-normalised waveform still scored **0.9945** against the golden while
    being 39% too loud; only SNR (7.99 dB, vs 19.62 dB after gain correction) and the peak/RMS
    ratios exposed it. Amplitude errors need an amplitude-sensitive metric.
    """
    if audio.dtype != torch.float32:
        audio = audio.float()
    peak = audio.abs().amax(dim=[1, 2], keepdim=True)
    if torch.any(peak > 1.0):
        audio = audio / peak.clamp(min=1.0)
    target_amp = 10.0 ** (target_db / 20.0)
    peak = audio.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-6)
    return audio * (target_amp / peak)


@dataclass
class PipelineOutput:
    """Everything a caller or a PCC gate could want from one generation."""

    audio: torch.Tensor
    """``[1, 2, T * 1920]`` fp32 waveform at 48 kHz."""
    final_latents: torch.Tensor
    """``[1, T, 64]`` host NSC latents at t=0 -- the VAE decoder's input."""
    encoder_hidden_states: torch.Tensor | None = None
    """``[1, enc_L, 2048]`` host copy of the condition-encoder output, when requested."""
    step_latents: list[torch.Tensor] | None = None
    """Per-step host latents, when requested. Index ``i`` is the latent *after* step ``i``."""


class AceStepPipeline:
    """Wires the three device blocks into one generation call.

    The blocks are passed in already constructed and weight-loaded. This class deliberately does
    no weight loading of its own: each block owns a different checkpoint subtree and its own
    ``_prepare_torch_state`` quirks, and duplicating that here would be a second place to keep in
    sync with the converter.

    Args:
        cond: ``TTNNAceStepConditionEncoder``, or ``None`` to supply ``encoder_hidden_states``
            directly to :meth:`generate` (used by the PCC gate to isolate Block 4).
        transformer: ``AceStepTransformer1DModel`` with ``prepare_rope(seq_len)`` already called.
        vae: ``OobleckDecoder``.
    """

    def __init__(self, *, cond, transformer, vae) -> None:
        self.cond = cond
        self.transformer = transformer
        self.vae = vae

    # ---------------------------------------------------------------- stages --
    def encode_conditions(
        self,
        text_hidden_states: torch.Tensor,
        lyric_hidden_states: torch.Tensor,
        timbre_latents: torch.Tensor,
        *,
        text_attention_mask: torch.Tensor | None = None,
        lyric_attention_mask: torch.Tensor | None = None,
        refer_audio_order_mask: torch.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Host Qwen3 outputs -> device ``encoder_hidden_states``."""
        if self.cond is None:
            msg = "pipeline was built without a condition encoder; pass encoder_hidden_states"
            raise ValueError(msg)
        enc, _mask = self.cond(
            text_hidden_states,
            lyric_hidden_states,
            timbre_latents,
            text_attention_mask=text_attention_mask,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_order_mask=refer_audio_order_mask,
        )
        return enc

    def decode_latents(
        self, final_latents_11TC: ttnn.Tensor, *, normalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Device NSC ``[1, 1, T, 64]`` -> ``(audio [1, 2, T*1920], host latents [1, T, 64])``.

        This is the device -> host -> device seam documented in the module docstring.

        ``normalize=True`` (the default, and what matches the reference) applies
        :func:`normalize_audio`. Pass ``False`` only to inspect the raw decoder output -- it peaks
        above 1.0 and will clip if written to a wav.
        """
        host = to_host(final_latents_11TC)  # [1, 1, T, 64] fp32
        host_1TC = host.reshape(1, *host.shape[-2:])  # [1, T, 64]
        latents_ncl = host_1TC.transpose(1, 2).contiguous()  # [1, 64, T] for the VAE
        audio = self.vae.decode(latents_ncl)
        if normalize:
            audio = normalize_audio(audio)
        return audio, host_1TC

    # ---------------------------------------------------------------- driver --
    def generate(
        self,
        *,
        context_latents_11TC: ttnn.Tensor,
        latents_11TC: ttnn.Tensor,
        encoder_hidden_states_11LC: ttnn.Tensor | None = None,
        text_hidden_states: torch.Tensor | None = None,
        lyric_hidden_states: torch.Tensor | None = None,
        timbre_latents: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        lyric_attention_mask: torch.Tensor | None = None,
        refer_audio_order_mask: torch.Tensor | None = None,
        num_steps: int = TURBO_NUM_STEPS,
        shift: float = TURBO_SHIFT,
        timesteps: Sequence[float] | None = None,
        normalize: bool = True,
        return_encoder_hidden_states: bool = False,
        return_step_latents: bool = False,
    ) -> PipelineOutput:
        """Run condition encoding (if needed), the denoising loop, and VAE decode.

        Either pass ``encoder_hidden_states_11LC`` directly, or the three host condition inputs
        for this method to encode. ``latents_11TC`` is ``x_1`` -- the caller owns noise sampling so
        that seeding stays reproducible and explicit.
        """
        if encoder_hidden_states_11LC is None:
            missing = [
                n
                for n, v in (
                    ("text_hidden_states", text_hidden_states),
                    ("lyric_hidden_states", lyric_hidden_states),
                    ("timbre_latents", timbre_latents),
                )
                if v is None
            ]
            if missing:
                msg = f"need encoder_hidden_states_11LC, or all of: {', '.join(missing)}"
                raise ValueError(msg)
            encoder_hidden_states_11LC = self.encode_conditions(
                text_hidden_states,
                lyric_hidden_states,
                timbre_latents,
                text_attention_mask=text_attention_mask,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_order_mask=refer_audio_order_mask,
            )

        enc_host = to_host(encoder_hidden_states_11LC) if return_encoder_hidden_states else None

        # Cross-attention K/V is step-invariant: 24 layers x 2 projections, hoisted out of the loop.
        cross_kv = self.transformer.precompute_cross_kv(encoder_hidden_states_11LC)

        capture = {} if return_step_latents else None
        final = denoise(
            self.transformer,
            latents_11TC,
            context_latents_11TC,
            cross_kv=cross_kv,
            num_steps=num_steps,
            shift=shift,
            timesteps=timesteps,
            capture=capture,
        )

        audio, final_host = self.decode_latents(final, normalize=normalize)

        steps = None
        if capture is not None:
            steps = [
                capture[k].reshape(1, *capture[k].shape[-2:])
                for k in (f"step_latents.call{i}" for i in range(num_steps))
                if k in capture
            ]

        return PipelineOutput(
            audio=audio,
            final_latents=final_host,
            encoder_hidden_states=(enc_host.reshape(1, *enc_host.shape[-2:]) if enc_host is not None else None),
            step_latents=steps,
        )
