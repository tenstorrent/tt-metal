# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC gate for Block 4b: ``pipeline/ace_step_pipeline.py``.

    x_1 [1, T, 64] + context_latents + encoder_hidden_states
      -> 8 x (DiT + Euler)  -> final latents [1, T, 64]
      -> Oobleck decoder    -> waveform      [1, 2, T*1920]

Real weights only, driven by ``golden/pipeline/s<S>`` and ``golden/dit/s<S>``. Skips cleanly if the
converted checkpoint is absent.

**Scope.** The condition encoder is deliberately *not* in the loop here: it has its own gate
(``test_cond_pcc``, PCC 0.9994+) and its ``encoder_hidden_states`` output is step-invariant, so it
is taken from the goldens. What this test adds over ``test_solver_pcc`` is the part nothing else
covers: **the VAE decoding a real denoised trajectory into the golden waveform**, plus the
NSC->NCL / device->host seam between the two blocks. ``test_vae_pcc`` cannot cover that — it runs
random-init self-hosted weights, because ``golden/vae/decoder_state_dict.pt`` is not in the dump.

Two gates, deliberately different:

* **latents** — PCC vs ``final_latents.pt``, target 0.99. Same quantity ``test_solver_pcc``
  gates; repeated here so a pipeline-wiring regression is attributable.
* **waveform** — PCC vs ``audio.pt``, target 0.95. Looser *on purpose*: the decoder applies ~1920x
  upsampling through five stages of transposed convolutions and Snake activations, so bf16/fp32
  differences in the latents are amplified in the time domain. A tight waveform gate would be a
  flaky proxy for latent accuracy rather than an independent check. Sample-domain SNR is also
  reported, since PCC on a waveform is dominated by the loudest passages.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v15.pipeline import AceStepPipeline
from models.experimental.ace_step_v15.tests import block4_reference as B4
from models.experimental.ace_step_v15.tests import dit_reference as R
from models.experimental.ace_step_v15.tt.ttnn_ace_step_common import (
    AceStepDiTConfig,
    to_device,
)
from models.experimental.ace_step_v15.tt.ttnn_ace_step_dit import AceStepTransformer1DModel
from models.experimental.ace_step_v15.tt.ttnn_ace_step_vae import (
    OobleckDecoder,
    prepare_decoder_state_dict,
)

TARGET_PCC_LATENTS = 0.99
TARGET_PCC_AUDIO = 0.99
#: Amplitude gate. PCC is a correlation and is nearly blind to a constant gain, so it alone let a
#: 1.39x-too-loud waveform through at PCC 0.9945 (the missing output normalisation -- see
#: `pipeline.normalize_audio`). These two catch amplitude errors that PCC cannot.
MIN_SNR_DB = 15.0
GAIN_TOLERANCE = 0.02  # best-fit gain must sit within 2% of 1.0
SEQ_LENS = (R.SEQ_LEN_BLOCK,)  # S=128 / 10.24 s


def _real_vae_state_dict() -> dict[str, torch.Tensor]:
    """The **raw** fp32 VAE checkpoint from the diffusers ``vae/`` subtree.

    Deliberately unprepared: ``prepare_decoder_state_dict`` strips the ``decoder.`` prefix *and*
    folds ``weight_norm`` itself, so pre-folding here (e.g. via
    ``ttnn_ace_step_weights.prepare_vae_decoder_weights``) double-prepares and trips its
    "expected 145 folded decoder tensors, got 217" guard.
    """
    from safetensors.torch import load_file

    root = Path(os.environ.get("ACE_STEP_PIPELINE", str(R.PIPELINE_PATH))) / "vae"
    path = root / "diffusion_pytorch_model.safetensors"
    if not path.exists():
        msg = f"{path} not found; set $ACE_STEP_PIPELINE to a diffusers-format ACE-Step 1.5 dir"
        raise FileNotFoundError(msg)
    return {k: v.to(torch.float32) for k, v in load_file(str(path)).items()}


def _snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    noise = (ref - got).pow(2).mean()
    if noise == 0:
        return float("inf")
    return float(10.0 * torch.log10(ref.pow(2).mean() / noise))


def _gain_analysis(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float]:
    """``(least-squares gain g minimising ||ref - g*got||, SNR in dB after applying it)``.

    PCC is a correlation and so is nearly blind to a constant scale factor. If ``g`` is far from
    1.0 while the gain-corrected SNR is much better than the raw SNR, the error is a systematic
    **amplitude** mismatch rather than distortion -- a very different bug from spectral error, and
    one a PCC-only gate would wave through.
    """
    g = float((ref * got).sum() / got.pow(2).sum())
    return g, _snr_db(ref, g * got)


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_pipeline_pcc(device, seq_len):
    try:
        sg = B4.SolverGoldens(seq_len)
        pg = B4.PipelineGoldens(seq_len)
        dit_goldens = R.DitGoldens(seq_len)
        dit_state = R.real_dit_state_dict()
        vae_state = _real_vae_state_dict()
    except (FileNotFoundError, KeyError) as exc:
        pytest.skip(f"Block 4 goldens or converted checkpoint unavailable: {exc}")

    config = AceStepDiTConfig.from_diffusers_config(sg.meta["transformer_config"])
    num_steps = sg.num_steps
    latent_t = seq_len * config.patch_size

    # ------------------------------------------------------------------------ build blocks --
    transformer = AceStepTransformer1DModel(config, mesh_device=device)
    transformer.load_torch_state_dict(dit_state)
    transformer.prepare_rope(seq_len)

    vae = OobleckDecoder(mesh_device=device, dtype=ttnn.float32)
    vae.load_torch_state_dict(prepare_decoder_state_dict(vae_state))

    pipe = AceStepPipeline(cond=None, transformer=transformer, vae=vae)

    def _to_11sc(t):
        return to_device(t.reshape(1, 1, *t.shape[-2:]), device)

    out = pipe.generate(
        context_latents_11TC=_to_11sc(dit_goldens["kw_context_latents"]),
        latents_11TC=_to_11sc(sg.x_at(0)),
        encoder_hidden_states_11LC=_to_11sc(dit_goldens["kw_encoder_hidden_states"]),
        timesteps=sg.timesteps,
        return_step_latents=True,
    )

    # ------------------------------------------------------------------------ comparison --
    _, p = comp_pcc(pg.final_latents, out.final_latents, pcc=0.0)
    latents_pcc = float(p)

    ref_audio = pg.audio
    got_audio = out.audio
    assert got_audio.shape == ref_audio.shape, f"audio {tuple(got_audio.shape)} != {tuple(ref_audio.shape)}"
    _, p = comp_pcc(ref_audio, got_audio, pcc=0.0)
    audio_pcc = float(p)
    snr = _snr_db(ref_audio, got_audio)

    print(f"\n=== pipeline PCC (S={seq_len}, T={latent_t}, {num_steps} steps) ===")
    if out.step_latents:
        for i, lat in enumerate(out.step_latents):
            _, sp = comp_pcc(pg.step_latents(i), lat, pcc=0.0)
            print(f"  step_latents.call{i}   pcc={float(sp):.6f}")
    print(f"  final_latents        pcc={latents_pcc:.6f}   (target {TARGET_PCC_LATENTS})")
    print(f"  audio {tuple(got_audio.shape)}  pcc={audio_pcc:.6f}   (target {TARGET_PCC_AUDIO})")
    gain, snr_gc = _gain_analysis(ref_audio, got_audio)
    print(f"  audio SNR            {snr:.2f} dB")
    print(f"  best-fit gain        {gain:.4f}  -> SNR after gain {snr_gc:.2f} dB")
    print(f"  rms  ref={ref_audio.pow(2).mean().sqrt().item():.5f}  got={got_audio.pow(2).mean().sqrt().item():.5f}")
    print(f"  peak |ref|={ref_audio.abs().max().item():.4f}  peak |got|={got_audio.abs().max().item():.4f}")

    assert latents_pcc >= TARGET_PCC_LATENTS, f"final latents PCC {latents_pcc:.6f} < {TARGET_PCC_LATENTS}"
    assert audio_pcc >= TARGET_PCC_AUDIO, f"audio PCC {audio_pcc:.6f} < {TARGET_PCC_AUDIO}"
    assert snr >= MIN_SNR_DB, f"audio SNR {snr:.2f} dB < {MIN_SNR_DB} dB"
    assert abs(gain - 1.0) <= GAIN_TOLERANCE, (
        f"best-fit gain {gain:.4f} is outside 1.0 +/- {GAIN_TOLERANCE} -- the waveform is "
        f"{1 / gain:.3f}x the reference amplitude. Output normalisation missing or wrong?"
    )
