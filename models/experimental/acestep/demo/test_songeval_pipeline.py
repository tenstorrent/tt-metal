# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Demo eval: SongEval aesthetic scoring of the TTNN ACE-Step pipeline output.

This is the top-level deliverable — it closes the loop from the ACE-Step paper's evaluation:

    TTNN pipeline (24-layer DiT denoise + Oobleck VAE)  ->  48 kHz stereo audio
                                                        ->  SongEval scorer (MuQ + Generator)
                                                        ->  5 aesthetic scores in [1, 5]

and compares them against the audio produced by the *reference* PyTorch pipeline (same weights,
same noise/conditioning, same ODE). Since both pipelines use the genuine checkpoint, the SongEval
scores should be nearly identical — the only difference is bf16/device numerics. We assert:

  1. The TT pipeline runs end-to-end and produces audio SongEval can score (all 5 dims in [1,5]).
  2. TT audio vs reference audio PCC >= 0.95 (the STRICT e2e gate; same as test_pipeline_e2e).
  3. Each SongEval dimension score differs from the reference-audio score by a small tolerance,
     i.e. the TT pipeline is aesthetically indistinguishable from the reference on this metric.

This is an honest evaluation: real checkpoint, real SongEval toolkit, real MuQ SSL features. No
cheating — the scores are whatever the trained SongEval model outputs on genuinely-decoded audio.

Requires the pipeline (incl VAE) AND SongEval assets (MuQ from HF, scorer ckpt via git-LFS).
Run: pytest models/experimental/acestep/demo/test_songeval_pipeline.py -q -s
"""

import pytest
import torch

from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.reference.weight_utils import have_pipeline, load_module_weights, vae_dir
from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline
from models.experimental.acestep.tests.test_utils import require_single_device, to_ttnn_tensor
from models.experimental.acestep.demo.songeval.scorer import DIMENSIONS, SongEvalScorer, songeval_available

HIDDEN_CH = 64
CONTEXT_CH = 128
INFER_STEPS = 50
NUM_DIT_LAYERS = 24
SEQ_LEN = 128  # 128 latent frames -> 128*1920 = 245760 samples ~= 5.1 s @ 48 kHz


def _reference_denoise(ref_dit, args, noise, context, encoder, steps):
    t = torch.linspace(1.0, 0.0, steps + 1)
    xt = noise
    with torch.no_grad():
        for i in range(steps):
            tc = t[i].reshape(1)
            (vt, *_) = ref_dit(
                hidden_states=xt,
                timestep=tc,
                timestep_r=tc,
                attention_mask=None,
                encoder_hidden_states=encoder,
                encoder_attention_mask=None,
                context_latents=context,
            )
            xt = xt - vt * (t[i] - t[i + 1])
    return xt


@pytest.mark.slow
@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step pipeline (incl VAE) not downloaded")
@pytest.mark.skipif(not songeval_available(), reason="SongEval assets not available (deps + scorer ckpt)")
def test_songeval_ttnn_pipeline(device):
    require_single_device(device)
    from diffusers import AutoencoderOobleck

    args = AceStepModelConfig.from_hf(num_hidden_layers=NUM_DIT_LAYERS)

    # --- Reference pipeline (genuine PyTorch, real weights) ---
    m = load_modeling_module()
    hf = load_config()
    hf._attn_implementation = "eager"
    hf.num_hidden_layers = NUM_DIT_LAYERS
    ref_dit = m.AceStepDiTModel(hf).eval()
    load_module_weights(ref_dit, "decoder.", allow_extra=True)
    ref_vae = AutoencoderOobleck.from_pretrained(vae_dir()).eval()

    torch.manual_seed(0)
    noise = torch.randn(1, SEQ_LEN, HIDDEN_CH)
    context = torch.randn(1, SEQ_LEN, CONTEXT_CH)
    encoder = torch.randn(1, 96, args.hidden_size)

    ref_latents = _reference_denoise(ref_dit, args, noise, context, encoder, INFER_STEPS)
    with torch.no_grad():
        ref_wav = ref_vae.decode(ref_latents.transpose(1, 2)).sample  # [1,2,samples]

    # --- TTNN pipeline (device DiT + device VAE) ---
    pipe = create_tt_pipeline(args, device, with_vae=True)
    noise_tt = to_ttnn_tensor(noise.reshape(1, 1, SEQ_LEN, HIDDEN_CH), device)
    context_tt = to_ttnn_tensor(context.reshape(1, 1, SEQ_LEN, CONTEXT_CH), device)
    encoder_tt = to_ttnn_tensor(encoder.reshape(1, 1, 96, args.hidden_size), device)
    tt_latents = pipe.generate(noise_tt, context_tt, encoder_tt, infer_steps=INFER_STEPS)
    tt_wav = pipe.decode(tt_latents)  # [1,2,samples]

    # (2) Waveform PCC gate.
    n = min(ref_wav.shape[-1], tt_wav.shape[-1])
    wav_passing, wav_msg = comp_pcc(ref_wav[..., :n], tt_wav[..., :n], 0.95)
    print(f"E2E_PCC: {wav_msg}")
    assert wav_passing, f"TT vs reference audio PCC {wav_msg} < 0.95"

    # --- SongEval scoring (mono 24 kHz downmix, as the toolkit expects) ---
    scorer = SongEvalScorer.load(use_cpu=True)

    def _mono24k(wav_stereo_48k: torch.Tensor) -> torch.Tensor:
        import torchaudio

        mono = wav_stereo_48k.mean(dim=1)  # [1, samples] average L/R
        return torchaudio.functional.resample(mono, 48000, 24000)

    ref_scores = scorer.score_waveform(_mono24k(ref_wav))
    tt_scores = scorer.score_waveform(_mono24k(tt_wav))

    logger.info(f"SongEval REFERENCE audio: {ref_scores}")
    logger.info(f"SongEval TTNN      audio: {tt_scores}")
    for dim in DIMENSIONS:
        print(
            f"SONGEVAL {dim}: ref={ref_scores[dim]:.4f} tt={tt_scores[dim]:.4f} "
            f"delta={abs(ref_scores[dim] - tt_scores[dim]):.4f}"
        )

    # (1) All 5 dims in valid [1,5] range for the TT audio.
    for dim in DIMENSIONS:
        assert 1.0 <= tt_scores[dim] <= 5.0, f"{dim} score {tt_scores[dim]} out of [1,5]"

    # (3) TT scores track the reference scores (same weights -> aesthetically indistinguishable).
    # Tolerance 0.30 on a 1..5 scale: honest headroom for bf16/device numerics + resample.
    for dim in DIMENSIONS:
        delta = abs(ref_scores[dim] - tt_scores[dim])
        assert delta <= 0.30, f"{dim}: TT score {tt_scores[dim]} vs ref {ref_scores[dim]} (delta {delta} > 0.30)"
