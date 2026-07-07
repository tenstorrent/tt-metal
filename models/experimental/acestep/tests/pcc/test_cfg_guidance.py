# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: TT CFG denoise (APG guidance) vs the genuine reference CFG sampling loop.

The reference runs the DiT twice per step (conditional context + a learned null_condition_emb) and
combines the two velocities with apg_forward (adaptive projected guidance, the base-model default).
Our pipeline's CFG path (_generate_cfg, pure ttnn apg_forward_ttnn) must match this numerically so
the whole model — WITH guidance — is prompt-faithful.

Reference denoise+APG mirrors modeling_acestep_v15_base.py sample() + apg_guidance.py exactly.
Compares the FINAL audio (DiT CFG denoise -> VAE decode) vs the reference's CFG audio.

PRECISION CEILING (measured, root-caused): under CFG the guidance amplifies the (cond-uncond)
velocity ~6x, and the DiT residual stream develops a massive-activation OUTLIER channel (absmax
~24600, 100x the median) whose bf16 representation error (quant step ~128 at that magnitude)
propagates. This is a DTYPE effect, not an implementation bug: the model's OWN bf16 mode (torch
.to(bfloat16)) only reaches ~0.863 audio PCC vs its fp32 self. So 0.95-vs-fp32 is unachievable for
ANY bf16 model. This test therefore gates TT against the achievable bf16 bar: TT (bf16) must match
the fp32 reference AT LEAST AS WELL AS the model's own bf16 mode does (with margin). TT actually
EXCEEDS native bf16 (~0.888 > 0.863) thanks to HiFi4-attention + fp32 accumulation.
"""

import pytest
import torch
import ttnn

from loguru import logger
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.common.utility_functions import comp_pcc
from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.reference.weight_utils import have_pipeline, load_module_weights, load_state_dict, vae_dir
from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline
from models.experimental.acestep.tt.apg_guidance import MomentumBuffer, apg_forward
from models.experimental.acestep.tests.test_utils import require_single_device, to_ttnn_tensor

NUM_DIT_LAYERS = 24
HIDDEN_CH = 64
CONTEXT_CH = 128
INFER_STEPS = 30
GUIDANCE_SCALE = 7.0  # reference base-model default


def _reference_cfg_denoise(ref_dit, rope, args, noise, context, enc_cond, null_emb, infer_steps, gs, cast=None):
    """Reference CFG ODE denoise: per step DiT(cond) + DiT(null), combine via apg_forward (dims=[1]).

    cast: optional torch dtype to cast the DiT inputs to per call (e.g. torch.bfloat16 to measure the
    model's own bf16 precision floor). Velocities are cast back to fp32 for the APG combine + Euler.
    """
    seq_len = noise.shape[1]
    t_prime = seq_len // args.patch_size
    pos = torch.arange(t_prime).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, t_prime, 128), pos)
    t = torch.linspace(1.0, 0.0, infer_steps + 1)
    enc_uncond = null_emb.reshape(1, 1, -1).expand(1, enc_cond.shape[1], -1).contiguous()
    mb = MomentumBuffer()

    def _c(x):
        return x.to(cast) if cast is not None else x

    xt = noise
    with torch.no_grad():
        for i in range(infer_steps):
            t_curr = t[i].reshape(1)
            (vt_c, *_) = ref_dit(
                hidden_states=_c(xt), timestep=_c(t_curr), timestep_r=_c(t_curr), attention_mask=None,
                encoder_hidden_states=_c(enc_cond), encoder_attention_mask=None, context_latents=_c(context),
            )
            (vt_u, *_) = ref_dit(
                hidden_states=_c(xt), timestep=_c(t_curr), timestep_r=_c(t_curr), attention_mask=None,
                encoder_hidden_states=_c(enc_uncond), encoder_attention_mask=None, context_latents=_c(context),
            )
            vt_c, vt_u = vt_c.float(), vt_u.float()
            vt = apg_forward(vt_c, vt_u, gs, momentum_buffer=mb, dims=[1])
            xt = xt - vt * (t[i] - t[i + 1])
    return xt


@pytest.mark.slow
@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step pipeline (incl VAE) not downloaded")
def test_cfg_guidance_e2e(device):
    require_single_device(device)
    from diffusers import AutoencoderOobleck

    args = AceStepModelConfig.from_hf(num_hidden_layers=NUM_DIT_LAYERS)
    m = load_modeling_module()
    hf = load_config()
    hf._attn_implementation = "eager"
    hf.num_hidden_layers = NUM_DIT_LAYERS
    ref_dit = m.AceStepDiTModel(hf).eval()
    load_module_weights(ref_dit, "decoder.", allow_extra=True)
    ref_vae = AutoencoderOobleck.from_pretrained(vae_dir()).eval()
    rope = Qwen3RotaryEmbedding(hf)

    sd = load_state_dict()
    null_emb = sd.get("null_condition_emb", sd.get("decoder.null_condition_emb")).float()

    seq_len = 128
    torch.manual_seed(0)
    noise = torch.randn(1, seq_len, HIDDEN_CH)
    context = torch.randn(1, seq_len, CONTEXT_CH)
    encoder = torch.randn(1, 96, args.hidden_size)

    # fp32 reference CFG denoise -> VAE decode (the golden target).
    ref_latents = _reference_cfg_denoise(ref_dit, rope, args, noise, context, encoder, null_emb, INFER_STEPS, GUIDANCE_SCALE)
    with torch.no_grad():
        ref_wav = ref_vae.decode(ref_latents.transpose(1, 2)).sample

    # bf16 PRECISION FLOOR: the SAME reference run in the model's own bf16 mode. Best any bf16 impl
    # can do vs the fp32 golden ref (the massive-activation outlier channel makes the bf16<->fp32 gap
    # ~0.86 under CFG 6x amplification). Measured here so the bar is data-derived, not hardcoded.
    ref_dit_bf16 = m.AceStepDiTModel(hf).eval()
    load_module_weights(ref_dit_bf16, "decoder.", allow_extra=True)
    ref_dit_bf16 = ref_dit_bf16.to(torch.bfloat16)
    bf16_latents = _reference_cfg_denoise(
        ref_dit_bf16, rope, args, noise, context, encoder, null_emb, INFER_STEPS, GUIDANCE_SCALE, cast=torch.bfloat16
    )
    with torch.no_grad():
        bf16_wav = ref_vae.decode(bf16_latents.transpose(1, 2)).sample
    nb = min(ref_wav.shape[-1], bf16_wav.shape[-1])
    _, bf16_floor = comp_pcc(ref_wav[..., :nb], bf16_wav[..., :nb], 0.0)
    print(f"CFG_BF16_FLOOR_PCC: {bf16_floor:.6f}")

    # TT CFG pipeline: same noise/context/encoder, null_condition_emb loaded.
    pipe = create_tt_pipeline(args, device, with_vae=True, with_encoders=False)
    pipe._null_condition_emb = null_emb  # inject (with_encoders=False skips the auto-load)
    noise_tt = to_ttnn_tensor(noise.reshape(1, 1, seq_len, HIDDEN_CH), device)
    context_tt = to_ttnn_tensor(context.reshape(1, 1, seq_len, CONTEXT_CH), device)
    enc_cond_tt = to_ttnn_tensor(encoder.reshape(1, 1, 96, args.hidden_size), device)
    enc_uncond_tt = pipe._uncond_context(enc_cond_tt)

    tt_latents = pipe.generate(
        noise_tt, context_tt, enc_cond_tt, infer_steps=INFER_STEPS,
        guidance_scale=GUIDANCE_SCALE, uncond_encoder_hidden_states=enc_uncond_tt,
    )
    tt_wav = pipe.decode(tt_latents)

    n = min(ref_wav.shape[-1], tt_wav.shape[-1])
    _, tt_pcc = comp_pcc(ref_wav[..., :n], tt_wav[..., :n], 0.0)
    logger.info(f"CFG e2e audio: ref={ref_wav.shape[-1]} tt={tt_wav.shape[-1]} gs={GUIDANCE_SCALE}")
    print(f"CFG_E2E_PCC: {tt_pcc:.6f}")

    # Correctness bar: TT (bf16) must match the fp32 reference AT LEAST AS WELL AS the model's own
    # bf16 mode does (minus a small tolerance for TT-vs-torch bf16 rounding differences). Gating
    # against a fixed 0.95-vs-fp32 would gate the bf16<->fp32 DTYPE gap, which no bf16 model can clear
    # (the bf16 floor itself is ~0.86). TT currently EXCEEDS the floor (~0.888 > ~0.863).
    bar = bf16_floor - 0.03
    assert tt_pcc >= bar, (
        f"CFG e2e audio PCC {tt_pcc:.4f} below the bf16 achievable bar {bar:.4f} "
        f"(bf16 floor {bf16_floor:.4f}); TT is worse than the model's own bf16 precision"
    )


@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step pipeline not downloaded")
def test_cfg_guidance_traced_matches_eager(device):
    """The traced CFG denoise loop (generate(use_trace=True) under CFG) must be numerically identical
    to the eager CFG loop. CFG is stateful (the APG momentum running-average), so tracing carries it
    across replays via the tracer input read-back pattern; this test guards that the state plumbing
    stays correct (a plain in-place buffer regresses to ~0.8 PCC). No reference/VAE needed — pure
    TT eager-vs-traced on the DiT latents, so it is fast."""
    require_single_device(device)

    args = AceStepModelConfig.from_hf(num_hidden_layers=NUM_DIT_LAYERS)
    sd = load_state_dict()
    null_emb = sd.get("null_condition_emb", sd.get("decoder.null_condition_emb")).float()

    seq_len = 128
    pipe = create_tt_pipeline(args, device, with_vae=False, with_encoders=False)
    pipe._null_condition_emb = null_emb

    def _fresh_inputs():
        # Fresh device tensors per call: the Tracer holds refs to its inputs and overwrites them, so
        # reusing the same handles across eager then traced calls corrupts the eager result.
        torch.manual_seed(0)
        noise = torch.randn(1, 1, seq_len, HIDDEN_CH)
        context = torch.randn(1, 1, seq_len, CONTEXT_CH)
        encoder = torch.randn(1, 1, 96, args.hidden_size)
        noise_tt = to_ttnn_tensor(noise, device)
        context_tt = to_ttnn_tensor(context, device)
        enc_tt = to_ttnn_tensor(encoder, device)
        return noise_tt, context_tt, enc_tt, pipe._uncond_context(enc_tt)

    nt, ct, et, uh = _fresh_inputs()
    eager = pipe.generate(
        nt, ct, et, infer_steps=INFER_STEPS, guidance_scale=GUIDANCE_SCALE,
        uncond_encoder_hidden_states=uh,
    )
    eager_t = ttnn.to_torch(eager).float().reshape(1, seq_len, HIDDEN_CH)

    nt, ct, et, uh = _fresh_inputs()
    traced = pipe.generate(
        nt, ct, et, infer_steps=INFER_STEPS, guidance_scale=GUIDANCE_SCALE,
        uncond_encoder_hidden_states=uh, use_trace=True,
    )
    traced_t = ttnn.to_torch(traced).float().reshape(1, seq_len, HIDDEN_CH)

    passing, msg = comp_pcc(eager_t, traced_t, 0.999)
    print(f"CFG_TRACED_VS_EAGER_PCC: {msg}")
    assert passing, f"traced CFG latents diverge from eager CFG: {msg} < 0.999"
