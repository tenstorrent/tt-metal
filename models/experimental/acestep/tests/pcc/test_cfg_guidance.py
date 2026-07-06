# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: TT CFG denoise (APG guidance) vs the genuine reference CFG sampling loop.

The reference runs the DiT twice per step (conditional context + a learned null_condition_emb) and
combines the two velocities with apg_forward (adaptive projected guidance, the base-model default).
Our pipeline's CFG path (_generate_cfg, pure ttnn apg_forward_ttnn) must match this numerically so
the whole model — WITH guidance — clears the e2e audio PCC gate. Without CFG the audio is noise-like;
this test gates the feature that makes it prompt-faithful.

Reference denoise+APG mirrors modeling_acestep_v15_base.py sample() + apg_guidance.py exactly.
Compares the FINAL audio (DiT CFG denoise -> VAE decode) vs the reference's CFG audio. Gate >= 0.95.
"""

import pytest
import torch

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


def _reference_cfg_denoise(ref_dit, rope, args, noise, context, enc_cond, null_emb, infer_steps, gs):
    """Reference CFG ODE denoise: per step DiT(cond) + DiT(null), combine via apg_forward (dims=[1])."""
    seq_len = noise.shape[1]
    t_prime = seq_len // args.patch_size
    pos = torch.arange(t_prime).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, t_prime, 128), pos)
    t = torch.linspace(1.0, 0.0, infer_steps + 1)
    enc_uncond = null_emb.reshape(1, 1, -1).expand(1, enc_cond.shape[1], -1).contiguous()
    mb = MomentumBuffer()

    xt = noise
    with torch.no_grad():
        for i in range(infer_steps):
            t_curr = t[i].reshape(1)
            (vt_c, *_) = ref_dit(
                hidden_states=xt, timestep=t_curr, timestep_r=t_curr, attention_mask=None,
                encoder_hidden_states=enc_cond, encoder_attention_mask=None, context_latents=context,
            )
            (vt_u, *_) = ref_dit(
                hidden_states=xt, timestep=t_curr, timestep_r=t_curr, attention_mask=None,
                encoder_hidden_states=enc_uncond, encoder_attention_mask=None, context_latents=context,
            )
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

    # Reference CFG denoise -> VAE decode.
    ref_latents = _reference_cfg_denoise(ref_dit, rope, args, noise, context, encoder, null_emb, INFER_STEPS, GUIDANCE_SCALE)
    with torch.no_grad():
        ref_wav = ref_vae.decode(ref_latents.transpose(1, 2)).sample

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
    passing, msg = comp_pcc(ref_wav[..., :n], tt_wav[..., :n], 0.95)
    logger.info(f"CFG e2e audio: ref={ref_wav.shape[-1]} tt={tt_wav.shape[-1]} gs={GUIDANCE_SCALE}")
    print(f"CFG_E2E_PCC: {msg}")
    assert passing, f"CFG e2e audio PCC {msg} < 0.95"
