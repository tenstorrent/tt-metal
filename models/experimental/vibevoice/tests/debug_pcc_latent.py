# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Debug script: compare TT vs reference diffusion latent for frame 0.
Run WITHOUT pytest, directly with: python models/experimental/vibevoice/tests/debug_pcc_latent.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # tt-metal root
_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for p in (_REFERENCE_DIR, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from models.experimental.vibevoice.common.config import MODEL_PATH, DEFAULT_TXT_PATH, VOICES_DIR
from models.experimental.vibevoice.tt.ttnn_dpm_scheduler import TTDPMSolverMultistepScheduler

MODEL_PATH = str(MODEL_PATH)
CFG_SCALE = 1.3
NUM_STEPS = 10
VOICE_PATH = str(next(iter(VOICES_DIR.glob("*.wav"))))
TEXT = open(DEFAULT_TXT_PATH).read().strip().replace("'", "'")


def run_ref_first_frame(ref_model, processor, inputs, cond_pos, cond_neg, seed):
    """Run reference diffusion for one frame with given conditions, return latent [1, 64]."""
    torch.manual_seed(seed)
    ref_model.set_ddpm_inference_steps(NUM_STEPS)
    ref_model.model.noise_scheduler.set_timesteps(NUM_STEPS)
    sched = ref_model.model.noise_scheduler

    # Replicate sample_speech_tokens exactly
    cond = torch.cat([cond_pos, cond_neg], dim=0)  # [2, hidden]
    speech = torch.randn(cond.shape[0], ref_model.config.acoustic_vae_dim).to(cond)
    print(f"  Ref initial noise (first 5): {speech[0, :5].tolist()}")

    for t in sched.timesteps:
        half = speech[: len(speech) // 2]
        combined = torch.cat([half, half], dim=0)
        with torch.no_grad():
            eps = ref_model.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=cond)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + CFG_SCALE * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        speech = sched.step(eps, t, speech).prev_sample

    result = speech[: len(speech) // 2]  # [1, 64]
    print(f"  Ref latent (first 5): {result[0, :5].tolist()}")
    return result


def run_tt_first_frame_cpu(pred_head_ref, cond_pos_tt, cond_neg_tt, seed):
    """Run TT-equivalent diffusion on CPU (float32) for one frame, return latent [1, 1, 1, 64]."""
    torch.manual_seed(seed)

    sched = TTDPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        solver_order=2,
        prediction_type="v_prediction",
    )
    sched.set_timesteps(NUM_STEPS)

    # Draw noise same way as TT: float32 [2,1,1,64], use first
    noise_2x = torch.randn(2, 1, 1, 64, dtype=torch.float32)
    noise = noise_2x[:1]  # [1, 1, 1, 64]
    print(f"  TT  initial noise (first 5): {noise[0, 0, 0, :5].tolist()}")

    # Reference draws [2, 64] = 128 float32 values
    # TT draws [2, 1, 1, 64] = 128 float32 values
    # Same count, same order — just different shape

    sample = noise  # [1, 1, 1, 64]

    # Build conditions: neg first, pos second (TT ordering)
    # cond_pos_tt: [1, 64], cond_neg_tt: [1, 64]
    # Reference: cat([pos, neg], dim=0) → [2, 64]
    # TT sample_speech_latents: cat([neg, pos], dim=0) for cond_combined

    cond_combined_ref = torch.cat([cond_pos_tt, cond_neg_tt], dim=0)  # [2, hidden] ref order

    for step_idx, t_val in enumerate(sched.timesteps):
        # TT uses [neg, pos] ordering for batch:
        # eps_uncond = result[0] (neg), eps_cond = result[1] (pos)
        # Equivalent: use ref order [pos, neg] and swap

        # For reference-equivalent CPU run, use [pos, neg] ordering:
        half = sample[0, 0]  # [1, 64] (flatten [1,1,64] for pred_head)
        combined = torch.cat([half, half], dim=0)  # [2, 64]
        t_tensor = torch.tensor([t_val, t_val], dtype=sample.dtype)
        with torch.no_grad():
            eps = pred_head_ref(combined, t_tensor, condition=cond_combined_ref)
        cond_eps, uncond_eps = torch.split(eps, 1, dim=0)
        half_eps = uncond_eps + CFG_SCALE * (cond_eps - uncond_eps)
        eps_full = torch.cat([half_eps, half_eps], dim=0)

        # Feed [2, 64] eps to scheduler as if sample is [2, 64]

        eps_4d = eps_full[0:1].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 64]

        # Use CPU scheduler step with scalar operations
        sigma = sched.sigmas[step_idx]
        alpha_t_val = 1.0 / (sigma**2 + 1) ** 0.5
        sigma_t_val = sigma * alpha_t_val

        x0_pred = sample * alpha_t_val - eps_4d * sigma_t_val

        # shift ring buffer
        for i in range(sched.solver_order - 1):
            sched.model_outputs[i] = sched.model_outputs[i + 1]
        sched.model_outputs[-1] = x0_pred

        lower_order_final = sched.lower_order_final and (step_idx == NUM_STEPS - 1) and NUM_STEPS < 15
        use_first_order = (sched.lower_order_nums < 1) or lower_order_final

        sigma_s = sched.sigmas[step_idx]
        sigma_t_next = sched.sigmas[step_idx + 1]

        import math

        alpha_s, sigma_s_v = 1.0 / (sigma_s**2 + 1) ** 0.5, sigma_s / (sigma_s**2 + 1) ** 0.5
        alpha_t, sigma_t_v = 1.0 / (sigma_t_next**2 + 1) ** 0.5, sigma_t_next / (sigma_t_next**2 + 1) ** 0.5

        lam_s = math.log(alpha_s) - math.log(sigma_s_v) if sigma_s_v > 1e-8 else 1e9
        lam_t = math.log(alpha_t) - math.log(sigma_t_v) if sigma_t_v > 1e-8 else 1e9
        h = lam_t - lam_s

        if use_first_order:
            ratio = sigma_t_v / (sigma_s_v if sigma_s_v > 1e-8 else 1e-8)
            coeff = -alpha_t * math.expm1(-h)
            sample = sample * ratio + x0_pred * coeff
        else:
            D0 = sched.model_outputs[-1]
            D1 = (sched.model_outputs[-1] - sched.model_outputs[-2]) / (sched.sigmas[step_idx - 1] / sigma_s)
            coeff_x = sigma_t_v / (sigma_s_v if sigma_s_v > 1e-8 else 1e-8)
            coeff_D0 = -alpha_t * math.expm1(-h)
            coeff_D1 = 0.5 * coeff_D0
            sample = sample * coeff_x + D0 * coeff_D0 + D1 * coeff_D1

        if sched.lower_order_nums < sched.solver_order:
            sched.lower_order_nums += 1
        sched._step_index += 1

    print(f"  TT  latent (first 5): {sample[0, 0, 0, :5].tolist()}")
    return sample


def main():
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

    print("Loading processor and reference model...")
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    ref_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )
    ref_model.eval()
    ref_model.set_ddpm_inference_steps(NUM_STEPS)

    inputs = processor(
        text=[TEXT], voice_samples=[[VOICE_PATH]], padding=True, return_tensors="pt", return_attention_mask=True
    )

    print("\nRunning reference generate (first frame condition capture)...")
    captured = {}

    def hook_fn(module, args, output):
        # Capture first call's conditions
        if "pos_cond" not in captured:
            pos_cond = args[0][:1] if args[0].shape[0] >= 1 else args[0]
            neg_cond = args[0][1:2] if args[0].shape[0] >= 2 else args[0]
            captured["pos_cond"] = pos_cond.detach()
            captured["neg_cond"] = neg_cond.detach()
            captured["latent_out"] = output.detach()
            print(f"  [hook] sample_speech_tokens conditions captured, pos_cond shape={pos_cond.shape}")

    # Hook into sample_speech_tokens — capture conditions at first call
    orig_sample = ref_model.sample_speech_tokens.__func__

    call_count = [0]

    def patched_sample(self, cond_pos, cond_neg, cfg_scale=3.0):
        if call_count[0] == 0:
            captured["pos_cond"] = cond_pos.detach().clone()
            captured["neg_cond"] = cond_neg.detach().clone()
        call_count[0] += 1
        return orig_sample(self, cond_pos, cond_neg, cfg_scale)

    import types

    ref_model.sample_speech_tokens = types.MethodType(patched_sample, ref_model)

    torch.manual_seed(0)
    with torch.no_grad():
        ref_out = ref_model.generate(
            **inputs,
            max_new_tokens=128,
            cfg_scale=CFG_SCALE,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            is_prefill=True,
        )

    print(f"  ref speech_outputs[0] shape: {ref_out.speech_outputs[0].shape}")
    print(f"  ref sequences shape: {ref_out.sequences.shape}")

    if "pos_cond" not in captured:
        print("ERROR: conditions not captured!")
        return

    pos_cond = captured["pos_cond"]  # [1, hidden]
    neg_cond = captured["neg_cond"]  # [1, hidden]
    print(f"  pos_cond shape: {pos_cond.shape}, norm: {pos_cond.norm():.4f}")
    print(f"  neg_cond shape: {neg_cond.shape}, norm: {neg_cond.norm():.4f}")

    # Now compare: reference diffusion latent vs TT equivalent
    print("\n--- Reference first frame ---")
    ref_latent = run_ref_first_frame(ref_model, processor, inputs, pos_cond, neg_cond, seed=0)
    print(f"  Ref latent norm: {ref_latent.norm():.4f}")

    # Unscale
    scale = float(ref_model.model.speech_scaling_factor.item())
    bias = float(ref_model.model.speech_bias_factor.item())
    ref_unscaled = ref_latent / scale - bias
    print(f"  scale={scale:.6f}, bias={bias:.6f}")
    print(f"  Ref unscaled latent (first 5): {ref_unscaled[0, :5].tolist()}")

    print("\n--- TT equivalent (CPU simulation) ---")
    print("  (TT uses float32 noise same shape/count as reference)")
    # The TT draws [2,1,1,64] float32 — same 128 values as reference's [2,64]
    # To simulate TT's first-frame diffusion, we need TT-side conditions (from bfloat16 LM)
    # For now, use same float32 conditions to isolate scheduler math
    tt_latent = run_tt_first_frame_cpu(ref_model.model.prediction_head, pos_cond, neg_cond, seed=0)
    tt_latent_2d = tt_latent[0, 0]  # [1, 64]
    print(f"  TT  latent norm: {tt_latent_2d.norm():.4f}")

    tt_unscaled = tt_latent_2d / scale - bias
    print(f"  TT  unscaled latent (first 5): {tt_unscaled[0, :5].tolist()}")

    from models.common.utility_functions import comp_pcc

    passed, pcc_val = comp_pcc(ref_unscaled.reshape(-1), tt_unscaled.reshape(-1), pcc=0.99)
    print(f"\n  PCC (ref vs TT-equivalent latent): {pcc_val:.6f} (pass={passed})")


if __name__ == "__main__":
    main()
