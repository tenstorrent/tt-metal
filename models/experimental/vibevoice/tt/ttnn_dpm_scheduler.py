# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice DPM Multistep Scheduler — TTNN port.

Reference: schedule/dpm_solver.py (DPMSolverMultistepScheduler)
           generate() lines 787–799 in modeling_vibevoice_inference.py

Strategy:
  - All noise-schedule math (sigmas, alphas, timesteps) is precomputed on host
    in __init__ / set_timesteps using torch (allowed: preprocess only).
  - step() and convert_model_output() operate exclusively on ttnn.Tensor;
    scalar coefficients are Python floats broadcast via ttnn.mul_sfloat.
  - sample_speech_latents() calls TT diffusion head each step; no torch on device.
"""

import math
from typing import List, Optional

import torch
import ttnn


_COMPUTE_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def _ttnn_scalar_mul(x: ttnn.Tensor, scalar: float) -> ttnn.Tensor:
    """Multiply TTNN tensor by a Python float scalar.

    Uses the direct scalar-operand form (not ``ttnn.mul(x, ttnn.full(...))``): ttnn.full is a
    host->device constant write, which is illegal inside a ttnn trace capture (the whole DPM
    loop is captured for --trace).  The scalar path broadcasts device-side, no host write.
    """
    return ttnn.mul(x, scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _ttnn_scalar_add(x: ttnn.Tensor, scalar: float) -> ttnn.Tensor:
    """Add a Python float scalar to a TTNN tensor (direct scalar form; see _ttnn_scalar_mul)."""
    return ttnn.add(x, scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class TTDPMSolverMultistepScheduler:
    """TTNN port of DPMSolverMultistepScheduler.

    Precomputes noise schedule on host; step() uses only ttnn ops on device.
    Only the subset of features used by VibeVoice is implemented:
      - algorithm_type = "dpmsolver++"
      - prediction_type = "v_prediction"
      - solver_order = 2 (multistep)
      - lower_order_final = True
      - thresholding = False
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
        solver_order: int = 2,
        prediction_type: str = "v_prediction",
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        timestep_spacing: str = "linspace",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.timestep_spacing = timestep_spacing

        # Build betas/alphas on host (torch, allowed in __init__)
        if beta_schedule == "cosine" or beta_schedule == "squaredcos_cap_v2":
            betas = self._betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alpha_t = torch.sqrt(alphas_cumprod).tolist()
        self.sigma_t = torch.sqrt(1 - alphas_cumprod).tolist()
        self.sigmas_full = ((1 - alphas_cumprod) / alphas_cumprod).sqrt().tolist()
        self.lambda_t = (torch.log(torch.sqrt(alphas_cumprod)) - torch.log(torch.sqrt(1 - alphas_cumprod))).tolist()

        self.num_inference_steps: Optional[int] = None
        self.timesteps: Optional[List[int]] = None
        self.sigmas: Optional[List[float]] = None
        self.model_outputs: List[Optional[ttnn.Tensor]] = [None] * solver_order
        self.lower_order_nums: int = 0
        self._step_index: Optional[int] = None

    @staticmethod
    def _betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
        def alpha_bar_fn(t: float) -> float:
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Compute inference timestep schedule on host (torch allowed here)."""
        import numpy as np

        self.num_inference_steps = num_inference_steps
        num_train = self.num_train_timesteps

        if self.timestep_spacing == "linspace":
            timesteps = np.linspace(0, num_train - 1, num_inference_steps + 1).round()[::-1][:-1].astype(np.int64)
        else:
            step_ratio = num_train // (num_inference_steps + 1)
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].astype(np.int64)

        sigmas_arr = np.array(self.sigmas_full)
        sigmas_sched = np.interp(timesteps, np.arange(len(sigmas_arr)), sigmas_arr)
        sigmas_sched = list(sigmas_sched.astype(float)) + [0.0]

        self.timesteps = timesteps.tolist()
        self.sigmas = sigmas_sched
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self._step_index = 0

    def _sigma_to_alpha_sigma_t(self, sigma: float):
        alpha_t = 1.0 / math.sqrt(sigma**2 + 1)
        sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def convert_model_output(self, model_output: ttnn.Tensor, sample: ttnn.Tensor) -> ttnn.Tensor:
        """Convert diffusion head output to x0_pred (device tensors only).

        v_prediction: x0_pred = alpha_t * sample - sigma_t * model_output
        """
        assert self._step_index is not None, "call set_timesteps before step"
        sigma = self.sigmas[self._step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        # x0_pred = alpha_t * sample - sigma_t * model_output
        x0_pred = ttnn.subtract(
            _ttnn_scalar_mul(sample, alpha_t),
            _ttnn_scalar_mul(model_output, sigma_t),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return x0_pred

    def dpm_solver_first_order_update(
        self,
        model_output: ttnn.Tensor,
        sample: ttnn.Tensor,
        prev_sample: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """First-order DPMSolver++ update (euler step)."""
        step_index = self._step_index
        sigma_s = self.sigmas[step_index]
        sigma_t = self.sigmas[step_index + 1]

        alpha_s, sigma_s_val = self._sigma_to_alpha_sigma_t(sigma_s)
        alpha_t, sigma_t_val = self._sigma_to_alpha_sigma_t(sigma_t)

        lambda_s = math.log(alpha_s) - math.log(sigma_s_val) if sigma_s_val > 1e-8 else 1e9
        lambda_t = math.log(alpha_t) - math.log(sigma_t_val) if sigma_t_val > 1e-8 else 1e9

        h = lambda_t - lambda_s
        # dpmsolver++: x_t = (sigma_t/sigma_s)*x_s - alpha_t*(expm1(-h))*model_s
        ratio = sigma_t_val / (sigma_s_val if sigma_s_val > 1e-8 else 1e-8)
        coeff_model = -alpha_t * math.expm1(-h)

        prev = ttnn.add(
            _ttnn_scalar_mul(sample, ratio),
            _ttnn_scalar_mul(model_output, coeff_model),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return prev

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: list,
        sample: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Second-order DPMSolver++ multistep update (midpoint)."""
        step_index = self._step_index
        sigma_s0 = self.sigmas[step_index]
        sigma_s1 = self.sigmas[step_index - 1]
        sigma_t = self.sigmas[step_index + 1]

        alpha_s0, sigma_s0_val = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1_val = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_t, sigma_t_val = self._sigma_to_alpha_sigma_t(sigma_t)

        lambda_s0 = math.log(alpha_s0) - math.log(sigma_s0_val) if sigma_s0_val > 1e-8 else 1e9
        lambda_s1 = math.log(alpha_s1) - math.log(sigma_s1_val) if sigma_s1_val > 1e-8 else 1e9
        lambda_t = math.log(alpha_t) - math.log(sigma_t_val) if sigma_t_val > 1e-8 else 1e9

        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0 = model_output_list[-1]
        D1 = _ttnn_scalar_mul(
            ttnn.subtract(model_output_list[-1], model_output_list[-2], memory_config=ttnn.DRAM_MEMORY_CONFIG),
            1.0 / r0,
        )

        if self.solver_type == "midpoint":
            coeff_x = sigma_t_val / (sigma_s0_val if sigma_s0_val > 0 else 1e-8)
            coeff_D0 = -alpha_t * math.expm1(-h)
            coeff_D1 = 0.5 * coeff_D0  # midpoint: D1 uses same exp factor, half weight

            prev = ttnn.add(
                ttnn.add(
                    _ttnn_scalar_mul(sample, coeff_x),
                    _ttnn_scalar_mul(D0, coeff_D0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                _ttnn_scalar_mul(D1, coeff_D1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            # heun — same as midpoint for our purposes
            prev = self.dpm_solver_first_order_update(D0, sample)
        return prev

    def step(
        self,
        model_output: ttnn.Tensor,
        sample: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Run one denoising step. Returns the updated sample (device tensor).

        No torch tensors on device — all scalars are Python floats.
        """
        assert self._step_index is not None, "call set_timesteps first"

        lower_order_final = (
            self.lower_order_final
            and (self._step_index == self.num_inference_steps - 1)
            and self.num_inference_steps < 15
        )

        converted = self.convert_model_output(model_output, sample)
        # shift model_outputs ring buffer
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = converted

        # Match reference: first order only when lower_order_nums < 1 or final step;
        # solver_order==2 triggers second order for all other steps.
        use_first_order = (self.lower_order_nums < 1) or lower_order_final
        if use_first_order:
            prev_sample = self.dpm_solver_first_order_update(converted, sample)
        else:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1
        return prev_sample


def sample_speech_latents(
    diffusion_head,
    condition_tt: ttnn.Tensor,
    neg_condition_tt: ttnn.Tensor,
    scheduler: TTDPMSolverMultistepScheduler,
    initial_latent: ttnn.Tensor,
    cfg_scale: float = 1.3,
    num_steps: int = 10,
    head_runner=None,
    t_tensors=None,
) -> ttnn.Tensor:
    """Run the DPM multistep loop using TT diffusion head.

    No torch tensors on device — all operations use ttnn.

    Args:
        diffusion_head: TTDiffusionHead instance
        condition_tt:     [1, 1, 1, hidden] positive conditioning
        neg_condition_tt: [1, 1, 1, hidden] negative conditioning (uncond)
        scheduler:        TTDPMSolverMultistepScheduler
        initial_latent:   [1, 1, 1, latent_size] initial noise
        cfg_scale:        classifier-free guidance scale
        num_steps:        number of denoising steps
        head_runner:      optional callable (noisy_images, timesteps, condition) -> eps,
                          drop-in for the head forward.  Used to ttnn-trace the head
                          (fixed B=2 shape, replayed every step); defaults to the eager
                          ``diffusion_head``.  The per-step scheduler.step stays host-side
                          (its coefficients are per-step Python floats that would bake into
                          a trace), so only the head forward is replaced.

    Returns:
        final denoised latent [1, 1, 1, latent_size]
    """
    if head_runner is None:
        head_runner = diffusion_head
    scheduler.set_timesteps(num_steps)

    sample = initial_latent
    latent_shape = initial_latent.shape

    # The CFG condition is step-INVARIANT (only the noisy latent + timestep change across the
    # num_steps loop), so hoist the condition concat + its Linear projection OUT of the loop:
    # computed once/frame instead of once/step.  Byte-identical (same ops on the same fixed inputs).
    # Requires the head to expose project_condition/forward_pre_cond (TTDiffusionHead); a custom
    # head_runner callable falls back to the original per-step full forward.
    cond_combined = ttnn.concat([neg_condition_tt, condition_tt], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _use_precond = head_runner is diffusion_head and hasattr(diffusion_head, "forward_pre_cond")
    cond_proj = diffusion_head.project_condition(cond_combined) if _use_precond else None

    for step_idx, t_val in enumerate(scheduler.timesteps):
        # Expand sample to CFG batch: [2, 1, 1, latent]
        sample_expanded = ttnn.concat([sample, sample], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Timestep tensor [2, 1, 1, 1].  Prefer pre-built tensors (t_tensors) when provided:
        # ttnn.full is a host->device write, illegal inside a trace capture, so the traced
        # (--trace) path passes tensors built once before capture; eager builds them here.
        if t_tensors is not None:
            t_tensor = t_tensors[step_idx]
        else:
            t_tensor = ttnn.full(
                (2, 1, 1, 1),
                float(t_val),
                dtype=ttnn.bfloat16,
                device=condition_tt.device(),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Run diffusion head on CFG batch (hoisted cond_proj when using the TTDiffusionHead)
        if _use_precond:
            eps_combined = diffusion_head.forward_pre_cond(sample_expanded, t_tensor, cond_proj)
        else:
            eps_combined = head_runner(sample_expanded, t_tensor, cond_combined)

        # Split CFG outputs
        eps_uncond = ttnn.slice(eps_combined, [0, 0, 0, 0], [1, 1, 1, eps_combined.shape[-1]])
        eps_cond = ttnn.slice(eps_combined, [1, 0, 0, 0], [2, 1, 1, eps_combined.shape[-1]])

        # CFG: eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        eps = ttnn.add(
            eps_uncond,
            _ttnn_scalar_mul(
                ttnn.subtract(eps_cond, eps_uncond, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                cfg_scale,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Scheduler step
        sample = scheduler.step(eps, sample)

    return sample
