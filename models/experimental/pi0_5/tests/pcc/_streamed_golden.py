# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared torch streamed-denoise golden (plan iter-3 §8, SC3).

Lifts the torch Euler loop from test_pcc_pi05_pipeline_denoise_vs_torch.py into a shared
helper so the e2e PCC test and the trace-parity test compute the SAME golden. The forward_fn
matches the device denoise step:
  velocity = project_output(forward_expert(embed_actions(x_t), adarms_cond(t), prefix_kv))
and the Euler loop is x_next = x_t + dt*v_t, fully fp32. ZERO tt_symbiote imports.
"""
from __future__ import annotations

import torch

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"


def compute_pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


def build_forward_fn(cfg, weights, suffix_ref, prefix_kv_torch):
    """forward_fn(x_t, t) -> velocity, matching the device denoise step."""
    from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone as TorchBackbone

    backbone = TorchBackbone(cfg, weights)

    def forward_fn(x_t, t):
        cond = suffix_ref.embed_timestep_adarms(torch.as_tensor(t, dtype=torch.float32).reshape(-1))
        h = suffix_ref.embed_actions(x_t)
        h, _ = backbone.forward_expert(h, adarms_cond=cond, past_key_values=prefix_kv_torch, use_cache=False)
        return suffix_ref.project_output(h)

    return forward_fn


def euler_golden(cfg, weights, suffix_ref, prefix_kv_torch, x_t_init, num_steps, action_horizon):
    """Full fp32 Euler loop -> golden actions [:, :action_horizon, :]."""
    forward_fn = build_forward_fn(cfg, weights, suffix_ref, prefix_kv_torch)
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]
    x = x_t_init.clone()
    with torch.no_grad():
        for i in range(num_steps):
            dt = timesteps[i + 1] - timesteps[i]
            x = x + dt * forward_fn(x, timesteps[i])
    return x[:, :action_horizon, :]
