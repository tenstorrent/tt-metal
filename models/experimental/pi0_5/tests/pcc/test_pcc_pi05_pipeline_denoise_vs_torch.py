# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end streamed-denoise PCC vs the torch golden (plan §11.1).

Golden: ``reference/torch_denoise.DenoisingModule.sample_actions`` (Euler
``x_next = x_t + dt*v_t``) with ``forward_fn(x_t, t, prefix_kv)`` =
``project_output(forward_expert(embed_actions(x_t), adarms_cond(t), prefix_kv))`` over the
SAME synthetic prefix KV the device path receives. Seed-fixed ``x_t_init`` fed to both.

Device path: ``StageDenoise.sample_actions(capture=True)`` on a standalone fresh >=4-chip
parent (splits (5,5,4,4)). The adapter builds the phantom mask + offset internally.

Skips cleanly without hardware or a checkpoint. (Note: the synthetic-KV PCC bar follows the
project's 0.95 e2e convention; the §11 0.99 target is for the real-KV path and is reachable
with the KV-bf16 / LoFi flips documented in PORT_NOTES.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

sys.path.insert(0, str(Path(__file__).parent))  # sibling _fabric_harness
from _fabric_harness import close_parent as _close_parent  # noqa: E402
from _fabric_harness import open_parent_with_retry as _open_parent  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/ttuser/salnahari/pi05_weights/pi05_base"))
SEED = 42
PCC_THRESHOLD = float(os.environ.get("PI05_PIPELINE_PCC", "0.95"))
_TRACE_REGION = 134_217_728


def _compute_pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


def _build_golden(cfg, weights, suffix_ref, prefix_kv_torch):
    """Return forward_fn(x_t, t) -> velocity, matching the device denoise step."""
    from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone as TorchBackbone

    backbone = TorchBackbone(cfg, weights)

    def forward_fn(x_t, t):
        cond = suffix_ref.embed_timestep_adarms(torch.as_tensor(t, dtype=torch.float32).reshape(-1))
        h = suffix_ref.embed_actions(x_t)
        h, _ = backbone.forward_expert(h, adarms_cond=cond, past_key_values=prefix_kv_torch, use_cache=False)
        return suffix_ref.project_output(h)

    return forward_fn


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(), reason=f"checkpoint not found at {CHECKPOINT_DIR}"
)
def test_e2e_streamed_denoise_vs_torch():
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_pipeline import StageDenoise
    from models.experimental.pi0_5.tt.tt_pipeline.weight_adapt import suffix_reference

    num_steps = int(os.environ.get("PI05_PIPELINE_STEPS", "5"))
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR), num_denoising_steps=num_steps)
    weights = Pi0_5WeightLoader(str(CHECKPOINT_DIR)).categorized_weights
    ec = cfg.expert_config
    B = 1
    ah = cfg.action_horizon
    ah_pad = ((ah + 31) // 32) * 32
    prefix_len = 288

    torch.manual_seed(SEED)
    x_t_init = torch.zeros(B, ah_pad, cfg.action_dim)
    x_t_init[:, :ah, :] = torch.randn(B, ah, cfg.action_dim)
    prefix_kv_torch = [
        (
            torch.randn(B, ec.num_kv_heads, prefix_len, ec.head_dim) * 0.5,
            torch.randn(B, ec.num_kv_heads, prefix_len, ec.head_dim) * 0.5,
        )
        for _ in range(ec.depth)
    ]

    # --- torch golden ---
    from models.experimental.pi0_5.common.configs import SuffixConfig

    sc = SuffixConfig(action_dim=cfg.action_dim, action_horizon=ah, expert_width=ec.width, pi05=True)
    suffix_ref = suffix_reference(weights["pi0_projections"], sc)
    forward_fn = _build_golden(cfg, weights, suffix_ref, prefix_kv_torch)
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]
    x = x_t_init.clone()
    with torch.no_grad():
        for i in range(num_steps):
            dt = timesteps[i + 1] - timesteps[i]
            x = x + dt * forward_fn(x, timesteps[i])
    torch_actions = x[:, :ah, :]

    # --- device streamed path ---
    parent = _open_parent(4)
    try:
        stage = StageDenoise(cfg, weights, None, parent_mesh=parent, n_submeshes=4)
        actions = stage.sample_actions(
            x_t_init_torch=x_t_init,
            prefix_kv_cache_torch=prefix_kv_torch,
            num_steps=num_steps,
            prefix_len=prefix_len,
            action_horizon=ah,
            capture=True,
        )
        stage.close()
    finally:
        _close_parent(parent)

    pcc = _compute_pcc(torch_actions, actions[:, :ah, :])
    print(f"\ne2e streamed denoise PCC vs torch ({num_steps} steps): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
