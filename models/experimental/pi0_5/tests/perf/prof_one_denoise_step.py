# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Drive ONE pi0.5 denoise step under the tt-metal Tracy device profiler.

  python -m tracy -v -r -p -m \\
    models.experimental.pi0_5.tests.perf.prof_one_denoise_step

Runs:
  - 1 warmup step (so JIT kernel cache + program cache are populated)
  - 1 measured step (the one we care about)
  - ttnn.synchronize_device at end so all device work finishes inside the capture
"""

from pathlib import Path

import torch
import ttnn

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_base"
PREFIX_LEN = 32
SEED = 0


def _build_model(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig(num_denoising_steps=10)
    return cfg, Pi0_5ModelTTNN(cfg, loader, device)


def _build_inputs(model, device):
    cfg = model.config
    ec = cfg.expert_config

    torch.manual_seed(SEED)
    # Host-pad to ah_padded (sample_actions:660-668) — the downstream sharded
    # RMSNorm requires tile-aligned logical shape (50 isn't tile-aligned, 64 is).
    ah_padded = model._action_horizon_padded
    noise_padded = torch.zeros(1, ah_padded, cfg.action_dim, dtype=torch.float32)
    noise_padded[:, : cfg.action_horizon, :] = torch.randn(1, cfg.action_horizon, cfg.action_dim)
    timestep = torch.tensor([0.5])

    noisy_ttnn = ttnn.from_torch(
        noise_padded,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    t_ttnn = ttnn.from_torch(timestep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Synthetic prefix KV cache — must match the suffix-side dtype (bf8_b) so
    # the keep_padded concat path validates. See note in test_perf_single_expert_block.py.
    prefix_kv_cache = []
    prefix_padded = ((PREFIX_LEN + 31) // 32) * 32
    for _ in range(ec.depth):
        k_t = torch.zeros(1, ec.num_kv_heads, prefix_padded, ec.head_dim, dtype=torch.float32)
        v_t = torch.zeros(1, ec.num_kv_heads, prefix_padded, ec.head_dim, dtype=torch.float32)
        k_t[:, :, :PREFIX_LEN] = torch.randn(1, ec.num_kv_heads, PREFIX_LEN, ec.head_dim) * 0.1
        v_t[:, :, :PREFIX_LEN] = torch.randn(1, ec.num_kv_heads, PREFIX_LEN, ec.head_dim) * 0.1
        prefix_kv_cache.append(
            (
                ttnn.from_torch(k_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.from_torch(v_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device),
            )
        )

    # Build the SDPA phantom mask for our synthetic prefix.
    sdpa_mask = model._build_sdpa_phantom_mask(PREFIX_LEN)
    model._sdpa_attn_mask = sdpa_mask
    model._sdpa_mask_kv_len = prefix_padded

    return noisy_ttnn, t_ttnn, prefix_kv_cache, sdpa_mask


def _one_denoise_step(model, x_t_ttnn, prefix_kv_cache, sdpa_mask, step_idx=0):
    """One full denoise step on the bs=1 fast path (precomputed mods)."""
    # Suffix embedding (action_in_proj only — adarms_cond is precomputed)
    suffix_embs = model.suffix_embedding.embed_actions(x_t_ttnn)
    adarms_cond = model._adarms_cond_per_step_bs1[step_idx]
    block_mods = model._block_mods_per_step[step_idx]
    final_mod = model._final_mod_per_step[step_idx]

    expert_out, _ = model.backbone.forward_expert(
        suffix_embs,
        adarms_cond=adarms_cond,
        past_key_values=prefix_kv_cache,
        precomputed_block_mods=block_mods,
        precomputed_final_mod=final_mod,
        attention_mask=sdpa_mask,
        keep_padded=True,
    )
    ttnn.deallocate(suffix_embs)

    velocity = model.suffix_embedding.project_output(expert_out)
    ttnn.deallocate(expert_out)

    # Euler integrate (matches sample_actions:737-744 non-fp32 path)
    dt = -0.1  # one step of 1/num_steps
    velocity_scaled = ttnn.mul(velocity, dt, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(velocity)
    x_t_new = ttnn.add(x_t_ttnn, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(velocity_scaled)
    return x_t_new


def main():
    # Open device (mesh of 1) on the visible chip.
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=24576,
    )
    try:
        cfg, model = _build_model(mesh)
        x_t, _t, kv, sdpa = _build_inputs(model, mesh)

        # Lift KV to keep_padded shape (matches sample_actions fast path).
        kv_lifted = [(ttnn.fill_implicit_tile_padding(k, 0.0), ttnn.fill_implicit_tile_padding(v, 0.0)) for k, v in kv]

        # 1× warmup so JIT / program cache + sharded-norm pcfg are built.
        x_t = _one_denoise_step(model, x_t, kv_lifted, sdpa, step_idx=0)
        ttnn.synchronize_device(mesh)

        # Signpost so the analysis script can isolate the measured step.
        ttnn.tracy_message("`TT_SIGNPOST: MEASURED_STEP_START`")
        x_t = _one_denoise_step(model, x_t, kv_lifted, sdpa, step_idx=1)
        ttnn.synchronize_device(mesh)
        ttnn.tracy_message("`TT_SIGNPOST: MEASURED_STEP_END`")

        ttnn.deallocate(x_t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
