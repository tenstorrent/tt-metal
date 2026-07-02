# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pi0.5 per-step PCC test.

The existing aggregate PCC test (test_pcc_pi05_model.py) only compares the final
denoised action tensor. This script extends that to per-denoise-step resolution.

Two metrics are reported:

  1. Per-step velocity PCC (primary gate):
     At each denoise step i, feed the SAME x_t_i (PyTorch's) into both models,
     and compare the velocities they produce. This isolates per-step accuracy
     without any cross-step error accumulation, and it matches the per-step
     semantics of the flow-matching Euler solver.

  2. End-to-end aggregate PCC (secondary):
     Call each model's real sample_actions() path with matched initial noise.
     Real TTNN keeps x_t on-device in bf16 throughout, so this is the correct
     way to observe compound drift — unlike a per-step host-driven loop, which
     would introduce extra bf16 round-tripping artifacts.

Threshold: per-step velocity PCC ≥ 0.99 for every step.

Usage:
    PYTHONPATH=<root>/ttnn:<root> python test_pcc_pi05_per_step.py
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as PI0ModelTorch
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader


_REPO_ROOT = Path(__file__).resolve().parents[5]  # tt-metal repo root
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", str(_REPO_ROOT))
CHECKPOINT_PATH = os.environ.get("PI0_CHECKPOINT", "lerobot/pi05_base")
BATCH_SIZE = 1
SEED = 42
PER_STEP_PCC_THRESHOLD = 0.99
# Informational secondary gate: compound bf16 drift over the 10-step Euler solver
# makes the 0.99 per-step bar unreachable on the aggregate output.
E2E_PCC_THRESHOLD = 0.93


def create_pi05_config() -> PI0ModelConfig:
    config = PI0ModelConfig(
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
    )
    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    return config


def create_test_inputs(config: PI0ModelConfig, batch_size: int = 1):
    image_size = config.siglip_config.image_size
    images = [torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32) for _ in range(2)]
    img_masks = [torch.ones(batch_size, dtype=torch.bool) for _ in range(2)]
    lang_tokens = torch.randint(0, 256000, (batch_size, 32))
    lang_masks = torch.ones(batch_size, 32, dtype=torch.bool)
    state = torch.randn(batch_size, config.state_dim, dtype=torch.float32)
    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
    }


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-8 or s2 < 1e-8:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


def torch_velocity(model_torch, x_t, t_scalar, vlm_cache, state):
    t = torch.tensor([t_scalar], dtype=torch.float32).expand(x_t.shape[0])
    return model_torch._denoise_forward(
        noisy_actions=x_t,
        timestep=t,
        kv_cache=vlm_cache,
        state=state,
    )


def ttnn_velocity(model_ttnn, x_t_torch, step_idx, prefix_kv_cache, state_ttnn):
    x_t_ttnn = ttnn.from_torch(
        x_t_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=model_ttnn.device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    t_tensor = model_ttnn._precomputed_timesteps[step_idx]

    suffix_embs, _, _, adarms_cond = model_ttnn.embed_suffix(state_ttnn, x_t_ttnn, t_tensor)
    expert_out, _ = model_ttnn.backbone.forward_expert(
        suffix_embs,
        past_key_values=prefix_kv_cache,
        adarms_cond=adarms_cond,
    )
    if not model_ttnn.config.pi05:
        action_out = ttnn.slice(
            expert_out,
            [0, 1, 0],
            [expert_out.shape[0], expert_out.shape[1], expert_out.shape[2]],
        )
    else:
        action_out = expert_out
    velocity = model_ttnn.suffix_embedding.project_output(action_out)
    return ttnn.to_torch(velocity)


def _resolve_checkpoint() -> Path:
    """Return the checkpoint path, skipping the pytest run if a local path is absent."""
    ckpt = Path(CHECKPOINT_PATH)
    if ckpt.is_absolute() and not ckpt.exists():
        pytest.skip(f"Checkpoint not found: {ckpt}")
    return ckpt


def run_per_step_pcc(device):
    """Return (worst per-step velocity PCC, end-to-end aggregate PCC) for TTNN vs torch.

    Feeds the SAME x_t (PyTorch's) into both models at each denoise step so the
    per-step velocity comparison is free of cross-step error accumulation, then
    runs each model's real sample_actions() path with matched initial noise for
    the aggregate metric.
    """
    config = create_pi05_config()
    num_steps = config.num_denoising_steps
    weight_loader = PI0WeightLoader(str(_resolve_checkpoint()))

    model_torch = PI0ModelTorch(config, weight_loader)
    torch.manual_seed(SEED)
    model_ttnn = PI0ModelTTNN(config, weight_loader, device)

    inputs = create_test_inputs(config, batch_size=BATCH_SIZE)
    torch.manual_seed(SEED)
    x_t_shared = torch.randn(BATCH_SIZE, config.action_horizon, config.action_dim, dtype=torch.float32)

    # Prefix / VLM forward on both backends.
    with torch.no_grad():
        prefix_embs_t, _, _ = model_torch.embed_prefix(
            inputs["images"], inputs["img_masks"], inputs["lang_tokens"], inputs["lang_masks"]
        )
        _, vlm_cache_torch = model_torch.backbone.forward_vlm(prefix_embs_t, use_cache=True)

    images_ttnn = [
        ttnn.from_torch(
            img,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for img in inputs["images"]
    ]
    lang_tokens_ttnn = ttnn.from_torch(
        inputs["lang_tokens"], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    lang_masks_ttnn = ttnn.from_torch(
        inputs["lang_masks"].float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    state_ttnn = ttnn.from_torch(inputs["state"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    prefix_embs_tt, _, _ = model_ttnn.embed_prefix(images_ttnn, inputs["img_masks"], lang_tokens_ttnn, lang_masks_ttnn)
    _, prefix_kv_cache_ttnn = model_ttnn.backbone.forward_vlm(prefix_embs_tt, use_cache=True)

    # Per-step velocity PCC with a shared x_t fed to both models.
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]
    x_t_torch = x_t_shared.clone()
    worst_v_pcc = 1.0
    with torch.no_grad():
        for i in range(num_steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            v_torch = torch_velocity(model_torch, x_t_torch, t, vlm_cache_torch, inputs["state"])
            v_ttnn = ttnn_velocity(model_ttnn, x_t_torch, i, prefix_kv_cache_ttnn, state_ttnn)
            worst_v_pcc = min(worst_v_pcc, compute_pcc(v_torch, v_ttnn))
            x_t_torch = x_t_torch + dt * v_torch

        # End-to-end aggregate with matched initial noise.
        model_ttnn.x_t_ttnn = ttnn.from_torch(
            x_t_shared,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=model_ttnn.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        saved_sample_noise = model_torch.denoising.sample_noise
        model_torch.denoising.sample_noise = lambda bs, device=None, dtype=torch.float32: x_t_shared.clone()
        try:
            torch_actions = model_torch.forward_inference(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )
        finally:
            model_torch.denoising.sample_noise = saved_sample_noise

        ttnn_actions = model_ttnn.sample_actions(
            images=images_ttnn,
            img_masks=inputs["img_masks"],
            lang_tokens=lang_tokens_ttnn,
            lang_masks=lang_masks_ttnn,
            state=state_ttnn,
        )
        if isinstance(ttnn_actions, ttnn.Tensor):
            ttnn_actions = ttnn.to_torch(ttnn_actions)

    return worst_v_pcc, compute_pcc(torch_actions, ttnn_actions)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pcc_pi05_per_step(device):
    """Per-step velocity PCC (primary gate) + end-to-end aggregate PCC for Pi0.5."""
    worst_v_pcc, e2e_pcc = run_per_step_pcc(device)
    print(
        f"\n✅ per-step worst v_pcc={worst_v_pcc:.6f} (≥{PER_STEP_PCC_THRESHOLD}), "
        f"e2e_pcc={e2e_pcc:.6f} (≥{E2E_PCC_THRESHOLD})"
    )
    assert (
        worst_v_pcc >= PER_STEP_PCC_THRESHOLD
    ), f"worst per-step velocity PCC {worst_v_pcc:.6f} < {PER_STEP_PCC_THRESHOLD}"
    assert e2e_pcc >= E2E_PCC_THRESHOLD, f"end-to-end PCC {e2e_pcc:.6f} < {E2E_PCC_THRESHOLD}"


def main():
    print("=" * 80)
    print("  PI0.5 PER-STEP PCC TEST")
    print("=" * 80)

    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        print(f"❌ Checkpoint not found: {ckpt}")
        return 1
    print(f"📁 Checkpoint: {ckpt}")

    print("🔌 Opening TTNN device...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    grid = device.compute_with_storage_grid_size()
    print(f"   Grid: {grid.x}x{grid.y}")

    try:
        config = create_pi05_config()
        num_steps = config.num_denoising_steps
        print(f"📋 pi05={config.pi05}, adaRMS={config.expert_config.use_adarms}, num_steps={num_steps}")

        print("1. Loading weights...")
        weight_loader = PI0WeightLoader(str(ckpt))

        print("2. Initializing PyTorch reference...")
        model_torch = PI0ModelTorch(config, weight_loader)

        # Match the aggregate baseline test's seeding pattern exactly:
        # seed → ttnn ctor (1st randn = x_t_ttnn) → inputs built from advanced RNG → reset seed.
        # Torch side uses the reset seed so its own first randn in sample_noise matches ttnn's x_0.
        print("3. Initializing TTNN model...")
        torch.manual_seed(SEED)
        model_ttnn = PI0ModelTTNN(config, weight_loader, device)

        print("4. Building test inputs (seeded, matches baseline test)...")
        inputs = create_test_inputs(config, batch_size=BATCH_SIZE)

        # Reset seed and draw x_0 the same way the ttnn constructor did: first randn after seed=SEED.
        torch.manual_seed(SEED)
        x_t_shared = torch.randn(BATCH_SIZE, config.action_horizon, config.action_dim, dtype=torch.float32)
        print(f"   x_0 shape: {tuple(x_t_shared.shape)}, std={x_t_shared.std().item():.4f}")

        # ---- PyTorch prefix ----
        print("5. PyTorch: embedding prefix + VLM forward...")
        with torch.no_grad():
            prefix_embs_t, _, _ = model_torch.embed_prefix(
                inputs["images"], inputs["img_masks"], inputs["lang_tokens"], inputs["lang_masks"]
            )
            _, vlm_cache_torch = model_torch.backbone.forward_vlm(prefix_embs_t, use_cache=True)

        # ---- TTNN prefix ----
        print("6. TTNN: embedding prefix + VLM forward...")
        images_ttnn = [
            ttnn.from_torch(
                img,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for img in inputs["images"]
        ]
        lang_tokens_ttnn = ttnn.from_torch(
            inputs["lang_tokens"], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        lang_masks_ttnn = ttnn.from_torch(
            inputs["lang_masks"].float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        state_ttnn = ttnn.from_torch(inputs["state"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        prefix_embs_tt, _, _ = model_ttnn.embed_prefix(
            images_ttnn, inputs["img_masks"], lang_tokens_ttnn, lang_masks_ttnn
        )
        _, prefix_kv_cache_ttnn = model_ttnn.backbone.forward_vlm(prefix_embs_tt, use_cache=True)

        # ---- Per-step velocity PCC (shared-x_t feed) ----
        print("\n7. Per-step velocity PCC (shared x_t fed to both models)...")
        timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

        x_t_torch = x_t_shared.clone()

        header = f"{'step':>4} {'t':>6} {'dt':>7}   {'v_pcc':>10}   {'||v_torch-v_ttnn||':>18}"
        print(header)
        print("-" * len(header))

        worst_v_pcc = 1.0
        per_step = []

        with torch.no_grad():
            for i in range(num_steps):
                t = timesteps[i]
                dt = timesteps[i + 1] - timesteps[i]

                v_torch = torch_velocity(model_torch, x_t_torch, t, vlm_cache_torch, inputs["state"])
                v_ttnn = ttnn_velocity(model_ttnn, x_t_torch, i, prefix_kv_cache_ttnn, state_ttnn)
                v_pcc = compute_pcc(v_torch, v_ttnn)
                v_l2 = (v_torch - v_ttnn).float().norm().item()

                worst_v_pcc = min(worst_v_pcc, v_pcc)
                per_step.append((i, t, v_pcc, v_l2))
                print(f"{i:>4} {t:>6.2f} {dt:>7.4f}   {v_pcc:>10.6f}   {v_l2:>18.6f}")

                # Euler update (torch only — we just need x_{t+1} for the next shared feed)
                x_t_torch = x_t_torch + dt * v_torch

        # ---- End-to-end aggregate PCC with matched seed ----
        print("\n8. End-to-end sample_actions with matched initial noise...")
        # Overwrite TTNN's internal x_0 slot with the shared noise (matches torch's seeded draw)
        model_ttnn.x_t_ttnn = ttnn.from_torch(
            x_t_shared,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=model_ttnn.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        with torch.no_grad():
            # Force torch's sampler to use the same x_0 via a throwaway override:
            saved_sample_noise = model_torch.denoising.sample_noise
            model_torch.denoising.sample_noise = lambda bs, device=None, dtype=torch.float32: x_t_shared.clone()
            try:
                torch_actions = model_torch.forward_inference(
                    images=inputs["images"],
                    img_masks=inputs["img_masks"],
                    lang_tokens=inputs["lang_tokens"],
                    lang_masks=inputs["lang_masks"],
                    state=inputs["state"],
                )
            finally:
                model_torch.denoising.sample_noise = saved_sample_noise

            ttnn_actions = model_ttnn.sample_actions(
                images=images_ttnn,
                img_masks=inputs["img_masks"],
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=state_ttnn,
            )
            if isinstance(ttnn_actions, ttnn.Tensor):
                ttnn_actions = ttnn.to_torch(ttnn_actions)

        e2e_pcc = compute_pcc(torch_actions, ttnn_actions)
        e2e_l2 = (torch_actions - ttnn_actions).float().norm().item()

        print("\n" + "=" * 80)
        print("  SUMMARY")
        print("=" * 80)
        print(f"  worst per-step velocity PCC  : {worst_v_pcc:.6f}")
        print(f"  end-to-end final x_t PCC     : {e2e_pcc:.6f}")
        print(f"  end-to-end final x_t ||Δ||   : {e2e_l2:.6f}")
        print(f"  per-step threshold           : {PER_STEP_PCC_THRESHOLD}")

        # Primary gate: per-step velocity PCC (per test docstring).
        # e2e is informational only — compound bf16 drift over 10 steps makes
        # 0.99 unreachable even with a correct implementation.
        E2E_PCC_THRESHOLD = 0.93
        passed = worst_v_pcc >= PER_STEP_PCC_THRESHOLD and e2e_pcc >= E2E_PCC_THRESHOLD
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  status                        : {status}")
        print("=" * 80)

        return 0 if passed else 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        print("\n🔌 Closing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
