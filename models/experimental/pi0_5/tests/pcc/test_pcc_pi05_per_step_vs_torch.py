# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pi0.5 per-step PCC test.

The existing aggregate PCC test (test_pcc_pi05_model.py) only compares the final
denoised action tensor. This script extends that to per-denoise-step resolution
and runs the e2e comparison across multiple random seeds.

Three metrics are reported:

  1. Per-step velocity PCC (informational, first seed only):
     At each denoise step i, feed the SAME x_t_i (PyTorch's) into both models,
     and compare the velocities they produce. This isolates per-step accuracy
     without any cross-step error accumulation, and it matches the per-step
     semantics of the flow-matching Euler solver.

  2. End-to-end aggregate PCC, per seed:
     Call each model's real sample_actions() path with matched initial noise.
     Real TTNN keeps x_t on-device in bf16 throughout, so this is the correct
     way to observe compound drift — unlike a per-step host-driven loop, which
     would introduce extra bf16 round-tripping artifacts.

  3. End-to-end PCC distribution across N seeds (primary gate):
     The e2e PCC is highly seed-sensitive because flow-matching with random
     initial noise is a chaotic dynamical system: a 10-step Euler integration
     of a learned nonlinear velocity field amplifies tiny per-step bf16 drift
     differently for every input. Single-seed PCC has stdev ≈ 0.006 across
     seeds. We therefore gate on the mean e2e PCC across SEEDS, not a single
     draw — this is the implementation-quality signal.

Pass condition: mean(e2e_pcc over SEEDS) ≥ MEAN_E2E_PCC_THRESHOLD (0.95).
The per-step worst-PCC is reported but informational only.

Usage:
    PYTHONPATH=<root>/ttnn:<root> python test_pcc_pi05_per_step_vs_torch.py
    # or with a custom seed list:
    PI0_PCC_SEEDS="42,7,100" python test_pcc_pi05_per_step_vs_torch.py
"""

import os
import statistics
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model as PI0ModelTorch
from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN as PI0ModelTTNN
from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader


_REPO_ROOT = Path(__file__).resolve().parents[5]  # tt-metal repo root
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", str(_REPO_ROOT))
CHECKPOINT_PATH = os.environ.get("PI0_CHECKPOINT", str(Path(__file__).resolve().parents[2] / "weights" / "pi05_base"))
BATCH_SIZE = 1
SEED = 42  # used for per-step velocity diagnostics (first seed in the sweep)

# Seed sweep for the e2e PCC distribution. Override with PI0_PCC_SEEDS="42,7,100"
_DEFAULT_SEEDS = [42, 0, 1, 7, 13, 100, 2024, 31337, 9001, 314]
SEEDS = [int(s) for s in os.environ["PI0_PCC_SEEDS"].split(",")] if os.environ.get("PI0_PCC_SEEDS") else _DEFAULT_SEEDS
PER_STEP_PCC_THRESHOLD = 0.99  # informational: per-step velocity PCC target
MEAN_E2E_PCC_THRESHOLD = 0.95  # primary gate: mean e2e PCC across SEEDS


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
    LANG_SEQ_LEN = 256
    lang_tokens = torch.zeros(batch_size, LANG_SEQ_LEN, dtype=torch.int64)
    lang_tokens[:, :32] = torch.randint(0, 256000, (batch_size, 32))
    lang_masks = torch.zeros(batch_size, LANG_SEQ_LEN, dtype=torch.bool)
    lang_masks[:, :32] = True
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
    # Host-pad action_horizon to the next tile-aligned multiple of 32 so the
    # expert's sharded RMSNorm can run (matches the contract in
    # Pi0_5ModelTTNN.__init__ / sample_actions). The trailing padded rows
    # are zero and don't influence the action_horizon-sized output below.
    ah = x_t_torch.shape[1]
    ah_padded = ((ah + 31) // 32) * 32
    if ah_padded != ah:
        padded = torch.zeros(x_t_torch.shape[0], ah_padded, x_t_torch.shape[2], dtype=x_t_torch.dtype)
        padded[:, :ah, :] = x_t_torch
        x_t_torch = padded
    x_t_ttnn = ttnn.from_torch(
        x_t_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=model_ttnn.device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    t_tensor = model_ttnn._timestep_per_step_bs1[step_idx]

    # Match the PRODUCTION fast path used by sample_actions:
    #   - skip the device-side time-MLP (sincos + 2 linears + silu)
    #   - feed the host-precomputed per-(step,layer) modulations
    #   - feed the host-precomputed final-norm modulation
    # Without this, the per-step measurement reflects a DIFFERENT code path
    # (device-side adarms_cond) than the e2e measurement (precomputed mods).
    suffix_embs = model_ttnn.suffix_embedding.embed_actions(x_t_ttnn)
    precomputed_block_mods = model_ttnn._block_mods_per_step[step_idx]
    precomputed_final_mod = model_ttnn._final_mod_per_step[step_idx]
    expert_out, _ = model_ttnn.backbone.forward_expert(
        suffix_embs,
        past_key_values=prefix_kv_cache,
        adarms_cond=None,  # unused when precomputed_*_mod are supplied
        precomputed_block_mods=precomputed_block_mods,
        precomputed_final_mod=precomputed_final_mod,
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
    v_torch_out = ttnn.to_torch(velocity)
    # Crop padded rows back to the actual action_horizon for the PCC compare.
    return v_torch_out[:, :ah, :]


def _build_ttnn_inputs(inputs, device):
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
    return images_ttnn, lang_tokens_ttnn, lang_masks_ttnn, state_ttnn


def _run_per_step_diagnostics(model_torch, model_ttnn, device, config, inputs, x_t_shared):
    """Per-step velocity diagnostics for a single seed. Returns worst per-step PCC."""
    num_steps = config.num_denoising_steps

    with torch.no_grad():
        prefix_embs_t, _, _ = model_torch.embed_prefix(
            inputs["images"], inputs["img_masks"], inputs["lang_tokens"], inputs["lang_masks"]
        )
        _, vlm_cache_torch = model_torch.backbone.forward_vlm(prefix_embs_t, use_cache=True)

    images_ttnn, lang_tokens_ttnn, lang_masks_ttnn, state_ttnn = _build_ttnn_inputs(inputs, device)
    prefix_embs_tt, _, _ = model_ttnn.embed_prefix(images_ttnn, inputs["img_masks"], lang_tokens_ttnn, lang_masks_ttnn)
    _, prefix_kv_cache_ttnn = model_ttnn.backbone.forward_vlm(prefix_embs_tt, use_cache=True)

    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]
    x_t_torch = x_t_shared.clone()

    header = (
        f"{'step':>4} {'t':>6} {'dt':>7}   {'v_pcc':>10}   {'||Δv||':>10}   "
        f"{'||v_t||':>10}   {'||v_n||':>10}   {'scale':>8}"
    )
    print(header)
    print("-" * len(header))

    worst_v_pcc = 1.0
    with torch.no_grad():
        for i in range(num_steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            v_torch = torch_velocity(model_torch, x_t_torch, t, vlm_cache_torch, inputs["state"])
            v_ttnn = ttnn_velocity(model_ttnn, x_t_torch, i, prefix_kv_cache_ttnn, state_ttnn)
            v_pcc = compute_pcc(v_torch, v_ttnn)
            v_l2 = (v_torch - v_ttnn).float().norm().item()
            n_torch = v_torch.float().norm().item()
            n_ttnn = v_ttnn.float().norm().item()
            scale = n_ttnn / max(n_torch, 1e-9)
            cos = torch.nn.functional.cosine_similarity(
                v_torch.flatten().float().unsqueeze(0),
                v_ttnn.flatten().float().unsqueeze(0),
            ).item()
            inner = (v_torch.flatten().float() * v_ttnn.flatten().float()).sum().item()
            err_parallel = n_torch - inner / max(n_torch, 1e-9)
            err_perp = (v_l2**2 - err_parallel**2) ** 0.5 if v_l2 > abs(err_parallel) else 0.0

            worst_v_pcc = min(worst_v_pcc, v_pcc)
            print(
                f"{i:>4} {t:>6.2f} {dt:>7.4f}   {v_pcc:>10.6f}   {v_l2:>10.4f}   "
                f"{n_torch:>10.4f}   {n_ttnn:>10.4f}   {scale:>8.4f}    cos={cos:.6f}  "
                f"err‖={err_parallel:.3f}  err⊥={err_perp:.3f}"
            )
            x_t_torch = x_t_torch + dt * v_torch

    return worst_v_pcc


def _e2e_for_seed(seed, model_torch, model_ttnn, device, config):
    """Run a single-seed e2e comparison and return (pcc, l2, cos)."""
    # Fresh-RNG inputs for this seed. (Note: this means seed=42 here is NOT the
    # same input pattern as the legacy "seed=42 after PI0ModelTTNN ctor"
    # pattern — we now sweep many seeds and report the distribution, so a
    # single fragile seed-consumption order no longer matters.)
    torch.manual_seed(seed)
    inputs = create_test_inputs(config, batch_size=BATCH_SIZE)
    torch.manual_seed(seed)
    x_0 = torch.randn(BATCH_SIZE, config.action_horizon, config.action_dim, dtype=torch.float32)

    # Pad action_horizon up to the next tile multiple (50 → 64) so the TTNN
    # x_t_ttnn buffer matches the expert's internal sharded LN layout.
    ah = x_0.shape[1]
    ah_padded = ((ah + 31) // 32) * 32
    if ah_padded != ah:
        x_0_padded = torch.zeros(x_0.shape[0], ah_padded, x_0.shape[2], dtype=x_0.dtype)
        x_0_padded[:, :ah, :] = x_0
    else:
        x_0_padded = x_0

    images_ttnn, lang_tokens_ttnn, lang_masks_ttnn, state_ttnn = _build_ttnn_inputs(inputs, device)
    model_ttnn.x_t_ttnn = ttnn.from_torch(
        x_0_padded,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    model_ttnn.resample_noise = False

    with torch.no_grad():
        saved = model_torch.denoising.sample_noise
        model_torch.denoising.sample_noise = lambda bs, device=None, dtype=torch.float32: x_0.clone()
        try:
            torch_actions = model_torch.forward_inference(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )
        finally:
            model_torch.denoising.sample_noise = saved

        ttnn_actions = model_ttnn.sample_actions(
            images=images_ttnn,
            img_masks=inputs["img_masks"],
            lang_tokens=lang_tokens_ttnn,
            lang_masks=lang_masks_ttnn,
            state=state_ttnn,
        )
        if isinstance(ttnn_actions, ttnn.Tensor):
            ttnn_actions = ttnn.to_torch(ttnn_actions)
        ttnn_actions = ttnn_actions[:, : config.action_horizon, : config.action_dim]

    pcc = compute_pcc(torch_actions, ttnn_actions)
    l2 = (torch_actions - ttnn_actions).float().norm().item()
    cos = torch.nn.functional.cosine_similarity(
        torch_actions.flatten().float().unsqueeze(0),
        ttnn_actions.flatten().float().unsqueeze(0),
    ).item()
    return pcc, l2, cos, inputs, x_0


def main():
    print("=" * 80)
    print("  PI0.5 PER-STEP + SEED-SWEEP PCC TEST")
    print("=" * 80)

    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        print(f"❌ Checkpoint not found: {ckpt}")
        return 1
    print(f"📁 Checkpoint: {ckpt}")
    print(f"🌱 Seeds:      {SEEDS}")

    print("🔌 Opening TTNN device...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    grid = device.compute_with_storage_grid_size()
    print(f"   Grid: {grid.x}x{grid.y}")

    try:
        config = create_pi05_config()
        num_steps = config.num_denoising_steps
        print(f"📋 pi05={config.pi05}, num_steps={num_steps}, batch_size={BATCH_SIZE}")

        print("1. Loading weights...")
        weight_loader = PI0WeightLoader(str(ckpt))

        print("2. Initializing PyTorch reference...")
        model_torch = PI0ModelTorch(config, weight_loader)

        print("3. Initializing TTNN model (seeded for ctor reproducibility)...")
        torch.manual_seed(SEEDS[0])
        model_ttnn = PI0ModelTTNN(config, weight_loader, device)

        # ---- Per-step velocity diagnostics for the first seed only ----
        first_seed = SEEDS[0]
        print(f"\n4. Per-step velocity diagnostics (seed={first_seed}, informational)...")
        torch.manual_seed(first_seed)
        diag_inputs = create_test_inputs(config, batch_size=BATCH_SIZE)
        torch.manual_seed(first_seed)
        x_t_shared = torch.randn(BATCH_SIZE, config.action_horizon, config.action_dim, dtype=torch.float32)
        worst_v_pcc = _run_per_step_diagnostics(model_torch, model_ttnn, device, config, diag_inputs, x_t_shared)

        # ---- E2E sweep across SEEDS ----
        print(f"\n5. E2E PCC across {len(SEEDS)} seed(s)...")
        print(f"{'seed':>10}   {'e2e_pcc':>10}   {'cos':>10}   {'L2':>10}")
        print("-" * 50)
        e2e_pccs = []
        for s in SEEDS:
            pcc, l2, cos, _, _ = _e2e_for_seed(s, model_torch, model_ttnn, device, config)
            e2e_pccs.append(pcc)
            print(f"{s:>10}   {pcc:>10.6f}   {cos:>10.6f}   {l2:>10.4f}")

        mean_pcc = statistics.mean(e2e_pccs)
        median_pcc = statistics.median(e2e_pccs)
        stdev_pcc = statistics.stdev(e2e_pccs) if len(e2e_pccs) > 1 else 0.0
        min_pcc = min(e2e_pccs)
        max_pcc = max(e2e_pccs)

        print("\n" + "=" * 80)
        print("  SUMMARY")
        print("=" * 80)
        print(
            f"  Per-step velocity PCC (worst, seed={first_seed}) : {worst_v_pcc:.6f}   "
            f"(threshold {PER_STEP_PCC_THRESHOLD}, informational)"
        )
        print(f"  E2E PCC distribution across {len(SEEDS)} seed(s):")
        print(f"    mean   : {mean_pcc:.6f}")
        print(f"    median : {median_pcc:.6f}")
        print(f"    stdev  : {stdev_pcc:.6f}")
        print(f"    min    : {min_pcc:.6f}")
        print(f"    max    : {max_pcc:.6f}")
        print(
            f"  Primary gate (mean ≥ {MEAN_E2E_PCC_THRESHOLD}): "
            f"{'✅ PASS' if mean_pcc >= MEAN_E2E_PCC_THRESHOLD else '❌ FAIL'}"
        )
        print("=" * 80)

        return 0 if mean_pcc >= MEAN_E2E_PCC_THRESHOLD else 1

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
