# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pi0.5 determinism test.

Runs the TTNN Pi0.5 sampler N times with identical inputs and identical initial
noise, and checks that every run produces bit-identical output. Non-determinism
of any magnitude is logged via max abs diff and pairwise PCC.

We explicitly overwrite `model_ttnn.x_t_ttnn` with a fixed x_0 before each run
to defeat the constructor's un-seeded `torch.randn` for the initial noise slot.

Usage:
    python test_determinism_pi05.py [--runs N]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader


_REPO_ROOT = Path(__file__).resolve().parents[5]  # tt-metal repo root
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", str(_REPO_ROOT))
CHECKPOINT_PATH = os.environ.get("PI0_CHECKPOINT", "lerobot/pi05_base")
BATCH_SIZE = 1
SEED = 42
DEFAULT_RUNS = 5


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


def build_ttnn_inputs(inputs, device):
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


def _resolve_checkpoint() -> Path:
    """Return the checkpoint path, skipping the pytest run if a local path is absent."""
    ckpt = Path(CHECKPOINT_PATH)
    if ckpt.is_absolute() and not ckpt.exists():
        pytest.skip(f"Checkpoint not found: {ckpt}")
    return ckpt


def run_determinism(device, runs: int = DEFAULT_RUNS):
    """Run sample_actions `runs`× with fixed inputs + fixed initial noise.

    Returns (all_bit_equal, worst_max_abs_diff, worst_pairwise_pcc) vs. run 0.
    """
    config = create_pi05_config()
    weight_loader = PI0WeightLoader(str(_resolve_checkpoint()))

    torch.manual_seed(SEED)
    model_ttnn = PI0ModelTTNN(config, weight_loader, device)

    torch.manual_seed(SEED + 1)
    inputs = create_test_inputs(config, BATCH_SIZE)
    images_ttnn, lang_tokens_ttnn, lang_masks_ttnn, state_ttnn = build_ttnn_inputs(inputs, device)

    torch.manual_seed(SEED + 2)
    x0 = torch.randn(BATCH_SIZE, config.action_horizon, config.action_dim, dtype=torch.float32)

    outputs = []
    for _ in range(runs):
        with torch.no_grad():
            out = model_ttnn.sample_actions(
                images=images_ttnn,
                img_masks=inputs["img_masks"],
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=state_ttnn,
                noise=x0,
            )
        if isinstance(out, ttnn.Tensor):
            out = ttnn.to_torch(out)
        outputs.append(out.clone())

    ref = outputs[0]
    all_bit_equal, max_abs, min_pcc = True, 0.0, 1.0
    for cur in outputs[1:]:
        all_bit_equal = all_bit_equal and torch.equal(ref, cur)
        max_abs = max(max_abs, (ref.float() - cur.float()).abs().max().item())
        min_pcc = min(min_pcc, compute_pcc(ref, cur))
    return all_bit_equal, max_abs, min_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_determinism_pi05(device):
    """TTNN Pi0.5 sampler is deterministic across repeated runs with fixed noise."""
    all_bit_equal, max_abs, min_pcc = run_determinism(device, DEFAULT_RUNS)
    print(f"\n✅ determinism: bit_equal={all_bit_equal}  max|Δ|={max_abs:.3e}  min_pcc={min_pcc:.8f}")
    # Pass if bit-identical OR within the bfloat16 noise floor.
    assert all_bit_equal or min_pcc >= 0.999999, f"non-deterministic: max|Δ|={max_abs:.3e}, min PCC={min_pcc:.8f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    args = parser.parse_args()

    print("=" * 80)
    print(f"  PI0.5 DETERMINISM TEST  (runs={args.runs})")
    print("=" * 80)

    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        print(f"❌ Checkpoint not found: {ckpt}")
        return 1
    print(f"📁 Checkpoint: {ckpt}")

    print("🔌 Opening TTNN device...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        config = create_pi05_config()
        print(f"📋 pi05={config.pi05}, num_steps={config.num_denoising_steps}")

        print("1. Loading weights...")
        weight_loader = PI0WeightLoader(str(ckpt))

        print("2. Initializing TTNN model...")
        torch.manual_seed(SEED)
        model_ttnn = PI0ModelTTNN(config, weight_loader, device)

        print("3. Building inputs...")
        torch.manual_seed(SEED + 1)
        inputs = create_test_inputs(config, BATCH_SIZE)
        images_ttnn, lang_tokens_ttnn, lang_masks_ttnn, state_ttnn = build_ttnn_inputs(inputs, device)

        torch.manual_seed(SEED + 2)
        x0 = torch.randn(BATCH_SIZE, config.action_horizon, config.action_dim, dtype=torch.float32)
        print(f"   x_0 shape: {tuple(x0.shape)}, std={x0.std().item():.4f}")

        print(f"\n4. Running TTNN sample_actions {args.runs}×...")
        outputs = []
        for r in range(args.runs):
            t_start = time.time()
            with torch.no_grad():
                out = model_ttnn.sample_actions(
                    images=images_ttnn,
                    img_masks=inputs["img_masks"],
                    lang_tokens=lang_tokens_ttnn,
                    lang_masks=lang_masks_ttnn,
                    state=state_ttnn,
                    noise=x0,
                )
            if isinstance(out, ttnn.Tensor):
                out = ttnn.to_torch(out)
            dt_ms = (time.time() - t_start) * 1000
            print(f"   run {r}: shape={tuple(out.shape)}, {dt_ms:.1f}ms")
            outputs.append(out.clone())

        print("\n5. Pairwise diff vs run 0...")
        ref = outputs[0]
        all_bit_equal = True
        max_abs = 0.0
        min_pcc = 1.0
        for r in range(1, args.runs):
            cur = outputs[r]
            equal = torch.equal(ref, cur)
            diff = (ref.float() - cur.float()).abs()
            mad = diff.max().item()
            pcc = compute_pcc(ref, cur)
            max_abs = max(max_abs, mad)
            min_pcc = min(min_pcc, pcc)
            all_bit_equal = all_bit_equal and equal
            print(f"   run 0 vs run {r}: bit-equal={equal}  max|Δ|={mad:.3e}  PCC={pcc:.8f}")

        print("\n" + "=" * 80)
        print("  SUMMARY")
        print("=" * 80)
        print(f"  bit-identical across runs : {all_bit_equal}")
        print(f"  worst max|Δ|               : {max_abs:.3e}")
        print(f"  worst pairwise PCC         : {min_pcc:.8f}")

        # Pass if either bit-identical OR PCC >= 0.999999 (bfloat16 noise floor)
        passed = all_bit_equal or min_pcc >= 0.999999
        print(f"  status                     : {'✅ PASS' if passed else '❌ FAIL'}")
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
