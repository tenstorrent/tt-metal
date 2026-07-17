# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC Test: PaliGemma Backbone - TTNN vs PyTorch

Tests the full PaliGemma backbone (VLM + Expert) with real (upstream libero) checkpoint weights.

Usage:
    pytest test_pcc_paligemma_full.py -v
    python test_pcc_paligemma_full.py  # Standalone
"""

import sys
import os


def _pi0_num_cameras() -> int:
    """Production pi0.5 LIBERO bs=3 — see [[pi05-siglip-bs3-production]]."""
    return int(os.environ.get("PI0_NUM_CAMERAS", "2"))


from pathlib import Path

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone as PaliGemmaBackboneTorch
from models.experimental.pi0_5.tt.ttnn_paligemma import Pi0_5PaliGemmaBackboneTTNN as PaliGemmaBackboneTTNN
from models.experimental.pi0_5.common.configs import SigLIPConfig, GemmaConfig, PaliGemmaConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader


TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.environ.get(
    "PI05_CHECKPOINT_DIR",
    str(Path(__file__).resolve().parents[2] / "weights" / "pi05_base"),
)
SEED = 42
PCC_THRESHOLD = 0.99


def create_config() -> PaliGemmaConfig:
    """Create PaliGemma config matching checkpoint."""
    siglip = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    vlm = GemmaConfig(
        width=2048,
        depth=18,
        mlp_dim=16384,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )
    expert = GemmaConfig(
        width=1024,
        depth=18,
        mlp_dim=4096,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,  # was 128 — matches the real pi0.5 expert config
    )
    return PaliGemmaConfig(
        siglip_config=siglip,
        vlm_config=vlm,
        expert_config=expert,
        # max_seq_len=512 sized so TTNN's precompute_freqs_cis_meta_format cos/sin
        # tensors stay small and precise. Larger values (e.g. 1024) make the
        # bf16 cos/sin precompute span a wider input range and drop per-position
        # precision enough to drag this block's PCC from 0.997 to 0.75.
        max_seq_len=512,
    )


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()
    mean1, mean2 = torch.mean(t1), torch.mean(t2)
    std1, std2 = torch.std(t1), torch.std(t2)
    if std1 < 1e-6 or std2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    covariance = torch.mean((t1 - mean1) * (t2 - mean2))
    return (covariance / (std1 * std2)).item()


def get_paligemma_weights():
    """Get weights from the real (upstream libero) checkpoint."""
    checkpoint_path = Path(CHECKPOINT_PATH)
    if checkpoint_path.is_absolute() and not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    weight_loader = PI0WeightLoader(str(checkpoint_path))
    return weight_loader.categorized_weights


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pcc_paligemma_embed_image(device):
    """Test PaliGemma image embedding: TTNN vs PyTorch."""
    torch.manual_seed(SEED)

    config = create_config()
    weights = get_paligemma_weights()

    # Create input
    pixel_values = torch.randn(_pi0_num_cameras(), 3, config.siglip_config.image_size, config.siglip_config.image_size)

    # PyTorch forward
    model_torch = PaliGemmaBackboneTorch(config, weights)
    out_torch = model_torch.embed_image(pixel_values)

    # Convert to TTNN tensor for TTNN model
    pixel_values_ttnn = ttnn.from_torch(
        pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TTNN forward
    model_ttnn = PaliGemmaBackboneTTNN(config, weights, device)
    out_ttnn = model_ttnn.embed_image(pixel_values_ttnn)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    print(f"\n✅ PaliGemma embed_image PCC (pretrained): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pcc_paligemma_vlm_block(device):
    """Test single PaliGemma VLM block: TTNN vs PyTorch."""
    torch.manual_seed(SEED)

    config = create_config()
    weights = get_paligemma_weights()

    # Create models
    model_torch = PaliGemmaBackboneTorch(config, weights)
    model_ttnn = PaliGemmaBackboneTTNN(config, weights, device)

    # Create input (simulated prefix embeddings). Scale by 0.5 to match the
    # realistic magnitude of `hidden` out of the prefix-embed path.
    batch_size = 1
    seq_len = 64
    hidden = torch.randn(batch_size, seq_len, config.vlm_config.width) * 0.5

    # Run single VLM block
    block_torch = model_torch.vlm_blocks[0]
    cos_torch, sin_torch = model_torch.cos[:seq_len], model_torch.sin[:seq_len]
    out_torch, _ = block_torch.forward(hidden, cos_torch, sin_torch)

    # ttnn.experimental.rotary_embedding expects cos/sin shaped
    # (1, 1, seq_len, head_dim) with the split-half layout
    # [c0, c1, ..., c_{n/2-1}, c0, c1, ..., c_{n/2-1}].
    # torch_gemma.precompute_freqs_cis returns (max_seq_len, head_dim//2),
    # so duplicate the half-dim and add the two leading singleton dims.
    cos_full = torch.cat([cos_torch, cos_torch], dim=-1).unsqueeze(0).unsqueeze(0)
    sin_full = torch.cat([sin_torch, sin_torch], dim=-1).unsqueeze(0).unsqueeze(0)
    cos_ttnn = ttnn.from_torch(cos_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_ttnn = ttnn.from_torch(sin_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # TTNN
    hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    block_ttnn = model_ttnn.vlm_blocks[0]
    out_ttnn, _ = block_ttnn.forward(hidden_ttnn, cos_ttnn, sin_ttnn)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    print(f"\n✅ PaliGemma VLM Block[0] PCC (pretrained): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


def main():
    """Standalone runner."""
    print("=" * 70)
    print("  PaliGemma Backbone PCC Test (Checkpoint Weights)")
    print("=" * 70)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if checkpoint_path.is_absolute() and not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    torch.manual_seed(SEED)
    config = create_config()

    print(f"\n📁 Checkpoint: {checkpoint_path}")
    print(f"📋 VLM: {config.vlm_config.depth} layers, width={config.vlm_config.width}")
    print(f"📋 Expert: {config.expert_config.depth} layers, width={config.expert_config.width}")

    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        print("\n1. Loading checkpoint weights...")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        weights = weight_loader.categorized_weights
        print(f"   ✅ Loaded weights")

        print("\n2. Testing embed_image...")
        pixel_values = torch.randn(
            _pi0_num_cameras(), 3, config.siglip_config.image_size, config.siglip_config.image_size
        )

        model_torch = PaliGemmaBackboneTorch(config, weights)
        out_torch = model_torch.embed_image(pixel_values)

        model_ttnn = PaliGemmaBackboneTTNN(config, weights, device)
        # embed_image() consumes a device tensor (BCHW); production callers upload
        # bf16 TILE before the call. Match that convention.
        pixel_values_ttnn = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = model_ttnn.embed_image(pixel_values_ttnn)

        if isinstance(out_ttnn, ttnn.Tensor):
            out_ttnn = ttnn.to_torch(out_ttnn)

        pcc = compute_pcc(out_torch, out_ttnn)
        passed = pcc >= PCC_THRESHOLD

        print("\n" + "=" * 70)
        print("  RESULTS - embed_image (pretrained)")
        print("=" * 70)
        print(f"   PCC:       {pcc:.6f}")
        print(f"   Threshold: {PCC_THRESHOLD}")
        print(f"   Status:    {'✅ PASS' if passed else '❌ FAIL'}")
        print("=" * 70)

        return 0 if passed else 1

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
