# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Localize numerical divergence between `ttnn_siglip` (device) and HF
`SiglipVisionModel` (host) on the upstream openpi pi05_libero weights.

This is the planning test for porting HF SigLIP semantics into the
device-side `ttnn_siglip.py` so the upstream checkpoint runs entirely on
Blackhole (without the `PI0_SIGLIP_HF=1` host bridge).

Usage:
    TT_METAL_HOME=/home/tt-admin/sdawle/pi0/tt-metal TT_VISIBLE_DEVICES=0 \
      PYTHONPATH=/home/tt-admin/sdawle/pi0/tt-metal:/storage/sdawle/openpi/src \
      python_env/bin/python -m pytest \
      models/experimental/pi0_5/tests/pcc/test_pcc_ttnn_siglip_vs_hf.py \
      -x -s --no-header
"""

import sys
import types as _types
from pathlib import Path

import pytest
import torch
import ttnn

# Bypass openpi's transformers_replace check
_fake = _types.ModuleType("transformers.models.siglip.check")
_fake.check_whether_transformers_replace_is_installed_correctly = lambda: True
sys.modules["transformers.models.siglip.check"] = _fake

UPSTREAM_CKPT = Path("/storage/sdawle/pi05_weights/pi05_libero_upstream")


def _hf_siglip_full_output(weights_vis: dict, pixel_values: torch.Tensor):
    """Run HF SiglipVisionModel on the host, return last_hidden_state (bf16)."""
    from models.experimental.pi0_5.reference.torch_siglip_hf import HFSigLIPVisionTower
    from models.experimental.pi0_5.common.configs import SigLIPConfig

    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    tower = HFSigLIPVisionTower(cfg, weights_vis)
    out = tower.forward(pixel_values.to(torch.float32))  # bf16 per the wrapper
    return out  # (B, 256, 1152)


def _ttnn_siglip_full_output(weights_vis: dict, pixel_values: torch.Tensor, device):
    """Run ttnn_siglip's SigLIPVisionTowerTTNN on device, return as torch bf16."""
    from models.experimental.pi0_5.tt.ttnn_siglip import SigLIPVisionTowerTTNN
    from models.experimental.pi0_5.common.configs import SigLIPConfig

    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    tower = SigLIPVisionTowerTTNN(cfg, weights_vis, device)

    # Upload pixel_values as ttnn (matches what PaliGemmaBackboneTTNN.embed_image does
    # in the BS=1 default path: bf16, TILE, DRAM).
    pix_ttnn = ttnn.from_torch(
        pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with torch.no_grad():
        out_ttnn = tower.forward(pix_ttnn)
    return ttnn.to_torch(out_ttnn)


@pytest.mark.skipif(
    not (UPSTREAM_CKPT / "model.safetensors").exists(),
    reason=f"upstream pi05_libero ckpt not found at {UPSTREAM_CKPT}",
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_ttnn_siglip_vs_hf_e2e(device):
    """First-cut diff: ttnn_siglip output vs HF on the same input + weights.

    Prints summary stats and the first three failing tokens so we can pin down
    whether the divergence is patch+pos, an early encoder layer, the post-LN,
    or accumulating gradually across the 27 layers.
    """
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader

    print(f"\n📋 Loading upstream weights from {UPSTREAM_CKPT}")
    loader = Pi0_5WeightLoader(str(UPSTREAM_CKPT))
    weights_vis = loader.categorized_weights["vlm_vision"]
    print(f"   {len(weights_vis)} vlm_vision keys")

    torch.manual_seed(0)
    pix = torch.randn(1, 3, 224, 224, dtype=torch.float32) * 0.1
    print(f"   input: {tuple(pix.shape)} std={pix.std().item():.3f}")

    print(f"\n🧪 Running HF SigLIP (host)")
    hf_out = _hf_siglip_full_output(weights_vis, pix).float()
    print(f"   hf_out: {tuple(hf_out.shape)} {hf_out.dtype}")
    print(f"           mean={hf_out.mean().item():+.5f} std={hf_out.std().item():.5f}")

    print(f"\n🧪 Running ttnn_siglip (device)")
    tt_out = _ttnn_siglip_full_output(weights_vis, pix, device).float()
    print(f"   tt_out: {tuple(tt_out.shape)} {tt_out.dtype}")
    print(f"           mean={tt_out.mean().item():+.5f} std={tt_out.std().item():.5f}")

    if hf_out.shape != tt_out.shape:
        pytest.fail(f"shape mismatch: hf {hf_out.shape} vs tt {tt_out.shape}")

    diff = (hf_out - tt_out).abs()
    per_tok = diff.max(dim=-1).values[0]
    print(f"\n📊 DIFF SUMMARY (HF − ttnn_siglip)")
    print(f"   overall max:  {diff.max().item():.5e}")
    print(f"   overall mean: {diff.mean().item():.5e}")
    print(f"   #tokens with max-per-tok > 0.1: {(per_tok > 0.1).sum().item()} / {per_tok.numel()}")
    print(f"   #tokens with max-per-tok > 0.5: {(per_tok > 0.5).sum().item()} / {per_tok.numel()}")
    worst3 = per_tok.topk(3).indices.tolist()
    for idx in worst3:
        print(f"     tok[{idx}]: hf  first8={hf_out[0, idx, :8].tolist()}")
        print(f"             tt  first8={tt_out[0, idx, :8].tolist()}")

    # PCC across the full output (Pearson correlation of flattened vectors)
    a = hf_out.flatten()
    b = tt_out.flatten()
    a_mean, b_mean = a.mean(), b.mean()
    pcc = ((a - a_mean) * (b - b_mean)).sum() / (
        ((a - a_mean) ** 2).sum().sqrt() * ((b - b_mean) ** 2).sum().sqrt() + 1e-12
    )
    print(f"   PCC: {pcc.item():.6f}")

    # No strict assertion yet — this is a diagnostic test. We just want to
    # see the numbers to plan the port.
    assert hf_out.shape == tt_out.shape


if __name__ == "__main__":
    # Allow direct invocation: python -u test_pcc_ttnn_siglip_vs_hf.py
    pytest.main([__file__, "-x", "-s"])
