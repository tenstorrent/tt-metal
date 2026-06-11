# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage PCC + mesh-carve smoke tests for the BH-Galaxy host-bounce pipeline.

Run a single test:
    pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py::test_mesh_carve_smoke
"""

from __future__ import annotations

import torch
import ttnn

import os
from pathlib import Path

import pytest

from models.experimental.pi0_5.tt.tt_bh_glx import stages
from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.tt_bh_glx.stage_vision import StageVision
from models.experimental.pi0_5.tt.tt_bh_glx.transport import send_via_host


CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


def test_mesh_carve_smoke():
    """Open the parent 8x4 mesh, carve 4/18/6 submeshes + per-chip 1x1s, host-bounce a probe tensor."""
    with open_galaxy_mesh(l1_small_size=24576) as h:
        assert h.parent.get_num_devices() == 32
        assert h.vision_submesh.get_num_devices() == stages.VISION_NUM_CHIPS
        assert h.prefill_submesh.get_num_devices() == stages.PREFILL_NUM_CHIPS
        assert h.denoise_submesh.get_num_devices() == stages.DENOISE_NUM_CHIPS
        assert len(h.vision_per_chip) == stages.VISION_NUM_CHIPS
        assert len(h.prefill_per_chip) == stages.PREFILL_NUM_CHIPS
        assert len(h.denoise_per_chip) == stages.DENOISE_NUM_CHIPS
        for sm in h.vision_per_chip + h.prefill_per_chip + h.denoise_per_chip:
            assert sm.get_num_devices() == 1

        # Host-bounce: tile-aligned probe tensor across the three stage boundaries.
        probe = torch.randn(1, 32, 32, dtype=torch.float32)
        x0 = ttnn.from_torch(
            probe,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=h.vision_per_chip[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x1 = send_via_host(x0, h.vision_per_chip[3])
        x2 = send_via_host(x1, h.prefill_per_chip[0])
        x3 = send_via_host(x2, h.denoise_per_chip[0])
        out = ttnn.to_torch(x3)
        assert out.shape == probe.shape
        # bf16 round-trip — loose tolerance suffices.
        assert torch.allclose(out.float(), probe, atol=1e-2)


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)
def test_vision_stage_pcc():
    """4-chip SigLIP slices + mm_projector vs torch reference. Target PCC ≥ 0.997."""
    from models.experimental.pi0_5.common.configs import SigLIPConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower as TorchSigLIPVisionTower
    from models.experimental.pi0_5.reference.torch_siglip import MultiModalProjector as TorchMMProjector

    # Match the single-chip default config (verified vs pi05_libero_upstream).
    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )

    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    vision_w = loader.categorized_weights["vlm_vision"]
    projector_w = loader.categorized_weights["vlm_projector"]

    bs = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
    torch.manual_seed(SEED)
    pixel_values = torch.randn(bs, 3, cfg.image_size, cfg.image_size)

    # Torch reference: SigLIP vision tower + mm_projector → (B, 256, 2048).
    ref_tower = TorchSigLIPVisionTower(cfg, vision_w)
    ref_proj = TorchMMProjector(projector_w)
    with torch.no_grad():
        ref_feat = ref_tower.forward(pixel_values)
        ref_out = ref_proj.forward(ref_feat)

    # TTNN sliced pipeline on 4 chips.
    with open_galaxy_mesh(l1_small_size=24576) as h:
        stage = StageVision(cfg, loader.categorized_weights, h)
        out_ttnn = stage.run(pixel_values)
        out = ttnn.to_torch(out_ttnn)

    assert out.shape == ref_out.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(ref_out.shape)}"
    pcc = _compute_pcc(ref_out, out)
    print(f"\n✅ Vision stage PCC vs torch: {pcc:.6f}  (shape {tuple(out.shape)})")
    assert pcc >= 0.997, f"PCC {pcc:.6f} < 0.997"


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)
def test_prefill_stage_pcc():
    """18-chip VLM prefill chain vs torch PaliGemmaBackbone.forward_vlm. Target PCC ≥ 0.99."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone as TorchBackbone
    from models.experimental.pi0_5.tt.tt_bh_glx.stage_prefill import StagePrefill

    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR), num_denoising_steps=5)
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    weights = loader.categorized_weights

    # Modest tile-aligned prefix length keeps the 18-chip chain fast.
    B = 1
    seq_len = 128
    torch.manual_seed(SEED)
    prefix_embs = torch.randn(B, seq_len, cfg.vlm_config.width) * 0.5

    # Torch reference (full 18-block chain + final RMS norm).
    ref = TorchBackbone(cfg, weights)
    with torch.no_grad():
        ref_out, _ = ref.forward_vlm(prefix_embs, attention_mask=None, position_ids=None, use_cache=False)

    with open_galaxy_mesh(l1_small_size=24576) as h:
        stage = StagePrefill(cfg, weights, h)
        prefix_ttnn = ttnn.from_torch(
            prefix_embs,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=h.prefill_per_chip[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_ttnn, _ = stage.run(prefix_ttnn, attention_mask=None, position_ids=None)
        out = ttnn.to_torch(out_ttnn)

    assert out.shape == ref_out.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(ref_out.shape)}"
    pcc = _compute_pcc(ref_out, out)
    print(f"\n✅ Prefill stage PCC vs torch: {pcc:.6f}  (shape {tuple(out.shape)})")
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"
