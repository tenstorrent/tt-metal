# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Adapter drop-in PCC tests for the tt_pipeline StageDenoise (plan §11.5).

  * Topology raise: requesting n_submeshes=8 on the Galaxy drop-in path raises ValueError.
  * Drop-in equivalence (expert chain): on a standalone N-submesh parent, build a bare
    expert chain via build_expert_only_pipeline and assert it returns the raw post-expert
    hidden PCC >= 0.95 vs the torch ``Pi0_5PaliGemmaBackbone.forward_expert`` (NO embed /
    final / project) -- proving the byte-faithful embed-bypass / raw-hidden contract.

Mesh: opens its OWN fresh >=N-chip parent under FABRIC_1D with a trace region (standalone
path). Skips cleanly without hardware or a checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/ttuser/salnahari/pi05_weights/pi05_base"))
SEED = 42
_TRACE_REGION = 134_217_728


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1, t2 = a.flatten().float(), b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


def _open_parent(n):
    """Open a fresh >=n-chip parent mesh under FABRIC_1D with a trace region."""
    n_dev = ttnn.get_num_devices()
    if n_dev < n:
        pytest.skip(f"need >={n} chips, have {n_dev}")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, n), l1_small_size=24576, trace_region_size=_TRACE_REGION
    )
    return parent


def _close_parent(parent):
    try:
        ttnn.close_mesh_device(parent)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _mk_config():
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig

    return Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR), num_denoising_steps=5)


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(), reason=f"checkpoint not found at {CHECKPOINT_DIR}"
)
def test_galaxy_dropin_path_n8_raises():
    """Requesting n_submeshes=8 on the Galaxy drop-in path raises the clear ValueError."""
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_pipeline import StageDenoise

    cfg = _mk_config()
    weights = Pi0_5WeightLoader(str(CHECKPOINT_DIR)).categorized_weights

    class _MH:
        class _M:
            shape = (6, 1)

        denoise_submesh = _M()

    with pytest.raises(ValueError, match="only n_submeshes=4"):
        StageDenoise(cfg, weights, _MH(), n_submeshes=8)


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(), reason=f"checkpoint not found at {CHECKPOINT_DIR}"
)
def test_expert_chain_dropin_equivalence():
    """run_expert_chain raw hidden PCC >= 0.95 vs torch forward_expert (standalone n=6)."""
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone as TorchBackbone
    from models.experimental.pi0_5.tt.tt_pipeline import StageDenoise

    cfg = _mk_config()
    weights = Pi0_5WeightLoader(str(CHECKPOINT_DIR)).categorized_weights
    ec = cfg.expert_config
    B = 1
    suffix_len = ((cfg.action_horizon + 31) // 32) * 32
    prefix_len = 288  # tile-aligned (9*32); the synthetic VLM prefix length

    torch.manual_seed(SEED)
    suffix_hidden = torch.randn(B, suffix_len, ec.width) * 0.5
    adarms_cond = torch.randn(B, ec.width) * 0.5
    prefix_kv_torch = [
        (
            torch.randn(B, ec.num_kv_heads, prefix_len, ec.head_dim) * 0.5,
            torch.randn(B, ec.num_kv_heads, prefix_len, ec.head_dim) * 0.5,
        )
        for _ in range(ec.depth)
    ]

    ref = TorchBackbone(cfg, weights)
    with torch.no_grad():
        ref_out, _ = ref.forward_expert(
            suffix_hidden, adarms_cond=adarms_cond, past_key_values=prefix_kv_torch, use_cache=False
        )

    parent = _open_parent(6)
    try:
        # Standalone n=6 so the chip->stage identity holds (3 layers/chip).
        stage = StageDenoise(cfg, weights, None, parent_mesh=parent, n_submeshes=6, splits=(3, 3, 3, 3, 3, 3))
        # Re-pack the synthetic torch KV into the Galaxy 6x3 per-chip device layout the adapter expects.
        n_per = 3
        prefix_kv_per_chip = []
        for c in range(6):
            chip_kv = []
            for j in range(n_per):
                kt, vt = prefix_kv_torch[c * n_per + j]
                sm = stage._submeshes[c]
                k_dst = ttnn.from_torch(kt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=sm)
                v_dst = ttnn.from_torch(vt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=sm)
                chip_kv.append((k_dst, v_dst))
            prefix_kv_per_chip.append(chip_kv)
        hidden_on_chip0 = ttnn.from_torch(
            suffix_hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=stage._submeshes[0]
        )
        adarms_per_chip = [
            ttnn.from_torch(adarms_cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=stage._submeshes[i])
            for i in range(6)
        ]
        out_ttnn = stage.run_expert_chain(hidden_on_chip0, adarms_per_chip, prefix_kv_per_chip)
        out = ttnn.to_torch(out_ttnn)
        stage.close()
    finally:
        _close_parent(parent)

    assert out.shape == ref_out.shape, f"shape mismatch {tuple(out.shape)} vs {tuple(ref_out.shape)}"
    pcc = _compute_pcc(ref_out, out)
    print(f"\nexpert-chain drop-in PCC vs torch forward_expert: {pcc:.6f}")
    assert pcc >= 0.95, f"PCC {pcc:.6f} < 0.95"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
