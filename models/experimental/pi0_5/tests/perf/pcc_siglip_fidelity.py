# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full single-chip SigLIP vision tower PCC vs torch, for matmul-fidelity A/B.

Uses the COMPLETE SigLIPVisionTowerTTNN (all 27 layers + post_ln + projector)
on one chip — a stricter, more complete validation than the 4-chip sliced
bench. Runs on a 1x1 submesh carved from a 4x4 mesh (this host's open_device
path is broken by the split fabric).

Reports PCC for the current PI0_SIGLIP_MM_HIFI / PI0_SIGLIP_MM_FP32_DEST config.
Gate: must stay >= the repo threshold (0.90) AND we track vs HiFi2 baseline.

    source models/experimental/pi0_5/local_env.sh
    PI0_SIGLIP_MM_HIFI=0 PI0_SIGLIP_MM_FP32_DEST=0 \
      python_env/bin/python models/experimental/pi0_5/tests/perf/pcc_siglip_fidelity.py
"""

from __future__ import annotations

import os

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower as TorchTower
from models.experimental.pi0_5.tt.ttnn_siglip import SigLIPVisionTowerTTNN

CKPT = os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream")
SEED = 42
BS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
GATE = float(os.environ.get("PCC_GATE", "0.90"))


def _pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    m1, m2 = t1.mean(), t2.mean()
    s1, s2 = t1.std(), t2.std()
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (((t1 - m1) * (t2 - m2)).mean() / (s1 * s2)).item()


def main():
    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    loader = PI0WeightLoader(CKPT)
    vision_weights = loader.get_vlm_vision_weights()
    torch.manual_seed(SEED)
    pixel_values = torch.randn(BS, 3, cfg.image_size, cfg.image_size)

    model_torch = TorchTower(cfg, vision_weights)
    with torch.no_grad():
        out_torch = model_torch.forward(pixel_values)

    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4), l1_small_size=24576)
    sm = None
    try:
        sm = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        pv = ttnn.from_torch(
            pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sm, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        model = SigLIPVisionTowerTTNN(cfg, vision_weights, sm)
        out = model.forward(pv)
        if isinstance(out, ttnn.Tensor):
            out = ttnn.to_torch(out)
        pcc = _pcc(out_torch, out)
        cfg_str = (
            f"MM_HIFI={os.environ.get('PI0_SIGLIP_MM_HIFI','2')} "
            f"FP32_DEST={os.environ.get('PI0_SIGLIP_MM_FP32_DEST','1')} "
            f"ATTN={os.environ.get('PI0_SIGLIP_ATTN_HIFI','-')} "
            f"MLP={os.environ.get('PI0_SIGLIP_MLP_HIFI','-')}"
        )
        print(f"\n=== Full SigLIP tower PCC (bs={BS}) [{cfg_str}] ===")
        print(f"  shape {tuple(out.shape)}  PCC={pcc:.6f}  gate={GATE}")
        print(f"METRIC full_tower_pcc={pcc:.6f}")
        ok = pcc >= GATE
        print(f"  PCC_GATE_OK={ok}")
        if not ok:
            raise SystemExit(f"PCC {pcc:.6f} < gate {GATE}")
    finally:
        if sm is not None:
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)


if __name__ == "__main__":
    main()
