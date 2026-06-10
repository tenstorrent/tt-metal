# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 real-HF-weights PCC harness for dots.ocr (integration skill).

One pytest case per block, parametrized; each block helper loads real
checkpoint weights through tt/weight_loader.py, runs the pure-PyTorch
reference on a production-distribution input, runs the TTNN block at the
production operating point (fp32, 1x4 mesh), and returns ONE float PCC.
Rows are appended per real_weights tick — vision_patch_embed first.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import ttnn

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


wl = _load_module("dots_ocr_weight_loader", MODEL_DIR / "tt" / "weight_loader.py")
ref = _load_module("dots_ocr_reference_functional", MODEL_DIR / "reference" / "functional.py")
_patch_embed_mod = _load_module("dots_ocr_tt_vision_patch_embed", MODEL_DIR / "tt" / "vision_patch_embed.py")
_vision_rmsnorm_mod = _load_module("dots_ocr_tt_vision_rmsnorm", MODEL_DIR / "tt" / "vision_rmsnorm.py")


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _t_vision_patch_embed(mesh_device) -> tuple[float, int]:
    sd = wl.vision_patch_embed_weights()
    # Production-distribution input: the HF preprocessor emits normalized,
    # pre-flattened patches. Reuse the golden's real preprocessed image patches.
    golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_patch_embed.pt")
    x = golden["input"]  # [num_patches, C*P*P]
    ref_out = ref.vision_patch_embed_forward(x, sd)

    block = _patch_embed_mod.TtVisionPatchEmbed(mesh_device, sd, dtype=ttnn.float32)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = block.forward(x_tt)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()
    assert out.shape == ref_out.shape, f"{out.shape} != {ref_out.shape}"
    return _pcc(ref_out, out), wl.count_params(sd)


def _t_vision_rmsnorm(mesh_device) -> tuple[float, int]:
    # Real weights from three sites (blocks.0.norm1, blocks.41.norm2,
    # post_trunk_norm) exercise the loader across the per-layer index and the
    # tower-level key; PCC is gated on each, the min is reported.
    # Production-distribution input: the golden's real residual-stream
    # activation (RMSNorm is its own first op, so a real activation suffices).
    golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_rmsnorm.pt")
    x, eps = golden["input"], golden["eps"]
    sites = [(0, "norm1"), (wl.VISION_NUM_BLOCKS - 1, "norm2"), (0, "post_trunk_norm")]
    pccs, n_params = [], 0
    for layer_idx, which in sites:
        sd = wl.vision_rmsnorm_weights(layer_idx=layer_idx, which=which)
        n_params += wl.count_params(sd)
        ref_out = ref.vision_rmsnorm_forward(x, sd["weight"], eps=eps)
        # Production operating point: the vision tower runs an fp32 residual
        # stream with fp32 TILE gammas (vision_transformer ttnn phase).
        block = _vision_rmsnorm_mod.TtVisionRMSNorm(mesh_device, sd, dtype=ttnn.float32, eps=eps)
        x_tt = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out = ttnn.to_torch(ttnn.get_device_tensors(block.forward(x_tt))[0]).float()
        assert out.shape == ref_out.shape, f"{which}: {out.shape} != {ref_out.shape}"
        pcc = _pcc(ref_out, out)
        print(f"  vision_rmsnorm[{which}@{layer_idx}] PCC = {pcc:.6f}")
        pccs.append(pcc)
    return min(pccs), n_params


BLOCKS = [
    ("vision_patch_embed", _t_vision_patch_embed),
    ("vision_rmsnorm", _t_vision_rmsnorm),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("name,fn", BLOCKS, ids=[b[0] for b in BLOCKS])
def test_real_hf_weights(name, fn, mesh_device):
    pcc, n_params = fn(mesh_device)
    print(f"[{name}] real-HF PCC = {pcc:.6f} ({n_params} params loaded)")
    assert pcc > 0.99, f"{name}: PCC {pcc:.6f} <= 0.99"
