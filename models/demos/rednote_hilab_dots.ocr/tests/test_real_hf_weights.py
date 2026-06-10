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


BLOCKS = [
    ("vision_patch_embed", _t_vision_patch_embed),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("name,fn", BLOCKS, ids=[b[0] for b in BLOCKS])
def test_real_hf_weights(name, fn, mesh_device):
    pcc, n_params = fn(mesh_device)
    print(f"[{name}] real-HF PCC = {pcc:.6f} ({n_params} params loaded)")
    assert pcc > 0.99, f"{name}: PCC {pcc:.6f} <= 0.99"
