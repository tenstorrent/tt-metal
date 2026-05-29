# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DIT (Diffusion Transformer) correctness test — compare one TTNN forward pass
against the PyTorch reference (HuggingFace ZImageTransformer2DModel).
"""

import pytest
import torch

import ttnn
from models.demos.z_image_turbo.tt.dit import model_pt
from models.demos.z_image_turbo.tt.dit.model_ttnn import ZImageTransformerTTNN

CAP_TOKENS = 128
IMG_LATENT_H = 64
IMG_LATENT_W = 64
LATENT_CHANNELS = 16

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _to_device_bf16(pt, mesh_device):
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_to_torch(tt_tensor, mesh_device):
    host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    return host[: host.shape[0] // 4].float()


def pcc(a, b):
    a_flat = a.flatten().double()
    b_flat = b.flatten().double()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = (a_centered * b_centered).sum()
    den = a_centered.norm() * b_centered.norm()
    return (num / den).item() if den > 0 else 0.0


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_dit_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()

    torch.manual_seed(42)
    latent = torch.randn(LATENT_CHANNELS, 1, IMG_LATENT_H, IMG_LATENT_W, dtype=torch.bfloat16)
    cap_feats = torch.randn(CAP_TOKENS, 2560, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], dtype=torch.bfloat16)

    # --- PyTorch reference ---
    pt_model = model_pt.load_model()
    model_pt.pad_heads(pt_model)
    pt_out = model_pt.forward(pt_model, [latent], timestep.float(), cap_feats)[0]
    del pt_model

    # --- TTNN ---
    tt_model = ZImageTransformerTTNN(mesh_device)
    tt_model.set_cap_feats(cap_feats.unsqueeze(0))

    tt_lat = _to_device_bf16(latent, mesh_device)
    tt_ts = _to_device_bf16(timestep, mesh_device)

    # Compile run (populates program cache)
    compile_out = tt_model._forward_impl([tt_lat], tt_ts)
    ttnn.synchronize_device(mesh_device)
    for t in compile_out:
        ttnn.deallocate(t, True)

    # Second run (from cache)
    tt_lat = _to_device_bf16(latent, mesh_device)
    tt_ts = _to_device_bf16(timestep, mesh_device)
    tt_out = tt_model._forward_impl([tt_lat], tt_ts)
    ttnn.synchronize_device(mesh_device)
    tt_result = _tt_to_torch(tt_out[0], mesh_device)

    correlation = pcc(pt_out.float(), tt_result)
    print(f"\nDIT PyTorch vs TTNN: PCC={correlation:.6f}")
    print(f"  PT  output: shape={pt_out.shape}, range=[{pt_out.float().min():.4f}, {pt_out.float().max():.4f}]")
    print(f"  TT  output: shape={tt_result.shape}, range=[{tt_result.min():.4f}, {tt_result.max():.4f}]")
    assert correlation > 0.998, f"PyTorch vs TTNN PCC too low: {correlation:.6f}"
