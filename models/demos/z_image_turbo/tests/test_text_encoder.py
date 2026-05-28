# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Text encoder correctness test — compare TTNN forward pass against
the PyTorch reference (HuggingFace Qwen3Model).
"""

import pytest
import torch

import ttnn
from models.demos.z_image_turbo.tt.text_encoder import model_pt
from models.demos.z_image_turbo.tt.text_encoder.model_ttnn import TextEncoderTTNN

CAP_TOKENS = 128
DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _to_device_int32(pt, mesh_device):
    return ttnn.from_torch(
        pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
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
def test_text_encoder_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()

    input_ids = model_pt.tokenize("a beautiful sunset over the ocean")

    # --- PyTorch reference ---
    pt_model = model_pt.load_model()
    pt_result = model_pt.forward(pt_model, input_ids)
    del pt_model

    # --- TTNN ---
    tt_model = TextEncoderTTNN(mesh_device, seq_len=CAP_TOKENS)

    # Compile run
    tt_ids = _to_device_int32(input_ids, mesh_device)
    tt_out = tt_model(tt_ids)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(tt_out, True)
    ttnn.deallocate(tt_ids, True)

    # Second run (from cache)
    tt_ids = _to_device_int32(input_ids, mesh_device)
    tt_out = tt_model(tt_ids)
    ttnn.synchronize_device(mesh_device)
    tt_result = _tt_to_torch(tt_out, mesh_device)

    correlation = pcc(pt_result, tt_result)
    print(f"\nText encoder PyTorch vs TTNN: PCC={correlation:.6f}")
    print(f"  PT  output: shape={pt_result.shape}, range=[{pt_result.min():.4f}, {pt_result.max():.4f}]")
    print(f"  TT  output: shape={tt_result.shape}, range=[{tt_result.min():.4f}, {tt_result.max():.4f}]")
    assert correlation > 0.986, f"PyTorch vs TTNN PCC too low: {correlation:.6f}"
