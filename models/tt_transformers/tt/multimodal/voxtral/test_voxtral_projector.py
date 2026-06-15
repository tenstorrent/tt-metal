# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase-2 PCC gate for the Voxtral multimodal projector.

get_audio_features = audio_tower → reshape(-1, audio_intermediate_size=5120) → linear_1(5120→3072,
no bias) → gelu → linear_2(3072→3072, no bias). This isolates the projector: feed the reshaped HF
encoder output to a ttnn projector and gate vs the HF projector reference.
"""
import pytest
import torch
import transformers

import ttnn
from tests.ttnn.utils_for_testing import comp_pcc

MODEL_NAME = "mistralai/Voxtral-Mini-3B-2507"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_voxtral_projector(mesh_device):
    torch.manual_seed(0)
    full = transformers.VoxtralForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    proj = full.multi_modal_projector.eval()
    inter = full.config.audio_config.intermediate_size  # 5120

    # Encoder output is [B, 1500, 1280]; reshape(-1, 5120) groups 4 frames -> [B*375, 5120]
    audio_hidden = torch.randn(1, 1500, 1280, dtype=torch.float32) * 0.5
    feats = audio_hidden.reshape(-1, inter)
    with torch.no_grad():
        golden = proj(feats)

    w1 = ttnn.from_torch(
        proj.linear_1.weight.t().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    w2 = ttnn.from_torch(
        proj.linear_2.weight.t().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x = ttnn.from_torch(
        feats,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    h = ttnn.linear(x, w1, activation="gelu")
    h = ttnn.linear(h, w2)
    out = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[: golden.shape[0]]

    passed, pcc = comp_pcc(golden, out, 0.99)
    print(f"Voxtral projector PCC: {pcc}")
    assert passed, f"Voxtral projector PCC too low: {pcc}"
