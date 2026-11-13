# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Decode matmul geometry (input cores, readers per DRAM bank) only moves data: the result does not change."""

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.gemma4_31b_qb2.tt import decoder as decoder_module
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.model import WeightReader, checkpoint_path


@pytest.mark.parametrize("layer_index", [0, 5], ids=["sliding", "full"])
def test_o_projection_readers_per_bank_do_not_change_the_result(qb2_mesh, layer_index, monkeypatch):
    path = checkpoint_path()
    config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
    weights = WeightReader(path).layer(layer_index)
    cores, _, k_block = decoder_module.DECODE_GEOMETRY["o"]
    outputs = {}
    for readers in (1, 2):
        monkeypatch.setitem(decoder_module.DECODE_GEOMETRY, "o", (cores, readers, k_block))
        layer = Decoder.from_state_dict(weights, hf_config=config, layer_idx=layer_index, mesh_device=qb2_mesh)
        k = layer.decode_weights["o"].shape[-2]
        x = torch.randn(1, 1, 32, k, generator=torch.Generator().manual_seed(1)).bfloat16()
        device_x = ttnn.from_torch(
            x,
            device=qb2_mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(qb2_mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        y = layer._dlinear(device_x, "o")
        outputs[readers] = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(qb2_mesh, dim=0)).float()
    assert torch.equal(outputs[1], outputs[2])
