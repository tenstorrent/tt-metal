# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from transformers import AutoConfig
from transformers.models.siglip.modeling_siglip import SiglipMLP

import ttnn
from models.demos.siglip.compare import comp_pcc
from models.demos.siglip.reference.functional import SiglipMLP as FunctionalSiglipMLP
from models.demos.siglip.tt.mlp import TtSiglipMLP


def test_functional_mlp():
    config = AutoConfig.from_pretrained(os.getenv("HF_MODEL"))
    assert hasattr(
        config, "vision_config"
    ), f"Unexpected model config provided. Expected a vision_config field to be present in: {config}"
    config = config.vision_config

    reference_mlp = SiglipMLP(config=config)
    functional_mlp = FunctionalSiglipMLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size)

    # Copy weights from reference to functional implementation
    functional_mlp.fc1.weight.data = reference_mlp.fc1.weight.data.clone()
    functional_mlp.fc1.bias.data = reference_mlp.fc1.bias.data.clone()
    functional_mlp.fc2.weight.data = reference_mlp.fc2.weight.data.clone()
    functional_mlp.fc2.bias.data = reference_mlp.fc2.bias.data.clone()

    batch = 1
    seq_len = 4096
    expected_max_input_scale = 15
    expected_min_input_scale = -15

    # Generate random inputs scaled to expected range
    random_inputs = (
        torch.rand(batch, seq_len, config.hidden_size, dtype=reference_mlp.state_dict()["fc1.weight"].dtype)
        * (expected_max_input_scale - expected_min_input_scale)
    ) - ((expected_max_input_scale - expected_min_input_scale) / 2)

    reference_output = reference_mlp(random_inputs)
    functional_output = functional_mlp(random_inputs)

    comparison_result, pcc = comp_pcc(reference_output, functional_output)

    if comparison_result:
        print(f"✅ Functional SigLIP MLP passes with PCC: {pcc}")
    else:
        print(f"❌ Functional SigLIP MLP fails with PCC: {pcc}")
        assert False


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_ttnn_mlp(mesh_device):
    config = AutoConfig.from_pretrained(os.getenv("HF_MODEL"))
    assert hasattr(
        config, "vision_config"
    ), f"Unexpected model config provided. Expected a vision_config field to be present in: {config}"
    config = config.vision_config

    reference_mlp = SiglipMLP(config=config)

    batch = 1
    seq_len = 4096
    expected_max_input_scale = 15
    expected_min_input_scale = -15

    # Generate random inputs scaled to expected range
    random_inputs = (
        torch.rand(batch, seq_len, config.hidden_size, dtype=reference_mlp.state_dict()["fc1.weight"].dtype)
        * (expected_max_input_scale - expected_min_input_scale)
    ) - (
        (expected_max_input_scale - expected_min_input_scale) / 2
    )  # Random inputs scaled to range of first MLP inputs

    reference_output = reference_mlp(random_inputs)

    try:
        # Instantiate the TtSiglipMLP class
        ttnn_mlp = TtSiglipMLP(
            mesh_device=mesh_device,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            state_dict=reference_mlp.state_dict(),
            state_dict_prefix="",
            dtype=ttnn.bfloat16,
        )

        # Convert input to ttnn tensor
        ttnn_input = ttnn.from_torch(
            random_inputs,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = ttnn_mlp(ttnn_input)

        result = ttnn.to_torch(ttnn_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]

        ttnn.deallocate(ttnn_input)
        ttnn.deallocate(ttnn_output)

        comparison_result, pcc = comp_pcc(reference_output, result)

        if comparison_result:
            print(f"✅ TTNN SigLIP MLP passes with PCC: {pcc}")
        else:
            print(f"❌ TTNN SigLIP MLP fails with PCC: {pcc}")
            assert False

    finally:
        ttnn.close_mesh_device(mesh_device)
