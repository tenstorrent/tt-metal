# models/demos/deepseek_v3/tests/test_embedding_1d.py
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

# Import from local reference files instead of HuggingFace
from torch.nn import Embedding

import ttnn
from models.demos.deepseek_v3.tt.embedding_1d import Embedding1D
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    load_reference_io_tensors_for_module,
    load_state_dict,
    run_module_forward,
)


@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 1),  # Single token decode
        ("decode", 32),  # Batch decode
        ("prefill", 128),  # Short prefill
        ("prefill", 512),  # Medium prefill
        ("prefill", 2048),  # Long prefill
    ],
)
@pytest.mark.parametrize(
    "generate_reference_io",
    [True, False],
)
def test_embedding_forward_pass(
    hf_config,
    mode,
    seq_len,
    generate_reference_io,
    tmp_path,
    mesh_device,
    model_path,
):
    module_path = "model.embed_tokens"

    if generate_reference_io:
        reference_model = Embedding(
            hf_config.vocab_size,
            hf_config.hidden_size,
            hf_config.pad_token_id,
        ).eval()
        state_dict = reference_model.state_dict()

        torch_input = torch.randint(0, min(1000, hf_config.vocab_size), (1, 1, seq_len))
        reference_output = reference_model(torch_input)
    else:
        state_dict = load_state_dict(model_path, module_path)
        torch_input, reference_output = load_reference_io_tensors_for_module(mode, module_path, seq_len, 1)

    # Generate module configs and state
    weight_config = Embedding1D.convert_weights(hf_config, state_dict, tmp_path, mesh_device)
    model_config = get_model_config(Embedding1D, mode, hf_config, mesh_device)
    model_state = Embedding1D.create_state(hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    tt_input_ids = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # TTNN forward pass
    tt_output = run_module_forward(Embedding1D, mode, tt_input_ids, run_config)

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            dims=(0, -1),
            mesh_shape=tuple(mesh_device.shape),
        ),
    )[:1, ...]

    # Cleanup
    ttnn.deallocate(tt_input_ids)
    ttnn.deallocate(tt_output)

    # Check PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
