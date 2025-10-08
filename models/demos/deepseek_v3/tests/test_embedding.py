# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

# Import from local reference files instead of HuggingFace
from torch.nn import Embedding as EmbeddingReference

import ttnn
from models.demos.deepseek_v3.tt.embedding import Embedding
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    load_reference_io_tensors_for_module,
    load_state_dict,
    run_module_forward,
)


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
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
    mesh_device,
    ccl,
    model_path,
    tmp_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    logger.info("Setting up reference IO")
    module_path = "model.embed_tokens"

    if generate_reference_io:
        reference_model = EmbeddingReference(
            hf_config.vocab_size,
            hf_config.hidden_size,
            hf_config.pad_token_id,
        ).eval()
        state_dict = reference_model.state_dict()

        torch_input = torch.randint(0, hf_config.vocab_size, (1, 1, seq_len))
        reference_output = reference_model(torch_input)

        # Do not cache random weights
        cache_path = tmp_path
        force_recalculate_weight_config = True
    else:
        state_dict = load_state_dict(model_path, module_path)
        torch_input, reference_output = load_reference_io_tensors_for_module(mode, module_path, seq_len, 1)

    # Generate module configs and state
    logger.info("Setting up TTNN configs")
    weight_config = get_test_weight_config(
        Embedding, hf_config, (state_dict,), cache_path, mesh_device, force_recalculate_weight_config
    )
    model_config = get_model_config(Embedding, mode, hf_config, mesh_device)
    model_state = Embedding.create_state(hf_config, mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    logger.info("Preparing TTNN inputs")
    tt_input_ids = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # TTNN forward pass
    logger.info("Running TTNN forward pass")
    tt_output = run_module_forward(Embedding, mode, tt_input_ids, run_config)

    # Convert output back to torch
    logger.info("Validating output")
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            dims=(0, -1),
            mesh_shape=tuple(mesh_device.shape),
        ),
    )[:1]

    # Cleanup
    ttnn.deallocate(tt_input_ids)
    ttnn.deallocate(tt_output)

    # Check PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
