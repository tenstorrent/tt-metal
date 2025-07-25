# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3RMSNorm
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.tt.rms_norm.rms_norm import RMSNorm
from models.demos.deepseek_v3.utils.config_helpers import even_int_div
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.utility_functions import comp_pcc


@pytest.fixture
def reference_model(hf_config, request):
    """Get the actual DeepSeek RMSNorm model using local implementation."""
    param = getattr(request, "param", None)
    if param is None or not hasattr(hf_config, param):
        raise ValueError("Test function must be parametrized with the config parameter for the norm size.")
    return DeepseekV3RMSNorm(
        hidden_size=getattr(hf_config, param),
        eps=hf_config.rms_norm_eps,
    )


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 32),
        ("prefill", 128),
    ],
)
def test_forward_pass_for_decoder_norm(
    mode,
    seq_len,
    hf_config,
    tmp_path,
    mesh_row,
):
    """Test rmsnorm forward pass against reference model."""
    # Create the test input
    torch_input = torch.randn(1, 1, seq_len, hf_config.hidden_size)

    # Get the reference output
    reference_model = DeepseekV3RMSNorm(
        hidden_size=hf_config.hidden_size,
        eps=hf_config.rms_norm_eps,
    )
    reference_output = reference_model(torch_input)

    # Generate the weight config
    hf_state_dict = reference_model.state_dict()
    weight_config = DistributedRMSNorm.convert_weights(hf_config, hf_state_dict, tmp_path, mesh_row)

    # Generate the model config
    if mode == "prefill":
        model_config = DistributedRMSNorm.prefill_model_config(hf_config, mesh_row)
    else:
        model_config = DistributedRMSNorm.decode_model_config(hf_config, mesh_row)

    # Generate a new model state
    model_state = DistributedRMSNorm.create_state(hf_config, mesh_row)

    # Generate the run config
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert the input to TTNN tensor
    if mode == "prefill":
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        shard_core_grid = ttnn.CoreGrid(x=4, y=7)
        memory_config = ttnn.create_sharded_memory_config(
            shape=(
                ttnn.core.roundup(even_int_div(seq_len, tuple(mesh_row.shape)[0]), ttnn.TILE_SIZE),
                ttnn.core.roundup(
                    even_int_div(hf_config.hidden_size, shard_core_grid.num_cores * tuple(mesh_row.shape)[1]),
                    ttnn.TILE_SIZE,
                ),
            ),
            core_grid=shard_core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_row,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_row, dims=(-2, -1), mesh_shape=tuple(mesh_row.shape)),
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN forward pass
    if mode == "prefill":
        tt_output = DistributedRMSNorm.forward_prefill(tt_input, run_config)
    else:
        tt_output = DistributedRMSNorm.forward_decode(tt_input, run_config)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_row, dims=(-2, -1), mesh_shape=tuple(mesh_row.shape)),
    )

    # Compare outputs
    pcc_required = 0.98
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"PCC: {pcc_message}")
    assert passing, f"RMS output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)


@pytest.mark.parametrize(
    "mode, seq_len, reference_model, hf_config_size_attr",
    [
        ("decode", 32, "q_lora_rank", "q_lora_rank"),
        ("prefill", 128, "q_lora_rank", "q_lora_rank"),
        ("decode", 32, "kv_lora_rank", "kv_lora_rank"),
        ("prefill", 128, "kv_lora_rank", "kv_lora_rank"),
    ],
    indirect=["reference_model"],
)
def test_rmsnorm_forward_pass_for_kq_norm(
    mode,
    seq_len,
    hf_config_size_attr,
    reference_model,
    hf_config,
    tmp_path,
    mesh_row,
):
    """Test rmsnorm forward pass against reference model."""
    # Get the hidden_size from the given hf_config attribute
    hidden_size = getattr(hf_config, hf_config_size_attr)

    # Create the test input
    torch_input = torch.randn(1, 1, seq_len, hidden_size)

    # Get the reference output
    reference_model = DeepseekV3RMSNorm(
        hidden_size=hidden_size,
        eps=hf_config.rms_norm_eps,
    )
    reference_output = reference_model(torch_input)

    # Generate the weight config
    hf_state_dict = reference_model.state_dict()
    weight_config = RMSNorm.convert_weights(hf_config, hf_state_dict, tmp_path, mesh_row)

    # Generate the model config
    if mode == "prefill":
        model_config = RMSNorm.prefill_model_config(hf_config, mesh_row)
    else:
        model_config = RMSNorm.decode_model_config(hf_config, mesh_row)

    # Generate a new model state
    model_state = RMSNorm.create_state(hf_config, mesh_row)

    # Generate the run config
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert the input to TTNN tensor
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_row,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_row),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN forward pass
    if mode == "prefill":
        tt_output = RMSNorm.forward_prefill(tt_input, run_config)
    else:
        tt_output = RMSNorm.forward_decode(tt_input, run_config)

    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])

    # Compare outputs
    pcc_required = 0.98
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"PCC: {pcc_message}")
    assert passing, f"RMS output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
