import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.mochi.final_layer import TtFinalLayer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from genmo.mochi_preview.dit.joint_model.asymm_models_joint import FinalLayer as RefFinalLayer
from models.experimental.mochi.common import get_mochi_dir, get_cache_path, compute_metrics


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_tt_final_layer_inference(mesh_device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16
    hidden_size = 3072
    patch_size = 2
    out_channels = 12
    seq_len = 44 * 1024  # Same as in feedforward test
    batch_size = 1

    mesh_device.enable_async(True)
    from safetensors.torch import load_file

    weights_path = os.path.join(get_mochi_dir(), "dit.safetensors")
    state_dict = load_file(weights_path)

    # Filter state dict for final layer
    final_layer_path = "final_layer"
    partial_state_dict = {
        k[len(final_layer_path) + 1 :]: v for k, v in state_dict.items() if k.startswith(final_layer_path)
    }
    print(partial_state_dict.keys())

    # Create reference model
    reference_model = RefFinalLayer(
        hidden_size=hidden_size,
        patch_size=patch_size,
        out_channels=out_channels,
    )
    reference_model.load_state_dict(partial_state_dict)

    # Create TT model
    weight_cache_path = get_cache_path(os.environ.get("FAKE_DEVICE"))
    tt_model = TtFinalLayer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix=final_layer_path,
        weight_cache_path=weight_cache_path,
        dtype=dtype,
        hidden_size=hidden_size,
    )

    # Create input tensors
    torch_x = torch.randn(1, batch_size, seq_len, hidden_size)
    torch_c = torch.randn(batch_size, hidden_size)

    tt_x = ttnn.from_torch(
        torch_x,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_c = ttnn.from_torch(
        torch_c.unsqueeze(0).unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run TtFinalLayer")
    tt_output = tt_model(tt_x, tt_c)

    # Output is already replicated
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    # Get reference output
    reference_output = reference_model(torch_x, torch_c)

    # Log output shapes
    logger.info(f"Reference output shape: {reference_output.shape}")
    logger.info(f"TT output shape: {tt_output_torch.shape}")

    # Compute metrics
    pcc, mse, mae = compute_metrics(reference_output, tt_output_torch)

    # Check if model meets requirements
    pcc_required = 0.99
    passing = pcc >= pcc_required

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    if passing:
        logger.info("TtFinalLayer Passed!")
    else:
        logger.warning("TtFinalLayer Failed!")

    assert passing, f"TtFinalLayer output does not meet PCC requirement {pcc_required}: {pcc}, MSE: {mse}, MAE: {mae}."
