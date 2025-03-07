import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.mochi.tt.dit.embed import PatchEmbed as TtPatchEmbed

from models.utility_functions import skip_for_grayskull
from genmo.mochi_preview.dit.joint_model.layers import PatchEmbed
from models.experimental.mochi.tt.common import get_cache_path, compute_metrics
from models.experimental.mochi.tests.dit.common import load_model_weights


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
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 1024}],
    indirect=True,
)
@pytest.mark.parametrize(
    "patch_size, in_chans, embed_dim, bias",
    [(2, 12, 3072, True)],
)
@pytest.mark.parametrize("B, T, H, W", [(1, 28, 60, 106)])
def test_tt_patch_embed_inference(
    mesh_device, patch_size, in_chans, embed_dim, bias, B, T, H, W, use_program_cache, reset_seeds
):
    dtype = ttnn.bfloat16
    mesh_device.enable_async(True)

    state_dict, partial_state_dict = load_model_weights("x_embedder")

    # Create reference model
    reference_model = PatchEmbed(
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        bias=bias,
    )
    reference_model.load_state_dict(partial_state_dict)

    # Create TT model
    weight_cache_path = get_cache_path(os.environ.get("FAKE_DEVICE"))
    tt_model = TtPatchEmbed(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix="x_embedder",
        weight_cache_path=weight_cache_path,
        dtype=dtype,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        bias=bias,
    )

    # Create input tensor
    torch_input = torch.randn(B, in_chans, T, H, W)
    print(f"torch_input: {torch_input.shape}")

    # Prepare tt input in flattened format
    tt_input = torch_input.squeeze(0).reshape(in_chans, T, H // patch_size, patch_size, W // patch_size, patch_size)
    tt_input = tt_input.permute(1, 2, 4, 0, 3, 5).reshape(
        1, B, T * H // patch_size * W // patch_size, in_chans * patch_size * patch_size
    )
    print(f"tt_input: {tt_input.shape}")
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run TtPatchEmbed")
    tt_output = tt_model(tt_input)

    # Output is replicated, so we need to take the first element
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    # Get reference output
    reference_output = reference_model(torch_input)

    # Compute metrics
    passing = True
    for b in range(B):
        pcc, mse, mae = compute_metrics(reference_output[b], tt_output_torch[b])
        # Check if model meets requirements
        pcc_required = 0.99
        passing = passing and (pcc >= pcc_required)
        logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    if passing:
        logger.info("TtPatchEmbed Passed!")
    else:
        logger.warning("TtPatchEmbed Failed!")

    assert passing, f"TtPatchEmbed output does not meet PCC requirement {pcc_required}: {pcc}, MSE: {mse}, MAE: {mae}."
