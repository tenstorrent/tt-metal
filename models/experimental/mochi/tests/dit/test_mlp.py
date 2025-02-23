import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.mochi.tt.dit.mlp import FeedForward as TtFeedForward
from models.utility_functions import (
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from genmo.mochi_preview.dit.joint_model.layers import FeedForward as RefFeedForward
from models.experimental.mochi.tt.common import get_mochi_dir, get_cache_path, compute_metrics


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
    "ff_path, in_feat, seq_len, seq_shard",
    [
        ("blocks.0.mlp_x", 3072, 44 * 1024, True),
        # ("blocks.0.mlp_x", 3072, 44520),
        ("blocks.0.mlp_y", 1536, 256, False),
        # ("blocks.0.mlp_y", 1536, 118),
    ],
)
def test_tt_feedforward_inference(mesh_device, seq_len, use_program_cache, reset_seeds, ff_path, in_feat, seq_shard):
    dtype = ttnn.bfloat16

    mesh_device.enable_async(True)
    from safetensors.torch import load_file

    weights_path = os.path.join(get_mochi_dir(), "dit.safetensors")
    state_dict = load_file(weights_path)
    partial_state_dict = {k[len(ff_path) + 1 :]: v for k, v in state_dict.items() if k.startswith(ff_path)}
    print(partial_state_dict.keys())

    multiple_of = 256
    mlp_ratio = 4.0
    mlp_hidden_dim = int(in_feat * mlp_ratio)

    reference_model = RefFeedForward(
        in_features=in_feat,
        hidden_size=mlp_hidden_dim,
        multiple_of=multiple_of,
        ffn_dim_multiplier=None,
    )
    reference_model.load_state_dict(partial_state_dict)

    weight_cache_path = get_cache_path(os.environ.get("FAKE_DEVICE"))
    tt_model = TtFeedForward(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        layer_num=0,
        dtype=dtype,
        in_features=in_feat,
        hidden_size=mlp_hidden_dim,
        multiple_of=multiple_of,
        ffn_dim_multiplier=None,
        state_dict_prefix=ff_path,
        seq_shard=seq_shard,
    )
    torch_input = torch.randn(1, 1, seq_len, in_feat)
    if seq_shard:
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-2)
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run TtFeedForward")
    tt_output = tt_model(tt_input)

    if seq_shard:
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-2))
    else:
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    # Get reference output from the reference model
    reference_output = reference_model(torch_input)

    # Compute metrics
    pcc, mse, mae = compute_metrics(reference_output, tt_output_torch)
    # Check if model meets requirements
    pcc_required = 0.99
    passing = pcc >= pcc_required

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    if passing:
        logger.info("TtFeedForward Passed!")
    else:
        logger.warning("TtFeedForward Failed!")

    assert passing, f"TtFeedForward output does not meet PCC requirement {pcc_required}: {pcc}, MSE: {mse}, MAE: {mae}."
