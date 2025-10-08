import os

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssRMSNorm
from models.demos.gpt_oss.tt.model_config import ModelArgs
from models.demos.gpt_oss.tt.rms_norm import RMSNorm
from tests.ttnn.utils_for_testing import comp_pcc

# ModelArgs will be instantiated inside test functions to avoid import-time loading


@pytest.fixture
def hf_config():
    """Load GPT-OSS config for testing"""
    path = os.getenv("HF_MODEL", "models/demos/gpt_oss/reference")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


@pytest.fixture
def reference_model(hf_config):
    model = GptOssRMSNorm(hf_config.hidden_size, hf_config.rms_norm_eps)
    return model


@pytest.mark.parametrize(
    "mesh_device",
    [
        (4, 8),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    [
        1,
        32,
        64,
        # 128,
        # 512,
        # 1024,
    ],
)
def test_rms_norm(
    mesh_device,
    device_params,
    seq_len,
    hf_config,
    reference_model,
):
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 8)))
    print(mesh_device.shape)

    # Get paths from ModelArgs to avoid code duplication
    model_args = ModelArgs(mesh_device=None, dummy_weights=True)  # dummy_weights=True to avoid loading actual weights
    gpt_dir = model_args.model_path

    torch.manual_seed(0)
    mesh_shape = tuple(mesh_device.shape)
    ref_rms_norm = reference_model.eval()

    # create a random tensor with shape (batch_size, seq_len, hidden_size)
    torch_input = torch.randn(1, seq_len, hf_config.hidden_size) * 2 + 3

    # forward pass through the reference model
    with torch.no_grad():
        ref_output = ref_rms_norm(torch_input)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        # mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, -1)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    # per_device_input_shape = tuple(tt_input.shape)
    # num_sharded_cores = 45
    # input_memory_config = ttnn.create_sharded_memory_config(
    #     shape=(ttnn.TILE_SIZE, per_device_input_shape[-1] // num_sharded_cores),
    #     core_grid=ttnn.CoreGrid(y=9, x=5),
    #     strategy=ttnn.ShardStrategy.WIDTH,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #     use_height_and_width_as_shard_shape=True,
    # )
    # tt_input = ttnn.to_memory_config(tt_input, input_memory_config)

    tt_rms_norm = RMSNorm(mesh_device, hf_config, reference_model.state_dict())
    tt_output = tt_rms_norm(tt_input)

    # tt_output_torch = ttnn.to_torch(
    #     tt_output,
    #     mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, dims=(0, -1)),
    # )[0]

    tt_output_tensors = ttnn.get_device_tensors(tt_output)

    expected_pcc = 0.99
    for i in range(len(tt_output_tensors)):
        tt_output_torch = ttnn.to_torch(tt_output_tensors[i])
        passing, pcc_message = comp_pcc(tt_output_torch, ref_output, pcc=expected_pcc)
        mse = torch.nn.functional.mse_loss(tt_output_torch, ref_output)
        print(f"PCC: {pcc_message}, MSE: {mse}")
        assert passing, f"PCC is {pcc_message}, expected {expected_pcc}"
