# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn

from ...utils.tensor import bf16_tensor
from ...utils.check import assert_quality
from ...models.transformers.attention_sd35 import SD35JointAttention
from ...parallel.manager import CCLManager
from ....stable_diffusion_35_large.reference import SD3Transformer2DModel as TorchSD3Transformer2DModel
from ...utils.padding import PaddingConfig


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links",
    [
        [(2, 4), (1, 1), 0, 1, 1],
        [(2, 4), (1, 2), 0, 1, 1],
        [(2, 4), (1, 2), 1, 0, 1],
        [(2, 4), (2, 1), 0, 1, 1],
        [(2, 4), (2, 1), 1, 0, 1],
        [(2, 4), (2, 2), 0, 1, 1],
        [(2, 4), (2, 2), 1, 0, 1],
        [(2, 4), (2, 4), 0, 1, 1],  # fails because we don't have padded heads yet
        [(2, 4), (2, 4), 1, 0, 1],
        [(4, 8), (4, 4), 0, 1, 4],
    ],
    ids=[
        "1x1sp0tp1",
        "1x2sp0tp1",
        "1x2sp1tp0",
        "2x1sp0tp1",
        "2x1sp1tp0",
        "2x2sp0tp1",
        "2x2sp1tp0",
        "2x4sp0tp1",
        "2x4sp1tp0",
        "4x4sp0tp1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, spatial_seq_len, prompt_seq_len"),
    [
        (1, 4096, 333),  # SD3.5 large config
    ],
)
# @pytest.mark.parametrize("context_pre_only", [True, False])
# TODO: add more parametrizations of Attention module options
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_sd35_joint_attention(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    # context_pre_only: bool,
    model_location_generator,
) -> None:
    torch_dtype = torch.bfloat16

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Model configuration
    query_dim = 2432
    head_dim = 64
    heads = 38
    bias = True
    out_bias = True
    eps = 1e-6

    # Create Torch model
    model_name = model_location_generator(
        f"stabilityai/stable-diffusion-3.5-large", model_subdir="StableDiffusion_35_Large"
    )
    parent_torch_model = TorchSD3Transformer2DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model = parent_torch_model.transformer_blocks[0].attn
    torch_model.eval()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    # Create a simple parallel config mock for the attention module
    class SimpleParallelConfig:
        def __init__(self, mesh_axis, factor):
            self.mesh_axis = mesh_axis
            self.factor = factor

    class MockParallelConfig:
        def __init__(self, tp_axis, tp_factor, sp_axis, sp_factor):
            self.tensor_parallel = SimpleParallelConfig(tp_axis, tp_factor)
            self.sequence_parallel = SimpleParallelConfig(sp_axis, sp_factor)

    parallel_config = MockParallelConfig(tp_axis, tp_factor, sp_axis, sp_factor)

    if heads % tp_factor != 0:
        padding_config = PaddingConfig.from_tensor_parallel_factor(heads, head_dim, tp_factor)
    else:
        padding_config = None

    # Create TT model
    tt_model = SD35JointAttention(
        query_dim=query_dim,
        head_dim=head_dim,
        heads=heads,
        out_dim=query_dim,
        bias=bias,
        out_bias=out_bias,
        context_pre_only=torch_model.context_pre_only,
        eps=eps,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Create input tensors
    torch.manual_seed(0)
    spatial_input = torch.randn((B, spatial_seq_len, query_dim), dtype=torch_dtype)
    prompt_input = torch.randn((B, prompt_seq_len, query_dim), dtype=torch_dtype)

    torch_spatial, torch_prompt = torch_model(spatial=spatial_input, prompt=prompt_input)

    spatial_input_4d = spatial_input.unsqueeze(0)
    prompt_input_4d = prompt_input.unsqueeze(0)

    # Convert to TT tensors - replicated across mesh
    tt_spatial = bf16_tensor(spatial_input_4d, device=mesh_device, mesh_axis=sp_axis, shard_dim=2)
    tt_prompt = bf16_tensor(prompt_input_4d, device=mesh_device)

    # Run TT model
    tt_spatial_out, tt_prompt_out = tt_model(tt_spatial, tt_prompt, N=spatial_seq_len)

    spatial_shard_dims = [None, None]
    spatial_shard_dims[sp_axis] = 2
    spatial_shard_dims[tp_axis] = 3
    tt_spatial_torch = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=spatial_shard_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    assert_quality(torch_spatial, tt_spatial_torch, pcc=0.997_000)

    if not torch_model.context_pre_only:
        prompt_shard_dims = [None, None]
        prompt_shard_dims[sp_axis] = 0
        prompt_shard_dims[tp_axis] = 3
        tt_prompt_torch = ttnn.to_torch(
            tt_prompt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=prompt_shard_dims, mesh_shape=tuple(mesh_device.shape)
            ),
        )

        for i in range(tt_prompt_torch.shape[0]):
            assert_quality(torch_prompt, tt_prompt_torch[i], pcc=0.998_000)
