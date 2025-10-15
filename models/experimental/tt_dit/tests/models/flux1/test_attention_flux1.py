# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import diffusers.models.transformers.transformer_flux as reference
import pytest
import torch
import ttnn

from ....models.transformers.attention_flux1 import Flux1Attention
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.padding import PaddingConfig
from ....utils.tensor import bf16_tensor


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis"),
    [
        pytest.param((1, 1), 0, 1, id="1x1sp0tp1"),
        pytest.param((1, 2), 0, 1, id="1x2sp0tp1"),
        pytest.param((1, 2), 1, 0, id="1x2sp1tp0"),
        pytest.param((2, 1), 0, 1, id="2x1sp0tp1"),
        pytest.param((2, 1), 1, 0, id="2x1sp1tp0"),
        pytest.param((2, 2), 0, 1, id="2x2sp0tp1"),
        pytest.param((2, 2), 1, 0, id="2x2sp1tp0"),
        pytest.param((2, 4), 0, 1, id="2x4sp0tp1"),
        pytest.param((2, 4), 1, 0, id="2x4sp1tp0"),
        pytest.param((4, 8), 0, 1, id="4x8sp0tp1"),
        pytest.param((4, 8), 1, 0, id="4x8sp1tp0"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "spatial_seq_len", "prompt_seq_len"),
    [
        (1, 4096, 512),
        # (1, 4096 + 512, 0),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attention(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
) -> None:
    torch_dtype = torch.bfloat16

    joint_attention = prompt_seq_len != 0

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Create Torch model
    parent_torch_model = reference.FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch_dtype
    )
    if joint_attention:
        torch_model = parent_torch_model.transformer_blocks[0].attn
    else:
        torch_model = parent_torch_model.single_transformer_blocks[0].attn
    assert isinstance(torch_model, reference.Attention)
    torch_model.eval()

    query_dim = torch_model.query_dim
    heads = torch_model.heads
    head_dim = torch_model.inner_dim // heads

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=1,
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
    tt_model = Flux1Attention(
        query_dim=query_dim,
        head_dim=head_dim,
        heads=heads,
        out_dim=torch_model.out_dim,
        added_kv_proj_dim=query_dim if joint_attention else 0,
        context_pre_only=torch_model.context_pre_only,
        pre_only=torch_model.pre_only,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Create input tensors
    torch.manual_seed(0)
    spatial_input = torch.randn((batch_size, spatial_seq_len, query_dim), dtype=torch_dtype)
    prompt_input = torch.randn((batch_size, prompt_seq_len, query_dim), dtype=torch_dtype) if joint_attention else None
    rope_cos = torch.randn([spatial_seq_len + prompt_seq_len, 128])
    rope_sin = torch.randn([spatial_seq_len + prompt_seq_len, 128])

    torch_output = torch_model.forward(spatial_input, prompt_input, image_rotary_emb=(rope_cos, rope_sin))
    torch_spatial, torch_prompt = torch_output if joint_attention else (torch_output, None)

    # Convert to TT tensors - replicated across mesh
    tt_spatial = bf16_tensor(spatial_input, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_prompt = bf16_tensor(prompt_input, device=mesh_device) if prompt_input is not None else None
    tt_spatial_rope_cos = bf16_tensor(rope_cos[prompt_seq_len:], device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_spatial_rope_sin = bf16_tensor(rope_sin[prompt_seq_len:], device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_prompt_rope_cos = bf16_tensor(rope_cos[:prompt_seq_len], device=mesh_device) if joint_attention else None
    tt_prompt_rope_sin = bf16_tensor(rope_sin[:prompt_seq_len], device=mesh_device) if joint_attention else None

    # Run TT model
    tt_spatial_out, tt_prompt_out = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin) if joint_attention else None,
        spatial_sequence_length=spatial_seq_len,
    )

    spatial_shard_dims = [None, None]
    spatial_shard_dims[sp_axis] = 1
    spatial_shard_dims[tp_axis] = 2
    tt_spatial_torch = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=spatial_shard_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    assert_quality(torch_spatial, tt_spatial_torch, pcc=0.9939, mse=0.01)

    if torch_prompt is not None:
        prompt_shard_dims = [None, None]
        prompt_shard_dims[sp_axis] = 0
        prompt_shard_dims[tp_axis] = 2
        tt_prompt_torch = ttnn.to_torch(
            tt_prompt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=prompt_shard_dims, mesh_shape=tuple(mesh_device.shape)
            ),
        )[:batch_size]

        assert_quality(torch_prompt, tt_prompt_torch, pcc=0.9939, mse=0.02)
