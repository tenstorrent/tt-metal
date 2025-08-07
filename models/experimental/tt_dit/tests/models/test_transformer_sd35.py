# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn

from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard
from ...utils.check import assert_quality
from ...models.transformers.transformer_sd35 import SD35TransformerBlock, SD35Transformer2DModel
from ...parallel.manager import CCLManager
from ....stable_diffusion_35_large.reference import SD3Transformer2DModel as TorchSD3Transformer2DModel


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        # [(1, 1), 0, 1], # fail L1 OOM
        [(1, 2), 0, 1],
        # [(1, 2), 1, 0], # fail L1 OOM
        # [(2, 1), 0, 1], # fail L1 OOM
        [(2, 1), 1, 0],
        [(2, 2), 0, 1],
        [(2, 2), 1, 0],
        # [(2, 4), 0, 1], # fails because we don't have padded heads yet
        [(2, 4), 1, 0],
    ],
    ids=["1x2sp0tp1", "2x1sp1tp0", "2x2sp0tp1", "2x2sp1tp0", "2x4sp1tp0"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, spatial_seq_len, prompt_seq_len"),
    [
        (1, 4096, 333),  # SD3.5 large config
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_sd35_transformer_block(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    B: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Model configuration
    dim = 2432
    head_dim = 64
    num_heads = 38
    use_dual_attention = False
    qk_norm = "rms_norm"

    # Create Torch model
    # dummy = TorchTransformerBlock(
    #     dim=dim,
    #     num_heads=num_heads,
    #     head_dim=head_dim,
    #     context_pre_only=False,
    #     qk_norm=qk_norm,
    #     use_dual_attention=use_dual_attention,
    # ).to(torch_dtype)
    # print("random model")
    # print(dummy)
    parent_torch_model = TorchSD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-large", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model = parent_torch_model.transformer_blocks[0]
    torch_model.eval()
    # print("pretrained model")
    # print(torch_model)
    # breakpoint()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )

    # Create a simple parallel config mock for the transformer block
    class SimpleParallelConfig:
        def __init__(self, mesh_axis, factor):
            self.mesh_axis = mesh_axis
            self.factor = factor

    class MockParallelConfig:
        def __init__(self, tp_axis, tp_factor, sp_axis, sp_factor):
            self.tensor_parallel = SimpleParallelConfig(tp_axis, tp_factor)
            self.sequence_parallel = SimpleParallelConfig(sp_axis, sp_factor)

    parallel_config = MockParallelConfig(tp_axis, tp_factor, sp_axis, sp_factor)

    # Create TT model
    tt_model = SD35TransformerBlock(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        context_pre_only=torch_model.context_pre_only,
        use_dual_attention=use_dual_attention,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        init=False,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Create input tensors
    torch.manual_seed(0)
    spatial_input = torch.randn((B, spatial_seq_len, dim), dtype=torch_dtype)
    prompt_input = torch.randn((B, prompt_seq_len, dim), dtype=torch_dtype)
    time_embed_input = torch.randn((B, dim), dtype=torch_dtype)

    spatial_input_4d = spatial_input.unsqueeze(0).clone()
    prompt_input_4d = prompt_input.unsqueeze(0).clone()
    time_embed_input_4d = time_embed_input.unsqueeze(0).unsqueeze(0).clone()

    # Convert to TT tensors - spatial sharded on sequence parallel axis
    tt_spatial = bf16_tensor_2dshard(spatial_input_4d, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(prompt_input_4d, device=mesh_device, mesh_axis=tp_axis, shard_dim=3)
    tt_time_embed = bf16_tensor(time_embed_input_4d, device=mesh_device)

    # NOTE: DO NOT run torch model before creating TT tensors. Torch model will modify the input tensors in place.
    # Run torch model
    torch_spatial, torch_prompt = torch_model(spatial=spatial_input, prompt=prompt_input, time_embed=time_embed_input)

    # Run TT model
    tt_spatial_out, tt_prompt_out = tt_model(tt_spatial, tt_prompt, tt_time_embed, spatial_seq_len, prompt_seq_len)

    # Convert outputs back to torch and compare
    spatial_shard_dims = [None, None]
    spatial_shard_dims[sp_axis] = 2
    spatial_shard_dims[tp_axis] = 3
    tt_spatial_torch = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=spatial_shard_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    tt_spatial_torch = tt_spatial_torch.squeeze(0)
    # tt_spatial_torch += spatial_input # DEBUG! Do residual add in torch!

    assert_quality(torch_spatial, tt_spatial_torch, pcc=0.995_000)

    if not torch_model.context_pre_only:
        assert tt_prompt_out is not None
        prompt_shard_dims = [None, None]
        prompt_shard_dims[sp_axis] = 0  # No sequence sharding for prompt
        prompt_shard_dims[tp_axis] = 3
        tt_prompt_torch = ttnn.to_torch(
            tt_prompt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=prompt_shard_dims, mesh_shape=tuple(mesh_device.shape)
            ),
        )
        for i in range(tt_prompt_torch.shape[0]):
            assert_quality(torch_prompt, tt_prompt_torch[i], pcc=0.999_800)
    else:
        assert tt_prompt_out is None
        assert torch_prompt is None


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 2), 0, 1],
        [(2, 1), 1, 0],
        [(2, 2), 0, 1],
        [(2, 2), 1, 0],
    ],
    ids=["1x2sp0tp1", "2x1sp1tp0", "2x2sp0tp1", "2x2sp1tp0"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, H, W, spatial_seq_len, prompt_seq_len"),
    [
        (1, 128, 128, 4096, 333),  # SD3.5 large config
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_sd35_transformer2d_model(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    B: int,
    H: int,
    W: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
) -> None:
    """Test the full SD35Transformer2DModel against the reference implementation."""
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Model configuration (SD3.5 Large)
    sample_size = 128
    patch_size = 2
    in_channels = 16
    num_layers = 38
    attention_head_dim = 64
    num_attention_heads = 38
    joint_attention_dim = 4096
    caption_projection_dim = 2432
    pooled_projection_dim = 2048
    out_channels = 16
    pos_embed_max_size = 192
    dual_attention_layers = ()

    # Create Torch reference model
    torch_model = TorchSD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model.eval()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )

    # Create parallel config mock
    class SimpleParallelConfig:
        def __init__(self, mesh_axis, factor):
            self.mesh_axis = mesh_axis
            self.factor = factor

    class MockParallelConfig:
        def __init__(self, tp_axis, tp_factor, sp_axis, sp_factor):
            self.tensor_parallel = SimpleParallelConfig(tp_axis, tp_factor)
            self.sequence_parallel = SimpleParallelConfig(sp_axis, sp_factor)

    parallel_config = MockParallelConfig(tp_axis, tp_factor, sp_axis, sp_factor)

    # Create TT model
    tt_model = SD35Transformer2DModel(
        sample_size=sample_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_layers=num_layers,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        joint_attention_dim=joint_attention_dim,
        caption_projection_dim=caption_projection_dim,
        pooled_projection_dim=pooled_projection_dim,
        out_channels=out_channels,
        pos_embed_max_size=pos_embed_max_size,
        dual_attention_layers=dual_attention_layers,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        init=False,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Create input tensors
    torch.manual_seed(0)
    # Spatial input - latent tensor in NHWC format
    spatial_input_nchw = torch.randn((B, in_channels, H, W), dtype=torch_dtype)
    # Text prompt embeddings
    prompt_input = torch.randn((B, prompt_seq_len, joint_attention_dim), dtype=torch_dtype)
    # Pooled text projections
    pooled_projections = torch.randn((B, pooled_projection_dim), dtype=torch_dtype)
    # Timestep
    timestep = torch.randn((B,), dtype=torch_dtype)

    # Clone inputs for TT model (to avoid in-place modifications)
    spatial_input_nhwc_tt = spatial_input_nchw.permute(0, 2, 3, 1).clone()
    prompt_input_tt = prompt_input.clone()
    pooled_projections_tt = pooled_projections.clone()
    timestep_tt = timestep.clone()

    # Run torch model
    with torch.no_grad():
        torch_output = torch_model(
            spatial=spatial_input_nchw,
            prompt_embed=prompt_input,
            pooled_projections=pooled_projections,
            timestep=timestep,
        )

    # Convert inputs to TT tensors with proper sharding
    # Spatial: sharded on sequence dimension (sp_axis) and feature dimension (tp_axis)
    tt_spatial = bf16_tensor(spatial_input_nhwc_tt, device=mesh_device, mesh_axis=sp_axis, shard_dim=1)  # Sharded on H

    # Prompt: replicated
    prompt_4d = prompt_input_tt.unsqueeze(0)
    tt_prompt = bf16_tensor(prompt_4d, device=mesh_device)

    # Pooled projections: replicated
    pooled_4d = pooled_projections_tt.unsqueeze(0).unsqueeze(0)
    tt_pooled = bf16_tensor(pooled_4d, device=mesh_device)

    # Timestep: replicated
    timestep_4d = timestep_tt.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    tt_timestep = bf16_tensor(timestep_4d, device=mesh_device)

    # Calculate sequence lengths
    N = spatial_seq_len  # H * W // (patch_size * patch_size)
    L = prompt_seq_len

    # Run TT model
    tt_output = tt_model(tt_spatial, tt_prompt, tt_pooled, tt_timestep, N, L)

    # Convert output back to torch and compare
    # Output should be replicated
    output_shard_dims = [None]
    output_shard_dims[sp_axis] = 0  # Sequence dimension
    output_shard_dims[tp_axis] = 1  # Feature dimension

    # tt_output_torch = ttnn.to_torch(
    #     tt_output,
    #     mesh_composer=ttnn.ConcatMesh2dToTensor(
    #         mesh_device, dims=output_shard_dims, mesh_shape=tuple(mesh_device.shape)
    #     ),
    # )
    tt_output_tensors = ttnn.get_device_tensors(tt_output)

    # Remove the batch dimension added by TT model and reshape to match torch output
    # TT output shape: (1, B, N, patch_size*patch_size*out_channels)
    # Torch output shape: (B, H//patch_size, W//patch_size, patch_size*patch_size*out_channels)
    tt_output_torch = tt_output_torch.squeeze(0)  # Remove first dimension

    # # Reshape to match torch output format
    # patches_h = H // patch_size
    # patches_w = W // patch_size
    # patch_embed_dim = patch_size * patch_size * out_channels

    # tt_output_reshaped = tt_output_torch.view(B, patches_h, patches_w, patch_embed_dim)

    # Compare outputs
    for i in range(len(tt_output_tensors)):
        tt_output_torch = ttnn.to_torch(tt_output_tensors[i])
        assert_quality(torch_output, tt_output_torch, pcc=0.990_000)  # Lower PCC due to full model complexity
    # assert_quality(torch_output, tt_output_reshaped, pcc=0.990_000)  # Lower PCC due to full model complexity

    print("SD35Transformer2DModel test passed successfully!")
