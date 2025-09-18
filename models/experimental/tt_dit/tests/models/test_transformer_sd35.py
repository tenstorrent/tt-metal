# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from loguru import logger

from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard
from ...utils.check import assert_quality
from ...models.transformers.transformer_sd35 import SD35TransformerBlock, SD35Transformer2DModel
from ...parallel.manager import CCLManager
from ....stable_diffusion_35_large.reference import SD3Transformer2DModel as TorchSD3Transformer2DModel
from ...utils.padding import PaddingConfig
from ...utils.cache import get_cache_path, get_and_create_cache_path, save_cache_dict, load_cache_dict
import time
import os


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, sp_axis, tp_axis, num_links",
    [
        [(2, 4), (1, 2), 0, 1, 1],
        [(2, 4), (2, 1), 1, 0, 1],
        [(2, 4), (2, 2), 0, 1, 1],
        [(2, 4), (2, 2), 1, 0, 1],
        [(2, 4), (2, 4), 0, 1, 1],
        [(2, 4), (2, 4), 1, 0, 1],
        [(4, 8), (4, 4), 0, 1, 4],
    ],
    ids=["1x2sp0tp1", "2x1sp1tp0", "2x2sp0tp1", "2x2sp1tp0", "2x4sp0tp1", "2x4sp1tp0", "4x4sp0tp1"],
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
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    model_location_generator,
) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    # Model configuration
    dim = 2432
    head_dim = 64
    num_heads = 38
    use_dual_attention = False

    # Create Torch model
    model_name = model_location_generator(
        f"stabilityai/stable-diffusion-3.5-large", model_subdir="StableDiffusion_35_Large"
    )
    parent_torch_model = TorchSD3Transformer2DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model = parent_torch_model.transformer_blocks[0]
    torch_model.eval()
    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=submesh_device,
        num_links=num_links,
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

    # Create padding config if needed for tensor parallelism
    if num_heads % tp_factor != 0:
        padding_config = PaddingConfig.from_tensor_parallel_factor(num_heads, head_dim, tp_factor)
    else:
        padding_config = None

    # Create TT model
    tt_model = SD35TransformerBlock(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        context_pre_only=torch_model.context_pre_only,
        use_dual_attention=use_dual_attention,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
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
    tt_spatial = bf16_tensor_2dshard(spatial_input_4d, device=submesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(prompt_input_4d, device=submesh_device, mesh_axis=tp_axis, shard_dim=3)
    tt_time_embed = bf16_tensor(time_embed_input_4d, device=submesh_device)

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
            submesh_device, dims=spatial_shard_dims, mesh_shape=tuple(submesh_device.shape)
        ),
    )
    tt_spatial_torch = tt_spatial_torch.squeeze(0)

    assert_quality(torch_spatial, tt_spatial_torch, pcc=0.994_000)

    if not torch_model.context_pre_only:
        assert tt_prompt_out is not None
        prompt_shard_dims = [None, None]
        prompt_shard_dims[sp_axis] = 0  # No sequence sharding for prompt
        prompt_shard_dims[tp_axis] = 3
        tt_prompt_torch = ttnn.to_torch(
            tt_prompt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                submesh_device, dims=prompt_shard_dims, mesh_shape=tuple(submesh_device.shape)
            ),
        )
        for i in range(tt_prompt_torch.shape[0]):
            assert_quality(torch_prompt, tt_prompt_torch[i], pcc=0.999_800)
    else:
        assert tt_prompt_out is None
        assert torch_prompt is None


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, sp_axis, tp_axis, num_links",
    [
        [(2, 4), (1, 2), 0, 1, 1],
        [(2, 4), (2, 1), 1, 0, 1],
        [(2, 4), (2, 2), 0, 1, 1],
        [(2, 4), (2, 2), 1, 0, 1],
        [(2, 4), (2, 4), 0, 1, 1],
        [(2, 4), (2, 4), 1, 0, 1],
        [(4, 8), (4, 4), 0, 1, 4],
    ],
    ids=["1x2sp0tp1", "2x1sp1tp0", "2x2sp0tp1", "2x2sp1tp0", "2x4sp0tp1", "2x4sp1tp0", "4x4sp0tp1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, H, W, spatial_seq_len, prompt_seq_len"),
    [
        (1, 128, 128, 4096, 333),  # SD3.5 large config
    ],
)
@pytest.mark.parametrize("load_cache", [True, False], ids=["yes_load_cache", "no_load_cache"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_sd35_transformer2d_model(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    H: int,
    W: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    load_cache: bool,
) -> None:
    """Test the full SD35Transformer2DModel against the reference implementation."""
    torch.manual_seed(0)
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

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
        mesh_device=submesh_device,
        num_links=num_links,
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

    # Create padding config if needed for tensor parallelism
    if num_attention_heads % tp_factor != 0:
        padding_config = PaddingConfig.from_tensor_parallel_factor(num_attention_heads, attention_head_dim, tp_factor)
    else:
        padding_config = None

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
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    if load_cache:
        cache_path = get_cache_path(
            model_name="stable-diffusion-3.5-large",
            subfolder="transformer",
            parallel_config=parallel_config,
            dtype="bf16",
        )
        assert os.path.exists(
            cache_path
        ), "Cache path does not exist. Run test_sd35_transformer_model_caching first with the desired parallel config."
        start = time.time()
        cache_dict = load_cache_dict(cache_path)
        tt_model.from_cached_state_dict(cache_dict)
        end = time.time()
        logger.info(f"Time taken to load cached state dict: {end - start} seconds")
    else:
        start = time.time()
        tt_model.load_state_dict(torch_model.state_dict())
        end = time.time()
        logger.info(f"Time taken to load state dict: {end - start} seconds")

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
    tt_spatial = bf16_tensor(
        spatial_input_nhwc_tt, device=submesh_device, mesh_axis=sp_axis, shard_dim=1
    )  # Sharded on H

    # Prompt: replicated
    prompt_4d = prompt_input_tt.unsqueeze(0)
    tt_prompt = bf16_tensor(prompt_4d, device=submesh_device)

    # Pooled projections: replicated
    pooled_4d = pooled_projections_tt.unsqueeze(0).unsqueeze(0)
    tt_pooled = bf16_tensor(pooled_4d, device=submesh_device)

    # Timestep: replicated
    timestep_4d = timestep_tt.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    tt_timestep = bf16_tensor(timestep_4d, device=submesh_device)

    # Calculate sequence lengths
    N = spatial_seq_len  # H * W // (patch_size * patch_size)
    L = prompt_seq_len

    # Run TT model
    tt_output = tt_model(tt_spatial, tt_prompt, tt_pooled, tt_timestep, N, L)
    logger.info(f"TT output shape: {tt_output.shape}")

    tt_output_tensors = ttnn.get_device_tensors(tt_output)

    # Compare outputs
    for i in range(len(tt_output_tensors)):
        logger.info(f"Checking output tensor {i}")
        tt_output_torch = ttnn.to_torch(tt_output_tensors[i])
        assert_quality(torch_output.squeeze(), tt_output_torch.squeeze(), pcc=0.999_940)

    print("SD35Transformer2DModel test passed successfully!")


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, sp_axis, tp_axis, num_links",
    [
        [(2, 4), (2, 2), 0, 1, 1],
        [(4, 8), (4, 4), 0, 1, 4],
    ],
    ids=[
        "2x2sp0tp1",
        "4x4sp0tp1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, H, W, spatial_seq_len, prompt_seq_len"),
    [
        (1, 128, 128, 4096, 333),
    ],
    ids=["sd35_large"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_sd35_transformer_model_caching(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    H: int,
    W: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
) -> None:
    """Test caching functionality for SD35Transformer2DModel."""
    torch.manual_seed(0)
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

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

    # Tight error bounds based on test config
    MIN_PCC = 0.999_500
    MIN_RMSE = 0.14

    # Create Torch reference model
    torch_model = TorchSD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model.eval()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=submesh_device,
        num_links=num_links,
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

    cache_path = get_and_create_cache_path(
        model_name="stable-diffusion-3.5-large",
        subfolder="transformer",
        parallel_config=parallel_config,
        dtype="bf16",
    )

    # Create padding config if needed for tensor parallelism
    if num_attention_heads % tp_factor != 0:
        padding_config = PaddingConfig.from_tensor_parallel_factor(num_attention_heads, attention_head_dim, tp_factor)
    else:
        padding_config = None
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
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    start = time.time()
    tt_model.load_state_dict(torch_model.state_dict())
    end = time.time()
    logger.info(f"Time taken to load state dict: {end - start} seconds")

    start = time.time()
    cache_dict = tt_model.to_cached_state_dict(cache_path)
    save_cache_dict(cache_dict, cache_path)
    end = time.time()
    logger.info(f"Time taken to cache state dict: {end - start} seconds")

    start = time.time()
    del tt_model

    cache_model = SD35Transformer2DModel(
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
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    loaded_cache_dict = load_cache_dict(cache_path)
    cache_model.from_cached_state_dict(loaded_cache_dict)
    end = time.time()
    logger.info(f"Time taken to load cached state dict: {end - start} seconds")
