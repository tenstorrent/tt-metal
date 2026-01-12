# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import os
import pytest
import torch
import ttnn
from loguru import logger

from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard
from ....utils.check import assert_quality
from ....models.transformers.transformer_mochi import MochiTransformerBlock, MochiTransformer3DModel
from ....parallel.manager import CCLManager
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....utils.padding import pad_vision_seq_parallel
from ....utils.cache import get_cache_path, get_and_create_cache_path, save_cache_dict, load_cache_dict
from diffusers import MochiTransformer3DModel as TorchMochiTransformer3DModel
from models.tt_transformers.tt.common import get_rot_transformation_mat


def stack_cos_sin(cos, sin):
    cos = torch.stack([cos, cos], dim=-1).flatten(-2)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2)
    return cos, sin


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links",
    [
        [(1, 1), (1, 1), 0, 1, 1],
        [(1, 2), (1, 2), 0, 1, 1],
        [(1, 2), (1, 2), 1, 0, 1],
        [(2, 1), (2, 1), 0, 1, 1],
        [(2, 1), (2, 1), 1, 0, 1],
        [(2, 2), (2, 2), 0, 1, 1],
        [(2, 2), (2, 2), 1, 0, 1],
        [(2, 4), (2, 4), 0, 1, 1],
        [(2, 4), (2, 4), 1, 0, 1],
        [(1, 8), (1, 8), 1, 0, 1],
        [(4, 8), (2, 8), 1, 0, 4],
        [(4, 8), (4, 8), 0, 1, 4],
        [(4, 8), (4, 8), 1, 0, 4],
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
        "1x8sp1tp0",
        "2x8sp1tp0",
        "4x8sp0tp1",
        "4x8sp1tp0",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, spatial_seq_len, prompt_seq_len"),
    [
        (1, 4000, 118),  # Similar to SD3.5 config
        (1, 44520, 118),  # Similar to SD3.5 config
    ],
    ids=["short_seq", "long_seq"],
)
@pytest.mark.parametrize("is_fsdp", [True, False], ids=["yes_fsdp", "no_fsdp"])
@pytest.mark.parametrize("context_pre_only", [True, False], ids=["yes_context_pre", "no_context_pre"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_mochi_transformer_block(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    context_pre_only: bool,
    is_fsdp: bool,
) -> None:
    torch_dtype = torch.float32
    parent_mesh_device = mesh_device
    mesh_device = parent_mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Model configuration
    dim = 3072
    pooled_projection_dim = 1536  # Different from dim
    num_attention_heads = 24
    attention_head_dim = 128
    activation_fn = "swiglu"
    eps = 1e-6
    layer_id = 0 if not context_pre_only else -1

    # Tight error bounds based on test config
    if not context_pre_only:
        MIN_PCC = 0.999_400 if spatial_seq_len == 4000 else 0.999_050
        MIN_RMSE = 0.036 if spatial_seq_len == 4000 else 0.046
    else:
        MIN_PCC = 0.991_500 if spatial_seq_len == 4000 else 0.990_300
        MIN_RMSE = 0.14 if spatial_seq_len == 4000 else 0.14

    parent_torch_model = TorchMochiTransformer3DModel.from_pretrained(
        f"genmo/mochi-1-preview", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model = parent_torch_model.transformer_blocks[layer_id]
    torch_model.eval()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )

    # Create TT model
    tt_model = MochiTransformerBlock(
        dim=dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        pooled_projection_dim=pooled_projection_dim,
        activation_fn=activation_fn,
        context_pre_only=context_pre_only,
        eps=eps,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Initialize weights randomly for testing
    torch.manual_seed(0)
    # Create input tensors
    spatial_input = torch.randn((B, spatial_seq_len, dim), dtype=torch_dtype)
    prompt_input = torch.randn((B, prompt_seq_len, pooled_projection_dim), dtype=torch_dtype)
    temb_input = torch.randn((B, dim), dtype=torch_dtype)

    # Create ROPE embeddings
    rope_cos = torch.randn(spatial_seq_len, num_attention_heads, attention_head_dim // 2)
    rope_sin = torch.randn(spatial_seq_len, num_attention_heads, attention_head_dim // 2)

    rope_cos_stack, rope_sin_stack = stack_cos_sin(
        rope_cos.unsqueeze(0).permute(0, 2, 1, 3), rope_sin.unsqueeze(0).permute(0, 2, 1, 3)
    )

    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    rope_cos_padded = pad_vision_seq_parallel(rope_cos_stack, num_devices=sp_factor)
    rope_sin_padded = pad_vision_seq_parallel(rope_sin_stack, num_devices=sp_factor)

    # Sequence fractured spatial
    tt_spatial = bf16_tensor(spatial_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    # Replicated prompt
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
    # Replicated time embedding
    tt_temb = bf16_tensor(temb_input.reshape(1, 1, B, dim), device=mesh_device)

    # Rope cos and sin sequence fractured and head fractured
    tt_rope_cos = bf16_tensor_2dshard(rope_cos_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_rope_sin = bf16_tensor_2dshard(rope_sin_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Create transformation matrix for RoPE
    trans_mat = get_rot_transformation_mat(None)
    tt_trans_mat = bf16_tensor(trans_mat, device=mesh_device)

    # Run TT model
    logger.info(
        f"Running TT model with spatial shape {tt_spatial.shape}, prompt shape {tt_prompt.shape}, rope_cos shape {tt_rope_cos.shape}, rope_sin shape {tt_rope_sin.shape}"
    )
    tt_spatial_out, tt_prompt_out = tt_model(
        spatial_1BND=tt_spatial,
        prompt_1BLP=tt_prompt,
        temb_11BD=tt_temb,
        N=spatial_seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
    )

    spatial_concat_dims = [None, None]
    spatial_concat_dims[sp_axis] = 2
    spatial_concat_dims[tp_axis] = 0
    tt_spatial_out = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=spatial_concat_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    tt_spatial_out = tt_spatial_out[:, :, :spatial_seq_len, :]

    # Run torch model
    logger.info(f"Running torch model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}")
    attention_mask = torch.ones((B, prompt_seq_len), dtype=torch_dtype)
    torch_spatial_out, torch_prompt_out = torch_model(
        hidden_states=spatial_input,
        encoder_hidden_states=prompt_input,
        temb=temb_input,
        encoder_attention_mask=attention_mask,
        image_rotary_emb=[rope_cos, rope_sin],
    )

    logger.info(f"Checking spatial outputs")
    for i in range(tt_spatial_out.shape[0]):
        assert_quality(torch_spatial_out, tt_spatial_out[i], pcc=MIN_PCC, relative_rmse=MIN_RMSE)

    if not context_pre_only:
        prompt_concat_dims = [None, None]
        prompt_concat_dims[sp_axis] = 0
        prompt_concat_dims[tp_axis] = 1
        tt_prompt_out = ttnn.to_torch(
            tt_prompt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=prompt_concat_dims, mesh_shape=tuple(mesh_device.shape)
            ),
        )
        tt_prompt_out = tt_prompt_out[:, :, :prompt_seq_len, :]
        # Get all replicas into the first dimension for checking
        tt_prompt_out = tt_prompt_out.reshape(-1, prompt_seq_len, pooled_projection_dim)

        logger.info(f"Checking prompt outputs")
        for i in range(tt_prompt_out.shape[0]):
            assert_quality(torch_prompt_out, tt_prompt_out[i], pcc=MIN_PCC, relative_rmse=MIN_RMSE)


@pytest.mark.parametrize(
    "dit_unit_test",
    [{"1": True, "0": False}.get(os.environ.get("DIT_UNIT_TEST"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links",
    [
        [(2, 2), 0, 1, 1],
        [(2, 2), 1, 0, 1],
        [(2, 4), 0, 1, 1],
        [(2, 4), 1, 0, 1],
        [(4, 8), 0, 1, 4],
        [(4, 8), 1, 0, 4],
    ],
    ids=[
        "2x2sp0tp1",
        "2x2sp1tp0",
        "2x4sp0tp1",
        "2x4sp1tp0",
        "4x8sp0tp1",
        "4x8sp1tp0",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, T, H, W, prompt_seq"),
    [
        (1, 8, 40, 50, 118),  # small input
        (1, 16, 40, 100, 118),  # medium input
        (1, 28, 60, 106, 118),  # large input
    ],
    ids=["short_seq", "medium_seq", "long_seq"],
)
@pytest.mark.parametrize(
    "test_attention_mask", [True, False], ids=["yes_test_attention_mask", "no_test_attention_mask"]
)
@pytest.mark.parametrize("load_cache", [True, False], ids=["yes_load_cache", "no_load_cache"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_mochi_transformer_model(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq: int,
    load_cache: bool,
    test_attention_mask: bool,
    dit_unit_test: bool,
) -> None:
    torch_dtype = torch.float32

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Model configuration
    patch_size = 2
    num_attention_heads = 24
    attention_head_dim = 128
    num_layers = 48
    pooled_projection_dim = 1536
    in_channels = 12
    text_embed_dim = 4096
    time_embed_dim = 256
    activation_fn = "swiglu"

    # Tight error bounds based on test config
    MIN_PCC = 0.992_000
    MIN_RMSE = 0.14

    if dit_unit_test:
        torch_model = TorchMochiTransformer3DModel(num_layers=1)
        num_layers = torch_model.config.num_layers
    else:
        torch_model = TorchMochiTransformer3DModel.from_pretrained(
            f"genmo/mochi-1-preview", subfolder="transformer", torch_dtype=torch_dtype
        )
    torch_model.eval()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )

    torch.manual_seed(0)
    # Create input tensors
    spatial_input = torch.randn((B, in_channels, T, H, W), dtype=torch_dtype)
    prompt_input = torch.randn((B, prompt_seq, text_embed_dim), dtype=torch_dtype)
    timestep_input = torch.randint(0, 1000, (B,), dtype=torch_dtype)
    attention_mask = torch.ones((B, prompt_seq), dtype=torch_dtype)
    if test_attention_mask:
        # Test that masking prompt works
        attention_mask[:, prompt_seq // 2 :] = 0

    # Create TT model
    tt_model = MochiTransformer3DModel(
        patch_size=patch_size,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        num_layers=num_layers,
        pooled_projection_dim=pooled_projection_dim,
        in_channels=in_channels,
        text_embed_dim=text_embed_dim,
        time_embed_dim=time_embed_dim,
        activation_fn=activation_fn,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=True,
    )
    if load_cache:
        cache_path = get_cache_path(
            model_name="mochi-1-preview",
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            dtype="bf16",
        )
        assert os.path.exists(
            cache_path
        ), "Cache path does not exist. Run test_mochi_transformer_model_caching first with the desired parallel config."
        start = time.time()
        cache_dict = load_cache_dict(cache_path)
        tt_model.from_cached_state_dict(cache_dict)
        end = time.time()
        logger.info(f"Time taken to load cached state dict: {end - start} seconds")
    else:
        start = time.time()
        tt_model.load_torch_state_dict(torch_model.state_dict())
        end = time.time()
        logger.info(f"Time taken to load state dict: {end - start} seconds")

    # Run TT model
    logger.info(
        f"Running TT model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}, timestep shape {timestep_input.shape}"
    )
    tt_spatial_out = tt_model(
        spatial=spatial_input,
        prompt=prompt_input,
        timestep=timestep_input,
        prompt_attention_mask=attention_mask,
    )

    # Run torch model
    logger.info(f"Running torch model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}")
    torch_spatial_out = torch_model(
        hidden_states=spatial_input,
        encoder_hidden_states=prompt_input,
        timestep=timestep_input,
        encoder_attention_mask=attention_mask,
        return_dict=False,
    )
    torch_spatial_out = torch_spatial_out[0]

    logger.info(f"Checking spatial outputs")
    assert_quality(torch_spatial_out, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MIN_RMSE)


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links",
    [
        [(1, 8), 1, 0, 1],
        [(2, 4), 1, 0, 1],
        [(2, 4), 0, 1, 1],
        [(4, 8), 1, 0, 4],
    ],
    ids=[
        "1x8sp1tp0",
        "2x4sp1tp0",
        "2x4sp0tp1",
        "4x8sp1tp0",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, T, H, W, prompt_seq"),
    [
        (1, 8, 40, 50, 118),
    ],
    ids=["short_seq"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_mochi_transformer_model_caching(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq: int,
) -> None:
    torch_dtype = torch.float32

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Model configuration
    patch_size = 2
    num_attention_heads = 24
    attention_head_dim = 128
    num_layers = 48
    pooled_projection_dim = 1536
    in_channels = 12
    text_embed_dim = 4096
    time_embed_dim = 256
    activation_fn = "swiglu"

    # Tight error bounds based on test config
    MIN_PCC = 0.992_500
    MIN_RMSE = 0.14

    torch_model = TorchMochiTransformer3DModel.from_pretrained(
        f"genmo/mochi-1-preview", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model.eval()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )

    cache_path = get_and_create_cache_path(
        model_name="mochi-1-preview",
        subfolder="transformer",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        dtype="bf16",
    )

    logger.info(f"Cache path {cache_path}")

    # Create TT model
    tt_model = MochiTransformer3DModel(
        patch_size=patch_size,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        num_layers=num_layers,
        pooled_projection_dim=pooled_projection_dim,
        in_channels=in_channels,
        text_embed_dim=text_embed_dim,
        time_embed_dim=time_embed_dim,
        activation_fn=activation_fn,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=True,
    )
    start = time.time()
    tt_model.load_torch_state_dict(torch_model.state_dict())
    end = time.time()
    logger.info(f"Time taken to load state dict: {end - start} seconds")

    start = time.time()
    cache_dict = tt_model.to_cached_state_dict(cache_path)
    save_cache_dict(cache_dict, cache_path)
    end = time.time()
    logger.info(f"Time taken to cache state dict: {end - start} seconds")

    start = time.time()
    del tt_model

    cache_model = MochiTransformer3DModel(
        patch_size=patch_size,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        num_layers=num_layers,
        pooled_projection_dim=pooled_projection_dim,
        in_channels=in_channels,
        text_embed_dim=text_embed_dim,
        time_embed_dim=time_embed_dim,
        activation_fn=activation_fn,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=True,
    )
    loaded_cache_dict = load_cache_dict(cache_path)
    cache_model.from_cached_state_dict(loaded_cache_dict)
    end = time.time()
    logger.info(f"Time taken to load cached state dict: {end - start} seconds")
