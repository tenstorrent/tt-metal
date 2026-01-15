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
from ....models.transformers.wan2_2.transformer_wan import WanTransformerBlock, WanTransformer3DModel
from ....parallel.manager import CCLManager
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....utils.padding import pad_vision_seq_parallel
from ....utils.cache import get_cache_path, get_and_create_cache_path, save_cache_dict, load_cache_dict
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.test import ring_params, line_params
from diffusers import WanTransformer3DModel as TorchWanTransformer3DModel


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, False],
        [(2, 4), (2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 1, 0, 1, line_params, ttnn.Topology.Linear, True],
        # WH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False],
    ],
    ids=[
        "2x2sp0tp1",
        "2x4sp0tp1",
        "2x4sp1tp0",
        "wh_4x8sp1tp0",
        "bh_4x8sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B, T, H, W, prompt_seq_len"),
    [
        (1, 31, 40, 80, 118),  # 5B-720p
        (1, 21, 60, 104, 118),  # 14B-480p
        (1, 21, 90, 160, 118),  # 14B-720p
    ],
    ids=["5b-720p", "14b-480p", "14b-720p"],
)
def test_wan_transformer_block(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    torch_dtype = torch.float32
    parent_mesh_device = mesh_device
    mesh_device = parent_mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Wan2.2 Model configuration
    dim = 5120
    ffn_dim = 13824
    num_attention_heads = 40
    attention_head_dim = dim // num_attention_heads
    cross_attention_norm = True
    eps = 1e-6
    patch_size = (1, 2, 2)
    in_channels = 16
    p_t, p_h, p_w = patch_size
    patch_F, patch_H, patch_W = T // p_t, H // p_h, W // p_w
    spatial_seq_len = patch_F * patch_H * patch_W
    layer_id = 0

    # Tight error bounds based on test config
    MIN_PCC = 0.999_500
    MAX_RMSE = 0.032

    # Load Wan2.2-T2V-14B model from HuggingFace
    parent_torch_model = TorchWanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer", torch_dtype=torch_dtype, trust_remote_code=True
    )
    torch_model = parent_torch_model.blocks[layer_id]
    torch_model.eval()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=topology,
    )

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    # Create TT model
    tt_model = WanTransformerBlock(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_attention_heads,
        cross_attention_norm=cross_attention_norm,
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
    prompt_input = torch.randn((B, prompt_seq_len, dim), dtype=torch_dtype)
    temb_input = torch.randn((B, 6, dim), dtype=torch_dtype)

    # Create ROPE embeddings
    rope_cos = torch.randn(B, spatial_seq_len, 1, attention_head_dim // 2)
    rope_sin = torch.randn(B, spatial_seq_len, 1, attention_head_dim // 2)

    torch_rope_cos, torch_rope_sin = stack_cos_sin(rope_cos, rope_sin)

    rope_cos_stack = torch_rope_cos.permute(0, 2, 1, 3)
    rope_sin_stack = torch_rope_sin.permute(0, 2, 1, 3)

    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    rope_cos_padded = pad_vision_seq_parallel(rope_cos_stack, num_devices=sp_factor)
    rope_sin_padded = pad_vision_seq_parallel(rope_sin_stack, num_devices=sp_factor)

    # Sequence fractured spatial
    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    # Replicated prompt
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
    # Replicated time embedding
    tt_temb = bf16_tensor(temb_input.unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)

    # Rope cos and sin sequence fractured and head fractured
    tt_rope_cos = bf16_tensor(rope_cos_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_rope_sin = bf16_tensor(rope_sin_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)

    # Create transformation matrix for RoPE
    trans_mat = get_rot_transformation_mat()
    tt_trans_mat = bf16_tensor(trans_mat, device=mesh_device)

    # Run TT model
    logger.info(
        f"Running TT model with spatial shape {tt_spatial.shape}, prompt shape {tt_prompt.shape}, rope_cos shape {tt_rope_cos.shape}, rope_sin shape {tt_rope_sin.shape}"
    )
    tt_spatial_out = tt_model(
        spatial_1BND=tt_spatial,
        prompt_1BLP=tt_prompt,
        temb_1BTD=tt_temb,
        N=spatial_seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
    )

    spatial_concat_dims = [None, None]
    spatial_concat_dims[sp_axis] = 2
    spatial_concat_dims[tp_axis] = 3
    tt_spatial_out = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=spatial_concat_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    tt_spatial_out = tt_spatial_out[:, :, :spatial_seq_len, :]

    # Run torch model
    logger.info(f"Running torch model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}")

    with torch.no_grad():
        torch_spatial_out = torch_model(
            hidden_states=spatial_input,
            encoder_hidden_states=prompt_input,
            temb=temb_input,
            rotary_emb=[torch_rope_cos, torch_rope_sin],
        )

    logger.info(f"Checking spatial outputs")
    assert_quality(torch_spatial_out, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)


@pytest.mark.parametrize(
    "dit_unit_test",
    [{"1": True, "0": False}.get(os.environ.get("DIT_UNIT_TEST"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, False],
        [(2, 4), (2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 1, 0, 1, line_params, ttnn.Topology.Linear, True],
        # WH (ring) on 4x8
        [(4, 8), (4, 8), 0, 1, 4, ring_params, ttnn.Topology.Ring, True],
        [(4, 8), (4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 0, 1, 2, line_params, ttnn.Topology.Linear, False],
        [(4, 8), (4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear, False],
    ],
    ids=[
        "2x2sp0tp1",
        "2x4sp0tp1",
        "2x4sp1tp0",
        "wh_4x8sp0tp1",
        "wh_4x8sp1tp0",
        "bh_4x8sp0tp1",
        "bh_4x8sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B, T, H, W, prompt_seq_len"),
    [
        (1, 8, 40, 50, 118),  # small input
        (1, 31, 40, 80, 118),  # 5B-720p
        (1, 21, 60, 104, 118),  # 14B-480p
        (1, 21, 90, 160, 118),  # 14B-720p
    ],
    ids=["short_seq", "5b-720p", "14b-480p", "14b-720p"],
)
@pytest.mark.parametrize("load_cache", [True, False], ids=["yes_load_cache", "no_load_cache"])
def test_wan_transformer_model(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    load_cache: bool,
    topology: ttnn.Topology,
    is_fsdp: bool,
    dit_unit_test: bool,
) -> None:
    torch_dtype = torch.float32

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Wan2.2 Model configuration
    patch_size = (1, 2, 2)
    num_attention_heads = 40
    dim = 5120
    in_channels = 16
    out_channels = 16
    text_dim = 4096
    freq_dim = 256
    ffn_dim = 13824
    num_layers = 40
    cross_attn_norm = True
    eps = 1e-6
    rope_max_seq_len = 1024

    # Tight error bounds based on test config
    MIN_PCC = 0.992_000
    MAX_RMSE = 0.15

    if dit_unit_test:
        torch_model = TorchWanTransformer3DModel(num_layers=1)
        num_layers = torch_model.config.num_layers
    else:
        torch_model = TorchWanTransformer3DModel.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer", torch_dtype=torch_dtype, trust_remote_code=True
        )
    torch_model.eval()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=topology,
    )

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    torch.manual_seed(0)
    # Create input tensors
    spatial_input = torch.randn((B, in_channels, T, H, W), dtype=torch_dtype)
    prompt_input = torch.randn((B, prompt_seq_len, text_dim), dtype=torch_dtype)
    timestep_input = torch.randint(0, 1000, (B,), dtype=torch_dtype)

    # Create TT model
    tt_model = WanTransformer3DModel(
        patch_size=patch_size,
        num_heads=num_attention_heads,
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        text_dim=text_dim,
        freq_dim=freq_dim,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        cross_attn_norm=cross_attn_norm,
        eps=eps,
        rope_max_seq_len=rope_max_seq_len,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )

    if load_cache:
        cache_path = get_cache_path(
            model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            dtype="bf16",
        )
        assert os.path.exists(
            cache_path
        ), "Cache path does not exist. Run test_wan_transformer_model_caching first with the desired parallel config."
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
    )

    del tt_model

    # Run torch model
    logger.info(f"Running torch model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}")

    with torch.no_grad():
        torch_spatial_out = torch_model(
            hidden_states=spatial_input,
            encoder_hidden_states=prompt_input,
            timestep=timestep_input,
            return_dict=False,
        )
    torch_spatial_out = torch_spatial_out[0]

    logger.info(f"Checking spatial outputs")
    assert_quality(torch_spatial_out, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp",
    [
        [(2, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, False],
        [(2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True],
        # WH (ring) on 4x8
        [(4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True],
        # BH (linear) on 4x8
        [(4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear, False],
    ],
    ids=[
        "2x2sp0tp1",
        "2x4sp0tp1",
        "wh_4x8sp1tp0",
        "bh_4x8sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("subfolder", ["transformer", "transformer_2"], ids=["transformer_1", "transformer_2"])
def test_wan_transformer_model_caching(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    subfolder: str,
    topology: ttnn.Topology,
    is_fsdp: bool,
) -> None:
    torch_dtype = torch.float32

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Wan2.2 Model configuration
    patch_size = (1, 2, 2)
    num_attention_heads = 40
    dim = 5120
    in_channels = 16
    out_channels = 16
    text_dim = 4096
    freq_dim = 256
    ffn_dim = 13824
    num_layers = 40
    cross_attn_norm = True
    eps = 1e-6
    rope_max_seq_len = 1024

    # Tight error bounds based on test config
    MIN_PCC = 0.992_500
    MIN_RMSE = 0.15

    torch_model = TorchWanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder=subfolder, torch_dtype=torch_dtype, trust_remote_code=True
    )
    torch_model.eval()

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=topology,
    )

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    cache_path = get_and_create_cache_path(
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        subfolder=subfolder,
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        dtype="bf16",
    )

    # Create TT model
    tt_model = WanTransformer3DModel(
        patch_size=patch_size,
        num_heads=num_attention_heads,
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        text_dim=text_dim,
        freq_dim=freq_dim,
        ffn_dim=ffn_dim,
        cross_attn_norm=cross_attn_norm,
        eps=eps,
        rope_max_seq_len=rope_max_seq_len,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
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

    cache_model = WanTransformer3DModel(
        patch_size=patch_size,
        num_heads=num_attention_heads,
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        text_dim=text_dim,
        freq_dim=freq_dim,
        ffn_dim=ffn_dim,
        cross_attn_norm=cross_attn_norm,
        eps=eps,
        rope_max_seq_len=rope_max_seq_len,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    loaded_cache_dict = load_cache_dict(cache_path)
    cache_model.from_cached_state_dict(loaded_cache_dict)
    end = time.time()
    logger.info(f"Time taken to load cached state dict: {end - start} seconds")
