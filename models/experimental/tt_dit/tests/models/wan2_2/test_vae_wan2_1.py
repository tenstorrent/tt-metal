# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import time
from loguru import logger
from collections import defaultdict
from diffusers import AutoencoderKLWan

import ttnn
from ....utils.check import assert_quality
from ....layers.normalization import RMSNorm
from ....models.vae.vae_wan2_1 import (
    WanAttentionBlock,
    WanDecoder3d,
    WanDecoder,
    WanCausalConv3d,
    WanResidualBlock,
    WanMidBlock,
    WanResample,
    WanUpBlock,
)
from ....utils.conv3d import count_convs, conv_pad_in_channels, conv_pad_height, conv_unpad_height
from ....utils.tensor import bf16_tensor_2dshard
from ....parallel.manager import CCLManager
from ....parallel.config import VaeHWParallelConfig, ParallelFactor


def setup_hooks(model):
    """
    Set up forward and pre-forward hooks to track input/output shapes for all modules.

    Returns:
        tuple: (hook_handles, shapes_info) where shapes_info contains input/output shape data
    """
    shapes_info = defaultdict(dict)
    hook_handles = []

    def create_pre_forward_hook(name):
        def pre_forward_hook(module, input):
            if isinstance(input, tuple):
                input_shapes = [list(x.shape) if hasattr(x, "shape") else str(type(x)) for x in input]
            else:
                input_shapes = [list(input.shape)] if hasattr(input, "shape") else [str(type(input))]

            shapes_info[name]["input_shapes"] = input_shapes
            shapes_info[name]["input_dtypes"] = [
                str(x.dtype) if hasattr(x, "dtype") else str(type(x))
                for x in (input if isinstance(input, tuple) else [input])
            ]

        return pre_forward_hook

    def create_forward_hook(name):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                output_shapes = [list(x.shape) if hasattr(x, "shape") else str(type(x)) for x in output]
                output_dtypes = [str(x.dtype) if hasattr(x, "dtype") else str(type(x)) for x in output]
            else:
                output_shapes = [list(output.shape)] if hasattr(output, "shape") else [str(type(output))]
                output_dtypes = [str(output.dtype)] if hasattr(output, "dtype") else [str(type(output))]

            shapes_info[name]["output_shapes"] = output_shapes
            shapes_info[name]["output_dtypes"] = output_dtypes

        return forward_hook

    # Register hooks for all named modules
    for name, module in model.named_modules():
        if name:  # Skip the root module (empty name)
            # Register pre-forward hook
            pre_handle = module.register_forward_pre_hook(create_pre_forward_hook(name))
            hook_handles.append(pre_handle)

            # Register forward hook
            handle = module.register_forward_hook(create_forward_hook(name))
            hook_handles.append(handle)

    return hook_handles, shapes_info


def print_shapes_info(shapes_info):
    """
    Print the collected input/output shape information in a formatted way.
    """
    print("\n" + "=" * 100)
    print("MODULE INPUT/OUTPUT SHAPES ANALYSIS")
    print("=" * 100)

    for module_name, info in shapes_info.items():
        print(f"\nModule: {module_name}")
        print("-" * (len(module_name) + 8))

        if "input_shapes" in info:
            print(f"  Input shapes:  {info['input_shapes']}")
            print(f"  Input dtypes:  {info['input_dtypes']}")

        if "output_shapes" in info:
            print(f"  Output shapes: {info['output_shapes']}")
            print(f"  Output dtypes: {info['output_dtypes']}")

    print("=" * 100)


def test_autoencoder_kl_wan():
    """
    Test to construct a pretrained AutoencoderKLWan for wan2.2 14B t2v model and print it.
    """

    logger.info("Loading AutoencoderKLWan model for Wan2.2 14B T2V...")

    # Load the pretrained AutoencoderKLWan model for Wan2.2 14B T2V
    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    torch_dtype = torch.float32

    # Load VAE component from the Wan2.2 model
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch_dtype, trust_remote_code=True)
    vae.eval()

    logger.info("Successfully loaded AutoencoderKLWan model")

    # Print the model architecture
    print("\n" + "=" * 80)
    print("AutoencoderKLWan Model Architecture:")
    print("=" * 80)
    print(vae)
    print("=" * 80)

    # Print model configuration
    print(f"\nModel Configuration:")
    print(f"Model ID: {model_id}")
    print(f"Torch dtype: {torch_dtype}")
    print(f"Model type: {type(vae)}")
    print(f"Training mode: {vae.training}")
    print(f"vae.use_tiling: {vae.use_tiling}")

    # Verify model is the correct type
    assert isinstance(vae, AutoencoderKLWan), f"Expected AutoencoderKLWan, got {type(vae)}"
    assert not vae.training, "Model should be in evaluation mode"

    # Set up hooks to track input/output shapes
    logger.info("Setting up forward hooks to track input/output shapes...")
    hook_handles, shapes_info = setup_hooks(vae)

    vae.to("meta")

    # Create a sample input tensor to test the model
    # Typical VAE input shape for video: (batch, channels, frames, height, width)
    # Using smaller dimensions for testing
    B, C, T, H, W = 1, 16, 21, 90, 160
    sample_input = torch.randn(B, C, T, H, W, dtype=torch_dtype, device="meta")  # Small test input
    logger.info(f"Running forward pass with input shape: {sample_input.shape}")

    # Run forward pass through the encoder
    with torch.no_grad():
        # Run forward pass through the decoder

        decoded = vae.decode(sample_input)
        if hasattr(decoded, "sample"):
            reconstruction = decoded.sample
        else:
            reconstruction = decoded

        logger.info(f"Decoded reconstruction shape: {reconstruction.shape}")

    # Print the collected shape information
    print_shapes_info(shapes_info)


@pytest.mark.parametrize(
    ("B, C, T, H, W, images"),
    [
        (1, 384, 1, 90, 160, False),  # decoder.mid_block.resnets.0.norm1
        (1, 192, 2, 180, 320, False),  # decoder.up_blocks.1.resnets.0.norm1
        (1, 384, 2, 180, 320, False),  # decoder.up_blocks.1.resnets.0.norm2
        (1, 192, 4, 360, 640, False),  # decoder.up_blocks.2.resnets.0.norm1
        (1, 96, 4, 720, 1280, False),  # decoder.up_blocks.3.resnets.0.norm1
        (1, 384, 1, 90, 160, True),  # decoder.mid_block.attentions.0.norm
    ],
    ids=[
        "input_0",
        "input_1",
        "input_2",
        "input_3",
        "input_4",
        "input_5",
    ],
)
@pytest.mark.parametrize("mean, std", [(0, 1), (2, 3), (-2, -3)])
def test_wan_rmsnorm(device, B, C, T, H, W, images, mean, std):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    torch_dtype = torch.float32
    bias = False
    torch_model = WanRMS_norm(dim=C, images=images, bias=bias)
    torch_model.eval()

    tt_model = RMSNorm(
        embedding_dim=C,
        norm_eps=1e-6,
        norm_elementwise_affine=True,
        bias=bias,
        mesh_device=device,
    )
    state_dict = torch_model.state_dict()
    state_dict["weight"] = state_dict["gamma"].squeeze()  # remove broadcasting dimensions
    del state_dict["gamma"]
    tt_model.load_state_dict(state_dict)

    torch_input_tensor = torch.randn(B, C, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
    if images:
        torch_input_tensor = torch_input_tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)

    tt_input_tensor = ttnn.from_torch(tt_input_tensor, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    torch_output = torch_model(torch_input_tensor)
    tt_output = tt_model(tt_input_tensor)

    tt_output_torch = ttnn.to_torch(tt_output)
    if images:
        tt_output_torch = tt_output_torch.permute(0, 3, 1, 2)
    else:
        tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)

    assert_quality(torch_output, tt_output_torch, pcc=0.999_500)


@pytest.mark.parametrize(
    ("B, C, T, H, W"),
    [
        (1, 384, 1, 90, 160),  # decoder.mid_block.resnets.0.norm1
        (1, 384, 1, 60, 104),  # decoder.mid_block.resnets.0.norm1
    ],
    ids=[
        "720p",
        "480p",
    ],
)
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis",
    [
        ((1, 1), 0, 1),
        ((2, 4), 0, 1),
        ((2, 4), 1, 0),
        ((1, 8), 0, 1),
        ((1, 4), 1, 0),
        ((4, 8), 0, 1),
    ],
    ids=[
        "1x1_h0_w1",
        "2x4_h0_w1",
        "2x4_h1_w0",
        "1x8_h0_w1",
        "1x4_h1_w0",
        "4x8_h0_w1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_wan_attention(mesh_device, B, C, T, H, W, mean, std, h_axis, w_axis, reset_seeds):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock as TorchWanAttentionBlock

    torch_dtype = torch.float32
    torch_model = TorchWanAttentionBlock(dim=C)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )

    tt_model = WanAttentionBlock(
        dim=C,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(B, C, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
    tt_input_tensor, logical_h = conv_pad_height(tt_input_tensor, parallel_config.height_parallel.factor)
    if logical_h != tt_input_tensor.shape[2]:
        logger.info(f"padding from {logical_h} to {tt_input_tensor.shape[2]}")

    tt_input_tensor = bf16_tensor_2dshard(
        tt_input_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
    )

    torch_output = torch_model(torch_input_tensor)
    tt_output = tt_model(tt_input_tensor, logical_h=logical_h)

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )

    tt_output_torch = conv_unpad_height(tt_output_torch, logical_h)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)

    assert_quality(torch_output, tt_output_torch, pcc=0.999_980, relative_rmse=0.007)


@pytest.mark.parametrize(
    ("B, C_in, C_out, T, H, W, kernel_size, stride, padding"),
    [
        (1, 16, 384, 1, 90, 160, 3, 1, 1),  # decoder.conv_in
        (1, 384, 384, 1, 90, 160, 3, 1, 1),  # decoder.mid_block.resnets.0.conv1
        (1, 192, 384, 2, 180, 320, 3, 1, 1),  # decoder.up_blocks.1.resnets.0.conv1
        (1, 384, 384, 2, 180, 320, 3, 1, 1),  # decoder.up_blocks.1.resnets.0.conv2
        (1, 192, 192, 4, 360, 640, 3, 1, 1),  # decoder.up_blocks.2.resnets.0.conv1
        (1, 96, 96, 4, 720, 1280, 3, 1, 1),  # decoder.up_blocks.3.resnets.0.conv1
        (1, 96, 3, 4, 720, 1280, 3, 1, 1),  # decoder.conv_out
        (1, 384, 768, 1, 90, 160, (3, 1, 1), 1, (1, 0, 0)),  # decoder.up_blocks.0.upsamplers.0.time_conv
        (1, 384, 768, 2, 180, 320, (3, 1, 1), 1, (1, 0, 0)),  # decoder.up_blocks.0.upsamplers.0.time_conv
    ],
    ids=[
        "conv_0",
        "conv_1",
        "conv_2",
        "conv_3",
        "conv_4",
        "conv_5",
        "conv_6",
        "conv_7",
        "conv_8",
    ],
)
@pytest.mark.parametrize("cache_len", [None, 1, 2], ids=["cache_none", "cache_1", "cache_2"])
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis",
    [
        ((1, 1), 0, 1),
        ((2, 4), 0, 1),
        ((2, 4), 1, 0),
        ((1, 8), 0, 1),
        ((1, 4), 1, 0),
        ((4, 8), 0, 1),
    ],
    ids=[
        "1x1_h0_w1",
        "2x4_h0_w1",
        "2x4_h1_w0",
        "1x8_h0_w1",
        "1x4_h1_w0",
        "4x8_h0_w1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_wan_conv3d(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, stride, padding, cache_len, mean, std, h_axis, w_axis
):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d as TorchWanCausalConv3d

    torch_dtype = torch.float32
    torch_model = TorchWanCausalConv3d(
        in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, stride=stride, padding=padding
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )
    tt_model = WanCausalConv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        mesh_device=mesh_device,
        stride=stride,
        padding=padding,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(B, C_in, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
    tt_input_tensor = conv_pad_in_channels(tt_input_tensor)
    tt_input_tensor, logical_h = conv_pad_height(tt_input_tensor, parallel_config.height_parallel.factor)
    if logical_h != tt_input_tensor.shape[2]:
        logger.info(f"padding from {logical_h} to {tt_input_tensor.shape[2]}")
    tt_input_tensor = bf16_tensor_2dshard(
        tt_input_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
    )
    logger.info(f"torch_input_tensor.shape: {torch_input_tensor.shape}")
    logger.info(f"tt_input_tensor.shape: {tt_input_tensor.shape}")

    if cache_len is not None:
        torch_cache_tensor = torch.randn(B, C_in, cache_len, H, W, dtype=torch_dtype) * std + mean
        tt_cache_tensor = torch_cache_tensor.permute(0, 2, 3, 4, 1)
        tt_cache_tensor = conv_pad_in_channels(tt_cache_tensor)
        tt_cache_tensor, logical_h = conv_pad_height(tt_cache_tensor, parallel_config.height_parallel.factor)
        tt_cache_tensor = bf16_tensor_2dshard(
            tt_cache_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
        )
    else:
        torch_cache_tensor = tt_cache_tensor = None

    torch_output = torch_model(torch_input_tensor, cache_x=torch_cache_tensor)
    tt_output = tt_model(tt_input_tensor, cache_x_BTHWC=tt_cache_tensor, logical_h=logical_h)

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    if logical_h != tt_output_torch.shape[2]:
        logger.info(f"Checking that output padded portion is zeros")
        padding = tt_output_torch[:, :, logical_h:, :, :]
        assert torch.all(padding == 0.0), f"Padding must be zero, got {padding}"

    tt_output_torch = conv_unpad_height(tt_output_torch, logical_h)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)

    if tt_output_torch.shape != torch_output.shape:
        logger.warning(
            f"tt_output_torch.shape != torch_output.shape, got {tt_output_torch.shape} != {torch_output.shape}"
        )
        tt_output_torch = tt_output_torch[:, :C_out]
        logger.warning(f"Trimmed tt_output_torch to {tt_output_torch.shape}")

    assert_quality(torch_output, tt_output_torch, pcc=0.999_980, relative_rmse=0.007)


@pytest.mark.parametrize(
    ("B, in_dim, out_dim, T, H, W"),
    [
        (1, 384, 384, 1, 90, 160),  # decoder.mid_block.resnets.0
        (1, 192, 384, 2, 180, 320),  # decoder.up_blocks.1.resnets.0
        (1, 384, 384, 2, 180, 320),  # decoder.up_blocks.1.resnets.1
        (1, 192, 192, 4, 360, 640),  # decoder.up_blocks.2.resnets.0
        (1, 96, 96, 4, 720, 1280),  # decoder.up_blocks.3.resnets.0
    ],
    ids=[
        "conv_0",
        "conv_1",
        "conv_2",
        "conv_3",
        "conv_4",
    ],
)
@pytest.mark.parametrize("cache_len", [None, 1, 2], ids=["cache_none", "cache_1", "cache_2"])
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis",
    [
        ((1, 1), 0, 1),
        ((2, 4), 0, 1),
        ((2, 4), 1, 0),
        ((1, 8), 0, 1),
        ((1, 4), 1, 0),
        ((4, 8), 0, 1),
    ],
    ids=[
        "1x1_h0_w1",
        "2x4_h0_w1",
        "2x4_h1_w0",
        "1x8_h0_w1",
        "1x4_h1_w0",
        "4x8_h0_w1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_wan_residual_block(mesh_device, B, in_dim, out_dim, T, H, W, cache_len, mean, std, h_axis, w_axis):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock as TorchWanResidualBlock

    torch_dtype = torch.float32
    torch_model = TorchWanResidualBlock(
        in_dim=in_dim,
        out_dim=out_dim,
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )
    tt_model = WanResidualBlock(
        in_dim=in_dim,
        out_dim=out_dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(B, in_dim, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
    tt_input_tensor, logical_h = conv_pad_height(tt_input_tensor, parallel_config.height_parallel.factor)
    if logical_h != tt_input_tensor.shape[2]:
        logger.info(f"padding from {logical_h} to {tt_input_tensor.shape[2]}")
    tt_input_tensor = bf16_tensor_2dshard(
        tt_input_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
    )
    logger.info(f"torch_input_tensor.shape: {torch_input_tensor.shape}")
    logger.info(f"tt_input_tensor.shape: {tt_input_tensor.shape}")

    if cache_len is not None:
        torch_cache_tensor_1 = torch.randn(B, in_dim, cache_len, H, W, dtype=torch_dtype) * std + mean
        torch_cache_tensor_2 = torch.randn(B, out_dim, cache_len, H, W, dtype=torch_dtype) * std + mean
        torch_feat_cache = [torch_cache_tensor_1, torch_cache_tensor_2]
        torch_feat_idx = [0]

        tt_cache_tensor_1 = torch_cache_tensor_1.permute(0, 2, 3, 4, 1)
        tt_cache_tensor_2 = torch_cache_tensor_2.permute(0, 2, 3, 4, 1)
        tt_cache_tensor_1, _ = conv_pad_height(tt_cache_tensor_1, parallel_config.height_parallel.factor)
        tt_cache_tensor_2, _ = conv_pad_height(tt_cache_tensor_2, parallel_config.height_parallel.factor)
        tt_cache_tensor_1 = bf16_tensor_2dshard(
            tt_cache_tensor_1, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
        )
        tt_cache_tensor_2 = bf16_tensor_2dshard(
            tt_cache_tensor_2, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
        )
        tt_feat_cache = [tt_cache_tensor_1, tt_cache_tensor_2]
        tt_feat_idx = [0]
    else:
        torch_feat_cache = [None, None]
        torch_feat_idx = [0]
        tt_feat_cache = [None, None]
        tt_feat_idx = [0]

    torch_output = torch_model(
        torch_input_tensor,
        feat_cache=torch_feat_cache,
        feat_idx=torch_feat_idx,
    )

    tt_output = tt_model(
        tt_input_tensor,
        logical_h,
        feat_cache=tt_feat_cache,
        feat_idx=tt_feat_idx,
    )

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    tt_output_torch = conv_unpad_height(tt_output_torch, logical_h)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)

    logger.info(f"checking output")
    assert_quality(torch_output, tt_output_torch, pcc=0.999_900, relative_rmse=0.012)

    for i in range(len(tt_feat_cache)):
        tt_feat_cache[i] = ttnn.to_torch(
            tt_feat_cache[i],
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
        )
        tt_feat_cache[i] = conv_unpad_height(tt_feat_cache[i], logical_h)
        tt_feat_cache[i] = tt_feat_cache[i].permute(0, 4, 1, 2, 3)
        logger.info(f"checking feat_cache {i}")
        assert_quality(torch_feat_cache[i], tt_feat_cache[i], pcc=0.999_000, relative_rmse=0.04)


@pytest.mark.parametrize(
    ("B, dim, T, H, W"),
    [
        (1, 384, 1, 90, 160),  # decoder.mid_block.resnets.0
    ],
    ids=[
        "mid_block",
    ],
)
@pytest.mark.parametrize("cache_len", [None, 1, 2], ids=["cache_none", "cache_1", "cache_2"])
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis",
    [
        ((1, 1), 0, 1),
        ((2, 4), 0, 1),
        ((2, 4), 1, 0),
        ((1, 8), 0, 1),
        ((1, 4), 1, 0),
        ((4, 8), 0, 1),
    ],
    ids=[
        "1x1_h0_w1",
        "2x4_h0_w1",
        "2x4_h1_w0",
        "1x8_h0_w1",
        "1x4_h1_w0",
        "4x8_h0_w1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_wan_mid_block(mesh_device, B, dim, T, H, W, cache_len, mean, std, h_axis, w_axis):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanMidBlock as TorchWanMidBlock

    torch_dtype = torch.float32
    torch_model = TorchWanMidBlock(
        dim=dim,
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )
    tt_model = WanMidBlock(
        dim=dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    num_convs = count_convs(tt_model)

    torch_input_tensor = torch.randn(B, dim, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
    tt_input_tensor, logical_h = conv_pad_height(tt_input_tensor, parallel_config.height_parallel.factor)
    if logical_h != tt_input_tensor.shape[2]:
        logger.info(f"padding from {logical_h} to {tt_input_tensor.shape[2]}")
    tt_input_tensor = bf16_tensor_2dshard(
        tt_input_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
    )

    torch_feat_cache = []
    tt_feat_cache = []
    torch_feat_idx = [0]
    tt_feat_idx = [0]
    for i in range(num_convs):
        if cache_len is not None:
            torch_cache_tensor = torch.randn(B, dim, cache_len, H, W, dtype=torch_dtype) * std + mean
            torch_feat_cache.append(torch_cache_tensor)

            tt_cache_tensor = torch_cache_tensor.permute(0, 2, 3, 4, 1)
            tt_cache_tensor, _ = conv_pad_height(tt_cache_tensor, parallel_config.height_parallel.factor)
            tt_cache_tensor = bf16_tensor_2dshard(
                tt_cache_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
            )
            tt_feat_cache.append(tt_cache_tensor)

        else:
            torch_feat_cache.append(None)
            tt_feat_cache.append(None)

    torch_output = torch_model(
        torch_input_tensor,
        feat_cache=torch_feat_cache,
        feat_idx=torch_feat_idx,
    )

    tt_output = tt_model(
        tt_input_tensor,
        logical_h,
        feat_cache=tt_feat_cache,
        feat_idx=tt_feat_idx,
    )

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    tt_output_torch = conv_unpad_height(tt_output_torch, logical_h)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)

    logger.info(f"checking output")
    assert_quality(torch_output, tt_output_torch, pcc=0.999_900, relative_rmse=0.008)

    for i in range(len(tt_feat_cache)):
        tt_feat_cache[i] = ttnn.to_torch(
            tt_feat_cache[i],
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
        )
        tt_feat_cache[i] = conv_unpad_height(tt_feat_cache[i], logical_h)
        tt_feat_cache[i] = tt_feat_cache[i].permute(0, 4, 1, 2, 3)
        logger.info(f"checking feat_cache {i}")
        assert_quality(torch_feat_cache[i], tt_feat_cache[i], pcc=0.999_000, relative_rmse=0.016)


@pytest.mark.parametrize(
    ("B, dim, T, H, W, mode, upsample_out_dim"),
    [
        (1, 384, 1, 90, 160, "upsample3d", None),  # decoder.up_blocks.0.upsamplers.0
        (1, 384, 2, 180, 320, "upsample3d", None),  # decoder.up_blocks.1.upsamplers.0
        (1, 192, 4, 360, 640, "upsample2d", None),  # decoder.up_blocks.2.upsamplers.0
    ],
    ids=[
        "upsample_0",
        "upsample_1",
        "upsample_2",
    ],
)
@pytest.mark.parametrize("cache_len", [None, 1, 2], ids=["cache_none", "cache_1", "cache_2"])
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis",
    [
        ((1, 1), 0, 1),
        ((2, 4), 0, 1),
        ((2, 4), 1, 0),
        ((1, 8), 0, 1),
        ((1, 4), 1, 0),
        ((4, 8), 0, 1),
    ],
    ids=[
        "1x1_h0_w1",
        "2x4_h0_w1",
        "2x4_h1_w0",
        "1x8_h0_w1",
        "1x4_h1_w0",
        "4x8_h0_w1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_wan_resample(mesh_device, B, dim, T, H, W, mode, upsample_out_dim, cache_len, mean, std, h_axis, w_axis):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample as TorchWanResample

    torch_dtype = torch.float32
    torch_model = TorchWanResample(
        dim=dim,
        mode=mode,
        upsample_out_dim=upsample_out_dim,
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )
    tt_model = WanResample(
        dim=dim,
        mode=mode,
        upsample_out_dim=upsample_out_dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    num_convs = count_convs(tt_model)

    torch_input_tensor = torch.randn(B, dim, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
    tt_input_tensor, logical_h = conv_pad_height(tt_input_tensor, parallel_config.height_parallel.factor)
    if logical_h != tt_input_tensor.shape[2]:
        logger.info(f"padding from {logical_h} to {tt_input_tensor.shape[2]}")
    tt_input_tensor = bf16_tensor_2dshard(
        tt_input_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
    )

    torch_feat_cache = []
    tt_feat_cache = []
    torch_feat_idx = [0]
    tt_feat_idx = [0]
    for i in range(num_convs):
        if cache_len is not None:
            torch_cache_tensor = torch.randn(B, dim, cache_len, H, W, dtype=torch_dtype) * std + mean
            torch_feat_cache.append(torch_cache_tensor)

            tt_cache_tensor = torch_cache_tensor.permute(0, 2, 3, 4, 1)
            tt_cache_tensor, _ = conv_pad_height(tt_cache_tensor, parallel_config.height_parallel.factor)
            tt_cache_tensor = bf16_tensor_2dshard(
                tt_cache_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
            )
            tt_feat_cache.append(tt_cache_tensor)

        else:
            torch_feat_cache.append(None)
            tt_feat_cache.append(None)

    torch_output = torch_model(
        torch_input_tensor,
        feat_cache=torch_feat_cache,
        feat_idx=torch_feat_idx,
    )

    tt_output, new_logical_h = tt_model(
        tt_input_tensor,
        logical_h,
        feat_cache=tt_feat_cache,
        feat_idx=tt_feat_idx,
    )

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    tt_output_torch = conv_unpad_height(tt_output_torch, new_logical_h)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)

    logger.info(f"checking output")
    assert_quality(torch_output, tt_output_torch, pcc=0.999_900, relative_rmse=0.008)

    for i in range(len(tt_feat_cache)):
        logger.info(f"checking feat_cache {i}")
        if isinstance(tt_feat_cache[i], str) and tt_feat_cache[i] == "Rep":
            logger.info(f"feat_cache {i} is Rep")
            assert torch_feat_cache[i] == "Rep"
            continue
        tt_feat_cache[i] = ttnn.to_torch(
            tt_feat_cache[i],
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
        )
        tt_feat_cache[i] = tt_feat_cache[i][:, :, : torch_feat_cache[i].shape[3], :, :]
        tt_feat_cache[i] = tt_feat_cache[i].permute(0, 4, 1, 2, 3)
        assert_quality(torch_feat_cache[i], tt_feat_cache[i], pcc=0.999_000, relative_rmse=0.016)


@pytest.mark.parametrize(
    ("B, in_dim, out_dim, T, H, W, mode, num_res_blocks"),
    [
        (1, 384, 384, 1, 90, 160, "upsample3d", 2),  # decoder.up_blocks.0
        (1, 192, 384, 2, 180, 320, "upsample3d", 2),  # decoder.up_blocks.1
        (1, 192, 192, 4, 360, 640, "upsample2d", 2),  # decoder.up_blocks.2
        (1, 192, 192, 4, 720, 1280, None, 2),  # decoder.up_blocks.3 # OOM on 720p input
        # (1, 192, 192, 4, 480, 832, None, 2),  # decoder.up_blocks.3 on 480p input
    ],
    ids=[
        "upblock_0",
        "upblock_1",
        "upblock_2",
        "upblock_3",
    ],
)
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis",
    [
        ((1, 1), 0, 1),
        ((2, 4), 0, 1),
        ((2, 4), 1, 0),
        ((1, 8), 0, 1),
        ((1, 4), 1, 0),
        ((4, 8), 0, 1),
    ],
    ids=[
        "1x1_h0_w1",
        "2x4_h0_w1",
        "2x4_h1_w0",
        "1x8_h0_w1",
        "1x4_h1_w0",
        "4x8_h0_w1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_wan_upblock(mesh_device, B, in_dim, out_dim, T, H, W, mode, num_res_blocks, mean, std, h_axis, w_axis):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanUpBlock as TorchWanUpBlock

    torch_dtype = torch.float32
    torch_model = TorchWanUpBlock(
        in_dim=in_dim,
        out_dim=out_dim,
        num_res_blocks=num_res_blocks,
        upsample_mode=mode,
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )
    tt_model = WanUpBlock(
        in_dim=in_dim,
        out_dim=out_dim,
        num_res_blocks=num_res_blocks,
        upsample_mode=mode,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    num_convs = count_convs(tt_model)

    torch_feat_cache = [None for _ in range(num_convs)]
    tt_feat_cache = [None for _ in range(num_convs)]

    # Run 4 times to get models to create their own caches
    for i in range(4):
        torch_feat_idx = [0]
        tt_feat_idx = [0]
        logger.info(f"running test iteration {i}")

        torch_input_tensor = torch.randn(B, in_dim, T, H, W, dtype=torch_dtype) * std + mean
        tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
        tt_input_tensor, logical_h = conv_pad_height(tt_input_tensor, parallel_config.height_parallel.factor)
        if logical_h != tt_input_tensor.shape[2]:
            logger.info(f"padding from {logical_h} to {tt_input_tensor.shape[2]}")
        tt_input_tensor = bf16_tensor_2dshard(
            tt_input_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
        )

        logger.info(f"running torch model")
        torch_output = torch_model(
            torch_input_tensor,
            feat_cache=torch_feat_cache,
            feat_idx=torch_feat_idx,
        )

        logger.info(f"running tt model")
        tt_output, new_logical_h = tt_model(
            tt_input_tensor,
            logical_h,
            feat_cache=tt_feat_cache,
            feat_idx=tt_feat_idx,
        )

        concat_dims = [None, None]
        concat_dims[h_axis] = 2
        concat_dims[w_axis] = 3
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
        )
        tt_output_torch = conv_unpad_height(tt_output_torch, new_logical_h)
        tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)

        logger.info(f"checking output")
        assert_quality(torch_output, tt_output_torch, pcc=0.999_000, relative_rmse=0.02)

        for i in range(len(tt_feat_cache)):
            logger.info(f"checking feat_cache {i}")
            if isinstance(tt_feat_cache[i], str) and tt_feat_cache[i] == "Rep":
                logger.info(f"feat_cache {i} is Rep")
                assert torch_feat_cache[i] == "Rep"
                continue
            tt_feat_cache_back = ttnn.to_torch(
                tt_feat_cache[i],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims
                ),
            )
            tt_feat_cache_back = tt_feat_cache_back[:, :, : torch_feat_cache[i].shape[3], :, :]
            tt_feat_cache_back = tt_feat_cache_back.permute(0, 4, 1, 2, 3)
            assert_quality(torch_feat_cache[i], tt_feat_cache_back, pcc=0.999_000, relative_rmse=0.03)

        # Defrag the cache
        tt_feat_cache_host = []
        for i in range(len(tt_feat_cache)):
            if isinstance(tt_feat_cache[i], str) and tt_feat_cache[i] == "Rep":
                tt_feat_cache_host.append(tt_feat_cache[i])
            else:
                tt_feat_cache_host.append(
                    ttnn.to_torch(
                        tt_feat_cache[i],
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims
                        ),
                    )
                )
                ttnn.deallocate(tt_feat_cache[i])
        for i in range(len(tt_feat_cache)):
            if isinstance(tt_feat_cache[i], str) and tt_feat_cache[i] == "Rep":
                tt_feat_cache[i] = tt_feat_cache_host[i]
            else:
                tt_feat_cache[i] = bf16_tensor_2dshard(
                    tt_feat_cache_host[i],
                    mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    shard_mapping={h_axis: 2, w_axis: 3},
                )


@pytest.mark.parametrize(
    ("B, C, T, H, W"),
    [
        (1, 16, 1, 60, 104),  # 480p
        (1, 16, 1, 90, 160),  # 720p
    ],
    ids=[
        "480p",
        "720p",
    ],
)
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize("check_cache", [True])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis",
    [
        ((2, 4), 0, 1),
        ((1, 8), 0, 1),
        ((1, 4), 1, 0),
        ((4, 8), 0, 1),
    ],
    ids=[
        "2x4_h0_w1",
        "1x8_h0_w1",
        "1x4_h1_w0",
        "4x8_h0_w1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_wan_decoder3d(mesh_device, B, C, T, H, W, mean, std, h_axis, w_axis, check_cache, reset_seeds):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanDecoder3d as TorchWanDecoder3d

    # mesh_device.disable_and_clear_program_cache()
    torch_dtype = torch.float32
    base_dim = 96
    z_dim = 16
    dim_mult = [1, 2, 4, 4]
    num_res_blocks = 2
    attn_scales = []
    temperal_downsample = [False, True, True]
    temperal_upsample = temperal_downsample[::-1]
    dropout = 0.0
    out_channels = 3
    is_residual = False

    MIN_PCC = 0.99 if tuple(mesh_device.shape)[h_axis] == 4 else 0.997
    MAX_RMSE = 0.12 if tuple(mesh_device.shape)[h_axis] == 4 else 0.08

    torch_model = TorchWanDecoder3d(
        dim=base_dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_upsample=temperal_upsample,
        dropout=dropout,
        out_channels=out_channels,
        is_residual=is_residual,
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )

    tt_model = WanDecoder3d(
        dim=base_dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_upsample=temperal_upsample,
        out_channels=out_channels,
        is_residual=is_residual,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    num_convs = count_convs(tt_model)

    torch_feat_cache = [None for _ in range(num_convs)]
    tt_feat_cache = [None for _ in range(num_convs)]

    # Run 4 times to get models to create their own caches
    for i in range(3):
        torch_feat_idx = [0]
        tt_feat_idx = [0]
        logger.info(f"running test iteration {i}")

        torch_input_tensor = torch.randn(B, C, T, H, W, dtype=torch_dtype) * std + mean
        tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
        tt_input_tensor = conv_pad_in_channels(tt_input_tensor)
        tt_input_tensor, logical_h = conv_pad_height(tt_input_tensor, parallel_config.height_parallel.factor)
        tt_input_tensor = bf16_tensor_2dshard(
            tt_input_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
        )

        logger.info(f"running torch model")
        torch_output = torch_model(
            torch_input_tensor,
            feat_cache=torch_feat_cache,
            feat_idx=torch_feat_idx,
        )
        logger.info(f"torch output shape: {torch_output.shape}")

        logger.info(f"running tt model")
        tt_output, new_logical_h = tt_model(
            tt_input_tensor,
            logical_h,
            feat_cache=tt_feat_cache,
            feat_idx=tt_feat_idx,
        )

        concat_dims = [None, None]
        concat_dims[h_axis] = 2
        concat_dims[w_axis] = 3
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
        )
        tt_output_torch = conv_unpad_height(tt_output_torch, new_logical_h)
        tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)
        # Trim padding on output channels
        # DEBUG: REMOVING
        logger.info(f"trimming output channels from {tt_output_torch.shape} to {out_channels}")
        tt_output_torch = tt_output_torch[:, :out_channels]

        logger.info(f"checking output")
        assert_quality(torch_output, tt_output_torch, pcc=MIN_PCC, relative_rmse=MAX_RMSE)

        if check_cache:
            for i in range(len(tt_feat_cache)):
                logger.info(f"checking feat_cache {i}")
                if isinstance(tt_feat_cache[i], str) and tt_feat_cache[i] == "Rep":
                    logger.info(f"feat_cache {i} is Rep")
                    assert torch_feat_cache[i] == "Rep"
                    continue
                tt_feat_cache_back = ttnn.to_torch(
                    tt_feat_cache[i],
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims
                    ),
                )
                tt_feat_cache_back = tt_feat_cache_back.permute(0, 4, 1, 2, 3)
                if tt_feat_cache_back.shape[1] != torch_feat_cache[i].shape[1]:
                    tt_feat_cache_back = tt_feat_cache_back[:, : torch_feat_cache[i].shape[1]]
                    logger.warning(f"Trimmed tt_feat_cache_back to {tt_feat_cache_back.shape}")
                if tt_feat_cache_back.shape[3] != torch_feat_cache[i].shape[3]:
                    tt_feat_cache_back = tt_feat_cache_back[:, :, :, : torch_feat_cache[i].shape[3]]
                    logger.warning(f"Trimmed tt_feat_cache_back to {tt_feat_cache_back.shape}")
                logger.info(f"feat_cache {i} shape: {torch_feat_cache[i].shape}, {tt_feat_cache_back.shape}")
                try:
                    assert_quality(torch_feat_cache[i], tt_feat_cache_back, pcc=MIN_PCC, relative_rmse=MAX_RMSE)
                except Exception as e:
                    logger.error(
                        f"Error checking feat_cache {i}: {e}. Known issue where when T=2 in cache, T=0 is corrupted after cache is updated."
                    )
                    # breakpoint()
                    raise e

        # Defrag the cache
        tt_feat_cache_host = []
        for i in range(len(tt_feat_cache)):
            if isinstance(tt_feat_cache[i], str) and tt_feat_cache[i] == "Rep":
                tt_feat_cache_host.append(tt_feat_cache[i])
            else:
                tt_feat_cache_host.append(
                    ttnn.to_torch(
                        tt_feat_cache[i],
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims
                        ),
                    )
                )
                ttnn.deallocate(tt_feat_cache[i])
        for i in range(len(tt_feat_cache)):
            if isinstance(tt_feat_cache[i], str) and tt_feat_cache[i] == "Rep":
                tt_feat_cache[i] = tt_feat_cache_host[i]
            else:
                tt_feat_cache[i] = bf16_tensor_2dshard(
                    tt_feat_cache_host[i],
                    mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    shard_mapping={h_axis: 2, w_axis: 3},
                )


@pytest.mark.parametrize(
    ("B, C, H, W"),
    [
        (1, 16, 60, 104),  # 480p, 10 frames
        (1, 16, 90, 160),  # 720p, 10 frames
    ],
    ids=[
        "480p",
        "720p",
    ],
)
@pytest.mark.parametrize("T", [1, 10, 81], ids=["_1f", "10f", "81f"])
# @pytest.mark.parametrize("mean, std", [(0, 1), (2, 3), (-2, 3)])
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize("real_weights", [True, False], ids=["real_weights", "fake_weights"])
@pytest.mark.parametrize("skip_check", [True, False], ids=["skip_check", "check_output"])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    [
        ((1, 1), 0, 1, 1),
        ((2, 4), 0, 1, 1),
        ((2, 4), 1, 0, 1),
        ((1, 8), 0, 1, 1),
        ((1, 4), 1, 0, 1),
        ((4, 8), 0, 1, 4),
    ],
    ids=[
        "1x1_h0_w1",
        "2x4_h0_w1",
        "2x4_h1_w0",
        "1x8_h0_w1",
        "1x4_h1_w0",
        "4x8_h0_w1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_wan_decoder(mesh_device, B, C, T, H, W, mean, std, h_axis, w_axis, num_links, real_weights, skip_check):
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan as TorchAutoencoderKLWan

    torch_dtype = torch.float32
    base_dim = 96
    z_dim = 16
    dim_mult = [1, 2, 4, 4]
    num_res_blocks = 2
    attn_scales = []
    temperal_downsample = [False, True, True]
    dropout = 0.0
    out_channels = 3
    is_residual = False

    if real_weights:
        torch_model = TorchAutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae")
    else:
        torch_model = TorchAutoencoderKLWan(
            base_dim=base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            out_channels=out_channels,
            is_residual=is_residual,
        )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )
    tt_model = WanDecoder(
        base_dim=base_dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_downsample=temperal_downsample,
        out_channels=out_channels,
        is_residual=is_residual,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(B, C, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
    tt_input_tensor = conv_pad_in_channels(tt_input_tensor)
    tt_input_tensor, logical_h = conv_pad_height(tt_input_tensor, parallel_config.height_parallel.factor)
    if logical_h != tt_input_tensor.shape[2]:
        logger.info(f"padding from {logical_h} to {tt_input_tensor.shape[2]}")
    tt_input_tensor = bf16_tensor_2dshard(
        tt_input_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
    )

    logger.info(f"running tt model")
    start = time.time()
    tt_output, new_logical_h = tt_model(
        tt_input_tensor,
        logical_h,
    )

    concat_dims = [None, None]
    concat_dims[h_axis] = 3
    concat_dims[w_axis] = 4
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    logger.info(f"tt time taken: {time.time() - start}")
    logger.info(f"tt output shape: {tt_output_torch.shape}")

    if not skip_check:
        logger.info(f"running torch model")
        start = time.time()
        torch_output = torch_model.decode(
            torch_input_tensor,
            return_dict=False,
        )[0]
        logger.info(f"torch time taken: {time.time() - start}")
        logger.info(f"torch output shape: {torch_output.shape}")

        logger.info(f"checking output")
        if tt_output_torch.shape[1] != torch_output.shape[1]:
            logger.warning(f"Trimmed tt_output_torch to {tt_output_torch.shape}")
            tt_output_torch = tt_output_torch[:, : torch_output.shape[1]]
        if new_logical_h != tt_output_torch.shape[3]:
            tt_output_torch = tt_output_torch[:, :, :, :new_logical_h, :]
            logger.warning(f"Trimmed tt_output_torch to {tt_output_torch.shape}")
        assert_quality(torch_output, tt_output_torch, pcc=0.998_000, relative_rmse=0.08)
    else:
        logger.warning("Skipping check")
