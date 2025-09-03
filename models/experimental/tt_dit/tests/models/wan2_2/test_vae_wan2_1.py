# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from collections import defaultdict
from diffusers import AutoencoderKLWan

import ttnn
from ....utils.check import assert_quality
from ....layers.normalization import RMSNorm
from ....models.vae.vae_wan2_1 import (
    WanAttentionBlock,
    WanCausalConv3d,
    WanResidualBlock,
)


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
    ],
)
@pytest.mark.parametrize("mean, std", [(0, 1), (2, 3), (-2, -3)])
def test_wan_attention(device, B, C, T, H, W, mean, std):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock as TorchWanAttentionBlock

    torch_dtype = torch.float32
    torch_model = TorchWanAttentionBlock(dim=C)
    torch_model.eval()

    tt_model = WanAttentionBlock(
        dim=C,
        mesh_device=device,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(B, C, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)

    tt_input_tensor = ttnn.from_torch(tt_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    torch_output = torch_model(torch_input_tensor)
    tt_output = tt_model(tt_input_tensor)

    tt_output_torch = ttnn.to_torch(tt_output)
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
        (1, 384, 768, 1, 90, 160, (3, 1, 1), 1, 1),  # decoder.up_blocks.0.upsamplers.0.time_conv
        (1, 384, 768, 2, 180, 320, (3, 1, 1), 1, 1),  # decoder.up_blocks.0.upsamplers.0.time_conv
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
@pytest.mark.parametrize("mean, std", [(0, 1), (2, 3), (-2, 3)])
def test_wan_conv3d(device, B, C_in, C_out, T, H, W, kernel_size, stride, padding, cache_len, mean, std):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d as TorchWanCausalConv3d

    torch_dtype = torch.float32
    torch_model = TorchWanCausalConv3d(
        in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, stride=stride, padding=padding
    )
    torch_model.eval()

    tt_model = WanCausalConv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        mesh_device=device,
        stride=stride,
        padding=padding,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(B, C_in, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
    tt_input_tensor = ttnn.from_torch(tt_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    if cache_len is not None:
        torch_cache_tensor = torch.randn(B, C_in, cache_len, H, W, dtype=torch_dtype) * std + mean
        tt_cache_tensor = torch_cache_tensor.permute(0, 2, 3, 4, 1)
        tt_cache_tensor = ttnn.from_torch(
            tt_cache_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )
    else:
        torch_cache_tensor = tt_cache_tensor = None

    torch_output = torch_model(torch_input_tensor, cache_x=torch_cache_tensor)
    tt_output = tt_model(tt_input_tensor, cache_x_BTHWC=tt_cache_tensor)

    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)

    if tt_output_torch.shape != torch_output.shape:
        logger.warning(
            f"tt_output_torch.shape != torch_output.shape, got {tt_output_torch.shape} != {torch_output.shape}"
        )
        tt_output_torch = tt_output_torch[:, :C_out]
        logger.warning(f"Trimmed tt_output_torch to {tt_output_torch.shape}")

    assert_quality(torch_output, tt_output_torch, pcc=0.999_990, relative_rmse=0.005)


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
@pytest.mark.parametrize("mean, std", [(0, 1), (2, 3), (-2, 3)])
def test_wan_residual_block(device, B, in_dim, out_dim, T, H, W, cache_len, mean, std):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock as TorchWanResidualBlock

    torch_dtype = torch.float32
    torch_model = TorchWanResidualBlock(
        in_dim=in_dim,
        out_dim=out_dim,
    )
    torch_model.eval()

    tt_model = WanResidualBlock(
        in_dim=in_dim,
        out_dim=out_dim,
        mesh_device=device,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(B, in_dim, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
    tt_input_tensor = ttnn.from_torch(tt_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    if cache_len is not None:
        torch_cache_tensor_1 = torch.randn(B, in_dim, cache_len, H, W, dtype=torch_dtype) * std + mean
        torch_cache_tensor_2 = torch.randn(B, out_dim, cache_len, H, W, dtype=torch_dtype) * std + mean
        torch_feat_cache = [torch_cache_tensor_1, torch_cache_tensor_2]
        torch_feat_idx = [0]

        tt_cache_tensor_1 = torch_cache_tensor_1.permute(0, 2, 3, 4, 1)
        tt_cache_tensor_2 = torch_cache_tensor_2.permute(0, 2, 3, 4, 1)
        tt_cache_tensor_1 = ttnn.from_torch(
            tt_cache_tensor_1, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_cache_tensor_2 = ttnn.from_torch(
            tt_cache_tensor_2, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
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
        feat_cache=tt_feat_cache,
        feat_idx=tt_feat_idx,
    )

    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)

    logger.info(f"checking output")
    assert_quality(torch_output, tt_output_torch, pcc=0.999_900, relative_rmse=0.0001)

    for i in range(len(tt_feat_cache)):
        tt_feat_cache[i] = ttnn.to_torch(tt_feat_cache[i])
        tt_feat_cache[i] = tt_feat_cache[i].permute(0, 4, 1, 2, 3)
        logger.info(f"checking feat_cache {i}")
        assert_quality(torch_feat_cache[i], tt_feat_cache[i], pcc=0.999_999, relative_rmse=0.001)
