# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

# This is the Conv3d used by the mochi models in huggingface's diffusers module.
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d as RefContextParallelConv3d
from loguru import logger

import ttnn

from ...layers.conv3d import ContextParallelConv3d as TtContextParallelConv3d
from ...parallel.config import MochiVAEParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality
from ...utils.tensor import print_tensor_mem_info

# Common test configurations
PCC_REQUIRED = 0.999

conv3d_args = {
    "context_parallel": True,
    "causal": True,
    "padding_mode": "replicate",
    "bias": True,
    "groups": 1,
}


def create_random_models(
    mesh_device,
    **model_args,
):
    """Initialize both reference and TT models."""
    # CogVideoXCausalConv3d takes a narrower kwarg set than genmo's class: no
    # `causal`/`context_parallel`/`bias`/`groups` knobs (causality is always on;
    # bias=True and groups=1 are hardcoded). Rename `padding_mode` -> `pad_mode`.
    ref_args = {
        "in_channels": model_args["in_channels"],
        "out_channels": model_args["out_channels"],
        "kernel_size": model_args["kernel_size"],
        "stride": model_args["stride"],
        "pad_mode": model_args["padding_mode"],
    }
    reference_model = RefContextParallelConv3d(**ref_args)

    # Create TT model
    tt_model = TtContextParallelConv3d(mesh_device=mesh_device, **model_args)
    # Diffusers nests the conv at `self.conv`; flatten to TT's `weight`/`bias`.
    ref_state = {k.removeprefix("conv."): v for k, v in reference_model.state_dict().items()}
    tt_model.load_torch_state_dict(ref_state)

    return reference_model, tt_model


def validate_outputs(tt_output, ref_output, test_name):
    """Validate and compare model outputs."""
    pcc, mse, mae = compute_metrics(ref_output, tt_output)
    logger.info(f"Output - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = pcc >= PCC_REQUIRED

    if passing:
        logger.info(f"{test_name} Passed!")
    else:
        logger.warning(f"{test_name} Failed!")
        logger.error(f"Output failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"{test_name} output does not meet PCC requirement {PCC_REQUIRED}"


# TODO: Merge this with the conv3d tests in test_vae_mochi.py
@torch.no_grad()
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride",
    [
        [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1)],
        #  [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1)],
    ],
    ids=["768"],  # , "512"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
def test_context_parallel_conv3d_forward_noshard(
    mesh_device, input_shape, out_channels, kernel_size, stride, reset_seeds
):
    input_channels = input_shape[1]
    model_args = conv3d_args.copy()
    model_args.update(
        {
            "in_channels": input_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "parallel_config": MochiVAEParallelConfig(
                time_parallel=ParallelFactor(factor=1, mesh_axis=1),
                h_parallel=ParallelFactor(factor=1, mesh_axis=0),
                w_parallel=ParallelFactor(factor=1, mesh_axis=1),
            ),
            "ccl_manager": CCLManager(
                mesh_device=mesh_device,
                topology=ttnn.Topology.Linear,
            ),
        }
    )

    reference_model, tt_model = create_random_models(mesh_device, **model_args)

    # Create input tensor (NCTHW format for PyTorch)
    torch_input = torch.randn(input_shape)

    # Convert to NTHWC format for TT
    tt_input_NTHWC = torch_input.permute(0, 2, 3, 4, 1)
    logger.info(f"Creating tensor: {tt_input_NTHWC.shape}")
    tt_input_NTHWC = ttnn.from_torch(
        tt_input_NTHWC,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=None,
    )
    tt_input_NTHWC = ttnn.to_layout(tt_input_NTHWC, ttnn.ROW_MAJOR_LAYOUT)

    logger.info("Run TtContextParallelConv3d forward")
    tt_output = tt_model(tt_input_NTHWC)

    # Convert TT output to torch tensor (from NTHWC to NCHW format)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # NTHWC -> NCHW

    # Get reference output
    logger.info("Run reference model forward")
    with torch.no_grad():
        ref_output, _ = reference_model(torch_input)  # returns (output, new_conv_cache)

    # Compare shapes
    logger.info(f"TT output shape: {tt_output_torch.shape}, Ref output shape: {ref_output.shape}")
    assert tt_output_torch.shape == ref_output.shape, "Output shapes don't match"

    assert_quality(ref_output, tt_output_torch, pcc=PCC_REQUIRED)


# Original test iterated over the following arches/device combos. Unclear how to
# iterate in this exact fashion within the current test framework.
# {
#     "N150": (1, 1),
#     "N300": (1, 2),
#     "T3K": (1, 1),
#     "TG": (8, 4)
# }
@torch.no_grad()
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride",
    [
        [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1)],
        [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1)],
        [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1)],
        [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1)],
    ],
    ids=["768", "512", "256", "128"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],  # TODO: make this the full suite of meshes want to test.
    indirect=True,
)
@pytest.mark.skip(
    reason=(
        "This test is currently busted. "
        "Some  tt tensors hang on calls to from_torch,"
        "The sharding is also not quite right as written. "
        "Fix this test when we bring up the mochi model tests, as "
        "test_vae_mochi.py has a lot of overlap and we can consolidate them."
    )
)
def test_context_parallel_conv3d_forward(mesh_device, input_shape, out_channels, kernel_size, stride, reset_seeds):
    """Test complete forward pass of TtContextParallelConv3d."""
    input_channels = input_shape[1]

    model_args = conv3d_args.copy()
    model_args.update(
        {
            "in_channels": input_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            # TODO: see test_vae_mochi.py for how to call these with various mesh device shapes
            # "parallel_config": MochiVAEParallelConfig(
            #     time_parallel=ParallelFactor(factor=4, mesh_axis=1),
            #     h_parallel=ParallelFactor(factor=1, mesh_axis=0),
            #     w_parallel=ParallelFactor(factor=1, mesh_axis=1),
            # ),
            # "ccl_manager": CCLManager(
            #     mesh_device=mesh_device,
            #     topology=ttnn.Topology.Linear,
            # ),
        }
    )
    # Create the models
    reference_model, tt_model = create_random_models(mesh_device, **model_args)

    # Create input tensor (NCTHW format for PyTorch)
    torch_input = torch.randn(input_shape)

    # Convert to NTHWC format for TT
    tt_input_NTHWC = torch_input.permute(0, 2, 3, 4, 1)

    # TODO: see test_vae_mochi.py for how to set this up.
    dims = [None, 1]
    mesh_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [
                ttnn.PlacementReplicate() if dims[0] is None else ttnn.PlacementShard(dims[0]),
                ttnn.PlacementReplicate() if dims[1] is None else ttnn.PlacementShard(dims[1]),
            ],
            ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1]),
        ),
    )

    tt_input_NTHWC = ttnn.from_torch(
        tt_input_NTHWC,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    tt_input_NTHWC = ttnn.to_layout(tt_input_NTHWC, ttnn.ROW_MAJOR_LAYOUT)

    logger.info("Run TtContextParallelConv3d forward")
    tt_output = tt_model(tt_input_NTHWC)

    # Convert TT output to torch tensor (from NTHWC to NCHW format)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # NTHWC -> NCHW

    # Get reference output
    logger.info("Run reference model forward")
    with torch.no_grad():
        ref_output, _ = reference_model(torch_input)  # returns (output, new_conv_cache)

    # Compare shapes
    logger.info(f"TT output shape: {tt_output_torch.shape}, Ref output shape: {ref_output.shape}")
    assert tt_output_torch.shape == ref_output.shape, "Output shapes don't match"

    assert_quality(ref_output, tt_output_torch, pcc=PCC_REQUIRED)


@torch.no_grad()
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride",
    [
        [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1)],
    ],
    ids=["512"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.skip(reason="This allocation hangs on blackhole. TODO Debug and fix in the future.")
def test_sharding(mesh_device, input_shape, out_channels, kernel_size, stride):
    # Create input tensor (NCTHW format for PyTorch)
    torch_input = torch.randn(input_shape)

    logger.info(f"initial shape {torch_input.shape}")
    # Convert to NTHWC format for TT
    tt_input_NTHWC = torch_input.permute(0, 2, 3, 4, 1).contiguous()

    logger.info(f"initial reshape {tt_input_NTHWC.shape}")
    logger.info(f"contiguous? : {tt_input_NTHWC.is_contiguous()}")

    tt_input_NTHWC = ttnn.from_torch(
        tt_input_NTHWC,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=None,
    )
    logger.info("Create DONE!")
    print_tensor_mem_info(tt_input_NTHWC)

    logger.info("reorder")
    tt_input_NTHWC_rm = ttnn.to_layout(tt_input_NTHWC, ttnn.ROW_MAJOR_LAYOUT)
    print_tensor_mem_info(tt_input_NTHWC_rm)


D2S_C_IN = 32
D2S_KERNEL = (3, 3, 3)
D2S_PADDING = (1, 1, 1)
D2S_INPUT_THW = (6, 8, 8)


def _run_d2s_conv3d(device, c_out, c_out_block, **kwargs):
    torch.manual_seed(0)
    x = torch.randn(1, D2S_C_IN, *D2S_INPUT_THW)
    # Output channels are read as (p1, p2, p3, C) by the depth-to-space path.
    conv = torch.nn.Conv3d(D2S_C_IN, c_out, D2S_KERNEL, padding=D2S_PADDING)

    config = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=2,
        H_out_block=4,
        W_out_block=4,
        C_out_block=c_out_block,
        C_in_block=D2S_C_IN,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )
    tt_x = ttnn.from_torch(x.permute(0, 2, 3, 4, 1), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_w = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=ttnn.from_torch(conv.weight.data, dtype=ttnn.bfloat16),
        groups=1,
        C_in_block=D2S_C_IN,
        alignment=32,
        device=device,
    )
    tt_b = ttnn.from_torch(conv.bias.data.reshape(1, -1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = ttnn.experimental.conv3d(
        input_tensor=tt_x,
        weight_tensor=tt_w,
        bias_tensor=tt_b,
        device=device,
        dtype=ttnn.bfloat16,
        output_channels=c_out,
        kernel_size=D2S_KERNEL,
        stride=(1, 1, 1),
        padding=D2S_PADDING,
        padding_mode="zeros",
        groups=1,
        config=config,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        **kwargs,
    )
    return ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])


def _ref_depth_to_space(out, d2s, trim_t):
    # (N, T, H, W, p1 * p2 * p3 * C) -> (N, T * p1 - trim_t, H * p2, W * p3, C)
    p1, p2, p3 = d2s
    N, T, H, W, C_out = out.shape
    C = C_out // (p1 * p2 * p3)
    x = out.reshape(N, T, H, W, p1, p2, p3, C).permute(0, 1, 4, 2, 5, 3, 6, 7)
    return x.reshape(N, T * p1, H * p2, W * p3, C)[:, trim_t:]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "d2s, trim_t, C, c_out_block",
    [
        ((2, 2, 2), 1, 32, 256),  # all 8 groups in one C_out block
        ((2, 2, 2), 1, 64, 32),  # each group spans 2 C_out blocks
    ],
    ids=["s222_trim1", "group_spans_blocks"],
)
def test_conv3d_depth_to_space_bit_exact(mesh_device, d2s, trim_t, C, c_out_block):
    c_out = C * d2s[0] * d2s[1] * d2s[2]

    ref = _run_d2s_conv3d(mesh_device, c_out, c_out_block)
    out = _run_d2s_conv3d(mesh_device, c_out, c_out_block, output_depth_to_space=list(d2s), output_trim_t_front=trim_t)

    expected = _ref_depth_to_space(ref, d2s, trim_t)
    assert out.shape == expected.shape
    assert torch.equal(out, expected)
