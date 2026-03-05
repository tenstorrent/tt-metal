# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Minimal reproduction test for Welford algorithm precision bug on Blackhole.

This test demonstrates that DRAM GroupNorm with Welford algorithm enabled
produces incorrect results on Blackhole hardware.

Issue: ttnn.group_norm with use_welford=True has numerical precision issues
on Blackhole's DRAM architecture (8 banks vs Wormhole's 12 banks).

To run this test:
    TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="7,7" pytest \
        models/experimental/stable_diffusion_xl_base/vae/tests/pcc/test_welford_bug_reproduction.py -v

Expected behavior:
    - test_resnetblock2d_welford_disabled: PASSES (PCC > 0.99)
    - test_resnetblock2d_welford_enabled: FAILS on Blackhole (PCC varies, worst ~0.83)
"""

import gc
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.vae.tt.model_configs import load_vae_model_optimisations
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from diffusers import AutoencoderKL
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random, is_blackhole
from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_conv_params,
    prepare_linear_params,
)
from models.experimental.stable_diffusion_xl_base.vae.tt.vae_utility import (
    get_DRAM_GN_shape,
)


class TtResnetBlock2DWithWelford(LightweightModule):
    """
    Copy of TtResnetBlock2D that ALWAYS uses Welford algorithm (even on Blackhole).
    This is used to reproduce the Welford precision bug.
    """

    def __init__(
        self, device, state_dict, module_path, model_config, conv_shortcut=False, debug_mode=False, force_welford=False
    ):
        super().__init__()

        self.device = device
        self.force_welford = force_welford  # If True, always use Welford (to reproduce bug)

        # fixed for ResnetBlock
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1
        self.debug_mode = debug_mode

        self.norm_groups = 32
        self.norm_eps = 1e-6

        # loading weights
        self.norm_weights_1 = state_dict[f"{module_path}.norm1.weight"]
        self.norm_bias_1 = state_dict[f"{module_path}.norm1.bias"]

        conv_weights_1 = state_dict[f"{module_path}.conv1.weight"]
        conv_bias_1 = state_dict[f"{module_path}.conv1.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        self.norm_weights_2 = state_dict[f"{module_path}.norm2.weight"]
        self.norm_bias_2 = state_dict[f"{module_path}.norm2.bias"]

        conv_weights_2 = state_dict[f"{module_path}.conv2.weight"]
        conv_bias_2 = state_dict[f"{module_path}.conv2.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if conv_shortcut:
            conv_weights_3 = state_dict[f"{module_path}.conv_shortcut.weight"].squeeze()
            conv_bias_3 = state_dict[f"{module_path}.conv_shortcut.bias"]

        (
            self.groupnorm_config_1,
            self.groupnorm_memory_config_1,
            self.input_mask_1,
            self.input_negative_mask_1,
            self.gamma_t_1,
            self.beta_t_1,
        ) = model_config.get_groupnorm_params(
            f"{module_path}.norm1", self.norm_weights_1, self.norm_bias_1, self.norm_groups, device
        )
        assert (
            self.groupnorm_memory_config_1 == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            or self.groupnorm_memory_config_1 == ttnn.DRAM_MEMORY_CONFIG
        ), "Only L1_BLOCK_SHARDED_MEMORY_CONFIG and DRAM_MEMORY_CONFIG is supported for GN"

        if self.groupnorm_memory_config_1 == ttnn.DRAM_MEMORY_CONFIG:
            N, C, H, W = get_DRAM_GN_shape(module_path, 1)
            torch_reciprocals = ttnn.create_group_norm_reciprocals(
                N, C, H, W, self.norm_groups, self.groupnorm_config_1["core_grid"]
            )
            self.reciprocals_tensor_1 = ttnn.from_torch(
                torch_reciprocals,
                dtype=ttnn.DataType.FLOAT32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        (
            self.groupnorm_config_2,
            self.groupnorm_memory_config_2,
            self.input_mask_2,
            self.input_negative_mask_2,
            self.gamma_t_2,
            self.beta_t_2,
        ) = model_config.get_groupnorm_params(
            f"{module_path}.norm2", self.norm_weights_2, self.norm_bias_2, self.norm_groups, device
        )
        assert (
            self.groupnorm_memory_config_2 == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            or self.groupnorm_memory_config_2 == ttnn.DRAM_MEMORY_CONFIG
        ), "Only L1_BLOCK_SHARDED_MEMORY_CONFIG and DRAM_MEMORY_CONFIG is supported for GN"

        if self.groupnorm_memory_config_2 == ttnn.DRAM_MEMORY_CONFIG:
            N, C, H, W = get_DRAM_GN_shape(module_path, 2)
            torch_reciprocals = ttnn.create_group_norm_reciprocals(
                N, C, H, W, self.norm_groups, self.groupnorm_config_2["core_grid"]
            )
            self.reciprocals_tensor_2 = ttnn.from_torch(
                torch_reciprocals,
                dtype=ttnn.DataType.FLOAT32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.compute1_config = model_config.get_conv_compute_config(module_path=f"{module_path}.conv1")
        self.conv1_config = model_config.get_conv_config(conv_path=f"{module_path}.conv1")
        (
            self.tt_conv1_weights,
            self.tt_conv1_bias,
            self.conv1_params,
        ) = prepare_conv_params(
            conv_weights_1,
            conv_bias_1,
            self.conv1_config.weights_dtype,
        )
        self.conv1_slice_config = None  # auto slicing
        self.conv_output_dtype = model_config.get_conv_output_dtype()

        self.compute2_config = model_config.get_conv_compute_config(module_path=f"{module_path}.conv2")
        self.conv2_config = model_config.get_conv_config(conv_path=f"{module_path}.conv2")
        (
            self.tt_conv2_weights,
            self.tt_conv2_bias,
            self.conv2_params,
        ) = prepare_conv_params(
            conv_weights_2,
            conv_bias_2,
            self.conv2_config.weights_dtype,
        )
        self.conv2_slice_config = None  # auto slicing

        if conv_shortcut:
            self.tt_conv3_weights, self.tt_conv3_bias = prepare_linear_params(
                device, conv_weights_3, conv_bias_3, model_config.conv_w_dtype
            )
        else:
            self.tt_conv3_weights = self.tt_conv3_bias = None

    def forward(self, input_tensor, input_shape):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        if self.groupnorm_memory_config_1 == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
            mem_cfg = ttnn.create_sharded_memory_config(
                shape=hidden_states.shape,
                core_grid=self.groupnorm_config_1["core_grid"],
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            reciprocals_tensor = None
        else:
            sharded_mem_config = ttnn.create_sharded_memory_config(
                shape=self.reciprocals_tensor_1.shape,
                core_grid=self.groupnorm_config_1["core_grid"],
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            reciprocals_tensor = ttnn.to_memory_config(self.reciprocals_tensor_1, sharded_mem_config)

        hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)

        # force_welford=True reproduces the bug, force_welford=False uses the workaround
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask_1,
            negative_mask=self.input_negative_mask_1,
            weight=self.gamma_t_1,
            bias=self.beta_t_1,
            epsilon=self.norm_eps,
            memory_config=hidden_states.memory_config(),
            use_welford=self.force_welford,
            reciprocals=reciprocals_tensor if self.force_welford else None,
            **self.groupnorm_config_1,
        )

        if self.conv1_slice_config != ttnn.Conv2dL1FullSliceConfig:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        if reciprocals_tensor is not None:
            ttnn.deallocate(reciprocals_tensor)

        hidden_states = ttnn.silu(hidden_states)

        [hidden_states, [H, W], [tt_conv1_weights, tt_conv1_bias]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_conv1_weights,
            in_channels=self.conv1_params["input_channels"],
            out_channels=self.conv1_params["output_channels"],
            device=self.device,
            bias_tensor=self.tt_conv1_bias,
            kernel_size=self.conv1_params["kernel_size"],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv1_config,
            compute_config=self.compute1_config,
            groups=self.groups,
            memory_config=None,
            slice_config=self.conv1_slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conv_output_dtype,
        )
        C = self.conv1_params["output_channels"]
        if not self.debug_mode:
            self.tt_conv1_weights = tt_conv1_weights
            self.tt_conv1_bias = tt_conv1_bias

        mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        if self.groupnorm_memory_config_2 == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
            mem_cfg = ttnn.create_sharded_memory_config(
                shape=hidden_states.shape,
                core_grid=self.groupnorm_config_2["core_grid"],
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            reciprocals_tensor = None
        else:
            sharded_mem_config = ttnn.create_sharded_memory_config(
                shape=self.reciprocals_tensor_2.shape,
                core_grid=self.groupnorm_config_2["core_grid"],
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            reciprocals_tensor = ttnn.to_memory_config(self.reciprocals_tensor_2, sharded_mem_config)

        hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)

        # force_welford=True reproduces the bug
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask_2,
            negative_mask=self.input_negative_mask_2,
            weight=self.gamma_t_2,
            bias=self.beta_t_2,
            epsilon=self.norm_eps,
            memory_config=hidden_states.memory_config(),
            use_welford=self.force_welford,
            reciprocals=reciprocals_tensor if self.force_welford else None,
            **self.groupnorm_config_2,
        )

        if self.conv2_slice_config != ttnn.Conv2dL1FullSliceConfig:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        if reciprocals_tensor is not None:
            ttnn.deallocate(reciprocals_tensor)

        hidden_states = ttnn.silu(hidden_states)

        [hidden_states, [H, W], [tt_conv2_weights, tt_conv2_bias]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_conv2_weights,
            in_channels=self.conv2_params["input_channels"],
            out_channels=self.conv2_params["output_channels"],
            device=self.device,
            bias_tensor=self.tt_conv2_bias,
            kernel_size=self.conv2_params["kernel_size"],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv2_config,
            compute_config=self.compute2_config,
            groups=self.groups,
            memory_config=None,
            slice_config=self.conv2_slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conv_output_dtype,
        )
        C = self.conv2_params["output_channels"]
        if not self.debug_mode:
            self.tt_conv2_weights = tt_conv2_weights
            self.tt_conv2_bias = tt_conv2_bias

        if self.tt_conv3_weights is not None:
            input_tensor_pre_conv = input_tensor
            input_tensor = ttnn.linear(
                input_tensor,
                self.tt_conv3_weights,
                bias=self.tt_conv3_bias,
            )
            ttnn.deallocate(input_tensor_pre_conv)

        if input_tensor.is_sharded():
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)
        if hidden_states.is_sharded():
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.add(input_tensor, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(input_tensor)
        return hidden_states, [C, H, W]


def run_resnetblock2d_test(
    device,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_vae_location,
    input_shape,
    block_id,
    resnet_id,
    conv_shortcut,
    block,
    force_welford,
):
    """
    Run ResnetBlock2D test with specified parameters.
    """

    image_resolution = (1024, 1024)

    vae = AutoencoderKL.from_pretrained(
        sdxl_base_vae_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
        subfolder=None if is_ci_v2_env else "vae",
    )
    vae.eval()
    state_dict = vae.state_dict()

    is_encoder = False
    if block == "up_blocks":
        torch_resnet = vae.decoder.up_blocks[block_id].resnets[resnet_id]
        block_path = f"{block}.{block_id}"
    elif block == "down_blocks":
        torch_resnet = vae.encoder.down_blocks[block_id].resnets[resnet_id]
        block_path = f"{block}.{block_id}"
        is_encoder = True
    else:
        torch_resnet = vae.decoder.mid_block.resnets[resnet_id]
        block_path = block

    vae_block = "encoder" if is_encoder else "decoder"

    model_config = load_vae_model_optimisations(image_resolution)
    tt_resnet = TtResnetBlock2DWithWelford(
        device,
        state_dict,
        f"{vae_block}.{block_path}.resnets.{resnet_id}",
        model_config,
        conv_shortcut,
        debug_mode=False,
        force_welford=force_welford,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_resnet(torch_input_tensor, None)

    B, C, H, W = input_shape
    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    torch_input_tensor = torch_input_tensor.reshape(B, 1, H * W, C)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output_tensor, output_shape = tt_resnet.forward(ttnn_input_tensor, [B, C, H, W])
    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del vae, tt_resnet
    gc.collect()

    return torch_output_tensor, output_tensor


# All 1024x1024 test cases from the original test file
# Format: (input_shape, block_id, resnet_id, conv_shortcut, block, expected_pcc)
TEST_CASES_1024x1024 = [
    # mid_block
    ((1, 512, 128, 128), 0, 0, False, "mid_block", 0.999),
    # up_blocks
    ((1, 512, 128, 128), 0, 0, False, "up_blocks", 0.999),
    ((1, 512, 256, 256), 1, 0, False, "up_blocks", 0.999),
    ((1, 512, 512, 512), 2, 0, True, "up_blocks", 0.999),
    ((1, 256, 512, 512), 2, 1, False, "up_blocks", 0.999),
    ((1, 256, 1024, 1024), 3, 0, True, "up_blocks", 0.999),
    ((1, 128, 1024, 1024), 3, 1, False, "up_blocks", 0.999),
    # down_blocks
    ((1, 128, 1024, 1024), 0, 0, False, "down_blocks", 0.998),
    ((1, 128, 512, 512), 1, 0, True, "down_blocks", 0.999),
    ((1, 256, 512, 512), 1, 1, False, "down_blocks", 0.999),
    ((1, 256, 256, 256), 2, 0, True, "down_blocks", 0.999),
    ((1, 512, 256, 256), 2, 1, False, "down_blocks", 0.999),
    ((1, 512, 128, 128), 3, 0, False, "down_blocks", 0.999),
]


@pytest.mark.skipif(not is_blackhole(), reason="This test only reproduces on Blackhole")
@pytest.mark.parametrize(
    "input_shape, block_id, resnet_id, conv_shortcut, block, pcc",
    TEST_CASES_1024x1024,
    ids=[
        "mid_block-0-0",
        "up_blocks-0-0",
        "up_blocks-1-0",
        "up_blocks-2-0-conv_shortcut",
        "up_blocks-2-1",
        "up_blocks-3-0-conv_shortcut",
        "up_blocks-3-1",
        "down_blocks-0-0",
        "down_blocks-1-0-conv_shortcut",
        "down_blocks-1-1",
        "down_blocks-2-0-conv_shortcut",
        "down_blocks-2-1",
        "down_blocks-3-0",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_resnetblock2d_welford_enabled_EXPECT_FAIL(
    device,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_vae_location,
    reset_seeds,
    input_shape,
    block_id,
    resnet_id,
    conv_shortcut,
    block,
    pcc,
):
    """
    BUG REPRODUCTION: These tests SHOULD FAIL on Blackhole.

    When Welford algorithm is enabled for DRAM GroupNorm on Blackhole,
    the numerical precision is severely degraded.

    Known failing cases with approximate PCC on Blackhole:
    - up_blocks-1-0: PCC ~0.93 (expected 0.999)
    - up_blocks-3-0-conv_shortcut: PCC ~0.9988 (borderline)
    - down_blocks-0-0: PCC ~0.98 (expected 0.998)
    - down_blocks-2-0-conv_shortcut: PCC ~0.83 (WORST - expected 0.999)

    This demonstrates the bug that needs to be fixed in ttnn.group_norm.
    """
    torch_output, tt_output = run_resnetblock2d_test(
        device,
        is_ci_env,
        is_ci_v2_env,
        sdxl_base_vae_location,
        input_shape,
        block_id,
        resnet_id,
        conv_shortcut,
        block,
        force_welford=True,
    )

    # This will FAIL on Blackhole for several test cases
    assert_with_pcc(torch_output, tt_output, pcc)


@pytest.mark.skipif(not is_blackhole(), reason="This test only reproduces on Blackhole")
@pytest.mark.parametrize(
    "input_shape, block_id, resnet_id, conv_shortcut, block, pcc",
    TEST_CASES_1024x1024,
    ids=[
        "mid_block-0-0",
        "up_blocks-0-0",
        "up_blocks-1-0",
        "up_blocks-2-0-conv_shortcut",
        "up_blocks-2-1",
        "up_blocks-3-0-conv_shortcut",
        "up_blocks-3-1",
        "down_blocks-0-0",
        "down_blocks-1-0-conv_shortcut",
        "down_blocks-1-1",
        "down_blocks-2-0-conv_shortcut",
        "down_blocks-2-1",
        "down_blocks-3-0",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_resnetblock2d_welford_disabled_EXPECT_PASS(
    device,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_vae_location,
    reset_seeds,
    input_shape,
    block_id,
    resnet_id,
    conv_shortcut,
    block,
    pcc,
):
    """
    WORKAROUND: These tests SHOULD PASS on Blackhole.

    When Welford algorithm is disabled for DRAM GroupNorm on Blackhole,
    the standard variance algorithm produces correct results.

    This demonstrates the workaround currently used in production code.
    """
    torch_output, tt_output = run_resnetblock2d_test(
        device,
        is_ci_env,
        is_ci_v2_env,
        sdxl_base_vae_location,
        input_shape,
        block_id,
        resnet_id,
        conv_shortcut,
        block,
        force_welford=False,
    )

    # This will PASS on Blackhole P150
    assert_with_pcc(torch_output, tt_output, pcc)
