# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, OpParameter, get_mesh_grid_size
import ttnn
from models.common.utility_functions import skip_for_blackhole, is_blackhole
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE

NUM_DEVICES = ttnn.distributed.get_num_devices()
MESH_X = NUM_DEVICES if NUM_DEVICES <= 8 else 8
MESH_Y = 1 if NUM_DEVICES <= 8 else int(NUM_DEVICES / MESH_X)


class SdxlConvTest(OpTestBase):
    def __init__(
        self,
        *args,
        compute_with_storage_grid_size=(8, 8),
        input_channels=None,
        out_channels=None,
        filter_height=None,
        filter_width=None,
        stride_h=None,
        stride_w=None,
        pad_h=None,
        pad_w=None,
        dilation=None,
        batch_size=None,
        input_height=None,
        input_width=None,
        groups=None,
        slice_config=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_with_storage_grid_size = compute_with_storage_grid_size
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation = dilation
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.groups = groups
        self.slice_config = slice_config

        self.reader_patterns_cache = {}

    # Remove weights shape
    def generate_torch_activations(self, shape):
        torch_input_tensor_nchw = torch.randn(shape, dtype=torch.bfloat16)
        torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
        return torch_input_tensor

    def generate_tt_input_from_torch(self, torch_tensor, dtype, layout, mem_config, ind):
        if ind == 0:
            return ttnn.from_torch(torch_tensor, dtype)
        elif ind == 1:
            # hack
            bias_tensor = (
                torch.randn(self.arguments[ind].shape, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            return ttnn.from_torch(bias_tensor, dtype)
        else:
            raise Exception("Expected only two inputs")

    def deallocate_activations(self):
        # Do nothing in conv case as activations are on device
        pass

    def run_device_operation(self):
        [tt_output_tensor_on_device, [_, _], [self.inputs[0], self.inputs[1]]] = ttnn.conv2d(
            input_tensor=self.activations,
            weight_tensor=self.inputs[0],
            in_channels=self.input_channels,
            out_channels=self.out_channels,
            device=self.mesh_device,
            bias_tensor=self.inputs[1],
            kernel_size=(self.filter_height, self.filter_width),
            stride=(self.stride_h, self.stride_w),
            padding=(self.pad_h, self.pad_w),
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            conv_config=self.program_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_output_dim=True,
            return_weights_and_bias=True,
            slice_config=self.slice_config,
        )
        return tt_output_tensor_on_device


# Test cases for convs that hang in SDXL UNet and VAE
conv_test_cases = [
    {
        "id": "unet_resnet_1280x1280",
        "input_shape": [1, 1280, 32, 32],
        "output_channels": 1280,
        "weights_dtype": ttnn.bfloat8_b,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "act_block_h_override": 0,
        "slice_type": None,
        "num_slices": 1,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
        "input_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "packer_l1_acc": True,
        "fp32_dest_acc_en": False,
    },
    {
        "id": "unet_resnet_1920x1280",
        "input_shape": [1, 1920, 32, 32],
        "output_channels": 1280,
        "weights_dtype": ttnn.bfloat8_b,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "act_block_h_override": 0,
        "slice_type": None,
        "num_slices": 1,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
        "input_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "packer_l1_acc": True,
        "fp32_dest_acc_en": False,
    },
    {
        "id": "unet_resnet_2560x1280",
        "input_shape": [1, 2560, 32, 32],
        "output_channels": 1280,
        "weights_dtype": ttnn.bfloat8_b,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "act_block_h_override": 0,
        "slice_type": None,
        "num_slices": 1,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
        "input_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "packer_l1_acc": True,
        "fp32_dest_acc_en": False,
    },
    {
        "id": "vae_resnet_256x256",
        "input_shape": [1, 256, 512, 512],
        "output_channels": 256,
        "weights_dtype": ttnn.bfloat8_b,
        "shard_layout": None,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "act_block_h_override": 512,
        "slice_type": ttnn.Conv2dDRAMSliceWidth,
        "num_slices": 4,
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "input_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "packer_l1_acc": False,
        "fp32_dest_acc_en": True,
    },
    {
        "id": "vae_resnet_512x256",
        "input_shape": [1, 512, 512, 512],
        "output_channels": 256,
        "weights_dtype": ttnn.bfloat8_b,
        "shard_layout": None,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "act_block_h_override": 512,
        "slice_type": ttnn.Conv2dDRAMSliceWidth,
        "num_slices": 8,
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "input_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "packer_l1_acc": False,
        "fp32_dest_acc_en": True,
    },
    {
        "id": "vae_resnet_512x512",
        "input_shape": [1, 512, 128, 128],
        "output_channels": 512,
        "weights_dtype": ttnn.bfloat8_b,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "act_block_h_override": 512,
        "slice_type": None,
        "num_slices": 1,
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "input_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "packer_l1_acc": False,
        "fp32_dest_acc_en": True,
    },
    {
        "id": "vae_upsample_256",
        "input_shape": [1, 512, 256, 256],
        "output_channels": 512,
        "weights_dtype": ttnn.bfloat16,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "act_block_h_override": 256,
        "slice_type": ttnn.Conv2dDRAMSliceWidth,
        "num_slices": 2,
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "input_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "packer_l1_acc": False,
        "fp32_dest_acc_en": True,
    },
    {
        "id": "vae_upsample_512",
        "input_shape": [1, 512, 512, 512],
        "output_channels": 512,
        "weights_dtype": ttnn.bfloat16,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "act_block_h_override": 256,
        "slice_type": ttnn.Conv2dDRAMSliceWidth,
        "num_slices": 8,
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "input_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "packer_l1_acc": False,
        "fp32_dest_acc_en": True,
    },
    {
        "id": "vae_upsample_1024",
        "input_shape": [1, 256, 1024, 1024],
        "output_channels": 256,
        "weights_dtype": ttnn.bfloat8_b,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "act_block_h_override": 512,
        "slice_type": ttnn.Conv2dDRAMSliceWidth,
        "num_slices": 16,
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "input_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "packer_l1_acc": False,
        "fp32_dest_acc_en": True,
    },
]


@skip_for_blackhole("Blackhole has not been tested, see #25544")
@pytest.mark.parametrize("test_config", conv_test_cases, ids=[c["id"] for c in conv_test_cases])
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
        pytest.param((MESH_X, MESH_Y), id="all"),  # run on all available devices
    ],
    indirect=["mesh_device"],
)
def test_sdxl_conv(mesh_device, didt_workload_iterations, determinism_check_interval, test_config):
    groups = 1
    dilation = 1
    pad_w = 1
    pad_h = 1
    stride_w = 1
    stride_h = 1
    filter_height = 3
    filter_width = 3

    batch_size, input_channels, input_height, input_width = test_config["input_shape"]
    output_channels = test_config["output_channels"]

    compute_grid = get_mesh_grid_size(mesh_device)
    logger.info(f"Running on {compute_grid} cores")

    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]
    conv_bias_shape = [output_channels]

    in0_shape = conv_input_shape
    in1_shape = conv_weight_shape

    activations_dtype = ttnn.bfloat16
    weights_dtype = test_config["weights_dtype"]

    # Weird situation
    # act and weight dtype need to be bfp8, which is set through conv config
    # however when creating tensors they need to be different
    in0_dtype = ttnn.bfloat16
    in1_dtype = weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        shard_layout=test_config["shard_layout"],
        deallocate_activation=test_config["deallocate_activation"],
        reallocate_halo_output=test_config["reallocate_halo_output"],
        enable_act_double_buffer=test_config["enable_act_double_buffer"],
        enable_weights_double_buffer=test_config["enable_weights_double_buffer"],
        reshard_if_not_optimal=True,
        act_block_w_div=1,
        act_block_h_override=test_config["act_block_h_override"],
    )

    slice_type = test_config["slice_type"]
    num_slices = test_config["num_slices"]
    slice_config = (
        ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        )
        if slice_type is not None and num_slices > 1
        else None
    )

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_kernel_config = ComputeConfigClass(
        math_fidelity=test_config["math_fidelity"],
        math_approx_mode=True,
        fp32_dest_acc_en=test_config["fp32_dest_acc_en"],
        packer_l1_acc=test_config["packer_l1_acc"],
    )

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    sdxlConvTest = SdxlConvTest(
        mesh_device,
        OpParameter(in0_shape, in0_dtype, ttnn.ROW_MAJOR_LAYOUT, test_config["input_mem_config"]),  # activations
        [
            OpParameter(in1_shape, in1_dtype, ttnn.TILE_LAYOUT, mem_config),  # inputs
            OpParameter(conv_bias_shape, in1_dtype, ttnn.TILE_LAYOUT, mem_config),
        ],
        out_mem_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        out_dtype=None,
        program_config=conv_config,  # program config
        compute_config=compute_kernel_config,  # compute config
        input_channels=input_channels,
        out_channels=output_channels,
        filter_height=filter_height,
        filter_width=filter_width,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation=dilation,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        groups=groups,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=determinism_check_interval > 0,
        determinism_check_interval=determinism_check_interval,
        compute_with_storage_grid_size=compute_grid,
        slice_config=slice_config,
    )

    sdxlConvTest.run_op_test()


@pytest.mark.parametrize("test_config", conv_test_cases, ids=[c["id"] for c in conv_test_cases])
@pytest.mark.parametrize("logical_chip_id", range(32), ids=[f"logical_chip_{i}_" for i in range(32)])
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_specific_chip_sdxl_conv(
    mesh_device, logical_chip_id, didt_workload_iterations, determinism_check_interval, test_config
):
    assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_sdxl_conv(
        mesh_device.get_device(logical_chip_id),
        didt_workload_iterations,
        determinism_check_interval,
        test_config,
    )


@skip_for_blackhole("Multi-board Blackhole has not been tested")
@pytest.mark.parametrize("test_config", conv_test_cases, ids=[c["id"] for c in conv_test_cases])
@pytest.mark.parametrize(
    "t3k_single_board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["t3k_single_board_mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_specific_board_sdxl_conv(
    t3k_single_board_mesh_device, didt_workload_iterations, determinism_check_interval, test_config
):
    test_sdxl_conv(t3k_single_board_mesh_device, didt_workload_iterations, determinism_check_interval, test_config)
