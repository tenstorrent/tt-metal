import ttnn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class NormConfig:
    num_groups: int = 32
    eps: float = 1e-5
    num_out_blocks: int = 1
    core_grid: ttnn.CoreGrid = None

    def __post_init__(self):
        if self.core_grid is None:
            self.core_grid = ttnn.CoreGrid(y=8, x=8)


@dataclass
class ConvConfig:
    split_conv: bool = False
    split_in: int = 1
    split_out: int = 1
    # conv_act_dtype, also output_dtype in tests
    conv_output_dtype = ttnn.bfloat16
    # weights_dtype
    conv_w_dtype = ttnn.bfloat16

    groups: int = 1
    kernel_size: Tuple[int, int] = (3, 3)
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (1, 1)
    dilation: Tuple[int, int] = (1, 1)
    act_block_h_override: int = 0
    act_block_w_div: int = 1
    deallocate_activation: bool = True
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    fp32_accum: bool = False
    packer_l1_acc: bool = False
    enable_split_reader: bool = False
    split_input_channels_factor: int = 1
    split_output_channels_factor: int = 1
    act_db: bool = False
    w_db: bool = False

    conv2d_config: ttnn.Conv2dConfig = None
    compute_config: ttnn.WormholeComputeKernelConfig = None

    def __post_init__(self):
        if self.compute_config is None:
            self.compute_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=self.math_fidelity,
                fp32_dest_acc_en=self.fp32_accum,
                math_approx_mode=False,
                packer_l1_acc=self.packer_l1_acc,
            )
        if self.conv2d_config is None:
            self.conv2d_config = ttnn.Conv2dConfig(
                weights_dtype=self.conv_w_dtype,
                deallocate_activation=self.deallocate_activation,
                act_block_w_div=self.act_block_w_div,
                act_block_h_override=self.act_block_h_override,
            )


@dataclass
class ResnetBlockConfig:
    norm1: NormConfig = None
    conv1: ConvConfig = None
    norm2: NormConfig = None
    conv2: ConvConfig = None
    conv_shortcut: ConvConfig = None  # For projection shortcuts


def make_norm_config(num_out_blocks: int, core_grid: Tuple[int, int]) -> NormConfig:
    return NormConfig(num_out_blocks=num_out_blocks, core_grid=ttnn.CoreGrid(y=core_grid[0], x=core_grid[1]))


def make_conv_config(
    kernel: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int], split_in: int
) -> ConvConfig:
    return ConvConfig(kernel_size=kernel, stride=stride, padding=padding, split_in=split_in)


# default is 1 out_block, 8x8 grid
def make_block_config(
    norm1_out_blocks: int = 1,
    norm1_grid: Optional[Tuple[int, int]] = (8, 8),
    conv1_kernel: Optional[Tuple[int, int]] = (3, 3),
    conv1_stride: Optional[Tuple[int, int]] = (1, 1),
    conv1_padding: Optional[Tuple[int, int]] = (1, 1),
    conv1_split_in: int = 1,
    norm2_out_blocks: int = 1,
    norm2_grid: Optional[Tuple[int, int]] = (8, 8),
    conv2_kernel: Optional[Tuple[int, int]] = (3, 3),
    conv2_stride: Optional[Tuple[int, int]] = (1, 1),
    conv2_padding: Optional[Tuple[int, int]] = (1, 1),
    conv2_split_in: int = 1,
    conv_shortcut_kernel: Optional[Tuple[int, int]] = (1, 1),
    conv_shortcut_stride: Optional[Tuple[int, int]] = (1, 1),
    conv_shortcut_padding: Optional[Tuple[int, int]] = (0, 0),
    conv_shortcut_split_in: int = 1,
) -> ResnetBlockConfig:
    return ResnetBlockConfig(
        norm1=make_norm_config(norm1_out_blocks, norm1_grid),
        conv1=make_conv_config(conv1_kernel, conv1_stride, conv1_padding, conv1_split_in),
        norm2=make_norm_config(norm2_out_blocks, norm2_grid),
        conv2=make_conv_config(conv2_kernel, conv2_stride, conv2_padding, conv2_split_in),
        conv_shortcut=make_conv_config(
            conv_shortcut_kernel, conv_shortcut_stride, conv_shortcut_padding, conv_shortcut_split_in
        ),
    )


# SDXL Refiner ResNet block configurations
resnet_configs = {
    # Down blocks
    "down_blocks.0.resnets.0": make_block_config(
        norm1_out_blocks=3, norm1_grid=(4, 4), norm2_out_blocks=3, norm2_grid=(4, 4)
    ),
    "down_blocks.0.resnets.1": make_block_config(
        norm1_out_blocks=3, norm1_grid=(4, 4), norm2_out_blocks=3, norm2_grid=(4, 4)
    ),
    "down_blocks.1.resnets.0": make_block_config(norm1_grid=(4, 4)),
    "down_blocks.1.resnets.1": make_block_config(),
    "down_blocks.2.resnets.0": make_block_config(),
    "down_blocks.2.resnets.1": make_block_config(),
    "down_blocks.3.resnets.0": make_block_config(),
    "down_blocks.3.resnets.1": make_block_config(),
    # Up blocks
    "up_blocks.0.resnets.0": make_block_config(),
    "up_blocks.0.resnets.1": make_block_config(),
    "up_blocks.0.resnets.2": make_block_config(),
    "up_blocks.1.resnets.0": make_block_config(),
    "up_blocks.1.resnets.1": make_block_config(),
    "up_blocks.1.resnets.2": make_block_config(),
    "up_blocks.2.resnets.0": make_block_config(),
    "up_blocks.2.resnets.1": make_block_config(),
    "up_blocks.2.resnets.2": make_block_config(norm1_grid=(4, 4)),
    "up_blocks.3.resnets.0": make_block_config(
        norm1_out_blocks=3, norm1_grid=(4, 4), conv1_split_in=2, norm2_out_blocks=3, norm2_grid=(4, 4)
    ),
    "up_blocks.3.resnets.1": make_block_config(norm1_out_blocks=2, norm2_out_blocks=3, norm2_grid=(4, 4)),
    "up_blocks.3.resnets.2": make_block_config(norm1_out_blocks=2, norm2_out_blocks=3, norm2_grid=(4, 4)),
}


def get_resnet_config(module_path: str) -> ResnetBlockConfig:
    config = resnet_configs.get(module_path)
    if config is None:
        return make_block_config()
    return config
