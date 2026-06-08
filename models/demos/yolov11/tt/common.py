# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os

import ttnn
from models.common.utility_functions import roundup32
from models.experimental.yolo_common.yolo_utils import get_core_grid_from_num_cores


class Yolov11Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        activation=None,
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        reshard=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_detect=False,
        is_dfl=False,
        config_override=None,
        deallocate_activation=False,
        split_weights=False,
    ):
        self.is_detect = is_detect
        self.activation = activation
        self.is_dfl = is_dfl
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        if split_weights:
            self.out_channels = self.out_channels // 2
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.reshard = reshard
        self.deallocate_activation = deallocate_activation
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        self.activation_dtype = activation_dtype
        _dblbuf = os.environ.get("YOLO_NO_DBLBUF") != "1"  # disable to relieve L1 pressure (large batch)
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=_dblbuf,
            reshard_if_not_optimal=True
            if (self.reshard or os.environ.get("YOLO_RESHARD_ALL") == "1")
            else False,
            activation=self.activation,
            enable_weights_double_buffer=_dblbuf,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]
        if "bias" in conv_pth:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, x, output_rm_needed=False, to_interleaved=False, batch_size=None):
        if self.is_detect:
            # x is collapsed to [1, 1, N*H*W, C]; recover per-image side from H*W = shape[2]//N
            bs = batch_size if batch_size is not None else x.shape[0]
            input_height = int(math.sqrt(x.shape[2] // bs))
            input_width = int(math.sqrt(x.shape[2] // bs))
            batch_size = bs
        elif self.is_dfl:
            input_height = x.shape[1]
            input_width = x.shape[2]
            batch_size = x.shape[0]
        else:
            batch_size = self.conv.batch_size
            input_height = self.conv.input_height
            input_width = self.conv.input_width

        kernel_size = [self.kernel_size[0], self.kernel_size[1]]
        stride = [self.stride[0], self.stride[1]]
        padding = [self.padding[0], self.padding[1]]
        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_config=self.conv_config,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )
        # valid rows = batch_size * H' * W'; without the batch factor the slices
        # below would drop every image past the first when batch_size > 1.
        hw = batch_size * output_height * output_width
        if to_interleaved:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            x = x[:, :, :hw, :]
        else:
            if x.shape[2] != hw and output_rm_needed:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                x = x[:, :, :hw, :]
        return x


def reshard_if_possible(x, core_grid=None):  # reshards if shard_spec is not multiples of 32
    if x.is_sharded() and (
        x.memory_config().shard_spec.shape[0] % 32 != 0 or x.memory_config().shard_spec.shape[1] % 32 != 0
    ):
        aligned_h, aligned_w = roundup32(x.memory_config().shard_spec.shape[0]), roundup32(
            x.memory_config().shard_spec.shape[1]
        )
        resharded_memory_config = ttnn.create_sharded_memory_config(
            shape=(aligned_h, aligned_w),
            core_grid=x.memory_config().shard_spec.grid if core_grid is None else core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=x.memory_config().shard_spec.orientation,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.to_memory_config(x, resharded_memory_config)
    return x


def _concat_shard_grid(nhw, default_cores=64):
    """Pick a (num_cores, shard_grid) for height-sharded concat that keeps
    #shards == #cores (no fit error) and scales with batch.

    Concat is a numerically-lossless layout op, so changing the core grid does
    not affect PCC. On WH the defaults reproduce the original 8x8/64-core path;
    on Blackhole P150a set YOLO_MAX_CORES (e.g. 100) + YOLO_GRID_ROWS/COLS so the
    larger batch's shard count still fits within the 11x10=110 core grid.
    """
    max_cores = int(os.environ.get("YOLO_MAX_CORES", str(default_cores)))
    grid_rows = int(os.environ.get("YOLO_GRID_ROWS", "8"))
    grid_cols = int(os.environ.get("YOLO_GRID_COLS", "8"))
    # largest divisor of nhw that is <= max_cores -> exact, padding-free sharding
    num_cores = 1
    for c in range(min(max_cores, grid_rows * grid_cols), 0, -1):
        if nhw % c == 0:
            num_cores = c
            break
    shard_grid = get_core_grid_from_num_cores(num_cores, grid_rows=grid_rows, grid_cols=grid_cols)
    return num_cores, shard_grid


def sharded_concat(input_tensors, num_cores=64, dim=3, to_interleaved=True):
    if os.environ.get("YOLO_DEBUG_CONCAT"):
        import sys as _sys

        print(
            f"[concat] shapes={[tuple(t.shape) for t in input_tensors]} default_cores={num_cores}",
            file=_sys.stderr,
            flush=True,
        )
    num_cores, shard_grid = _concat_shard_grid(input_tensors[0].shape[2], default_cores=num_cores)
    in_shard_width = input_tensors[0].shape[-1]
    shard_height = input_tensors[0].shape[2] // num_cores
    input_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, in_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out_shard_width = 0
    for i in range(len(input_tensors)):
        out_shard_width += input_tensors[i].shape[-1]
        input_tensors[i] = ttnn.to_memory_config(input_tensors[i], input_sharded_memory_config)
    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    output = ttnn.concat(input_tensors, dim, memory_config=output_sharded_memory_config)
    if to_interleaved:
        output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG)

    return output


# for input tensor's whose shard_w is different from each other
def sharded_concat_2(
    input_tensor_1, input_tensor_2, num_cores=64, shard_grid_coord_min=0, shard_grid_coord_max=7, dim=-1
):
    if input_tensor_1.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        input_tensor_1 = ttnn.to_layout(input_tensor_1, ttnn.ROW_MAJOR_LAYOUT)

    if input_tensor_2.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        input_tensor_2 = ttnn.to_layout(input_tensor_2, ttnn.ROW_MAJOR_LAYOUT)

    # default 8x8 grid maps to coords 0..7; reproduce that as the default core count
    default_cores = (shard_grid_coord_max - shard_grid_coord_min + 1) ** 2
    num_cores, shard_grid = _concat_shard_grid(input_tensor_1.shape[2], default_cores=default_cores)
    shard_height = input_tensor_1.shape[2] // num_cores

    input_sharded_memory_config_1 = ttnn.create_sharded_memory_config(
        (shard_height, input_tensor_1.shape[-1]),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    input_sharded_memory_config_2 = ttnn.create_sharded_memory_config(
        (shard_height, input_tensor_2.shape[-1]),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_1 = ttnn.to_memory_config(input_tensor_1, input_sharded_memory_config_1)
    input_tensor_2 = ttnn.to_memory_config(input_tensor_2, input_sharded_memory_config_2)
    out_sharded_memory_config_ = ttnn.create_sharded_memory_config(
        (shard_height, input_tensor_1.shape[-1] + input_tensor_2.shape[-1]),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    output = ttnn.concat((input_tensor_1, input_tensor_2), dim, memory_config=out_sharded_memory_config_)
    return output


class TtnnConv:
    def __init__(
        self,
        device,
        parameter,
        conv_pt,
        enable_act=True,
        is_detect=False,
        reshard=False,
        activation=None,
        deallocate_activation=False,
        split_weights=False,
    ):
        self.enable_act = enable_act
        if self.enable_act:
            activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        self.conv = Yolov11Conv2D(
            parameter.conv,
            conv_pt.conv,
            device=device,
            is_detect=is_detect,
            reshard=reshard,
            activation=activation,
            deallocate_activation=deallocate_activation,
            split_weights=split_weights,
        )

    def __call__(self, device, x, output_rm_needed=False, to_interleaved=False, batch_size=None):
        x = self.conv(x, output_rm_needed, to_interleaved, batch_size=batch_size)
        return x


def deallocate_tensors(*tensors):
    for t in tensors:
        ttnn.deallocate(t)


def get_mesh_mappers(device):
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer


def get_mesh_mappers(device):
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer
