# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn


class Yolov11Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        activation="",
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        reshard=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_detect=False,
        is_dfl=False,
        config_override=None,
        deallocate_activation=False,
    ):
        self.is_detect = is_detect
        self.activation = activation
        self.is_dfl = is_dfl
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
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
            packer_l1_acc=True,
            math_approx_mode=True,
        )
        self.activation_dtype = activation_dtype
        # Convert activation string to proper ttnn activation object
        activation_param = None
        if self.activation == "silu":
            activation_param = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        elif self.activation == "relu":
            activation_param = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        # Add more activation types as needed

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True if self.reshard else False,
            activation=activation_param,
        )
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if "bias" in conv_pth and conv_pth.bias is not None:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, x):
        if self.is_detect:
            input_height = int(math.sqrt(x.shape[2]))
            input_width = int(math.sqrt(x.shape[2]))
            batch_size = x.shape[0]
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

        # 🔍 DEBUGGING: Analyze conv2d parameters
        print(f"🔍 [CONV2D DEBUG] Conv2D operation details:")
        print(f"    Input shape: {x.shape}")
        print(f"    Input memory config: {x.memory_config()}")
        print(f"    Input memory layout: {x.memory_config().memory_layout}")
        print(f"    Input buffer type: {x.memory_config().buffer_type}")
        print(f"    Input storage type: {x.storage_type()}")
        print(f"    Input is sharded: {x.is_sharded()}")
        print(f"    In channels: {self.in_channels}, Out channels: {self.out_channels}")
        print(f"    Kernel size: {kernel_size}, Stride: {stride}, Padding: {padding}")
        print(f"    Groups: {self.groups}")
        print(f"    Activation dtype: {self.activation_dtype}")
        print(f"    Conv config shard layout: {self.conv_config.shard_layout}")
        print(f"    Weight shape: {self.weight.shape}")
        print(f"    Weight memory config: {self.weight.memory_config()}")
        if self.bias is not None:
            print(f"    Bias shape: {self.bias.shape}")
            print(f"    Bias memory config: {self.bias.memory_config()}")
        else:
            print(f"    Bias: None")
        
        # Calculate expected output memory
        expected_output_elements = batch_size * self.out_channels * input_height * input_width
        expected_output_mb = (expected_output_elements * 2) / (1024 * 1024)  # bfloat16 = 2 bytes
        print(f"    Expected output elements: {expected_output_elements}")
        print(f"    Expected output memory: {expected_output_mb:.2f} MB")
        
        if expected_output_mb > 1.43:
            print(f"    ⚠️  WARNING: Expected output ({expected_output_mb:.2f}MB) exceeds L1 limit!")
        else:
            print(f"    ✅ Expected output ({expected_output_mb:.2f}MB) fits in L1")
        
        # 🔍 DEBUGGING: Check if we need to force DRAM via slice_config
        # According to PR #27147, the conv2d should auto-select DRAM based on input memory config
        # But if there are explicit slice configs, they might override this
        print(f"    🔧 Conv2D should auto-select DRAM based on input memory config")
        print(f"    🔧 Input is in DRAM: {x.memory_config().buffer_type == ttnn.BufferType.DRAM}")
        print(f"    🔧 Input is HEIGHT_SHARDED: {x.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED}")

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
            dtype=self.activation_dtype
        )
        
        # 🔍 DEBUGGING: Analyze conv2d output
        print(f"🔍 [CONV2D DEBUG] Conv2D operation completed:")
        print(f"    Output shape: {x.shape}")
        print(f"    Output memory config: {x.memory_config()}")
        print(f"    Output storage type: {x.storage_type()}")
        print(f"    Actual output height: {output_height}, width: {output_width}")
        
        hw = output_height * output_width
        print(f"    Expected hw: {hw}, Actual x.shape[2]: {x.shape[2]}")
        
        if x.shape[2] != hw:
            print(f"    🔧 Reshaping needed: {x.shape[2]} -> {hw}")
            print(f"    🔧 Converting to DRAM interleaved for reshaping, then back to sharded")
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            print(f"    🔧 After sharded_to_interleaved: {x.shape}, memory_config: {x.memory_config()}")
            x = x[:, :, :hw, :]
            print(f"    🔧 After slicing: {x.shape}")
            # Convert back to HEIGHT_SHARDED DRAM for next conv2d operations
            try:
                dram_sharded_config = ttnn.create_sharded_memory_config(
                    x.shape,
                    ttnn.CoreGrid(y=8, x=8),
                    ttnn.ShardStrategy.HEIGHT,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=False,
                )
                dram_sharded_config.buffer_type = ttnn.BufferType.DRAM
                x = ttnn.to_memory_config(x, dram_sharded_config)
                print(f"    🔧 Converted back to DRAM HEIGHT_SHARDED: {x.memory_config()}")
            except Exception as e:
                print(f"    ⚠️  Failed to convert back to DRAM sharded, keeping interleaved: {e}")
        else:
            print(f"    ✅ No reshaping needed")
        
        print(f"🔍 [CONV2D DEBUG] Final output memory config: {x.memory_config()}")
        return x


def sharded_concat(input_tensors, num_cores=64, dim=3, to_interleaved=True):
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    in_shard_width = input_tensors[0].shape[-1]
    shard_height = (input_tensors[0].shape[2] + num_cores - 1) // num_cores
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


# for input tensor's whose shape is different from each other
def sharded_concat_2(
    input_tensor_1, input_tensor_2, num_cores=64, shard_grid_coord_min=0, shard_grid_coord_max=7, dim=-1
):
    if input_tensor_1.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        input_tensor_1 = ttnn.to_layout(input_tensor_1, ttnn.ROW_MAJOR_LAYOUT)

    if input_tensor_2.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        input_tensor_2 = ttnn.to_layout(input_tensor_2, ttnn.ROW_MAJOR_LAYOUT)

    shard_height = (input_tensor_1.shape[2] + num_cores - 1) // num_cores

    input_sharded_memory_config_1 = ttnn.create_sharded_memory_config(
        (shard_height, input_tensor_1.shape[-1]),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(shard_grid_coord_min, shard_grid_coord_min),
                    ttnn.CoreCoord(shard_grid_coord_max, shard_grid_coord_max),
                )
            }
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    input_sharded_memory_config_2 = ttnn.create_sharded_memory_config(
        (shard_height, input_tensor_2.shape[-1]),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(shard_grid_coord_min, shard_grid_coord_min),
                    ttnn.CoreCoord(shard_grid_coord_max, shard_grid_coord_max),
                )
            }
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_1 = ttnn.to_memory_config(input_tensor_1, input_sharded_memory_config_1)
    input_tensor_2 = ttnn.to_memory_config(input_tensor_2, input_sharded_memory_config_2)
    out_sharded_memory_config_ = ttnn.create_sharded_memory_config(
        (shard_height, input_tensor_1.shape[-1] + input_tensor_2.shape[-1]),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(shard_grid_coord_min, shard_grid_coord_min),
                    ttnn.CoreCoord(shard_grid_coord_max, shard_grid_coord_max),
                )
            }
        ),
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
        activation="",
        deallocate_activation=False,
    ):
        self.enable_act = enable_act
        if self.enable_act:
            activation = "silu"
        self.conv = Yolov11Conv2D(
            parameter.conv,
            conv_pt.conv,
            device=device,
            is_detect=is_detect,
            reshard=reshard,
            activation=activation,
            deallocate_activation=deallocate_activation,
        )

    def __call__(self, device, x):
        x = self.conv(x)
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
