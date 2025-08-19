# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class TtYOLOv7Conv2D:
    def __init__(
        self,
        input_params,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        deallocate=False,
        deallocate_activation=False,
        height_sharding=True,
        activation="silu",
        groups=1,
        dtype=ttnn.bfloat8_b,
        num_cores_nhw=None,
        is_reshape=False,
        enable_split_reader=False,
        enable_act_double_buffer=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        math_approx_mode=False,
        input_channels_alignment=32,
        use_1d_systolic_array=True,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.dtype = dtype
        self.deallocate_activation = deallocate_activation
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.packer_l1_acc = packer_l1_acc
        self.math_approx_mode = math_approx_mode
        self.input_channels_alignment = input_channels_alignment
        self.use_1d_systolic_array = use_1d_systolic_array
        self.input_params = input_params
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.groups = groups
        self.deallocate = deallocate
        self.activation = activation
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.num_cores_nhw = num_cores_nhw
        self.is_reshape = is_reshape
        self.enable_split_reader = enable_split_reader
        self.enable_act_double_buffer = enable_act_double_buffer

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16 if self.use_1d_systolic_array else ttnn.bfloat8_b,
            activation=self.activation,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            enable_split_reader=self.enable_split_reader,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            deallocate_activation=self.deallocate_activation,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
            math_approx_mode=self.math_approx_mode,
        )
        if self.num_cores_nhw is not None:
            shard_grid = get_shard_grid_from_num_cores(self.num_cores_nhw, device)
            conv_config.core_grid = shard_grid
            conv_config.override_sharding_config = True

        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, [_out_height, _out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.input_params[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=(self.conv_params[0], self.conv_params[1]),
            stride=(self.conv_params[2], self.conv_params[3]),
            padding=(self.conv_params[4], self.conv_params[5]),
            dilation=(self.conv_params[6], self.conv_params[7]),
            batch_size=self.input_params[0],
            input_height=self.input_params[1],
            input_width=self.input_params[2],
            conv_config=conv_config,
            compute_config=compute_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=self.dtype,
        )
        if self.is_reshape:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
            output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.reshape(
                output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])
            )
            output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
        return output_tensor


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


def sharded_concat(
    input_tensors,
    num_cores=56,
    dim=3,
    shard_strategy=ttnn.ShardStrategy.HEIGHT,
    shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
):
    max_cores = 0
    max_id = 0
    for i, tensor in enumerate(input_tensors):
        cores = tensor.memory_config().shard_spec.grid.num_cores()
        if cores > max_cores:
            max_cores = cores
            max_id = i
    id = max_id

    shard_grid = input_tensors[id].memory_config().shard_spec.grid
    num_cores = shard_grid.num_cores()

    in_shard_width = input_tensors[id].shape[-1]
    shard_height = (input_tensors[id].shape[2] + num_cores - 1) // num_cores
    shard_height = ((shard_height + 31) // 32) * 32
    print("Shard height:", shard_height)

    input_sharded_memory_config = ttnn.create_sharded_memory_config_(
        (shard_height, in_shard_width),
        core_grid=shard_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    out_shard_width = 0
    for i in range(len(input_tensors)):
        out_shard_width += input_tensors[i].shape[-1]

        if input_tensors[i].shape[-1] != in_shard_width:
            print(f"Input tensor {i} does not have the correct memory config, converting...")
            temp_input_shared_memory_config = ttnn.create_sharded_memory_config_(
                (shard_height, input_tensors[i].shape[-1]),
                core_grid=shard_grid,
                strategy=shard_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            input_tensors[i] = ttnn.to_memory_config(input_tensors[i], temp_input_shared_memory_config)
        else:
            # If the input tensor already has the correct memory config, we can skip this step
            input_tensors[i] = ttnn.to_memory_config(input_tensors[i], input_sharded_memory_config)
        print(input_tensors[i].shape, input_tensors[i].memory_config().shard_spec.grid, input_tensors[i].layout)

    output_sharded_memory_config = ttnn.create_sharded_memory_config_(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    output = ttnn.concat(input_tensors, dim, memory_config=output_sharded_memory_config)
    return output
