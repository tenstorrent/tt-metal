# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from ttnn.model_preprocessing import preprocess_model

import ttnn
import tt_lib
import tt_lib.fallback_ops


class TtNeck:
    def __init__(
        self,
        device,
        parameters,
    ) -> None:
        self.device = device
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        # print("\n\n\nattributes of parameters.c3: ", parameters.c3.__dict__)
        self.c4 = parameters.c4
        self.c5 = parameters.c5
        self.c6 = parameters.c6
        self.c7 = parameters.c7
        self.c7_2 = parameters.c7_2
        self.c7_3 = parameters.c7_3
        self.c7_4 = parameters.c7_4
        self.c7_5 = parameters.c7_5
        self.c8 = parameters.c8
        self.c8_2 = parameters.c8_2
        self.c9 = parameters.c9
        self.c9_2 = parameters.c9_2
        self.c9_3 = parameters.c9_3
        self.c9_4 = parameters.c9_4
        self.c9_5 = parameters.c9_5
        self.c10 = parameters.c10
        self.c10_2 = parameters.c10_2
        #        self.p1 = parameters.p1
        #        self.p2 = parameters.p2
        #        self.p3 = parameters.p3

        #########conv3###############
        #        self.c3 = ttnn.Conv2d(
        #            in_channels=1024,
        #            out_channels=512,
        #            kernel_size=(1, 1),
        #            stride=(1, 1),
        #            padding=(0, 0),
        #            dtype=ttnn.bfloat8_b,
        #            device=device,
        #            use_1d_systolic_array=True,
        #            batch_size=1,
        #            input_height=10,
        #            input_width=10,
        #            reader_patterns_cache={},
        #            weight=parameters.c3.weight,
        #            # bias=parameters.c3.bias,
        #            math_fidelity=ttnn.MathFidelity.LoFi,
        #            weights_dtype=ttnn.bfloat8_b,
        #            use_shallow_conv_variant=False,
        #            deallocate_activation=True,
        #            # padded_input_channels=32,
        #            activation="relu",
        #            conv_blocking_and_parallelization_config_override=None,
        #            # compute_kernel_config=compute_kernel_config,
        #        )

        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}

        max_pool_parallel_config_override["grid_size"] = self.c3.conv.grid_size
        max_pool_parallel_config_override["num_cores_nhw"] = self.c3.conv.sliding_window_op_params.num_cores_nhw
        print(max_pool_parallel_config_override)
        print(max_pool_parallel_config_override["num_cores_nhw"])

        #        self.p1 = tt_lib.fallback_ops.MaxPool2d(
        #            kernel_size=(5, 5),
        #            stride=(1, 1),
        #            padding=(2, 2),
        #            dilation=(1, 1),
        #            channels_last=True
        #        )
        #        self.p2 = tt_lib.fallback_ops.MaxPool2d(
        #            kernel_size=(9, 9),
        #            stride=(1, 1),
        #            padding=(4, 4),
        #            dilation=(1, 1),
        #            channels_last=True
        #        )
        #        self.p3 = tt_lib.fallback_ops.MaxPool2d(
        #            kernel_size=(13, 13),
        #            stride=(1, 1),
        #            padding=(6, 6),
        #            dilation=(1, 1),
        #            channels_last=True
        #        )

        self.p1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.p2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        self.p3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)

    #        self.p1 = ttnn.MaxPool2d(
    #            kernel_size=(5, 5),
    #            stride=(1, 1),
    #            padding=(2, 2),
    #            dilation=(1, 1),
    #            dtype=ttnn.bfloat16,
    #            device=self.device,
    #            batch_size=1,
    #            input_height=10,
    #            input_width=10,
    #            reader_patterns_cache=self.max_pool_reader_patterns_cache,
    #            deallocate_activation=True,
    #            # parallel_config_override=max_pool_parallel_config_override,
    #            channels=512,
    #        )
    #        self.p2 = ttnn.MaxPool2d(
    #            kernel_size=(9, 9),
    #            stride=(1, 1),
    #            padding=(4, 4),
    #            dilation=(1, 1),
    #            dtype=ttnn.bfloat16,
    #            device=self.device,
    #            batch_size=1,
    #            input_height=10,
    #            input_width=10,
    #            reader_patterns_cache=self.max_pool_reader_patterns_cache,
    #            deallocate_activation=True,
    #            # parallel_config_override=max_pool_parallel_config_override,
    #            channels=512,
    #        )
    #        self.p3 = ttnn.MaxPool2d(
    #            kernel_size=(13, 13),
    #            stride=(1, 1),
    #            padding=(6, 6),
    #            dilation=(1, 1),
    #            dtype=ttnn.bfloat16,
    #            device=self.device,
    #            batch_size=1,
    #            input_height=10,
    #            input_width=10,
    #            reader_patterns_cache=self.max_pool_reader_patterns_cache,
    #            deallocate_activation=True,
    #            # parallel_config_override=max_pool_parallel_config_override,
    #            channels=512,
    #        )
    #
    def __call__(self, device, input_tensors):
        input_tensor0 = input_tensors[0].to(device, self.c1.conv.input_sharded_memory_config)

        #######

        #        # 3 CBN blocks
        #        x1 = self.c1(input_tensor)
        #        x1_b = self.b1(x1)
        #        x1_m = self.relu(x1_b)
        #
        #        x2 = self.c2(x1_m)
        #        x2_b = self.b2(x2)
        #        x2_m = self.relu(x2_b)
        #
        #        x3 = self.c3(x2_m)
        #        x3_b = self.b3(x3)
        #        x3_m = self.relu(x3_b)
        #
        #        # maxpools
        #        x4 = self.p1(x3_m)
        #        x5 = self.p2(x3_m)
        #        x6 = self.p3(x3_m)
        #
        #        # concat the outputs of maxpool and x3_m
        #        conc1 = torch.cat([x4, x5, x6, x3_m], dim=1)
        #
        #        # 4 back2back CBRs
        #        # CBR4-1
        #        x7 = self.c4(conc1)
        #        x7_b = self.b4(x7)
        #        x7_m = self.relu(x7_b)
        #
        #        # CBR4-2
        #        x8 = self.c5(x7_m)
        #        x8_b = self.b5(x8)
        #        x8_m = self.relu(x8_b)
        #
        #        # CBR4-3
        #        x9 = self.c6(x8_m)
        #        x9_b = self.b6(x9)
        #        x9_m = self.relu(x9_b)
        #
        #        # CBR4-4
        #        x10 = self.c7(x9_m)
        #        x10_b = self.b7(x10)
        #        x10_m = self.relu(x10_b)
        #
        #        # upsample
        #        u1 = self.u(x10_m)
        #
        #        # Next CBR block to be concatinated with output of u1
        #        # gets the output of downsample4 module which is dimensions: [1, 512, 20, 20] - make a random tensor with that shape for the purpose of running the neck unit test stand-alone
        #        outDownSample4 = torch.rand([1, 512, 20, 20])
        #        # CBR block for conc2
        #        x11 = self.c7(outDownSample4)
        #        x11_b = self.b7(x11)
        #        x11_m = self.relu(x11_b)
        #
        #        # concat CBR output with output from u1
        #        conc2 = torch.cat([u1, x11_m], dim=1)
        #
        #        # 6 back2back CBRs
        #        # CBR6_1
        #        x12 = self.c7(conc2)
        #        x12_b = self.b7(x12)
        #        x12_m = self.relu(x12_b)
        #
        #        # CBR6_2
        #        x13 = self.c8(x12_m)
        #        x13_b = self.b8(x13)
        #        x13_m = self.relu(x13_b)
        #
        #        # CBR6_3
        #        x14 = self.c7(x13_m)
        #        x14_b = self.b7(x14)
        #        x14_m = self.relu(x14_b)
        #
        #        # CBR6_4
        #        x15 = self.c8(x14_m)
        #        x15_b = self.b8(x15)
        #        x15_m = self.relu(x15_b)
        #
        #        # CBR6_5
        #        x16 = self.c7(x15_m)
        #        x16_b = self.b7(x16)
        #        x16_m = self.relu(x16_b)
        #
        #        # CBR6_6
        #        x17 = self.c9(x16_m)
        #        x17_b = self.b9(x17)
        #        x17_m = self.relu(x17_b)
        #
        #        # upsample
        #        u2 = self.u(x17_m)
        #
        #        # CBR block for conc3
        #        outDownSample3 = torch.rand([1, 256, 40, 40])
        #        x18 = self.c9(outDownSample3)
        #        x18_b = self.b9(x18)
        #        x18_m = self.relu(x18_b)
        #
        #        # concat CBR output with output from u2
        #        conc3 = torch.cat([u2, x18_m], dim=1)
        #
        #        # 5 CBR blocks
        #        # CBR5_1
        #        x19 = self.c9(conc3)
        #        x19_b = self.b9(x19)
        #        x19_m = self.relu(x19_b)
        #
        #        # CBR5_2
        #        x20 = self.c10(x19_m)
        #        x20_b = self.b10(x20)
        #        x20_m = self.relu(x20_b)
        #
        #        # CBR5_3
        #        x21 = self.c9(x20_m)
        #        x21_b = self.b9(x21)
        #        x21_m = self.relu(x21_b)
        #
        #        # CBR5_4
        #        x22 = self.c10(x21_m)
        #        x22_b = self.b10(x22)
        #        x22_m = self.relu(x22_b)
        #
        #        # CBR5_5
        #        x23 = self.c9(x22_m)
        #        x23_b = self.b9(x23)
        #        x23_m = self.relu(x23_b)
        #
        #        return x23_m, x9_m, x16_m
        #
        #        #######
        output_tensor = self.c1(input_tensor0)
        output_tensor = self.c2(output_tensor)
        output_tensor = self.c3(output_tensor)
        output_tensorc3 = output_tensor

        output_tensorc3 = tt_lib.tensor.sharded_to_interleaved(output_tensorc3, ttnn.L1_MEMORY_CONFIG)
        # output_tensorc3 = ttnn.to_layout(output_tensorc3, layout=ttnn.TILE_LAYOUT)
        custom_sharded_memory_config = ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        )
        # output_tensorc3 = tt_lib.tensor.interleaved_to_sharded(output_tensorc3, self.p1.max_pool.input_sharded_memory_config)
        # ouptut_tensorc3=ttnn.to_memory_config(output_tensorc3, self.p1.max_pool.input_sharded_memory_config)
        # input_tensor.to(device, mem_config = custom_sharded_memory_config)
        # output_tensorc3 = output_tensorc3.to(device, self.p1.max_pool.input_sharded_memory_config)
        # input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

        # reproduces maxpool padding error
        output_tensorc3 = ttnn.to_layout(output_tensorc3, ttnn.ROW_MAJOR_LAYOUT)
        # output_tensorc3 = tt_lib.tensor.interleaved_to_sharded(
        #    output_tensorc3, self.p1.max_pool.input_sharded_memory_config
        # )
        print("C3 sharding: ", self.c3.conv.input_sharded_memory_config)
        # print("P1 sharding: ", self.p1.max_pool.output_sharded_memory_config)
        # input_tensor.memory_config().memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        print("Input sharding: ", output_tensorc3.memory_config().memory_layout)
        # return [output_tensorc3, output_tensorc3, output_tensorc3]

        output_tensorc3 = ttnn.from_device(output_tensorc3)
        output_tensorc3 = ttnn.to_torch(output_tensorc3)
        output_tensorc3 = torch.reshape(output_tensorc3, (1, 10, 10, 512))
        output_tensorc3 = torch.permute(output_tensorc3, (0, 3, 1, 2))
        # print("p1 inp: ",output_tensorc3.shape)

        # output_tensorc3 = ttnn.reshape(output_tensorc3, (1, 10, 10, 512))
        # output_tensorc3 = ttnn.to_torch(output_tensorc3)
        # output_tensorc3 = torch.permute(output_tensorc3, (0, 3, 1, 2))
        # from models.utility_functions import torch_to_tt_tensor_rm
        # output_tensorc3 = torch_to_tt_tensor_rm(output_tensorc3, device, put_on_device=True)
        output_tensor = self.p1(output_tensorc3)
        output_tensorp1 = output_tensor
        output_tensor = self.p2(output_tensorc3)
        output_tensorp2 = output_tensor
        output_tensor = self.p3(output_tensorc3)
        output_tensorp3 = output_tensor
        print("p3 shape: ", output_tensorp1.shape)
        # output_tensorp1 = ttnn.to_layout(output_tensorp1, layout=ttnn.TILE_LAYOUT)
        # output_tensorp1 = ttnn.permute(output_tensorp1, (0, 2, 3, 1))
        # output_tensorp1 = ttnn.reshape(output_tensorp1, (1, 1, 100, 500))
        # output_tensorp2 = ttnn.to_layout(output_tensorp2, layout=ttnn.TILE_LAYOUT)
        # output_tensorp2 = ttnn.permute(output_tensorp2, (0, 2, 3, 1))
        # output_tensorp2 = ttnn.reshape(output_tensorp2, (1, 1, 100, 500))
        # output_tensorp3 = ttnn.to_layout(output_tensorp3, layout=ttnn.TILE_LAYOUT)
        # output_tensorp3 = ttnn.permute(output_tensorp3, (0, 2, 3, 1))
        # output_tensorp3 = ttnn.reshape(output_tensorp3, (1, 1, 100, 500))
        # output_tensorc3 = ttnn.to_layout(output_tensorc3, layout=ttnn.TILE_LAYOUT)
        # output_tensorc3 = ttnn.permute(output_tensorc3, (0, 2, 3, 1))
        # output_tensorc3 = ttnn.reshape(output_tensorc3, (1, 1, 100, 500))
        # output_tensorc3 = ttnn.permute(output_tensorc3, (0, 2, 3, 1))
        output_tensorp1 = torch.reshape(output_tensorp1, (1, 512, 1, 100))
        output_tensorp2 = torch.reshape(output_tensorp2, (1, 512, 1, 100))
        output_tensorp3 = torch.reshape(output_tensorp3, (1, 512, 1, 100))
        output_tensorc3 = torch.reshape(output_tensorc3, (1, 512, 1, 100))
        output_tensorp1 = torch.permute(output_tensorp1, (0, 2, 3, 1))
        output_tensorp2 = torch.permute(output_tensorp2, (0, 2, 3, 1))
        output_tensorp3 = torch.permute(output_tensorp3, (0, 2, 3, 1))
        output_tensorc3 = torch.permute(output_tensorc3, (0, 2, 3, 1))

        output_tensorp1 = ttnn.from_torch(output_tensorp1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensorp2 = ttnn.from_torch(output_tensorp2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensorp3 = ttnn.from_torch(output_tensorp3, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensorc3 = ttnn.from_torch(output_tensorc3, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensorp1 = output_tensorp1.to(device)
        output_tensorp2 = output_tensorp2.to(device)
        output_tensorp3 = output_tensorp3.to(device)
        output_tensorc3 = output_tensorc3.to(device)
        # output_tensorp1 = tt_lib.tensor.sharded_to_interleaved(output_tensorp1, ttnn.L1_MEMORY_CONFIG)
        # output_tensorp1 = ttnn.to_layout(output_tensorp1, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat([output_tensorp1, output_tensorp2, output_tensorp3, output_tensorc3], dim=3)
        # output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = output_tensor.to(device, self.c4.conv.input_sharded_memory_config)
        # print("DEBUG:", output_tensor.memory_config())
        output_tensor = self.c4(output_tensor)
        output_tensor = self.c5(output_tensor)
        output_tensor = self.c6(output_tensor)
        output_tensor_9m = output_tensor
        output_tensor = self.c7(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.upsample(output_tensor, (1, 4, 1), memory_config=output_tensor.memory_config())

        # TODO add ttnn tensor here for testing
        #    input_shape = torch_input_tensor.shape
        #    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        #
        #    input_tensor = input_tensor.reshape(
        #        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
        #    )

        outDownSample4 = input_tensors[1].to(device, self.c7_2.conv.input_sharded_memory_config)
        # CBR block for conc2
        outDownSample4_c7 = self.c7_2(outDownSample4)
        #        outDownSample4_b7 = self.b7(outDownSample4_c7)
        #        outDownSample4_r7 = self.relu(outDownSample4_b7)
        #
        # output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        # output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        outDownSample4_c7 = tt_lib.tensor.sharded_to_interleaved(outDownSample4_c7, ttnn.L1_MEMORY_CONFIG)
        outDownSample4_c7 = ttnn.to_layout(outDownSample4_c7, layout=ttnn.TILE_LAYOUT)
        print(outDownSample4_c7.memory_config())
        print(output_tensor.memory_config())
        output_tensor = ttnn.concat([output_tensor, outDownSample4_c7], dim=3)

        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c7_3.conv.input_sharded_memory_config)
        output_tensor = self.c7_3(output_tensor)
        output_tensor = self.c8(output_tensor)
        output_tensor = self.c7_4(output_tensor)
        output_tensor = self.c8_2(output_tensor)
        output_tensor = self.c7_5(output_tensor)
        output_tensor_16m = output_tensor
        print(output_tensor.shape)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = output_tensor.to(device, self.c9.conv.input_sharded_memory_config)

        print(self.c9.conv.input_sharded_memory_config)
        print("Last config:", output_tensor.memory_config())
        output_tensor = self.c9(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.upsample(output_tensor, (1, 4, 1), memory_config=output_tensor.memory_config())
        # output_tensor = self.u(output_tensor)
        #        # CBR block for conc3
        #        # TODO add ttnn random tensor here
        outDownSample3 = input_tensors[2].to(device, self.c9_2.conv.input_sharded_memory_config)
        outDownSample3_c9 = self.c9_2(outDownSample3)
        #        outDownSample3_b9 = self.b9(outDownSample3_c9)
        #        outDownSample3_r9 = self.relu(outDownSample3_b9)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat([output_tensor, outDownSample3_c9], dim=3)
        output_tensor = output_tensor.to(device, self.c9_3.conv.input_sharded_memory_config)
        output_tensor = self.c9_3(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c10.conv.input_sharded_memory_config)
        print("out: ", output_tensor.layout)
        # print("c10: ", self.c10.output_layout)
        output_tensor = self.c10(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c9_4.conv.input_sharded_memory_config)
        output_tensor = self.c9_4(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c10_2.conv.input_sharded_memory_config)
        output_tensor = self.c10_2(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c9_5.conv.input_sharded_memory_config)
        output_tensor = self.c9_5(output_tensor)
        #        #        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        #        #        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        #        #        output_tensor = ttnn.concat([output_tensor, output_tensor_c3], dim=3)
        #
        #        #        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c8.conv.input_sharded_memory_config)
        #        #        output_tensor = self.c8(output_tensor)
        #
        return ttnn.from_device(output_tensor), ttnn.from_device(output_tensor_9m), ttnn.from_device(output_tensor_16m)
