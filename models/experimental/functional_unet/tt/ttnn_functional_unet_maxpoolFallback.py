# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import tt_lib as ttl
import tt_lib.fallback_ops

from loguru import logger


def unet_reshard(
    ttnn_tensor,
    sharded_memory_config,
    use_reshard=True,
    interleaved_memory_config=ttnn.L1_MEMORY_CONFIG,
    dtype=None,
):
    if use_reshard:
        return ttnn.to_memory_config(
            ttnn_tensor,
            memory_config=sharded_memory_config,
        )
    else:
        ttl_tensor = ttnn_tensor.value
        ttl_tensor = ttl.tensor.sharded_to_interleaved(ttl_tensor, interleaved_memory_config, dtype)
        ttl_tensor = ttl.tensor.interleaved_to_sharded(
            ttl_tensor,
            sharded_memory_config,
            dtype,
        )
        return ttnn.Tensor(ttl_tensor)


class UNet:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c1_2 = parameters.c1_2
        self.p1 = parameters.p1
        self.c2 = parameters.c2
        self.c2_2 = parameters.c2_2
        self.p2 = parameters.p2
        self.c3 = parameters.c3
        self.c3_2 = parameters.c3_2
        self.p3 = parameters.p3
        self.c4 = parameters.c4
        self.c4_2 = parameters.c4_2
        self.p4 = parameters.p4
        self.bnc = parameters.bnc
        self.bnc_2 = parameters.bnc_2
        self.c5 = parameters.c5
        self.c5_2 = parameters.c5_2
        self.c5_3 = parameters.c5_3
        self.c6 = parameters.c6
        self.c6_2 = parameters.c6_2
        self.c6_3 = parameters.c6_3
        self.c7 = parameters.c7
        self.c7_2 = parameters.c7_2
        self.c7_3 = parameters.c7_3
        self.c8 = parameters.c8
        self.c8_2 = parameters.c8_2
        self.c8_3 = parameters.c8_3
        self.output_layer = parameters.output_layer

    def torch_call(self, torch_input_tensor):
        device_id = 0
        device = ttnn.open_device(device_id=device_id)
        input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)
        input_tensor = self.c1.copy_input_to_device(input_tensor)

        output_tensor = self.c1(input_tensor)
        output_tensor = self.c1_2(output_tensor)
        save_c1_2_out = output_tensor
        save_c1_2_out = ttnn.to_layout(save_c1_2_out, layout=ttnn.TILE_LAYOUT)
        output_tensor = self.p1(output_tensor)

        output_tensor = unet_reshard(
            output_tensor, self.c2.get_expected_memory_config(output_tensor.shape), use_reshard=False
        )
        output_tensor = self.c2(output_tensor)
        output_tensor = self.c2_2(output_tensor)
        save_c2_2_out = output_tensor
        save_c2_2_out = ttnn.to_layout(save_c2_2_out, layout=ttnn.TILE_LAYOUT)
        output_tensor = self.p2(output_tensor)

        output_tensor = unet_reshard(
            output_tensor, self.c3.get_expected_memory_config(output_tensor.shape), use_reshard=False
        )
        output_tensor = self.c3(output_tensor)
        output_tensor = self.c3_2(output_tensor)
        save_c3_2_out = output_tensor
        save_c3_2_out = ttnn.to_layout(save_c3_2_out, layout=ttnn.TILE_LAYOUT)

        output_tensor = self.p3(output_tensor)

        output_tensor = unet_reshard(
            output_tensor, self.c4.get_expected_memory_config(output_tensor.shape), use_reshard=False
        )
        output_tensor = self.c4(output_tensor)
        output_tensor = self.c4_2(output_tensor)
        save_c4_2_out = output_tensor

        if False:
            output_tensor = self.p4(output_tensor)
        else:
            output_tensor = self.c4_2.copy_output_from_device(output_tensor)
            output_tensor = ttnn.to_torch(output_tensor)
            output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
            output_tensor = output_tensor.to(torch_input_tensor.dtype)

            output_tensor = torch.nn.functional.max_pool2d(output_tensor, kernel_size=2, stride=2)
            output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
            output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
            output_tensor = self.bnc.copy_input_to_device(output_tensor)

        output_tensor = self.bnc(output_tensor)
        output_tensor = self.bnc_2(output_tensor)

        ## upsample block
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (1, 132, 10, 64))
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, 5280, 64))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        output_tensor = ttnn.concat([output_tensor, save_c4_2_out], dim=3)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (2, 132, 20, 96))
        output_tensor = self.c5.copy_input_to_device(output_tensor)
        output_tensor = self.c5(output_tensor)
        output_tensor = self.c5_2(output_tensor)
        output_tensor = self.c5_3(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (1, 264, 20, 32))
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, 21120, 32))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        output_tensor = ttnn.concat([output_tensor, save_c3_2_out], dim=3)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (2, 264, 40, 64))
        output_tensor = self.c6.copy_input_to_device(output_tensor)
        output_tensor = self.c6(output_tensor)
        output_tensor = self.c6_2(output_tensor)
        output_tensor = self.c6_3(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (1, 528, 40, 32))
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, 84480, 32))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        output_tensor = ttnn.concat([output_tensor, save_c2_2_out], dim=3)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (2, 528, 80, 48))
        output_tensor = self.c7.copy_input_to_device(output_tensor)
        output_tensor = self.c7(output_tensor)
        output_tensor = self.c7_2(output_tensor)
        output_tensor = self.c7_3(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (1, 1056, 80, 16))
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, 160 * 1056 * 2, 16))
        output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        output_tensor = ttnn.concat([output_tensor, save_c1_2_out], dim=3)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (2, 1056, 160, 32))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        output_tensor = self.c8.copy_input_to_device(output_tensor)
        output_tensor = self.c8(output_tensor)
        output_tensor = self.c8_2(output_tensor)
        output_tensor = self.c8_3(output_tensor)
        output_tensor = self.output_layer(output_tensor)
        output_tensor = self.output_layer.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        ttnn.close_device(device)
        return output_tensor
