# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import tt_lib as ttl
import tt_lib.fallback_ops

from loguru import logger


class UNet:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c1_2 = parameters.c1_2
        self.c2 = parameters.c2
        self.c2_2 = parameters.c2_2
        self.c3 = parameters.c3
        self.c3_2 = parameters.c3_2
        self.c4 = parameters.c4
        self.c4_2 = parameters.c4_2
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

    #    def __call__(self, x):
    #        identity = x
    #
    #        # Relu and bn1 are fused with conv1
    #        out = self.c1(x)
    #
    #        out = self.c1_2(out)
    #        out = self.c2(out)
    #        out = self.c2_2(out)
    #        out = self.c3(out)
    #        out = self.c3_2(out)
    #        out = self.c4(out)
    #        out = self.c4_2(out)
    #        out = self.bnc(out)
    #        out = self.bnc_2(out)
    #        out = self.c5(out)
    #        out = self.c5_2(out)
    #
    #        return out

    def torch_call(self, torch_input_tensor):
        device_id = 0
        device = ttnn.open_device(device_id=device_id)
        input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)
        input_tensor = self.c1.copy_input_to_device(input_tensor)

        output_tensor = self.c1(input_tensor)
        output_tensor = self.c1_2(output_tensor)
        output_tensor = self.c1_2.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)
        save_c1_2_out = output_tensor
        output_tensor = torch.nn.functional.max_pool2d(output_tensor, kernel_size=2, stride=2)

        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.c2.copy_input_to_device(output_tensor)
        output_tensor = self.c2(output_tensor)
        output_tensor = self.c2_2(output_tensor)
        output_tensor = self.c2_2.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)
        save_c2_2_out = output_tensor

        output_tensor = torch.nn.functional.max_pool2d(output_tensor, kernel_size=2, stride=2)

        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.c3.copy_input_to_device(output_tensor)
        output_tensor = self.c3(output_tensor)
        output_tensor = self.c3_2(output_tensor)
        output_tensor = self.c3_2.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)
        save_c3_2_out = output_tensor

        output_tensor = torch.nn.functional.max_pool2d(output_tensor, kernel_size=2, stride=2)

        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.c4.copy_input_to_device(output_tensor)
        output_tensor = self.c4(output_tensor)
        output_tensor = self.c4_2(output_tensor)
        output_tensor = self.c4_2.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)
        save_c4_2_out = output_tensor

        output_tensor = torch.nn.functional.max_pool2d(output_tensor, kernel_size=2, stride=2)
        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.bnc.copy_input_to_device(output_tensor)
        output_tensor = self.bnc(output_tensor)
        output_tensor = self.bnc_2(output_tensor)
        #        output_tensor_bnr2 = output_tensor
        #        output_tensor_bnr2 = self.bnc_2.copy_output_from_device(output_tensor_bnr2)
        #        output_tensor_bnr2 = ttnn.to_torch(output_tensor_bnr2)
        #        output_tensor_bnr2 = torch.permute(output_tensor_bnr2, (0, 3, 1, 2))
        #        output_tensor_bnr2 = output_tensor_bnr2.to(torch_input_tensor.dtype)

        # output_tensor = torch.nn.functional.interpolate(output_tensor, scale_factor=2, mode="bilinear")
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.to_torch(output_tensor)
        # output_tensor = torch.reshape(output_tensor, (2, 64, 132, 20))

        save_c4_2_out = torch.permute(save_c4_2_out, (0, 2, 3, 1))
        save_c4_2_out = ttnn.from_torch(save_c4_2_out, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        # output_tensor = ttnn.from_torch(output_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        output_tensor = ttnn.from_torch(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        output_tensor = ttnn.reshape(output_tensor, (2, 132, 20, 64))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        logger.debug(f"output_tensor: {output_tensor.shape}")
        logger.debug(f"output_tensor: {save_c4_2_out.shape}")
        output_tensor = ttnn.concat([output_tensor, save_c4_2_out], dim=3)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.c5.copy_input_to_device(output_tensor)
        output_tensor = self.c5(output_tensor)
        output_tensor = self.c5_2(output_tensor)
        output_tensor = self.c5_3(output_tensor)
        # output_tensor = self.c5_2.copy_output_from_device(output_tensor)
        # output_tensor = ttnn.to_torch(output_tensor)
        # output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        # output_tensor = output_tensor.to(torch_input_tensor.dtype)

        # output_tensor = torch.nn.functional.interpolate(output_tensor, scale_factor=2, mode="bilinear")

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.to_torch(output_tensor)

        save_c3_2_out = torch.permute(save_c3_2_out, (0, 2, 3, 1))
        save_c3_2_out = ttnn.from_torch(save_c3_2_out, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        output_tensor = ttnn.reshape(output_tensor, (2, 264, 40, 32))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        output_tensor = ttnn.concat([output_tensor, save_c3_2_out], dim=3)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.c6.copy_input_to_device(output_tensor)
        output_tensor = self.c6(output_tensor)
        output_tensor = self.c6_2(output_tensor)
        output_tensor = self.c6_3(output_tensor)
        # output_tensor = self.c6_2.copy_output_from_device(output_tensor)
        # output_tensor = ttnn.to_torch(output_tensor)
        # output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        # output_tensor = output_tensor.to(torch_input_tensor.dtype)

        # output_tensor = torch.nn.functional.interpolate(output_tensor, scale_factor=2, mode="bilinear")

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.to_torch(output_tensor)

        save_c2_2_out = torch.permute(save_c2_2_out, (0, 2, 3, 1))
        save_c2_2_out = ttnn.from_torch(save_c2_2_out, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        output_tensor = ttnn.reshape(output_tensor, (2, 528, 80, 32))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        output_tensor = ttnn.concat([output_tensor, save_c2_2_out], dim=3)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.c7.copy_input_to_device(output_tensor)
        output_tensor = self.c7(output_tensor)
        output_tensor = self.c7_2(output_tensor)
        output_tensor = self.c7_3(output_tensor)
        # output_tensor = self.c7_2.copy_output_from_device(output_tensor)
        # output_tensor = ttnn.to_torch(output_tensor)
        # output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        # output_tensor = output_tensor.to(torch_input_tensor.dtype)

        # output_tensor = torch.nn.functional.interpolate(output_tensor, scale_factor=2, mode="bilinear")

        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.to_torch(output_tensor)

        save_c1_2_out = torch.permute(save_c1_2_out, (0, 2, 3, 1))
        save_c1_2_out = ttnn.from_torch(save_c1_2_out, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        output_tensor = ttnn.reshape(output_tensor, (2, 1056, 160, 16))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        output_tensor = ttnn.concat([output_tensor, save_c1_2_out], dim=3)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
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
