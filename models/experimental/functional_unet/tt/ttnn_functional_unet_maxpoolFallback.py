# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import tt_lib as ttl
import tt_lib.fallback_ops


class UNet:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c1_2 = parameters.c1_2
        # self.p1 = parameters.p1
        self.c2 = parameters.c2
        self.c2_2 = parameters.c2_2
        self.c3 = parameters.c3
        self.c3_2 = parameters.c3_2
        self.c4 = parameters.c4
        self.c4_2 = parameters.c4_2
        self.bnc = parameters.bnc
        self.bnc_2 = parameters.bnc_2

    def __call__(self, x):
        identity = x

        # Relu and bn1 are fused with conv1
        out = self.c1(x)

        # Relu and bn2 are fused with conv1
        out = self.c1_2(out)
        out = self.c2(out)
        out = self.c2_2(out)
        out = self.c3(out)
        out = self.c3_2(out)
        out = self.c4(out)
        out = self.c4_2(out)
        out = self.bnc(out)
        out = self.bnc_2(out)
        # out = self.p1(out)

        # out = ttnn.add(out, identity, memory_config=ttnn.get_memory_config(out))
        # out = ttnn.to_memory_config(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # out = self.relu(out)

        return out

    def torch_call(self, torch_input_tensor):
        #        input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        #        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)
        #
        #        input_tensor = self.c1.copy_input_to_device(input_tensor)
        #        output_tensor = self(input_tensor)
        #        output_tensor = self.bnc_2.copy_output_from_device(output_tensor)
        #        # output_tensor = self.c1_2.copy_output_from_device(input_tensor)
        #
        #        output_tensor = ttnn.to_torch(output_tensor)
        #        print("the shape before change is: ", output_tensor.size())
        #        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        #        print("the shape after change is: ", output_tensor.size())
        #        print(" torch_input_tensor.shape: ", torch_input_tensor.shape)
        #        # output_tensor = torch.reshape(output_tensor, torch_input_tensor.shape)
        #        output_tensor = output_tensor.to(torch_input_tensor.dtype)
        #
        #
        #        #pt_nn = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        #        pt_nn = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #        pt_out = pt_nn(output_tensor)
        #        t0 = ttl.tensor.Tensor(
        #            output_tensor.reshape(-1).tolist(),
        #            output_tensor.shape,
        #            ttl.tensor.DataType.BFLOAT16,
        #            ttl.tensor.Layout.ROW_MAJOR,
        #        )
        #
        #        device_id = 0
        #        device = ttnn.open(device_id)
        #        kernel_size=2
        #        stride=2
        #        padding=0
        #        dilation=1
        #        return_indices=False
        #        ceil_mode=False
        #        t0 = t0.to(device)
        #        tt_nn = ttl.fallback_ops.MaxPool2d(
        #            kernel_size,
        #            stride,
        #            padding,
        #            dilation,
        #            return_indices,
        #            ceil_mode,
        #        )
        #        t1 = tt_nn(t0)
        #
        #        output_tensor = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)
        input_tensor = self.c1.copy_input_to_device(input_tensor)

        output_tensor = self.c1(input_tensor)
        output_tensor = self.c1_2(output_tensor)
        output_tensor = self.c1_2.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        # fall back maxpool to cpu
        pt_nn = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # pt_out = pt_nn(output_tensor)
        t0 = ttl.tensor.Tensor(
            output_tensor.reshape(-1).tolist(),
            output_tensor.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        device_id = 0
        device = ttnn.open(device_id)
        kernel_size = 2
        stride = 2
        padding = 0
        dilation = 1
        return_indices = False
        ceil_mode = False
        t0 = t0.to(device)
        tt_nn = ttl.fallback_ops.MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode,
        )
        t1 = tt_nn(t0)

        output_tensor = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        print("The size of torch tensor before permute is: ", output_tensor.size())
        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.c2.copy_input_to_device(output_tensor)
        output_tensor = self.c2(output_tensor)
        output_tensor = self.c2_2(output_tensor)
        output_tensor = self.c2_2.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        """

        input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)
        input_tensor = self.c1.copy_input_to_device(input_tensor)

        output_tensor = self.c1(input_tensor)
        output_tensor = self.c1_2(output_tensor)
        output_tensor = self.c1_2.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)
        """

        # fall back maxpool to cpu
        pt_nn = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # pt_out = pt_nn(output_tensor)
        t0 = ttl.tensor.Tensor(
            output_tensor.reshape(-1).tolist(),
            # output_tensor.tolist(),
            output_tensor.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        device_id = 0
        device = ttnn.open(device_id)
        kernel_size = 2
        stride = 2
        padding = 0
        dilation = 1
        return_indices = False
        ceil_mode = False
        t0 = t0.to(device)

        # t0_shape = t0.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().size()
        # print("t0_shape: ", t0_shape)

        tt_nn = ttl.fallback_ops.MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode,
        )
        t1 = tt_nn(t0)

        output_tensor = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.c3.copy_input_to_device(output_tensor)
        output_tensor = self.c3(output_tensor)
        output_tensor = self.c3_2(output_tensor)
        output_tensor = self.c3_2.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        # fall back maxpool to cpu
        pt_nn = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # pt_out = pt_nn(output_tensor)
        t0 = ttl.tensor.Tensor(
            output_tensor.reshape(-1).tolist(),
            output_tensor.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        device_id = 0
        device = ttnn.open(device_id)
        kernel_size = 2
        stride = 2
        padding = 0
        dilation = 1
        return_indices = False
        ceil_mode = False
        t0 = t0.to(device)
        tt_nn = ttl.fallback_ops.MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode,
        )
        t1 = tt_nn(t0)

        output_tensor = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.c4.copy_input_to_device(output_tensor)
        output_tensor = self.c4(output_tensor)
        output_tensor = self.c4_2(output_tensor)
        output_tensor = self.c4_2.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        # fall back maxpool to cpu
        pt_nn = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # pt_out = pt_nn(output_tensor)
        t0 = ttl.tensor.Tensor(
            output_tensor.reshape(-1).tolist(),
            output_tensor.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        device_id = 0
        device = ttnn.open(device_id)
        kernel_size = 2
        stride = 2
        padding = 0
        dilation = 1
        return_indices = False
        ceil_mode = False
        t0 = t0.to(device)
        tt_nn = ttl.fallback_ops.MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode,
        )
        t1 = tt_nn(t0)

        output_tensor = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        output_tensor = torch.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16)
        output_tensor = self.bnc.copy_input_to_device(output_tensor)
        output_tensor = self.bnc(output_tensor)
        output_tensor = self.bnc_2(output_tensor)
        output_tensor = self.bnc_2.copy_output_from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        return output_tensor
