# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# this code is testing the published example.rst documentation

import torch
import tt_lib as tt_lib

from tt_lib.fallback_ops import fallback_ops

if __name__ == "__main__":
    # Initialize TT Accelerator device on PCI slot 0
    tt_device = tt_lib.device.CreateDevice(0)

    # Create random PyTorch tensor
    py_tensor = torch.randn((1, 1, 32, 32))
    py_tensor_exp = torch.randint(0, 10, (1, 1, 32, 32))

    # Create TT tensor from PyTorch Tensor and send it to TT accelerator device
    tt_tensor = tt_lib.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),
        py_tensor.size(),
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
        tt_device,
    )

    # Run relu on TT accelerator device
    tt_relu_out = tt_lib.tensor.relu(tt_tensor)

    # Move TT Tensor tt_relu_out to host and convert it to PyTorch tensor py_relu_out
    tt_relu_out = tt_relu_out.cpu()
    py_relu_out = torch.Tensor(tt_relu_out.data()).reshape(tt_relu_out.get_legacy_shape())

    # Execute pow using PyTorch (since pow is not available from tt_lib)
    py_pow_out = torch.pow(py_relu_out, py_tensor_exp)

    # Create TT Tensor from py_pow_out and move it to TT accelerator device
    tt_pow_out = tt_lib.tensor.Tensor(
        py_pow_out.reshape(-1).tolist(),
        py_pow_out.size(),
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
        tt_device,
    )

    # Run silu on TT Tensor tt_pow_out
    # This is a fallback op and it will behave like regular ops on TT accelerator device,
    # even though under the hood this op is executed on host.
    tt_silu_out = tt_lib.tensor.silu(tt_pow_out)

    # Run exp on TT accelerator device
    tt_exp_out = tt_lib.tensor.exp(tt_silu_out)

    # Move TT Tensor output from TT accelerator device to host
    tt_output = tt_exp_out.cpu()

    # Print TT Tensor
    print(tt_output)

    # Close TT accelerator device
    tt_lib.device.CloseDevice(tt_device)
