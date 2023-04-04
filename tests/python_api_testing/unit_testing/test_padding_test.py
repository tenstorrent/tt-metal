import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np
import torch

from libs import tt_lib as ttl


@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_shape, input_tensor_start, pad_value",
    (((1, 1, 3, 4), (1, 1, 32, 32), (0, 0, 1, 1), 0),),
)
def test_run_padding_test(
    input_tensor_shape, output_tensor_shape, input_tensor_start, pad_value
):
    # Args for unpad
    output_tensor_start = input_tensor_start
    output_tensor_end = tuple(
        input_tensor_start[i] + input_tensor_shape[i] - 1
        for i in range(len(input_tensor_shape))
    )

    inp = torch.rand(*input_tensor_shape)
    ones = torch.ones(*input_tensor_shape)

    # Create tensor on host
    a = ttl.tensor.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    b = ttl.tensor.Tensor(
        ones.reshape(-1).tolist(),
        input_tensor_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    # Pad inputs on host
    a_pad = a.pad(output_tensor_shape, input_tensor_start, pad_value)
    b_pad = b.pad(output_tensor_shape, input_tensor_start, pad_value)

    # Run add op on device with padded tensors
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    a_dev = a_pad.to(ttl.tensor.Layout.TILE).to(device)
    b_dev = b_pad.to(ttl.tensor.Layout.TILE).to(device)
    out_dev = ttl.tensor.add(a_dev, b_dev)
    out_pad = out_dev.to(host).to(ttl.tensor.Layout.ROW_MAJOR)

    # Unpad out to get result
    out = out_pad.unpad(output_tensor_start, output_tensor_end)
    out_pt = torch.Tensor(out.data()).reshape(*input_tensor_shape)

    out_ref = inp + ones

    print("\n", out_pt)
    print("\n", out_ref)

    assert torch.allclose(out_pt, out_ref, rtol=1e-2)
