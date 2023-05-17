from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    channels_last,
    _nearest_32,
    pad_activation,
)
import torch


def test_tilize_hpadding_cl():
    pcie_0 = 0
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    pt_tensor = torch.randn(1, 32 * 9, 1, 4)
    cl_pt_tensor = channels_last(pt_tensor)
    cl_shape = cl_pt_tensor.shape[:]
    list_tensor = torch.flatten(cl_pt_tensor).tolist()
    tt_tensor = ttl.tensor.Tensor(
        list_tensor,
        pt_tensor.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.CHANNELS_LAST,
        device,
    )
    tt_res = ttl.tensor.tilize_with_zero_padding(tt_tensor)
    padded_pt_tensor_shape = [
        pt_tensor.shape[0],
        pt_tensor.shape[1],
        pt_tensor.shape[2],
        _nearest_32(pt_tensor.shape[3]),
    ]
    cl_shape_padded = [cl_shape[0], cl_shape[1], _nearest_32(cl_shape[2]), cl_shape[3]]
    assert tt_res.shape() == cl_shape_padded
    tt_res_array = np.array(tt_res.to(host).data(), dtype=float).reshape(
        cl_shape_padded
    )
    print("Shape of cl pt tensor - " + str(cl_pt_tensor.shape))
    cl_pt_tensor_padded = pad_activation(cl_pt_tensor)
    print("Shape of cl pt tensor padded - " + str(cl_pt_tensor_padded.shape))
    assert list(cl_pt_tensor_padded.shape) == cl_shape_padded
    golden_pt_tensor = tilize(cl_pt_tensor_padded)
    assert (
        abs(golden_pt_tensor - tt_res_array) < 0.02
    ).all(), "Max abs difference for tilize can be 0.02 due to bfloat conversions"

    del tt_res

    ttl.device.CloseDevice(device)
