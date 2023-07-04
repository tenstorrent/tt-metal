import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

import tt_lib as ttl
from tt_lib.utils import tilize_to_list, tilize, channels_last
import torch


def test_tilize_channels_last_tensor():
    pcie_0 = 0
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    pt_tensor = torch.randn(1, 32, 1, 32)
    cl_pt_tensor = channels_last(pt_tensor)
    cl_shape = cl_pt_tensor.shape[:]
    list_tensor = torch.flatten(cl_pt_tensor).tolist()
    print("Shape of cl pt tensor - " + str(cl_pt_tensor.shape))
    tt_tensor = ttl.tensor.Tensor(
        list_tensor,
        pt_tensor.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.CHANNELS_LAST,
        device,
    )
    tt_res = ttl.tensor.tilize(tt_tensor)
    tt_res_array = np.array(tt_res.to(host).data(), dtype=float).reshape(cl_shape)

    golden_pt_tensor = tilize(cl_pt_tensor)
    assert (
        abs(golden_pt_tensor - tt_res_array) < 0.02
    ).all(), "Max abs difference for tilize can be 0.02 due to bfloat conversions"

    del tt_res

    ttl.device.CloseDevice(device)
