from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import numpy as np

from pymetal import ttmetal as ttm
from pymetal.ttmetal.utils import tilize_to_list, tilize, channels_last, _nearest_32, pad_activation
import torch

if __name__ == "__main__":
    pcie_0 = 0
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    pt_tensor = torch.randn(1, 32, 1, 28)
    cl_pt_tensor = channels_last(pt_tensor)
    cl_shape = cl_pt_tensor.shape[:]
    list_tensor = torch.flatten(cl_pt_tensor).tolist()
    tt_tensor = ttm.tensor.Tensor(
        list_tensor,
        pt_tensor.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.CHANNELS_LAST,
        device
    )
    tt_res = ttm.tensor.tilize_with_zero_padding(tt_tensor)
    padded_pt_tensor_shape = [pt_tensor.shape[0],pt_tensor.shape[1],pt_tensor.shape[2],_nearest_32(pt_tensor.shape[3])]
    cl_shape_padded = [cl_shape[0],cl_shape[1],_nearest_32(cl_shape[2]),cl_shape[3]]
    assert(tt_res.shape() == cl_shape_padded)
    tt_res_array = np.array(tt_res.to(host).data(), dtype=float).reshape(cl_shape_padded)
    print("Shape of cl pt tensor - " + str(cl_pt_tensor.shape))
    cl_pt_tensor_padded = pad_activation(cl_pt_tensor)
    print("Shape of cl pt tensor padded - " + str(cl_pt_tensor_padded.shape))
    assert(list(cl_pt_tensor_padded.shape) == cl_shape_padded)
    golden_pt_tensor = tilize(cl_pt_tensor_padded)
    assert (abs(golden_pt_tensor - tt_res_array) < 0.02).all(), "Max abs difference for tilize can be 0.02 due to bfloat conversions"
    ttm.device.CloseDevice(device)
