
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch

from pymetal import tt_lib as ttl
from utility_functions import untilize, tilize, tilize_to_list




def tt_to_torch_tensor(tt_tensor, host):
    tt_tensor = tt_tensor.to(host).to(tt_lib.tensor.Layout.ROW_MAJOR)

    # create a 1D PyTorch tensor from values in TT Tensor obtained with data() member function
    # and then reshape PyTorch tensor to shape of TT Tensor
    py_tensor = torch.Tensor(tt_tensor.data()).reshape(tt_tensor.shape())


def torch_to_tt_tensor(py_tensor, device):
    tt_tensor = (
        tt_lib.tensor.Tensor(
            py_tensor.reshape(-1).tolist(), # PyTorch tensor flatten into a list of floats
            py_tensor.size(),               # shape of TT Tensor that will be created
            tt_lib.tensor.DataType.BFLOAT16, # data type that will be used in created TT Tensor
            tt_lib.tensor.Layout.ROW_MAJOR,  # memory layout that will be used in created TT Tensor
        )
        .to(tt_lib.tensor.Layout.TILE)     # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(device)                         # move TT Tensor from host to TT accelerator device (device is of type tt_lib.device.Device)
    )
    return tt_tensor
