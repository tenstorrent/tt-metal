import torch
import json
import numpy as np
import tt_lib


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = (
        tt_lib.tensor.Tensor(
            py_tensor.reshape(-1).tolist(),
            size,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        .to(tt_lib.tensor.Layout.TILE)
        .to(tt_device)
    )

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    host = tt_lib.device.GetHost()
    tt_output = tt_tensor.to(host).to(tt_lib.tensor.Layout.ROW_MAJOR)
    py_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())
    return py_output
