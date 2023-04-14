import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset

from libs import tt_lib as ttl
from utility_functions import tt2torch_tensor, torch2tt_tensor

from libs.tt_lib.utils import _nearest_32 as nearest_32


def pad_by_zero(x: torch.Tensor, device):
    initial_shape = x.shape
    if x.shape[3] % 32 != 0 or x.shape[2] % 32 != 0:
        tt_tensor = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        )
        x = tt_tensor.pad((x.shape[0], x.shape[1], nearest_32(x.shape[2]), nearest_32(x.shape[3])), (0, 0, 0, 0), 0)
        x = x.to(ttl.tensor.Layout.TILE).to(device)

    else:
        x = torch2tt_tensor(x, device)
    return x, initial_shape

def unpad_from_zero(x, desired_shape, host):
    if x.shape()[-1] == desired_shape[-1] and x.shape()[-2] == desired_shape[-2] :
        x = tt2torch_tensor(x)
    else:
        x = x.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
        x = x.unpad((0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1) )
        x = torch.Tensor(x.data()).reshape(x.shape())
    return x
