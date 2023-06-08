import torch
import numpy as np
from loguru import logger

import tt_lib


def get_shape(shape):
    """Insert 1's in the begining of shape list until the len(shape) = 4"""
    if len(shape) <= 4:
        new_shape = [1 for i in range(4 - len(shape))]
        new_shape.extend(shape)
    else:
        new_shape = shape
    return new_shape


def is_torch_tensor(x):
    if type(x) is torch.Tensor:
        return True
    else:
        return False
