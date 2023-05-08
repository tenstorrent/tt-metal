from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from abc import abstractmethod
import torch
import math
from torch.nn import functional as F

from libs import tt_lib as ttm
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
import numpy as np
import python_api_testing.models.bloom.bloom_utils as bloom_utils
from libs import tt_lib as ttl


def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0) -> ttm.tensor.Tensor:

    input_shape = input.shape()

    # print(f"input_shape {input_shape}")
    # print(f"batch1 shape {batch1.shape}")
    # print(f"batch2 shape {batch2.shape}")

    if beta != 1.0:
        input = ttm.tensor.mul(beta, input)

    tmp = ttm.tensor.bmm(batch1, batch2)

    if alpha != 1.0:
        tmp = ttm.tensor.mul(alpha, tmp)

    result = ttm.tensor.add(input, tmp)

    return result
