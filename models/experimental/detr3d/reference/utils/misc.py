# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import numpy as np
from collections import deque
from typing import List
from models.experimental.detr3d.reference.utils.dist import is_distributed, barrier, all_reduce_sum


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


@torch.jit.ignore
def to_list_1d(arr) -> List[float]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr


@torch.jit.ignore
def to_list_3d(arr) -> List[List[List[float]]]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr


def huber_loss(error, delta=1.0):
    """
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return loss


# From https://github.com/facebookresearch/detr/blob/master/util/misc.py
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_distributed():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        barrier()
        all_reduce_sum(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
