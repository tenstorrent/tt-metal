import os
import sys
import logging
import torch
from collections import namedtuple
from dataclasses import dataclass


# logger functions
logger = logging.getLogger("bytemlperf_micro_perf")


def setup_logger(loglevel: str):
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(loglevel.upper())
    logger.propagate = False


def default_creator(size, dtype, device):
    if dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
        return torch.randn(size=size, dtype=dtype, device=device)
    elif dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        return torch.randint(low=-16, high=17, size=size, dtype=dtype, device=device)
    elif dtype in [torch.uint64, torch.uint32, torch.uint16, torch.uint8]:
        return torch.randnint(low=0, high=17, size=size, dtype=dtype, device=device)
    else:
        raise NotImplementedError


# shape: list or tuple
# dtype: torch.dtype
# device: str
# creator: func, default is torch.zeros
OpTensorInfo = namedtuple(
    "OpTensorInfo", ["shape", "dtype", "device", "creator"], defaults=[torch.float32, "cpu", default_creator]
)


def calc_tensor_size(tensor_info: OpTensorInfo):
    tensor_size = 1
    for dim in tensor_info.shape:
        tensor_size *= dim
    dtype_size = torch.tensor([], dtype=tensor_info.dtype).element_size()
    tensor_size *= dtype_size
    return tensor_size


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()
