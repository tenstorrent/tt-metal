import sys
import pathlib
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from core.ops.xccl_ops import ReduceScatterOp

OP_MAPPING = {"torch": ReduceScatterOp}
