import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from core.ops.vector_reduction_ops import ReduceSumOp

OP_MAPPING = {"torch": ReduceSumOp}
