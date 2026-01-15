import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from core.ops.xccl_ops import P2POp

OP_MAPPING = {"torch": P2POp}
