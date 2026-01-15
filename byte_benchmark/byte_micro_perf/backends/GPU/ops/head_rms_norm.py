import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from core.ops.llm_ops import HeadRMSNormOp

OP_MAPPING = {"torch": HeadRMSNormOp}
