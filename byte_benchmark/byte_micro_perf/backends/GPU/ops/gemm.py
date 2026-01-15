import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from core.ops.tensor_gemm_ops import GemmOp

OP_MAPPING = {}


class GPUGemmOp(GemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        if self.dtype == "float32":
            # use float32 gemm
            torch.set_float32_matmul_precision("highest")
            # torch.backends.cuda.matmul.allow_tf32 = False
            # torch.backends.cudnn.allow_tf32 = False
        elif self.dtype == "tfloat32":
            # use tfloat32 gemm
            torch.set_float32_matmul_precision("high")
            # torch.backends.cuda.matmul.allow_tf32 = True
            # torch.backends.cudnn.allow_tf32 = True

        if self.dtype in ["int8"]:
            raise NotImplementedError("GPU does not support int8 gemm currently")


OP_MAPPING["torch"] = GPUGemmOp
