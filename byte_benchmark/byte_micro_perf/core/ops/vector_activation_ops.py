import sys
import pathlib
import torch
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[2]))

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp
from core.ops.vector_sfu_ops import CosOp


class GeluOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.gelu_run
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

    def gelu_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.gelu(src)
        return dst


class SiluOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.silu_run
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

    def silu_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.silu(src)
        return dst
