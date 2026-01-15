import sys
import pathlib
import torch
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[2]))

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


class GemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]

        if self.arg_type == "default":
            self.M = self.args_dict["M"]
            self.K = self.args_dict["K"]
            self.N = self.args_dict["N"]
        elif self.arg_type == "llm":
            self.M = self.args_dict["num_tokens"]
            self.K = self.args_dict["hidden_size"]
            self.N = self.args_dict["new_hidden_size"]

        # bf16 * bf16 --> bf16
        # fp16 * fp16 --> fp16
        if self.dtype in ["bfloat16", "float16"]:
            self.torch_dtype = getattr(torch, self.dtype)
            self.out_dtype = self.torch_dtype
        # fp32 * fp32 --> fp32
        elif self.dtype == "float32":
            self.torch_dtype = torch.float32
            self.out_dtype = torch.float32
        # fp32(tf32) * fp32(tf32) --> fp32
        elif self.dtype == "tfloat32":
            self.torch_dtype = torch.float32
            self.out_dtype = torch.float32
        elif self.dtype == "int8":
            self.torch_dtype = torch.int8
            self.out_dtype = torch.bfloat16
        else:
            raise NotImplementedError

        if self.dtype in ["float32", "tfloat32", "float16", "bfloat16"]:
            self.input_tensor_info = {
                "a": OpTensorInfo(
                    shape=[self.M, self.K],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "b": OpTensorInfo(
                    shape=[self.K, self.N],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
            }
            self.output_tensor_info = {
                "c": OpTensorInfo(
                    shape=[self.M, self.N],
                    dtype=self.out_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }
        elif self.dtype == "int8":
            self.input_tensor_info = {
                "a": OpTensorInfo(
                    shape=[self.M, self.K],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "b": OpTensorInfo(
                    shape=[self.K, self.N],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "a_scale": OpTensorInfo(
                    shape=[self.M],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                ),
                "b_scale": OpTensorInfo(
                    shape=[self.N],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                ),
            }
            self.output_tensor_info = {
                "c": OpTensorInfo(
                    shape=[self.M, self.N],
                    dtype=self.out_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = self.M * self.N * self.K * 2

        self._run_func = self.gemm_run

    def gemm_run(self, tensor_mapping):
        if self.dtype in ["float32", "tfloat32", "float16", "bfloat16"]:
            a = tensor_mapping["a"]
            b = tensor_mapping["b"]
            c = tensor_mapping["c"]
            torch.matmul(a, b, out=c)
            return c
        else:
            raise NotImplementedError
