import sys
import pathlib
import torch
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[2]))

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


"""
default:    [batch_size, dim_size]
llm:        [batch_size, q_seq_len, hidden_size]
        --> [num_tokens, hidden_size]
"""


class RMSNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.add_residual = self.args_dict.get("add_residual", False)
        if not self.add_residual in [True, False]:
            raise NotImplementedError

        self.epsilon = 1e-5

        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.torch_device_name,
                creator=torch.randn,
            ),
            "weight": OpTensorInfo(
                shape=[self.dim_size], dtype=self.torch_dtype, device=self.backend.torch_device_name, creator=torch.ones
            ),
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            ),
        }

        if self.add_residual:
            self.input_tensor_info["residual"] = OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        """
        https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html#rmsnorm
        1. vec * vec: [batch_size, hidden_size]
        2. vec reduce_sum: [batch_size, hidden_size]
        3. vec * scalar: [batch_size]
        4. vec + scalar: [batch_size]
        5. vec sqrt: [batch_size]
        6. vec / vec: [batch_size]
        7. vec * vec: [batch_size, hidden_size]
        """
        self.calc_flops = self.batch_size * (3 * self.dim_size + 4)
        if self.add_residual:
            self.calc_flops += self.batch_size * self.dim_size

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )

        self._run_func = self.rms_norm_run

    def rms_norm_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        weight = tensor_mapping["weight"]
        if self.add_residual:
            residual = tensor_mapping["residual"]
            src = src + residual
        dst = torch.nn.functional.rms_norm(
            input=src,
            normalized_shape=[self.dim_size],
            weight=weight,
            eps=self.epsilon,
        )
        return dst


class LayerNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]

            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "weight": OpTensorInfo(
                    shape=[self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "bias": OpTensorInfo(
                    shape=[self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
            }
            self.output_tensor_info = {
                "dst": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
            }
        else:
            raise NotImplementedError

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        self._run_func = self.layer_norm_run

    def layer_norm_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        weight = tensor_mapping["weight"]
        bias = tensor_mapping["bias"]
        dst = torch.nn.functional.layer_norm(src, (self.dim_size,), weight=weight, bias=bias)
        return dst


class SoftmaxOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
            self.dim = -1

            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }
            self.output_tensor_info = {
                "dst": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["batch_size"]
            self.head_num = self.args_dict["head_num"]
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.kv_seq_len = self.args_dict["kv_seq_len"]
            self.dim = -1

            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.head_num, self.q_seq_len, self.kv_seq_len],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }
            self.output_tensor_info = {
                "dst": OpTensorInfo(
                    shape=[self.batch_size, self.head_num, self.q_seq_len, self.kv_seq_len],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = self.input_tensor_size
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_in_out_tensors_func = partial(
            self._create_in_out_tensors, create_inputs=True, create_outputs=False
        )

        self._run_func = self.softmax_run

    def softmax_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.softmax(src, dim=self.dim)
        return dst
