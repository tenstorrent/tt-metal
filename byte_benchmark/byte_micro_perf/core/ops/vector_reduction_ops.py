import sys
import pathlib
import torch
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[2]))

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


class ReduceMaxOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }
        self.output_tensor_info = {
            "max_value": OpTensorInfo(
                shape=[self.batch_size, 1],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "max_indices": OpTensorInfo(
                shape=[self.batch_size, 1],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
            ),
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        self._run_func = self.reduce_max_run

    def reduce_max_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        max_value, max_indices = torch.max(src, dim=-1, keepdim=True)
        return max_value, max_indices


class ReduceMinOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }
        self.output_tensor_info = {
            "min_value": OpTensorInfo(
                shape=[self.batch_size, 1],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "min_indices": OpTensorInfo(
                shape=[self.batch_size, 1],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
            ),
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        self._run_func = self.reduce_min_run

    def reduce_min_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        min_value, min_indices = torch.min(src, dim=-1, keepdim=True)
        return min_value, min_indices


class ReduceSumOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }
        self.output_tensor_info = {
            "sum": OpTensorInfo(
                shape=[self.batch_size, 1],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        self._run_func = self.reduce_sum_run

    def reduce_sum_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        sum = torch.sum(src, dim=-1, keepdim=True)
        return sum


class TopkOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]
        self.k = self.args_dict["k"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "value": OpTensorInfo(
                shape=[self.batch_size, self.k],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "indice": OpTensorInfo(
                shape=[self.batch_size, self.k],
                dtype=torch.int64,
                device=self.backend.get_torch_device_name(),
            ),
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_in_out_tensors_func = partial(
            self._create_in_out_tensors, create_inputs=True, create_outputs=False
        )

        self._run_func = self.topk_run

    def topk_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        value, indice = torch.topk(src, self.k, dim=-1, largest=True, sorted=False)
        return value, indice
