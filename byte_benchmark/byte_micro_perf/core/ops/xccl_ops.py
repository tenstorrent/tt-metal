import sys
import pathlib
import torch
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[2]))

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


class AllReduceOp(BasicOp):
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

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        # [world_size, batch_size // world_size, dim_size] -->
        # [world_size, batch_size // world_size, dim_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.randn,
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.randn,
            )
        }

        # inplace operation
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = 2 * (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = self.batch_size * self.dim_size

        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        self._run_func = self.all_reduce_run

    def all_reduce_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_reduce(src, op=dist_module.ReduceOp.SUM, group=self.op_group)
        return src


class ReduceScatterOp(BasicOp):
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

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        # [world_size, batch_size // world_size, dim_size] -->
        # [1, batch_size // world_size, dim_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.randn,
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size // self.world_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.randn,
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = self.batch_size * self.dim_size

        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        self._run_func = self.reduce_scatter_run

    def reduce_scatter_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.reduce_scatter_tensor(dst, src, op=dist_module.ReduceOp.SUM, group=self.op_group)
        return dst


class AllGatherOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        # [1, batch_size // world_size, dim_size] -->
        # [world_size, batch_size // world_size, dim_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size // self.world_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.output_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        self._run_func = self.all_gather_run

    def all_gather_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_gather_into_tensor(dst, src, group=self.op_group)
        return dst


class AlltoAllOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        # [world_size, batch_size // world_size, dim_size]
        # --> [world_size, batch_size // world_size, dim_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.output_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        self._run_func = self.all_to_all_run

    def all_to_all_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_to_all_single(dst, src, group=self.op_group)
        return dst


class BroadcastOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        # [batch_size, dim_size] -->
        # [batch_size, dim_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.world_size, self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.world_size, self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size / self.world_size
        self.write_bytes = self.input_tensor_size / self.world_size * (self.world_size - 1)
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.tensor_size
        self.bus_size = self.algo_size
        self.calc_flops = 0

        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        self._run_func = self.broadcast

    def broadcast(self, tensor_mapping):
        src = tensor_mapping["src"]
        dist_module = self.backend.get_dist_module()
        for i in range(self.world_size):
            dist_module.broadcast(src[i], src=i, group=self.op_group)
        return src

    def summary(self, latency_us):
        target_dict, env_dict = super().summary(latency_us)
        if target_dict:
            target_dict["latency(us)"] = round(target_dict["latency(us)"] / self.world_size, 3)
            target_dict["algo_size(B)"] = round(target_dict["algo_size(B)"] / self.world_size, 3)
            target_dict["bus_size(B)"] = round(target_dict["bus_size(B)"] / self.world_size, 3)

        return target_dict, env_dict


class P2POp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.src_device = self.args_dict["src_device"]
        self.dst_device = self.args_dict["dst_device"]

        dist_module = self.backend.get_dist_module()
        self.world_size = self.group_size
        self.local_rank = dist_module.get_rank(group=self.op_group)

        # self.next_device = (self.local_rank + 1) % self.world_size
        # self.last_device = (self.local_rank - 1 + self.world_size) % self.world_size

        self.input_tensor_info = {
            "send": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }
        self.output_tensor_info = {
            "recv": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = self.algo_size

        self.calc_flops = 0

        self._run_func = self.p2p_run

    def p2p_run(self, tensor_mapping):
        send_tensor = tensor_mapping["send"]
        recv_tensor = tensor_mapping["recv"]
        dist_module = self.backend.get_dist_module()

        if self.src_device != self.dst_device:
            reqs = []
            if self.local_rank == self.src_device:
                reqs.append(dist_module.isend(send_tensor, self.dst_device, group=self.op_group))
            if self.local_rank == self.dst_device:
                reqs.append(dist_module.irecv(recv_tensor, self.src_device, group=self.op_group))
            for req in reqs:
                req.wait()
        else:
            recv_tensor.copy_(send_tensor)

        # # 0 --> 1
        # # 0 --> 1 --> 2 --> 3
        # # 0 --> 1 --> 2 --> 3 --> 4 --> 5 --> 6 --> 7 --> 8
        # reqs = []
        # if self.local_rank != self.world_size - 1:
        #     reqs.append(dist_module.isend(send_tensor, self.next_device, group=self.op_group))
        # if self.local_rank != 0:
        #     reqs.append(dist_module.irecv(recv_tensor, self.last_device, group=self.op_group))
        # for req in reqs:
        #     req.wait()


class Host2DeviceOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size], dtype=self.torch_dtype, device="cpu", creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }

        self.input_tensor_size = 0
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = 0
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.tensor_size
        self.bus_size = self.tensor_size

        self.calc_flops = 0

        self._run_func = self.host2device_run

    def host2device_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dst.copy_(src)
        return dst


class Device2HostOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size], dtype=self.torch_dtype, device="cpu", creator=torch.empty
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = 0
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.tensor_size
        self.bus_size = self.tensor_size

        self.calc_flops = 0

        self._run_func = self.device2host_run

    def device2host_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dst.copy_(src)
        return dst


class Device2DeviceOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        self.calc_flops = 0

        self._run_func = self.device2device_run

    def device2device_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dst.copy_(src)
        return dst
