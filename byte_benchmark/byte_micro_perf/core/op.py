import sys
import time
import torch
import copy
import pathlib
from typing import List
from collections import namedtuple
from functools import partial

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))


class BasicOp:
    def __init__(self, args_dict, backend, *args, **kwargs):
        # store args
        self.args_dict = args_dict

        # get backend instance
        self.backend = backend

        # store distributed info
        self.op_group = kwargs.get("op_group", None)
        self.group_size = kwargs.get("group_size", 1)

        # default create input and output tensors based on given tensor shapes
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        # whether run default testing process
        self._custom_run = False

        # default function call, empty run
        self._run_func = self._empty_run

        # op realization provider
        self._provider = "default"

        # tensor info
        self.input_tensor_info = None
        self.output_tensor_info = None

        # tensor size for allocate memory
        self.input_tensor_size = 0
        self.output_tensor_size = 0
        self.tensor_size = 0

        # read / write bytes for calc memory bandwidth
        self.read_bytes = 0
        self.write_bytes = 0
        self.io_bytes = 0

        # communication size through links (such as pcie or nvlink)
        self.algo_size = 0
        self.bus_size = 0

        # flops for calc flops power
        self.calc_flops = 0

        # 1. parse arguments
        # 2. prepare op and compile op if needed
        # 3. define input and output tensor info
        # 4. calculate tensor size
        # 5. calculate read / write bytes
        # 6. calculate algo / bus size
        # 7. calculate flops
        # 8. specify run function, custom run or obey standard test process
        self.prepare()

        self.is_concurrent = False

        # extra providers
        self.extra_providers = []

    def _empty_run(self):
        raise NotImplementedError("run func not implemented.")

    def _create_in_out_tensors(
        self,
        instance_num,
        create_inputs=True,
        create_outputs=True,
    ):
        all_tensor_list = []

        # create first instance
        first_tensor_mapping = {}
        if create_inputs:
            for key, value in self.input_tensor_info.items():
                first_tensor_mapping[key] = value.creator(size=value.shape, dtype=value.dtype, device=value.device)
                if value.device == "cpu":
                    first_tensor_mapping[key] = first_tensor_mapping[key].pin_memory()
        if create_outputs:
            for key, value in self.output_tensor_info.items():
                first_tensor_mapping[key] = value.creator(size=value.shape, dtype=value.dtype, device=value.device)
                if value.device == "cpu":
                    first_tensor_mapping[key] = first_tensor_mapping[key].pin_memory()
        all_tensor_list.append(first_tensor_mapping)

        # clone following instances
        for _ in range(instance_num - 1):
            tensor_mapping = {}
            for key, value in first_tensor_mapping.items():
                tensor_mapping[key] = value.clone()
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list

    def is_custom_run(self):
        return self._custom_run

    def core_run(self, *args, **kwargs):
        return self._run_func(*args, **kwargs)

    def prepare(self):
        pass

    def get_provider(self):
        return self._provider

    def create_tensors(self, instance_num):
        return self._create_tensors_func(instance_num)

    def summary(self, latency_us):
        target_dict = {}
        env_dict = {}

        if latency_us > 0:
            target_dict["latency(us)"] = round(latency_us, 3)
            if self.is_concurrent:
                target_dict["algo_size(B)"] = self.algo_size
                target_dict["bus_size(B)"] = self.bus_size
                target_dict["algo_bw(GB/s)"] = round(self.algo_size / latency_us / 1e3, 3)
                target_dict["bus_bw(GB/s)"] = round(self.bus_size / latency_us / 1e3, 3)
            else:
                target_dict["read_bytes(B)"] = self.read_bytes
                target_dict["write_bytes(B)"] = self.write_bytes
                target_dict["io_bytes(B)"] = self.io_bytes
                target_dict["mem_bw(GB/s)"] = round(self.io_bytes / latency_us / 1e3, 3)
                target_dict["calc_flops"] = self.calc_flops
                target_dict["calc_flops_power(tflops)"] = round(self.calc_flops / latency_us / 1e6, 3)
                target_dict["calc_mem_ratio"] = round(self.calc_flops / self.io_bytes, 3) if self.io_bytes != 0 else 0

            env_dict.update(self.backend.env_dict)
            if len(self.extra_providers) > 0:
                for extra_provider in self.extra_providers:
                    if extra_provider in self.backend.avail_providers:
                        env_dict.update(self.backend.avail_providers[extra_provider])
        return target_dict, env_dict

    def merge_summary(self, target_dict_list):
        if len(target_dict_list) == 0:
            return {}
        if not self.is_concurrent:
            return target_dict_list[0]

        latency_list = [target_dict["latency(us)"] for target_dict in target_dict_list]
        algo_bw_list = [target_dict["algo_bw(GB/s)"] for target_dict in target_dict_list]
        bus_bw_list = [target_dict["bus_bw(GB/s)"] for target_dict in target_dict_list]

        target_dict = copy.deepcopy(target_dict_list[0])
        target_dict["latency(us)"] = min(latency_list)

        target_dict["algo_bw(GB/s)"] = max(algo_bw_list)
        target_dict["bus_bw(GB/s)"] = max(bus_bw_list)
        target_dict["algo_bw_sum(GB/s)"] = round(sum(algo_bw_list), 3)
        target_dict["bus_bw_sum(GB/s)"] = round(sum(bus_bw_list), 3)
        target_dict["latency_list(us)"] = latency_list
        target_dict["algo_bw_list(GB/s)"] = algo_bw_list
        target_dict["bus_bw_list(GB/s)"] = bus_bw_list

        return target_dict
