import os
import sys
import math
import time
import random
import torch
import pathlib
import traceback
from datetime import timedelta
from abc import ABC, abstractmethod
from typing import List

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger


class Backend(ABC):
    def __init__(self):
        self.numa_world_size = 1
        self.numa_rank = 0

    """
    device management related
    """

    @abstractmethod
    def get_torch_device_name(self):
        raise NotImplementedError

    @abstractmethod
    def get_device_name(self, index: int = 0):
        raise NotImplementedError

    @abstractmethod
    def get_device_properties(self, index: int = 0):
        raise NotImplementedError

    @abstractmethod
    def get_mem_info(self, index: int = 0):
        raise NotImplementedError

    @abstractmethod
    def get_device_count(self):
        raise NotImplementedError

    @abstractmethod
    def set_device(self, index: int):
        raise NotImplementedError

    @abstractmethod
    def get_device(self):
        raise NotImplementedError

    @abstractmethod
    def device_synchronize(self):
        raise NotImplementedError

    @abstractmethod
    def empty_cache(self):
        raise NotImplementedError

    @abstractmethod
    def get_backend_env(self):
        raise NotImplementedError

    """
    ccl related
    """

    @abstractmethod
    def get_dist_module(self):
        raise NotImplementedError

    @abstractmethod
    def get_dist_backend(self):
        raise NotImplementedError

    def initialize_ccl(self, rank: int, world_size: int):
        os.environ["MASTER_PORT"] = os.environ["BACKEND_PORT"]
        dist_module = self.get_dist_module()
        dist_backend_name = self.get_dist_backend()

        dist_module.init_process_group(
            backend=dist_backend_name, world_size=world_size, rank=rank, timeout=timedelta(seconds=1800)
        )

        assigned_value = 1 if rank < world_size // 2 else -1
        data = torch.ones([1], dtype=torch.float32, device=self.get_torch_device_name()) * assigned_value
        dist_module.all_reduce(data, op=dist_module.ReduceOp.SUM)
        print(data)

        return True

    def new_group(self, ranks):
        dist_module = self.get_dist_module()

        if dist_module.is_initialized():
            return dist_module.new_group(ranks)
        else:
            return None

    def op_group_barrier(self, op_group=None, group_size=1):
        dist_module = self.get_dist_module()
        if dist_module.is_initialized() and group_size > 1:
            dist_module.all_reduce(
                torch.tensor([1], dtype=torch.int32, device=self.get_torch_device_name()),
                op=dist_module.ReduceOp.SUM,
                group=op_group,
            )

    def destroy_process_group(self):
        dist_module = self.get_dist_module()
        if dist_module.is_initialized():
            dist_module.destroy_process_group()

    def core_perf(self, op_instance, warmup_iterations, prefer_iterations, tensor_list, profiling=True):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        # nessary sync for host2device, device2host test
        self.op_group_barrier(op_group=op_group, group_size=group_size)
        self.device_synchronize()

        for i in range(warmup_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
        self.device_synchronize()
        start_time = time.perf_counter_ns()
        for i in range(prefer_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
        self.device_synchronize()
        end_time = time.perf_counter_ns()

        latency_us = (end_time - start_time) / 1e3 / prefer_iterations
        return latency_us, []

    def perf(self, op_instance, profiling=True):
        # op
        tensor_size = op_instance.tensor_size

        # device
        device_mem_info = self.get_mem_info()
        avail_memory = device_mem_info[0]

        # assume
        assume_avail_bytes = int(avail_memory * 0.9)
        assume_cache_size = 1 * (1024**3)

        # preset return values
        latency_us = 0.0

        try:
            min_test_iters = 32 if not op_instance.is_concurrent else 10
            sleep_time = 0.2
            max_test_time = 1e6
            max_data_cnt = 1
            if not op_instance.is_concurrent:
                if tensor_size > assume_avail_bytes:
                    raise RuntimeError("Not enough memory to run the op")
                elif 2 * tensor_size > assume_avail_bytes:
                    max_data_cnt = 1
                elif tensor_size > assume_cache_size:
                    max_data_cnt = 2
                else:
                    max_data_cnt = min(
                        math.floor(max(assume_avail_bytes, assume_cache_size) / tensor_size),
                        math.floor(assume_cache_size / tensor_size),
                    )

            tensor_list = op_instance.create_tensors(max_data_cnt)
            random.shuffle(tensor_list)

            latency_us, _ = self.core_perf(op_instance, 2, 2, tensor_list, profiling=False)
            prefer_iters = min(max(int(max_test_time / latency_us), 2), min_test_iters)
            if op_instance.group_size > 1:
                dist_module = self.get_dist_module()
                prefer_iters_list = [None for _ in range(op_instance.group_size)]
                dist_module.all_gather_object(prefer_iters_list, prefer_iters, group=op_instance.op_group)
                prefer_iters = max(prefer_iters_list)
            time.sleep(sleep_time)
            latency_us, kernel_list = self.core_perf(op_instance, 2, prefer_iters, tensor_list, profiling=profiling)

            del tensor_list
            self.empty_cache()
        except Exception as e:
            traceback.print_exc()

        return op_instance.summary(latency_us)

    def fake_perf(self, group_size, op_group):
        if group_size > 1:
            dist_module = self.get_dist_module()

            self.op_group_barrier(op_group=op_group, group_size=group_size)

            prefer_iters_list = [None for _ in range(group_size)]
            dist_module.all_gather_object(prefer_iters_list, 0, group=op_group)

            self.op_group_barrier(op_group=op_group, group_size=group_size)
