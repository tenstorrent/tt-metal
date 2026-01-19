import os
import sys
import json
import time
import copy
import signal
import pathlib
import traceback
import prettytable
from itertools import combinations

from typing import List, Dict, Any
import torch.distributed as dist
import torch.multiprocessing as mp

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.creators import create_backend_instance, get_op_info, create_op_instance


class Scheduler:
    def __init__(self, args):
        self._subprocesses = []

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, exiting...")
            self.__clean_subprocess()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # store args
        self.args_device = args.device
        self.backend_type = args.hardware_type
        self.disable_parallel = args.disable_parallel
        self.profiling = not args.disable_profiling
        self.report_dir = args.report_dir

        if self.profiling and self.backend_type == "Tenstorrent":
            os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
            os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
            os.environ["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "1"
            os.environ["TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES"] = "1"

        # create backend instance
        self.backend = create_backend_instance(self.backend_type)

        self.backend.node_world_size = args.node_world_size
        self.backend.node_rank = args.node_rank

        self.backend.numa_world_size = args.numa_world_size
        self.backend.numa_rank = args.numa_rank

        self.backend.all_process_size = args.all_process_size
        self.backend.all_process_rank = args.all_process_rank

        # for example, "NVIDIA A800-SXM4-80GB"
        self.backend.device_name = self.backend.get_device_name()

        # for example, "cuda"
        self.backend.torch_device_name = self.backend.get_torch_device_name()

        # get device info
        if self.args_device == "all":
            self.ori_target_devices = self.backend.avail_devices
        else:
            try:
                self.ori_target_devices = []
                for d in self.args_device.split(","):
                    if d.isdigit() and int(d) < self.backend.device_count:
                        self.ori_target_devices.append(int(d))
            except Exception as e:
                raise RuntimeError(f"invalid device config: {self.args_device}, error msg: {e}")
        self.ori_target_device_count = len(self.ori_target_devices)
        if self.ori_target_device_count == 0:
            raise RuntimeError("no valid device")

        # test cases
        self.test_cases = []

    def prepare_task(self, task, test_cases):
        # store task_name and test_cases
        self.op_name = task

        """
        test_mode:
        - single:           test one case on one card
        - concurrent:       test all cases on multiple cards simultaneously
        - concurrent_p2p:   test all cases on multiple cards, but iterate on each device pair
        """
        # get test mode and op_providers
        self.test_mode, self.op_cls_mapping = get_op_info(self.backend_type, task)
        if not self.op_cls_mapping:
            return False

        # for each node, each numa process, provide:
        # 1. ori_target_devices: [0, 1, 2, 3, 4, 5, 6, 7]
        # 2. total_target_devices: [0, 1, 2 ,3, 4, 5, 6, 7]
        # 3. target_devices: [0, 1, 2, 3] or [4, 5, 6, 7]
        # 4. device_num_per_numa: 4
        # 5. all_node_devices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7] for 2 nodes
        # one node, one process, specified devices
        if self.test_mode == "single" or self.test_mode == "mesh_single":
            if self.backend.numa_rank == 0:
                self.backend.ori_target_devices = self.ori_target_devices
                self.backend.target_devices = (
                    self.ori_target_devices if not self.disable_parallel else self.ori_target_devices[0:1]
                )
                self.backend.total_target_devices = (
                    self.ori_target_devices if not self.disable_parallel else self.ori_target_devices[0:1]
                )
                self.backend.device_num_per_numa = self.ori_target_device_count if not self.disable_parallel else 1
                self.backend.all_node_devices = self.ori_target_devices
            else:
                return False
        # multiple nodes, multiple processes, specified devices on each
        elif self.test_mode == "concurrent" or self.test_mode == "concurrent_p2p":
            target_devices = []
            total_target_devices = []
            device_num_per_numa = self.ori_target_device_count // self.backend.numa_world_size

            if device_num_per_numa == 0:
                if self.backend.numa_rank < self.ori_target_device_count:
                    target_devices = self.ori_target_devices[self.backend.numa_rank : self.backend.numa_rank + 1]
                total_target_devices = self.ori_target_devices
            else:
                target_devices = self.ori_target_devices[
                    self.backend.numa_rank * device_num_per_numa : (self.backend.numa_rank + 1) * device_num_per_numa
                ]
                total_target_devices = self.ori_target_devices[0 : self.backend.numa_world_size * device_num_per_numa]

            if len(target_devices) == 0:
                return False

            all_node_devices = []
            if self.backend.all_process_size > 1:
                # gather all nodes devices
                temp_all_node_devices = [None for _ in range(self.backend.all_process_size)]
                dist.all_gather_object(temp_all_node_devices, target_devices)

                device_num_per_process = [len(d) for d in temp_all_node_devices]
                if len(set(device_num_per_process)) > 1:
                    raise RuntimeError("all process device num per numa must be same")

                for process_target_devices in temp_all_node_devices:
                    all_node_devices.extend(process_target_devices)
            else:
                all_node_devices = total_target_devices

            self.backend.ori_target_devices = self.ori_target_devices
            self.backend.total_target_devices = total_target_devices
            self.backend.target_devices = target_devices
            self.backend.device_num_per_numa = device_num_per_numa
            self.backend.all_node_devices = all_node_devices

        if self.test_mode == "single" or self.test_mode == "mesh_single":
            self.test_cases = test_cases

        # if concurrent or concurrent_p2p
        # check world_size
        elif self.test_mode == "concurrent":
            avail_world_size = len(self.backend.all_node_devices)
            self.test_cases = []

            for case in test_cases:
                if "world_size" not in case:
                    continue

                if case["world_size"] > avail_world_size:
                    continue

                self.test_cases.append(case)

        elif self.test_mode == "concurrent_p2p":
            avail_world_size = len(self.backend.total_target_devices)
            self.test_cases = []

            for case in test_cases:
                new_case_template = copy.deepcopy(case)
                new_case_template["world_size"] = avail_world_size
                new_case_template["src_device"] = 0
                new_case_template["dst_device"] = 1

                pairs = list(combinations(self.backend.total_target_devices, 2))
                for i in range(avail_world_size):
                    pairs.append((i, i))

                for device_pair in pairs:
                    new_case = copy.deepcopy(new_case_template)
                    new_case["src_device"] = device_pair[0]
                    new_case["dst_device"] = device_pair[1]
                    self.test_cases.append(new_case)

        # print readable config
        pt = prettytable.PrettyTable()
        pt.align = "l"
        pt.field_names = ["config", "value"]
        pt.add_row(["backend", self.backend_type])
        pt.add_row(["op_name", self.op_name])
        pt.add_row(["test_mode", self.test_mode])
        pt.add_row(["op_cls_mapping", self.op_cls_mapping])

        pt.add_row(["node_rank", f"{self.backend.node_rank} / {self.backend.node_world_size}"])
        pt.add_row(["numa_rank", f"{self.backend.numa_rank} / {self.backend.numa_world_size}"])
        pt.add_row(["process_rank", f"{self.backend.all_process_rank} / {self.backend.all_process_size}"])
        pt.add_row(["avail_devices", self.backend.avail_devices])
        pt.add_row(["ori_target_devices", self.backend.ori_target_devices])
        pt.add_row(["total_target_devices", self.backend.total_target_devices])
        pt.add_row(["target_devices", self.backend.target_devices])
        pt.add_row(["all_node_devices", self.backend.all_node_devices])
        pt.add_row(["device_num_per_numa", self.backend.device_num_per_numa])
        pt.add_row(["test_cases", len(self.test_cases)])

        pt.add_row(["", ""])
        for key, value in self.backend.env_dict.items():
            pt.add_row([key, value])
        print(pt)

        return True

    def __del__(self):
        self.__clean_subprocess()

    def __clean_subprocess(self):
        for process in self._subprocesses:
            if process.is_alive():
                pid = process.pid
                if pid is not None:
                    os.kill(pid, signal.SIGTERM)
        self._subprocesses.clear()

    def run(self):
        self.__clean_subprocess()

        # For mesh_single mode, use only 1 instance (single process manages all devices via MeshDevice)
        # This is needed for Tenstorrent CCL ops that use ttnn MeshDevice
        # For Tenstorrent backend in single mode, also use 1 instance because ttnn ops
        # internally manage device 0 and would cause lock contention with multiple processes
        if self.test_mode == "mesh_single":
            instance_num = 1
        elif self.test_mode == "single" and self.backend_type == "Tenstorrent":
            instance_num = 1
        else:
            instance_num = len(self.backend.target_devices)
        print("Instance num for subprocesses:", instance_num)
        input_queues = mp.Queue()
        output_queues = mp.Queue()
        try:
            _subprocess = mp.spawn(
                fn=self.subprocess_func,
                args=(input_queues, output_queues),
                nprocs=instance_num,
                join=False,
                daemon=False,
            )
        except Exception as e:
            logger.error(f"Create subprocesses failed, error msg: {e}")
            return []

        self._subprocesses = _subprocess.processes
        for _ in range(instance_num):
            assert "ready" == output_queues.get()
        logger.info("all ranks are ready and listening, init done")

        result_list = []

        if self.test_mode == "single" or self.test_mode == "mesh_single":
            """
            single mode:
            each node only have one numa process, each dispatch tasks
            all device process will receive tasks alternately

            mesh_single mode (for Tenstorrent CCL):
            Same as single but only 1 subprocess manages all devices via MeshDevice
            """
            if self.backend.numa_rank == 0:
                valid_case_set = set()
                for index, test_case in enumerate(self.test_cases):
                    test_case["index"] = index
                    valid_case_set.add(index)
                    input_queues.put(test_case, False)
                for _ in range(instance_num):
                    input_queues.put(None, False)

                while len(valid_case_set) > 0:
                    result_dict = output_queues.get()
                    result_list.extend(result_dict["results"])
                    valid_case_set.remove(result_dict["index"])

                result_list = sorted(result_list, key=lambda x: x["arguments"]["index"])
                for result in result_list:
                    result["sku_name"] = self.backend.get_device_name()
                    result["op_name"] = self.op_name
                    if "arguments" in result and "index" in result["arguments"]:
                        result["arguments"].pop("index")

        elif self.test_mode == "concurrent" or self.test_mode == "concurrent_p2p":
            """
            concurrent mode:
            multiple nodes will have multiple numa process
            only the first numa node dispatch task
            only the first device process receive task and broadcast to other devicce proccesses
            """
            if self.backend.all_process_rank == 0:
                valid_case_set = set()
                for index, test_case in enumerate(self.test_cases):
                    test_case["index"] = index
                    valid_case_set.add(index)
                    input_queues.put(test_case, False)
                input_queues.put(None, False)

                while len(valid_case_set) > 0:
                    result_dict = output_queues.get()
                    result_list.extend(result_dict["results"])
                    valid_case_set.remove(result_dict["index"])

                result_list = sorted(result_list, key=lambda x: x["arguments"]["index"])
                for result in result_list:
                    result["sku_name"] = self.backend.get_device_name()
                    result["op_name"] = self.op_name
                    if "arguments" in result and "index" in result["arguments"]:
                        result["arguments"].pop("index")

        for process in self._subprocesses:
            process.join()

        return result_list

    def subprocess_func(self, instance_rank: int, *args):
        try:
            input_queues, output_queues = args
            backend = self.backend

            # computation ops
            if self.test_mode == "single":
                # world_size: 8
                # index: [0, 1, 2, 3, 4, 5, 6, 7]
                # device_id: [0, 1, 2, 3, 4, 5, 6, 7]
                true_world_size = len(backend.target_devices)
                true_rank = backend.numa_rank * backend.device_num_per_numa + instance_rank
                true_device_index = backend.target_devices[true_rank]
                print(
                    f"true_world_size: {true_world_size}, true_rank: {true_rank}, true_device_index: {true_device_index}"
                )
                backend.set_device(true_device_index)

                # device process is ready
                output_queues.put("ready")

                # loop function
                while True:
                    # get test case and check exit
                    test_case = input_queues.get()
                    if test_case is None:
                        break

                    # results for multiple providers
                    provider_results = []
                    for op_provider, op_cls in self.op_cls_mapping.items():
                        # try create op instance on current device
                        op_instance = None
                        try:
                            op_instance = op_cls(test_case, backend)
                            op_instance.is_concurrent = False
                        except Exception as e:
                            print(f"op {self.op_name} provider {op_provider} init failed, error msg: {e}")
                            continue

                        # try bench op and get result
                        target_dict = {}
                        env_dict = {}
                        try:
                            target_dict, env_dict = backend.perf(op_instance, profiling=self.profiling)
                        except:
                            pass

                        if target_dict:
                            arguments_str = json.dumps(test_case)
                            targets_str = json.dumps(target_dict, indent=4)
                            print(f"{self.op_name}\t{op_provider}\n{arguments_str}\n{targets_str}\n")

                            template_dict = {
                                "provider": op_provider,
                                "arguments": copy.deepcopy(test_case),
                                "targets": target_dict,
                                "env": env_dict,
                            }
                            provider_results.append(template_dict)

                    output_queues.put({"index": test_case["index"], "results": provider_results}, block=False)

            # Tenstorrent CCL ops using MeshDevice - single process manages all devices
            elif self.test_mode == "mesh_single":
                # In mesh_single mode, we have only 1 subprocess (instance_rank=0)
                # This single process manages all devices via ttnn MeshDevice
                # No backend.set_device() needed - the op will create MeshDevice internally
                print(f"mesh_single mode: single process managing {len(backend.target_devices)} devices via MeshDevice")

                # device process is ready
                output_queues.put("ready")

                # loop function
                while True:
                    # get test case and check exit
                    test_case = input_queues.get()
                    if test_case is None:
                        break

                    # results for multiple providers
                    provider_results = []
                    for op_provider, op_cls in self.op_cls_mapping.items():
                        # try create op instance - MeshDevice will be created inside
                        op_instance = None
                        try:
                            op_instance = op_cls(test_case, backend)
                            op_instance.is_concurrent = False
                        except Exception as e:
                            print(f"op {self.op_name} provider {op_provider} init failed, error msg: {e}")
                            traceback.print_exc()
                            continue

                        # try bench op and get result
                        target_dict = {}
                        env_dict = {}
                        try:
                            target_dict, env_dict = backend.perf(op_instance, profiling=self.profiling)
                        except Exception as e:
                            print(f"op {self.op_name} provider {op_provider} perf failed, error msg: {e}")
                            traceback.print_exc()

                        if target_dict:
                            arguments_str = json.dumps(test_case)
                            targets_str = json.dumps(target_dict, indent=4)
                            print(f"{self.op_name}\t{op_provider}\n{arguments_str}\n{targets_str}\n")

                            template_dict = {
                                "provider": op_provider,
                                "arguments": copy.deepcopy(test_case),
                                "targets": target_dict,
                                "env": env_dict,
                            }
                            provider_results.append(template_dict)

                    output_queues.put({"index": test_case["index"], "results": provider_results}, block=False)

            # communication ops
            elif self.test_mode == "concurrent":
                """
                assume that:
                ---
                2 nodes, 4 numas, 16 devices
                all_node_devices: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
                true_rank: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                all_node_devices + true_rank --> true_device_index
                """
                true_world_size = len(backend.all_node_devices)
                true_rank = backend.all_process_rank * backend.device_num_per_numa + instance_rank
                true_device_index = backend.all_node_devices[true_rank]
                device_world_size = backend.numa_world_size * backend.device_num_per_numa
                device_true_rank = backend.numa_rank * backend.device_num_per_numa + instance_rank
                print(
                    f"true_world_size: {true_world_size}, true_rank: {true_rank}, true_device_index: {true_device_index}"
                )
                backend.set_device(true_device_index)

                # init dist env on all nodes
                dist_module = backend.get_dist_module()
                backend.initialize_ccl(true_rank, true_world_size)

                # device process is ready
                output_queues.put("ready")

                # store process group for each group_size config
                process_groups_mapping = {true_world_size: None}

                # loop function
                while True:
                    if true_rank == 0:
                        test_case = input_queues.get()
                    else:
                        test_case = None

                    exchange_area = [None for _ in range(true_world_size)]
                    if true_world_size > 1:
                        dist_module.all_gather_object(exchange_area, {"rank": true_rank, "result": test_case})
                    sorted_exchange_area = sorted(exchange_area, key=lambda x: x["rank"])

                    test_case = sorted_exchange_area[0]["result"]
                    if test_case is None:
                        break

                    provider_results = []
                    for op_provider, op_cls in self.op_cls_mapping.items():
                        # create corresponding process group
                        world_size = test_case.get("world_size", 1)
                        if world_size > 1 and world_size not in process_groups_mapping:
                            process_groups_mapping[world_size] = backend.new_group(range(world_size))

                        # create op instance on all target devices
                        op_instance = None
                        if true_rank < world_size:
                            try:
                                op_instance = op_cls(
                                    test_case,
                                    backend,
                                    op_group=process_groups_mapping.get(world_size, None),
                                    group_size=world_size,
                                )
                                op_instance.is_concurrent = True
                            except:
                                traceback.print_exc()

                        # create current exchange area for all cooperative devices
                        # maybe on different node or numa
                        # check whether needed devices have created op instance
                        exchange_area = [None for _ in range(true_world_size)]
                        if true_world_size > 1:
                            dist_module.all_gather_object(
                                exchange_area,
                                {
                                    "rank": true_rank,
                                    "result": op_instance is not None,
                                },
                            )
                        sorted_exchange_area = sorted(exchange_area, key=lambda x: x["rank"])
                        if not all([item["result"] for item in sorted_exchange_area[:world_size]]):
                            continue

                        # according to given world_size,
                        # some devices work, some devices sleep
                        target_dict = {}
                        env_dict = {}
                        if true_rank < world_size:
                            try:
                                target_dict, env_dict = backend.perf(op_instance, profiling=self.profiling)
                            except:
                                pass

                        # sync on all cooperative devices
                        exchange_area = [None for _ in range(true_world_size)]
                        if true_world_size > 1:
                            dist_module.all_gather_object(
                                exchange_area,
                                {
                                    "rank": true_rank,
                                    "target_dict": target_dict,
                                    "env_dict": env_dict,
                                },
                            )
                        sorted_exchange_area = sorted(exchange_area, key=lambda x: x["rank"])
                        target_dict_list = [item["target_dict"] for item in sorted_exchange_area]
                        env_dict_list = [item["env_dict"] for item in sorted_exchange_area]
                        if not all(target_dict_list[:world_size]):
                            continue

                        if backend.node_rank == 0 and backend.numa_rank == 0 and instance_rank == 0:
                            new_target_dict = op_instance.merge_summary(target_dict_list[:world_size])
                            new_env_dict = env_dict_list[0]
                            arguments_str = json.dumps(test_case)
                            targets_str = json.dumps(new_target_dict, indent=4)
                            print(
                                f"{self.op_name}\t{op_provider}\ndevice {backend.total_target_devices[:world_size]}\n{arguments_str}\n{targets_str}\n"
                            )

                            template_dict = {
                                "provider": op_provider,
                                "arguments": copy.deepcopy(test_case),
                                "targets": new_target_dict,
                                "env": new_env_dict,
                            }
                            provider_results.append(template_dict)

                    if backend.numa_rank == 0 and instance_rank == 0:
                        output_queues.put(
                            {
                                "index": test_case["index"],
                                "results": provider_results,
                            },
                            block=False,
                        )
                backend.destroy_process_group()

            elif self.test_mode == "concurrent_p2p":
                """
                assume that:
                ---
                2 nodes, 4 numas, 16 devices
                all_node_devices: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
                true_rank: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                all_node_devices + true_rank --> true_device_index
                """

                # world_size: 16
                # index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                # device_id: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
                true_world_size = len(backend.all_node_devices)
                true_rank = backend.all_process_rank * backend.device_num_per_numa + instance_rank
                true_device_index = backend.all_node_devices[true_rank]
                device_world_size = backend.numa_world_size * backend.device_num_per_numa
                device_true_rank = backend.numa_rank * backend.device_num_per_numa + instance_rank
                print(
                    f"true_world_size: {true_world_size}, true_rank: {true_rank}, true_device_index: {true_device_index}"
                )
                backend.set_device(true_device_index)

                # init dist env on each node
                dist_module = backend.get_dist_module()
                backend.initialize_ccl(true_rank, true_world_size)

                # device process is ready
                output_queues.put("ready")

                # loop function
                while True:
                    if true_rank == 0:
                        test_case = input_queues.get()
                    else:
                        test_case = None

                    exchange_area = [None for _ in range(true_world_size)]
                    if true_world_size > 1:
                        dist_module.all_gather_object(exchange_area, {"rank": true_rank, "result": test_case})
                    sorted_exchange_area = sorted(exchange_area, key=lambda x: x["rank"])

                    test_case = sorted_exchange_area[0]["result"]
                    if test_case is None:
                        break

                    src_device = test_case["src_device"]
                    dst_device = test_case["dst_device"]

                    target_device_num = 2 if (src_device != dst_device) else 1

                    provider_results = []
                    for op_provider, op_cls in self.op_cls_mapping.items():
                        # create op instance on all target devices
                        op_instance = None
                        if true_rank in [src_device, dst_device]:
                            try:
                                op_instance = op_cls(test_case, backend, op_group=None, group_size=true_world_size)
                                op_instance.is_concurrent = True
                            except:
                                pass

                        # create current exchange area for all cooperative devices
                        # maybe on different node or numa
                        exchange_area = [None for _ in range(true_world_size)]
                        if true_world_size > 1:
                            dist_module.all_gather_object(
                                exchange_area,
                                {
                                    "rank": true_rank,
                                    "result": op_instance is not None,
                                },
                            )
                        sorted_exchange_area = sorted(exchange_area, key=lambda x: x["rank"])
                        if sum([1 if item["result"] else 0 for item in sorted_exchange_area]) != target_device_num:
                            continue

                        target_dict = {}
                        env_dict = {}
                        if op_instance is not None:
                            try:
                                target_dict, env_dict = backend.perf(op_instance, profiling=self.profiling)
                            except:
                                pass
                        else:
                            backend.fake_perf(group_size=true_world_size, op_group=None)

                        # sync on all cooperative devices
                        exchange_area = [None for _ in range(true_world_size)]
                        if true_world_size > 1:
                            dist_module.all_gather_object(
                                exchange_area, {"rank": true_rank, "target_dict": target_dict, "env_dict": env_dict}
                            )
                        sorted_exchange_area = sorted(exchange_area, key=lambda x: x["rank"])
                        target_dict_list = [item["target_dict"] for item in sorted_exchange_area]
                        env_dict_list = [item["env_dict"] for item in sorted_exchange_area]
                        if sum([1 if item else 0 for item in target_dict_list]) != target_device_num:
                            continue

                        if backend.node_rank == 0 and backend.numa_rank == 0 and instance_rank == 0:
                            new_target_dict = target_dict_list[dst_device]
                            new_env_dict = env_dict_list[dst_device]
                            arguments_str = json.dumps(test_case)
                            targets_str = json.dumps(new_target_dict, indent=4)
                            print(
                                f"{self.op_name}\t{op_provider}\nsrc: {src_device}, dst: {dst_device}\n{arguments_str}\n{targets_str}\n"
                            )

                            if new_target_dict:
                                template_dict = {
                                    "provider": op_provider,
                                    "arguments": copy.deepcopy(test_case),
                                    "targets": new_target_dict,
                                    "env": new_env_dict,
                                }
                                provider_results.append(template_dict)

                    if backend.numa_rank == 0 and instance_rank == 0:
                        output_queues.put(
                            {
                                "index": test_case["index"],
                                "results": provider_results,
                            },
                            block=False,
                        )
                backend.destroy_process_group()

        except Exception as e:
            traceback.print_exc()
            sys.exit(1)
