# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import signal
import pathlib
import logging
import argparse
import subprocess
import traceback
import psutil
import torch.multiprocessing as mp


# directory config
FILE_DIR = pathlib.Path(__file__).parent.absolute()
BYTE_MLPERF_ROOT = FILE_DIR
sys.path.insert(0, str(BYTE_MLPERF_ROOT))

from core.perf_engine import engine_run
from core.utils import logger, setup_logger


def get_numa_configs():
    num_threads = 0  # 192
    num_threads_per_core = 0  # 2
    num_cores_per_socket = 0  # 48
    num_sockets = 0  # 2
    ret = subprocess.run("lscpu", capture_output=True, text=True)
    for line in ret.stdout.splitlines():
        if line.startswith("CPU(s):"):
            num_threads = int(line.split(":")[1].strip())
        if line.startswith("Thread(s) per core:"):
            num_threads_per_core = int(line.split(":")[1].strip())
        if line.startswith("Core(s) per socket:"):
            num_cores_per_socket = int(line.split(":")[1].strip())
        if line.startswith("Socket(s):"):
            num_sockets = int(line.split(":")[1].strip())
    num_threads_per_socket = num_threads // num_sockets

    is_core_continuous = True
    is_thread_continuous = False
    for line in ret.stdout.splitlines():
        if line.startswith("NUMA node0 CPU(s):"):
            numa_config_str = line.split(":")[1].strip()

            if "-" in numa_config_str:
                is_core_continuous = True
                split_list = numa_config_str.split("-")
                start_core_id = int(split_list[0])
                end_core_id = int(split_list[-1])

                if end_core_id - start_core_id > num_threads_per_socket:
                    is_thread_continuous = False
                else:
                    is_thread_continuous = True
            else:
                is_core_continuous = False
                is_thread_continuous = True
            break

    numa_configs = []
    try:
        numa_node_num = int(
            subprocess.check_output("lscpu | grep 'NUMA node(s)' | awk -F: '{print $2}'", shell=True).decode().strip()
        )
        for i in range(numa_node_num):
            numa_cores = (
                subprocess.check_output(f"lscpu | grep 'NUMA node{i}' | awk -F: '{{print $2}}'", shell=True)
                .decode()
                .strip()
            )
            core_list = []
            for item in numa_cores.split(","):
                item_split = item.split("-")
                start = int(item_split[0])
                end = int(item_split[1]) + 1
                core_list.extend(range(start, end))
            numa_configs.append(core_list)
    except:
        for socket_id in range(num_sockets):
            core_ids = []

            # 0: 0-95
            # 1: 96-191
            if is_core_continuous and is_thread_continuous:
                core_start = socket_id * num_threads_per_socket
                core_ids.extend(range(core_start, core_start + num_threads_per_socket))
            # 0: 0-47, 96-143
            # 1: 48-95,144-191
            elif is_core_continuous and not is_thread_continuous:
                for thread_id in range(num_threads_per_core):
                    core_start = thread_id * (num_sockets * num_cores_per_socket) + socket_id * num_cores_per_socket
                    core_ids.extend(range(core_start, core_start + num_cores_per_socket))
            # 0: 0,2,4,8,...,94,...
            # 1: 1,3,5,7,...,95,...
            else:
                core_ids.extend(range(socket_id, num_threads, num_sockets))

            numa_configs.append(core_ids)

    return numa_configs


# get numa config, for example:
# 0: 0-31,64-95
# 1: 32-63,96-127
numa_configs = get_numa_configs()
avail_numa_node = [-1]
for i, numa_config in enumerate(numa_configs):
    avail_numa_node.append(i)


def parse_args():
    parser = argparse.ArgumentParser()

    # hardware
    parser.add_argument(
        "--hardware_type",
        type=str,
        default="GPU",
        help="The backend going to be evaluted, refs to backends/",
    )
    parser.add_argument(
        "--show_hardware_list",
        action="store_true",
        help="Print all hardware bytemlperf supported",
    )

    # task
    parser.add_argument(
        "--task_dir",
        type=str,
        default=str(BYTE_MLPERF_ROOT.joinpath("workloads", "basic")),
        help="The direcotry of tasks going to be evaluted, e.g., default set to workloads",
    )
    parser.add_argument(
        "--task",
        default="all",
        help="The task going to be evaluted, refs to workloads/, default use all tasks in workloads/",
    )
    parser.add_argument("--show_task_list", action="store_true", help="Print all available task names")

    # report dir
    parser.add_argument(
        "--report_dir",
        type=str,
        default=str(BYTE_MLPERF_ROOT.joinpath("reports")),
        help="Report dir, default is reports/",
    )

    parser.add_argument("--node_world_size", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--gloo_port", type=str, default="49373")
    parser.add_argument("--backend_port", type=str, default="49374")

    parser.add_argument(
        "--numa_node",
        type=int,
        choices=avail_numa_node,
        help="NUMA node id, -1 means normal run, default is None which means numa_balance.",
    )
    parser.add_argument("--device", type=str, default="all", help="Device id, seperated by comma, default is all")
    parser.add_argument("--disable_parallel", action="store_true", help="Disable parallel run for normal op.")
    parser.add_argument("--disable_profiling", action="store_true", help="Disable profiling op kernels.")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logger(args.log_level)

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["GLOO_PORT"] = args.gloo_port
    os.environ["BACKEND_PORT"] = args.backend_port

    return args


if __name__ == "__main__":
    args = parse_args()

    # get task and report dirs
    task_dir = pathlib.Path(args.task_dir).absolute()
    report_dir = pathlib.Path(args.report_dir).absolute()
    report_dir.mkdir(parents=True, exist_ok=True)

    # hardware_list
    hardware_list = []
    for file in BYTE_MLPERF_ROOT.joinpath("backends").iterdir():
        if file.is_dir() and file.stem.startswith("_") is False:
            hardware_list.append(file.stem)

    # task_list
    csv_task_list = [task_csv.stem for task_csv in task_dir.rglob("*.csv")]
    json_task_list = [task_json.stem for task_json in task_dir.rglob("*.json")]
    task_list = list(set(json_task_list) | set(csv_task_list))

    if args.show_hardware_list:
        logger.info("***************** Supported Hardware Backend *****************")
        print(hardware_list)
        exit(0)

    if args.show_task_list:
        logger.info("******************* Supported Task *******************")
        print(task_list)
        exit(0)

    # check hardware
    if args.hardware_type not in hardware_list:
        logger.error(f"Hardware {args.hardware_type} not found in {BYTE_MLPERF_ROOT.joinpath('backends')}")
        exit(1)
    logger.info(f"******************* Hardware: *****************")
    logger.info(f"{args.hardware_type}\n")

    # check task
    if args.task == "all":
        test_cases = task_list
    else:
        test_cases = []
        specified_tasks = args.task.split(",")
        for task in specified_tasks:
            if task not in task_list:
                logger.error(f"Task {task} not found in {args.task_dir}")
                continue
            test_cases.append(task)
    test_cases.sort()
    args.task = ",".join(test_cases)

    logger.info(f"******************* Tasks: *****************")
    logger.info(f"{test_cases}\n")

    # Note: Tenstorrent CCL operations (all_reduce, all_gather, reduce_scatter, all_to_all)
    # now use native ttnn CCL with MeshDevice in "single" test_mode.
    # This is a single-process model where one process manages all devices via MeshDevice.
    # No special environment configuration is needed here.

    # launch num_numa_node subprocesses, and set cpu affinity for each subprocess
    # For Tenstorrent backend, force single process since ttnn ops manage devices internally
    # and multiple processes would cause device lock contention
    if args.hardware_type == "Tenstorrent" and args.numa_node is None:
        world_size = 1
        cpu_affinity = []
        for numa_config in numa_configs:
            cpu_affinity.extend(numa_config)
        cpu_affinity = [cpu_affinity]
    elif args.numa_node is None:
        world_size = len(numa_configs)
        cpu_affinity = []
        for i in range(world_size):
            cpu_affinity.append(numa_configs[i])
    # launch one subprocess
    elif args.numa_node == -1:
        world_size = 1
        cpu_affinity = []
        for numa_config in numa_configs:
            cpu_affinity.extend(numa_config)
    # launch one subprocess and set cpu affinity
    else:
        world_size = 1
        cpu_affinity = [numa_configs[args.numa_node]]

    try:
        mp.set_start_method("spawn", force=True)
    except Exception as e:
        logger.exception(f"failed to set spawn context: {e}")
        traceback.print_exc()
        sys.exit(-1)

    # terminate core task perf process
    cur_process_id = os.getpid()
    subprocess_pids = []

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, exiting...")
        if len(subprocess_pids) > 0:
            logger.info(f"terminate subprocess: {subprocess_pids}")
            for pid in subprocess_pids:
                os.kill(pid, signal.SIGTERM)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        queue_instance = mp.Queue()
        _subprocess = mp.spawn(
            fn=engine_run, args=(world_size, queue_instance, args), nprocs=world_size, join=False, daemon=False
        )
        for process in _subprocess.processes:
            subprocess_pids.append(process.pid)
        print(f"current process id: {cur_process_id}")
        print(f"start subprocesses: {subprocess_pids}")
    except Exception as e:
        logger.error(f"Create subprocesses failed, error msg: {e}")
        traceback.print_exc()
        sys.exit(-1)

    # block subprocess and set cpu affinity
    for i in range(world_size):
        child_process = psutil.Process(subprocess_pids[i])
        child_process.cpu_affinity(cpu_affinity[i])

    # start subprocess
    for _ in range(world_size):
        queue_instance.put("start")

    # wait for subprocess
    for process in _subprocess.processes:
        process.join()
