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
import csv
import json
import argparse
import pathlib
import itertools
import jsonlines
import shutil
import psutil
import prettytable
from datetime import timedelta

from typing import Any, Dict, List
import itertools
from collections import namedtuple

import traceback
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger, setup_logger
from core.scheduler import Scheduler


def parse_task(task_dir, task):
    task_dir = pathlib.Path(task_dir).absolute()
    target_task_files = task_dir.rglob(task + ".json")
    task_cases = []
    for task_file in target_task_files:
        with open(task_file, "r") as f:
            json_data = json.load(f)
        json_data_list = json_data.get("cases", [])
        if json_data_list:
            for argument_case in json_data_list:
                removed_keys = []
                for key in argument_case:
                    if key.startswith("__"):
                        removed_keys.append(key)
                for removed_key in removed_keys:
                    argument_case.pop(removed_key)

                keys = list(argument_case.keys())
                values = list(argument_case.values())
                for i, value in enumerate(values):
                    if isinstance(value, str):
                        values[i] = [value]
                total_cases = list(itertools.product(*values))
                for case in total_cases:
                    case_dict = dict(zip(keys, case))

                    added_key_value = {}
                    removed_key = []
                    for key in case_dict:
                        if "." in key:
                            split_keys = key.split(".")
                            for i, split_key in enumerate(split_keys):
                                added_key_value[split_key] = case_dict[key][i]
                            removed_key.append(key)
                    for key in removed_key:
                        del case_dict[key]
                    case_dict.update(added_key_value)
                    task_cases.append(case_dict)
    return task_cases


def engine_run(rank, *args):
    world_size, queue_instance, configs = args
    _ = queue_instance.get()

    # assign node
    node_world_size = configs.node_world_size
    node_rank = configs.node_rank

    # assign numa
    numa_world_size = world_size
    numa_rank = rank
    configs.numa_world_size = numa_world_size
    configs.numa_rank = numa_rank

    # assign all process
    all_process_size = node_world_size * numa_world_size
    all_process_rank = node_rank * numa_world_size + numa_rank
    configs.all_process_size = all_process_size
    configs.all_process_rank = all_process_rank

    # try init process dist
    os.environ["MASTER_PORT"] = os.environ["GLOO_PORT"]
    try:
        dist.init_process_group(
            backend="gloo", world_size=all_process_size, rank=all_process_rank, timeout=timedelta(seconds=1800)
        )
        data = torch.ones(1)
        dist.all_reduce(data)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

    # try create scheduler
    try:
        scheduler = Scheduler(configs)
    except Exception as e:
        traceback.print_exc()
        sys.exit(-1)

    for i, task in enumerate(configs.task.split(",")):
        # sync for all processes
        dist.barrier()
        if numa_rank == 0:
            print("")
            print("=" * 100)
            print(f"{i+1}: {task}")
            print("=" * 100)

        try:
            # parse task simutaneously
            task_cases = parse_task(configs.task_dir, task)
            if len(task_cases) == 0:
                print(f"{task} has no task case")
                continue

            # prepare task, exit if needed
            result_list = []
            prepare_ret = scheduler.prepare_task(task, task_cases)
            if prepare_ret:
                result_list = scheduler.run()

            if configs.numa_rank == 0:
                if len(result_list) == 0:
                    logger.error("No result available")
                    continue

                sku_name = result_list[0]["sku_name"]
                report_dir = pathlib.Path(configs.report_dir).absolute()
                backend_dir = report_dir.joinpath(configs.hardware_type)
                sku_dir = backend_dir.joinpath(sku_name)
                op_dir = sku_dir.joinpath(task)
                op_dir.mkdir(parents=True, exist_ok=True)

                # grouped by arg_type
                result_mapping = {}
                for result in result_list:
                    # check arguments and targets
                    arguments = result.get("arguments", {})
                    targets = result.get("targets", {})
                    if arguments == {} or targets == {}:
                        continue

                    arg_type = arguments.get("arg_type", "default")
                    if arg_type not in result_mapping:
                        result_mapping[arg_type] = {}

                    provider = result.get("provider", "torch")
                    world_size = arguments.get("world_size", 1)
                    dtype = arguments["dtype"]

                    key = (provider, world_size, dtype)
                    if key not in result_mapping[arg_type]:
                        result_mapping[arg_type][key] = []
                    result_mapping[arg_type][key].append(result)

                for arg_type in result_mapping:
                    target_folder = op_dir.joinpath(arg_type)
                    if target_folder.exists():
                        shutil.rmtree(target_folder)
                    target_folder.mkdir(parents=True, exist_ok=True)

                    for key in result_mapping[arg_type]:
                        provider, world_size, dtype = key

                        if world_size == 1:
                            file_name = f"{provider}-{dtype}"
                        else:
                            file_name = f"{provider}-group{world_size}-{dtype}"

                        result_list = result_mapping[arg_type][key]
                        jsonl_file_path = target_folder.joinpath(file_name + ".jsonl")
                        with jsonlines.open(jsonl_file_path, "w") as writer:
                            for result in result_list:
                                writer.write(result)

                        keys = ["task_name"]
                        keys.extend(list(result_list[0]["arguments"].keys()))
                        keys.extend(list(result_list[0]["targets"].keys()))

                        csv_file_path = target_folder.joinpath(file_name + ".csv")
                        with open(csv_file_path, "w") as f:
                            writer = csv.writer(f)
                            writer.writerow(keys)
                            for result in result_list:
                                row = [task]
                                row.extend(list(result["arguments"].values()))
                                row.extend(list(result["targets"].values()))
                                writer.writerow(row)

        except Exception as e:
            traceback.print_exc()
            continue

    dist.destroy_process_group()
