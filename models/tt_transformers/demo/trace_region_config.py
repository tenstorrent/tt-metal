# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import re

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0

# NOTE: We need to override trace_region_size before the mesh device is opened
# NOTE: When using DP, we need to have the imlpemented logic because when we parametrize the test with a specific trace region size, all submeshes will have that trace region size
# example of the above : T3K (DP-8-b1 ; @parametrize(trace_region_size=X) -> we efectivly have 8 N150's with trace_region_size=X which could leed to OOM if X is too large)

# TODO: For now, each confest.py should call get_supported_trace_region_size if they want to override the trace region size


def get_base_model_name(model_name: str) -> str:
    # Remove the suffix after B- (case insensitive), e.g. "Llama-3.1-70B-Instruct" -> "Llama-3.1-70B"
    match = re.search(r"(.*?\d+[bB])-", model_name)
    return match.group(1) if match else model_name


def get_mesh_device_name(num_devices, mesh_device_name):
    if mesh_device_name == "P100":
        return "P100"

    arch_name = ttnn.get_arch_name()

    if num_devices == 0:
        return "CPU"

    if is_blackhole():
        dict_device_names = {
            1: "P150",  # We don't need to check for P100, because using DP would not use the P100 device config (7x1)
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }
    elif is_wormhole_b0():
        dict_device_names = {
            1: "N150",
            2: "N300",
            4: "N150x4",
            8: "T3K",
            32: "TG",
        }
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    if num_devices in dict_device_names:
        return dict_device_names[num_devices]
    else:
        logger.info(f"Unsupported number of devices: {num_devices} for {arch_name}")
        return None


def base_model_name_from_env():
    HF_MODEL = os.getenv("HF_MODEL")

    if HF_MODEL:
        model_name = HF_MODEL.strip("/").split("/")[-1]
    else:
        return None

    return get_base_model_name(model_name)


def device_name_based_on_data_parallel(request, num_devices, mesh_device_name):
    data_parallel = request.node.callspec.params.get("data_parallel", 1)
    num_devices = num_devices[0] * num_devices[1] if isinstance(num_devices, tuple) else num_devices
    num_devices_data_parallel = num_devices // data_parallel
    return get_mesh_device_name(num_devices_data_parallel, mesh_device_name)


def get_supported_trace_region_size(request, mesh_device):
    # TODO: If no specific trace region size is listed for a model and device, the default one will be used (the one set in simple_text_demo.py @parametrize)
    trace_region_size_dict = {
        "Llama-3.1-8B": {
            "N150": 25000000,
            "N300": 38000000,
            "T3K": 50000000,
            "TG": 50000000,
        },
        "Llama-3.3-70B": {
            "T3K": 80000000,
            "TG": 80000000,
            "P150": 80000000,
            "P300": 80000000,
            "P150x4": 80000000,
            "P150x8": 80000000,
        },
        "Llama-3.1-70B": {
            "T3K": 90000000,
            "TG": 90000000,
            "P150": 90000000,
            "P300": 90000000,
            "P150x4": 90000000,
            "P150x8": 90000000,
        },
        "Qwen3-32B": {
            "T3K": 90000000,
            "TG": 90000000,
            "P150": 90000000,
            "P300": 90000000,
            "P150x4": 90000000,
            "P150x8": 90000000,
        },
        "GPT-OSS-20B": {
            "T3K": 50000000,
            "TG": 50000000,
        },
        "GPT-OSS-120B": {
            "T3K": 50000000,
            "TG": 50000000,
        },
        "Qwen2.5-72B": {
            "T3K": 70000000,
            "TG": 70000000,
        },
        "gemma-3-27b": {
            "T3K": 70000000,
            "TG": 70000000,
        },
        "DeepSeek-R1-Distill-Llama-70B": {
            "P150x4": 90000000,
        },
    }

    device_name_based_on_dp = device_name_based_on_data_parallel(request, mesh_device, os.getenv("MESH_DEVICE"))
    base_model_name = base_model_name_from_env()
    return trace_region_size_dict.get(base_model_name, {}).get(device_name_based_on_dp, None)
