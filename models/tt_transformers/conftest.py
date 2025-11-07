# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pytest

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.tt_transformers.tt.common import get_base_model_name


def get_mesh_device_name(num_devices, mesh_device_name):
    if mesh_device_name == "P100":
        return "P100"

    arch_name = ttnn.get_arch_name()

    if num_devices == 0:
        return "CPU"

    if is_blackhole():
        dict_device_names = {
            1: "P150",  # We don't need to check for P150, because using DP would not use the P100 device config (7x1)
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
        raise ValueError(f"Unsupported number of devices: {num_devices} for {arch_name}")


def get_data_parallel_device_name(request, num_devices, mesh_device_name):
    data_parallel = request.node.callspec.params.get("data_parallel", 1)
    num_devices = num_devices[0] * num_devices[1] if isinstance(num_devices, tuple) else num_devices
    num_devices_data_parallel = num_devices // data_parallel
    return get_mesh_device_name(num_devices_data_parallel, os.getenv("MESH_DEVICE"))


def get_supported_trace_region_size(device_name):
    # TODO: If no specific trace region size is listed for a model and device, the default one will be used (the one set in simple_text_demo.py)
    trace_region_size_dict = {
        "Llama-3.1-8B": {
            "N150": 25000000,
            "N300": 33000000,
            "T3K": 50000000,
            "TG": 50000000,
        }
    }

    LLAMA_DIR = os.getenv("LLAMA_DIR")
    HF_MODEL = os.getenv("HF_MODEL")

    if LLAMA_DIR:
        model_name = os.path.basename(LLAMA_DIR.strip("/"))
    elif HF_MODEL:
        model_name = HF_MODEL.strip("/").split("/")[-1]
    model_name = get_base_model_name(model_name)

    return trace_region_size_dict.get(model_name, {}).get(device_name, None)


@pytest.fixture
def device_params(request, galaxy_type):
    # Get param dict passed in from test parametrize (or default to empty dict)
    params = getattr(request, "param", {}).copy()

    mesh_device = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
        os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
    )
    is_single_device = (mesh_device == (1, 1)) if isinstance(mesh_device, tuple) else (mesh_device == 1)

    data_parallel_device_name = get_data_parallel_device_name(request, mesh_device, os.getenv("MESH_DEVICE"))

    override_trace_region_size = get_supported_trace_region_size(data_parallel_device_name)
    if override_trace_region_size:
        params["trace_region_size"] = override_trace_region_size

    if "fabric_config" in params:
        if is_single_device:
            params["fabric_config"] = None
        elif params["fabric_config"] == True:
            params["fabric_config"] = (
                ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
            )

    return params
