# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import re

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, is_quasar, is_wormhole_b0
from models.demos.utils.model_targets import normalize_sku
from models.demos.utils.trace_region_sizes import hf_model_name_candidates, resolve_trace_region_size_for_candidates

# NOTE: We need to override trace_region_size before the mesh device is opened
# NOTE: When using DP, we need to have the implemented logic because when we parametrize the test with a specific trace region size, all submeshes will have that trace region size
# example of the above : T3K (DP-8-b1 ; @parametrize(trace_region_size=X) -> we effectively have 8 N150's with trace_region_size=X which could lead to OOM if X is too large)

# TODO: For now, each conftest.py should call get_supported_trace_region_size if they want to override the trace region size


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
    elif is_quasar():
        dict_device_names = {
            1: "QUASAR_BOARD",
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


def _model_name_candidates() -> list[str]:
    hf_model = os.getenv("HF_MODEL")
    if not hf_model:
        base = base_model_name_from_env()
        return [base] if base else []

    return hf_model_name_candidates(hf_model)


def get_logical_sku(request, mesh_device):
    """Canonical SKU for the submesh actually opened.

    Derives the SKU from the parametrized mesh shape, ``data_parallel``, and
    ``MESH_DEVICE`` rather than the physical cluster, so a logical submesh
    (e.g. a 1x4 slice of a Galaxy, or a ``MESH_DEVICE=N300`` run on a T3K) maps
    to the SKU of the mesh that is actually opened. Returns ``None`` for CPU or
    when the device count is unsupported.
    """
    device_name_based_on_dp = device_name_based_on_data_parallel(request, mesh_device, os.getenv("MESH_DEVICE"))
    if not device_name_based_on_dp or device_name_based_on_dp == "CPU":
        return None
    return normalize_sku(device_name_based_on_dp)


def get_supported_trace_region_size(request, mesh_device):
    sku = get_logical_sku(request, mesh_device)
    if sku is None:
        return None

    candidates = _model_name_candidates()
    if not candidates:
        return None

    # Unconfigured (model, SKU) pairs fall back to dynamic allocation (0).
    return resolve_trace_region_size_for_candidates(candidates, sku)
