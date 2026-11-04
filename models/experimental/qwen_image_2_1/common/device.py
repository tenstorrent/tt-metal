# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device lifetime for the single-p150 Tensix-dispatch implementation."""
from __future__ import annotations

import os
from contextlib import contextmanager

import ttnn

DEFAULT_L1_SMALL = 32768
DEFAULT_TRACE_REGION = int(os.environ.get("QWEN_IMAGE_TRACE_REGION", 1024 * 1024 * 1024))


def dispatch_core_config(eth: bool | None = None) -> ttnn.DispatchCoreConfig:
    if eth is None:
        eth = os.environ.get("QWEN_IMAGE_ETH_DISPATCH", "0") == "1"
    return ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH if eth else ttnn.DispatchCoreType.WORKER)


def open_device(
    *,
    eth_dispatch: bool | None = None,
    trace_region_size: int = DEFAULT_TRACE_REGION,
    l1_small_size: int = DEFAULT_L1_SMALL,
    num_command_queues: int = 1,
    worker_l1_size: int | None = None,
):
    kw = dict(
        mesh_shape=ttnn.MeshShape(1, 1),
        dispatch_core_config=dispatch_core_config(eth_dispatch),
        trace_region_size=trace_region_size,
        l1_small_size=l1_small_size,
        num_command_queues=num_command_queues,
    )
    if worker_l1_size is not None:
        kw["worker_l1_size"] = worker_l1_size
    dev = ttnn.open_mesh_device(**kw)
    return dev


def close_device(dev):
    ttnn.close_mesh_device(dev)


@contextmanager
def device(**kw):
    dev = open_device(**kw)
    try:
        yield dev
    finally:
        close_device(dev)


def core_grid(dev) -> ttnn.CoreCoord:
    return dev.compute_with_storage_grid_size()
