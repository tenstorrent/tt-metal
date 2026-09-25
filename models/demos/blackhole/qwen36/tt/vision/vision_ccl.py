# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""All-reduce to a replicated full-width activation when dim does not split into whole tiles.
Weights stay sharded; used when vision_replicated_acts is set.
"""

import os

import ttnn
from models.common.utility_functions import is_wormhole_b0

_VISION_CCL_UNTUNED = (10, 2)
_VISION_CCL_TUNED = (10, 4)
_VISION_CCL_TUNED_DEVICES = ("N300", "T3K")


def vision_ccl_tuning(device_name=None):
    """(chunks_per_sync, num_workers_per_link) for vision-tower collectives."""
    override = os.environ.get("QWEN36_VISION_CCL")
    if override == "0":
        return _VISION_CCL_UNTUNED
    if override and "," in override:
        chunks_per_sync, num_workers_per_link = (int(t) for t in override.split(","))
        return chunks_per_sync, num_workers_per_link
    if is_wormhole_b0() and device_name in _VISION_CCL_TUNED_DEVICES:
        return _VISION_CCL_TUNED
    return _VISION_CCL_UNTUNED


def vision_ccl_kwargs(device_name=None):
    chunks_per_sync, num_workers_per_link = vision_ccl_tuning(device_name)
    return {"chunks_per_sync": chunks_per_sync, "num_workers_per_link": num_workers_per_link}


def all_reduce_replicated(x, tt_ccl, topology, memory_config=ttnn.DRAM_MEMORY_CONFIG, ccl_kwargs=None):
    """Sum partials to a replicated full-width result; gather on dim 0, not dim 3."""
    assert x.shape[0] == 1, f"all_reduce_replicated expects a leading dim of 1, got {tuple(x.shape)}"
    gathered = ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=0,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=tt_ccl.get_num_links(1),
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_buffers_per_channel=2,
        **(ccl_kwargs if ccl_kwargs is not None else vision_ccl_kwargs()),
    )
    reduced = ttnn.experimental.fast_reduce_nc(
        gathered,
        dims=[0],
        output=None,
        compute_kernel_config=None,
        memory_config=memory_config,
    )
    ttnn.deallocate(gathered)
    return reduced
