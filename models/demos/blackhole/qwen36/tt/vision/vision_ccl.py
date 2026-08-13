# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CCL helper for the vision tower's replicated-activation mode.

The tower's normal TP contract keeps activations FRACTURED along dim=3, restored after each
row-parallel projection by ``tt_all_reduce(dim=3)`` (a reduce_scatter on T3K/QB2). That requires
``dim`` to split into a whole number of TILES per device — and unlike ``hidden_dim`` (which
``VisionModelArgs`` pads to ``tile_size * num_devices``), ``dim`` comes straight from the HF config.
Qwen3.6-27B's vision dim 1152 is 36 tiles: 9 per device at TP=4, but 4.5 at TP=8, and a tile cannot
be split across devices.

So at TP=8 the tower switches to replicated activations (``args.vision_replicated_acts``): the
row-parallel out-projections all-reduce to a full-width replicated tensor instead of
reduce-scattering to a fractured one. Weights stay sharded, so no TP compute is given up — the
tower is not run redundantly. The cost is that an all-reduce moves more data than a reduce_scatter.
"""

import ttnn


def all_reduce_replicated(x, tt_ccl, topology, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Sum per-device partials, leaving the FULL-width result replicated on every device.

    Gathers along dim 0 rather than dim 3: stacking the partials on the batch axis carries no
    tile-divisibility requirement, whereas splitting dim 3 is exactly what is impossible here.
    ``x`` is [1, 1, S, dim] (a partial sum of the full dim); the result is [1, 1, S, dim].
    """
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
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
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
