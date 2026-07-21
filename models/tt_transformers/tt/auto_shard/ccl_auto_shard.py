# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Collectives for auto-sharded layers, on top of reduce_scatter and all_gather.

sharding.py picks which mesh axis a matmul's contraction was split over; these run the collective
across that named axis, so they work on a 2D mesh too.
"""

import math
import os

import ttnn
from loguru import logger

from models.tt_transformers.tt.inline_profile import section

WIDTH = 3  # tensors are [1, 1, seq, hidden]; we always collect along hidden


# Pin the width-sharded matmul to <env_var>=<n> cores, or None for ttnn's default grid.
# num_cores must be <=8 or a multiple of 8.
# TODO(auto-shard): drive core selection from the cost model instead of an env var.
def core_grid_program_config(args, env_var, M, K, N, label):
    n = os.environ.get(env_var)
    if not n:
        return None
    grid = _num_to_coregrid(int(n))
    pc = args.matmul_1d_config(m=M, k=K, n=N, grid=grid)
    logger.info(
        f"[cores] {label}: 1D width-sharded grid=(x={grid.x}, y={grid.y}) = {grid.num_cores} cores (M={M} K={K} N={N})"
    )
    return pc


def _num_to_coregrid(x):
    if x <= 8:
        return ttnn.CoreGrid(y=1, x=x)
    if x % 8 == 0:
        return ttnn.CoreGrid(y=x // 8, x=8)
    raise ValueError(f"{x} cores: use <=8, or a multiple of 8")


_logged_grids = set()


def log_default_grid(x, weight, label):
    # ttnn won't tell us the grid it picks for an interleaved matmul, so run one throwaway
    # width-sharded matmul and read the core count off its shard spec.
    if not os.environ.get("AUTO_SHARD_LOG_CORES") or label in _logged_grids:
        return
    if x.shape[-2] > 32:
        return  # the probe hits out_subblock limits on large-M prefill
    try:
        probe = ttnn.linear(x, weight, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        ss = probe.memory_config().shard_spec
        cores = ss.grid.num_cores() if ss is not None else "interleaved(?)"
        logger.info(f"[cores] {label}: ttnn default auto grid = {cores} cores (K={x.shape[-1]} N={weight.shape[-1]})")
        ttnn.deallocate(probe)
    except Exception as e:
        logger.info(f"[cores] {label}: probe failed ({type(e).__name__}: {e}); K={x.shape[-1]} N={weight.shape[-1]}")
    _logged_grids.add(label)


def all_reduce(
    tensor,
    mesh_device,
    axis,
    *,
    replicate,
    dtype=ttnn.bfloat16,
    topology=ttnn.Topology.Linear,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    label="all_reduce",
):
    """Sum `tensor`'s partials across mesh `axis`.

    axis=None (or a size-1 axis) means nothing was split there, so there's nothing to reduce.
    replicate=True leaves the full sum on every chip; on a 1D line, replicate=False stops after
    the reduce_scatter so each chip keeps only its summed slice. A 2D reduce always replicates.
    `label` names this collective in the AUTO_SHARD_PROFILE=1 table.
    """
    mesh_shape = tuple(mesh_device.shape)
    if axis is None or mesh_shape[axis] == 1:
        return tensor

    # AUTO_SHARD_NO_CCL=1 measures tok/s without the collectives. Each all_reduce is undone by the
    # decoder's paired all_gather (also stubbed), so shapes still line up. Output is garbage.
    if os.environ.get("AUTO_SHARD_NO_CCL") == "1":
        return tensor

    with section(label, mesh_device):
        if tensor.dtype != dtype:
            tensor = ttnn.typecast(tensor, dtype)

        tensor = ttnn.reduce_scatter(
            tensor, dim=WIDTH, cluster_axis=axis, topology=topology, memory_config=memory_config
        )

        if replicate or 1 not in mesh_shape:
            tensor = ttnn.all_gather(
                tensor, dim=WIDTH, cluster_axis=axis, topology=topology, memory_config=memory_config
            )

    return tensor


def all_gather(
    tensor,
    mesh_device,
    axis,
    *,
    topology=ttnn.Topology.Ring,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    label="all_gather",
):
    """Reassemble a width-sharded `tensor` into a full replicated copy across mesh `axis`.

    axis=None (or a size-1 axis) means nothing was split over it, so there's nothing to gather.
    `label` names this collective in the AUTO_SHARD_PROFILE=1 table.
    """
    mesh_shape = tuple(mesh_device.shape)
    if axis is None or mesh_shape[axis] == 1:
        return tensor

    # See all_reduce. Only the decoder's gathers are skipped, since those cancel with a stubbed
    # all_reduce. The embedding and lm_head gathers aren't paired, so skipping them would fracture
    # the tensor rather than just slow it down.
    if os.environ.get("AUTO_SHARD_NO_CCL") == "1" and label.startswith("decoder"):
        return tensor

    with section(label, mesh_device):
        out = ttnn.all_gather(tensor, dim=WIDTH, cluster_axis=axis, topology=topology, memory_config=memory_config)
    return out


# The matmul and the reduce_scatter workers run concurrently, so they must not share cores -- two
# data-movement kernels on one core trip "Illegal NOC usage". Matmul gets rows [0, _RS_GRID_ROW),
# the RS workers get the rest. Same split as
# tests/ttnn/unit_tests/operations/ccl/test_new_matmul_reduce_scatter.py.
_RS_GRID_ROW = 6
_RS_CORE_GRID_OFFSET = (0, _RS_GRID_ROW)


# Persistent buffers for the fused op, keyed per call site so two sites with the same shape don't
# alias. These must be allocated once, not per call: ttnn.zeros writes from the host, which is
# illegal during trace capture. The eager compile run fills the cache before capture starts.
_rs_buffers = {}


def _persistent_rs_buffers(mesh_device, label, M, N, num_devices, dtype, memory_config):
    key = (label, M, N, num_devices, dtype)
    if key not in _rs_buffers:
        alloc = lambda width: ttnn.zeros(  # noqa: E731
            [1, 1, M, width], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=memory_config
        )
        # intermediate holds the per-chip matmul output; output holds this chip's scattered slice
        _rs_buffers[key] = (alloc(N), alloc(N // num_devices))
    return _rs_buffers[key]


def _rs_matmul_program_config(x, weight, mesh_device):
    """Build the 2D-multicast matmul program config the fused matmul_reduce_scatter op requires.

    matmul_reduce_scatter_async won't auto-pick a grid (it derefs program_config unconditionally,
    so None gives "bad optional access") and only accepts MatmulMultiCoreReuseMultiCastProgramConfig.
    in0_block_w=1 and 1x1 out-subblocks are always legal, so this works for any M/K/N.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.y <= _RS_GRID_ROW:
        raise ValueError(
            f"fused matmul_reduce_scatter needs a compute grid taller than {_RS_GRID_ROW} rows to keep "
            f"the matmul and reduce_scatter workers apart, but this device has {grid.x}x{grid.y}. "
            f"Lower _RS_GRID_ROW, or run with AUTO_SHARD_FUSE_RS unset to use linear + reduce_scatter."
        )
    rows = _RS_GRID_ROW
    M, N = x.shape[-2], weight.shape[-1]
    per_core_M = max(1, math.ceil(M / ttnn.TILE_SIZE / rows))
    per_core_N = max(1, math.ceil(N / ttnn.TILE_SIZE / grid.x))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, rows),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def matmul_reduce_scatter(
    x,
    weight,
    mesh_device,
    tt_ccl,
    axis,
    *,
    replicate,
    bias=None,
    dtype=ttnn.bfloat16,
    compute_kernel_config=None,
    program_config=None,
    topology=ttnn.Topology.Linear,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    activation=None,
    label="matmul_reduce_scatter",
):
    """Fused matmul + reduce_scatter: overlap the matmul with the reduce over `axis`.

    Drop-in for a serialized `ttnn.linear(x, weight)` -> `all_reduce(...)`. `replicate=True` still
    needs a trailing all_gather, since only the reduce_scatter half is fused. `bias` is added
    before the reduce_scatter, matching unfused numerics exactly.

    The fused kernel rings over the whole mesh (no cluster_axis), so it only applies on a 1D ring.
    Anything else falls back to linear + all_reduce.
    """
    mesh_shape = tuple(mesh_device.shape)

    # Fusion is off by default: on a 1x4 at batch 1 it loses in both regimes -- prefill 48ms fused
    # vs 45ms unfused, decode 33 vs 41.5 tok/s.
    #
    # Decode's loss is structural. The 2D-multicast matmul maps M across core rows, and decode has
    # M=32 (one tile), so one row works and the rest idle. Decode is DRAM-bound anyway, with no
    # compute to hide the collective behind. You'd need M>=192 to fill the grid.
    #
    # Prefill should win (M=3584 fills the grid, compute-bound) but loses by ~3ms. Untested
    # suspects: the grid split leaves the matmul 48 cores instead of 64, and
    # _rs_matmul_program_config picks the worst legal config for shape-independence. Tuning both
    # might flip it, but prefill is <1% of a 200-token generation.
    is_prefill = x.shape[-2] > ttnn.TILE_SIZE
    fused_ok = (
        os.environ.get("AUTO_SHARD_FUSE_RS") == "1"
        and is_prefill  # decode is a guaranteed loss
        and axis is not None
        and mesh_shape[axis] > 1
        and 1 in mesh_shape  # 1D line/ring only
        and topology == ttnn.Topology.Ring
    )

    if not fused_ok:
        with section(f"{label} linear", mesh_device):
            out = ttnn.linear(
                x,
                weight,
                bias=bias,
                dtype=dtype,
                compute_kernel_config=compute_kernel_config,
                program_config=program_config,
                memory_config=memory_config,
            )
        # all_reduce opens its own section; nesting would double-count it
        out = all_reduce(
            out,
            mesh_device,
            axis,
            replicate=replicate,
            dtype=dtype,
            topology=topology,
            memory_config=memory_config,
            label=f"{label} all_reduce",
        )
        return out

    M, N = x.shape[-2], weight.shape[-1]
    num_devices = mesh_shape[axis]
    # Override whatever the caller passed: the fused op only takes a 2D-multicast config, and the
    # grid isn't free anyway since it has to dodge the reduce_scatter workers.
    program_config = _rs_matmul_program_config(x, weight, mesh_device)
    intermediate_buffer, output_buffer = _persistent_rs_buffers(
        mesh_device, label, M, N, num_devices, dtype, memory_config
    )
    _, reduced = ttnn.experimental.matmul_reduce_scatter_async(
        x,
        weight,
        persistent_intermediate_buffer=intermediate_buffer,
        persistent_output_buffer=output_buffer,
        dim=WIDTH,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(axis),
        reduce_scatter_core_grid_offset=_RS_CORE_GRID_OFFSET,
        bias=bias,
        num_links=tt_ccl.get_num_links(axis),
        memory_config_rs=memory_config,
        memory_config_mm=memory_config,
        topology=topology,
        dtype=dtype,
        program_config=program_config,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
    )

    if replicate:
        reduced = all_gather(reduced, mesh_device, axis, topology=topology, memory_config=memory_config)
    return reduced


def partition(tensor, mesh_device, axis, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Shard a replicated `tensor`'s width across mesh `axis`, each chip keeping its slice.

    The inverse of all_gather, for a module that splits its contraction over `axis`. axis=None (or
    a size-1 axis) means the module wants the full width. No network traffic -- data is on-chip.
    """
    mesh_shape = tuple(mesh_device.shape)
    if axis is None or mesh_shape[axis] == 1:
        return tensor
    return ttnn.mesh_partition(tensor, dim=WIDTH, cluster_axis=axis, memory_config=memory_config)
