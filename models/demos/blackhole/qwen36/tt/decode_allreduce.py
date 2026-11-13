# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replicated decode residual for the TP=8 MTP verify of Qwen3.8-27B.

Without it, the decode residual is fractured ([1, 1, R, dim / tp] per device): every norm all-gathers it and every
sublayer reduce-scatters its row-parallel output back into it, then adds the residual (2 collectives and 1 add per
sublayer). With it, the residual is replicated full width in L1, width-sharded over a 10x4 core grid ("RES40",
shard [32, dim / 40]); norms run locally on it, and each sublayer's row-parallel output goes through one
all_reduce_async that also adds the residual (one_shot=True, residual_tensor=).

Only the shape the operation was tested at takes this path: a (1, 8) mesh on a 1D fabric (one_shot routes by hop
count), dim 5120, dense MLP, at most one tile of rows. Prefill and the plain decode keep the fractured residual; the
residual layout stays private to each forward.
"""

from loguru import logger

import ttnn

RES40_GRID = (10, 4)
SUPPORTED_MESH_SHAPE = (1, 8)
SUPPORTED_DIM = 5120
_UNSET = object()


class ResidualAllReduce:
    """One sublayer's all-reduce: sum the row-parallel partials over the TP axis and add ``residual``."""

    def __init__(self, owner, residual):
        self.owner = owner
        self.residual = residual

    def all_reduce(self, partial):
        return self.owner.all_reduce(partial, self.residual)


class DecodeResidualAllReduce:
    """The RES40 layout, its local norms, the model-entry gather and the residual-adding all-reduce.

    One instance per TT_CCL (get_decode_allreduce), shared by the model and every decoder layer, so the persistent
    all-reduce buffer is allocated once, at model construction, never inside a trace.
    """

    def __init__(self, mesh_device, args, tt_ccl):
        self.mesh = mesh_device
        self.tt_ccl = tt_ccl
        self.dim = args.dim
        self.tp = list(mesh_device.shape)[1]
        self.topology = args.ccl_topology()
        self.num_links = tt_ccl.get_num_links(1)
        ncores = RES40_GRID[0] * RES40_GRID[1]
        grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(RES40_GRID[0] - 1, RES40_GRID[1] - 1))}
        )

        def width_mem(width):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, width], ttnn.ShardOrientation.ROW_MAJOR),
            )

        shard_w = self.dim // ncores
        self.res_mem = width_mem(shard_w)
        block_w = shard_w // ttnn.TILE_SIZE
        self.norm_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=RES40_GRID,
            subblock_w=min(4, block_w),
            block_h=1,
            block_w=block_w,
            inplace=False,
        )
        # all_reduce_async stages every device's partial here: tp x the output shard on the output's cores.
        self.buffer = ttnn.empty(
            [1, 1, ttnn.TILE_SIZE, self.dim * self.tp],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=width_mem(self.dim * self.tp // ncores),
        )

    def enter(self, x):
        """Model entry: the fractured embedding [1, 1, R, dim / tp] -> the replicated RES40 residual [1, 1, R, dim]."""
        gathered = ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.num_links,
            topology=self.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        out = ttnn.to_memory_config(gathered, self.res_mem)
        if out is not gathered:
            ttnn.deallocate(gathered)
        return out

    def rms_norm(self, x, norm):
        """Local sharded rms_norm of the RES40 residual with the framework RMSNorm ``norm``'s replicated weight."""
        return ttnn.rms_norm(
            x,
            epsilon=norm.eps,
            weight=norm.weight,
            program_config=self.norm_pc,
            memory_config=self.res_mem,
            compute_kernel_config=norm.compute_kernel_config_hifi2,
        )

    def final_norm(self, x, norm):
        """The final norm before the LM head: the local norm, then DRAM interleaved as the LM head reads it."""
        normed = self.rms_norm(x, norm)
        out = ttnn.to_memory_config(normed, ttnn.DRAM_MEMORY_CONFIG)
        if out is not normed:
            ttnn.deallocate(normed)
        return out

    def with_residual(self, residual):
        """The all-reduce a sublayer whose input is ``residual`` ends with; its output is the next residual."""
        return ResidualAllReduce(self, residual)

    def all_reduce(self, partial, residual):
        """Sum the per-device row-parallel ``partial`` [.., R, dim] over the TP axis and add ``residual`` (RES40).

        The partial is resharded to RES40 and cast to bfloat16 (the residual's dtype) first. one_shot gives the same
        bytes as the default mode and lets back-to-back calls share the one persistent buffer (each device starts call
        n + 1 only after every owner has read its slots of call n)."""
        shape = list(partial.shape)
        rows = shape[-2]
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
            partial = ttnn.reshape(partial, (1, 1, rows, shape[-1]))
        if list(residual.shape) != [1, 1, rows, self.dim] or residual.memory_config() != self.res_mem:
            raise ValueError(
                f"the residual must be the RES40 tensor [1, 1, {rows}, {self.dim}], got {list(residual.shape)} in "
                f"{residual.memory_config()}"
            )
        packed = ttnn.to_memory_config(partial, self.res_mem)
        if packed is not partial:
            ttnn.deallocate(partial)
        if packed.dtype != ttnn.bfloat16:
            cast = ttnn.typecast(packed, ttnn.bfloat16)
            ttnn.deallocate(packed)
            packed = cast
        out = ttnn.experimental.all_reduce_async(
            packed,
            self.buffer,
            cluster_axis=1,
            mesh_device=self.mesh,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
            topology=self.topology,
            num_links=self.num_links,
            memory_config=self.res_mem,
            residual_tensor=residual,
            one_shot=True,
        )
        ttnn.deallocate(packed)
        return out


def decode_allreduce_unsupported(args, mesh_device):
    """Why the replicated decode residual does not apply to this model and mesh, or None when it does."""
    shape = tuple(mesh_device.shape)
    if shape != SUPPORTED_MESH_SHAPE:
        return f"mesh {list(shape)} is not {list(SUPPORTED_MESH_SHAPE)}"
    if args.dim != SUPPORTED_DIM:
        return f"hidden dim {args.dim} is not {SUPPORTED_DIM}"
    fabric = ttnn.get_fabric_config()
    if fabric not in (ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_1D_RING):
        return f"fabric {fabric} is not a 1D fabric"
    if getattr(args, "moe_num_experts", 0) > 0:
        return "the MoE MLP reduce-scatters into the fractured residual"
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < RES40_GRID[0] or grid.y < RES40_GRID[1]:
        return f"compute grid {grid.x}x{grid.y} is smaller than {RES40_GRID[0]}x{RES40_GRID[1]}"
    return None


def get_decode_allreduce(args, tt_ccl):
    """The DecodeResidualAllReduce shared by every user of ``tt_ccl``, or None (fractured residual) when the shape is
    not the tested one. Built on the first call and cached on ``tt_ccl``, so the model and its layers always agree."""
    if tt_ccl is None:
        return None
    cached = getattr(tt_ccl, "_qwen36_decode_ar", _UNSET)
    if cached is not _UNSET:
        return cached
    reason = decode_allreduce_unsupported(args, tt_ccl.mesh_device)
    inst = None
    if reason is None:
        inst = DecodeResidualAllReduce(tt_ccl.mesh_device, args, tt_ccl)
        logger.info(f"the MTP verify runs on the replicated RES40 residual (num_links={inst.num_links})")
    else:
        logger.info(f"the MTP verify keeps the fractured residual: {reason}")
    tt_ccl._qwen36_decode_ar = inst
    return inst
