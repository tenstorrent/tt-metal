# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh plan, CCL and weight helpers shared by the MiMo TTNN modules.

Parallel plan (Blackhole Galaxy, 4x8):

* TP=8 along mesh axis 1. One replica is one Galaxy row (a 1x8 submesh). Every collective in a
  replica runs on ``cluster_axis=1``, so it stays inside the row's ring.
* DP=4 along mesh axis 0. The four rows are independent replicas with their own weights, KV cache
  and users (see ``tt/generator.py``). No collective crosses rows.

Per chip, both attention layer types see the same shape: 8 of the 64 Q heads and 1 KV head (SWA
layers have 8 KV heads, one per chip; global layers have 4, so chips 2i and 2i+1 both hold KV head i).
Routed experts are tensor-parallel too: every chip holds all 256 experts with 1/8 of
``moe_intermediate_size`` (2048 / 8 = 256), and the expert output is all-reduced across the row.
"""

import math
import os

import torch

import ttnn

TILE = ttnn.TILE_SIZE


class MeshPlan:
    """TP on ``tp_axis`` of a (sub)mesh. For a 1x8 replica ``dp == 1``; a 4x8 parent has ``dp == 4``."""

    def __init__(self, mesh_device, tp_axis=1, num_links=2, topology=ttnn.Topology.Ring):
        self.mesh_device = mesh_device
        self.mesh_shape = tuple(mesh_device.shape)
        self.tp_axis = tp_axis
        self.tp = self.mesh_shape[tp_axis]
        self.num_links = num_links
        self.topology = topology if self.tp > 2 else ttnn.Topology.Linear

    def shard(self, dim):
        """Shard tensor ``dim`` across the TP axis, replicate across the other axis."""
        dims = [None, None]
        dims[self.tp_axis] = dim
        return ttnn.ShardTensor2dMesh(self.mesh_device, dims=tuple(dims), mesh_shape=self.mesh_shape)

    def replicate(self):
        return ttnn.ReplicateTensorToMesh(self.mesh_device)

    def all_reduce(self, x, memory_config=None):
        if self.tp == 1:
            return x
        out = ttnn.all_reduce(
            x,
            cluster_axis=self.tp_axis,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        x.deallocate(True)
        return out

    def all_gather(self, x, dim, memory_config=None):
        if self.tp == 1:
            return x
        return ttnn.all_gather(
            x,
            dim=dim,
            cluster_axis=self.tp_axis,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    def to_torch_tp_concat(self, x, dim):
        """Host readback of a TP-sharded tensor from the first replica row: concat the TP shards on ``dim``."""
        shards = ttnn.get_device_tensors(x)
        row = [ttnn.to_torch(shards[i]) for i in range(self.tp)]
        return torch.cat(row, dim=dim)

    def to_torch_replicated(self, x):
        return ttnn.to_torch(ttnn.get_device_tensors(x)[0])


def cache_name(cache_dir, *parts):
    if cache_dir is None:
        return None
    return os.path.join(cache_dir, "_".join(str(p) for p in parts))


def cache_exists(cache_file, dtype, layout=ttnn.TILE_LAYOUT):
    """True when ``as_weight`` would load ``cache_file`` without needing the host tensor."""
    if cache_file is None:
        return False
    return os.path.exists(f"{cache_file}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


def as_weight(torch_tensor, plan, mapper, dtype, cache_file, layout=ttnn.TILE_LAYOUT, memory_config=None):
    """``ttnn.as_tensor`` that loads from the cache when ``torch_tensor`` is None and the cache is warm."""
    return ttnn.as_tensor(
        torch_tensor,
        device=plan.mesh_device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=mapper,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_file,
    )


def compute_config(fidelity=ttnn.MathFidelity.HiFi2, fp32_acc=True, packer_l1_acc=True):
    # WormholeComputeKernelConfig is the compute config type for Blackhole too.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


def mcast_1d_config(grid, m, n, k, in0_block_w=4, out_subblock_w=4):
    """1D in0-mcast matmul config over ``grid`` (x, y) cores; N split across cores. Used by sparse_matmul."""
    core_x, core_y = grid
    num_cores = core_x * core_y
    nt = math.ceil(n / TILE)
    kt = math.ceil(k / TILE)
    per_core_n = math.ceil(nt / num_cores)
    while kt % in0_block_w:
        in0_block_w -= 1
    out_subblock_w = min(out_subblock_w, per_core_n)
    while per_core_n % out_subblock_w:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=out_subblock_w,
        per_core_M=max(TILE, m) // TILE,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def pad_to(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple


class DramShardedMatmul:
    """Decode matmul [1, 1, 32, K] x [K, N] with the weight width-sharded over the DRAM banks.

    One worker core per DRAM bank reads only its own bank, which for these M = 32 decode matmuls is
    2-5x faster than a DRAM-interleaved weight (BH, measured: router 50 -> 11 us, SWA qkv 72 -> 34 us).
    N is padded per chip to a multiple of 32 x banks; ``__call__`` returns the padded width and callers
    slice when it matters. Prefill keeps using ``ttnn.linear`` on ``interleaved(weight)``.
    """

    def __init__(self, plan, k, n, out_dtype=ttnn.bfloat16, compute_kernel_config=None):
        device = plan.mesh_device
        self.banks = banks = device.dram_grid_size().x
        self.k, self.n = k, n
        self.n_pad = math.ceil(n / (TILE * banks)) * TILE * banks
        assert k % (TILE * banks) == 0, (k, banks)
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        self.weight_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (k, self.n_pad // banks), ttnn.ShardOrientation.ROW_MAJOR),
        )
        cores = ttnn.num_cores_to_corerangeset(banks, device.compute_with_storage_grid_size(), row_wise=True)

        def l1_width(width):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, (TILE, width), ttnn.ShardOrientation.ROW_MAJOR),
            )

        self.in_memory_config = l1_width(k // banks)
        self.out_memory_config = l1_width(self.n_pad // banks)
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=k // TILE // banks,
            per_core_M=1,
            per_core_N=self.n_pad // TILE // banks,
            fused_activation=None,
        )
        self.out_dtype = out_dtype
        self.compute_kernel_config = compute_kernel_config or compute_config(ttnn.MathFidelity.HiFi2)

    def pad_chip_blocks(self, w, tp):
        """Host [K, tp * n] (chip blocks side by side) -> [K, tp * n_pad], zero columns at the end of each block."""
        blocks = w.reshape(w.shape[0], tp, self.n)
        return torch.nn.functional.pad(blocks, (0, self.n_pad - self.n)).reshape(w.shape[0], tp * self.n_pad)

    def __call__(self, x, weight, memory_config=None):
        """x [1, 1, 32, K] (interleaved) -> [1, 1, 32, n_pad] (``memory_config``, default DRAM interleaved)."""
        x_sharded = ttnn.to_memory_config(x, self.in_memory_config)
        out = ttnn.matmul(
            x_sharded,
            weight,
            program_config=self.program_config,
            memory_config=self.out_memory_config,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        x_sharded.deallocate(True)
        result = ttnn.to_memory_config(out, memory_config or ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return result
