# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The MTP drafter's own LM head under TP: a bfloat4_b copy of the vocab-sharded head weight, width-sharded over the
DRAM banks and read by the multi-core DRAM-sharded matmul pipeline (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
with use_multicore_pipeline).

The verify keeps the shared bfloat8_b head (Qwen36Model.lm_head_weight), so the served tokens do not change; the
drafts come from this copy, so draft acceptance can move. The copy adds its size to DRAM (DraftHeadPlan.weight_bytes)
and replaces nothing: the verify still reads the shared head.

DraftHeadPlan is the host arithmetic of the pipeline's layout and L1 for the drafter's call (bfloat16 activation,
bfloat4_b weight, float32 logits, fp32 Dest), after create_program_dram_sharded_multicore_spec in
matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp at one tile of rows. A shape the pipeline does not take,
or whose L1 does not fit, raises ValueError, and the drafter keeps the shared head.
"""

import math

from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

TILE = 32
# Logits tiles per core of the head's L1 width-sharded output (ceil(N / OUT_TILES_PER_CORE) cores). Every L1 buffer
# takes its address range on every core, so a bank-wide shard (122 float32 tiles, 488 KB at Qwen3.8 TP=8) would take
# L1 the pipeline's circular buffers need; 10 tiles put 40 KB on each of 97 cores.
OUT_TILES_PER_CORE = 10
# L1 of one Blackhole core left for the head's circular buffers and L1 buffers: 1536 KB minus the unreserved base
# (memory map end and the 69 KB kernel config buffer, 112000 B) minus the serving l1_small_size (24576 B).
L1_BANK_BYTES = 1536 * 1024 - 112000 - 24576
# L1 the decode traces hold for their whole life: DecodeResidualAllReduce.buffer ([32, dim * tp] bfloat16 over 40
# cores, 64 KB at dim 5120 and TP 8).
L1_RESERVED_BYTES = 64 * 1024
# The pipeline's constants on Blackhole: tiles per column pass, two weight streams of depth-3 blocks, DRAM alignment.
_DST_TILES = 8
_WEIGHT_STREAMS, _STREAM_DEPTH = 2, 3
_DRAM_ALIGN = 64
_ACT_TILE, _WEIGHT_TILE, _OUT_TILE = 2048, 576, 4096  # bfloat16, bfloat4_b, float32
# The pipeline's role table holds one byte per core.
_MAX_WORK_ITEMS = 255


class DraftHeadPlan:
    """The drafter head's layout for one device's head weight [k, n] (pure arithmetic, no device).

    banks: DRAM banks of the device (dram_grid_size().x); grid: (x, y) of its compute grid."""

    def __init__(self, k, n, banks, grid):
        if k % TILE:
            raise ValueError(f"K {k} is not a whole number of tiles")
        self.banks = banks
        self.grid = tuple(grid)
        self.k_tiles = k // TILE
        self.n_tiles = math.ceil(n / TILE)
        # The plain head's auto-configured program streams K in 2-tile blocks when K is even.
        self.in0_block_w = 2 if self.k_tiles % 2 == 0 else 1
        self.bank_tiles = math.ceil(self.n_tiles / banks)
        # About one reader per 8 tiles of a bank, 2 to 4: the pipeline needs at least 2.
        self.readers = min(4, max(2, (self.bank_tiles + 7) // 8))
        if self.bank_tiles * banks - self.n_tiles >= self.bank_tiles:
            raise ValueError(f"{self.n_tiles} vocab tiles leave a padding-only bank of {self.bank_tiles} tiles")
        if self.k_tiles % (banks * self.in0_block_w):
            raise ValueError(
                f"K {self.k_tiles} tiles do not split into {banks} activation shards of {self.in0_block_w}-tile blocks"
            )
        if banks > self.grid[0]:
            raise ValueError(f"{banks} activation cores do not fit a grid row of {self.grid[0]}")
        # Activation: width-sharded over one core per bank.
        self.act_shard_w = self.k_tiles // banks * TILE
        self.per_core_N = OUT_TILES_PER_CORE
        self.out_cores = math.ceil(self.n_tiles / self.per_core_N)
        cores = self.grid[0] * self.grid[1]
        if self.out_cores > cores:
            raise ValueError(f"{self.out_cores} logits cores exceed the {self.grid[0]}x{self.grid[1]} grid")
        self.layout = self._layout()
        if self.layout["work"] > cores:
            raise ValueError(f"{self.layout['work']} pipeline cores exceed the {self.grid[0]}x{self.grid[1]} grid")
        need, avail = self.l1_bytes(), L1_BANK_BYTES - L1_RESERVED_BYTES
        if need["total"] > avail:
            raise ValueError(f"the head needs {need['total'] // 1024} KB of L1 per core, {avail // 1024} KB are free")

    def _layout(self):
        """Per bank cn column groups and ck K-row groups, p column passes of ncp tiles, kb K tiles per streamed
        weight block, max_own output tiles per core, work items. Raises on the layouts the factory refuses."""
        nbt, kt, w = self.bank_tiles, self.k_tiles, self.readers
        cn = min(w, nbt)
        ck = min(w // nbt, kt) if w > nbt else 1
        ncmax = math.ceil(nbt / cn)
        p = math.ceil(ncmax / _DST_TILES)
        ncp = math.ceil(ncmax / p)
        if ck > 1 and p > 1:
            raise ValueError(f"{w} readers per bank: K row groups need a single column pass ({nbt} tiles per bank)")
        if nbt // cn < p:
            raise ValueError(f"{w} readers per bank leave a core fewer columns than its {p} passes")
        kb = min(self.in0_block_w, math.ceil(kt / ck))
        max_own = 0
        for c in range(cn):
            width = nbt * (c + 1) // cn - nbt * c // cn
            for kidx in range(ck):
                max_own = max(max_own, width * (kidx + 1) // ck - width * kidx // ck)
        work = math.ceil(self.n_tiles / nbt) * cn * ck
        if work > _MAX_WORK_ITEMS:
            raise ValueError(f"{work} work items exceed the pipeline's one-byte role table")
        max_blocks = math.ceil(math.ceil(kt / ck) / kb)
        return dict(cn=cn, ck=ck, p=p, ncp=ncp, kb=kb, max_own=max_own, max_blocks=max_blocks, work=work)

    def l1_bytes(self):
        """L1 per core of one call in bytes: ``cb`` (the circular buffers on every core of the work rectangle: the
        whole-K activation copy, the weight streams, the output staging, the K-group partials when ck > 1) and the
        L1 buffers ``act_shard`` and ``out_shard``, each of which takes its size on every core."""
        lay = self.layout
        weight_tile = -(-_WEIGHT_TILE // _DRAM_ALIGN) * _DRAM_ALIGN
        x = lay["max_blocks"] * lay["kb"] * _ACT_TILE
        w = _WEIGHT_STREAMS * _STREAM_DEPTH * lay["kb"] * lay["ncp"] * weight_tile
        out = lay["max_own"] * _OUT_TILE
        part = (lay["ncp"] + lay["ck"] * lay["max_own"]) * _OUT_TILE if lay["ck"] > 1 else 0
        cb = x + w + out + part
        act_shard = self.act_shard_w // TILE * _ACT_TILE
        out_shard = self.per_core_N * _OUT_TILE
        return dict(cb=cb, act_shard=act_shard, out_shard=out_shard, total=cb + act_shard + out_shard)

    def weight_memory_config(self):
        """DRAM width-sharded, one [K, bank_tiles] shard per bank; the last bank's padding tiles lie past N."""
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.banks - 1, 0))})
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(bank_grid, [self.k_tiles * TILE, self.bank_tiles * TILE], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def weight_bytes(self):
        """DRAM per device of the bfloat4_b copy, padding tiles included."""
        return self.k_tiles * self.bank_tiles * self.banks * _WEIGHT_TILE


class DraftHead:
    """The bfloat4_b DRAM-sharded head weight and the program that reads it.

    linear(x, cfg) takes one tile of rows of the replicated full-width hidden [1, 1, R, k] (TILE, DRAM) and returns
    this device's float32 logits shard [1, 1, R, n] in DRAM interleaved, the form ttnn.linear on the shared head
    returns."""

    def __init__(self, weight, plan):
        self.weight = weight
        self.plan = plan
        self.act_memcfg = ttnn.create_sharded_memory_config(
            shape=(TILE, plan.act_shard_w),
            core_grid=ttnn.CoreGrid(x=plan.banks, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.progcfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=plan.in0_block_w,
            per_core_M=1,
            per_core_N=plan.per_core_N,
            fused_activation=None,
            num_workers_per_dram_bank=plan.readers,
            use_multicore_pipeline=True,
        )

    def applies(self, x):
        shape = tuple(x.shape)
        return (
            len(shape) == 4
            and shape[:2] == (1, 1)
            and shape[2] <= TILE
            and shape[3] == self.plan.k_tiles * TILE
            and x.layout == ttnn.TILE_LAYOUT
        )

    def linear(self, x, compute_kernel_config):
        x_sh = ttnn.to_memory_config(x, self.act_memcfg)
        out = ttnn.linear(
            x_sh,
            self.weight,
            program_config=self.progcfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.float32,
        )
        ttnn.deallocate(x_sh)
        logits = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return logits


def build_draft_head(lm_head_weight, mesh_device, cache_path):
    """The drafter head for the checkpoint head ``lm_head_weight`` [vocab, k] vocab-sharded over the TP axis, or None
    when DraftHeadPlan does not take the shape (the drafter then reads the shared head)."""
    vocab, k = (int(s) for s in lm_head_weight.shape)
    tp = tuple(mesh_device.shape)[1] if tpc.is_2d_mesh(mesh_device) else mesh_device.get_num_devices()
    grid = mesh_device.compute_with_storage_grid_size()
    try:
        if vocab % tp:
            raise ValueError(f"vocab {vocab} does not split over {tp} devices")
        plan = DraftHeadPlan(k, vocab // tp, mesh_device.dram_grid_size().x, (grid.x, grid.y))
    except ValueError as e:
        logger.info(f"MTP drafter: shared LM head ({e})")
        return None
    interleaved = ttnn.as_tensor(
        lm_head_weight,
        preprocess=lambda t: t.T.contiguous(),  # [k, vocab]
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
        mesh_mapper=tpc.tp_shard_mapper(mesh_device, -1),
    )
    if tuple(interleaved.shape)[-2:] != (k, vocab // tp):
        raise ValueError(f"the drafter head shard is {tuple(interleaved.shape)}, expected [{k}, {vocab // tp}]")
    weight = ttnn.to_memory_config(interleaved, plan.weight_memory_config())
    ttnn.deallocate(interleaved)
    logger.info(
        f"MTP drafter: bfloat4_b LM head, DRAM-sharded {plan.banks} x {plan.bank_tiles} tiles, "
        f"{plan.readers} readers per bank, {plan.weight_bytes() / 2**20:.1f} MiB per device"
    )
    return DraftHead(weight, plan)
