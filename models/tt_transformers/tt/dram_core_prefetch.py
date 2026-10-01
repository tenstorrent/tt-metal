# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Decode weight streaming through the Blackhole DRAM-core tensor prefetcher.

The DRAM controller RISCs (two senders per bank) stream each projection's weights into a shallow global
circular buffer on a 64-core ring; gather_in0 ring matmuls with ``stream_in1`` consume them block by block.
Weights are kept in the receiver-contiguous ND-sharded DRAM layout (64 shards round-robin over the 8 banks).
Because the stream runs off the command queue, it keeps filling the window while attention/norm ops run,
so that non-matmul time is hidden under the weight stream. Measured on one P150 for a Llama-3.1-8B layer:
392 us vs 420 us for the 2-reader DRAM-sharded matmuls with ~37 us of other ops between them.
"""
import atexit

from loguru import logger

import ttnn

TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


class DramCorePrefetch:
    RING = 64
    GRID = (8, 8)
    # per receiver, in units of the largest page (w2: 7x2 bfp8 tiles = 15 KB); 32 -> 487 KB of L1. Deeper windows
    # buffer more weight stream during attention/norms but must leave room for the ring matmul's full-K in0.
    WINDOW_PAGES = int(__import__("os").environ.get("TT_DRAM_CORE_PREFETCH_WINDOW", "32"))

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.num_banks = mesh_device.dram_grid_size().x
        assert self.num_banks * (self.RING // self.num_banks) == self.RING
        self.entries = {}  # name -> (program_config, weight) of the first layer registered
        self.gcb = None
        self.started = False
        self.receiver_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.GRID[0] - 1, self.GRID[1] - 1))}
        )

    # ---- layouts ----
    def weight_mem_config(self, k, n):
        assert n % self.RING == 0 and (n // self.RING) % ttnn.TILE_SIZE == 0, f"n={n} must split over the ring"
        assert k % (self.RING * ttnn.TILE_SIZE) == 0, f"k={k} must split over the ring in whole tiles"
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.num_banks - 1, 0))})
        nd = ttnn.NdShardSpec(
            ttnn.Shape([k, n // self.RING]),
            dram_grid,
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        return ttnn.MemoryConfig(ttnn.BufferType.DRAM, nd)

    def make_weight(self, torch_weight, dtype, cache_file_name=None):
        """torch_weight: (1, 1, k, n)."""
        k, n = torch_weight.shape[-2], torch_weight.shape[-1]
        kwargs = dict(
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, memory_config=self.weight_mem_config(k, n)
        )
        if cache_file_name is not None:
            try:
                return ttnn.as_tensor(torch_weight, cache_file_name=cache_file_name, **kwargs)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"DramCorePrefetch: caching {cache_file_name} failed ({e}); loading without cache")
        return ttnn.as_tensor(torch_weight, **kwargs)

    def program_config(self, k, n):
        out_block_w = n // self.RING // ttnn.TILE_SIZE
        sub_w = min(out_block_w, 8)
        while sub_w > 1 and out_block_w % sub_w:
            sub_w -= 1
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=self.GRID,
            in0_block_w=1,  # the factory derives the K block from the in0 shard width
            out_subblock_h=1,
            out_subblock_w=sub_w,
            per_core_M=1,
            per_core_N=out_block_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=self.RING // self.num_banks,
            untilize_out=False,
            stream_in1=True,
        )

    def out_mem_config(self, n):
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, n // self.RING),
            core_grid=self.receiver_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def in_mem_config(self, k):
        return self.out_mem_config(k)

    def page_bytes(self, k, n, dtype):
        return (k // ttnn.TILE_SIZE // self.RING) * (n // self.RING // ttnn.TILE_SIZE) * TILE_BYTES[dtype]

    # ---- lifecycle ----
    def register(self, name, weight, k, n):
        if name not in self.entries:
            self.entries[name] = (self.program_config(k, n), weight)

    def _bank_receivers(self, bank):
        recv_per_bank = self.RING // self.num_banks
        cores = []
        for s in range(recv_per_bank):
            ring_pos = bank + s * self.num_banks
            cores.append(
                ttnn.CoreRange(
                    ttnn.CoreCoord(ring_pos % self.GRID[0], ring_pos // self.GRID[0]),
                    ttnn.CoreCoord(ring_pos % self.GRID[0], ring_pos // self.GRID[0]),
                )
            )
        return ttnn.CoreRangeSet(cores)

    def ensure_built(self):
        """Create the GCB for a decode phase. Called from the model's switch to decode mode; the GCB is released
        for prefill (its 487 KB/core would clash with prefill's L1 use) and rebuilt here. Decode traces carry the
        GCB address, so a rebuild must land on the same address: allocation is deterministic given the same
        persistent buffers, and the check below turns any surprise into an error instead of silent garbage."""
        if self.gcb is None:
            self.build()

    def rebuild_if_released(self):
        """At a decode-mode switch: rebuild a GCB that prefill released (decode traces hold its address).
        The first build is left to the first eager decode forward, so the GCB lands below the persistent
        decode inputs and releasing it hands prefill back a contiguous region."""
        if self.gcb is None and getattr(self, "gcb_address", None) is not None:
            self.build()

    def release(self):
        if self.gcb is not None:
            ttnn.synchronize_device(self.mesh_device)  # outstanding stream requests have been consumed
            self.gcb = None
            import gc

            gc.collect()

    def build(self):
        assert self.entries, "no decode weights registered"
        pcs = [pc for pc, _ in self.entries.values()]
        weights = [w for _, w in self.entries.values()]
        max_page = max(self.page_bytes(w.padded_shape[-2], w.padded_shape[-1], w.dtype) for w in weights)
        size = self.WINDOW_PAGES * max_page
        bank_to_receivers = [(b, self._bank_receivers(b)) for b in range(self.num_banks)]
        self.gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
            self.mesh_device, pcs, weights, bank_to_receivers=bank_to_receivers, size=size
        )
        addr = self.gcb.buffer_address()
        prev = getattr(self, "gcb_address", None)
        if prev is not None and prev != addr:
            raise RuntimeError(
                f"DramCorePrefetch: GCB rebuilt at {addr:#x} but decode traces were captured with {prev:#x}"
            )
        self.gcb_address = addr
        if not self.started:
            ttnn.experimental.start_tensor_prefetcher(self.mesh_device)
            self.started = True
            atexit.register(self.stop)
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(self.mesh_device, 0)  # weights landed before any read
        logger.info(
            f"DramCorePrefetch: {len(self.entries)} projections, GCB window {self.WINDOW_PAGES} x {max_page} B = {size} B per receiver at {addr:#x}, ring {self.RING}"
        )

    def stop(self):
        if self.started:
            self.started = False
            try:
                ttnn.experimental.stop_tensor_prefetcher(self.mesh_device)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"DramCorePrefetch.stop: {e}")

    # ---- the matmul ----
    def linear(self, x, weight, k, n, dtype, compute_kernel_config):
        return ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
            x,
            weight,
            global_cb=self.gcb,
            program_config=self.program_config(k, n),
            memory_config=self.out_mem_config(n),
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
        )
