# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Tensor-prefetcher linear for Inkling-2-Mini decode on 1x Blackhole p100a.

The DRAM cores' DRISC senders stream a weight's K-blocks into L1 pipes on a 7 x R receiver ring
while earlier ops still run; an mcast-in0 1D matmul on the ring then reads them in place.
Measured: k/v 37 -> 17 us, o 64 -> 40 us; q (2304 x 6560) gets no faster.

Usage (one object per device, shared by every prefetched weight):
    pf = PrefetchedLinear(device, recv_rows=3, ring_kb=768)
    w = pf.weight(host_w)          # torch [1,1,K,N] -> K-row-major bank-sharded bfp8, N padded
    pf.start()                     # once, before the decode trace is captured
    y = pf(x, w)                   # inside the trace; y [1,1,32,N] interleaved (DRAM) bf16
    pf.stop()                      # before ttnn.close_device / before any op that needs the ring's L1

The caller must stop() a started prefetcher before it closes the device. When the device cannot run the
prefetcher (Blackhole firmware older than 19.12, or the streaming profiler on), ``supported`` is False:
``linear()`` then returns the plain DRAM-sharded linear and start()/stop() do nothing, so a model built with
the prefetcher still runs, without the speedup.

shortcut: one pipe set for all weights, sized to the largest K-block; per-weight rings are
the upgrade if L1 gets tight.
"""

import torch
from loguru import logger

import ttnn
from models.demos.inkling2_mini.tt.attention import _DramShardedLinear

TILE = 32
BFP8_TILE_BYTES = 1088


class PrefetchedLinear:
    def __init__(self, device, recv_rows=3, ring_kb=768, dtype=ttnn.bfloat8_b):
        self.device = device
        self.dtype = dtype
        self._running = False
        self.supported = ttnn.experimental.is_tensor_prefetcher_supported(device)
        if not self.supported:
            logger.warning("tensor prefetcher unsupported (needs Blackhole fw >= 19.12, no streaming profiler); "
                           "using the plain DRAM-sharded linear")  # fmt: skip
            return
        self.banks = device.dram_grid_size().x
        self.rows = recv_rows
        self.ring = self.banks * recv_rows
        self.ring_bytes = ring_kb * 1024
        self.recv = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.banks - 1, recv_rows - 1))}
        )

        # Bank b feeds ring positions b*R .. b*R+R-1, laid row-major on the banks x R grid, so shard
        # index == ring position (tests' bank_receivers_contiguous).
        def arc(b):
            cells = [divmod(b * recv_rows + s, self.banks) for s in range(recv_rows)]
            return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c, r), ttnn.CoreCoord(c, r)) for r, c in cells])

        bank_to_receivers = [(b, arc(b)) for b in range(self.banks)]
        # Carve the pipes before any program runs on the ring cores (the arena refuses sealed cores).
        # K-row-major weights need one sender pipe per bank (support_multi_receiver_shards=True).
        self._space = ttnn.experimental.create_prefetcher_pipe_space(
            device,
            sender_cores=ttnn.CoreRangeSet(set()),
            receiver_domain=self.recv,
            ring_size=self.ring_bytes,
            max_receivers_per_pipe=recv_rows,
            num_dram_senders=2 * self.banks,
        )
        self.pipes = ttnn.experimental.create_prefetcher_pipes_for_tensor_prefetcher(
            self._space, bank_to_receivers, support_multi_receiver_shards=True
        )
        self.ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def padded_n(self, n):
        m = TILE * self.ring
        return (n + m - 1) // m * m

    def weight(self, w):
        """torch [1,1,K,N] (or [K,N]) -> device weight; returns (tensor, N_true)."""
        w = w.float().reshape(1, 1, *w.shape[-2:])
        k, n = w.shape[-2:]
        npad = self.padded_n(n)
        if npad != n:
            w = torch.nn.functional.pad(w, (0, npad - n))
        # K-row-major: one WIDTH_SHARDED (K, N/banks) shard per DRAM bank
        banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.banks - 1, 0))})
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(banks, [k, npad // self.banks], ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.as_tensor(w, device=self.device, dtype=self.dtype, memory_config=mem, layout=ttnn.TILE_LAYOUT), n

    def _program(self, k, n):
        pcn = n // TILE // self.ring
        ibw = next(
            b for b in (8, 6, 4, 3, 2, 1) if (k // TILE) % b == 0 and 2 * b * pcn * BFP8_TILE_BYTES <= self.ring_bytes
        )
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(self.banks, self.rows),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=max(s for s in range(1, 5) if pcn % s == 0),
            out_block_h=1,
            out_block_w=pcn,
            per_core_M=1,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=self.rows,
            untilize_out=False,
            stream_in1=False,
        )

    def linear(self, w):
        """Callable with `_DramShardedLinear`'s signature (x, dtype, ckc, memory_config) for torch w [K, N]."""
        if not self.supported:
            return _DramShardedLinear(self.device, w, self.dtype)
        wn = self.weight(w)
        return lambda x, dtype, ckc=None, memory_config=None: self(x, wn, dtype=dtype)

    def start(self):
        if self.supported and not self._running:
            ttnn.experimental.start_tensor_prefetcher(self.device)
            self._running = True

    def stop(self):
        if self._running:
            ttnn.experimental.stop_tensor_prefetcher(self.device)
            ttnn.synchronize_device(self.device)
            self._running = False

    def __call__(self, x, wn, dtype=ttnn.bfloat16, interleaved=True):
        """x [1,1,32,K] bf16 interleaved; returns [1,1,32,N_true] (sliced if N was padded)."""
        w, n_true = wn
        k, n = w.shape[-2], w.shape[-1]
        out_mem = ttnn.create_sharded_memory_config(
            (TILE, n // self.ring),
            self.recv,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        y = ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
            x,
            w,
            prefetcher_pipes=self.pipes,
            program_config=self._program(k, n),
            memory_config=out_mem,
            compute_kernel_config=self.ckc,
            dtype=dtype,
        )
        if not interleaved:
            return y
        y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        if n_true != n:
            y = ttnn.slice(y, [0, 0, 0, 0], [1, 1, y.shape[-2], n_true])
        return y
