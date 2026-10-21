# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Temperature / top-k / top-p sampling of the decode users on device, inside the decode trace (vLLM mode).

vLLM's sampler (``apply_top_k_top_p``) keeps the top_k largest logits, then the smallest set of the most likely tokens
whose probability reaches top_p, and draws from it. Here, per user (rows 0..users-1 of the logit tile):

1. SampleScan, on 32 cores per chip, each over a range of the chip's vocab shard: the compute kernel sums
   exp((logit - the range's largest) / T) over the range in fp32 (the SFPU's exact exp) while the reader and the
   writer keep two users' K largest fp32 logits each (exact values, integer keys); the writer then adds up the sums
   per user in fixed point;
2. SampleChipMerge, one core per user: the chip's K largest, sorted, and its normalizer, the sum of
   exp((logit - the chip's largest) / T) over the shard;
3. after one row all-gather, SampleMerge walks the row's candidates (the chips' lists, merged) in descending
   order: probabilities are softmax(logits / T) over the vocabulary, or over the first top_k candidates when
   top_k is set, as vLLM normalizes; the nucleus keeps a candidate while the probability ranked above it is below
   top_p; the token is drawn by inverse CDF with one uniform per row from a splitmix64 stream on device (its state
   persistent, seeded by the host with the parameters), so a sampled step copies nothing either. A greedy row (temperature 0 or top_k 1) takes the first candidate: the lowest id among the
   largest logits, as GreedyHead and the host's argmax.

The data movement cores have no FPU: steps 2 and 3 run in fixed point (kernels/sample_fixed.hpp), probabilities to
~1e-9 of the largest. The candidates hold the row's K most likely tokens and, past that rank, each chip's K most
likely, so the draw equals vLLM's whenever the nucleus has at most K tokens (99.7% of the decode rows of GPQA
reasoning at temperature 1, top_p 0.95; 99.9% fit in the tp * K candidates) or lies inside the candidates;
otherwise it is the nucleus's tokens among the candidates. top_k beyond tp * K is not supported
(``max_device_top_k``). Seeded requests and requests for logprobs stay on host sampling (the plugin routes them).
"""

import numpy as np
import torch

import ttnn

from .common import TILE

KERNEL_DIR = "models/experimental/mimo_v2_6_flash/tt/kernels"
SCAN_CORES = 32
CANDIDATES_PER_CHIP = 32  # K: each scan core's and each chip's list per user
PARAM_WORDS = 8  # per row: log2(e) / T (fp32 bits), top_p (Q24), top_k, greedy, 1 / T (fp32 bits), unused
# Circular buffers: the scan's logit tiles, its shift and scale tiles, the compute kernel's sum tile, the reader's and
# writer's scratch, the tiles' address from the reader to the writer.
CB_X, CB_M, CB_A, CB_S, CB_SCRATCH, CB_OUT, CB_READY = 0, 1, 2, 3, 4, 5, 6


def _is_dram(t):
    return int(t.memory_config().buffer_type == ttnn.BufferType.DRAM)


def _cb(index, nbytes, cores, dtype=ttnn.uint32, page=None):
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page or nbytes)
    return ttnn.CBDescriptor(total_size=nbytes, core_ranges=cores, format_descriptors=[fmt])


def _kernel(name, cores, crs, compile_args, runtime_args, config):
    rt = ttnn.RuntimeArgs()
    for core, args in zip(cores, runtime_args):
        rt[core.x][core.y] = args
    return ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=compile_args,
        runtime_args=rt,
        config=config,
    )


def _ranges(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


class DeviceSampler:
    def __init__(self, plan, vocab_local, users, max_batch, scan_cores=SCAN_CORES):
        assert vocab_local % TILE == 0 and 1 <= users <= 4 and max_batch * 4 <= 1024
        self.plan, self.vocab_local, self.users, self.max_batch = plan, vocab_local, users, max_batch
        self.k = CANDIDATES_PER_CHIP
        self.candidates = self.k * plan.tp
        device = plan.mesh_device
        grid = device.compute_with_storage_grid_size()
        assert scan_cores % 4 == 0 and scan_cores <= min(grid.x * grid.y, 128)
        self.num_scan = scan_cores
        self.scan_cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(scan_cores)]
        self.merge_cores = self.scan_cores[:users]
        self.final_core = ttnn.CoreCoord(0, 0)
        tiles = vocab_local // TILE
        bounds = [tiles * i // scan_cores for i in range(scan_cores + 1)]
        self.ranges = list(zip(bounds[:-1], bounds[1:]))
        self.max_tiles = max(e - b for b, e in self.ranges)
        rep = plan.replicate()

        def persistent(shape, dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, value=None):
            torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.int32
            t = torch.zeros(shape, dtype=torch_dtype) if value is None else value
            return ttnn.from_torch(
                t, device=device, dtype=dtype, layout=layout, memory_config=memory_config, mesh_mapper=rep
            )

        # Scan partials: 4 user slots of K (key, index) pairs per scan core, and 4 uint64 sums per scan core, as raw
        # 4096-byte pages. The chip candidates page: the users' sorted lists, then their normalizers at byte 2048.
        partial_pages = scan_cores * 4 * self.k * 8 // 4096
        self.partials = persistent((1, 1, TILE, TILE * partial_pages), ttnn.uint32)
        self.sums = persistent((1, 1, TILE, TILE), ttnn.uint32)
        self.chip_cands = persistent((1, 1, TILE, TILE), ttnn.uint32)
        # Per-row parameters, rewritten in place (``set_params``); rows default to greedy.
        dram_rm = {"layout": ttnn.ROW_MAJOR_LAYOUT, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
        self.params = persistent(
            (1, max_batch * PARAM_WORDS), ttnn.uint32, value=self._params(*self._greedy()), **dram_rm
        )
        self.rng = persistent((1, max_batch * 2), ttnn.uint32, **dram_rm)
        # The tokens of the last step, persistent (the trace writes them; reads and the feedback use them).
        self.next_tokens = persistent((1, max_batch), ttnn.uint32, **dram_rm)
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (CB_X, CB_M, CB_A):
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        self.compute_config.unpack_to_dest_mode = modes

    # ------------------------------------------------------------------ host side
    def _greedy(self):
        n = self.max_batch
        return torch.zeros(n), torch.zeros(n), torch.ones(n)

    def _params(self, temperature, top_k, top_p):
        """int32 [1, max_batch * PARAM_WORDS] of per-row temperature, top_k (0: none), top_p (each [max_batch])."""
        greedy = (temperature == 0) | (top_k == 1)
        inv = torch.where(greedy, torch.ones_like(temperature), 1.0 / temperature.double().clamp(min=1e-2).float())
        scale = (inv.double() * 1.4426950408889634).float()
        zeros = torch.zeros(self.max_batch, dtype=torch.int32)
        words = [
            torch.from_numpy(scale.numpy().view(np.int32).copy()),
            (top_p.double().clamp(max=1.0) * (1 << 24)).round().to(torch.int32),
            top_k.to(torch.int32),
            greedy.to(torch.int32),
            torch.from_numpy(inv.float().numpy().view(np.int32).copy()),
            zeros,
            zeros,
            zeros,
        ]
        return torch.stack(words, dim=1).reshape(1, -1)

    def host_params(self, temperature, top_k, top_p, seeds=None):
        """torch [rows] each (rows <= max_batch; the rest greedy) -> (host tensors for ``set_params``, whether any row
        samples). top_k < 1 or at least the vocabulary means no top_k; top_k 1 or temperature 0 is greedy. ``seeds``:
        int64 [max_batch] splitmix64 states for the rows' draws (default: from torch's generator)."""
        rows = temperature.numel()
        temp, k, p = self._greedy()
        temp[:rows] = temperature.float()
        kk = top_k.float()
        k[:rows] = torch.where((kk >= 1) & (kk < self.vocab_local * self.plan.tp), kk, torch.zeros_like(kk))
        p[:rows] = top_p.float()
        greedy = (temp == 0) | (k == 1)
        if seeds is None:
            seeds = torch.randint(-(2**63), 2**63 - 1, (self.max_batch,), dtype=torch.int64)
        words = seeds.numpy().astype(np.int64).view(np.int32).reshape(1, -1)  # low word first
        rep = self.plan.replicate()
        hosts = [
            ttnn.from_torch(self._params(temp, k, p), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep),
            ttnn.from_torch(
                torch.from_numpy(words.copy()), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
            ),
        ]
        return hosts, bool((~greedy[:rows]).any())

    def set_params(self, hosts):
        ttnn.copy_host_to_device_tensor(hosts[0], self.params)
        ttnn.copy_host_to_device_tensor(hosts[1], self.rng)

    # ------------------------------------------------------------------ device side (trace-capturable)
    def _scan(self, logits):
        cores, crs = self.scan_cores, _ranges(self.scan_cores)
        params_bytes = self.max_batch * PARAM_WORDS * 4
        block_bytes = 4 * self.k * 8
        reader_args = [
            [logits.buffer_address(), self.partials.buffer_address(), self.params.buffer_address(), b, e, i]
            for i, (b, e) in enumerate(self.ranges)
        ]
        reader = _kernel(
            "sample_scan.cpp",
            cores,
            crs,
            [self.users, self.k, self.max_batch, CB_X, CB_M, CB_A, CB_SCRATCH, CB_READY],
            reader_args,
            ttnn.ReaderConfigDescriptor(),
        )
        compute = _kernel(
            "sample_norm_compute.cpp",
            cores,
            crs,
            [CB_X, CB_M, CB_A, CB_S],
            [[e - b] for b, e in self.ranges],
            self.compute_config,
        )
        writer = _kernel(
            "sample_scan_writer.cpp",
            cores,
            crs,
            [self.users, self.k, CB_S, CB_OUT, CB_READY],
            [
                [self.partials.buffer_address(), self.sums.buffer_address(), b, e, i]
                for i, (b, e) in enumerate(self.ranges)
            ],
            ttnn.WriterConfigDescriptor(),
        )
        tile = 4096
        cbs = [
            _cb(CB_X, self.max_tiles * tile, crs, ttnn.float32, tile),
            _cb(CB_M, tile, crs, ttnn.float32),
            _cb(CB_A, tile, crs, ttnn.float32),
            _cb(CB_S, tile, crs, ttnn.float32),
            _cb(CB_SCRATCH, block_bytes // 2 + params_bytes, crs),
            _cb(CB_OUT, block_bytes // 2 + 32, crs),
            _cb(CB_READY, 16, crs),
        ]
        prog = ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=cbs)
        ttnn.generic_op([logits, self.partials, self.sums, self.params], prog)

    def _chip_merge(self):
        cores, crs = self.merge_cores, _ranges(self.merge_cores)
        addrs = [
            self.partials.buffer_address(),
            self.sums.buffer_address(),
            self.params.buffer_address(),
            self.chip_cands.buffer_address(),
        ]
        scratch = self.num_scan * (self.k * 8 + 32) + self.max_batch * PARAM_WORDS * 4 + self.k * 8 + 16
        kernel = _kernel(
            "sample_chip_merge.cpp",
            cores,
            crs,
            [self.num_scan, self.k, CB_SCRATCH, self.max_batch],
            [addrs + [u] for u in range(self.users)],
            ttnn.ReaderConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, crs)])
        ttnn.generic_op([self.partials, self.sums, self.params, self.chip_cands], prog)

    def _merge(self, cands, token_input):
        tp = self.plan.tp
        cores, crs = [self.final_core], _ranges([self.final_core])
        args = [
            cands.buffer_address(),
            _is_dram(cands),
            self.params.buffer_address(),
            self.rng.buffer_address(),
            self.next_tokens.buffer_address(),
            _is_dram(self.next_tokens),
            token_input.buffer_address() if token_input is not None else 0,
            _is_dram(token_input) if token_input is not None else 0,
        ]
        scratch = tp * (self.users * self.k * 8 + 64) + self.max_batch * (PARAM_WORDS * 4 + 8)
        scratch += 2 * self.candidates * 4 + self.max_batch * 4
        kernel = _kernel(
            "sample_merge.cpp",
            cores,
            crs,
            [self.users, tp, self.k, self.vocab_local, self.max_batch, CB_SCRATCH],
            [args],
            ttnn.ReaderConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, crs)])
        io = [cands, self.params, self.rng, self.next_tokens] + ([token_input] if token_input is not None else [])
        ttnn.generic_op(io, prog)

    def __call__(self, logits, token_input=None):
        """logits [1, 1, 32, vocab_local] fp32 TILE (DRAM interleaved; users in rows 0..users-1).

        Returns the persistent ``next_tokens`` [1, max_batch] uint32 row-major (zeros past ``users``); also writes it
        into ``token_input`` (a [1, max_batch] uint32 row-major tensor) when given."""
        assert logits.layout == ttnn.TILE_LAYOUT and logits.dtype == ttnn.float32 and _is_dram(logits)
        assert logits.shape[-1] == self.vocab_local and logits.shape[-2] == TILE
        self._scan(logits)
        self._chip_merge()
        cands = self.plan.all_gather(self.chip_cands, dim=-1)
        self._merge(cands, token_input)
        cands.deallocate(True)
        return self.next_tokens
