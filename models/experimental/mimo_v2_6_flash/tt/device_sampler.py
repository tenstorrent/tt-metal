# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Temperature / top-k / top-p sampling of the decode users on device, inside the decode trace (vLLM mode).

vLLM's sampler (``apply_top_k_top_p``) keeps the top_k largest logits, then the smallest set of the most likely tokens
whose probability reaches top_p, and draws from it. Here, per user (rows 0..users-1 of the logit tile):

1. SampleScan: 32 cores per chip each keep the K largest fp32 logits of a range of the chip's vocab shard (the
   GreedyScan read pattern; exact values, integer keys);
2. SampleChipMerge: one core per user sorts those into the chip's K largest and writes the chip's largest logit;
3. each chip's sum of exp((logit - its largest) / T) over its shard (ttnn ops on the logit tile), the normalizer;
4. after two row all-gathers, SampleMerge walks the row's candidates (the chips' lists, merged) in descending
   order: probabilities are softmax(logits / T) over the vocabulary, or over the first top_k candidates when
   top_k is set, as vLLM normalizes; the nucleus keeps a candidate while the probability ranked above it is below
   top_p; the token is drawn by inverse CDF with one uniform per row (``set_uniforms``, from the host every
   sampled step). A greedy row (temperature 0 or top_k 1) takes the first candidate: the lowest id among the
   largest logits, as GreedyHead and the host's argmax.

The candidates hold the row's K most likely tokens and, past that rank, each chip's K most likely, so the draw equals
vLLM's whenever the nucleus has at most K tokens (99.7% of the decode rows of GPQA reasoning at temperature 1,
top_p 0.95; 99.9% fit in the tp * K candidates) or lies inside the candidates; otherwise it is the nucleus's tokens
among the candidates. top_k beyond tp * K is not supported (``max_device_top_k``). Seeded requests and requests for
logprobs stay on host sampling (the plugin routes them).
"""

import numpy as np
import torch

import ttnn

from .common import TILE

KERNEL_DIR = "models/experimental/mimo_v2_6_flash/tt/kernels"
SCAN_CORES = 32
SCRATCH_CB = 0
CANDIDATES_PER_CHIP = 32  # K: each scan core's and each chip's list per user


def _is_dram(t):
    return int(t.memory_config().buffer_type == ttnn.BufferType.DRAM)


class DeviceSampler:
    def __init__(self, plan, vocab_local, users, max_batch, scan_cores=SCAN_CORES):
        assert vocab_local % TILE == 0 and 1 <= users <= 4 and max_batch * 4 <= 1024
        self.plan, self.vocab_local, self.users, self.max_batch = plan, vocab_local, users, max_batch
        self.k = CANDIDATES_PER_CHIP
        self.candidates = self.k * plan.tp
        device = plan.mesh_device
        grid = device.compute_with_storage_grid_size()
        assert scan_cores % 4 == 0 and scan_cores <= grid.x * grid.y
        self.num_scan = scan_cores
        self.scan_cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(scan_cores)]
        self.merge_cores = self.scan_cores[:users]
        self.final_core = ttnn.CoreCoord(0, 0)
        tiles = vocab_local // TILE
        bounds = [tiles * i // scan_cores for i in range(scan_cores + 1)]
        self.ranges = list(zip(bounds[:-1], bounds[1:]))
        self.max_tiles = max(e - b for b, e in self.ranges)
        self.tiles = tiles
        rep = plan.replicate()

        def persistent(shape, dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, value=None):
            torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.int32
            t = torch.zeros(shape, dtype=torch_dtype) if value is None else value
            return ttnn.from_torch(
                t, device=device, dtype=dtype, layout=layout, memory_config=memory_config, mesh_mapper=rep
            )

        # Scan partials: 4 user slots of K (key, index) pairs per scan core, as raw 4096-byte pages.
        partial_pages = scan_cores * 4 * self.k * 8 // 4096
        self.partials = persistent((1, 1, TILE, TILE * partial_pages), ttnn.uint32)
        self.chip_cands = persistent((1, 1, TILE, TILE), ttnn.uint32)
        self.chip_max = persistent((1, 1, TILE, 1), ttnn.float32)
        # Per-row parameters, rewritten in place (``set_params``); rows default to greedy.
        self.inv_temp = persistent((1, 1, TILE, 1), ttnn.float32, value=torch.ones(1, 1, TILE, 1))
        dram_rm = {"layout": ttnn.ROW_MAJOR_LAYOUT, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
        default = torch.tensor([0, 1 << 24, 0, 1], dtype=torch.int32).repeat(max_batch).reshape(1, -1)
        self.params = persistent((1, max_batch * 4), ttnn.uint32, value=default, **dram_rm)
        self.uniform = persistent((1, max_batch), ttnn.uint32, **dram_rm)
        self.widen = persistent((1, 1, 1, TILE), ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # The tokens of the last step, persistent (the trace writes them; reads and the feedback use them).
        self.next_tokens = persistent((1, max_batch), ttnn.uint32, **dram_rm)

    # ------------------------------------------------------------------ host side
    def host_params(self, temperature, top_k, top_p):
        """torch [rows] each (rows <= max_batch; the rest greedy) -> (host tensors for ``set_params``, whether any row
        samples). top_k < 1 or at least the vocabulary means no top_k; top_k 1 or temperature 0 is greedy."""
        rows = temperature.numel()
        temp = torch.zeros(self.max_batch)
        temp[:rows] = temperature.float()
        k = torch.zeros(self.max_batch)
        kk = top_k.float()
        k[:rows] = torch.where((kk >= 1) & (kk < self.vocab_local * self.plan.tp), kk, torch.zeros_like(kk))
        p = torch.ones(self.max_batch)
        p[:rows] = top_p.float().clamp(max=1.0)
        greedy = (temp == 0) | (k == 1)
        inv = torch.where(greedy, torch.ones_like(temp), 1.0 / temp.double().clamp(min=1e-2).float())
        # SampleMerge's fixed-point inputs: log2(e) / T as fp32 bits, top_p in Q24, top_k, greedy.
        scale = (inv.double() * 1.4426950408889634).float().numpy().view(np.int32)
        params = torch.stack(
            [
                torch.from_numpy(scale.copy()),
                (p.double() * (1 << 24)).round().to(torch.int32),
                k.to(torch.int32),
                greedy.to(torch.int32),
            ],
            dim=1,
        ).reshape(1, -1)
        rep = self.plan.replicate()
        hosts = [
            ttnn.from_torch(inv.reshape(1, 1, -1, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep),
            ttnn.from_torch(params, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep),
        ]
        return hosts, bool((~greedy[:rows]).any())

    def set_params(self, hosts):
        ttnn.copy_host_to_device_tensor(hosts[0], self.inv_temp)
        ttnn.copy_host_to_device_tensor(hosts[1], self.params)

    def set_uniforms(self, uniforms):
        """uniforms: torch [max_batch] in [0, 1), one draw per row for the next step (Q24 on device)."""
        h = ttnn.from_torch(
            (uniforms.double() * (1 << 24)).floor().clamp(max=(1 << 24) - 1).to(torch.int32).reshape(1, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.plan.replicate(),
        )
        ttnn.copy_host_to_device_tensor(h, self.uniform)

    # ------------------------------------------------------------------ device side (trace-capturable)
    @staticmethod
    def _program(kernel, cores, compile_args, runtime_args, scratch_bytes):
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        cb = ttnn.CBDescriptor(
            total_size=scratch_bytes,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(SCRATCH_CB, ttnn.uint32, scratch_bytes)],
        )
        rt = ttnn.RuntimeArgs()
        for core, args in zip(cores, runtime_args):
            rt[core.x][core.y] = args
        desc = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/{kernel}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=compile_args,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(kernels=[desc], semaphores=[], cbs=[cb])

    def _scan(self, logits):
        elem_bytes = 4 if logits.dtype == ttnn.float32 else 2
        args = [
            [logits.buffer_address(), self.partials.buffer_address(), b, e, i] for i, (b, e) in enumerate(self.ranges)
        ]
        scratch = self.max_tiles * 2 * self.users * 16 * elem_bytes + 4 * self.k * 8
        prog = self._program(
            "sample_scan.cpp",
            self.scan_cores,
            [self.users, SCRATCH_CB, self.max_tiles, elem_bytes, self.k],
            args,
            scratch,
        )
        ttnn.generic_op([logits, self.partials], prog)

    def _chip_merge(self):
        addrs = [self.partials.buffer_address(), self.chip_cands.buffer_address(), self.chip_max.buffer_address()]
        args = [addrs + [u] for u in range(self.users)]
        scratch = self.num_scan * self.k * 8 + self.k * 8 + 64
        prog = self._program(
            "sample_chip_merge.cpp", self.merge_cores, [self.num_scan, self.k, SCRATCH_CB], args, scratch
        )
        ttnn.generic_op([self.partials, self.chip_cands, self.chip_max], prog)

    def _normalizer(self, logits):
        """Each row's sum of exp((logit - chip max) / T) over this chip's shard, in every column of a [1, 1, 32, 32]
        fp32 tile (a one-column tensor does not all-gather intact)."""
        shifted = ttnn.multiply(ttnn.subtract(logits, self.chip_max), self.inv_temp)
        e = ttnn.exp(shifted)
        shifted.deallocate(True)
        col = ttnn.sum(e, dim=-1, keepdim=True)
        e.deallocate(True)
        out = ttnn.add(col, self.widen)
        col.deallocate(True)
        return out

    def _merge(self, cands, sums, token_input):
        tp = self.plan.tp
        args = [
            cands.buffer_address(),
            _is_dram(cands),
            sums.buffer_address(),
            _is_dram(sums),
            self.params.buffer_address(),
            self.uniform.buffer_address(),
            self.next_tokens.buffer_address(),
            _is_dram(self.next_tokens),
            token_input.buffer_address() if token_input is not None else 0,
            _is_dram(token_input) if token_input is not None else 0,
        ]
        scratch = tp * self.users * self.k * 8 + tp * 256 + self.max_batch * 16 + 1024 + 2 * self.candidates * 4
        scratch += self.max_batch * 4
        compile_args = [self.users, tp, self.k, self.vocab_local, self.max_batch, SCRATCH_CB]
        prog = self._program("sample_merge.cpp", [self.final_core], compile_args, [args], scratch)
        io = [cands, sums, self.params, self.uniform, self.next_tokens] + (
            [token_input] if token_input is not None else []
        )
        ttnn.generic_op(io, prog)

    def __call__(self, logits, token_input=None):
        """logits [1, 1, 32, vocab_local] bf16 or fp32 TILE (DRAM interleaved; users in rows 0..users-1).

        Returns the persistent ``next_tokens`` [1, max_batch] uint32 row-major (zeros past ``users``); also writes it
        into ``token_input`` (a [1, max_batch] uint32 row-major tensor) when given."""
        assert logits.layout == ttnn.TILE_LAYOUT and logits.dtype in (ttnn.bfloat16, ttnn.float32) and _is_dram(logits)
        assert logits.shape[-1] == self.vocab_local and logits.shape[-2] == TILE
        self._scan(logits)
        self._chip_merge()
        sums = self._normalizer(logits)
        cands = self.plan.all_gather(self.chip_cands, dim=-1)
        sums_all = self.plan.all_gather(sums, dim=-1)
        sums.deallocate(True)
        self._merge(cands, sums_all, token_input)
        cands.deallocate(True)
        sums_all.deallocate(True)
        return self.next_tokens
