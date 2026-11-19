# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel building block: the decode DS matmul y[M <= 32, N] = x[M, K] @ W[K, N] as a generic_op program
over a DSLinear's DRAM width-sharded weight (read in place: no copy, no re-layout).

16 bank-reader cores (DSLinear.readers per bank, the DS placement query of the matmul factory), reader (bank b, index r)
owns output tiles [b * SHARD_W + r * NR, + NR) of the padded N. Per K block of KB tile rows the reader streams its
weight block (BRISC / NoC0, 3-deep ring, one transaction id per block), pulls the activation block (NCRISC / NoC1) and
accumulates matmul_block subblocks onto fp32 partials with packer L1 accumulation (TRISC, HiFi2 / fp32 DST as the
decode DS linears); the partials are packed to the output dtype after the last block (kernels/ds_mm/). In the
megakernel this is the phase body of P2 / P5 / P8 / P9 (the activation then comes from the previous phase's owner
cores and the output goes to the next phase's cores instead of tensors).
"""

from __future__ import annotations

import os

import ttnn

KERNEL_DIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "ds_mm")
TILE = 32
BF16, BF8, FP32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
L1 = ttnn.L1_MEMORY_CONFIG
CB = dict(IN0=0, W=1, PART=2, OUT=3)
TILE_BYTES = {BF16: 2048, BF8: 1088, FP32: 4096, ttnn.bfloat4_b: 576}
MAX_BURST = 16384


def _tiles_per_txn(nr: int, tb: int) -> int:
    best = 1
    for d in range(1, nr + 1):
        if nr % d == 0 and d * tb <= MAX_BURST:
            best = d
    return best


class DSMatmul:
    """y = x @ W for a DSLinear (its DRAM-sharded weight, padded N, readers per bank); call with x [1,1,M<=32,K]."""

    def __init__(self, device, ds, *, kb: int | None = None, fidelity=ttnn.MathFidelity.HiFi2, out_dtype=BF16,
                 readers: list | None = None):  # fmt: skip
        self.device, self.ds = device, ds
        w = ds.weight
        self.k, self.n_pad, self.n = ds.k, ds.n_pad, ds.n
        self.banks = device.dram_grid_size().x
        self.readers = ds.readers
        self.kt = self.k // TILE
        self.shard_w = self.n_pad // TILE // self.banks
        if self.shard_w % self.readers:
            raise ValueError(f"{ds.role}: shard width {self.shard_w} tiles not divisible by {self.readers} readers")
        self.nr = self.shard_w // self.readers
        self.kb = kb or ds.block_w
        if self.kt % self.kb:
            raise ValueError(f"{ds.role}: K block {self.kb} does not divide {self.kt} K tiles")
        self.nb = self.kt // self.kb
        self.w_dtype = w.dtype
        self.tb = TILE_BYTES[self.w_dtype]
        self.tpt = _tiles_per_txn(self.nr, self.tb)
        self.subw = max(s for s in (4, 2, 1) if self.nr % s == 0)
        self.fidelity, self.out_dtype = fidelity, out_dtype
        # tenstorrent/tt-metal: the bank-adjacent readers come from mk_readers.reader_assignments (the public optimal
        # bank -> worker assignment plus the nearest free cores), not the fork's DramReaderPlacement
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import reader_assignments

        self.assign = reader_assignments(device, self.readers, ds.ocfg.ds_placement)
        if readers is not None:  # explicit [(core, bank, index)] (megakernel: a rectangle, for multicast dispatch)
            self.assign = [(c, int(b), int(i)) for c, b, i in readers]
        xs = sorted({int(c.x) for c, _, _ in self.assign})
        ys = sorted({int(c.y) for c, _, _ in self.assign})
        if (
            len(xs) * len(ys) == len(self.assign)
            and xs == list(range(xs[0], xs[-1] + 1))
            and ys == list(range(ys[0], ys[-1] + 1))
        ):  # a full rectangle: one core range
            self.cores = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(xs[0], ys[0]), ttnn.CoreCoord(xs[-1], ys[-1]))]
            )
        else:
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges

            self.cores = merged_ranges([(c.x, c.y) for c, _, _ in self.assign])
        self.calls = 0

    def _rt_core(self, fn):
        rt = ttnn.RuntimeArgs()
        for c, bank, idx in self.assign:
            rt[c.x][c.y] = [int(v) for v in fn(c, bank, idx)]
        return rt

    def _rt(self, fn):
        rt = ttnn.RuntimeArgs()
        for c, bank, idx in self.assign:
            rt[c.x][c.y] = [int(v) for v in fn(bank, idx)]
        return rt

    def __call__(self, x: ttnn.Tensor, *, out_memory_config=L1) -> ttnn.Tensor:
        """x [1,1,M<=32,K] (tile layout, interleaved) -> y [1,1,M,N_pad] out_dtype (interleaved)."""
        y = self.output(x, out_memory_config)
        kernels, cbs = self.parts(x, y)
        ttnn.generic_op([x, self.ds.weight, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        self.calls += 1
        return y

    def output(self, x, out_memory_config=L1):
        m = int(x.shape[-2])
        if x.layout != ttnn.TILE_LAYOUT or int(x.shape[-1]) != self.k or m > TILE:
            raise ValueError(f"ds_mm {self.ds.role}: x {tuple(x.shape)} {x.layout}")
        return ttnn.empty((1, 1, m, self.n_pad), self.out_dtype, ttnn.TILE_LAYOUT, self.device, out_memory_config)

    def parts(self, x, y, *, cb_base: int = 0, wait=None, mk_probe=None):
        """(kernels, cbs) of the matmul on the reader cores (megakernel phase: cb_base shifts the CB indices,
        wait = (semaphore id, count) makes the activation mover wait for the producers' hand-off first)."""
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        defines = [("DSMM_CB_BASE", str(cb_base))] if cb_base else []
        mover_defines = list(defines)
        if wait is not None:
            mover_defines += [("DSMM_WAIT_SEM_ID", str(wait[0])), ("DSMM_WAIT_COUNT", str(wait[1]))]
        w_addr = self.ds.weight.buffer_address()
        wdef, mdef = list(defines), list(mover_defines)
        pidx = {(int(c.x), int(c.y)): i for i, (c, _, _) in enumerate(self.assign)}
        if mk_probe is not None:  # megakernel phase probe: reader i -> slots base + 2i (BRISC), base + 2i + 1 (NCRISC)
            wdef += [("MK_PROBE_ARG", "4"), ("MK_PROBE_CB", "31")]
            mdef += [("MK_PROBE_ARG", "3"), ("MK_PROBE_CB", "31")]
        pargs = lambda c, k: [mk_probe[0], mk_probe[1] + 2 * pidx[(int(c.x), int(c.y))] + k] if mk_probe else []
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        wreader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, "wreader.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores, compile_time_args=[self.nb, self.kb, self.nr, self.tb, self.tpt, self.shard_w],
            runtime_args=self._rt_core(lambda c, bank, idx: [w_addr, bank, idx, bank & 3] + pargs(c, 0)), defines=wdef,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
        )  # fmt: skip
        mover = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, "mover.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores, compile_time_args=[self.nb, self.kb, self.nr] + acc(x) + acc(y),
            runtime_args=self._rt_core(lambda c, bank, idx: [x.buffer_address(), y.buffer_address(),
                                                             bank * self.shard_w + idx * self.nr] + pargs(c, 1)),
            defines=mdef, config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
        )  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, math_approx_mode=False, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[cb_base + CB["PART"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, "compute.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores, compile_time_args=[self.nb, self.kb, self.nr, self.subw],
            runtime_args=ttnn.RuntimeArgs(), defines=defines, config=cc,
        )  # fmt: skip
        xt = TILE_BYTES[x.dtype]
        ot = TILE_BYTES[self.out_dtype]
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(
            total_size=page * n, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )  # fmt: skip
        b = cb_base
        cbs = [cb(b + CB["IN0"], x.dtype, xt, 2 * self.kb), cb(b + CB["W"], self.w_dtype, self.tb, 3 * self.kb * self.nr),
               cb(b + CB["PART"], FP32, 4096, self.nr), cb(b + CB["OUT"], self.out_dtype, ot, self.nr)]  # fmt: skip
        return [wreader, mover, compute], cbs

    def noc_coords(self):
        """(x, y) NoC coordinates of the reader cores (megakernel hand-off targets)."""
        out = []
        for c, _, _ in self.assign:
            w = self.device.worker_core_from_logical_core(c)
            out.append((int(w.x), int(w.y)))
        return out

    def l1_bytes(self, x_dtype=BF16) -> int:
        """CB bytes per reader core."""
        return (2 * self.kb * TILE_BYTES[x_dtype] + 3 * self.kb * self.nr * self.tb + self.nr * 4096
                + self.nr * TILE_BYTES[self.out_dtype])  # fmt: skip
