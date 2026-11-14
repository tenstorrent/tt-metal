# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview skinny (M = 1..16 rows) bank-local streaming kernels, JIT only (no C++ op change), default OFF: HY4_SKINNY=shared,router,oproj.

Derived from the DSV4 bank-local chain (deepseek_v41_flash/tt/skinny_shared.py + skinny_kernels, imported for the helpers, not edited):
one worker core per DRAM bank (8 per chip), one L1 ring per core fed by the blaze DRAMStreamingMatmul reader with 12-15-tile pages, custom_mm (LoFi,
fp32 dest accumulation) on tiny 8x32 in0 tiles, intermediates exchanged over the NoC.

SkinnyShared: Hy4 shared expert partial per mesh column c (TP8): partial_c = (silu(min(x w0_c, L)) * clamp(x w1_c, -L, L)) w2_c,
hidden 6144 (192 tiles), 256 intermediate columns per mesh column = 8 tiles: bank core b owns the gate tile b and the up tile b (K = 192 each),
all-gathers act_b (8 tiles) to every core and computes the down columns [24 b, 24 b + 24) (K = 8).  Per core 384 + 192 bfp8 tiles (627 KB), 5.0 MB per chip.
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import skinny as SK
from models.demos.blackhole.deepseek_v41_flash.tt.skinny import NB, TB8, _acc, _core_set, _hash
from models.demos.blackhole.deepseek_v41_flash.tt.skinny_shared import _alias_cb, _fb, bank_seq_weight

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skinny_kernels")


def enabled(name):
    """HY4_SKINNY=shared,router,oproj (or all)"""
    v = os.environ.get("HY4_SKINNY", "")
    s = {x.strip() for x in v.split(",") if x.strip()}
    return name in s or "all" in s


def bank_cores(md, noc=0):
    """worker core of each DRAM bank; noc=1: assignment optimal for NOC1 (the default NCRISC reader NoC), noc=0: NOC0 (what the DSV4 chain used)."""
    n = ttnn.NOC.RISCV_1_default if noc else ttnn.NOC.RISCV_0_default
    return list(ttnn.get_optimal_dram_bank_to_logical_worker_assignment(md, n))


# banks 0 and 4 sit on logical cores (0,9) / (6,9): on 5 of the 8 chips of .46 slot 0 these two read DRAM at 41 GB/s instead of 59 (profile, skinny-kernels/logs/run_sh5.log);
# the neighbouring column (1,9) / (7,9) reads at full speed on all 8 chips.  Per-host/harvest dependent: re-check with HY4_SKINNY_REMAP="" (the NoC0-optimal assignment) on a new host.
DEFAULT_REMAP = "0@1.9/4@7.9"


class SkinnyShared:
    KT = 192  # hidden tiles (6144)
    KA = 8  # intermediate tiles per mesh column (256)
    NCOL = 24  # down columns (tiles) per bank core: 192 / 8

    def __init__(self, md, w0, w1, w2, limit=10.0, sbk1=None, sbk2=None, ring=None, xsplit=None, approx=None, fp32=None, banknoc=None, remap=None, tiny=None, act8=None):
        """w0 (gate), w1 (up): [1,1,6144,2048]; w2 (down): [1,1,2048,6144] host tensors ([in, out])."""
        self.md = md
        rows, cols = tuple(md.shape)
        assert cols == 8 and w0.shape[-1] == 2048 and w0.shape[-2] == 6144
        self.limit = float(limit)
        ic = 256
        w0 = w0.float().reshape(6144, 2048)
        w1 = w1.float().reshape(6144, 2048)
        w2 = w2.float().reshape(2048, 6144)
        tl = lambda m, k, n: m[32 * k : 32 * k + 32, 32 * n : 32 * n + 32]  # tile (k, n)
        w01 = torch.zeros(cols, NB, 2 * self.KT, 32, 32)
        w2t = torch.zeros(cols, NB, self.NCOL * self.KA, 32, 32)
        for c in range(cols):
            g, u, d = w0[:, ic * c : ic * (c + 1)], w1[:, ic * c : ic * (c + 1)], w2[ic * c : ic * (c + 1)]
            for b in range(NB):
                w01[c, b] = torch.stack([tl(g, k, b) for k in range(self.KT)] + [tl(u, k, b) for k in range(self.KT)])
                w2t[c, b] = torch.stack([tl(d, k, self.NCOL * b + j) for j in range(self.NCOL) for k in range(self.KA)])
        self.w01 = bank_seq_weight(md, w01)
        self.w2 = bank_seq_weight(md, w2t)
        self.tiny = int(os.environ.get("HY4_SKINNY_TINY", "1") if tiny is None else tiny)  # in0 tiles with R = 1/2/4/8 rows (x and act exchange 8/R times smaller)
        self.act8 = int(os.environ.get("HY4_SKINNY_ACT8", "0") if act8 is None else act8)
        self.banknoc = int(os.environ.get("HY4_SKINNY_BANKNOC", "0") if banknoc is None else banknoc)
        self.cores = bank_cores(md, self.banknoc)
        for ent in filter(None, (remap if remap is not None else os.environ.get("HY4_SKINNY_REMAP", DEFAULT_REMAP)).split("/")):  # "bank@x.y/bank@x.y": move a bank's worker core
            b, xy = ent.split("@")
            x, y = xy.split(".")
            self.cores[int(b)] = ttnn.CoreCoord(int(x), int(y))
        self.vcs = SK._vcs(self.cores)
        self.phys = [md.worker_core_from_logical_core(ttnn.CoreCoord(c.x, c.y)) for c in self.cores]
        self.fp32 = int(os.environ.get("HY4_SKINNY_FP32", "1") if fp32 is None else fp32)
        self.sbk1 = int(sbk1 or os.environ.get("HY4_SKINNY_SBK1", "48"))  # K tiles per reader block / compute sub-block, phase A
        self.sbk2 = int(sbk2 or os.environ.get("HY4_SKINNY_SBK2", "48"))  # tiles per reader block, phase C (6 columns of 8)
        self.ring = int(ring or os.environ.get("HY4_SKINNY_RING", "384"))
        self.xsplit = int(os.environ.get("HY4_SKINNY_XSPLIT", "1") if xsplit is None else xsplit)
        self.approx = int(os.environ.get("HY4_SKINNY_APPROX", "1") if approx is None else approx)
        assert self.KT % self.sbk1 == 0 and self.sbk2 % self.KA == 0 and (self.NCOL * self.KA) % self.sbk2 == 0
        assert self.ring // self.sbk1 <= 15 and self.ring // self.sbk2 <= 15 and self.ring % self.sbk1 == 0 and self.ring % self.sbk2 == 0

    def __call__(self, h, T, out=None):
        """h: std tile tensor [1,1,T,6144] bf16 (DRAM or L1); -> out [1,1,32,6144] bf16 TILE (rows 0..T-1 valid)."""
        assert T <= 16
        MT = 1 if T <= 8 else 2
        R = 8 if (MT == 2 or not self.tiny) else (2 if T <= 2 else 4 if T <= 4 else 8)  # rows of the in0 tiles
        RA = 8 if self.act8 else R
        md = self.md
        cores = self.cores
        core_set = _core_set(cores)
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 32, 6144]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
            )
        ring, sbk1, sbk2 = self.ring, self.sbk1, self.sbk2
        (CB_X, CB_W, CB_GU, CB_GU16, CB_ACTL, CB_ACTL16, CB_ACT, CB_O, CB_ZERO, CB_ZERO16, CB_TMP, CB_TMP16) = range(12)
        cbs = [
            SK._tiny_cb(core_set, CB_X, self.KT * MT, rows=R),
            SK._ring_cb(core_set, CB_W, ring),
            _alias_cb(core_set, CB_GU, CB_GU16, 2 * MT),
            _alias_cb(core_set, CB_ACTL, CB_ACTL16, MT),
            SK._tiny_cb(core_set, CB_ACT, self.KA * MT, rows=RA),
            SK._tiny_cb(core_set, CB_O, self.NCOL * MT),
            _alias_cb(core_set, CB_ZERO, CB_ZERO16, 1),
            _alias_cb(core_set, CB_TMP, CB_TMP16, 2),
        ]
        nsub1 = self.KT // sbk1
        readers = []
        for i, cc in enumerate(cores):
            ph = [
                dict(addr=self.w01.buffer_address(), sbk=sbk1, pn=2, nsub=nsub1, nbuf=ring // sbk1),
                dict(pn=0),
                dict(
                    addr=self.w2.buffer_address(),
                    sbk=sbk2,
                    pn=self.NCOL * self.KA // sbk2,
                    nsub=1,
                    nbuf=ring // sbk2,
                ),
            ]
            readers.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/hsk_reader.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=_core_set([cc]),
                    compile_time_args=SK.reader_ct(CB_W, i, self.vcs[i], ph),
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=[],
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
        SBWC = (4 if self.fp32 else 5) // MT
        while self.NCOL % SBWC:
            SBWC -= 1
        rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, cc in enumerate(cores):
            rt_w[cc.x][cc.y] = [i, i * self.NCOL]
            rt_c[cc.x][cc.y] = [i]
        common_w = [h.buffer_address(), out.buffer_address()]
        for p in self.phys:
            common_w += [p.x, p.y]
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/hsk_sh_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[CB_X, CB_ZERO, CB_ACTL, CB_ACT, CB_O, NB, self.KT, self.NCOL, SBWC, self.KA, self.xsplit, MT, R, RA]
            + _acc(h)
            + _acc(out),
            runtime_args=rt_w,
            common_runtime_args=common_w,
            config=ttnn.WriterConfigDescriptor(),
        )
        L = self.limit
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/hsk_sh_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[
                CB_X, CB_W, CB_GU, CB_GU16, CB_ACTL, CB_ACTL16, CB_ACT, CB_O, CB_ZERO, CB_ZERO16, CB_TMP, CB_TMP16,
                self.KT, self.fp32, _fb(L), _fb(-L), _fb(-3.0e38), sbk1, nsub1, self.KA, self.NCOL, SBWC, MT,
            ],
            runtime_args=rt_c,
            common_runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=bool(self.fp32),
                math_approx_mode=bool(self.approx),
            ),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=readers + [writer, compute],
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(3)],
            cbs=cbs,
        )
        prog.custom_program_hash = _hash(
            0x7B1, MT, R, RA, self.xsplit, self.approx, self.fp32, ring, sbk1, sbk2, self.w01.buffer_address(),
            self.w2.buffer_address(), tuple(_acc(h)), tuple(_acc(out)),
        )
        ttnn.generic_op([h, self.w01, self.w2, out], prog)
        return out
