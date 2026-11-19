# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 rung 3, milestone M1: the drafter's matmuls as DS reader-phase programs (tt/mk_readers.ReaderPhases on
DSLinear DRAM-width-sharded weights, kernels/mk/reader_{w,a,c}.cpp: the Stage 14 layer megakernel's matmul phases)
instead of the 7a mm_stream programs (tt/mm_stream.py) -- `GLM53_DF2_MM=ds` (default `stream`: the rung-2 kind).

Why (dflash2_intrace.md §8): u3 measured ~72 us per traced launch of the rung-2 step (380 launches, +27.4 ms at TP2):
dispatch-bound on the per-launch command stream, whose size grows with the cores carrying per-core runtime args.
An mm_stream program runs on up to 128 cores (3 kernels, per-core args on two); a ReaderPhases program runs on the
8 x readers (16-24) bank-adjacent DS reader cores with the same arithmetic per output tile (ds_mm compute: K blocks of
`kb` tiles, packer-L1 fp32 accumulation, exact fp32 unpack, HiFi2 -- bitwise the mm_stream result when kb is equal).
Phases of one ReaderPhases program share ONE launch: the matmuls that take the SAME input are grouped (`DsGroup`):
the ctx program's 8 qkv projections of hidden_norm's output (9 -> 2 launches with fc) and the head's lm_head + hproj
of the final norm (2 -> 1). The per-layer chain (kp_attn -> conv -> qkv -> ... -> down) cannot share a program while
its glue is stock ops: a program's phases wait on semaphores only owner phases of the SAME program can raise, and a
command queue starts a program after the previous one completed -- the per-layer program is M2 (owner phases).

Geometry (DSLinear): the weight [K, N] bfloat8_b width-sharded over the 8 DRAM banks, `readers` cores per bank each
streaming its NR = N / 32 / 8 / readers column tiles in K blocks of `kb` tiles; N must be a multiple of 256 x readers
(else padded with zero columns and the output sliced: lm_head). Per-core L1 = the weight ring (2-3 blocks) + the
activation block + the fp32 partials + the output tiles (`ds_l1_bytes`): a matmul over the budget (GLM53_DF2_DS_L1,
default 1,100,000 B: the TP2 lm_head, 2,420 column tiles) stays an mm_stream program.
"""

from __future__ import annotations

import os

import torch

from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import RING_BYTES_MAX, RING_SLOTS, MMPhase, ReaderPhases

TILE = 32
BANKS = 8
BF16, FP32 = torch.bfloat16, torch.float32
TB8 = 1088  # bfloat8_b tile bytes
L1_BUDGET = int(os.environ.get("GLM53_DF2_DS_L1", "1100000"))


def ds_readers(n: int) -> int:
    """readers per bank for an N-column weight: the most (<= 3) whose 256 x readers divides N (no padding); 3 (padded)
    when none does"""
    for r in (3, 2, 1):
        if n % (TILE * BANKS * r) == 0:
            return r
    return 3


def ds_geometry(n: int, readers: int) -> dict:
    """n_pad, shard_w (column tiles per bank), rp (readers that take part), nr (column tiles per reader)"""
    q = TILE * BANKS * readers
    n_pad = -(-n // q) * q
    shard_w = n_pad // TILE // BANKS
    rp = max(r for r in range(1, readers + 1) if shard_w % r == 0)
    return dict(n_pad=n_pad, shard_w=shard_w, rp=rp, nr=shard_w // rp)


def ds_l1_bytes(n: int, kb: int, readers: int, out_fp32: bool) -> int:
    """mk_readers.ReaderPhases.l1_bytes for a one-phase program (+ the fp32 output CB of an fp32-output phase)"""
    g = ds_geometry(n, readers)
    nr = g["nr"]
    slots = RING_SLOTS if RING_SLOTS * kb * nr * TB8 <= RING_BYTES_MAX else 2
    return slots * kb * nr * TB8 + 2 * kb * 2048 + nr * (4096 + 2048) + (nr * 4096 if out_fp32 else 0)


class DsMM:
    """one drafter matmul y = x @ W as a DSLinear weight + a one-phase ReaderPhases program (one launch per call);
    parts = the per-chip host shards [K, N] (bf16 values; packed bfloat8_b on device); out = BF16 / FP32"""

    def __init__(self, ops, parts: list, *, out=BF16, name: str = "", kb: int = 8):
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import DSLinear, OptimizedDecoderConfig

        ttnn = ops.ttnn
        k, n = int(parts[0].shape[0]), int(parts[0].shape[1])
        r = ds_readers(n)
        g = ds_geometry(n, r)
        kt = k // TILE
        kb = max(d for d in range(1, kb + 1) if kt % d == 0)
        self.ops, self.name, self.k, self.n, self.n_pad, self.readers, self.kb, self.out = (
            ops,
            name,
            k,
            n,
            g["n_pad"],
            r,
            kb,
            out,
        )
        self.l1 = ds_l1_bytes(n, kb, r, out == FP32)
        pad = g["n_pad"] - n
        hosts = [
            ttnn.from_torch(torch.nn.functional.pad(p.to(BF16), (0, pad)) if pad else p.to(BF16),
                            dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT) for p in parts
        ]  # fmt: skip
        w = ops._mesh(hosts)  # [K, n_pad] interleaved; DSLinear re-lays it width-sharded over the banks (then freed)
        ocfg = OptimizedDecoderConfig()
        ocfg.ds_readers_per_role = f"{name}:{r}"
        self.ds = DSLinear(ops.mesh, w, role=name, cfg=ocfg, ckc=ops.hifi2, dtype=ops.dt[out])
        del w
        self.rp = ReaderPhases(ops.mesh, [MMPhase(self.ds, kb=kb)], readers_per_bank=r)
        assert self.rp.l1_bytes() + (g["nr"] * 4096 if out == FP32 else 0) == self.l1, (self.rp.l1_bytes(), self.l1)
        self.geo = g
        self.sm = None  # (no mm_stream program: the breakdown job's recorder reads this attribute)

    def raw(self, x):
        """the one-phase program (no launch count): the padded output [1, 1, 32, n_pad]"""
        return self.rp([x], out_memory_config=self.ops.DRAM)[0]

    def __call__(self, x):
        self.ops._tick(1, "ds_mm")
        return self.ops._unpad(self.raw(x), self.n)


class DsGroup:
    """several DsMM of the SAME input as the phases of ONE ReaderPhases program (one launch): g(x) -> [y_i]"""

    def __init__(self, ops, mms: list, *, name: str = ""):
        assert mms and all(isinstance(m, DsMM) for m in mms), [type(m) for m in mms]
        self.ops, self.mms, self.name = ops, mms, name
        r = max(m.readers for m in mms)
        self.rp = ReaderPhases(ops.mesh, [MMPhase(m.ds, kb=m.kb) for m in mms], readers_per_bank=r)
        self.l1 = self.rp.l1_bytes() + max((m.geo["nr"] * 4096 for m in mms if m.out == FP32), default=0)
        assert self.l1 <= L1_BUDGET, (name, self.l1, L1_BUDGET)
        self.sm = None

    def __call__(self, x):
        self.ops._tick(1, "ds_group")
        ys = self.rp([x] * len(self.mms), out_memory_config=self.ops.DRAM)
        return [self.ops._unpad(y, m.n) for y, m in zip(ys, self.mms)]


def ds_plan(parts_n: int, out, kb: int = 8) -> dict:
    """host arithmetic for the census / the doc: readers, n_pad, nr, cores, L1 bytes, fits"""
    r = ds_readers(parts_n)
    g = ds_geometry(parts_n, r)
    l1 = ds_l1_bytes(parts_n, kb, r, out == FP32)
    return dict(
        readers=r, cores=BANKS * r, active=BANKS * g["rp"], n_pad=g["n_pad"], nr=g["nr"], l1=l1, fits=l1 <= L1_BUDGET
    )
