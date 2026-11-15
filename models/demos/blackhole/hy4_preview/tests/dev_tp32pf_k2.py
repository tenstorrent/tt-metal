# SPDX-License-Identifier: Apache-2.0
"""tp32-prefill-build K2 (v1 W2): down grouped kernel with variable rows per expert (plan = per-core unit list), expert-resident weights, SwiGLU epilogue.
One chip, NC cores. Rows are given expert-grouped (padded to even M-tile counts); the pair-list build / row gather / score are NOT in K1a.
Env: NC=110 NE=256 PAIRS=16384 SKEW=1.0 XL1=0|1 NREP=3 OUT=json"""
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.skinny import _acc, _core_set, _hash
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import TB4, bank_tiles_tensor

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tt", "tp32pf_kernels")
NC = int(os.environ.get("NC", "110"))
NE = int(os.environ.get("NE", "256"))
PAIRS = int(os.environ.get("PAIRS", "16384"))
SKEW = float(os.environ.get("SKEW", "1.0"))
XL1 = int(os.environ.get("XL1", "0"))
NREP = int(os.environ.get("NREP", "3"))
WRITE = int(os.environ.get("WRITE", "0"))
L = 10.0


def tile_cb(cs, idx, ntiles, fmt, page):
    return ttnn.CBDescriptor(
        total_size=ntiles * page,
        core_ranges=cs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=idx, data_format=fmt, page_size=page, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32]))
            )
        ],
    )


def make_plan(counts, nc):
    mts = [((int(c) + 31) // 32 + 1) // 2 * 2 for c in counts]
    start, s = [], 0
    for m in mts:
        start.append(s)
        s += m
    order = sorted([e for e in range(len(counts)) if mts[e] > 0], key=lambda e: -mts[e])
    load = [0.0] * nc
    units = [[] for _ in range(nc)]
    for e in order:
        c = min(range(nc), key=lambda i: load[i])
        units[c].append((e, start[e], mts[e]))
        load[c] += mts[e] * 10 + 8
    plan = torch.zeros(nc, 512, dtype=torch.int32)
    for c in range(nc):
        assert len(units[c]) <= 100
        plan[c, 0] = len(units[c])
        for u, (e, s0, m) in enumerate(units[c]):
            plan[c, 1 + 3 * u : 4 + 3 * u] = torch.tensor([e, s0, m])
    return plan, mts, start, max(load)


def run(md, w2, act, plan, out, cores, nc):
    cs = _core_set(cores)
    CB_A, CB_W, CB_O, CB_META, CB_WS = range(5)
    cbs = [
        tile_cb(cs, CB_A, 8, ttnn.bfloat16, 2048),
        tile_cb(cs, CB_W, 2 * 384, ttnn.bfloat4_b, TB4),
        tile_cb(cs, CB_O, 32, ttnn.bfloat16, 2048),
        ttnn.CBDescriptor(
            total_size=2048,
            core_ranges=cs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_META, data_format=ttnn.uint32, page_size=2048)],
        ),
        ttnn.CBDescriptor(
            total_size=2048,
            core_ranges=cs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_WS, data_format=ttnn.uint32, page_size=2048)],
        ),
    ]
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [i]
    rd = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/k2_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=[CB_W, CB_A, CB_META] + _acc(plan) + _acc(act),
        runtime_args=rt,
        common_runtime_args=[w2.buffer_address(), plan.buffer_address(), act.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    wr = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/k2_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=[CB_O, CB_WS, WRITE] + _acc(plan) + _acc(out),
        runtime_args=rt,
        common_runtime_args=[plan.buffer_address(), out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    cp = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/k2_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=[CB_A, CB_W, CB_O, CB_META],
        runtime_args=rt,
        common_runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=True
        ),
    )
    prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x7F12, nc, WRITE, w2.buffer_address(), act.buffer_address(), plan.buffer_address(), out.buffer_address()
    )
    ttnn.generic_op([act, w2, plan, out], prog)


def traced_us(md, fn, n=8, rep_=3):
    def body():
        for _ in range(rep_):
            fn()

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / rep_ * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def main():
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=16384, trace_region_size=50_000_000)
    torch.manual_seed(0)
    tiles = (torch.randn(1, 8, 6 * NE * 8, 32, 32) * 0.05).bfloat16()
    w2 = bank_tiles_tensor(md, tiles)
    wr = (
        ttnn.to_torch(w2)
        .reshape(1, 6 * NE * 8, 32, 1, 8, 32)
        .permute(0, 3, 4, 1, 2, 5)
        .reshape(8, 6 * NE * 8, 32, 32)
        .float()
    )
    g = md.compute_with_storage_grid_size()
    allc = [ttnn.CoreCoord(xx, yy) for yy in range(g.y) for xx in range(g.x)]
    cores = allc[:NC]
    for rep in range(NREP):
        pop = torch.exp(torch.randn(NE) * 0.7 * SKEW)
        cnt = torch.multinomial(pop / pop.sum(), PAIRS, replacement=True).bincount(minlength=NE)
        plan_t, mts, start, maxload = make_plan(cnt.tolist(), NC)
        MT = sum(mts)
        A = torch.zeros(MT * 32, 64)
        for e in range(NE):
            A[start[e] * 32 : start[e] * 32 + int(cnt[e])] = torch.randn(int(cnt[e]), 64)
        Ab = A.bfloat16()
        at = Ab.reshape(MT, 32, 2, 32).permute(0, 2, 1, 3).reshape(1, 1, MT * 2 * 32, 32)
        act = ttnn.from_torch(
            at, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        plan = ttnn.from_torch(
            plan_t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = ttnn.from_torch(
            torch.zeros(1, 1, MT * 192 * 32, 32).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        f = lambda: run(md, w2, act, plan, out, cores, NC)
        f()
        ttnn.synchronize_device(md)
        worst, nchk = 1.0, 0
        if WRITE:
            o = ttnn.to_torch(out).reshape(MT, 192, 32, 32).permute(0, 2, 1, 3).reshape(MT * 32, 6144).float()
            for e in torch.argsort(cnt, descending=True)[:4].tolist() + torch.randperm(NE)[:4].tolist():
                c = int(cnt[e])
                if c == 0:
                    continue
                # W2 slice [64, 6144]: tile (b, (s*256+e)*8 + k*4 + n) -> rows k*32+r, cols (24b+4s+n)*32 + c
                w = torch.zeros(64, 6144)
                for b in range(8):
                    for s_ in range(6):
                        t = wr[b, (s_ * NE + e) * 8 : (s_ * NE + e) * 8 + 8].reshape(2, 4, 32, 32)  # k, n, r, c
                        cc = (24 * b + 4 * s_) * 32
                        w[:, cc : cc + 128] = t.permute(0, 2, 1, 3).reshape(64, 128)
                ref = Ab[start[e] * 32 : start[e] * 32 + c].float() @ w
                worst = min(worst, pcc(ref, o[start[e] * 32 : start[e] * 32 + c]))
                nchk += 1
        us = traced_us(md, f)
        p(
            "rep",
            rep,
            "NC",
            NC,
            "WRITE",
            WRITE,
            "Mtiles",
            MT,
            "us",
            round(us, 1),
            "pcc",
            round(worst, 5),
            "checked",
            nchk,
        )
        ttnn.deallocate(act)
        ttnn.deallocate(plan)
        ttnn.deallocate(out)
    ttnn.close_mesh_device(md)


main()
