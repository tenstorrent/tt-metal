# SPDX-License-Identifier: Apache-2.0
"""tp32-prefill-build checkpoint 1: custom gate/up weight-streaming matmul floor (one chip, NC cores, each owns experts me, me+NC, ...).
Per (expert, pass): x [MT rows-tiles x 192 K tiles] (resident L1, bf16) @ W01 expert slice [192 x 4 tiles] (bfp4 bank runs, LAYOUT v1/v2 W01) -> [MT x 4] tiles.
Env: NC=1,8,110  MT=1|2  PASSES=1  NE=256 (experts)  OUT=json. Correctness: PCC vs fp32 reference on the bfp4-rounded tiles (first 4 experts of core 0..)."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.skinny import _acc, _core_set, _hash
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import TB4, bank_tiles_tensor

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tt", "tp32pf_kernels")
MT = int(os.environ.get("MT", "2"))
PASSES = int(os.environ.get("PASSES", "1"))
NE = int(os.environ.get("NE", "256"))
NCS = [int(x) for x in os.environ.get("NC", "1,8,110").split(",")]
NB = int(os.environ.get("NB", "4"))


def run(md, w01, x, out, cores, nc):
    cs = _core_set(cores)
    CB_X, CB_W, CB_O = 0, 1, 2
    cbs = [
        ttnn.CBDescriptor(
            total_size=MT * 192 * 2048,
            core_ranges=cs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_X,
                    data_format=ttnn.bfloat16,
                    page_size=2048,
                    tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=NB * 96 * TB4,
            core_ranges=cs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_W,
                    data_format=ttnn.bfloat4_b,
                    page_size=TB4,
                    tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * MT * 4 * 2048,
            core_ranges=cs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_O,
                    data_format=ttnn.bfloat16,
                    page_size=2048,
                    tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                )
            ],
        ),
    ]
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [i]
    rd = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/fl_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=[CB_W, CB_X, MT, PASSES, NB, NE, nc] + _acc(x),
        runtime_args=rt,
        common_runtime_args=[w01.buffer_address(), x.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    wr = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/fl_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=[CB_O, MT, PASSES, NE, nc] + _acc(out),
        runtime_args=rt,
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    cp = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/fl_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=[CB_X, CB_W, CB_O, MT, PASSES, NE, nc],
        runtime_args=rt,
        common_runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=True
        ),
    )
    prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x7F10, MT, PASSES, NE, nc, NB, w01.buffer_address(), x.buffer_address(), out.buffer_address()
    )
    ttnn.generic_op([x, w01, out], prog)


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
    tiles = (torch.randn(1, 8, NE * 96, 32, 32) * 0.05).bfloat16()
    w01 = bank_tiles_tensor(md, tiles)
    xt = (torch.randn(MT * 192, 32, 32)).bfloat16()  # tile i = (b, m, kl) = (i // (MT*24), (i // 24) % MT, i % 24)
    x = ttnn.from_torch(
        xt.reshape(1, 1, MT * 192 * 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.zeros(1, 1, NE * PASSES * MT * 4 * 32, 32).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g = md.compute_with_storage_grid_size()
    allc = [ttnn.CoreCoord(xx, yy) for yy in range(g.y) for xx in range(g.x)]
    p("grid", g.x, g.y)
    # reference on the bfp4-rounded weights
    wback = ttnn.to_torch(w01).reshape(1, NE * 96 * 32, 8, 32) if False else None
    wr = (
        ttnn.to_torch(w01)
        .reshape(1, NE * 96, 32, 1, 8, 32)
        .permute(0, 3, 4, 1, 2, 5)
        .reshape(8, NE * 96, 32, 32)
        .float()
    )
    res = {}
    for nc in NCS:
        cores = allc[:nc]
        f = lambda: run(md, w01, x, out, cores, nc)
        f()
        ttnn.synchronize_device(md)
        o = ttnn.to_torch(out).reshape(NE, PASSES, MT, 4, 32, 32).float()
        xf = xt.float().reshape(8, MT, 24, 32, 32)
        worst = 1.0
        for e in [q for q in range(NE) if q % nc == 0][
            :4
        ]:  # only core 0 loads real x (others read L1 garbage: DRAM traffic of x excluded)
            w = wr[:, e * 96 : (e + 1) * 96].reshape(8, 24, 4, 32, 32)
            ref = torch.einsum("bmkrc,bkjcn->mjrn", xf, w)  # [m, j, 32, 32]
            worst = min(worst, pcc(ref, o[e, 0]))
        us = traced_us(md, f)
        tp_ns = us * 1e3 / (NE / nc * PASSES * 768 * MT) if nc else 0
        wb = NE * PASSES * 442368 / 1e6
        res[f"nc{nc}"] = dict(us=us, pcc=worst, weights_MB=wb, GBps=wb * 1e3 / us / 1e3 * 1e3 / 1e3)
        p(
            "NC",
            nc,
            "MT",
            MT,
            "PASSES",
            PASSES,
            "us",
            round(us, 1),
            "pcc",
            round(worst, 5),
            "ns/tile-product(max core)",
            round(tp_ns, 1),
            "weight GB/s",
            round(wb / us * 1e3, 1),
        )
    if os.environ.get("OUT"):
        json.dump(res, open(os.environ["OUT"], "w"), indent=1)
    ttnn.close_mesh_device(md)


main()
