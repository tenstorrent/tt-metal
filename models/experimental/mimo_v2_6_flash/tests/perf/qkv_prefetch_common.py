# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the qkv prefetch scripts (kagent_qkv_prefetch_bench.py / kagent_qkv_prefetch_check.py) on
Galaxy row 0.

The unit is BankStreamMatmul.section as tt/moe.py runs it: the next SWA layer's qkv projection inside the MoE reduce
launch. Per chip x [1, 1, 32, 4096] bf16 in DRAM (users in rows 0-3, zeros elsewhere) @ w [4096, 1920] BFP8 (DRAM
interleaved, padded once by ``prepare``), HiFi2 with fp32 accumulation -> [1, 1, 32, 1920] bf16 in L1. The rest of
that launch is the stub kernels/qkv_prefetch_stub.cpp on one reduce core: it raises the start semaphore of every
stream core START_COUNT times (the fabric senders, done reading DRAM), and X_DELAY_US later writes x's user rows and
raises the x-ready semaphore of every loader X_READY_COUNT times (the reduce cores, norm written): x is ready about
17 us after the stream may start, as in the model's merged launch (device profile). x holds other data until then.
"""

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.experimental.mimo_v2_6_flash.tt.common import DramShardedMatmul, MeshPlan, compute_config
from models.experimental.mimo_v2_6_flash.tt.moe_stream import (
    FabricRowGather,
    ReduceExpand,
    reduce_launch_cores,
    semaphore_descriptor,
)
from models.experimental.mimo_v2_6_flash.tt.proj_stream import BankStreamMatmul, RouterScoreStream

HIDDEN, QKV_WIDTH, O_IN, EXPERTS, USERS = 4096, 1920, 1024, 256, 4
START_COUNT = FabricRowGather.SENDERS
X_READY_COUNT = 8  # ReduceExpand's cores
X_DELAY_US = 12.0  # + the stub's x write and signals (~5 us): x is ready ~17 us after the launch starts
CLOCK_MHZ = 1350
STUB = "models/experimental/mimo_v2_6_flash/tests/perf/kernels/qkv_prefetch_stub.cpp"
STUB_CORE = ttnn.CoreCoord(0, ReduceExpand.CORE_ROW)
CB_STUB = 0
TILE_BYTES = 2048

open_row, close = common.open_row, common.close


def _dev(row, t, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t,
        device=row,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(row),
    )


def build(row, seed=0):
    """The prefetching qkv stream and its weight (random, real shape and dtype), plus the stub's x buffers."""
    plan = MeshPlan(row)
    mm = compute_config(ttnn.MathFidelity.HiFi2)
    stream = BankStreamMatmul.prefetching(plan, HIDDEN, QKV_WIDTH, mm, avoid=reduce_launch_cores(row))
    assert stream is not None
    assert (STUB_CORE.x, STUB_CORE.y) in reduce_launch_cores(row)
    g = torch.Generator().manual_seed(seed)
    w = _dev(row, torch.randn(HIDDEN, QKV_WIDTH, generator=g) * 0.02, ttnn.bfloat8_b)
    return plan, stream, w, stream.prepare(w)


def x_host(step, seed=0):
    """[1, 1, 32, HIDDEN] bf16: users in rows 0-3, zeros elsewhere."""
    g = torch.Generator().manual_seed(1000 + step + 100 * seed)
    x = torch.zeros(1, 1, 32, HIDDEN)
    x[0, 0, :USERS] = torch.randn(USERS, HIDDEN, generator=g)
    return x.bfloat16()


def x_tensor(row, x):
    return _dev(row, x, ttnn.bfloat16)


def poison(row, x):
    """Overwrite x's user rows with NaN (what the section must not read before the x-ready semaphore); the other
    rows stay zero, as in the model."""
    t = torch.zeros(1, 1, 32, HIDDEN)
    t[0, 0, :USERS] = float("nan")
    host = ttnn.from_torch(
        t.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(row),
    )
    ttnn.copy_host_to_device_tensor(host, x)


def launch(row, stream, w_stream, x, x_src, delay_us=X_DELAY_US):
    """One launch: the section plus the stub, which copies x_src into x once delay_us have passed. Returns the
    section's [1, 1, 32, QKV_WIDTH] L1 output."""
    section, out = stream.section(x, w_stream, start_count=START_COUNT, x_ready_count=X_READY_COUNT)
    stream_cores, loaders = stream.stream_cores(), stream.loaders()
    stub_noc = row.worker_core_from_logical_core(STUB_CORE)
    assert all((c.x, c.y) != (stub_noc.x, stub_noc.y) for c in stream_cores), "the section takes the stub core"

    def coords(cores):
        return [len(cores)] + [v for c in cores for v in (c.x, c.y)]

    x_tiles = HIDDEN // 32
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(STUB_CORE, STUB_CORE)])
    rt = ttnn.RuntimeArgs()
    rt[STUB_CORE.x][STUB_CORE.y] = [x_src.buffer_address(), x.buffer_address()] + coords(stream_cores) + coords(loaders)
    stub = ttnn.KernelDescriptor(
        kernel_source=STUB,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=ranges,
        compile_time_args=[
            stream.SEM_START,
            START_COUNT,
            stream.SEM_X_READY,
            X_READY_COUNT,
            int(delay_us * CLOCK_MHZ),
            x_tiles,
            CB_STUB,
        ],
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=CB_STUB,
        data_format=ttnn.bfloat16,
        page_size=640,
        tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
    )
    cb = ttnn.CBDescriptor(total_size=x_tiles * 640, core_ranges=ranges, format_descriptors=[fmt])
    sems = list(section["semaphores"]) + [
        semaphore_descriptor(ranges, stream.SEM_START),
        semaphore_descriptor(ranges, stream.SEM_X_READY),
    ]
    program = ttnn.ProgramDescriptor(
        kernels=list(section["kernels"]) + [stub], cbs=list(section["cbs"]) + [cb], semaphores=sems
    )
    ttnn.generic_op([x, x_src] + list(section["tensors"]) + [out], program)
    return out


def standalone(row, seed=0):
    """The other BankStreamMatmul / RouterScoreStream users (same kernels): the model's standalone qkv stream, the
    o stream (x in L1) and the router scores. Returns a function of the step that runs all three."""
    plan = MeshPlan(row)
    mm = compute_config(ttnn.MathFidelity.HiFi2)
    g = torch.Generator().manual_seed(seed + 1)
    qkv = BankStreamMatmul(plan, HIDDEN, QKV_WIDTH, mm, skip_bank_core=True)
    w_qkv = qkv.prepare(_dev(row, torch.randn(HIDDEN, QKV_WIDTH, generator=g) * 0.02, ttnn.bfloat8_b))
    o = BankStreamMatmul(plan, O_IN, HIDDEN, mm)
    w_o = _dev(row, torch.randn(O_IN, HIDDEN, generator=g) * 0.02, ttnn.bfloat8_b)
    router_mm = DramShardedMatmul(
        plan, HIDDEN, EXPERTS, out_dtype=ttnn.float32, compute_kernel_config=compute_config(ttnn.MathFidelity.HiFi4)
    )
    w_router = _dev(
        row, torch.randn(HIDDEN, EXPERTS, generator=g) * 0.02, ttnn.bfloat16, router_mm.weight_memory_config
    )
    router = RouterScoreStream(plan, HIDDEN, EXPERTS, torch.randn(EXPERTS, generator=g) * 0.1)

    def run(step):
        x_h = x_tensor(row, x_host(step, seed=1))
        xo = torch.zeros(1, 1, 32, O_IN)
        xo[0, 0, :USERS] = torch.randn(USERS, O_IN, generator=torch.Generator().manual_seed(2000 + step))
        x_o = _dev(row, xo.bfloat16(), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG)
        assert router.accepts(x_h, w_router)
        scores, biased = router(x_h, w_router)
        outs = {"qkv": qkv(x_h, w_qkv), "o": o(x_o, w_o), "scores": scores, "biased": biased}
        return outs, (x_h, x_o)

    return run


def rows(t):
    """Rows 0-3 of every chip's copy (all chips must agree), as float. Every other row must be finite."""
    full = [ttnn.to_torch(d).float()[0, 0] for d in ttnn.get_device_tensors(t)]
    assert all(bool(torch.isfinite(f[USERS:]).all()) for f in full), "non-finite values outside the user rows"
    chips = [f[:USERS] for f in full]
    for c in range(1, len(chips)):
        assert torch.equal(chips[c].nan_to_num(7.0), chips[0].nan_to_num(7.0)), f"chip {c} differs from chip 0"
    return chips[0]
