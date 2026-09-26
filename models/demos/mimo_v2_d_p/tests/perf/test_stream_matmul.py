# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Streamed-weight matmul prototype (phase 2 of the streamed-expert plan): y = x @ W with W streamed from DRAM at
full bandwidth and N-split over 64 compute cores, on one Blackhole chip.

  16 reader cores   = each DRAM bank's NOC0-optimal core + its east neighbour (2 readers per bank)
    BRISC  (NOC0)   kernels/stream_mm/sm_reader.cpp   streams its bank region in consumption order into a chunk CB
    NCRISC (NOC1)   kernels/stream_mm/sm_forward.cpp  sends each receiver its K-block, credit-gated, + data counter
  64 compute cores  = R = 4 per reader, the free cores closest along NOC1 (downstream of the forwarder)
    BRISC           kernels/stream_mm/sm_recv.cpp     grants slot credits, publishes landed blocks, publishes x
    TRISC           kernels/stream_mm/sm_compute.cpp  K-streamed matmul_block, partials via packer L1 accumulation

x ([M, K], M <= 128) is resident in every compute core's L1, pre-blocked by K-block; each core owns PCN = N / 32 / 64
output tile columns. W is re-laid out host-side so every reader's region is contiguous: expert -> K-block -> receiver ->
[kblk x pcn] tiles. NUM_EXPERTS copies of W are streamed back to back (distinct DRAM bytes), to measure steady state;
the output keeps the last one, checked against x @ W. Weights and x bf8 (bf4 weights later), y bf16, LoFi.

Tag ``streammm_K{K}_N{N}_M{M}_kb{kblk}_E{E}``; weight GB/s = E * K * N * 1.0625 B / device kernel time.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mimo_v2_d_p.tests.perf.test_dram_read_fwd import noc_hops

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


def _env_list(name, default, cast=str):
    return [cast(v) for v in os.environ.get(name, default).split(",") if v]


KDIR = "models/demos/mimo_v2_d_p/tests/perf/kernels/stream_mm"
COMPUTE = f"{KDIR}/sm_compute.cpp"
SHAPES = _env_list("MIMO_SMM_SHAPES", "4096x4096,2048x4096")
MS = _env_list("MIMO_SMM_M", "32,64,128", int)
KBLK = _env_list("MIMO_SMM_KBLK", "8", int)
EXPERTS = int(os.environ.get("MIMO_SMM_EXPERTS", "4"))
SLOTS = int(os.environ.get("MIMO_SMM_SLOTS", "3"))
READ_BATCH = int(os.environ.get("MIMO_SMM_READ_BATCH", "2"))
R = 4  # receivers per reader
ITERS = int(os.environ.get("MIMO_SMM_ITERS", "3"))
STATS_PATH = Path(os.environ.get("MIMO_SMM_STATS", "generated/mimo_stream_mm/cases.jsonl"))
BF8_TILE = 1088


def _pick_cores(device):
    grid = device.compute_with_storage_grid_size()
    opt = list(device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0))
    readers = opt + [ttnn.CoreCoord(c.x + 1, c.y) for c in opt]  # reader r serves bank r % 8, half r // 8
    taken = {(c.x, c.y) for c in readers}
    free = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
    phys = lambda c: device.worker_core_from_logical_core(c)
    receivers = []
    for r in readers:
        cand = sorted(
            (f for f in free if (f.x, f.y) not in taken), key=lambda f: (noc_hops(phys(r), phys(f), 1), f.y, f.x)
        )[:R]
        assert len(cand) == R, "not enough free cores"
        taken |= {(f.x, f.y) for f in cand}
        receivers += cand  # compute core ci = r * R + j
    return readers, receivers, phys


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("kblk", KBLK, ids=lambda k: f"kb{k}")
@pytest.mark.parametrize("m", MS, ids=lambda m: f"M{m}")
@pytest.mark.parametrize("shape", SHAPES)
def test_stream_matmul(device, shape, m, kblk):
    K, N = (int(v) for v in shape.split("x"))
    banks = device.dram_grid_size().x
    readers, receivers, phys = _pick_cores(device)
    n_rd, n_cc = len(readers), len(receivers)
    Kt, Nt, Mt = K // 32, N // 32, m // 32
    assert Nt % n_cc == 0 and Kt % kblk == 0, (Kt, Nt, kblk)
    pcn, nk, E = Nt // n_cc, Kt // kblk, EXPERTS
    blk_tiles = kblk * pcn
    chunk_tiles = R * blk_tiles

    torch.manual_seed(0)
    W = torch.randn(K, N) * 0.02
    x = torch.randn(m, K)

    # ---- weights: bank b holds reader (b, 0)'s region then reader (b, 1)'s; a region = E x nk x R blocks ----
    Wt = W.view(Kt, 32, Nt, 32).permute(0, 2, 1, 3)  # [Kt, Nt, 32, 32]
    per_reader = []
    for r in range(n_rd):
        blocks = []
        for c in range(nk):
            for j in range(R):
                ci = r * R + j
                blocks.append(Wt[c * kblk : (c + 1) * kblk, ci * pcn : (ci + 1) * pcn].reshape(-1, 32, 32))
        per_reader.append(torch.cat(blocks).repeat(E, 1, 1))  # [E * nk * R * blk_tiles, 32, 32]
    halves = n_rd // banks
    bank_cols = [torch.cat([per_reader[b + h * banks] for h in range(halves)]).reshape(-1, 32) for b in range(banks)]
    w_host = torch.cat(bank_cols, dim=1)  # [T_bank * 32, banks * 32]; column b = bank b's tiles in order
    t_bank = w_host.shape[0] // 32
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    w_dev = ttnn.from_torch(
        w_host,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (t_bank * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    region_bytes = t_bank // halves * BF8_TILE

    # ---- compute-core tensors: x (pre-blocked, replicated), in1 landing ring, output ----
    cc_crs = _crs(receivers)
    cc_order = ttnn.corerange_to_cores(cc_crs, None, True)  # shard i -> this core
    hs = lambda h, w: ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cc_crs, (h, w), ttnn.ShardOrientation.ROW_MAJOR),
    )
    x_blk = torch.cat([x[:, c * kblk * 32 : (c + 1) * kblk * 32] for c in range(nk)])  # [nk * M, kblk * 32]
    x_dev = ttnn.from_torch(
        x_blk.repeat(n_cc, 1),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=hs(nk * m, kblk * 32),
    )
    land = ttnn.allocate_tensor_on_device(
        ttnn.Shape([n_cc * SLOTS * kblk * 32, pcn * 32]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
        hs(SLOTS * kblk * 32, pcn * 32),
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([n_cc * m, pcn * 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, hs(m, pcn * 32)
    )
    in0_tiles, out_tiles = nk * Mt * kblk, Mt * pcn

    # ---- program ----
    rd_crs = _crs(readers)
    all_crs = _crs(readers + receivers)
    DATA_SEM = R
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in range(R + 1)]
    pk = lambda c: (phys(c).x << 16) | phys(c).y
    rd_rt, fw_rt, rv_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for r, c in enumerate(readers):
        rd_rt[c.x][c.y] = [w_dev.buffer_address(), r % banks, (r // banks) * region_bytes]
        fw_rt[c.x][c.y] = [land.buffer_address()] + [pk(receivers[r * R + j]) for j in range(R)]
        for j in range(R):
            rc = receivers[r * R + j]
            rv_rt[rc.x][rc.y] = [pk(c), j]
    num_chunks = E * nk
    batch = READ_BATCH if num_chunks % READ_BATCH == 0 else 1
    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
    FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
    osh = max(h for h in range(1, Mt + 1) if Mt % h == 0 and h * pcn <= 8)  # out subblock h x pcn <= 8 DEST tiles
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/sm_reader.cpp",
            source_type=FP,
            core_ranges=rd_crs,
            compile_time_args=[0, chunk_tiles, BF8_TILE, num_chunks, batch],
            runtime_args=rd_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/sm_forward.cpp",
            source_type=FP,
            core_ranges=rd_crs,
            compile_time_args=[0, R, blk_tiles, BF8_TILE, num_chunks, SLOTS, 0, DATA_SEM],
            runtime_args=fw_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/sm_recv.cpp",
            source_type=FP,
            core_ranges=cc_crs,
            compile_time_args=[0, in0_tiles, 1, blk_tiles, nk, E, SLOTS, 16, out_tiles, DATA_SEM],
            runtime_args=rv_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE,
            source_type=FP,
            core_ranges=cc_crs,
            compile_time_args=[kblk, Mt, pcn, nk, osh, E],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
        ),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * batch * chunk_tiles * BF8_TILE,
            core_ranges=rd_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat8_b, page_size=BF8_TILE)
            ],
        ),
        ttnn.cb_descriptor_from_sharded_tensor(0, x_dev),
        ttnn.cb_descriptor_from_sharded_tensor(1, land),
        ttnn.cb_descriptor_from_sharded_tensor(16, out),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    tag = f"streammm_K{K}_N{N}_M{m}_kb{kblk}_E{E}"
    w_bytes = E * K * N * BF8_TILE / 1024
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(
            json.dumps(
                {
                    "tag": tag,
                    "K": K,
                    "N": N,
                    "M": m,
                    "kblk": kblk,
                    "E": E,
                    "weight_bytes": w_bytes,
                    "flops": 2 * E * m * K * N,
                }
            )
            + "\n"
        )
    for it in range(1 + ITERS):
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_start")
        ttnn.generic_op([w_dev, x_dev, land, out], program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
        if it == 0:
            got_sh = ttnn.to_torch(out).float().view(n_cc, m, pcn * 32)
            got = torch.zeros(m, N)
            ci_of = {(c.x, c.y): i for i, c in enumerate(receivers)}
            for s, core in enumerate(cc_order):
                ci = ci_of[(core.x, core.y)]
                got[:, ci * pcn * 32 : (ci + 1) * pcn * 32] = got_sh[s]
            if os.environ.get("MIMO_SMM_DUMP"):
                torch.save(
                    {
                        "got_sh": got_sh,
                        "ref": x @ W,
                        "land": ttnn.to_torch(land).float(),
                        "W": W,
                        "x": x,
                        "x_dev": ttnn.to_torch(x_dev).float(),
                        "nk": nk,
                        "kblk": kblk,
                        "slots": SLOTS,
                        "order": [(c.x, c.y) for c in cc_order],
                        "recv": [(c.x, c.y) for c in receivers],
                        "pcn": pcn,
                    },
                    os.environ["MIMO_SMM_DUMP"],
                )
            ok, pcc = comp_pcc(x @ W, got, 0.99)
            logger.info(f"{tag}: PCC {pcc}")
            assert ok, pcc
    logger.info(f"ran {tag}: {w_bytes / 1e6:.0f} MB of weights streamed per run")
