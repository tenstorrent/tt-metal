# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for rms_norm perf idea I8 — eliding the return multicast's
PRE_HANDSHAKE receiver ack.

WHAT IS ISOLATED. Only the cross-core combine's RENDEZVOUS, on the writer's NoC1:

    every member  -> column-valid unicast of its ROWS_T stat tiles into the ROOT's
                     cb_stat_gather slot r*G+slot, barrier, gather_sem.up
    root          -> gather_sem.wait_min((b+1)*(G-1)), then SenderPipe::send() of
                     the ROWS_T rstd tiles (minus the last tile's face 3) to the
                     group rectangle, src != dst so INCLUDE_SRC loops the root's own
                     copy back into cb_rstd
    every core    -> consume the landing tile(s) to DRAM (the op's store_block)

Removed: reader, compute, reduce, finalize, apply, untilize, gamma, sharding. The
payload bytes, the transaction counts, the semaphore topology and the per-block loop
structure are the op's.

VARIANTS (all under the SAME precision contract — bf16 payload; no compute kernel
exists here, so math_fidelity/fp32_dest_acc are not even reachable):
    baseline      PRE_HANDSHAKE=true  (the op today)
    elide         PRE_HANDSHAKE=false, stock kernel_lib ReceiverPipe
    elide_noinit  PRE_HANDSHAKE=false, ctor does not init data_ready (host does)

CORRECTNESS GATE. The landing buffer is POISONED with bf16 -inf before the loop and
is a SINGLE slot reused across blocks. Block b broadcasts the exact value 2**b. Every
core writes its landing tile(s) out to DRAM after every block, so the host sees, per
(core, block), either the exact broadcast value (correct), -inf (consumed before the
broadcast landed) or a previous block's value (stale slot). Face 3 of the last tile is
never sent (the op's column-valid trim), so it is masked out of the comparison.

MEASURED (blackhole_p150b @1350MHz, DEVICE KERNEL DURATION [ns], one fresh dispatch
per (geometry, variant); the decode rows are the median of 3 repeats, spread < 1.5%).
Every row CORRECT: zero poison reads, zero stale/wrong values.

    geometry       G    rows_t blocks | baseline   elide    elide_noinit
    decode110_b1   110    1      1    |   7907      7276  (1.09x)   7319  (1.08x)
    decode110_b2   110    1      2    |  12770     11610  (1.10x)  11536  (1.11x)
    decode110_b4   110    1      4    |  22644     20054  (1.13x)  20090  (1.13x)
    bshard64_b1      8   16      1    |  65267     65556  (1.00x)  65422  (1.00x)
    bshard64_b2      8   16      2    |  91194     91183  (1.00x)  90953  (1.00x)
    wshard8_b1       8    1      1    |   4769      4748  (1.00x)   4767  (1.00x)

READ: the saving is ~600-650 ns PER BLOCK and it scales with the ACK COUNT (G-1
remote atomic increments converging on one root), not with the payload. At G=110 it
is 8-13% of this rendezvous; at G=8 it is inside the noise. Nothing regresses.

MULTI-BLOCK IS CORRECT, not just num_blocks==1: cb_rstd here is a SINGLE slot reused
every block, poisoned with -inf, and the 2- and 4-block runs come back exact. The
back-pressure that replaces the ack is the GATHER (see the kernel head comment).
"""

from __future__ import annotations

from pathlib import Path

import torch
import ttnn

TILE = 32
KERNEL_DIR = Path(__file__).parent / "kernels"
BF16_TILE_BYTES = 2 * TILE * TILE

CB_STAT_PARTIAL = 7
CB_STAT_GATHER = 8
CB_RSTD_SEND = 10
CB_RSTD = 11

SEM_GATHER = 0
SEM_MCAST_READY = 1
SEM_MCAST_CONSUMED = 2

VARIANTS = {"baseline": 0, "elide": 1, "elide_noinit": 2}
BASELINE = "baseline"


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _rect(cores):
    x0 = min(c[0] for c in cores)
    x1 = max(c[0] for c in cores)
    y0 = min(c[1] for c in cores)
    y1 = max(c[1] for c in cores)
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _cb(index, core_ranges, num_pages, page_size=BF16_TILE_BYTES, fmt=ttnn.bfloat16):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)],
    )


# ---------------------------------------------------------------- geometries
# Each geometry mirrors one of the op's real combine topologies.
#   name -> dict(groups=[[core,...], ...], rows_t, num_blocks)
def geometry(name):
    if name == "decode110_b1":  # focus shape (1,1,32,7168) interleaved: 110 cores, 1 group, 1 block
        return dict(groups=[[(i % 11, i // 11) for i in range(110)]], rows_t=1, num_blocks=1)
    if name == "decode110_b2":  # same rendezvous, 2 blocks (landing slot reused once)
        return dict(groups=[[(i % 11, i // 11) for i in range(110)]], rows_t=1, num_blocks=2)
    if name == "decode110_b4":  # 4 blocks: the single-slot landing buffer is reused 3x
        return dict(groups=[[(i % 11, i // 11) for i in range(110)]], rows_t=1, num_blocks=4)
    if name == "bshard64_b2":  # (1,1,8192,1024) BLOCK_SHARDED [1024,128] (8,8): 8 row-groups, 2 blocks
        return dict(groups=[[(x, y) for x in range(8)] for y in range(8)], rows_t=16, num_blocks=2)
    if name == "bshard64_b1":  # same grid, one block (rows_t 16, no slot reuse)
        return dict(groups=[[(x, y) for x in range(8)] for y in range(8)], rows_t=16, num_blocks=1)
    if name == "wshard8_b1":  # (1,1,32,1024) WIDTH_SHARDED [32,128] (8,1): one 8-core group
        return dict(groups=[[(x, 0) for x in range(8)]], rows_t=1, num_blocks=1)
    raise KeyError(name)


def make_out(device, n_cores, rows_t, num_blocks):
    n_tiles = n_cores * rows_t * num_blocks
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, TILE * n_tiles, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def descriptor(device, variant, geo, tt_out):
    groups, rows_t, num_blocks = geo["groups"], geo["rows_t"], geo["num_blocks"]
    G = len(groups[0])
    assert all(len(g) == G for g in groups)
    all_cores = [c for g in groups for c in g]
    n_cores = len(all_cores)
    crs = _crs(all_cores)
    pre_handshake = VARIANTS[variant] == 0

    cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED])
    mcasts = [ttnn.Mcast2D(device, _rect(g), ttnn.CoreCoord(*g[-1]), cfg, G - 1) for g in groups]
    mcast_ct = list(mcasts[0].compile_time_args(pre_handshake))
    for m in mcasts[1:]:
        assert list(m.compile_time_args(pre_handshake)) == mcast_ct

    ct = [num_blocks, rows_t, G, VARIANTS[variant]] + mcast_ct
    assert len(ct) == 9
    ct.extend(ttnn.TensorAccessorArgs(tt_out).get_compile_time_args())

    rt = ttnn.RuntimeArgs()
    cid = 0
    for gi, g in enumerate(groups):
        root = g[-1]
        rv = device.worker_core_from_logical_core(ttnn.CoreCoord(*root))
        for slot, (cx, cy) in enumerate(g):
            own = [
                1 if (cx, cy) == root else 0,
                tt_out.buffer_address(),
                slot,
                int(rv.x),
                int(rv.y),
                cid,
                n_cores,
                0,
            ]
            rt[cx][cy] = own + list(mcasts[gi].runtime_args(ttnn.CoreCoord(cx, cy)))
            cid += 1

    cbs = [
        _cb(CB_STAT_PARTIAL, crs, rows_t),
        _cb(CB_STAT_GATHER, crs, rows_t * G),
        _cb(CB_RSTD_SEND, crs, rows_t),
        _cb(CB_RSTD, crs, rows_t),  # SINGLE slot, reused across blocks (the op's shape)
    ]
    sems = [
        ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=crs, initial_value=0),
        # 0 == INVALID: with the ack elided this host value is what the receivers rest on.
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=crs, initial_value=0),
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "bench_writer.cpp"),
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.WriterConfigDescriptor(),
        )
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)


def check(out_tensor, geo):
    """Exact-value gate: every (core, block) landing tile must hold 2**block."""
    rows_t, num_blocks = geo["rows_t"], geo["num_blocks"]
    n_cores = sum(len(g) for g in geo["groups"])
    got = ttnn.to_torch(out_tensor)[0, 0].to(torch.float32)
    bad_poison = 0
    bad_value = 0
    worst = None
    for b in range(num_blocks):
        expect = float(2**b)
        for c in range(n_cores):
            for r in range(rows_t):
                t = (b * n_cores + c) * rows_t + r
                tile = got[t * TILE : (t + 1) * TILE, :].clone()
                if r == rows_t - 1:
                    tile[16:, 16:] = expect  # face 3 of the last tile is never sent
                bad_poison += int((tile == float("-inf")).sum())
                mism = tile != expect
                if mism.any():
                    bad_value += int(mism.sum())
                    if worst is None:
                        worst = (b, c, r, float(tile[mism][0]))
    return {
        "elems": n_cores * num_blocks * rows_t * TILE * TILE,
        "poison_reads": bad_poison,
        "wrong_values": bad_value,
        "first_wrong": worst,
    }


def device_kernel_ns(device):
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    ns = None
    for programs in per_chip.values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get("DEVICE KERNEL DURATION [ns]")
            if entry is None:
                continue
            d = float(entry.duration)
            ns = d if ns is None else max(ns, d)
    return ns


def run(device, variant, geo_name, verify=True):
    geo = geometry(geo_name)
    n_cores = sum(len(g) for g in geo["groups"])
    tt_out = make_out(device, n_cores, geo["rows_t"], geo["num_blocks"])
    desc = descriptor(device, variant, geo, tt_out)
    # generic_op wants >= 2 io tensors (>= 1 in, 1 out); this bench needs no input,
    # so a 1-tile dummy rides along untouched.
    dummy = ttnn.from_torch(
        torch.zeros((1, 1, TILE, TILE), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn.ReadDeviceProfiler(device)
    out = ttnn.generic_op([dummy, tt_out], desc)
    ns = device_kernel_ns(device)
    stats = check(out, geo) if verify else None
    del out, tt_out, dummy
    return ns, stats


def main(device, geo_names=("decode110_b1", "decode110_b2", "bshard64_b2"), variants=tuple(VARIANTS), verify=True):
    """One fresh dispatch per (geometry, variant). Prints ns + the correctness gate."""
    results = {}
    for gname in geo_names:
        base = None
        for v in variants:
            ns, stats = run(device, v, gname, verify=verify)
            results[(gname, v)] = (ns, stats)
            if v == BASELINE:
                base = ns
            ratio = f"{base / ns:.3f}x" if (base and ns) else "-"
            ok = "CORRECT"
            if stats is not None and (stats["poison_reads"] or stats["wrong_values"]):
                ok = f"WRONG poison={stats['poison_reads']} bad={stats['wrong_values']} first={stats['first_wrong']}"
            print(f"{gname:16s} {v:14s} {ns:>9.0f} ns  {ratio:>7s}  {ok}", flush=True)
    return results
