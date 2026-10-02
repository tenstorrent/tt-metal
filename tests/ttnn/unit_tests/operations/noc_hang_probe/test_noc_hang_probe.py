# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NoC erratum probes (Blackhole, one chip): traffic patterns that the known NoC hardware bugs describe, each with a
rule-respecting control of the same volume, to see which of them actually wedge the NoC and how often.

There is no flow control between cores: every write lands in scratch nobody reads, so the kernels cannot deadlock on a
protocol; a dispatch timeout here is a NoC-level hang. Geometry follows FlatRoutedExpert's east x relay: a multicaster
at logical (7,3) sending into the rectangle x 8..10, y 0..7, a helper at (7,2) that receives one atomic from it per
chain, the rectangle cores sending data + atomics back to the multicaster tile, and DRAM readers in columns 6, 7.

    NOC_PROBE_LAUNCHES (2000), NOC_PROBE_ITERS (per-launch loop count, 2000), NOC_PROBE_SYNC (100),
    NOC_PROBE_DRAM (rw | r | w: which DRAM traffic the DRAM_RW scenarios issue), NOC_PROBE_DRAM_NOCS (e.g. 0 / 1 / 0,1),
    NOC_PROBE_DRAM_CORES (all | rect: the multicast's receivers | out: columns 6, 7 only | east: column 11 + x 8..10
    rows 8, 9, where the op's down cores sit: their read responses cross the rectangle), NOC_PROBE_MC (0 no multicaster,
    1 on NOC0 (default), 2 on NOC1),
    NOC_PROBE_MC_DELAY (riscv_wait cycles before the first multicast, 0),
    NOC_PROBE_RTA (dummy runtime args appended per core, 0: the dispatcher writes them into every core each launch,
    while the previous launch may still run), NOC_PROBE_COMPUTE_SCALE (3)
"""

import os
import time

import pytest
from loguru import logger

import ttnn

KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// scratch layout (CB 0, same address on every core)
constexpr uint32_t SRC = 0, MC = 16384, FLOOD = 4 * 16384, CNT = 5 * 16384, WORD = CNT + 64, RD = 6 * 16384;

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);  // 0 multicaster, 1 flooder, 2 DRAM reader
    const uint32_t iters = get_arg_val<uint32_t>(0);
    const uint32_t base = get_write_ptr(0);
    if constexpr (role == 0) {
        const uint32_t a0 = get_arg_val<uint32_t>(1), a1 = get_arg_val<uint32_t>(2), ndest = get_arg_val<uint32_t>(3);
        const uint32_t piece = get_arg_val<uint32_t>(4), chain = get_arg_val<uint32_t>(5), hxy = get_arg_val<uint32_t>(6);
        const uint64_t rect = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
        const uint64_t helper = get_noc_addr(hxy >> 16, hxy & 0xFFFF, base + CNT);
        const uint32_t start_delay = get_arg_val<uint32_t>(7);
        if (start_delay) {
            riscv_wait(start_delay);  // control: the receivers' startup reads are done before the first multicast
        }
        for (uint32_t it = 0; it < iters; ++it) {
            for (uint32_t c = 0; c < chain; ++c) {
                const bool linked = MC_LINKED && c + 1 < chain;
                noc_async_write_multicast(base + SRC, rect | (base + MC + (c % 3) * piece), piece, ndest, linked);
#if ATOMIC_IN_CHAIN
                if (linked && c == 0) {
                    noc_semaphore_inc(helper, 1);  // violates: another command buffer while the chain is open
                }
#endif
            }
#if SEM_MCAST
            noc_semaphore_set_multicast(base + WORD, rect | (base + CNT), ndest);
#endif
#if ATOMIC_AFTER
            noc_semaphore_inc(helper, 1);  // relay pattern: an atomic between chains, not drained
#endif
#if DRAIN_ATOMICS
            noc_async_atomic_barrier();
#endif
            noc_async_writes_flushed();
        }
    } else if constexpr (role == 1) {
        const uint32_t txy = get_arg_val<uint32_t>(1), fb = get_arg_val<uint32_t>(2), every = get_arg_val<uint32_t>(3);
        const uint64_t tgt = get_noc_addr(txy >> 16, txy & 0xFFFF, base + FLOOD);
        const uint32_t axy = get_arg_val<uint32_t>(4) ? get_arg_val<uint32_t>(4) : txy;  // atomics' target
        const uint64_t cnt = get_noc_addr(axy >> 16, axy & 0xFFFF, base + CNT);
        for (uint32_t it = 0; it < iters; ++it) {
            if (fb) {
                noc_async_write(base + SRC, tgt + (fb <= 4096 ? (it % 4) * fb : 0), fb);
            }
            if (every && it % every == 0) {
#if FLOOD_INLINE == 1
                noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(cnt, it);
#elif FLOOD_INLINE == 2
                noc_inline_dw_write<InlineWriteDst::DEFAULT, false>(cnt, it);
#else
                noc_semaphore_inc(cnt, 1);
#endif
            }
            if (it % 8 == 7) {
                noc_async_writes_flushed();
            }
        }
    } else if constexpr (role == 4) {
        // FlatRoutedExpert's se_dyn_load on every receiver at kernel start: counts / regions rows and the id table,
        // page 0 of three interleaved tensors (DRAM bank 0 for every core), one barrier; then the receiver is idle
        const uint32_t dram = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2), bytes = get_arg_val<uint32_t>(3);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(get_noc_addr_from_bank_id<true>(0, dram + i * 4096), base + RD + i * 4096, bytes);
        }
        noc_async_read_barrier();
    } else if constexpr (role == 3) {
        // DRAM reads and writes from one NOC, each on its own VC (BH-76: different VCs to the DRAM endpoint deadlock)
        const uint32_t dram = get_arg_val<uint32_t>(1), rb = get_arg_val<uint32_t>(2), first_bank = get_arg_val<uint32_t>(3);
        const uint32_t rvc = get_arg_val<uint32_t>(4), wvc = get_arg_val<uint32_t>(5);
        const uint32_t do_rd = get_arg_val<uint32_t>(6), do_wr = get_arg_val<uint32_t>(7);
        for (uint32_t it = 0; it < iters; ++it) {
            const uint32_t bank = (first_bank + it) % 8;
            if (do_rd) {
                noc_async_read(get_noc_addr_from_bank_id<true>(bank, dram), base + RD, rb, noc_index, rvc);
            }
            if (do_wr) {
                noc_async_write(base + SRC, get_noc_addr_from_bank_id<true>(bank, dram + 16384), rb, noc_index, wvc);
            }
            if (it % 8 == 7) {
                noc_async_read_barrier();
                noc_async_writes_flushed();
            }
        }
    } else {
        const uint32_t dram = get_arg_val<uint32_t>(1), rb = get_arg_val<uint32_t>(2), first_bank = get_arg_val<uint32_t>(3);
        for (uint32_t it = 0; it < iters; ++it) {
            noc_async_read(get_noc_addr_from_bank_id<true>((first_bank + it) % 8, dram), base + RD, rb);
            if (it % 8 == 7) {
                noc_async_read_barrier();
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_read_barrier();
    noc_async_full_barrier();
}
"""

COMPUTE = r"""
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
// L1 port pressure: unpack streams matmul operands, pack writes results, on CBs nobody synchronizes (garbage data)
void kernel_main() {
    const uint32_t iters = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(1, 1, 2);
    matmul_block_init(1, 1, false, 4, 2, 8);
    for (uint32_t it = 0; it < iters; ++it) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < 8; ++k) {
            matmul_block(1, 1, 2 * k, 4 * k, 0, false, 4, 2, 8);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < 8; ++t) {
            pack_tile<true>(t, 2, t);
        }
        tile_regs_release();
    }
}
"""

# scenarios whose rectangle cores and multicaster also run COMPUTE
WITH_COMPUTE = {
    "relay_compute",
    "relay_unlinked_compute",
    "relay_sibling_compute",
    "mcast_compute",
    "relay_full",
    "relay_full_unlinked",
}
# DRAM read+write roles: scenario -> (read VC, write VC, NOCs used); every rectangle + reader core reads and writes DRAM
DRAM_RW = {
    "dram_mixed_vc": (2, 1, (0,)),
    "dram_same_vc": (1, 1, (0,)),
    "dram_mixed_vc_both": (2, 1, (0, 1)),
    "dram_same_vc_both": (1, 1, (0, 1)),
}
# the receivers' startup reads (se_dyn_load): scenario -> (reads, bytes each)
STARTUP_READS = {"startup_reads": (3, 1024), "startup_reads_big": (3, 4096)}
# scenarios with NOC1 weight forwarders: the DRAM-reader cores' NCRISC writes 16 KB bursts into the rectangle cores
WITH_FWD = {"relay_full", "relay_full_unlinked"}

# scenario -> (defines, multicaster args, flooder args, readers on[, sibling NOC])
#   sibling NOC: a DRAM reader on the multicaster's (and the helper's) other RISC, on that NOC (as se11_xrd, NOC1)
#   mc: piece bytes, chain length;  flood: bytes per write (0: none), atomic every N writes (0: none)
SCENARIOS = {
    # documented: linked multicast + another command buffer before the chain completes
    "linked_atomic_in_chain": (dict(MC_LINKED=1, ATOMIC_IN_CHAIN=1), (8704, 8), (0, 0), False),
    "linked_atomic_in_chain_ctrl": (dict(MC_LINKED=1, ATOMIC_AFTER=1, DRAIN_ATOMICS=1), (8704, 8), (0, 0), False),
    # the relay: linked chains, an undrained atomic between them, rectangle sends data + atomics back, DRAM readers
    "relay": (dict(MC_LINKED=1, ATOMIC_AFTER=1, SEM_MCAST=1), (8704, 8), (2048, 4), True),
    "relay_unlinked": (dict(MC_LINKED=0, ATOMIC_AFTER=1, SEM_MCAST=1), (8704, 8), (2048, 4), True),
    "relay_drained": (dict(MC_LINKED=1, ATOMIC_AFTER=1, DRAIN_ATOMICS=1, SEM_MCAST=1), (8704, 8), (2048, 4), True),
    # documented: multicast into a congested grid (no atomics anywhere)
    "mcast_congested": (dict(MC_LINKED=0), (8704, 8), (8192, 0), True),
    "mcast_congested_linked": (dict(MC_LINKED=1), (8704, 8), (8192, 0), True),
    "mcast_alone": (dict(MC_LINKED=0), (8704, 8), (0, 0), False),
    # 4-byte transactions into the multicaster tile from 24 cores
    # the relay core's sibling RISC streams DRAM reads on the other NOC while the multicaster runs
    "linked_sibling_noc1": (dict(MC_LINKED=1), (8704, 8), (0, 0), False, 1),
    "unlinked_sibling_noc1": (dict(MC_LINKED=0), (8704, 8), (0, 0), False, 1),
    "relay_sibling_noc1": (dict(MC_LINKED=1, ATOMIC_AFTER=1, SEM_MCAST=1), (8704, 8), (2048, 4), True, 1),
    "relay_unlinked_sibling_noc1": (dict(MC_LINKED=0, ATOMIC_AFTER=1, SEM_MCAST=1), (8704, 8), (2048, 4), True, 1),
    # the same with the receivers' (and the multicaster's) L1 under unpack/pack load, as in the op
    "relay_compute": (dict(MC_LINKED=1, ATOMIC_AFTER=1, SEM_MCAST=1), (8704, 8), (2048, 4), True),
    "relay_unlinked_compute": (dict(MC_LINKED=0, ATOMIC_AFTER=1, SEM_MCAST=1), (8704, 8), (2048, 4), True),
    "relay_sibling_compute": (dict(MC_LINKED=1, ATOMIC_AFTER=1, SEM_MCAST=1), (8704, 8), (2048, 4), True, 1),
    "relay_full": (dict(MC_LINKED=1, ATOMIC_AFTER=1, SEM_MCAST=1), (8704, 8), (2048, 4), True, 1),
    "relay_full_unlinked": (dict(MC_LINKED=0, ATOMIC_AFTER=1, SEM_MCAST=1), (8704, 8), (2048, 4), True, 1),
    **{
        name: (dict(MC_LINKED=0), (8704, 8), (0, 0), False)
        for name in ("dram_mixed_vc", "dram_same_vc", "dram_mixed_vc_both", "dram_same_vc_both")
    },
    "startup_reads": (dict(MC_LINKED=1, SEM_MCAST=1), (8704, 8), (0, 0), False),
    "startup_reads_big": (dict(MC_LINKED=1, SEM_MCAST=1), (8704, 8), (0, 0), False),
    "mcast_compute": (dict(MC_LINKED=0), (8704, 8), (0, 0), False),
    "atomics_storm": (dict(MC_LINKED=0), (8704, 8), (0, 1), False),
    "inline_posted_storm": (dict(MC_LINKED=0, FLOOD_INLINE=1), (8704, 8), (0, 1), False),
    "inline_nonposted_storm": (dict(MC_LINKED=0, FLOOD_INLINE=2), (8704, 8), (0, 1), False),
}

ALL_DEFS = ("MC_LINKED", "ATOMIC_IN_CHAIN", "ATOMIC_AFTER", "DRAIN_ATOMICS", "SEM_MCAST", "FLOOD_INLINE")


def _program(device, scenario, iters, dram):
    defs, (piece, chain), (fb, every), readers, *sib = SCENARIOS[scenario]
    defines = [(k, str(defs.get(k, 0))) for k in ALL_DEFS]
    C = ttnn.CoreCoord
    mc, helper = C(7, 3), C(7, 2)
    rect = [C(x, y) for y in range(0, 8) for x in range(8, 11)]
    rd = [C(x, y) for x in (6, 7) for y in range(10) if (x, y) not in ((7, 3), (7, 2))] if readers else []
    virt = lambda c: device.worker_core_from_logical_core(c)
    pk = lambda c: (virt(c).x << 16) | virt(c).y
    cset = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    used = [mc, helper] + rect + rd
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    k = lambda role, cores, rt, risc, noc: ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=src,
        core_ranges=cset(cores),
        compile_time_args=[role],
        defines=defines,
        runtime_args=rt,
        config=dm(risc, noc),
    )
    mrt, frt, rrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    mrt[mc.x][mc.y] = [
        iters,
        pk(rect[0]),
        pk(rect[-1]),
        len(rect),
        piece,
        chain,
        pk(helper),
        int(os.environ.get("NOC_PROBE_MC_DELAY", "0")),
    ]
    for c in rect:
        frt[c.x][c.y] = [iters * chain if fb else iters, pk(mc), fb, every, 0]
    for i, c in enumerate(rd):
        rrt[c.x][c.y] = [iters * 4, dram.buffer_address(), 8192, i]
    mc_noc = int(os.environ.get("NOC_PROBE_MC", "1"))
    if mc_noc == 2:  # a NOC1 multicast names the rectangle from its NOC1 start (the far corner)
        mrt[mc.x][mc.y] = [iters, pk(rect[-1]), pk(rect[0])] + list(mrt[mc.x][mc.y])[3:]
    kernels = [k(0, [mc], mrt, "RISCV_1", (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[mc_noc - 1])] if mc_noc else []
    if fb or every:
        kernels.append(k(1, rect, frt, "RISCV_0", ttnn.NOC.NOC_0))
    if rd:
        kernels.append(k(2, rd, rrt, "RISCV_0", ttnn.NOC.NOC_0))
    if scenario in STARTUP_READS:
        srt4 = ttnn.RuntimeArgs()
        n, nbytes = STARTUP_READS[scenario]
        for c in rect:
            srt4[c.x][c.y] = [0, dram.buffer_address(), n, nbytes]
        kernels.append(k(4, rect, srt4, "RISCV_0", ttnn.NOC.NOC_0))
    if scenario in DRAM_RW:
        rvc, wvc, nocs = DRAM_RW[scenario]
        if os.environ.get("NOC_PROBE_DRAM_NOCS"):
            nocs = tuple(int(v) for v in os.environ["NOC_PROBE_DRAM_NOCS"].split(","))
        out = [C(x, y) for x in (6, 7) for y in range(10) if (x, y) not in ((7, 3), (7, 2))]
        east = [C(11, y) for y in range(10)] + [C(x, y) for x in (8, 9, 10) for y in (8, 9)]  # the op's down cores
        workers = {"all": rect + out, "rect": rect, "out": out, "east": east}[
            os.environ.get("NOC_PROBE_DRAM_CORES", "all")
        ]
        for noc in nocs:
            drt = ttnn.RuntimeArgs()
            for i, c in enumerate(workers):
                mode = os.environ.get("NOC_PROBE_DRAM", "rw")
                drt[c.x][c.y] = [
                    iters * 2,
                    dram.buffer_address(),
                    8192,
                    i + 4 * noc,
                    rvc,
                    wvc,
                    "r" in mode,
                    "w" in mode,
                ]
            kernels.append(k(3, workers, drt, ("RISCV_0", "RISCV_1")[noc], (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[noc]))
        used = list(dict.fromkeys(used + workers))
    if scenario in WITH_FWD:
        wrt = ttnn.RuntimeArgs()
        for i, c in enumerate(rd):
            wrt[c.x][c.y] = [iters * 2, pk(rect[i % len(rect)]), 16384, 0, 0]
        kernels.append(k(1, rd, wrt, "RISCV_1", ttnn.NOC.NOC_1))
    if sib:
        srt = ttnn.RuntimeArgs()
        for i, c in enumerate((mc, helper)):
            srt[c.x][c.y] = [iters * 4, dram.buffer_address(), 8192, 3 + i]
        kernels.append(k(2, [mc, helper], srt, "RISCV_0", (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[sib[0]]))
    if scenario in WITH_COMPUTE:
        crt = ttnn.RuntimeArgs()
        for c in rect + [mc]:
            crt[c.x][c.y] = [iters * int(os.environ.get("NOC_PROBE_COMPUTE_SCALE", "3"))]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE,
                source_type=src,
                core_ranges=cset(rect + [mc]),
                runtime_args=crt,
                config=ttnn.ComputeConfigDescriptor(),
            )
        )
    pad = int(os.environ.get("NOC_PROBE_RTA", "0"))
    if pad:
        for kd in kernels:
            rt = kd.runtime_args
            for c in used:
                try:
                    cur = list(rt[c.x][c.y])
                except Exception:
                    continue
                if cur:
                    rt[c.x][c.y] = cur + list(range(pad))
            kd.runtime_args = rt
        logger.info(f"NOCPROBE runtime args on the multicaster: {len(list(kernels[0].runtime_args[mc.x][mc.y]))}")
    program = ttnn.ProgramDescriptor()
    program.kernels = kernels
    program.cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cset(used),
            format_descriptors=[ttnn.CBFormatDescriptor(idx, ttnn.bfloat16, 2048)],
        )
        for idx, size in ((1, 32 * 2048), (2, 8 * 2048))
    ] + [
        ttnn.CBDescriptor(
            total_size=7 * 16384,
            core_ranges=cset(used),
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048)],
        )
    ]
    mesh_desc = ttnn.MeshProgramDescriptor()
    mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = program
    return mesh_desc


ROLEMAP = os.path.join(os.path.dirname(__file__), "flat_expert_rolemap.json")
SKEL_PARTS = (
    "mcast",
    "helper",
    "gu_h",
    "gu_atomics",
    "fwd",
    "rd_dram",
    "dn_dram",
    "dn_y",
    "dn_chain",
    "x_dram",
    "compute",
)


def _skeleton(device, iters, dram):
    """FlatRoutedExpert's NoC traffic on its own cores (flat_expert_rolemap.json, MIMO_FL_SHOW's ROLEMAP of the 2048
    token model shape), without its protocol: every role streams its traffic for the whole launch. NOC_PROBE_SKEL_DROP
    names parts (SKEL_PARTS) to leave out; NOC_PROBE_MC_LINKED (1); NOC_PROBE_SKEL_RELAYS (0,1: west, east); NOC_PROBE_SKEL_NOC1 (parts moved to NOC1:
    mcast, gu_h, rd_dram); NOC_PROBE_SKEL_GU_SPLIT (h1 | a1: the gate/up h writes / atomics alone on NOC1)."""
    import json

    rm = json.load(open(ROLEMAP))
    drop = set(v for v in os.environ.get("NOC_PROBE_SKEL_DROP", "").split(",") if v)
    C = ttnn.CoreCoord
    cc = lambda xy: C(xy[0], xy[1])
    readers, gu, down, heads = (
        [cc(c) for c in rm["readers"]],
        [cc(c) for c in rm["gu"]],
        [cc(c) for c in rm["down"]],
        [cc(c) for c in rm["heads"]],
    )
    relays = [cc(c) for c in rm["relays"]]  # primaries then helpers
    prim, helpers = relays[:2], relays[2:]
    rects = [((2, 0), (5, 9)), ((8, 0), (10, 7))]
    virt = lambda c: device.worker_core_from_logical_core(c)
    pk = lambda c: (virt(c).x << 16) | virt(c).y
    cset = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    linked = os.environ.get("NOC_PROBE_MC_LINKED", "1")
    defines = [
        ("MC_LINKED", linked),
        ("ATOMIC_IN_CHAIN", "0"),
        ("ATOMIC_AFTER", "0" if "helper" in drop else "1"),
        ("DRAIN_ATOMICS", "0"),
        ("SEM_MCAST", "1"),
        ("FLOOD_INLINE", "0"),
    ]
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    N0, N1 = ttnn.NOC.NOC_0, ttnn.NOC.NOC_1
    on1 = set(v for v in os.environ.get("NOC_PROBE_SKEL_NOC1", "").split(",") if v)  # parts moved to NOC1
    # a RISC's sibling takes the other NOC, as the op's own switches do (XNOC=1: the x reader; RD_SWAP=1: the forwarder)
    if "mcast" in on1:
        on1.update(("x_dram!0", "helper"))  # the helpers' landing writes move with the multicast
    if "rd_dram" in on1:
        on1.add("fwd!0")
    if "dn_dram" in on1:  # the down cores' h chain / y writes take NOC0 (the builder's DN_NOC=0)
        on1.update(("dn_y!0", "dn_chain!0"))
    nocof = lambda part, default: N0 if part + "!0" in on1 else (N1 if part in on1 else default)
    kernels = []

    def add(role, per_core, risc, noc):
        if not per_core:
            return
        rt = ttnn.RuntimeArgs()
        for c, args in per_core:
            rt[c.x][c.y] = args
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=src,
                core_ranges=cset([c for c, _ in per_core]),
                compile_time_args=[role],
                defines=defines,
                runtime_args=rt,
                config=dm(risc, noc),
            )
        )

    rd_args = lambda i, n: [n, dram.buffer_address(), 8192, i]
    # relays: NCRISC NOC0 multicast into their rectangle (+ atomic to their helper), BRISC NOC1 x reads from DRAM
    if "mcast" not in drop:
        which = [int(v) for v in os.environ.get("NOC_PROBE_SKEL_RELAYS", "0,1").split(",")]  # 0 west, 1 east
        s0, s1 = (1, 0) if "mcast" in on1 else (0, 1)  # a NOC1 multicast starts at the far corner
        add(
            0,
            [
                (
                    c,
                    [
                        iters,
                        pk(C(*r[s0])),
                        pk(C(*r[s1])),
                        (r[1][0] - r[0][0] + 1) * (r[1][1] - r[0][1] + 1),
                        8704,
                        8,
                        pk(h),
                        0,
                    ],
                )
                for j, (c, r, h) in enumerate(zip(prim, rects, helpers))
                if j in which
            ],
            "RISCV_1",
            nocof("mcast", N0),
        )
    if "x_dram" not in drop:
        add(2, [(c, rd_args(i, iters * 2)) for i, c in enumerate(relays)], "RISCV_0", nocof("x_dram", N1))
    # helpers: NCRISC NOC0 landing writes + an atomic into their primary
    if "helper" not in drop:
        add(1, [(h, [iters * 8, pk(p), 8704, 8, 0]) for h, p in zip(helpers, prim)], "RISCV_1", nocof("helper", N0))
    # gate/up: BRISC NOC0 h slices into the chain heads, atomics (freed words) to their rectangle's relay
    rect_of = lambda c: 0 if c.x <= 5 else 1
    split = os.environ.get("NOC_PROBE_SKEL_GU_SPLIT")  # "h1": h writes NCRISC NOC1, atomics BRISC NOC0; "a1": reverse
    if split:
        h_noc, a_noc = (N1, N0) if split == "h1" else (N0, N1)
        risc = lambda noc: "RISCV_1" if noc == N1 else "RISCV_0"
        add(1, [(c, [iters * 4, pk(heads[i % len(heads)]), 2048, 0, 0]) for i, c in enumerate(gu)], risc(h_noc), h_noc)
        add(1, [(c, [iters * 4, pk(prim[rect_of(c)]), 0, 4, 0]) for c in gu], risc(a_noc), a_noc)
    elif "gu_h" not in drop or "gu_atomics" not in drop:
        fb = 0 if "gu_h" in drop else 2048
        every = 0 if "gu_atomics" in drop else 4
        add(
            1,
            [(c, [iters * 4, pk(heads[i % len(heads)]), fb, every, pk(prim[rect_of(c)])]) for i, c in enumerate(gu)],
            "RISCV_0",
            nocof("gu_h", N0),
        )
    # readers: BRISC NOC0 weight reads from DRAM, NCRISC NOC1 16 KB forwards into their gate/up cores
    if "rd_dram" not in drop:
        add(2, [(c, rd_args(i, iters * 4)) for i, c in enumerate(readers)], "RISCV_0", nocof("rd_dram", N0))
    if "fwd" not in drop:
        add(
            1,
            [(c, [iters * 2, pk(gu[(4 * i) % len(gu)]), 16384, 0, 0]) for i, c in enumerate(readers)],
            "RISCV_1",
            nocof("fwd", N1),
        )
    # down: NCRISC NOC0 weight reads from DRAM, BRISC NOC1 h chain to the next down core / y writes to DRAM
    if "dn_dram" not in drop:
        add(2, [(c, rd_args(i, iters * 4)) for i, c in enumerate(down)], "RISCV_1", nocof("dn_dram", N0))
    if "dn_y" not in drop:
        add(
            3,
            [(c, [iters * 2, dram.buffer_address(), 4096, i, 1, 1, 0, 1]) for i, c in enumerate(down)],
            "RISCV_0",
            nocof("dn_y", N1),
        )
    elif "dn_chain" not in drop:
        add(
            1,
            [(c, [iters * 4, pk(down[(i + 1) % len(down)]), 4096, 0, 0]) for i, c in enumerate(down)],
            "RISCV_0",
            nocof("dn_chain", N1),
        )
    used = list(dict.fromkeys(readers + gu + down + relays))
    if "compute" not in drop:
        crt = ttnn.RuntimeArgs()
        for c in gu + down:
            crt[c.x][c.y] = [iters * int(os.environ.get("NOC_PROBE_COMPUTE_SCALE", "3"))]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE,
                source_type=src,
                core_ranges=cset(gu + down),
                runtime_args=crt,
                config=ttnn.ComputeConfigDescriptor(),
            )
        )
    program = ttnn.ProgramDescriptor()
    program.kernels = kernels
    program.cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cset(used),
            format_descriptors=[ttnn.CBFormatDescriptor(idx, ttnn.bfloat16, 2048)],
        )
        for idx, size in ((0, 7 * 16384), (1, 32 * 2048), (2, 8 * 2048))
    ]
    mesh_desc = ttnn.MeshProgramDescriptor()
    mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = program
    return mesh_desc


@pytest.mark.timeout(86400)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("scenario", list(SCENARIOS) + ["op_skeleton"])
def test_noc_hang_probe(mesh_device, scenario):
    launches = int(os.environ.get("NOC_PROBE_LAUNCHES", "2000"))
    iters = int(os.environ.get("NOC_PROBE_ITERS", "2000"))
    sync = int(os.environ.get("NOC_PROBE_SYNC", "100"))
    dram = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 8 * 32, 32 * 16]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    io = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    desc = (
        _skeleton(mesh_device, iters, dram)
        if scenario == "op_skeleton"
        else _program(mesh_device, scenario, iters, dram)
    )
    ttnn.generic_op([io, io], desc)
    ttnn.synchronize_device(mesh_device)
    t0 = t_last = time.perf_counter()
    for i in range(launches):
        ttnn.generic_op([io, io], desc)
        if (i + 1) % sync == 0:
            ttnn.synchronize_device(mesh_device)
            now = time.perf_counter()
            logger.info(
                f"NOCPROBE {scenario} launch {i + 1}/{launches}: {(now - t_last) / sync * 1e3:.2f} ms/launch, "
                f"elapsed {now - t0:.0f} s"
            )
            t_last = now
    ttnn.synchronize_device(mesh_device)
    logger.info(f"NOCPROBE {scenario} done: {launches} launches x {iters} iters, no hang")
