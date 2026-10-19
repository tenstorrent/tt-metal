# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""NoC wedge probe: the GLM-5.3-Flash flat routed expert's NoC traffic on its own cores, with NOTHING that can
logically hang - no semaphore waits, no CBs handshakes, no compute (every write lands in scratch nobody reads), so a
dispatch timeout here is a NoC hardware wedge. After tt-metal-mimo's noc_hang_probe op_skeleton (the MiMo NOC0 wedge),
for the DRISC-only plan (MIMO_FL_CMB_DRISC=1, MIMO_FL_DRISC_DUAL=0, MIMO_FL_ND=32, MIMO_FL_NH=1): the roles come from the
real plan (flat_routed_expert_plan), so the cores match the op's.

Parts (NOC_PROBE_DROP: comma list to leave out; NOC_PROBE_NOC1 / NOC_PROBE_NOC0: parts moved to that NoC):
  mcast   x relays' (primaries) multicasts into their gate/up rectangle, chains of MT pieces (NOC0), each chain followed
          by a semaphore-set multicast (NOC_PROBE_LINKED 1/0, NOC_PROBE_POSTED 0/1)
  helper  helper relays' sends into their primary (NOC0)
  x_dram  relays' x reads from DRAM (NOC1)
  drisc   the DRISC gate/up weight senders, emulated from the worker cell nearest each slot's DRAM tile (role map
          NOC_PROBE_ROLEMAP): 9216 B blocks + a data-semaphore increment to the slot's 4 receivers, on the slot's NoC
          (NOC_PROBE_DRISC_MODE default / amp / one_tile_noc1)
  NOC_PROBE_X_MODE mcast (default) | chain: x as unicast column chains (relay -> column heads, each core -> the next)
  NOC_PROBE_LEFT_PRIMARY x,y: the left rectangle's multicasting relay on another cell (e.g. 1,0)
  gu      gate/up receivers' h writes to the down chain heads + freed-slot atomics to their relay (NOC1)
  dn_dram down cores' down-weight reads from DRAM (NOC0, NCRISC)
  dn_y    down cores' y row writes to DRAM (NOC0; NOC_PROBE_Y_VC, default 1: VC 2 as SeYRmWriter, next to the
          probe's full-rate DRAM reads on the default VC, deadlocks every chip on the first launch - BH-76, "different
          VCs to the DRAM endpoint", not the wedge under study)
  dn_h    down cores' h chain to the next down core (NOC1, BRISC)
All 8 chips run the same program each launch. NOC_PROBE_LAUNCHES (2000), NOC_PROBE_ITERS (1500), NOC_PROBE_SYNC (50).

  BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 \\
  scripts/run_safe_pytest.sh --mesh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/hang/test_noc_wedge_probe.py -s
"""

import os
import time

from loguru import logger

from models.demos.common.bringup.testing.harness import mesh_parametrize

FLAT_ENV = {"MIMO_FL_CMB_DRISC": "1", "MIMO_FL_DRISC_DUAL": "0", "MIMO_FL_ND": "32", "MIMO_FL_NH": "1"}
H, I, E, NG, M = 4096, 2048, 36, 288, 8192
CB_BYTES = 256 * 1024

KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
// scratch (CB 0, the same address on every core): nothing here is ever read for its value
constexpr uint32_t SRC = 0, MC = 16384, FLOOD = 65536, CNT = 196608, WORD = CNT + 64, RD = 200704;

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    size_t a = 0;
    const uint32_t iters = get_arg_val<uint32_t>(a++);
    const uint32_t base = get_write_ptr(0);
    if constexpr (role == 0) {  // multicaster: chains of linked pieces into a rectangle, then a semaphore-set multicast
        const uint32_t a0 = get_arg_val<uint32_t>(a++), a1 = get_arg_val<uint32_t>(a++), ndest = get_arg_val<uint32_t>(a++);
        const uint32_t piece = get_arg_val<uint32_t>(a++), chain = get_arg_val<uint32_t>(a++);
        const uint64_t rect = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
        for (uint32_t it = 0; it < iters; ++it) {
            for (uint32_t c = 0; c < chain; ++c) {
                const bool linked = PROBE_LINKED && c + 1 < chain;
#if PROBE_POSTED
                while (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
                }
                ncrisc_noc_fast_write_any_len<noc_mode>(noc_index, write_cmd_buf, base + SRC,
                    rect | (base + MC + (c % 3) * piece), piece, NOC_MULTICAST_WRITE_VC, true, linked, ndest, true, true);
#else
                noc_async_write_multicast(base + SRC, rect | (base + MC + (c % 3) * piece), piece, ndest, linked);
#endif
            }
            noc_semaphore_set_multicast(base + WORD, rect | (base + CNT), ndest);
            noc_async_writes_flushed();
        }
    } else if constexpr (role == 1) {  // flooder: round-robin writes (+ an atomic every `every`) to n targets
        const uint32_t n = get_arg_val<uint32_t>(a++), bytes = get_arg_val<uint32_t>(a++);
        const uint32_t every = get_arg_val<uint32_t>(a++), vcp = get_arg_val<uint32_t>(a++);
        const uint32_t vc = vcp & 0xFF, paced = vcp >> 8;  // (bit 8: a DRISC emulator, paced)
        const size_t t0 = a;
        for (uint32_t it = 0; it < iters; ++it) {
            const uint32_t xy = get_arg_val<uint32_t>(t0 + it % n);
            if (bytes) {
                noc_async_write(base + SRC, get_noc_addr(xy >> 16, xy & 0xFFFF, base + FLOOD + (it % 8) * bytes), bytes,
                                noc_index, vc);
            }
            if (every && it % every == 0) {
                noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, base + CNT), 1);
            }
            if (it % 8 == 7) {
                noc_async_writes_flushed();
            }
#if PROBE_DRISC_GAP
            if (paced) {  // (the DRISC emulators: paced to a DRAM tile's rate, ~28 GB/s per tile)
                riscv_wait(PROBE_DRISC_GAP);
            }
#endif
        }
    } else {  // 2: DRAM reader, 3: DRAM writer (row pages over the banks)
        const uint32_t dram = get_arg_val<uint32_t>(a++), bytes = get_arg_val<uint32_t>(a++);
        const uint32_t first = get_arg_val<uint32_t>(a++), vc = get_arg_val<uint32_t>(a++);
        for (uint32_t it = 0; it < iters; ++it) {
            const uint64_t d = get_noc_addr_from_bank_id<true>((first + it) % 8, dram + (it % 4) * 16384);
            if constexpr (role == 2) {
                noc_async_read(d, base + RD, bytes);
                if (it % 8 == 7) {
                    noc_async_read_barrier();
                }
            } else {
                noc_async_write(base + SRC, d, bytes, noc_index, vc);
                if (it % 8 == 7) {
                    noc_async_writes_flushed();
                }
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_read_barrier();
}
"""

# each bank's active DRISC tile (tile 1), physical (x, y) - from the hang triages' callstacks
DRISC_PHYS = [(0, 1), (0, 10), (0, 4), (0, 7), (9, 1), (9, 10), (9, 4), (9, 7)]


def _phys_to_logical(px, py):
    """Nearest worker cell east of a DRAM-column tile on the 12x10 grid (phys x 1..7 -> logical 0..6, 11..15 -> 7..11;
    phys y 2..11 -> logical 0..9)."""
    lx = 0 if px == 0 else 7
    ly = min(9, max(0, py - 2))
    return lx, ly


def _program(mesh, plan, iters, dram, env):
    import ttnn

    C = ttnn.CoreCoord
    drop = set(v for v in env.get("NOC_PROBE_DROP", "").split(",") if v)
    on1 = set(v for v in env.get("NOC_PROBE_NOC1", "").split(",") if v)
    on0 = set(v for v in env.get("NOC_PROBE_NOC0", "").split(",") if v)
    N0, N1 = ttnn.NOC.NOC_0, ttnn.NOC.NOC_1
    nocof = lambda part, default: N1 if part in on1 else (N0 if part in on0 else default)  # noqa: E731
    gu = [C(*c) for c in plan["gu"]]
    down = [C(*c) for c in plan["down"]]
    relays = [C(*c) for c in plan["relays"]]
    prim, helpers = relays[:2], relays[2:4]
    left = [c for c in gu if c.x <= 6]
    right = [c for c in gu if c.x > 6]
    bbox = lambda cs: (
        (min(c.x for c in cs), min(c.y for c in cs)),
        (max(c.x for c in cs), max(c.y for c in cs)),
    )  # noqa: E731
    rects = [bbox(left), bbox(right)]
    prim = sorted(prim, key=lambda c: c.x)  # (west primary -> the left rectangle)
    if env.get("NOC_PROBE_LEFT_PRIMARY"):  # (relay placement: the left rectangle's multicaster on another free cell)
        prim[0] = C(*[int(v) for v in env["NOC_PROBE_LEFT_PRIMARY"].split(",")])
    helpers = sorted(helpers, key=lambda c: c.x)
    virt = lambda c: mesh.worker_core_from_logical_core(c)  # noqa: E731
    pk = lambda c: (virt(c).x << 16) | virt(c).y  # noqa: E731
    cset = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])  # noqa: E731
    defines = [
        ("PROBE_LINKED", env.get("NOC_PROBE_LINKED", "1")),
        ("PROBE_POSTED", env.get("NOC_PROBE_POSTED", "0")),
        # (DRISC emulator pacing, riscv cycles per 9 KB block: 9216 B at ~28 GB/s ~ 330 ns ~ 450 cycles at 1.35 GHz)
        ("PROBE_DRISC_GAP", env.get("NOC_PROBE_DRISC_GAP", "0")),
    ]
    kernels, busy, entries = [], {}, []

    def add(role, per_core, risc, noc, what):  # (collected; grouped into kernels at the end)
        for c, args in per_core:
            if (c.x, c.y, risc) in busy:
                continue
            busy[(c.x, c.y, risc)] = what
            entries.append((role, c, args, risc, noc))

    mt = int(plan.get("mt", 4))
    a = dram.buffer_address()
    # the DRISC emulation first (it takes its cells' BRISC / NCRISC over whatever else ran there)
    if "drisc" not in drop:
        # each slot's source DRAM tile and NoC from the role map (bh-lb-09: drisc_slots[r].tile_logical -> banks' tiles'
        # phys_noc0), emulated from the nearest worker cell; NOC_PROBE_DRISC_MODE: default (the map: slots 0-7 free tile
        # NOC1, 8-15 alt tile NOC0), amp (every slot NOC0 = MIMO_FL_DRISC_NOC1=0), one_tile_noc1 (no alt tiles, every
        # slot from its bank's free tile on NOC1 = MIMO_FL_DRISC_ALT_MAXCNT=0 + JOB1_NOC=1)
        import json

        rm = json.load(
            open(env.get("NOC_PROBE_ROLEMAP", "/localdev/mstaletovic/hang_campaign/2026-10-18/probe/glm_rolemap.json"))
        )
        tiles = {tuple(t["logical"]): t for b in rm["drisc_banks"] for t in b["tiles"]}
        free_of = {b["bank"]: next(t for t in b["tiles"] if t["role"] == "free tile") for b in rm["drisc_banks"]}
        mode = env.get("NOC_PROBE_DRISC_MODE", "default")
        streams = {}  # (cell, noc) -> receivers
        for sl in rm["drisc_slots"]:
            t = tiles[tuple(sl["tile_logical"])]
            noc = sl["noc"]
            if mode == "amp":
                noc = 0
            elif mode == "one_tile_noc1":
                t, noc = free_of[sl["bank"]], 1
            cell = C(*_phys_to_logical(*t["phys_noc0"]))
            streams.setdefault((cell.x, cell.y, noc), []).extend(C(*rc["logical"]) for rc in sl["receivers"])
        for (x, y, noc), recv in streams.items():
            risc = "RISCV_0" if noc == 0 else "RISCV_1"
            if (x, y, risc) in busy:  # (two slots' streams on one cell and NoC: merged into the first)
                continue
            dvc = int(env.get("NOC_PROBE_DRISC_VC", "3"))  # (se_drisc_gu fwd_vc 3; MIMO_FL_FWD_VC in the op)
            add(
                1,
                [(C(x, y), [iters * 6, len(recv), 9216, 1, 0x100 | dvc] + [pk(c) for c in recv])],
                risc,
                N1 if noc else N0,
                "drisc",
            )
    if "mcast" not in drop and env.get("NOC_PROBE_X_MODE", "mcast") == "chain":
        # x as unicast chains instead of a multicast: the relay streams blocks (+ an arrival increment) to its
        # rectangle's column heads, every gate/up core the same to the next core of its column - the traffic of a
        # store-and-forward chain, unthrottled, nothing waits
        xn = nocof("mcast", N0)
        for c, r in zip(prim, rects):
            (x0, y0), (x1, y1) = r
            heads = [C(x, y0) for x in range(x0, x1 + 1)]
            add(1, [(c, [iters * mt, len(heads), 8704, 1, 1] + [pk(h) for h in heads])], "RISCV_1", xn, "x_chain_relay")
            fwd = [
                (C(x, y), [iters * mt, 1, 8704, 1, 1, pk(C(x, y + 1))])
                for x in range(x0, x1 + 1)
                for y in range(y0, y1)
            ]
            add(1, fwd, "RISCV_1", xn, "x_chain_fwd")
    elif "mcast" not in drop:
        s0, s1 = (1, 0) if nocof("mcast", N0) == N1 else (0, 1)  # (a NOC1 multicast starts at the far corner)
        add(
            0,
            [
                (c, [iters, pk(C(*r[s0])), pk(C(*r[s1])), (r[1][0] - r[0][0] + 1) * (r[1][1] - r[0][1] + 1), 8704, mt])
                for c, r in zip(prim, rects)
            ],
            "RISCV_1",
            nocof("mcast", N0),
            "mcast",
        )
    if "x_dram" not in drop:
        add(
            2,
            [(c, [iters * 2, a, 8192, i, 1]) for i, c in enumerate(prim + helpers)],
            "RISCV_0",
            nocof("x_dram", N1),
            "x_dram",
        )
    if "helper" not in drop:
        add(
            1,
            [(h, [iters * 4, 1, 8704, 4, 1, pk(p)]) for h, p in zip(helpers, prim)],
            "RISCV_1",
            nocof("helper", N0),
            "helper",
        )
    if "gu" not in drop:
        heads = down[:7]
        rect_prim = lambda c: prim[0] if c.x <= 6 else prim[1]  # noqa: E731
        add(
            1,
            [(c, [iters * 4, 2, 2048, 4, 1, pk(heads[i % len(heads)]), pk(rect_prim(c))]) for i, c in enumerate(gu)],
            "RISCV_0",
            nocof("gu", N1),
            "gu",
        )
    if "dn_dram" not in drop:
        add(
            2, [(c, [iters * 4, a, 8192, i, 1]) for i, c in enumerate(down)], "RISCV_1", nocof("dn_dram", N0), "dn_dram"
        )
    if "dn_y" not in drop:  # (the same NCRISC as the reads in the op: alternate cores here, each RISC one stream)
        add(
            3,
            [(c, [iters * 2, a + 65536, 8192, i, int(env.get("NOC_PROBE_Y_VC", "1"))]) for i, c in enumerate(down)],
            "RISCV_0",
            nocof("dn_y", N0),
            "dn_y",
        )
    if "dn_h" not in drop:
        add(
            1,
            [(c, [iters * 4, 1, 4096, 0, 1, pk(down[(i + 1) % len(down)])]) for i, c in enumerate(down)],
            "RISCV_0",
            nocof("dn_h", N1),
            "dn_h",
        )
    # both RISCs of a core on one NoC: dynamic-NoC mode for that core's kernels (per-RISC command buffers and
    # counters - in dedicated mode they would count each other's transactions and the barriers never settle)
    nocs_of = {}
    for role, c, args, risc, noc in entries:
        nocs_of.setdefault((c.x, c.y), []).append(noc)
    shared = {xy for xy, ns in nocs_of.items() if len(ns) != len(set(ns))}
    groups = {}
    for role, c, args, risc, noc in entries:
        dyn = (c.x, c.y) in shared
        groups.setdefault((role, risc, noc, dyn), []).append((c, args))
    for (role, risc, noc, dyn), per_core in groups.items():
        rt = ttnn.RuntimeArgs()
        for c, args in per_core:
            rt[c.x][c.y] = args
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cset([c for c, _ in per_core]),
                compile_time_args=[role],
                defines=defines,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=getattr(ttnn.DataMovementProcessor, risc),
                    noc=noc,
                    noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC if dyn else ttnn.NOC_MODE.DM_DEDICATED_NOC,
                ),
            )
        )
    logger.info(f"NOCPROBE {len(shared)} cores with both RISCs on one NoC (dynamic-NoC mode there)")
    used = sorted({(x, y) for (x, y, _) in busy} | {(c.x, c.y) for c in gu + down + relays})
    program = ttnn.ProgramDescriptor()
    program.kernels = kernels
    program.cbs = [
        ttnn.CBDescriptor(
            total_size=CB_BYTES,
            core_ranges=cset([C(x, y) for x, y in used]),
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048)],
        )
    ]
    desc = ttnn.MeshProgramDescriptor()
    rows, cols = tuple(mesh.shape)
    desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows - 1, cols - 1))] = program
    roles = {}
    for (x, y, risc), what in busy.items():
        roles.setdefault(what, 0)
        roles[what] += 1
    return desc, roles, rects, prim


@mesh_parametrize
def test_noc_wedge_probe(mesh_device):
    import ttnn

    for k, v in FLAT_ENV.items():
        os.environ.setdefault(k, v)
    env = dict(os.environ)
    plan = dict(
        ttnn._ttnn.operations.bringup.flat_routed_expert_plan(
            mesh_device, H, I, E, NG, M, weights_bf8=False, pin=1, x_bf16=False, h_bf16=False
        )
    )
    logger.info(
        f"NOCPROBE plan: drisc {plan.get('drisc')} dual {plan.get('dual')} gu {len(plan['gu'])} down {len(plan['down'])} "
        f"relays {plan['relays']} mt {plan.get('mt')}"
    )
    launches = int(env.get("NOC_PROBE_LAUNCHES", "2000"))
    iters = int(env.get("NOC_PROBE_ITERS", "1500"))
    sync = int(env.get("NOC_PROBE_SYNC", "50"))
    dram = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 8 * 32, 32 * 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    io = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    desc, roles, rects, prim = _program(mesh_device, plan, iters, dram, env)
    tag = ",".join(f"{k}={env[k]}" for k in sorted(env) if k.startswith("NOC_PROBE_")) or "default"
    logger.info(f"NOCPROBE [{tag}] roles {roles}, rectangles {rects}, primaries {[(c.x, c.y) for c in prim]}")
    ttnn.generic_op([io, io], desc)
    ttnn.synchronize_device(mesh_device)
    t0 = t_last = time.perf_counter()
    for i in range(launches):
        ttnn.generic_op([io, io], desc)
        if (i + 1) % sync == 0:
            ttnn.synchronize_device(mesh_device)
            now = time.perf_counter()
            logger.info(
                f"NOCPROBE launch {i + 1}/{launches}: {(now - t_last) / sync * 1e3:.2f} ms/launch, elapsed {now - t0:.0f} s"
            )
            t_last = now
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"NOCPROBE done [{tag}]: {launches} launches x {iters} iters x {mesh_device.get_num_devices()} chips, no hang"
    )
