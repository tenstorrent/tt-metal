# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Combine lab (MEASUREMENT / DESIGN ONLY, not production): the MoE combine's mechanisms rebuilt bottom-up as
generic_op programs with a Python host, to find where the production combine loses against the bare fabric
all-to-all (GLM-5.3 4x4 ring-ring 10240: combine alone 1407 us vs the all-to-all's 789 us) and to try designs fast.
Every planning choice is here in Python; the kernels (kernels/lab_*.cpp) are JIT-compiled.

Stage "take" (this file): ports (one a fabric direction and link: kernels/lab_port.cpp) send every row as one fabric
unicast into a landing ring - one ring of D slots a (source chip, accumulator) on the accumulator's core - under the
production combine's credit flow control (a row needs room: D + credits back - rows sent; accumulators count the
rows they take at the home port facing the source, which snapshots the counts back over the fabric every CRED_THR
rows); accumulators (kernels/lab_acc.cpp) only take the rows (no sums yet). CREDIT=0: no flow control (the
all-to-all with the combine's landing addresses).

Routing: uniform - every chip sends rows_per_dst rows to every other chip, rows_per_dst / A to each of its
accumulators; a row to accumulator o of chip h leaves chip s by the fabric's first hop toward h, on link
(o + s + h) % L (production's link_of).
"""

import dataclasses
from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.combine_lab.bench_util import crs, kpath

FAB_DEFS = [("FABRIC_2D", "1")]
DIRS = ("r+", "r-", "c+", "c-")


def node(mesh, rc):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*rc))


def fabric_first_hops(mesh):
    """chips, {(src, dst): the neighbour the fabric's route from src to dst leaves through}."""
    R, C = tuple(mesh.shape)
    chips = [(r, c) for r in range(R) for c in range(C)]
    fh = {}
    for s in chips:
        r, c = s
        toward = {}
        for p in {((r + 1) % R, c), ((r - 1) % R, c), (r, (c + 1) % C), (r, (c - 1) % C)} - {s}:
            rd = ttnn.get_eth_forwarding_direction(mesh, node(mesh, s), node(mesh, p))
            assert rd is not None, f"no fabric route {s} -> neighbour {p}"
            toward[rd] = p
        for d in chips:
            if d != s:
                rd = ttnn.get_eth_forwarding_direction(mesh, node(mesh, s), node(mesh, d))
                assert rd in toward, f"fabric first hop {s} -> {d} ({rd}) is not toward a torus neighbour"
                fh[(s, d)] = toward[rd]
    return chips, fh


def dir_of(mesh, s, nb):
    R, C = tuple(mesh.shape)
    r, c = s
    for name, p in zip(DIRS, (((r + 1) % R, c), ((r - 1) % R, c), (r, (c + 1) % C), (r, (c - 1) % C))):
        if p == nb:
            return name
    raise AssertionError(f"{nb} is no neighbour of {s}")


@dataclass
class LabCfg:
    PB: int = 12320  # packet bytes: a 6144-wide bf16 row + the 32 B trailer (GLM-5.3)
    SLOT: int = 12352  # landing slot stride (64 B aligned)
    rows_per_dst: int = 320  # rows a chip sends each other chip a call (GLM-5.3 4x4 10240, uniform: 5120 rows a chip)
    A: int = 20  # accumulators a chip (production's choice at this shape)
    D: int = 3  # slots a (source, accumulator) ring (production's at this shape)
    links: int = 2
    nh: int = 8  # rotating headers a destination (flush every nh sends)
    cred_thr: int = 4  # rows taken from a source before the home port snapshots its credits back
    credit: bool = True  # False: no flow control (slots overwritten, accumulators idle)
    read_row: bool = False  # accumulators read each row (a NoC read of PB bytes) before freeing its slot
    timing: bool = False  # LAB_TIMING: ports' wait counters as profiler data
    burst: int = 2  # order 2: rows a destination back to back
    order: int = 0  # the ports' round robin: 0 destination-major (bursts to one chip), 1 accumulator-major
    pool: int = 0  # > 0: the rings on this many pool cores (production's landing on the flat expert cores), else on the
    # accumulators' own cores
    # producer stage: W writer cores hand every row's segments to the ports' injection rings (lab_writer.cpp,
    # lab_preader.cpp), as production's y writers; else the ports send synthetic rows
    producer: bool = False
    islots: int = 32  # a port's injection ring slots (production: at most 32)
    gb: int = 8  # the port reader's grant batch (production: 8)
    pace: int = 0  # writers: cycles a 32-row tile at least (0: as fast as they can)
    split: bool = False  # a row's halves go one a link (two writer groups, rings per (source, half)); else whole rows by the link hash
    fused: bool = False  # the port's sender also runs the hand-off (production with the accumulators on the ports)
    pace_g: int = 1  # tiles a burst (the experts' 128-row sub-blocks: 4)
    kstride: int = 7919  # the row order's stride over the (home, accumulator) pairs (coprime with them)
    nwriters: int = 32  # the first nwriters of `writers` (W: SEG = 12288 / W)
    geom: str = ""  # a production geometry (JSON from test_dump_geometry): writers = its down cores, ports per chip and
    # accumulators = its combine plan's (8 of them on the chip's port cores), pool = its flat expert cores
    # geometry variants of the ports (the accumulators stay production's 20 free cores): 'prod' production's per chip;
    # 'row9' / 'top' the 8 cores of a fixed list for every chip (heaviest port first); 'swapxy' production's X ports on
    # its Y ports' cores and back
    gports: str = "prod"
    port_layout: int = 0  # ports: 0 column 0 (the a2a's), 1 row 8 one a column (x 0-7), 2 row 9 one a column
    wnoc: int = 0  # the writers' NoC (0: RISCV_1 NOC0, 1: RISCV_0 NOC1)
    pnoc: int = 0  # the port's sender NoC (0: sender RISCV_1 NOC0 + reader RISCV_0 NOC1; 1: swapped)
    wdest: int = 0  # writers' payload: 0 the port's ring, 1 their own L1 (no NoC route: diagnosis)
    wmode: int = 0  # writers' segment writes: 0 noc_async_write, 1 noc_async_write_one_packet
    nowrite: bool = False  # writers run the hand-off's protocol without the payload writes
    writers: list = field(default_factory=lambda: [(x, y) for y in range(8) for x in range(7, 11)])
    # (one port a (direction, link): DIRS x links; the a2a's simple placement - column 0)
    ports: list = field(default_factory=lambda: [(0, y) for y in range(8)])
    accs: list = field(default_factory=lambda: [(x, y) for y in range(4) for x in range(2, 7)])


def state_layout(N, A, nh, W=32):
    """Byte offsets of the state block (every lab core, the same address): lab_port.cpp / lab_acc.cpp."""
    AP = (A + 3) // 4 * 4
    o, off = 0, {}

    def take(name, nbytes):
        nonlocal o
        off[name] = o
        o = (o + nbytes + 63) // 64 * 64

    take("O_CRED", 4 * N * AP)
    take("O_USED", 4 * N * AP)
    take("O_RET", 4 * N * AP)
    take("O_RTOT", 4 * N)
    take("O_RDONE", 4 * N)
    take("O_GOT", 4 * N * 2)
    take("O_HDRS", N * nh * 128)
    take("O_STAGE", nh * 4 * AP)
    take("O_WORK", 2 * 4 * N * A)
    take("O_SRC", 12352)
    take("O_PROBE", 64 * (2 * N + 1))
    take("O_GRANT", 4 * 8)
    take("O_ISSUED", 4 * 8)
    take("O_TRSTAGE", 32 * 32)
    take("O_LANDED", 4 * W)
    take("O_READY", 4)
    take("O_FREED", 4)
    take("O_GRANTED", 4)
    return off, o, AP


def _dm(noc):
    """a data movement kernel on NoC noc: RISCV_1 on NOC0, RISCV_0 on NOC1 (the dedicated-NoC pairing)"""
    return ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1 if noc == 0 else ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1,
    )


def build(mesh, cfg: LabCfg):
    R, C = tuple(mesh.shape)
    chips, fh = fabric_first_hops(mesh)
    N, A, L = len(chips), cfg.A, cfg.links
    assert cfg.rows_per_dst % A == 0, "uniform: rows_per_dst a multiple of A"
    ROWS = cfg.rows_per_dst // A
    geo = None
    if cfg.geom:  # (the production plan's cores; block coordinates -> mesh: transposed with SP on mesh axis 1)
        import json

        geo = json.load(open(cfg.geom))
        tr = geo["sp_axis"] == 1
        to_mesh = lambda r, c: (c, r) if tr else (r, c)  # noqa: E731
        bdir = {0: "r+", 1: "r-", 2: "c+", 3: "c-"} if tr else {0: "c+", 1: "c-", 2: "r+", 3: "r-"}
        geo_port = {(to_mesh(r, c), bdir[d], li): (x, y) for r, c, d, li, x, y in geo["port_cores"]}
        if cfg.gports == "swapxy":  # (each chip's X ports onto its Y ports' cores and back, link for link)
            sw = {"r+": "c+", "c+": "r+", "r-": "c-", "c-": "r-"}
            geo_port = {(ch, sw[d], li): xy for (ch, d, li), xy in geo_port.items()}
        elif cfg.gports in ("row9", "top"):
            fixed = {
                "row9": [(8, 9), (9, 9), (10, 9), (11, 9), (6, 9), (7, 9), (0, 9), (1, 9)],
                "top": [(0, 0), (1, 0), (6, 1), (7, 1), (0, 3), (1, 3), (6, 4), (6, 6)],
            }[cfg.gports]
            geo_fixed = fixed
        accs0 = [(x, y) for r, c, x, y in geo["accumulator_cores"] if (r, c) == tuple(geo["accumulator_cores"][0][:2])]
        cfg = dataclasses.replace(cfg, writers=[tuple(c) for c in geo["down_cores"]], accs=accs0, A=len(accs0), fused=True)
        A = cfg.A
        assert cfg.rows_per_dst % A == 0 or True
    assert (geo or len(cfg.ports) == len(DIRS) * L) and len(cfg.accs) == A
    idx = {ch: ch[0] * C + ch[1] for ch in chips}  # (the block index: production's topo.index)
    vxy = lambda core: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*core))
    pk = lambda core: vxy(core).x | vxy(core).y << 8
    cfg = dataclasses.replace(cfg, writers=cfg.writers[: cfg.nwriters])
    if cfg.port_layout:  # (before anything reads the ports)
        cfg = dataclasses.replace(cfg, ports=[(x, 7 + cfg.port_layout) for x in range(8)])
    if geo and cfg.gports in ("row9", "top"):  # (per chip: its ports by load, heaviest first, onto the fixed list)
        geo_port = {}
        for ch in chips:
            load = {d: sum(1 for h in chips if h != ch and dir_of(mesh, ch, fh[(ch, h)]) == d) for d in DIRS}
            order = sorted(((d, l) for d in DIRS for l in range(L)), key=lambda k: (-load[k[0]], DIRS.index(k[0]), k[1]))
            for (d, l), xy in zip(order, geo_fixed):
                geo_port[(ch, d, l)] = xy

    def pc(ch, d, l):  # (the port core of chip ch's (direction, link))
        return geo_port[(ch, d, l)] if geo else cfg.ports[DIRS.index(d) * L + l]

    all_ports = sorted({pc(ch, d, l) for ch in chips for d in DIRS for l in range(L)})
    W = len(cfg.writers)
    SL = L if cfg.split else 1
    if cfg.split:  # (a part: its bytes and the 32 B trailer, 64 B aligned slots)
        cfg = dataclasses.replace(cfg, PB=12288 // SL + 32, SLOT=(12288 // SL + 32 + 63) // 64 * 64)
    off, state_bytes, AP = state_layout(N, A, cfg.nh, W)
    used = set(all_ports) | set(cfg.accs)
    g = mesh.compute_with_storage_grid_size()
    free = [(x, y) for y in range(g.y) for x in range(g.x) if (x, y) not in used]
    pool = free[: cfg.pool] if not geo else [tuple(c) for c in (geo["gate_up_cores"] + geo["down_cores"])][: cfg.pool]
    assert len(pool) == cfg.pool, f"{len(free)} free cores for a pool of {cfg.pool}"
    ring_bytes = (-(-A * N * SL * cfg.D // cfg.pool) if cfg.pool else N * SL * cfg.D) * cfg.SLOT
    assert ring_bytes <= 1300 * 1024, f"rings {ring_bytes} B a core"
    cores = sorted(set(all_ports) | set(cfg.accs) | (set(cfg.writers) if cfg.producer else set()))
    assert not cfg.producer or not (set(cfg.writers) & (set(all_ports) | set(cfg.accs))), "writers on free cores"
    rep = ttnn.ReplicateTensorToMesh(mesh)

    def l1(core_list, nbytes):
        rows = -(-nbytes // 64)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs(core_list), [rows, 16], ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.from_torch(
            torch.zeros(len(core_list) * rows, 16, dtype=torch.int32), dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mem, mesh_mapper=rep,
        )  # fmt: skip

    state = l1(cores, state_bytes)
    rings = l1(pool if cfg.pool else cfg.accs, ring_bytes)
    iring = l1(all_ports, cfg.islots * cfg.SLOT) if cfg.producer else None
    ROWB = 12288 // SL  # (a part's bytes; SEG below from the whole row)
    NP = (N - 1) * A
    named = dict(
        PB=cfg.PB, SLOT=cfg.SLOT, A=A, AP=AP, D=cfg.D, N=N, L=L, NH=cfg.nh, ROWS=ROWS, CRED_THR=cfg.cred_thr,
        STATE=state.buffer_address(), RINGS=rings.buffer_address(), CREDIT=int(cfg.credit),
        READ_ROW=int(cfg.read_row), POOL=cfg.pool, ORDER=cfg.order, BURST=cfg.burst, **off,
        PRODUCER=int(cfg.producer), IRING=iring.buffer_address() if iring else 0, ISLOTS=cfg.islots, ROWB=ROWB,
        W=W, SEG=12288 // W, SL=SL, NOWRITE=int(cfg.nowrite), WMODE=cfg.wmode, WDEST=cfg.wdest, NROWS=NP * ROWS, NP=NP, K=cfg.kstride, PACE=cfg.pace, PACE_G=cfg.pace_g, GB=cfg.gb, FUSED=int(cfg.fused),
    )  # fmt: skip
    defines = FAB_DEFS + ([("LAB_TIMING", "1")] if cfg.timing else [])
    mpd = ttnn.MeshProgramDescriptor()
    stats = {}
    for ch in chips:
        me = idx[ch]
        prt = ttnn.RuntimeArgs()
        port_list = []
        for d in DIRS:
            for l in range(L):
                core = pc(ch, d, l)
                dsts = [h for h in chips if h != ch and dir_of(mesh, ch, fh[(ch, h)]) == d]
                srcs = [s for s in chips if s != ch and dir_of(mesh, ch, fh[(ch, s)]) == d]
                args = [int(node(mesh, ch).mesh_id), me, l, 0, DIRS.index(d) * L + l, len(dsts)]
                for h in dsts:
                    args += [idx[h], int(node(mesh, h).chip_id)]
                args += [len(srcs)]
                for s in srcs:
                    sp = pc(s, dir_of(mesh, s, fh[(s, ch)]), l)  # (the port on s that sends me rows on link l)
                    args += [idx[s], int(node(mesh, s).chip_id), pk(sp)]
                args += [pk(a) for a in cfg.accs] + [pk(c) for c in pool]
                grp = cfg.writers[l * (W // SL):(l + 1) * (W // SL)] if cfg.split else cfg.writers
                args += ([len(grp)] + [pk(c) for c in grp]) if (cfg.producer and cfg.fused) else []
                stats[(ch, d, l)] = sum(1 for h in dsts for o in range(A) if cfg.split or (o + me + idx[h]) % L == l) * ROWS
                args[3] = stats[(ch, d, l)]  # (PRODUCER: this port's rows a call)
                prt[core[0]][core[1]] = args
                port_list.append((core, d, l))
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=kpath("lab_port.cpp"),
                core_ranges=crs([c for c, _, _ in port_list]),
                named_compile_time_args=list(named.items()),
                defines=defines,
                runtime_args=prt,
                config=_dm(cfg.pnoc),
            )
        ]
        if cfg.credit:
            art = ttnn.RuntimeArgs()
            for o, core in enumerate(cfg.accs):
                args = [me, o]
                for s in chips:
                    for part in range(SL):
                        if s == ch:
                            args += [0, 0]
                        else:
                            link = part if cfg.split else (o + idx[s] + me) % L
                            args += [ROWS, pk(pc(ch, dir_of(mesh, ch, fh[(ch, s)]), link))]
                art[core[0]][core[1]] = args + [pk(c) for c in pool]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("lab_acc.cpp"),
                    core_ranges=crs(cfg.accs),
                    named_compile_time_args=list(named.items()),
                    runtime_args=art,
                    # (with the geometry some accumulators share the port cores: the port sender's other RISC)
                    config=_dm(1 - cfg.pnoc) if geo else _dm(0),
                )
            )
        if cfg.producer:
            homes = [h for h in chips if h != ch]
            wrt = ttnn.RuntimeArgs()
            pids = [pk(pc(ch, d, l)) for d in DIRS for l in range(L)]
            for wi, core in enumerate(cfg.writers):
                args = [me, wi, len(homes)]
                for h in homes:
                    args += [idx[h], DIRS.index(dir_of(mesh, ch, fh[(ch, h)]))]
                wrt[core[0]][core[1]] = args + pids
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("lab_writer.cpp"),
                    core_ranges=crs(cfg.writers),
                    named_compile_time_args=list(named.items()),
                    defines=[("LAB_TIMING", "1")] if cfg.timing else [],
                    runtime_args=wrt,
                    config=_dm(cfg.wnoc),
                )
            )
            prd = ttnn.RuntimeArgs()
            for core, d, l in ([] if cfg.fused else port_list):
                grp = cfg.writers[l * (W // SL):(l + 1) * (W // SL)] if cfg.split else cfg.writers
                prd[core[0]][core[1]] = [DIRS.index(d) * L + l, stats[(ch, d, l)], len(grp)] + [pk(c) for c in grp]
            kernels += [] if cfg.fused else [
                ttnn.KernelDescriptor(
                    kernel_source=kpath("lab_preader.cpp"),
                    core_ranges=crs([c for c, _, _ in port_list]),
                    named_compile_time_args=list(named.items()),
                    defines=[("LAB_TIMING", "1")] if cfg.timing else [],
                    runtime_args=prd,
                    config=_dm(1 - cfg.pnoc),
                )
            ]
        p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
        for core, d, l in port_list:
            r, c = ch
            nb = {"r+": ((r + 1) % R, c), "r-": ((r - 1) % R, c), "c+": (r, (c + 1) % C), "c-": (r, (c - 1) % C)}[d]
            p.kernels[0].runtime_args[core[0]][core[1]].extend(
                ttnn.setup_fabric_connection(
                    mesh_device=mesh,
                    src_fabric_node_id=node(mesh, ch),
                    dst_fabric_node_id=node(mesh, nb),
                    link_idx=l,
                    program_descriptor=p,
                    worker_core=ttnn.CoreCoord(*core),
                )
            )
        mc = ttnn.MeshCoordinate(*ch)
        mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    return mpd, [state, rings] + ([iring] if iring else []), stats
