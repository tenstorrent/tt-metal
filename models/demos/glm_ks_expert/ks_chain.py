# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Stage 2: GLM-5.3 gate/up cores on x chains (kernels/kc_*.cpp), one chip, generic_op.

A rectangle of gate/up cores (GLM's west rectangle: logical columns 2-5, rows 0-9) fed x along unicast chains as the
flat expert does: an injector (relay) unicasts each x block, in XP pieces, into its chain's head; every core forwards
pieces to its successor (NCRISC, NOC0, DEPTH in flight) once the successor has freed the block's slot. Chains run down
a column from the relay's row (wrapping inside the rectangle), then the next column. Weights resident (stage 2 measures
the x path and the partial exchange; DRAM weight streams come next).

  mode 0 (today): one chain through every core; each core one pair over all K (192 K tiles, 24 x blocks a sub-block).
  mode 1 (K split in pairs): columns paired (x0 + 2j, x0 + 2j + 1); chain A through the even columns carries K half 0,
    chain B through the odd ones half 1 (12 x blocks a sub-block each); partners exchange bf16 partials (one hop).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.glm_ks_expert.bench_util import (
    ACT_ARGS,
    GLM,
    TILE_BYTES,
    cb,
    crs,
    kpath,
    pcc,
    quantize,
    read_stamps,
    rel_err,
    stamps_tensor,
    to_dev,
)


@dataclass
class ChainCfg:
    mode: int = 1
    x0: int = 2
    y0: int = 0
    W: int = 4
    H: int = 10
    relay_x: int = 1
    relay_y: int = 4
    mt: int = 4  # row tiles a sub-block (128 rows)
    kb: int = 8  # K tiles an x block
    xs: int = 24  # x ring slots (blocks)
    xp: int = 4  # pieces a block
    depth: int = 6
    nsb: int = 48  # sub-blocks a launch (GLM e16 M320: 16 experts x 3)
    ps: int = 2  # partial ring slots
    w_stream: bool = False  # weights streamed from DRAM (each core's bank), one expert's blocks a region
    sbe: int = 3  # sub-blocks an expert (GLM M 320 -> 3 with a short last one; here all full)
    wpieces: int = 0  # reads a weight block (0: 1 today, 2 K split - the same bytes a read)
    wdepth: int = 2  # weight reads in flight
    geometry: object = None  # (device, ks) -> geometry dict; None: the rectangle above (columns paired)
    down_load: bool = False  # the down side as the flat expert's: h to the down chains, down compute, y to DRAM
    hbuf: int = 3  # h buffers a down core
    hp: int = 16  # pieces an h buffer on the down chains
    hdepth: int = 4
    down_chains: int = 7


def l1_tiles(device, cores, n_tiles, dtype):
    """A HEIGHT_SHARDED L1 tile tensor of n_tiles a core (its address the same on every core)."""
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(cores), [n_tiles * 32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        torch.zeros(len(cores) * n_tiles * 32, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
    )


def cb_on(index, tensor, core_ranges, n_tiles, dtype):
    """A CB over an L1 tensor's shard (globally allocated: its address the tensor's) on core_ranges."""
    d = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
    d.core_ranges = core_ranges
    d.total_size = n_tiles * TILE_BYTES[dtype]
    return d


def chain_order(cols, y0, H, start_row):
    """Down a column from start_row (wrapping inside rows y0 .. y0 + H - 1), then the next column from that row."""
    seq, row = [], start_row
    for x in cols:
        for r in range(H):
            y = y0 + (row - y0 + r) % H
            seq.append((x, y))
            if r + 1 == H:
                row = y
    return seq


def rect_geometry(cfg, ks):
    """One rectangle; K split pairs columns (x0 + 2j, x0 + 2j + 1): chain A the even columns, B the odd ones."""
    cols = list(range(cfg.x0, cfg.x0 + cfg.W))
    if ks:
        assert cfg.W % 2 == 0
        chains = [chain_order(cols[0::2], cfg.y0, cfg.H, cfg.relay_y), chain_order(cols[1::2], cfg.y0, cfg.H, cfg.relay_y + 1)]
        relays = [(cfg.relay_x, cfg.relay_y), (cfg.relay_x, cfg.relay_y + 1)]
        chain_half = [0, 1]
        partner = {c: ((c[0] + 1, c[1]) if (c[0] - cfg.x0) % 2 == 0 else (c[0] - 1, c[1])) for ch in chains for c in ch}
        half = {c: (c[0] - cfg.x0) % 2 for ch in chains for c in ch}
    else:
        chains = [chain_order(cols, cfg.y0, cfg.H, cfg.relay_y)]
        relays = [(cfg.relay_x, cfg.relay_y)]
        chain_half, partner, half = [0], {}, {}
    gu = [c for ch in chains for c in ch]
    pair_of = {c: (c[0] - cfg.x0) * cfg.H + (c[1] - cfg.y0) for c in gu}
    bank = {c: i % 8 for i, c in enumerate(gu)}
    slot = {c: i // 8 for i, c in enumerate(gu)}
    return dict(chains=chains, relays=relays, chain_half=chain_half, partner=partner, half=half, pair_of=pair_of, bank=bank, slot=slot)


def plan_geometry(pairing="snake", relay_rows=None):
    """The flat plan's GLM e16 cores (its gate/up rectangles, relays, banks: gate/up index ci in bank ci // 8; down
    cores and their banks). Today: one snake a rectangle (column-major from the primary relay's row, as the flat
    expert). K split, pairing "snake": consecutive snake positions pair up, chain A the even positions (K half 0, fed
    by the primary), B the odd ones (half 1, fed by the helper); "cols": columns paired (even-width rectangles)."""

    def build(device, ks):
        mbu = ttnn._ttnn.operations.experimental.moe_block
        d = mbu.flat_routed_expert_plan(
            device, GLM["H"], GLM["I"], 16, 256, 5120, weights_bf8=False, pin=1, layout=mbu.FlatLayoutOptions()
        ).as_dict()
        gcores = [tuple(c) for c in d["gate_up_cores"]]
        relays4 = [tuple(c) for c in d["relay_cores"]]  # primaries (rect 0, 1), helpers (rect 0, 1)
        bank = {c: i // 8 for i, c in enumerate(gcores)}
        slot = {c: i % 8 for i, c in enumerate(gcores)}
        rects = []
        for side in (0, 1):  # (west: left of the plan's relay column split, east)
            cs = [c for c in gcores if (c[0] < 7) == (side == 0)]
            xs_, ys_ = sorted({c[0] for c in cs}), sorted({c[1] for c in cs})
            rects.append((xs_[0], xs_[-1], ys_[0], ys_[-1]))
        chains, relays, chain_half, partner, half = [], [], [], {}, {}
        for k, (x0, x1, y0, y1) in enumerate(rects):
            H = y1 - y0 + 1
            yr = min(max(relays4[k][1], y0), y1)
            snake = chain_order(list(range(x0, x1 + 1)), y0, H, yr)
            if not ks:
                chains.append(snake)
                relays.append(relays4[k])
                chain_half.append(0)
                continue
            if pairing == "cols":
                cols = list(range(x0, x1 + 1))
                assert len(cols) % 2 == 0
                a = chain_order(cols[0::2], y0, H, yr)
                b = chain_order(cols[1::2], y0, H, yr)
                for c in a:
                    partner[c], partner[(c[0] + 1, c[1])] = (c[0] + 1, c[1]), c
            else:
                a, b = snake[0::2], snake[1::2]
                for ca, cb_ in zip(a, b):
                    partner[ca], partner[cb_] = cb_, ca
            for c in a:
                half[c] = 0
            for c in b:
                half[c] = 1
            chains += [a, b]
            relays += [relays4[k], relays4[2 + k]]
            chain_half += [0, 1]
        gu = [c for ch in chains for c in ch]
        if ks:  # a pair's two h pairs: the half-0 core owns 2j, its partner 2j + 1
            pair_of, j = {}, 0
            for c in gu:
                if half[c] == 0:
                    pair_of[c], pair_of[partner[c]] = 2 * j, 2 * j + 1
                    j += 1
        else:
            pair_of = {c: i for i, c in enumerate(gu)}
        down = [(tuple(c), pat[0]) for c, pat in zip(d["down_cores"], d["down_bank_patterns"])]
        return dict(chains=chains, relays=relays, chain_half=chain_half, partner=partner, half=half, pair_of=pair_of,
                    bank=bank, slot=slot, down=down)

    return build


def run_chain(device, cfg: ChainCfg, check=True):
    H_t, I_t = GLM["H"] // 32, GLM["I"] // 32
    ks = cfg.mode == 1
    nb = (H_t // 2 if ks else H_t) // cfg.kb
    kt = nb * cfg.kb
    wc = 4 if ks else 2
    g = cfg.geometry(device, ks) if cfg.geometry else rect_geometry(cfg, ks)
    chains, relays, chain_half = g["chains"], g["relays"], g["chain_half"]
    partner, half, w_bank, w_slot = g["partner"], g["half"], g["bank"], g["slot"]
    down = g.get("down", [])
    gu = [c for ch in chains for c in ch]
    idx = {c: i for i, c in enumerate(gu)}
    n = len(gu)
    pair_of = g["pair_of"]
    vxy = lambda c: device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
    pk = lambda c: (vxy(c).x << 16) | vxy(c).y

    # ---- data
    torch.manual_seed(0)
    rows = cfg.mt * 32
    x = torch.randn(rows, H_t * 32)
    w = torch.randn(H_t * 32, 2 * n * 32) / (H_t * 32) ** 0.5  # pair p: gate cols 64p .. 64p + 31, up 64p + 32 ..
    xq, wq = quantize(x, ttnn.bfloat8_b), quantize(w, ttnn.bfloat4_b)
    # x in block order: block b (K tiles b kb ..) tile (r, k) at b * mt * kb + r * kb + k
    xt = x.reshape(cfg.mt, 32, H_t, 32).permute(2, 0, 1, 3)  # [K tile][row tile][32][32]
    xblk = xt.reshape(H_t // cfg.kb, cfg.kb, cfg.mt, 32, 32).permute(0, 2, 1, 3, 4).reshape(-1, 32, 32)
    x_dev = to_dev(xblk.reshape(-1, 32), ttnn.bfloat8_b, device)
    gcol = lambda p: w[:, 64 * p : 64 * p + 32]
    ucol = lambda p: w[:, 64 * p + 32 : 64 * p + 64]
    wl = []
    for c in gu:
        p = pair_of[c]
        if ks:
            q = pair_of[partner[c]]
            k0 = half[c] * kt * 32
            sl = slice(k0, k0 + kt * 32)
            wl.append(torch.cat([gcol(q)[sl], ucol(q)[sl], gcol(p)[sl], ucol(p)[sl]], 1))
        else:
            wl.append(torch.cat([gcol(p), ucol(p)], 1))
    w_dev = to_dev(torch.cat(wl, 0), ttnn.bfloat4_b, device)
    ident = torch.zeros(64, 64)
    ident[:32, :32] = torch.eye(32)
    ident[32:, 32:] = torch.eye(32)
    i_dev = to_dev(ident, ttnn.bfloat16, device)
    h_dev = to_dev(torch.zeros(n * cfg.mt * 32, 32), ttnn.bfloat8_b, device)
    st = stamps_tensor(device, gu)

    # ---- L1 rings (same address on every core)
    # (L1 tensors take the same address on every core: the relays' x buffer (one sub-block's blocks) and the gate/up
    # cores' x ring share one tensor over both)
    wpieces = cfg.wpieces or (2 if ks else 1)
    ws = 2 * nb  # weight ring: two experts
    per_bank = {}
    for c in gu:
        per_bank[w_bank[c]] = max(per_bank.get(w_bank[c], 0), w_slot[c] + 1)
    wdram = None
    if cfg.w_stream:
        R = kt * wc  # tiles a core's region (one expert)
        S = max(per_bank.values()) * R
        shards = torch.zeros(8, S, 32, 32)
        for c in gu:
            tiles = wl[idx[c]].reshape(kt, 32, wc, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)
            shards[w_bank[c], w_slot[c] * R : (w_slot[c] + 1) * R] = tiles
        dram_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [S * 32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        wdram = ttnn.from_torch(
            shards.reshape(-1, 32), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
        )
    down_cores = [c for c, _ in down] if cfg.down_load else []
    D_BYTES = I_t * (H_t // 32) * TILE_BYTES[ttnn.bfloat4_b]  # a down core's weights an expert (6 columns, K = I)
    heads = []
    # (one L1 tensor over the relays' x buffers, the gate/up x rings and the down cores' weight rings)
    xring = l1_tiles(device, gu + relays + down_cores, max(cfg.xs, nb) * cfg.mt * cfg.kb, ttnn.bfloat8_b)
    assert not down_cores or 2 * D_BYTES <= max(cfg.xs, nb) * cfg.mt * cfg.kb * TILE_BYTES[ttnn.bfloat8_b]
    ddram = None
    DC, DKB = H_t // 32, 8  # y tile columns a down core (6), K tiles a down weight block
    if down_cores:
        nd = len(down_cores)
        assert nd * DC == H_t
        wd = torch.randn(I_t * 32, H_t * 32) / (I_t * 32) ** 0.5
        dper, dslot = {}, {}
        for c, b in down:
            dslot[c] = dper.get(b, 0)
            dper[b] = dslot[c] + 1
        DR = I_t * DC  # tiles a down core's region (one expert)
        DS = max(dper.values()) * DR
        dsh = torch.zeros(8, DS, 32, 32)
        for d, (c, b) in enumerate(down):
            reg = wd[:, d * DC * 32 : (d + 1) * DC * 32].reshape(I_t, 32, DC, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)
            dsh[b, dslot[c] * DR : (dslot[c] + 1) * DR] = reg
        dmem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))]),
                [DS * 32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        ddram = ttnn.from_torch(dsh.reshape(-1, 32), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dmem)
        y_dev = to_dev(torch.zeros(nd * cfg.mt * DC * 32, 32), ttnn.bfloat8_b, device)
        # down chains: column-major, split into down_chains runs; the coordinator is the first head
        order = sorted(down_cores)
        sizes = [len(order) // cfg.down_chains + (1 if i < len(order) % cfg.down_chains else 0) for i in range(cfg.down_chains)]
        dchains, at = [], 0
        for sz in sizes:
            dchains.append(order[at : at + sz])
            at += sz
        heads = [ch[0] for ch in dchains]
    xbuf = xring
    pring = l1_tiles(device, gu, cfg.ps * cfg.mt * 2, ttnn.bfloat16)

    named = [
        ("mode", cfg.mode),
        ("mt", cfg.mt),
        ("kb", cfg.kb),
        ("nb", nb),
        ("xs", cfg.xs),
        ("xp", cfg.xp),
        ("depth", cfg.depth),
        ("nsb", cfg.nsb),
        ("wc", wc),
        ("ps", cfg.ps),
        ("x_tile_bytes", TILE_BYTES[ttnn.bfloat8_b]),
        ("p_tile_bytes", TILE_BYTES[ttnn.bfloat16]),
        ("w_stream", int(cfg.w_stream)),
        ("sbe", cfg.sbe),
        ("ws", ws),
        ("wpieces", wpieces),
        ("wdepth", cfg.wdepth),
        ("w_tile_bytes", TILE_BYTES[ttnn.bfloat4_b]),
        ("hbuf", cfg.hbuf),
        ("ngu", n),
    ]
    ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    g_crs, r_crs = crs(gu), crs(relays)
    inj_rt, fwd_rt, rcv_rt, cmp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for k, (ch, rl) in enumerate(zip(chains, relays)):
        inj_rt[rl[0]][rl[1]] = [x_dev.buffer_address(), chain_half[k] * nb * cfg.mt * cfg.kb, xbuf.buffer_address(), pk(ch[0]), xring.buffer_address()]
        for j, c in enumerate(ch):
            pred = pk(ch[j - 1]) if j else 0
            succ = pk(ch[j + 1]) if j + 1 < len(ch) else 0
            fwd_rt[c[0]][c[1]] = [xring.buffer_address(), pred, succ, pk(rl)] + (
                [wdram.buffer_address() + w_slot[c] * kt * wc * TILE_BYTES[ttnn.bfloat4_b], w_bank[c]] if cfg.w_stream else [0, 0]
            )
            rcv_rt[c[0]][c[1]] = [
                w_dev.buffer_address(),
                idx[c] * kt * wc,
                pk(partner[c]) if ks else 0,
                h_dev.buffer_address(),
                idx[c] * cfg.mt,
                i_dev.buffer_address(),
                pring.buffer_address(),
            ] + ([len(heads), xring.buffer_address(), pair_of[c]] + [pk(hd) for hd in heads] if down_cores else [0, 0, 0])
            cmp_rt[c[0]][c[1]] = [st.buffer_address()]
    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=kpath("kc_inject.cpp"),
            core_ranges=r_crs,
            compile_time_args=ta(x_dev),
            named_compile_time_args=named,
            runtime_args=inj_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("kc_forward.cpp"),
            core_ranges=g_crs,
            named_compile_time_args=named,
            runtime_args=fwd_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("kc_receive.cpp"),
            core_ranges=g_crs,
            compile_time_args=ta(w_dev) + ta(h_dev) + ta(i_dev),
            named_compile_time_args=named,
            runtime_args=rcv_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("kc_compute.cpp"),
            core_ranges=g_crs,
            named_compile_time_args=named + ACT_ARGS,
            runtime_args=cmp_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=False
            ),
        ),
    ]
    if down_cores:
        dnamed = [
            ("nsb", cfg.nsb), ("sbe", cfg.sbe), ("mt", cfg.mt), ("ngu", n), ("hbuf", cfg.hbuf), ("hp", cfg.hp),
            ("hdepth", cfg.hdepth), ("dc", DC), ("dkb", DKB), ("dpieces", 4), ("ddepth", 2), ("nd", nd),
            ("h_tile_bytes", TILE_BYTES[ttnn.bfloat8_b]), ("w_tile_bytes", TILE_BYTES[ttnn.bfloat4_b]),
            ("y_tile_bytes", TILE_BYTES[ttnn.bfloat8_b]),
        ]  # fmt: skip
        d_crs = crs(down_cores)
        dl_rt, dr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        coord = heads[0]
        dindex = {c: d for d, (c, _) in enumerate(down)}
        for c, b in down:
            dl_rt[c[0]][c[1]] = [ddram.buffer_address() + dslot[c] * DR * TILE_BYTES[ttnn.bfloat4_b], b]
        for ch in dchains:
            for j, c in enumerate(ch):
                succ = pk(ch[j + 1]) if j + 1 < len(ch) else 0
                dr_rt[c[0]][c[1]] = [
                    succ, xring.buffer_address(), int(j == 0), pk(coord), int(c == coord), y_dev.buffer_address(),
                    dindex[c] * cfg.mt * DC,
                ] + ([n] + [pk(g_) for g_ in gu] if c == coord else [0])  # fmt: skip
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=kpath("kd_load.cpp"), core_ranges=d_crs, named_compile_time_args=dnamed,
                runtime_args=dl_rt, config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("kd_recv.cpp"), core_ranges=d_crs, compile_time_args=ta(y_dev),
                named_compile_time_args=dnamed, runtime_args=dr_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("kd_compute.cpp"), core_ranges=d_crs, named_compile_time_args=dnamed,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, dst_full_sync_en=False),
            ),
        ]  # fmt: skip
    cbs = [
        cb_on(0, xring, g_crs, cfg.xs * cfg.mt * cfg.kb, ttnn.bfloat8_b),
        cb(1, ttnn.bfloat4_b, (ws * cfg.kb * wc) if cfg.w_stream else kt * wc, g_crs),
        ttnn.cb_descriptor_from_sharded_tensor(2, pring),
        cb(3, ttnn.bfloat16, 4, g_crs),
        cb(16, ttnn.bfloat8_b, 2 * cfg.mt, g_crs),
        cb(17, ttnn.bfloat16, 2 * cfg.mt * 2, g_crs),
    ]
    if down_cores:
        cbs += [
            cb(4, ttnn.bfloat4_b, 2 * (I_t // DKB) * DKB * DC, d_crs),
            cb_on(5, xring, d_crs, cfg.hbuf * n * cfg.mt, ttnn.bfloat8_b),
            cb(18, ttnn.bfloat8_b, 2 * cfg.mt * DC, d_crs),
        ]
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=crs(gu + relays + down_cores), initial_value=0) for s in range(13)]
    io = [x_dev, w_dev, i_dev, h_dev, st, xring, pring] + [t for t in (wdram, ddram) if t is not None] + ([y_dev] if down_cores else [])
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
    ttnn.synchronize_device(device)

    s = read_stamps(st, n)
    t0, t1, t2 = s[:, 0], s[:, 1], s[:, 2]
    start = int(t0.min())
    first = float((t1 - start).median()) / 1350
    total = float((t2 - start).max()) / 1350
    steady = float(((t2 - t1) & 0xFFFFFFFF).median()) / max(1, cfg.nsb - 1) / 1350
    out = dict(cores=n, total_us=total, first_sb_us=first, steady_sb_us=steady, nb=nb)
    if cfg.w_stream:
        out["dram_gbs"] = cfg.nsb / cfg.sbe * (n * kt * wc * TILE_BYTES[ttnn.bfloat4_b] + len(down_cores) * D_BYTES) / (total * 1e3)
    if check:
        h = ttnn.to_torch(h_dev).float().reshape(n, cfg.mt * 32, 32)
        if cfg.w_stream:  # the weights as the device holds them (the sharded upload rounds bfp4 its own way)
            back = ttnn.to_torch(wdram).float().reshape(8, -1, 32, 32)
            R = kt * wc
            region = lambda c: (
                back[w_bank[c], w_slot[c] * R : (w_slot[c] + 1) * R]
                .reshape(kt, wc, 32, 32)
                .permute(0, 2, 1, 3)
                .reshape(kt * 32, wc * 32)
            )
        else:
            region = lambda c: quantize(wl[idx[c]], ttnn.bfloat4_b)

        def own(c):  # this core's pair's (gate, up) over all K, as the device holds it
            m = region(c)
            if not ks:
                return m[:, :32], m[:, 32:64]
            mp = region(partner[c])
            g, u = torch.zeros(H_t * 32, 32), torch.zeros(H_t * 32, 32)
            mine, other = slice(half[c] * kt * 32, (half[c] + 1) * kt * 32), slice((1 - half[c]) * kt * 32, (2 - half[c]) * kt * 32)
            g[mine], u[mine] = m[:, 64:96], m[:, 96:128]
            g[other], u[other] = mp[:, 0:32], mp[:, 32:64]
            return g, u

        errs, pccs = [], []
        for c in gu:
            g, u = own(c)
            ref = torch.nn.functional.silu(xq @ g) * (xq @ u)
            errs.append(rel_err(h[idx[c]], ref))
            pccs.append(pcc(h[idx[c]], ref))
        out["max_err"] = max(errs)
        out["min_pcc"] = min(pccs)
        if down_cores:  # y of the last sub-block from h as the device holds it (bfp8) and the device's down weights
            hcols = torch.zeros(cfg.mt * 32, n * 32)
            for c in gu:
                g_, u_ = own(c)
                hcols[:, 32 * pair_of[c] : 32 * pair_of[c] + 32] = torch.nn.functional.silu(xq @ g_) * (xq @ u_)
            hq = quantize(hcols, ttnn.bfloat8_b)
            dback = ttnn.to_torch(ddram).float().reshape(8, -1, 32, 32)
            yd = ttnn.to_torch(y_dev).float().reshape(nd, cfg.mt, DC, 32, 32)
            ypcc = []
            for d, (c, b) in enumerate(down):
                wreg = dback[b, dslot[c] * DR : (dslot[c] + 1) * DR].reshape(I_t, DC, 32, 32).permute(0, 2, 1, 3).reshape(I_t * 32, DC * 32)
                yref = hq @ wreg
                ygot = yd[d].permute(0, 2, 1, 3).reshape(cfg.mt * 32, DC * 32)
                ypcc.append(pcc(ygot, yref))
            out["y_min_pcc"] = min(ypcc)
    return out
