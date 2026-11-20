#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""noc_a2a.py: N2, run.sh's on-chip network-on-chip all-to-all check,
right after G3b (hc/fleet/chunk_load.py rw_after).

Every pass moves one uint32 ROW_MAJOR tensor over the whole compute grid of every chip and reads it back after each leg:
  host>DRAM   host write to interleaved DRAM (page r in bank r % banks)
  DRAM>L1h    interleaved DRAM to height-sharded L1 (rows split over every core of the grid; every bank to every core)
  L1h>L1w     height-sharded to width-sharded L1 (every core sends a slice to every other core: the all-to-all)
  L1w>L1h     width-sharded back to height-sharded L1 (the all-to-all the other way)
  L1h>DRAM    height-sharded L1 back to interleaved DRAM (every core to every bank)
  L1mc        right after DRAM>L1h, one generic_op kernel on every core: the 2D multicast of the matmul
              (row and column senders, as the 2D-multicast matmul's in0 and in1 senders). Each pass one column of row
              senders (x = (pass // 2) % gx) multicasts the first MCW words of its height shard to every core of its row, and
              one row of column senders (y = (pass // 2) % gy) to every core of its column, then increments two semaphores on
              each receiver INC times (one per sender direction). Every receiver waits for both counts and records them;
              each must be exactly INC (a lost or doubled increment, or a wait that timed out, is a semaphore FAIL).
              Even passes run on NOC0 (data movement RISC 0), odd passes on NOC1 (RISC 1, multicast corners swapped).
The shard copies (to_memory_config) read on NOC0 and write on NOC1; chunk_load.py logs the NOC of every leg (NOCS).
Each readback is compared bit for bit with the host's own copy (L1mc with the DRAM>L1h readback it multicasts). A word
is charged to the first leg where it went wrong, with that leg's source and destination (logical core x,y in the grid,
ROW_MAJOR shard order, or DRAM bank). The value of word (row, col) is a hash of row, col and pass, and odd passes invert
it, so every bit toggles. A mismatch that most chips share word for word is the op's or software's (a WARN, never a chip
FAIL); a chip with any other bad word or semaphore count is its NUMERIC FAIL (fleet/chunk_sched.noc_lines), named by
BDF, tray 1-4 and ASIC. Passes loop until the wall cap. If this ttnn cannot build or run the multicast kernel, L1mc is
'not checked' (a WARN) and the other legs go on.
"""
import re
import time

RPC, CPC = 8, 16  # rows and 32-bit words per core: the tensor is (cores*RPC) x (cores*CPC) words
LEGS = ["host>DRAM", "DRAM>L1h", "L1h>L1w", "L1w>L1h", "L1h>DRAM"]
MC = "L1mc"  # the multicast and semaphore leg
MCW = 64  # words each multicast sender sends (256 B, the first MCW words of its height shard)
INC = 4  # increments each sender makes on each receiver's semaphore
SPIN = 1 << 22  # a receiver's wait for its counts, in polls (well under a second), then 'timed out'
NOCS = {
    "host>DRAM": "host write over PCIe",
    **{L: "reader NOC0, writer NOC1" for L in LEGS[1:]},
    MC: "NOC0 (RISC 0) on even passes, NOC1 (RISC 1) on odd passes",
}
HOST = -1000  # end codes: >= 0 core index, -1-b DRAM bank b, HOST the host
KEEP = 8  # example words kept per chip

# L1mc's kernel, runtime args: src, dst, bytes, INC, SPIN, then per direction (row, column): is_sender, the multicast
# rectangle's NOC start x,y and end x,y, the receiver count nd, and the nd receivers' NOC x,y. The output shard per core
# is 3 rows of MCW words: the row sender's data, the column sender's data, then [row count, column count, timed out].
MC_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const uint32_t src = get_arg_val<uint32_t>(i++);
    const uint32_t dst = get_arg_val<uint32_t>(i++);
    const uint32_t nbytes = get_arg_val<uint32_t>(i++);
    const uint32_t inc = get_arg_val<uint32_t>(i++);
    const uint32_t spin = get_arg_val<uint32_t>(i++);
    const uint32_t sem[2] = {get_semaphore(0), get_semaphore(1)};
    for (uint32_t d = 0; d < 2; d++) {
        const uint32_t sender = get_arg_val<uint32_t>(i++);
        const uint32_t x0 = get_arg_val<uint32_t>(i++);
        const uint32_t y0 = get_arg_val<uint32_t>(i++);
        const uint32_t x1 = get_arg_val<uint32_t>(i++);
        const uint32_t y1 = get_arg_val<uint32_t>(i++);
        const uint32_t nd = get_arg_val<uint32_t>(i++);
        if (sender) {
            const uint64_t mc = get_noc_multicast_addr(x0, y0, x1, y1, dst + d * nbytes);
            noc_async_write_multicast_loopback_src(src, mc, nbytes, nd);
            noc_async_write_barrier();
            for (uint32_t r = 0; r < nd; r++) {
                const uint32_t rx = get_arg_val<uint32_t>(i + 2 * r);
                const uint32_t ry = get_arg_val<uint32_t>(i + 2 * r + 1);
                const uint64_t a = get_noc_addr(rx, ry, sem[d]);
                for (uint32_t j = 0; j < inc; j++) {
                    noc_semaphore_inc(a, 1);
                }
            }
            noc_async_atomic_barrier();
        }
        i += 2 * nd;
    }
    volatile tt_l1_ptr uint32_t* sr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem[0]);
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem[1]);
    uint32_t n = 0;
    while ((sr[0] < inc || sc[0] < inc) && n < spin) {
        invalidate_l1_cache();
        n++;
    }
    for (uint32_t j = 0; j < spin / 64; j++) {  // settle: a surplus increment lands before the counts are read
        invalidate_l1_cache();
    }
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + 2 * nbytes);
    st[0] = sr[0];
    st[1] = sc[0];
    st[2] = n >= spin;
}
"""


def grid(mesh):
    """(gx, gy): the compute grid every chip shards over."""
    try:
        g = mesh.compute_with_storage_grid_size()
    except Exception:  # noqa: BLE001
        g = mesh.get_devices()[0].compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def end_name(e, gx):
    """An end code's name: 'core x,y', 'bank b' or 'host'."""
    e = int(e)
    if e == HOST:
        return "host"
    if e < 0:
        return f"bank {-1 - e}"
    return f"core {e % gx},{e // gx}"


def gold_data(torch, R, C, k):
    """Pass k's R x C words (int32 view of uint32): a hash of (row, col, pass), inverted on odd passes."""
    v = torch.arange(R * C, dtype=torch.int64) * 2654435761 + (k + 1) * 0x632BE5AB
    v = v & 0xFFFFFFFF
    v = v ^ (v >> 15)
    v = (v * 0x2C1B3C6D) & 0xFFFFFFFF
    v = v ^ (v >> 12)
    if k % 2:
        v = v ^ 0xFFFFFFFF
    v = torch.where(v >= 2**31, v - 2**32, v)
    return v.to(torch.int32).reshape(R, C)


def leg_ends(torch, R, C, n, nb):
    """Per leg, (src, dst) end codes as R x 1 or 1 x C tensors (broadcast over the R x C words)."""
    r = torch.arange(R).reshape(R, 1)
    c = torch.arange(C).reshape(1, C)
    bank = -1 - (r % nb)
    rc = r // (R // n)  # height shard: the core holding row r
    cc = c // (C // n)  # width shard: the core holding column c
    host = torch.full((R, 1), HOST)
    return [(host, bank), (bank, rc), (rc, cc), (cc, rc), (rc, bank)]


def mcfg(ttnn, kind, crs, R, C, n):
    """A height ('h') or width ('w') sharded L1 memory config over the whole grid."""
    shape = [R // n, C] if kind == "h" else [R, C // n]
    lay = ttnn.TensorMemoryLayout.HEIGHT_SHARDED if kind == "h" else ttnn.TensorMemoryLayout.WIDTH_SHARDED
    return ttnn.MemoryConfig(lay, ttnn.BufferType.L1, ttnn.ShardSpec(crs, shape, ttnn.ShardOrientation.ROW_MAJOR))


def inject(reads, chips, k, R, n, gx):
    """With TRAY_INJECT naming N2 and a chip: one bit flipped in that chip's DRAM>L1h readback, in the first word of the
    spec's core's height shard (default core 0,0), so the word's leg is 'DRAM>L1h bank b -> core x,y'."""

    import tray_inject

    for i in range(len(reads[1])):
        s = tray_inject.want("N2", chips.get(i, {}).get("pci"), k)
        if s is None:
            continue
        x, y = s["core"] or (0, 0)
        e = (y * gx + x) % n
        tray_inject.applied(s, core=(e % gx, e // gx), where="leg=DRAM>L1h")
        reads[1][i] = tray_inject.flip_tensor(s, reads[1][i], e * (R // n) * reads[1][i].shape[1])


def mc_senders(k, gx, gy):
    """Pass k's (row senders' column, column senders' row) and NOC (0 or 1): k's NOC alternates and the senders move on
    every second pass, so every sender column and row gets both NOCs, even grid sides included."""
    return (k // 2) % gx, (k // 2) % gy, k % 2


def mc_program(ttnn, mesh, crs, gx, gy, k, src, dst):
    """Pass k's L1mc program (MC_KERNEL on every core, two semaphores at 0) reading the height shard at src and writing
    the output shard at dst."""
    cx, cy, p = mc_senders(k, gx, gy)
    wc = {}
    for y in range(gy):
        for x in range(gx):
            c = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            wc[x, y] = (int(c.x), int(c.y))

    def rect(a, b):  # NOC0 multicasts from the low corner to the high one, NOC1 the other way round
        lo = (min(wc[a][0], wc[b][0]), min(wc[a][1], wc[b][1]))
        hi = (max(wc[a][0], wc[b][0]), max(wc[a][1], wc[b][1]))
        return [*hi, *lo] if p else [*lo, *hi]

    rt = ttnn.RuntimeArgs()
    for y in range(gy):
        for x in range(gx):
            row = [int(x == cx), *rect((0, y), (gx - 1, y)), gx] + [v for i in range(gx) for v in wc[i, y]]
            col = [int(y == cy), *rect((x, 0), (x, gy - 1)), gy] + [v for j in range(gy) for v in wc[x, j]]
            rt[x][y] = [src, dst, MCW * 4, INC, SPIN] + row + col
    dm = ttnn.DataMovementProcessor
    kd = ttnn.KernelDescriptor(
        kernel_source=MC_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs,
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=dm.RISCV_1 if p else dm.RISCV_0, noc=ttnn.NOC.NOC_1 if p else ttnn.NOC.NOC_0
        ),
    )
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0) for i in (0, 1)]
    return ttnn.ProgramDescriptor(kernels=[kd], semaphores=sems, cbs=[])


def mc_expect(torch, h, k, gx, gy):
    """The L1mc output (3*cores x MCW words) a chip should return, from its own DRAM>L1h readback h (so a word already
    bad in that leg is not charged again), and the (src, dst) end code of each of its rows."""
    n = gx * gy
    cx, cy, _ = mc_senders(k, gx, gy)
    exp = torch.zeros(3 * n, MCW, dtype=torch.int32)
    src, dst = [], []
    for y in range(gy):
        for x in range(gx):
            e = y * gx + x
            exp[3 * e] = h[(y * gx + cx) * RPC, :MCW]
            exp[3 * e + 1] = h[(cy * gx + x) * RPC, :MCW]
            exp[3 * e + 2, :2] = INC
            src += [y * gx + cx, cy * gx + x, e]
            dst += [e, e, e]
    return exp, torch.tensor(src).reshape(-1, 1), torch.tensor(dst).reshape(-1, 1)


def mc_inject(o, chips, k, gx):
    """With TRAY_INJECT naming N2m and a chip: one bit flipped in the row-semaphore count that chip's spec core (default
    core 0,0) recorded, so the semaphore check must name that chip and core."""

    import tray_inject

    for i in range(len(o)):
        s = tray_inject.want("N2m", chips.get(i, {}).get("pci"), k)
        if s is None:
            continue
        x, y = s["core"] or (0, 0)
        e = (y * gx + x) % (o[i].shape[0] // 3)
        tray_inject.applied(s, core=(e % gx, e // gx), where="leg=L1mc semaphore")
        o[i] = tray_inject.flip_tensor(s, o[i], (3 * e + 2) * MCW)


SETUP_ERRS = (AttributeError, ImportError, NotImplementedError, TypeError)
SETUP_RE = re.compile(r"build|compil|generate binaries|not supported|unsupported", re.I)
# word starts only, a letter not before: "exchange" inside a build error is not a hang, "op_timeout" and "x_hang" are
HANG_RE = re.compile(r"(?<![A-Za-z])(?:time ?out|timed out|hang|waiting for)", re.I)


def mc_setup_error(e):
    """True for an L1mc error that leaves the leg not checked (a NUMERIC WARN): an older ttnn without generic_op or its
    classes, or a kernel that does not build. False for the rest, a FAIL: a multicast or semaphore wait that timed out,
    a dispatch error. Only the message before its backtrace is matched (every backtrace names the program compile)."""
    msg = str(e).split("backtrace:")[0]
    if isinstance(e, TimeoutError) or HANG_RE.search(msg):
        return False
    return isinstance(e, SETUP_ERRS) or bool(SETUP_RE.search(msg))


def a2a_check(ttnn, torch, mesh, chips, wall_s, nb=8, beat=None, say=None, k0=0):
    """N2: the legs above, read back and compared, on every chip until wall_s s. chips: mesh index -> dict(pci, tray,
    dev). k0: the L1mc steps of this run's earlier chunks, so the senders go on rotating across chunks. The first pass
    runs two L1mc steps (NOC0 and NOC1), each later pass one (the wall loop estimates a pass by one step); mc_passes
    counts the steps. Returns
    dict(passes, legs, nocs, grid, cores, mb, wall_s, banks, mc_passes, mc_nocs, mc_error, mc_fail (L1mc timed out or
    failed to dispatch: N2 stops there, no chip named), common=[shared
    mismatches], chips={mesh index: dict(chip, passes, bad_passes, bad (words), first_pass, first_t, per_leg, xor, pairs
    ({'leg|src|dst': words}), examples, sem (wrong semaphore counts), sem_ex)})."""
    gx, gy = grid(mesh)
    n = gx * gy
    R, C = n * RPC, n * CPC
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    H, W, D = mcfg(ttnn, "h", crs, R, C, n), mcfg(ttnn, "w", crs, R, C, n), ttnn.DRAM_MEMORY_CONFIG
    O = mcfg(ttnn, "h", crs, 3 * n, MCW, n)  # L1mc's output: 3 rows of MCW words per core
    ends = leg_ends(torch, R, C, n, nb)
    new = lambda: dict(  # noqa: E731
        passes=0,
        bad_passes=0,
        bad=0,
        first_pass=None,
        first_t=None,
        per_leg={L: 0 for L in LEGS + [MC]},
        xor=0,
        pairs={},
        examples=[],
        sem=0,
        sem_ex=[],
    )
    st = {i: new() for i in chips}
    common = []
    mc_err, mc_fail, mc_passes, mc_nocs = None, None, 0, set()
    t0 = time.monotonic()
    k, last = 0, 0.0

    def read(x, shape):
        return [
            ttnn.to_torch(d).contiguous().view(torch.int32).reshape(shape) for d in ttnn.get_device_tensors(x.cpu())
        ]

    def vote(L, sig, nchip):
        """The chips whose mismatch most chips share word for word (the op's or software's)."""
        shared = set()
        if nchip >= 3:
            for key, who in sig.items():
                if len(who) > nchip // 2:
                    shared.update(who)
                    common.append(dict(leg=L, k=k, chips=len(who), of=nchip, words=key[0]))
                    if say:
                        say(
                            f"{L} pass {k}: {len(who)} of {nchip} chips differ the same way ({key[0]} words): "
                            f"the op's or software's, not a chip's"
                        )
        return shared

    def charge(s, L, g, w, sr, ds, r, c):
        """Adds bad words (got g, want w, source and destination end codes sr, ds at rows r, cols c) to chip stats s."""
        s["bad"] += int(r.numel())
        s["per_leg"][L.split()[0]] += int(r.numel())
        xb = torch.bitwise_xor(g, w).to(torch.int64) & 0xFFFFFFFF  # int64: bit 31 overflows an int32 mask
        for bit in range(32):
            if bool(((xb >> bit) & 1).any()):
                s["xor"] |= 1 << bit
        for (a, b), cnt in zip(*torch.unique(torch.stack([sr, ds], 1), dim=0, return_counts=True)):
            key = f"{L}|{end_name(a, gx)}|{end_name(b, gx)}"
            s["pairs"][key] = s["pairs"].get(key, 0) + int(cnt)
        for j in range(min(KEEP - len(s["examples"]), int(r.numel()))):
            s["examples"].append(
                dict(
                    leg=L,
                    k=k,
                    row=int(r[j]),
                    col=int(c[j]),
                    src=end_name(sr[j], gx),
                    dst=end_name(ds[j], gx),
                    got=int(g[j]) & 0xFFFFFFFF,
                    want=int(w[j]) & 0xFFFFFFFF,
                )
            )

    def first(s):
        if s["first_pass"] is None:
            s["first_pass"], s["first_t"] = k, round(time.time(), 1)

    while time.monotonic() - t0 + last < wall_s or k == 0:
        p0 = time.monotonic()
        extra = 0.0  # pass 0's second L1mc step, run and compared: left out of the per-pass estimate (one step a pass)
        gold = gold_data(torch, R, C, k)
        reads, live, mcs = [], [], []  # mcs: this pass's L1mc steps, (step, readback, expected)
        try:
            x = ttnn.from_torch(
                gold,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=D,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            live.append(x)
            for li, cfg in enumerate([None, H, W, H, D]):
                if li:
                    x = ttnn.to_memory_config(x, cfg)
                    live.append(x)
                ttnn.synchronize_device(mesh)
                reads.append(read(x, (R, C)))
                if beat:
                    beat(f"noc pass {k} leg {li}")
                # L1mc multicasts from the height shards DRAM>L1h just wrote. Step j runs on NOC j%2 (mc_senders);
                # the first pass runs two steps, so both NOCs are checked even when the wall allows one pass (plain)
                tried = li == 1 and mc_err is None
                for j in range(k0 + mc_passes, k0 + mc_passes + (2 if k == 0 else 1)) if tried else ():
                    if mc_err is not None:
                        break
                    s0 = time.monotonic()
                    try:
                        o = ttnn.from_torch(
                            torch.zeros(3 * n, MCW, dtype=torch.int32),
                            dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=mesh,
                            memory_config=O,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                        )
                        live.append(o)
                        prog = mc_program(ttnn, mesh, crs, gx, gy, j, x.buffer_address(), o.buffer_address())
                        ttnn.generic_op([x, o], prog)
                        ttnn.synchronize_device(mesh)
                        mcs.append((j, read(o, (3 * n, MCW)), [mc_expect(torch, h, j, gx, gy) for h in reads[1]]))
                        mc_passes += 1
                        mc_nocs.add(j % 2)
                        if len(mcs) > 1:
                            extra += time.monotonic() - s0
                    except Exception as e:  # noqa: BLE001  (mc_setup_error sorts not checked from failed)
                        mc_err = f"{type(e).__name__}: {str(e)[:300]}"
                        if not mc_setup_error(e):
                            mc_fail = mc_err  # the mesh may be hung: no more legs or passes on it
                            if say:
                                say(f"{MC} FAILED in pass {k} (timed out or failed to dispatch): {mc_err}")
                        elif say:
                            say(f"{MC} not checked: {mc_err}")
                if mc_fail:
                    break
                if tried and beat:
                    beat(f"noc pass {k} leg {MC}")
        finally:
            for t in live:
                try:
                    ttnn.deallocate(t)
                except Exception:  # noqa: BLE001
                    pass
        if mc_fail:
            break
        inject(reads, chips, k, R, n, gx)  # TRAY_INJECT: test-only flip of one chip's host copy, before the compare
        nchip = len(reads[0])
        prev = [None] * nchip
        bad_pass = set()
        for li, L in enumerate(LEGS):
            src, dst = ends[li]
            nw, sig = [], {}
            for i in range(nchip):
                d = reads[li][i] != gold
                m = d if prev[i] is None else d & ~prev[i]
                prev[i] = d
                nw.append(m)
                cnt = int(m.sum())
                if cnt:
                    idx = torch.nonzero(m.reshape(-1)).reshape(-1)[:32]
                    key = (cnt, tuple(idx.tolist()), tuple(reads[li][i].reshape(-1)[idx].tolist()))
                    sig.setdefault(key, []).append(i)
            shared = vote(L, sig, nchip)
            for i in range(nchip):
                m = nw[i]
                if i in shared or not bool(m.any()):
                    continue
                s = st.setdefault(i, new())
                r, c = torch.nonzero(m, as_tuple=True)
                bad_pass.add(i)
                first(s)
                sr = src.expand(R, C)[r, c]
                ds = dst.expand(R, C)[r, c]
                charge(s, L, reads[li][i][r, c], gold[r, c], sr, ds, r, c)
                if say:
                    ch = chips.get(i, {})
                    say(
                        f"BAD mesh{i} {ch.get('pci')} tray {ch.get('tray') or '?'}: {int(r.numel())} words in {L} "
                        f"pass {k}, first {end_name(sr[0], gx)} -> {end_name(ds[0], gx)}"
                    )
        m1 = None
        for si, (j, mco, mce) in enumerate(mcs):
            if si == 1:
                m1 = time.monotonic()
            if not si:
                mc_inject(mco, chips, k, gx)  # TRAY_INJECT: test-only flip of one chip's semaphore count
            cx, cy, p = mc_senders(j, gx, gy)
            L = f"{MC} NOC{p}"
            diffs, sig = [], {}
            for i in range(nchip):
                d = mco[i] != mce[i][0]
                diffs.append(d)
                cnt = int(d.sum())
                if cnt:
                    idx = torch.nonzero(d.reshape(-1)).reshape(-1)[:32]
                    key = (cnt, tuple(idx.tolist()), tuple(mco[i].reshape(-1)[idx].tolist()))
                    sig.setdefault(key, []).append(i)
            shared = vote(MC, sig, nchip)
            for i in range(nchip):
                d = diffs[i]
                if i in shared or not bool(d.any()):
                    continue
                s = st.setdefault(i, new())
                bad_pass.add(i)
                first(s)
                exp, msrc, mdst = mce[i]
                data = d.clone()
                data[2::3] = False
                r, c = torch.nonzero(data, as_tuple=True)
                if r.numel():
                    sr, ds = msrc.expand(3 * n, MCW)[r, c], mdst.expand(3 * n, MCW)[r, c]
                    charge(s, L, mco[i][r, c], exp[r, c], sr, ds, r, c)
                nsem = 0
                for e in torch.nonzero(d[2::3].any(1)).reshape(-1).tolist():
                    got = [int(v) & 0xFFFFFFFF for v in mco[i][3 * e + 2, :3]]
                    nsem += 1
                    s["sem"] += 1
                    if len(s["sem_ex"]) < KEEP:
                        s["sem_ex"].append(
                            dict(
                                k=k,
                                core=end_name(e, gx),
                                noc=f"NOC{p}",
                                row=got[0],
                                col=got[1],
                                want=INC,
                                timed_out=bool(got[2]),
                                row_sender=end_name(e // gx * gx + cx, gx),
                                col_sender=end_name(cy * gx + e % gx, gx),
                            )
                        )
                if say:
                    ch = chips.get(i, {})
                    say(
                        f"BAD mesh{i} {ch.get('pci')} tray {ch.get('tray') or '?'}: {int(r.numel())} words and {nsem} "
                        f"semaphore counts wrong in {L} pass {k}"
                    )
        if m1 is not None:
            extra += time.monotonic() - m1
        for i in range(nchip):
            s = st.setdefault(i, new())
            s["passes"] += 1
            s["bad_passes"] += i in bad_pass
        k += 1
        last = time.monotonic() - p0 - extra
    el = time.monotonic() - t0
    return dict(
        passes=k,
        legs=LEGS + ([MC] if mc_passes else []),
        nocs=NOCS,
        grid=f"{gx}x{gy}",
        cores=n,
        mb=round(R * C * 4 / 2**20, 1),
        wall_s=round(el, 1),
        banks=nb,
        mc_passes=mc_passes,
        mc_nocs=sorted(mc_nocs),
        mc_error=mc_err,
        mc_fail=mc_fail,
        common=common,
        chips={str(i): dict(chips.get(i, {}), **s) for i, s in st.items()},
    )
