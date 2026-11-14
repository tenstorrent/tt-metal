# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MOE_REPLICAS=prev (P.56, owner 2026-11-13, EXPERIMENTAL): expert replicas learned from the previous chunk.

Terms
-----
* slot / spare slot: a chip's experts (12 on the 8x4) and the 13th slot (MOE_REPLICAS=runtime's layout) that holds one
  replica (a copy of another chip's expert) per layer.
* call ordinal o: the MoE calls on one MoE block (mesh, chunk), counted from the model's load, on the host (this
  module) and on the device (the planner's global state word 0) alike: every MoE call is followed by exactly one
  planner run. A chunk makes NCALL calls (one a MoE layer, 40 on DeepSeek-V4.1-Flash): a layer called at o is called
  next at o + NCALL.
* plan (after layer L's call at o, on the device, kernels/moe_prev_planner.cpp): the planner core of every chip gets
  every chip's expert counts of that call (the sender cores send each chip's own slots over the 2D fabric: no host
  round trip, no sync), adds each spare slot's tokens to the expert it held, and plans layer L's NEXT call (o + NCALL,
  the next chunk) with integer arithmetic, the same on every chip: repeatedly the busiest chip's biggest remaining
  expert gets a receiver - the least loaded chip whose spare slot is free this plan - and x = (busiest - receiver -
  MOE_PREV_FIXED) / 2 of its tokens (at most what it has left; stop below MOE_PREV_MIN); a receiver that already holds
  the expert (filled earlier, no copy) is kept when it is within MOE_PREV_SLACK tokens of the best one. The slot map
  (router_rows.hpp slot_of's uneven split, the shares as hash ranges: tt/moe_block_main_adapter.exact_map_words) goes
  straight into layer L's router bias.
* fill (P.53's piggyback, on REC_FX's device pages): a new copy (home chip h's slot -> chip k's spare slot of layer L)
  is cut into pieces (expert_slot_copy's chunk, 10880 B on the 8x4; ~1830 a copy) and given to the calls o + 1 ..
  o + NCALL - 2 in order, at most MOE_FILL_PIECES pieces a source port a call and 30 jobs a port page (the budgets of
  every chip, simulated identically on every chip, live in the global state): chip h's port facing the first hop
  toward k, link 0 (the route the next call's start barrier takes to k), carries them in that call's
  routed_experts_combine (combine_port_writer.cpp REC_FX: a piece only in a round that sent no row, while the channel
  holds fewer than MOE_FILL_CAP packets; the call's rest after the main loop). The jobs are written by the planner into
  a ring of W page sets (one a call ordinal mod W): MOE_PREV_HOST=legacy (P.56) W job tensors, the host passes each
  call ring[o % W]; device (P.60) one job tensor of W page sets, every port reads G[0] mod W itself, so the call's
  fill args never change (no per-call host write). A job names the copied layer's weight buffers (words 6 / 7), so a
  call carries copies of any number of layers. A copy that does not fit in its
  window is not made (that receiver is dropped for the plan, counted). Call o + NCALL - 1 is the fence: its start
  barrier travels the pieces' routes after them, so every piece has landed before layer L's next call.
* guard (host, no transfer): a layer's plan is used only when its next call comes at least NCALL ordinals after the
  plan (every copy was then carried and fenced); otherwise (a request's first chunk, a skipped layer, a sequence of
  another length) the layer's router bias gets its empty map back (a device copy of the load-time bias) and the call
  runs replicas off (counted: `stats`).
* request start (begin_chunk at start 0 or a replay): the reset program empties every job page and the global state
  (a new epoch: every layer state reads as empty) and every layer runs its first chunk off. Nothing crosses requests.
* MOE_PLANNER=folded (P.62b; with MOE_PREV_HOST=device and MOE_PREV_HOST2=lean): no planner launch a call. The
  combine's link-0 port senders send the chip's counts row after their main loop (set_combine_fill_exact prev_*: the
  same fused write + atomic increment as the sender cores; the chip's first port writes its own row too, tagged with
  the ordinal + 1), and the NEXT MoE call's fused head plans it on its watcher core after its own work (fused_head
  tail: kernels/prev_planner.hpp, the planner's L1 the arena past the head's; the pending layer's buffers from a
  64 B descriptor a layer). The block's last layer a chunk plans at once with a standalone folded planner (one launch a
  chunk, no senders), so nothing is pending at a chunk's end; flush() plans a pending call (a request's reset, a
  switch). Same arithmetic, inputs and order: same plans, maps, pages and states.

Data (uint32, DRAM, one page each unless said): global state [GW] (word 0 the ordinal, 1 the epoch, 2..4 cumulative
new copies / pieces / dropped, stamps [W] at 16, budgets [W][chips][4 directions] at 16 + W: pieces | jobs << 16), a
layer's state [SW] (0 epoch, 1 the plan's ordinal, 2 receivers, 3 kept (no copy), 4 new copies, 5 dropped, 6 / 7 the
busiest chip's tokens before / after the plan (predicted), 8 the mean, 9 planned; hold [chips] at 16 (the expert in
each spare slot, or NONE), ready [chips] (the first ordinal it may be used), receivers [chips] (this plan's expert a
spare slot serves), shares [chips]), the landing rows [2 parities x chips, SPW] (the exchanged counts), the
directions [chips x chips] (combine_fill_port).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

import ttnn

from .settings import settings

NONE = 0xFFFFFFFF
KERNEL = Path(__file__).parent / "kernels" / "moe_prev_planner.cpp"
SHARE_ONE = 1 << 15


def _mb():
    return ttnn.bringup.moe_block_main


_HOST2: list = [None]  # (MOE_PREV_HOST2 override: tests switch it in a loaded model, set_host2)


def set_host2(mode: str | None) -> None:
    """Tests: MOE_PREV_HOST2 (legacy | lean) for every PrevReplicas from now on (None: the setting's)."""
    assert mode in (None, "legacy", "lean"), mode
    _HOST2[0] = mode


def host2() -> str:
    return _HOST2[0] or settings.get("MOE_PREV_HOST2")


_PLANNER: list = [None]  # (MOE_PLANNER override: tests switch it in a loaded model, set_planner)


def set_planner(mode: str | None) -> None:
    """Tests: MOE_PLANNER (launch | folded) for every PrevReplicas from now on (None: the setting's)."""
    assert mode in (None, "launch", "folded"), mode
    _PLANNER[0] = mode


def planner() -> str:
    return _PLANNER[0] or settings.get("MOE_PLANNER")


def _u32(host: np.ndarray, mesh, shard: bool = False) -> ttnn.Tensor:
    t = torch.from_numpy(host.astype(np.uint32).view(np.int32).copy())
    return ttnn.from_torch(
        t,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if shard else ttnn.ReplicateTensorToMesh(mesh),
    )


def _bytes_to_words(vals) -> list:
    """Bytes (each < 256) packed 4 a word, little end first (the kernel's arg_byte)."""
    v = [int(x) for x in vals] + [0] * (-len(vals) % 4)
    assert all(0 <= x < 256 for x in v), v
    return [v[i] | v[i + 1] << 8 | v[i + 2] << 16 | v[i + 3] << 24 for i in range(0, len(v), 4)]


def _2bits_to_words(vals) -> list:
    """Directions 0..3 packed 16 a word, low bits first (the kernel's arg_2bit); 0xFF (unused) packs as 0."""
    v = [0 if int(x) == 0xFF else int(x) for x in vals] + [0] * (-len(vals) % 16)
    assert all(0 <= x < 4 for x in v), v
    return [sum(v[i + j] << (2 * j) for j in range(16)) for i in range(0, len(v), 16)]


def _al16(x: int) -> int:
    return -(-int(x) // 16) * 16


class Geometry:
    """The planner's constants (compile-time args) on one mesh."""

    def __init__(self, n: int, epc: int, spc: int, ncall: int, pieces: int, links: int, cpage: int):
        self.n, self.epc, self.spc = int(n), int(epc), int(spc)
        self.E = self.n * self.epc
        assert self.E % 32 == 0
        self.spw = _al16(self.spc)
        self.ncall = int(ncall)
        self.W = self.ncall + 4
        self.pcs = int(pieces)
        self.bud = int(settings.get("MOE_FILL_PIECES"))
        assert 0 < self.bud < 1 << 16, "MOE_REPLICAS=prev: MOE_FILL_PIECES in 1 .. 65535"
        self.cap = max(1, int(settings.get("MOE_FILL_CAP")))
        self.maxj = int(_mb().COMBINE_FX_MAX_JOBS)
        self.fix = int(settings.get("MOE_PREV_FIXED"))
        self.minx = max(1, int(settings.get("MOE_PREV_MIN")))
        self.slack = int(settings.get("MOE_PREV_SLACK"))
        self.maxnew = int(settings.get("MOE_PREV_NEW"))
        self.links = int(links)
        self.cpage = int(cpage)
        self.gw = _al16(16 + self.W + self.W * self.n * 4)
        self.sw = _al16(16 + 4 * self.n)
        self.mappage = self.E // 32

    def ct(self, role: int, ringpg: int = 0, mesh1: int = 0) -> list:
        """ringpg: 0 - W job tensors (MOE_PREV_HOST=legacy); > 0 - one job tensor of W page sets of ringpg pages.
        mesh1 (P.62, MOE_PREV_HOST2=lean): 1 - one program for the whole mesh (positions from a table)."""
        return [
            role,
            self.n,
            self.epc,
            self.spc,
            self.spw,
            self.W,
            self.ncall,
            self.pcs,
            self.bud,
            self.cap,
            self.maxj,
            self.fix,
            self.minx,
            self.slack,
            self.maxnew,
            self.links,
            self.cpage,
            self.gw,
            self.sw,
            self.mappage,
            int(ringpg),
            int(mesh1),
        ]

    def l1_bytes(self) -> int:
        a = lambda x: -(-x // 64) * 64  # noqa: E731
        b = a(self.cpage) + a(self.n * self.spw * 4) + a(self.gw * 4) + a(self.sw * 4) + a(self.n * self.n * 4)
        return (
            64 + b + 4096 + 64 + 1024 + a(self.ncall * 16) + self.ncall * 1024 + 3 * self.E * 4 + 3 * self.n * 4 + 256
        )


# --------------------------------------------------------------------------------------------------------- host mirror
def plan_host(geo: Geometry, rows: np.ndarray, G: np.ndarray, st: np.ndarray, dirs: np.ndarray, me: int | None = None):
    """The planner's arithmetic on the host, word for word (tests compare the device's map, states and pages with it).
    rows [chips, spc]: every chip's own slot counts of the call; G, st: the global and layer states before (copied);
    dirs [chips, chips]. -> (map words [1024] row-major, G after, st after, jobs {(ring slot, chip, dir): [8 words]}
    (every chip's when me is None))."""
    n, epc, E, W = geo.n, geo.epc, geo.E, geo.W
    G = G.astype(np.uint64).copy()
    st = st.astype(np.uint64).copy()
    g = int(G[0])
    epoch = int(G[1])
    SH, SR, SV, SX = 16, 16 + n, 16 + 2 * n, 16 + 3 * n
    GS, GB = 16, 16 + W
    if int(st[0]) != epoch:
        st[0] = epoch
        st[SH : SH + n] = NONE
        st[SR : SR + n] = 0
    st[SV : SV + n] = NONE
    st[SX : SX + n] = 0
    used = [False] * n
    fail = [False] * n
    load = [int(rows[b // epc, b % epc]) for b in range(E)]
    nr = [0] * E
    for k in range(n):
        if int(st[SH + k]) != NONE:
            load[int(st[SH + k])] += int(rows[k, epc])
    T = [sum(load[c * epc : (c + 1) * epc]) for c in range(n)]
    tsum, tmax0 = sum(T), max(T)
    share = list(load)
    nrec = nreuse = nnew = ndrop = pieces = 0
    jobs: dict = {}

    def alloc(h, k, tab):
        nonlocal pieces
        p = int(dirs[h, k])
        room = 0
        for o in range(g + 1, g + geo.ncall - 1):
            s = o % W
            v = int(G[GB + (s * n + h) * 4 + p]) if int(G[GS + s]) == o else 0
            if (v >> 16) < geo.maxj and (v & 0xFFFF) < geo.bud:
                room += geo.bud - (v & 0xFFFF)
        if room < geo.pcs:
            return 0
        rem, p0, last = geo.pcs, 0, 0
        for o in range(g + 1, g + geo.ncall - 1):
            if not rem:
                break
            s = o % W
            if int(G[GS + s]) != o:
                G[GS + s] = o
                G[GB + s * n * 4 : GB + (s + 1) * n * 4] = 0
            v = int(G[GB + (s * n + h) * 4 + p])
            u, j = v & 0xFFFF, v >> 16
            if j >= geo.maxj or u >= geo.bud:
                continue
            take = min(geo.bud - u, rem)
            G[GB + (s * n + h) * 4 + p] = (u + take) | (j + 1) << 16
            if me is None or h == me:
                jobs[(s, h, p, j)] = [tab | k << 8, p0, p0 + take, 0, 0, 0]
            p0 += take
            rem -= take
            last = o
        pieces += geo.pcs
        return last

    for _ in range(2 * n):
        h = max(range(n), key=lambda c: (T[c], -c))
        b = max(range(h * epc, (h + 1) * epc), key=lambda x: (share[x], -x))
        if share[b] == 0:
            break
        kr = kn = None
        for k in range(n):
            if k == h or used[k]:
                continue
            if int(st[SH + k]) == b and int(st[SR + k]) <= g + geo.ncall and (kr is None or T[k] < T[kr]):
                kr = k
            if not fail[k] and int(st[SR + k]) <= g and nnew < geo.maxnew and (kn is None or T[k] < T[kn]):
                kn = k
        fresh = False
        if kr is not None and (kn is None or T[kr] <= T[kn] + geo.slack):
            k = kr
        elif kn is not None:
            k = kn
            fresh = int(st[SH + kn]) != b
        else:
            break
        gap = T[h] - T[k] - geo.fix
        x = gap >> 1 if gap > 0 else 0
        x = min(x, share[b])
        if x < geo.minx:
            break
        if fresh:
            last = alloc(h, k, b % epc)
            if last == 0:
                fail[k] = True
                ndrop += 1
                continue
            st[SH + k] = b
            st[SR + k] = last + 2
            nnew += 1
        else:
            nreuse += 1
        used[k] = True
        st[SV + k] = b
        st[SX + k] = x
        nr[b] += 1
        nrec += 1
        share[b] -= x
        T[h] -= x
        T[k] += x + geo.fix
    words = np.zeros(1024, dtype=np.uint64)
    for b in range(E):
        words[b] = (b // epc) * geo.spc + b % epc
    first = 0
    for b in range(E):
        if nr[b] == 0:
            continue
        words[E + b] = nr[b] | first << 6
        tot, acc, i = load[b], share[b], 0
        for k in range(n):
            if int(st[SV + k]) != b:
                continue
            lb = min(max(1, (acc * SHARE_ONE + tot // 2) // tot), SHARE_ONE - 1)
            words[2 * E + first + i] = (k * geo.spc + epc) | lb << 9
            acc += int(st[SX + k])
            i += 1
        first += nr[b]
    st[1] = g
    st[2:10] = [nrec, nreuse, nnew, ndrop, tmax0, max(T), tsum // n, 1]
    st[10:14] = 0  # (the device's diagnostic cycle stamps: not mirrored, masked in comparisons)
    G[0] = g + 1
    G[2] += nnew
    G[3] += pieces
    G[4] += ndrop
    return words.astype(np.uint32), G.astype(np.uint32), st.astype(np.uint32), jobs


def map_words_of_tile(tile: np.ndarray) -> np.ndarray:
    """The router bias's map tile (fp32 [32, 32] as read back, tile layout undone by ttnn) -> integer words [1024]."""
    return np.rint(np.asarray(tile, dtype=np.float64).reshape(-1)).astype(np.uint64).astype(np.uint32)


# ------------------------------------------------------------------------------------------------------------ device
class PrevReplicas:
    """One MoE block's prev-replica machinery (its resources): the ring of job tensors, the global state, the landing
    rows, the directions, the planner / reset programs (prepared, OV-23) and the host ordinal with the guard."""

    def __init__(self, layer, res, ncall: int):
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan

        from .moe_block_main_adapter import _exact_fx
        from .window_send import _torus

        mb = _mb()
        self.mesh, self.res = layer.mesh, res
        mesh = self.mesh
        n = mesh.get_num_devices()
        self.R, self.C = tuple(mesh.shape)
        fx = _exact_fx(layer, res)
        self.tables, self.limit = fx["tables"], fx["limit"]
        cpage = -(-4 * n * layer.spc // 64) * 64  # (head.counts: one row-major page of chips x spc uint32)
        self.geo = Geometry(n, layer.epc, layer.spc, ncall, self.tables.pieces, int(fx["links"]), cpage)
        geo = self.geo
        dirs = np.zeros((n, n), dtype=np.uint32)
        for a in range(n):
            for b in range(n):
                if a != b:
                    dirs[a, b] = int(mb.combine_fill_port(res, a, b))
        assert int(dirs.max()) < 4
        self.dirs = dirs
        # (the directions, then the empty map's tile: fp32 bits of each candidate's own slot, in tile order)
        base = np.zeros(1024, dtype=np.uint32)
        own = np.arange(geo.E)
        base[: geo.E] = ((own // geo.epc) * geo.spc + own % geo.epc).astype(np.float32).view(np.uint32)
        r, c = np.arange(1024) // 32, np.arange(1024) % 32
        tiled = np.zeros(1024, dtype=np.uint32)
        tiled[((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)] = base
        self.consts = _u32(np.concatenate([dirs.reshape(-1), tiled]).reshape(1, 1, 1, n * n + 1024), mesh)
        G = np.zeros(geo.gw, dtype=np.uint32)
        G[1] = 1  # (epoch 1: the layers' zeroed states, epoch 0, read as empty)
        G[16 : 16 + geo.W] = NONE
        self.G = _u32(G.reshape(1, 1, 1, geo.gw), mesh)
        self.landing = _u32(np.zeros((1, 1, 2 * n, geo.spw), dtype=np.uint32), mesh)
        self.pages, self.pw = int(mb.COMBINE_FX_PAGES), int(mb.COMBINE_FX_PAGE_WORDS)
        # (the job page ring: built per host mode at first use - legacy W tensors, device one tensor of W page sets)
        self._rings: dict = {}
        self.mode = settings.get("MOE_PREV_HOST")
        self._host2 = settings.get("MOE_PREV_HOST2")  # (read once: lean() runs every call)
        cc = ttnn.CoreCoord(0, 0)
        self.sem = ttnn.create_global_semaphore(
            mesh, ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)]), 0, ttnn.BufferType.L1_SMALL
        )
        self.sem_addr = int(ttnn.get_global_semaphore_address(self.sem))
        v = mesh.worker_core_from_logical_core(cc)
        self.pxy = (int(v.x), int(v.y))
        self.torus = _torus(mesh)
        self.chips, _, self.wraps = mesh_plan(mesh, 1, self.torus)
        self.src = KERNEL.read_text()
        self.ord = 0  # (host ordinal = the device's G[0])
        self._plan_pp: dict = {}  # (host mode -> the prepared planner / reset)
        self._reset_pp: dict = {}
        self._fx_w = None  # (device mode: the fill args' gate_up / down - fixed; the planner's jobs name their own)
        self._fx_on = False  # (P.62 lean: the combine's fill args set (device mode); fill_off() clears it)
        self._fx_fold = False  # (P.62b: the set fill args carry the prev counts (folded))
        # P.62b (MOE_PLANNER=folded): the head's watcher core (the planner tail runs there; the arrival word too), the
        # tail, the call whose plan is pending (the next folded head plans it), the block's last layer (it plans at once)
        blk = layer.block()
        tx, ty, _ = mb.fused_head_tail_core(blk.resources, blk.dtypes, blk.accumulators, blk.l1_free)
        self.tail_core = (int(tx), int(ty))
        tc = ttnn.CoreCoord(*self.tail_core)
        self.sem_t = ttnn.create_global_semaphore(
            mesh, ttnn.CoreRangeSet([ttnn.CoreRange(tc, tc)]), 0, ttnn.BufferType.L1_SMALL
        )
        self.sem_t_addr = int(ttnn.get_global_semaphore_address(self.sem_t))
        self.tail = mb.HeadTail(ct=geo.ct(0, self.pages), l1=64 + geo.l1_bytes())
        self._pending = None
        self.last_layer = None  # (set by the adapter: the largest main layer index on this block)
        self._fold_pp = None
        self.stats = {"calls": 0, "ready": 0, "off": 0, "off_copies": 0, "plans": 0}
        logger.info(
            f"MOE_REPLICAS=prev: {n} chips, {ncall} MoE calls a chunk, ring {geo.W}, {geo.pcs} pieces a copy, "
            f"{geo.bud} a port a call, planner L1 {geo.l1_bytes()} B, host {self.mode}"
        )

    # ------------------------------------------------------------------ the job page ring
    def _ringpg(self) -> int:
        return self.pages if self.mode == "device" else 0

    def rings(self) -> list:
        """The ring's job tensors in the current host mode: legacy W tensors [chips, 1, pages, page words] (sharded by
        chip), device one tensor [chips, 1, W x pages, page words]."""
        r = self._rings.get(self.mode)
        if r is None:
            assert self.mode in ("legacy", "device"), self.mode
            n, W = self.geo.n, self.geo.W
            if self.mode == "device":
                shapes = [(n, 1, W * self.pages, self.pw)]
            else:
                shapes = [(n, 1, self.pages, self.pw)] * W
            r = self._rings[self.mode] = [_u32(np.zeros(sh, dtype=np.uint32), self.mesh, shard=True) for sh in shapes]
        return r

    def _ring_args(self) -> list:
        return [int(t.buffer_address()) for t in self.rings()]

    # ------------------------------------------------------------------ programs
    def _node(self, ch):
        return self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch))

    def _planner_desc(self, io: list, cpage: int):
        from ttnn.bringup.combine_rf.sparse_combine_l1 import route

        geo = self.geo
        counts, landing, G, st, rb, gu, d, consts = io
        md = ttnn.MeshProgramDescriptor()
        rings = self._ring_args()
        l1 = geo.l1_bytes()
        cores = want = 0
        for r in range(self.R):
            for c in range(self.C):
                me = r * self.C + c
                groups: dict = {}
                for k in range(geo.n):
                    if k == me:
                        continue
                    kr, kc = divmod(k, self.C)
                    hops = route((r, c), (kr, kc), self.R, self.C, self.wraps[0], self.wraps[1], 0)
                    assert hops and hops[0][0] == (r, c), ((r, c), (kr, kc), hops)
                    groups.setdefault(hops[0][1], []).append((kr, kc))
                prog = ttnn.ProgramDescriptor()
                kernels, cbs = [], []
                dm = ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                )
                cc = ttnn.CoreCoord(0, 0)
                crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
                cbs.append(
                    ttnn.CBDescriptor(
                        total_size=l1, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, l1)]
                    )
                )
                rt = ttnn.RuntimeArgs()
                rt[0][0] = [
                    int(counts.buffer_address()),
                    int(landing.buffer_address()),
                    int(G.buffer_address()),
                    int(st.buffer_address()),
                    int(rb.buffer_address()),
                    int(gu.buffer_address()),
                    int(d.buffer_address()),
                    int(consts.buffer_address()),
                    me,
                    self.sem_addr,
                ] + rings
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=self.src,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=crs,
                        compile_time_args=geo.ct(0, self._ringpg()),
                        runtime_args=rt,
                        config=dm,
                    )
                )
                cores += 1
                want += 8
                for gi, (port_dir, dsts) in enumerate(sorted(groups.items())):
                    sc = ttnn.CoreCoord(1 + gi, 0)
                    scr = ttnn.CoreRangeSet([ttnn.CoreRange(sc, sc)])
                    sl1 = -(-geo.cpage // 64) * 64 + 64 + geo.spw * 4 + 128
                    cbs.append(
                        ttnn.CBDescriptor(
                            total_size=sl1,
                            core_ranges=scr,
                            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, sl1)],
                        )
                    )
                    args = [
                        int(counts.buffer_address()),
                        int(landing.buffer_address()),
                        int(G.buffer_address()),
                        me,
                        self.sem_addr,
                        self.pxy[0],
                        self.pxy[1],
                        len(dsts),
                    ]
                    for ch in dsts:
                        dn = self._node(ch)
                        args += [int(dn.mesh_id), int(dn.chip_id)]
                    port = self.chips[(r, c)][port_dir]
                    args += list(
                        ttnn.setup_fabric_connection(
                            self._node((r, c)), self._node(port["peer"]), port["links"][0], prog, sc
                        )
                    )
                    srt = ttnn.RuntimeArgs()
                    srt[sc.x][sc.y] = args
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=self.src,
                            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                            core_ranges=scr,
                            compile_time_args=geo.ct(1, self._ringpg()),
                            runtime_args=srt,
                            config=dm,
                        )
                    )
                    cores += 1
                    want += 3
                prog.cbs = cbs
                prog.kernels = kernels
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
        return md, cores, want

    def lean(self) -> bool:
        """MOE_PREV_HOST2=lean (P.62) on the device host path: one planner program for the mesh, the fill args once."""
        return self.mode == "device" and (_HOST2[0] or self._host2) == "lean"

    def folded(self) -> bool:
        """MOE_PLANNER=folded (P.62b) on the lean path: the plan in the next MoE call's fused head, the counts from the
        combine's ports (no planner launch); needs two MoE layers a chunk at least (a layer's own head never plans it).
        """
        return self.lean() and planner() == "folded" and self.geo.ncall >= 2

    def _mesh_tables(self):
        """P.62: the chip-independent tables of the one-program planner (memoized): (mesh id, me by fabric chip id,
        chip id by position, first hop direction by (me, k), eth channel by (me, direction), a chip of each direction
        and its port)."""
        if getattr(self, "_mt", None) is not None:
            return self._mt
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import DIRS
        from ttnn.bringup.combine_rf.sparse_combine_l1 import route

        n, R, C = self.geo.n, self.R, self.C
        nodes = [self._node((k // C, k % C)) for k in range(n)]
        mesh_ids = {int(x.mesh_id) for x in nodes}
        assert len(mesh_ids) == 1, mesh_ids
        chip_of = [int(x.chip_id) for x in nodes]
        ndev = max(chip_of) + 1
        assert ndev < 255 and len(set(chip_of)) == n, chip_of
        me_of = [0xFF] * ndev
        for k, c in enumerate(chip_of):
            me_of[c] = k
        hop = [0xFF] * (n * n)
        for me in range(n):
            r, c = divmod(me, C)
            for k in range(n):
                if k == me:
                    continue
                hops = route((r, c), divmod(k, C), R, C, self.wraps[0], self.wraps[1], 0)
                assert hops and hops[0][0] == (r, c), ((r, c), divmod(k, C), hops)
                hop[me * n + k] = DIRS.index(hops[0][1])
        chan = [0xFF] * (n * 4)
        rep: dict = {}
        for me in range(n):
            ports = self.chips[divmod(me, C)]
            for di, dn in enumerate(DIRS):
                port = ports.get(dn)
                if port is None:
                    continue
                scratch = ttnn.ProgramDescriptor()
                a = list(
                    ttnn.setup_fabric_connection(
                        nodes[me], self._node(port["peer"]), port["links"][0], scratch, ttnn.CoreCoord(1 + di, 0)
                    )
                )
                # (device-init fabric: [eth channel, two semaphore ids]; only the channel is the chip's)
                assert len(a) == 3 and 0 <= int(a[0]) < 255, a
                chan[me * 4 + di] = int(a[0])
                rep.setdefault(di, (me, port))
        used = sorted({h for h in hop if h != 0xFF})
        for di in used:
            assert di in rep, (di, rep)
            assert all(chan[me * 4 + di] != 0xFF for me in range(n) if di in hop[me * n : (me + 1) * n]), di
        self._mt = (mesh_ids.pop(), ndev, me_of, chip_of, hop, chan, rep, used, nodes)
        return self._mt

    def _planner_desc_mesh(self, io: list):
        """P.62 (MOE_PREV_HOST2=lean): the planner as ONE program for the whole mesh (DS41-225): the planner core and
        one sender core a first-hop direction (cores 1 + d), every chip the same runtime args; a chip finds its
        position, targets and fabric channel in tables keyed by its fabric chip id (kernel MESH1)."""
        geo = self.geo
        counts, landing, G, st, rb, gu, d, consts = io
        mesh_id, ndev, me_of, chip_of, hop, chan, rep, used, nodes = self._mesh_tables()
        prog = ttnn.ProgramDescriptor()
        dm = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        )
        l1 = geo.l1_bytes()
        cc = ttnn.CoreCoord(0, 0)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
        cbs = [
            ttnn.CBDescriptor(
                total_size=l1, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, l1)]
            )
        ]
        rt = ttnn.RuntimeArgs()
        rt[0][0] = (
            [int(t.buffer_address()) for t in (counts, landing, G, st, rb, gu, d, consts)]
            + [0, self.sem_addr]
            + self._ring_args()
            + _bytes_to_words(me_of)
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=self.src,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=geo.ct(0, self._ringpg(), 1),
                runtime_args=rt,
                config=dm,
            )
        ]
        tables = _bytes_to_words(me_of) + _bytes_to_words(chip_of) + _2bits_to_words(hop) + _bytes_to_words(chan)
        for di in used:
            sc = ttnn.CoreCoord(1 + di, 0)
            scr = ttnn.CoreRangeSet([ttnn.CoreRange(sc, sc)])
            sl1 = -(-geo.cpage // 64) * 64 + 64 + geo.spw * 4 + 128
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=sl1, core_ranges=scr, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, sl1)]
                )
            )
            me, port = rep[di]
            fab = list(ttnn.setup_fabric_connection(nodes[me], self._node(port["peer"]), port["links"][0], prog, sc))
            assert len(fab) == 3, fab
            args = [
                int(counts.buffer_address()),
                int(landing.buffer_address()),
                int(G.buffer_address()),
                self.sem_addr,
                self.pxy[0],
                self.pxy[1],
                di,
                mesh_id,
                ndev,
            ]
            args += fab + tables
            assert len(args) <= 200, len(args)  # (runtime args: at most 341 a core)
            srt = ttnn.RuntimeArgs()
            srt[sc.x][sc.y] = args
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=self.src,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=scr,
                    compile_time_args=geo.ct(1, self._ringpg(), 1),
                    runtime_args=srt,
                    config=dm,
                )
            )
        prog.cbs = cbs
        prog.kernels = kernels
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(self.R - 1, self.C - 1))] = prog
        return md, 1 + len(used), 8 + 3 * len(used)

    def _planner_desc_fold(self, io: list):
        """P.62b: the standalone folded planner (the block's last layer a chunk, flush()): the planner core alone on the
        head's watcher core (the arrival word is there), one mesh program, the own row from the landing rows (counts arg
        0); the combine's ports sent the counts."""
        geo = self.geo
        landing, G, st, rb, gu, d, consts = io
        mesh_id, ndev, me_of, chip_of, hop, chan, rep, used, nodes = self._mesh_tables()
        prog = ttnn.ProgramDescriptor()
        l1 = geo.l1_bytes()
        cc = ttnn.CoreCoord(*self.tail_core)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=l1, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, l1)]
            )
        ]
        rt = ttnn.RuntimeArgs()
        rt[cc.x][cc.y] = (
            [0]
            + [int(t.buffer_address()) for t in (landing, G, st, rb, gu, d, consts)]
            + [0, self.sem_t_addr]
            + self._ring_args()
            + _bytes_to_words(me_of)
        )
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=self.src,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=geo.ct(0, self._ringpg(), 1),
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        ]
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(self.R - 1, self.C - 1))] = prog
        return md

    def _desc(self, layer) -> ttnn.Tensor:
        """P.62b: the layer's descriptor for the head's planner tail (made at its first folded call, one 64 B page):
        its state, router bias, gate/up, down, then the landing rows, global state, directions + empty map, the arrival
        word, the job ring; word 15 the mark 'P62B'."""
        d = getattr(layer, "_prev_desc", None)
        if d is None:
            w = np.zeros(16, dtype=np.uint32)
            w[:9] = [
                int(t.buffer_address())
                for t in (layer._prev_state, layer.head_w.router_bias, layer.w_gu, layer.w_d, self.landing, self.G)
            ] + [int(self.consts.buffer_address()), self.sem_t_addr, int(self.rings()[0].buffer_address())]
            w[15] = 0x50364232
            d = layer._prev_desc = _u32(w.reshape(1, 1, 1, 16), self.mesh)
        return d

    def head_tail(self, layer) -> dict:
        """P.62b: the fused head's kwargs of a folded call - the planner tail and the pending call's layer descriptor
        (none pending: this layer's; the tail then finds no tag and plans nothing)."""
        return {"tail": self.tail, "tail_desc": self._desc(self._pending if self._pending is not None else layer)}

    def on_head(self, layer, head) -> None:
        """P.62b: after a folded call's head - the combine's fill args with the prev counts (once: the head's counts
        tensor is the block's persistent one)."""
        if self._fx_on and self._fx_fold:
            return
        if self._fx_w is None:
            self._fx_w = (layer.w_gu, layer.w_d)
        _mb().set_combine_fill_exact(
            self.res,
            True,
            self._fx_w[0],
            self._fx_w[1],
            self.tables.runs,
            self.rings()[0],
            self.limit,
            ring=self.geo.W,
            ordinal=self.G,
            wait=False,
            prev_counts=head.counts,
            prev_landing=self.landing,
            prev_sem=self.sem_t_addr,
            prev_spc=self.geo.spc,
            prev_core=self.tail_core,
        )
        self._fx_on, self._fx_fold = True, True

    def _plan_now(self, layer) -> None:
        """P.62b: the standalone folded planner for layer's last call (its counts already sent by the combine)."""
        io = [
            self.landing,
            self.G,
            layer._prev_state,
            layer.head_w.router_bias,
            layer.w_gu,
            layer.w_d,
            self.consts,
        ]
        if self._fold_pp is None:
            self._fold_pp = ttnn.bringup.prepare_program(self._planner_desc_fold(io), io)
            assert self._fold_pp.patch_count == 7, f"folded planner: {self._fold_pp.patch_count} patched slots, want 7"
            logger.info(f"MOE_REPLICAS=prev: folded planner prepared (core {self.tail_core}, one mesh program)")
        ttnn.bringup.run_prepared(self._fold_pp, io)

    def flush(self) -> None:
        """P.62b: a pending folded plan planned now (a request's reset, a switch of path or side, a call that is not
        folded): afterwards the device state is complete, as after the launch path."""
        if self._pending is not None:
            layer, self._pending = self._pending, None
            self._plan_now(layer)

    def _reset_desc(self):
        geo = self.geo
        md = ttnn.MeshProgramDescriptor()
        rings = self._ring_args()
        l1 = -(-geo.gw * 4 // 64) * 64 + 256
        one = self.lean()  # (P.62: ROLE 2 reads arg 0 and the rings only - arg 1, the position, is not read)
        for r in range(1 if one else self.R):
            for c in range(1 if one else self.C):
                prog = ttnn.ProgramDescriptor()
                cc = ttnn.CoreCoord(0, 0)
                crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
                rt = ttnn.RuntimeArgs()
                rt[0][0] = [int(self.G.buffer_address()), 0 if one else r * self.C + c] + rings
                prog.cbs = [
                    ttnn.CBDescriptor(
                        total_size=l1, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, l1)]
                    )
                ]
                prog.kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=self.src,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=crs,
                        compile_time_args=geo.ct(2, self._ringpg()),
                        runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                        ),
                    )
                ]
                rng = (
                    ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(self.R - 1, self.C - 1))
                    if one
                    else ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))
                )
                md[rng] = prog
        return md

    # ------------------------------------------------------------------ the model's hooks
    def reset(self, layers) -> None:
        """A request starts (begin_chunk at start 0 / a replay): every job page and the global state emptied on the
        device (a new epoch), every layer's plan forgotten (its first call runs off)."""
        self.flush()
        key = (self.mode, self.lean())
        pp = self._reset_pp.get(key)
        if pp is None:
            pp = self._reset_pp[key] = ttnn.bringup.prepare_program(self._reset_desc(), [self.G])
        ttnn.bringup.run_prepared(pp, [self.G])
        for m in layers:
            m._prev_ord = None

    def before(self, layer) -> None:
        """Before layer's MoE call: the guard (its plan is used only when every copy was carried and fenced), then the
        combine's exact fill args with this call's job page ring slot."""
        o = self.ord
        ok = layer._prev_ord is not None and o >= layer._prev_ord + self.geo.ncall
        self.stats["calls"] += 1
        if ok:
            self.stats["ready"] += 1
        else:
            self.stats["off"] += 1
            if layer._prev_dirty:
                ttnn.copy(layer._prev_off_rb, layer.head_w.router_bias)
                layer._prev_dirty = False
                self.stats["off_copies"] += 1
        layer._prev_used = ok
        if getattr(layer, "_prev_fold", False) and self.folded():  # (P.62b: on_head sets the fill args)
            return
        self.flush()  # (P.62b: a call that is not folded - a pending plan first)
        if self._fx_on and not self._fx_fold:  # (P.62 lean: the device mode's fill args are already set)
            return
        if self.mode == "device":  # (P.60: the same args every call - the ports read their ring slot from G[0])
            if self._fx_w is None:
                self._fx_w = (layer.w_gu, layer.w_d)  # (unused by the ports: every job names its layer's buffers)
            _mb().set_combine_fill_exact(
                self.res,
                True,
                self._fx_w[0],
                self._fx_w[1],
                self.tables.runs,
                self.rings()[0],
                self.limit,
                ring=self.geo.W,
                ordinal=self.G,
                wait=False,
            )
            self._fx_on, self._fx_fold = self.lean(), False
            return
        _mb().set_combine_fill_exact(
            self.res, True, layer.w_gu, layer.w_d, self.tables.runs, self.rings()[o % self.geo.W], self.limit
        )

    def fill_off(self) -> None:
        """The combine's exact fill switched off (prev off in a loaded model, tests): the next before() sets it again."""
        self.flush()
        _mb().set_combine_fill_exact(self.res, False)
        self._fx_on = self._fx_fold = False

    def set_mode(self, mode: str) -> None:
        """Tests: the host path (MOE_PREV_HOST) switched in a loaded model; the fill args are set again."""
        self.flush()
        self.mode = mode
        self._fx_on = self._fx_fold = False

    def after(self, layer, counts: ttnn.Tensor) -> None:
        """After layer's MoE call (counts: its fused head's per-chip counts): the planner (device) for its next call."""
        if getattr(layer, "_prev_fold", False) and self.folded():  # (P.62b: planned by the next head, or now)
            self.after_host(layer)
            return
        lean = self.lean()
        pp = self._plan_pp.get((self.mode, lean))
        if lean and pp is not None:  # (P.62 lean: the steady call - dispatch and the ordinals, nothing else)
            ttnn.bringup.run_prepared(
                pp,
                [
                    counts,
                    self.landing,
                    self.G,
                    layer._prev_state,
                    layer.head_w.router_bias,
                    layer.w_gu,
                    layer.w_d,
                    self.consts,
                ],
            )
            layer._prev_ord = self.ord
            layer._prev_dirty = True
            self.ord += 1
            self.stats["plans"] += 1
            return
        io = [
            counts,
            self.landing,
            self.G,
            layer._prev_state,
            layer.head_w.router_bias,
            layer.w_gu,
            layer.w_d,
            self.consts,
        ]
        if pp is None:
            assert int(counts.shape[-1]) == self.geo.n * self.geo.spc, (counts.shape, self.geo.n, self.geo.spc)
            assert counts.dtype == ttnn.uint32 and counts.layout == ttnn.ROW_MAJOR_LAYOUT, (counts.dtype, counts.layout)
            md, cores, want = self._planner_desc_mesh(io) if lean else self._planner_desc(io, self.geo.cpage)
            if lean:
                cores *= self.geo.n  # (cores over the mesh, as the per-chip descriptor counts them)
            pp = self._plan_pp[(self.mode, lean)] = ttnn.bringup.prepare_program(md, io)
            self._plan_cores = cores
            # (8 io addresses on each planner core, 3 on each sender core: anything else matched a frozen value)
            assert pp.patch_count == want, f"planner: {pp.patch_count} patched slots, want {want}"
            logger.info(
                f"MOE_REPLICAS=prev: planner prepared ({self.mode}, {'one mesh program' if lean else 'a program a chip'}),"
                f" {cores} cores ({cores / self.geo.n:.1f} a chip)"
            )
        ttnn.bringup.run_prepared(pp, io)
        self.after_host(layer)

    def after_host(self, layer) -> None:
        """after()'s host bookkeeping alone: the call ordinal (= the device's G[0], which the planner advances) and the
        layer's plan ordinal for the guard. MOE_TRACE=moe (P.61): a replayed MoE half ran the planner from the trace."""
        layer._prev_ord = self.ord
        layer._prev_dirty = True
        self.ord += 1
        self.stats["plans"] += 1
        if getattr(layer, "_prev_fold", False) and self.folded():  # (P.62b: this call's head planned the pending one)
            self._pending = layer
            if layer.layer == self.last_layer:  # (the chunk's last MoE call: planned now, nothing left pending)
                self.flush()

    # ------------------------------------------------------------------ read back (tests, reports)
    def read_global(self) -> np.ndarray:
        return _read(self.G)

    def read_ring(self, s: int) -> np.ndarray:
        """Ring slot s's job pages of every chip: [chips, pages, page words]."""
        r = self.rings()
        if self.mode == "device":
            return read_all(r[0]).reshape(self.geo.n, self.geo.W, self.pages, self.pw)[:, s]
        return read_all(r[s]).reshape(self.geo.n, -1, self.pw)


def _words(t) -> np.ndarray:
    return (ttnn.to_torch(t).reshape(-1).to(torch.int64).numpy() & 0xFFFFFFFF).astype(np.uint32)


def _read(t: ttnn.Tensor, chip: int = 0) -> np.ndarray:
    return _words(ttnn.get_device_tensors(t)[chip])


def read_all(t: ttnn.Tensor) -> np.ndarray:
    """Every chip's copy of a uint32 tensor: [chips, words]."""
    return np.stack([_words(d) for d in ttnn.get_device_tensors(t)])
