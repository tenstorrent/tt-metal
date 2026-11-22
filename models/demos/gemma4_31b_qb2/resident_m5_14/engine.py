# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""resident_m5_14 (M7.1; LEAD_DECISIONS_M7 M118, bench/resident/m7/m71/M5_14_K2A_STRADDLE_FIX.md): the frozen
resident_m5_13 plus F1, the K2a straddle fix: a verify block that straddles an absolute chunk boundary (256 positions
sliding, 2,048 full) gets its tiles reduced in their own chunks (kernels/k2a_common.hpp Builder; k2a_sched.py replays
it), so K2a rows no longer depend on the block's alignment (W1PROBE v3: onset 1280). One boundary stays: a block
straddling 16,384 would switch the full layers' flat merge to the group-node tree for the whole block, so the serving
class caps such a block with verify_cap() (M131: its rows from 16,384 on are sent inert, and the schedule ends a block at
its last live row). NVs 9 -> 8; the K2a unit-count guard counts a straddling block's extra unit. The M1L path is
unchanged (F2 open). Below, m5_13's history.

resident_m5_13 (M7; LEAD_DECISIONS_M7 M5): the frozen resident_m5_11 plus (1) every global semaphore in L1_SMALL
(model.py SEM_L1_SMALL, GRES_SEM_L1_SMALL=0 restores m5_11's L1; bench/resident/m7/L1_FLOOR_ANSWER.md) and (2) the
runtime tap mask set_taps_enabled() (status word ST_TAPS_OFF; M7_REBASE_v2_ADDENDUM_2 F1: PLAIN = taps built, masked).
Neither changes a token, logit or KV row. Below, the history as in m5_11.

The resident engine's serving API (bench/resident/M1_INTERFACE.md, INTERFACE_VERSION 1). resident_m3: the frozen
resident_m3 (the M1L engine, capacity 262,144, narrow / wide full layers, with resident_m2's API deltas:
EngineConfig.l1_top per R, cb_tops(), drop_rows(), close() releasing the weight owner; bench/resident/M3_MERGE.md)
plus M2b stage 1: F1 (block-copy K/V splice) and F9 (per-R seam split); bench/resident/M3_1_DELTA.md. resident_m3_2:
resident_m3_1 plus F2 (the slices' h multicast from two RISCs) and F5v (the merges' SFPU ops on the faces they read);
bench/resident/M3_2_DELTA.md. resident_m3_3: resident_m3_2 plus F10c8 (the slices' h pushed and multicast in 8-view
chunks, kChunk 8) and the act all-gather's rotated peer order; bench/resident/M3_3_DELTA.md. resident_m5_2 (M5 phase 1,
bench/resident/m5/M5_FOUNDATION.md): resident_m3_3 plus per-user blocks in the hub (validation, per-user status
nibbles, feed per live row; EngineConfig.block) and per-unit page-table reads on the attention cores. resident_m5_2
(M5 K0b, bench/resident/m5_1/M5_1_DELTA.md): resident_m5 plus resident_m4_2's R = 32 items (the x0 seed offset at
R = 32, the o = q attention stub and layer_offset for teacher-forced checks, the writer's timeline marks 6 .. 9),
dflash-builder's m4x0 x0 -> cb_y slice alias at R = 32, unit regions for the attention tables and all 32 feed rows on
the attention writer. resident_m5_2 (M5 K1, bench/resident/m5_2/M5_2_DELTA.md): resident_m5_1 plus the R = 32 taps
and logits row layouts (M4_M5_GATES sites 2 and 3), the slices' debug x dump (dump_rounds) and the host guard: until
the attention serves every user's block (K2a), launch(R) refuses a launch whose rows give a user u >= 1 a valid block
(M1_INTERFACE_M5_ADDENDUM s5); `test_allow_users = True` lifts it for tests.

EngineConfig / KVPools describe what the adapter hands over; the engine owns the per-step device tensors (tokens,
positions, slots, tables, logits, status, taps) that every captured R program reads, validates host inputs before
anything reaches the device, and exposes launch / warmup / readback / lifecycle calls. StubEngine implements the same
API without weights (deterministic fake outputs) so the serving adapter can be built against it; ResidentEngine runs
the resident programs (model.py)."""

from dataclasses import dataclass, field

import torch

import ttnn

INTERFACE_VERSION = 1
VOCAB = 262144
HIDDEN = 5376
ROWS = 32  # every feed / output tensor carries 32 rows (one tile); rows >= R are inert
GROUPS = 6  # table rows per slot: the vLLM kv-cache groups (1 full + 5 sliding for Gemma-4-31B)
TAB_STAGE = 4  # (M5 A5) slots per row-granular write_tables call (more: the whole-table upload)
PAGE = 128
FEEDS = ("argmax", "external", "none")
SLIDING = (4, 128, 256)  # per-chip page geometry [kv heads, rows, head_dim]
FULL = (1, 128, 512)

# status words (M1_INTERFACE.md s5); M5: user u's sticky flags in word ST_USER + u // 8, bits 4 (u % 8) .. + 3
ST_LAUNCHES, ST_FLAGS, ST_ROW, ST_POS, ST_PAGE, ST_ENTRY = range(6)
ST_USER = 8
ST_TAPS_OFF = 6  # (m5_13, M7 addendum 2 F1) the host's tap mask, read by the slices each launch (1 = no tap writes)
F_POSITION, F_SLOT, F_TABLE_COLUMN, F_PAGE = 0x1, 0x2, 0x4, 0x8
F_SCHED = 0x10  # (K2b step 3) a K2a schedule overflow: the launch ran inert (status word 1 only; no row moved)
# (m5_14 F1, LEAD_DECISIONS_M7 M131) the positions a K2a verify block must not straddle: the full layers' switch from
# the flat merge (k <= 8 chunks) to the group-node tree at 8 x 2,048 = 16,384 (k2a.FLAT_POS). The only one: the host
# replay sweeps every 2,048 c boundary to 262,144 at M = 2 / 4 / 8 (bench/resident/m7/m71/m5_14_build, check C9)
K2A_MERGE_BOUNDARIES = (16384,)


def verify_cap(p, M):
    """(m5_14 F1, M131) how many rows a K2a verify block of M rows at position p may use: M, or fewer so that its live
    rows end before a merge boundary (K2A_MERGE_BOUNDARIES). The rows from the cap on are sent inert (position -1):
    the unit cores' schedule then ends the block at its last live row, and the next block starts at the boundary."""
    for b in K2A_MERGE_BOUNDARIES:
        if p < b <= p + M - 1:
            return b - p
    return M


class EngineError(ValueError):
    pass


@dataclass(frozen=True)
class EngineConfig:
    weight_cache: str = "/home/ttuser/experiments/gemma4/resident_cache_raw"
    rows: tuple = (1,)
    max_users: int = 1
    capacity: int = 32768
    tap_layers: tuple = ()
    l1_top: object = 1_330_000  # int (every R) or {R: int} (M2: R = 1 keeps M1's 1_330_000, R = 8 its own)
    logits: bool = True
    embedding: str = "resident"
    block: object = None  # M5: rows per user block, int (every R) or {R: int, or (A2, R = 32) a tuple of them};
    # None = R (the launch is one user)

    def l1_top_for(self, R):
        """the absolute L1 address where R's streamer CBs must end"""
        return int(self.l1_top[R]) if isinstance(self.l1_top, dict) else int(self.l1_top)

    def block_for(self, R):
        """M5: the rows of each user block in R's launches (user u: rows [u M, (u + 1) M)); with several blocks (A2),
        the first: launch(R)'s default"""
        return self.blocks_of(R)[0]

    def blocks_of(self, R):
        """(M5 A2) every block R's programs are built for (one program and trace pair per block)"""
        b = self.block.get(R) if isinstance(self.block, dict) else self.block
        if b is None:
            return (R,)
        return tuple(int(x) for x in b) if isinstance(b, (tuple, list)) else (int(b),)

    def validate(self, max_capacity):
        if not self.rows or any(not isinstance(r, int) or not 1 <= r <= ROWS for r in self.rows):
            raise EngineError(f"rows must be ints in 1..{ROWS}: {self.rows}")
        if not 1 <= self.max_users <= ROWS:
            raise EngineError(f"max_users must be 1..{ROWS}: {self.max_users}")
        if self.capacity % PAGE or not PAGE <= self.capacity <= max_capacity:
            raise EngineError(f"capacity must be a multiple of {PAGE} in {PAGE}..{max_capacity}: {self.capacity}")
        if len(set(self.tap_layers)) != len(self.tap_layers) or any(not 0 <= l < 60 for l in self.tap_layers):
            raise EngineError(f"tap_layers must be distinct layer indices 0..59: {self.tap_layers}")
        if isinstance(self.l1_top, dict) and set(self.l1_top) != set(self.rows):
            raise EngineError(f"l1_top per R needs exactly the rows {self.rows}: {sorted(self.l1_top)}")
        if self.embedding not in ("resident", "ttnn"):
            raise EngineError(f"embedding must be 'resident' or 'ttnn' (M5 E): {self.embedding!r}")
        for R in self.rows:
            ms = self.blocks_of(R)
            if any(not 1 <= M <= R or R % M for M in ms) or len(set(ms)) != len(ms):
                raise EngineError(f"blocks for R={R} must be distinct divisors of it: {ms}")
            if len(ms) > 1 and R not in (8, 32):
                raise EngineError(f"several blocks are for R = 8 / 32 (K2a) only: R={R} {ms}")


def _l1_top_value(v):
    return {int(R): int(t) for R, t in v.items()} if isinstance(v, dict) else int(v)


@dataclass
class KVPools:
    kv: list  # 60 (K, V) ttnn tensors in layer order
    layer_group: list  # 60 ints: the `tables` row each layer reads
    scratch: list  # scratch[g][slot]: the slot's scratch page id in group g's geometry
    rope: dict = field(default_factory=dict)  # kind -> (cos, sin) row-major [positions, D] bf16

    def validate(self, max_users, layer_kinds):
        if len(self.kv) != len(layer_kinds) or len(self.layer_group) != len(layer_kinds):
            raise EngineError("kv and layer_group need one entry per layer")
        pages, kinds = {}, {}
        for l, ((k, v), g, full) in enumerate(zip(self.kv, self.layer_group, layer_kinds)):
            geom = FULL if full else SLIDING
            for t in (k, v):
                if tuple(t.shape)[1:] != geom or t.dtype != ttnn.bfloat8_b or t.layout != ttnn.TILE_LAYOUT:
                    raise EngineError(f"layer {l}: expected [P, {geom}] bf8 TILE, got {tuple(t.shape)} {t.dtype}")
                if t.memory_config().buffer_type != ttnn.BufferType.DRAM or t.memory_config().is_sharded():
                    raise EngineError(f"layer {l}: KV must be interleaved DRAM")
            if k.buffer_address() == v.buffer_address():
                raise EngineError(f"layer {l}: K and V share storage")
            if not 0 <= g < GROUPS:
                raise EngineError(f"layer {l}: group {g} outside 0..{GROUPS - 1}")
            p = int(k.shape[0])
            if pages.setdefault(g, p) != p:
                raise EngineError(f"group {g}: layers disagree on pool pages ({pages[g]} vs {p})")
            if kinds.setdefault(g, full) != full:
                raise EngineError(f"group {g}: holds both sliding and full layers (M5 per-unit tables)")
        if len(self.scratch) < max(self.layer_group) + 1 or any(len(s) < max_users for s in self.scratch):
            raise EngineError("scratch needs a page per slot for every group")
        for g, p in pages.items():
            if any(not 0 <= s < p for s in self.scratch[g][:max_users]):
                raise EngineError(f"group {g}: scratch page outside the pool")
        return pages


@dataclass
class EngineStatus:
    launches: int
    flags: int
    row: int
    position: int
    page: int
    entry: int
    users: tuple = ()  # M5: per user block u of the launches since the last clear, its sticky flags (32 entries)


class _EngineBase:
    """the tensors and host-side calls StubEngine and ResidentEngine share"""

    MAX_CAPACITY = 262144  # M1L: chunked full-layer workers (attention L1 constant in the context)

    def __init__(self, mesh, cfg, pools, layer_kinds):
        cfg.validate(self.MAX_CAPACITY)
        self.mesh, self.cfg, self.pools = mesh, cfg, pools
        self.pool_pages = pools.validate(cfg.max_users, layer_kinds)
        self.max_pages = cfg.capacity // PAGE
        self.launches = 0
        self.feed, self.write_logits = "none", False
        rep = ttnn.ReplicateTensorToMesh(mesh)
        dram = ttnn.DRAM_MEMORY_CONFIG

        def rm(x, dtype):
            return ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep,
                                   memory_config=dram)

        self.tokens = rm(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.uint32)
        self.positions = rm(torch.full((1, 1, 1, ROWS), -1, dtype=torch.int32), ttnn.int32)
        self.slots = rm(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.int32)
        self.status = rm(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.uint32)
        self.control = rm(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.uint32)
        # (M5 A4) every launch's per-row argmax ids (the hub writes them in every feed mode; inert / flagged rows 0),
        # and the adapter's greedy-row mask for lane S's restore where(greedy_mask, argmax_ids, sampled)
        self.argmax_ids = rm(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.uint32)
        self.greedy_mask = rm(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.int32)
        self._tables_host = torch.zeros(cfg.max_users, GROUPS, self.max_pages, dtype=torch.int32)
        for g in range(GROUPS):
            if g in self.pool_pages:
                self._tables_host[:, g, :] = torch.tensor(pools.scratch[g][: cfg.max_users], dtype=torch.int32)[:, None]
        self.tables = rm(self._tables_host.clone(), ttnn.int32)
        # (M5 A5) row-granular write_tables: staged slot rows + an index row, scattered on device (table_scatter.cpp)
        self._tab_stage = rm(torch.zeros(TAB_STAGE, GROUPS, self.max_pages, dtype=torch.int32), ttnn.int32)
        self._tab_idx = rm(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.uint32)
        self._scatter_pd = None
        self.logits = (
            ttnn.from_torch(torch.zeros(1, 1, ROWS, VOCAB), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3), memory_config=dram)
            if cfg.logits else None
        )
        n = max(1, len(cfg.tap_layers))
        self._taps = [
            ttnn.from_torch(torch.zeros(1, 1, ROWS, n * HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh, mesh_mapper=rep, memory_config=dram)
            for _ in range(2)
        ] if cfg.tap_layers else []
        self._taps_off = False  # (m5_13) the runtime tap mask (set_taps_enabled); False = m5_11's behaviour
        # host mirror of the feed rows, for permute_slots / release_slot without a device read
        self._rows = (torch.zeros(ROWS, dtype=torch.int32), torch.full((ROWS,), -1, dtype=torch.int32),
                      torch.zeros(ROWS, dtype=torch.int32))

    # ---- host -> device
    def _write(self, host, dev):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host.contiguous(), dtype=dev.dtype, layout=dev.layout,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)),
            dev,
        )

    def write_rows(self, slots, positions, tokens):
        slots, positions, tokens = (list(map(int, x)) for x in (slots, positions, tokens))
        n = len(slots)
        if not (n == len(positions) == len(tokens)) or n > ROWS:
            raise EngineError(f"write_rows needs three sequences of one length <= {ROWS}")
        prev = None
        for r, (s, p, t) in enumerate(zip(slots, positions, tokens)):
            if not 0 <= t < VOCAB:
                raise EngineError(f"row {r}: token {t} outside the vocabulary")
            if p == -1:
                prev = None
                continue
            if not 0 <= s < self.cfg.max_users:
                raise EngineError(f"row {r}: slot {s} outside 0..{self.cfg.max_users - 1}")
            if not 0 <= p < self.cfg.capacity:
                raise EngineError(f"row {r}: position {p} outside 0..{self.cfg.capacity - 1}")
            if prev is not None and prev[0] == s and p != prev[1] + 1:
                raise EngineError(f"row {r}: rows of slot {s} must have consecutive positions")
            prev = (s, p)
        seen, last = [], None
        for r, (s, p) in enumerate(zip(slots, positions)):
            if p < 0:
                last = None
                continue
            if last != s:
                if s in seen:
                    raise EngineError(f"slot {s}: its rows must be consecutive")
                seen.append(s)
            last = s
        t = torch.zeros(ROWS, dtype=torch.int32)
        pos = torch.full((ROWS,), -1, dtype=torch.int32)
        sl = torch.zeros(ROWS, dtype=torch.int32)
        t[:n], pos[:n], sl[:n] = torch.tensor(tokens), torch.tensor(positions), torch.tensor(slots)
        # (F1) rows no captured program can serve are refused here; launch(R) checks R's own program
        whys = [self._unservable(R, pos, sl, M) for R in self.cfg.rows if R >= n for M in self.cfg.blocks_of(R)]
        if whys and all(w is not None for w in whys):
            raise EngineError(f"write_rows: {whys[0]}")
        self._rows = (t, pos, sl)
        self._write(t.reshape(1, 1, 1, ROWS), self.tokens)
        self._write(pos.reshape(1, 1, 1, ROWS), self.positions)
        self._write(sl.reshape(1, 1, 1, ROWS), self.slots)

    def write_tables(self, tables, slots=None):
        tables = torch.as_tensor(tables).to(torch.int32)
        idx = list(range(self.cfg.max_users)) if slots is None else [int(s) for s in slots]
        if tuple(tables.shape) != (len(idx), GROUPS, self.max_pages):
            raise EngineError(f"tables must be [{len(idx)}, {GROUPS}, {self.max_pages}], got {tuple(tables.shape)}")
        if any(not 0 <= s < self.cfg.max_users for s in idx):
            raise EngineError("write_tables: slot outside max_users")
        for g, p in self.pool_pages.items():
            col = tables[:, g, :]
            if bool(((col < 0) | (col >= p)).any()):
                raise EngineError(f"group {g}: table entries must be in [0, {p})")
        self._tables_host[idx] = tables
        if slots is not None and len(idx) <= TAB_STAGE:  # (M5 A5) only the slots' rows reach the device
            st = torch.zeros(TAB_STAGE, GROUPS, self.max_pages, dtype=torch.int32)
            st[: len(idx)] = tables
            iw = torch.zeros(ROWS, dtype=torch.int32)
            iw[0] = len(idx)
            iw[1: 1 + len(idx)] = torch.tensor(idx, dtype=torch.int32)
            self._write(st, self._tab_stage)
            self._write(iw.reshape(1, 1, 1, ROWS), self._tab_idx)
            self._scatter()
        else:
            self._write(self._tables_host, self.tables)

    def _scatter(self):
        """(M5 A5) table_scatter.cpp on one core per chip, eagerly on CQ 0: after the previous launch, before the next"""
        if self._scatter_pd is None:
            from models.demos.gemma4_31b_qb2.resident_m5_14 import PKG_DIR

            core = ttnn.CoreCoord(0, 0)
            cr = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
            page = self.max_pages * 4
            cb = ttnn.CBDescriptor(total_size=page + 128, core_ranges=cr, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=page + 128)])
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = [self._tab_stage.buffer_address(), self._tab_idx.buffer_address(),
                                  self.tables.buffer_address()]
            kd = ttnn.KernelDescriptor(
                kernel_source=f"{PKG_DIR}/kernels/table_scatter.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cr, compile_time_args=[GROUPS, page], runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                         noc=ttnn.NOC.RISCV_1_default))
            prog = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])
            pd = ttnn.MeshProgramDescriptor()
            for chip in range(self.mesh.get_num_devices()):
                coord = ttnn.MeshCoordinate(0, chip)
                pd[ttnn.MeshCoordinateRange(coord, coord)] = prog
            self._scatter_pd = pd
        ttnn.generic_op([self._tab_stage, self._tab_idx, self.tables], self._scatter_pd)

    def set_mode(self, feed, write_logits):
        if feed not in FEEDS:
            raise EngineError(f"feed must be one of {FEEDS}")
        if write_logits and self.logits is None:
            raise EngineError("write_logits needs EngineConfig(logits=True)")
        self.feed, self.write_logits = feed, bool(write_logits)
        c = torch.zeros(1, 1, 1, ROWS, dtype=torch.int32)
        c[..., 0], c[..., 1] = FEEDS.index(feed), int(write_logits)
        self._write(c, self.control)

    def clear_status(self):
        s = torch.zeros(1, 1, 1, ROWS, dtype=torch.int32)
        s[..., ST_LAUNCHES] = self.read_status().launches
        s[..., ST_TAPS_OFF] = int(self._taps_off)  # (m5_13) the tap mask survives a clear
        self._write(s, self.status)

    def set_taps_enabled(self, on):
        """(m5_13, M7 addendum 2 F1) the runtime tap mask: False = the programs skip every tap write (PLAIN with taps built,
        about 0 cost), True = taps written as built (the default, m5_11's behaviour). Status word ST_TAPS_OFF, which the
        hub preserves (it only ORs flags and bumps the launch counter) and the slices read at every launch, so it is trace
        safe. Call between synced launches. The taps are side outputs: tokens, logits and KV are the same either way."""
        if not self._taps:
            raise EngineError("no taps (EngineConfig(tap_layers=()))")
        self._taps_off = not bool(on)
        st = self._read0(self.status)[:ROWS].to(torch.int32)
        st[ST_TAPS_OFF] = int(self._taps_off)
        self._write(st.reshape(1, 1, 1, ROWS), self.status)

    @property
    def taps_enabled(self):
        return not self._taps_off

    # ---- device -> host
    def _read0(self, t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1)

    def read_argmax_ids(self, n):
        """(M5 A4) the last launch's argmax ids of rows 0..n-1 (0 for inert or flagged rows), in every feed mode"""
        return self._read0(self.argmax_ids)[:n].to(torch.int64) & 0xFFFFFFFF

    def write_greedy_mask(self, mask):
        """(M5 A4) rows whose next token is the argmax (1) or a sampler's (0): lane S's restore reads it on device"""
        m = torch.zeros(ROWS, dtype=torch.int32)
        m[: len(mask)] = torch.tensor([int(bool(x)) for x in mask], dtype=torch.int32)
        self._write(m.reshape(1, 1, 1, ROWS), self.greedy_mask)

    def read_tokens(self, n):
        return self._read0(self.tokens)[:n].to(torch.int64) & 0xFFFFFFFF

    def read_logits(self, n):
        if self.logits is None:
            raise EngineError("no logits output (EngineConfig(logits=False))")
        return ttnn.to_torch(self.logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3)).float()[0, 0, :n]

    def read_status(self):
        w = self._read0(self.status).to(torch.int64) & 0xFFFFFFFF
        signed = lambda v: v - (1 << 32) if v >= 1 << 31 else v
        users = tuple(int(w[ST_USER + u // 8] >> (4 * (u % 8))) & 0xF for u in range(ROWS))
        return EngineStatus(int(w[ST_LAUNCHES]), int(w[ST_FLAGS]), signed(int(w[ST_ROW])), signed(int(w[ST_POS])),
                            signed(int(w[ST_PAGE])), signed(int(w[ST_ENTRY])), users)

    def taps_buffer(self, k):
        if not self._taps:
            raise EngineError("no taps (EngineConfig(tap_layers=()))")
        return self._taps[k % 2]

    # ---- lifecycle
    def release_slot(self, slot):
        """(M5 A7) reset only `slot`: its rows' positions to -1 and its tables to its scratch pages. The positions are a
        read-modify-write of the DEVICE copy after a sync, so every other row keeps its device-advanced position (the
        host mirror of the last write_rows is stale under device feeds; m5_3 re-uploaded it and rewound co-tenants)."""
        if not 0 <= slot < self.cfg.max_users:
            raise EngineError(f"slot {slot} outside max_users")
        ttnn.synchronize_device(self.mesh)
        pos = (self._read0(self.positions)[:ROWS].to(torch.int64) & 0xFFFFFFFF).to(torch.int64)
        pos = torch.where(pos >= 1 << 31, pos - (1 << 32), pos).to(torch.int32)
        sl = self._read0(self.slots)[:ROWS].to(torch.int32)
        pos[(sl == slot) & (pos >= 0)] = -1
        self._rows = (self._rows[0].clone(), pos.clone(), sl.clone())
        self._write(pos.reshape(1, 1, 1, ROWS), self.positions)
        rows = self._tables_host[slot].clone()
        for g in self.pool_pages:
            rows[g, :] = self.pools.scratch[g][slot]
        self.write_tables(rows[None], slots=[slot])

    def permute_slots(self, moves):
        moves = {int(a): int(b) for a, b in moves.items()}
        if any(not 0 <= x < self.cfg.max_users for kv in moves.items() for x in kv):
            raise EngineError("permute_slots: slot outside max_users")
        ttnn.synchronize_device(self.mesh)
        tables = self._tables_host.clone()
        for old, new in moves.items():
            tables[new] = self._tables_host[old]
        self._tables_host = tables
        self._write(self._tables_host, self.tables)
        sl = self._read0(self.slots)[:ROWS].to(torch.int32)
        sl = torch.tensor([moves.get(int(s), int(s)) for s in sl], dtype=torch.int32)
        self._rows = (self._rows[0], self._rows[1], sl)
        self._write(sl.reshape(1, 1, 1, ROWS), self.slots)


class StubEngine(_EngineBase):
    """The M1_INTERFACE API with fake, deterministic outputs and no weights, for building the serving adapter.

    launch(R) is eager and synchronizing (it computes on the host): active rows (position >= 0) get
    out = (token + 1) % VOCAB; feed 'argmax' writes out to `tokens` and advances positions, 'external' only advances
    positions (the sampler writes `tokens`), 'none' writes out and leaves positions; with write_logits, logits rows are
    one-hot (value 20.0, capped as the real engine caps: 30 tanh(20 / 30)) at `out`. Inactive rows keep their tokens.
    Status word 0 counts launches. Taps: row r of tap i holds float(i + 1) * (position + 1) in every column."""

    def __init__(self, mesh, cfg, pools, layer_kinds=None):
        kinds = layer_kinds if layer_kinds is not None else [l % 6 == 5 for l in range(60)]
        super().__init__(mesh, cfg, pools, kinds)
        self._launch_count = 0

    def warmup(self):
        ttnn.synchronize_device(self.mesh)

    def cb_top(self):
        return 0

    def cb_tops(self):
        return {R: 0 for R in self.cfg.rows}

    def rebuild(self, l1_top):
        self.cfg = EngineConfig(**{**self.cfg.__dict__, "l1_top": _l1_top_value(l1_top)})
        self.cfg.validate(self.MAX_CAPACITY)

    def drop_rows(self, rows):
        keep = tuple(R for R in self.cfg.rows if R not in {int(r) for r in rows})
        if not keep:
            raise EngineError("drop_rows would leave no rows; use close()")
        l1 = {R: t for R, t in self.cfg.l1_top.items() if R in keep} if isinstance(self.cfg.l1_top, dict) else self.cfg.l1_top
        self.cfg = EngineConfig(**{**self.cfg.__dict__, "rows": keep, "l1_top": l1})

    def close(self):
        pass

    def launch(self, R):
        if R not in self.cfg.rows:
            raise EngineError(f"R {R} not captured (rows {self.cfg.rows})")
        ttnn.synchronize_device(self.mesh)
        tok = self._read0(self.tokens)[:ROWS].to(torch.int64) & 0xFFFFFFFF
        pos = self._read0(self.positions)[:ROWS].to(torch.int64)
        active = pos >= 0
        if bool(active[R:].any()):
            raise EngineError(f"rows >= R={R} must be inert")
        out = torch.where(active, (tok + 1) % VOCAB, tok)
        if self.feed in ("argmax", "none"):
            self._write(out.to(torch.int32).reshape(1, 1, 1, ROWS), self.tokens)
        if self.feed in ("argmax", "external"):
            self._write(torch.where(active, pos + 1, pos).to(torch.int32).reshape(1, 1, 1, ROWS), self.positions)
        if self.write_logits:
            lg = torch.zeros(1, 1, ROWS, VOCAB)
            for r in range(R):
                if active[r]:
                    lg[0, 0, r, int(out[r])] = 30.0 * torch.tanh(torch.tensor(20.0 / 30.0))
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(lg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=3)),
                self.logits,
            )
        if self._taps:
            n = len(self.cfg.tap_layers)
            tp = torch.zeros(1, 1, ROWS, n * HIDDEN)
            for i in range(n):
                tp[0, 0, :, i * HIDDEN : (i + 1) * HIDDEN] = ((i + 1) * (pos.float() + 1))[:, None]
            self._write(tp, self._taps[self.launches % 2])
        st = self._read0(self.status)[:ROWS].to(torch.int32)
        st[ST_LAUNCHES] += 1
        self._write(st.reshape(1, 1, 1, ROWS), self.status)
        self.launches += 1


@dataclass
class ServeIO:
    """what the resident programs (model.py) read and write: the engine's tensors and the pools' geometry"""

    tokens: object
    positions: object
    slots: object
    control: object
    status: object
    tables: object
    scratch_t: object
    logits_raw: object
    argmax_ids: object
    x0: object  # (M5 E) the engine's TILE x0 [1, 1, 32, 5376] (embedding "ttnn"), else None: the slices read the table
    taps: list
    tap_layers: tuple
    capacity: int
    max_users: int
    layer_group: list
    pool_pages: dict
    kv: list
    rope: dict


class ResidentEngine(_EngineBase):
    """The resident programs behind the M1_INTERFACE API: one program per captured R (all reading the same engine
    tensors, one weight copy), two traces each (with / without the logits cap, ttnn.unary_chain as tt/model.py)."""

    def __init__(self, mesh, cfg, pools, *, layers=60, embedding_t=None, ccl=None, **model_kw):
        """(M5 E) embedding "ttnn": embedding_t is ttnn's embedding (the generator's model.embedding: [262144, 5376]
        bf16 ROW_MAJOR, sharded by column over the chips); ccl its MeshCCL (default: a new MeshCCL(mesh, 2, Ring))"""
        from models.demos.gemma4_31b_qb2.resident_m5_14.common import is_full

        if cfg.capacity % 2048:
            raise EngineError("capacity must be a multiple of 2048 (table rows are 64 B aligned DRAM pages)")
        kinds = [is_full(l) for l in range(layers)]
        super().__init__(mesh, cfg, pools, kinds)
        if set(pools.rope) != {"sliding_attention", "full_attention"}:
            raise EngineError("rope needs the sliding and full decode (cos, sin) tables")
        rep = ttnn.ReplicateTensorToMesh(mesh)
        scr = torch.zeros(cfg.max_users, 1, 16, dtype=torch.int32)
        for g in self.pool_pages:
            scr[:, 0, g] = torch.tensor(pools.scratch[g][: cfg.max_users], dtype=torch.int32)
        self.scratch_t = ttnn.from_torch(scr, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                         mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.logits_raw = (
            ttnn.from_torch(torch.zeros(1, 1, ROWS, VOCAB), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if cfg.logits else None
        )
        # (M5 E) x0 = all_gather(embedding(tokens, ttnn's shard) * bf16(sqrt(5376))) into x0_t, ahead of every launch
        # (eager and in every trace); bitwise the resident table's rows (bench/resident/m5_5/e_x0_probe3.py)
        self.x0_t = None
        if cfg.embedding == "ttnn":
            if embedding_t is None:
                raise EngineError("embedding 'ttnn' needs embedding_t (ttnn's column-sharded embedding)")
            if ccl is None:
                from models.demos.gemma4_31b_qb2.tt.decoder import MeshCCL

                ccl = MeshCCL(mesh, 2, ttnn.Topology.Ring)
            self._emb_t, self._ccl = embedding_t, ccl
            self._emb_scale = torch.tensor(HIDDEN ** 0.5, dtype=torch.bfloat16).item()
            sh = ttnn.ShardTensorToMesh(mesh, dim=2)
            self._emb_o, self._emb_sc = (
                ttnn.from_torch(torch.zeros(1, ROWS, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                mesh_mapper=sh, memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(2))
            self.x0_t = ttnn.from_torch(torch.zeros(1, 1, ROWS, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        device=mesh, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._tok_v = ttnn.reshape(self.tokens, ttnn.Shape([1, ROWS]))
        self.io = ServeIO(
            tokens=self.tokens, positions=self.positions, slots=self.slots, control=self.control, status=self.status,
            tables=self.tables, scratch_t=self.scratch_t, logits_raw=self.logits_raw, argmax_ids=self.argmax_ids,
            x0=self.x0_t,
            taps=self._taps,
            tap_layers=tuple(cfg.tap_layers), capacity=cfg.capacity, max_users=cfg.max_users,
            layer_group=list(pools.layer_group), pool_pages=dict(self.pool_pages), kv=list(pools.kv), rope=pools.rope,
        )
        self.layers, self.model_kw = layers, model_kw
        self.models, self.block_models, self.traces = {}, {}, {}
        self.test_allow_users = False  # (tests only: launch blocks of users >= 1 the attention does not serve yet)
        # (M5 D18 / D21, XCHIP_SWITCH_DESIGN option c) the last launched program (R, M): a launch of another program
        # first finishes the mesh, so no chip starts it while a peer still runs the previous one (fabric packets of
        # different programs land at the same peer L1 addresses; a repeated program is safe by parity slots)
        self._last_prog = None
        self.switch_syncs = 0
        self._owner = None
        self._build()

    def _build(self):
        from models.demos.gemma4_31b_qb2.resident_m5_14.model import ResidentGemma

        self.models, self.block_models = {}, {}
        for R in sorted(self.cfg.rows):
            for M in self.cfg.blocks_of(R):  # (M5 A2: one program per block; models[R] is the default block's)
                m = ResidentGemma(self.mesh, self.cfg.weight_cache, self.io, layers=self.layers, rows=R,
                                  taps=self.cfg.tap_layers, shared=self._owner, l1_top=self.cfg.l1_top_for(R),
                                  block=M, **self.model_kw)
                if self._owner is None:
                    self._owner = m  # holds the one weight copy for every later program
                self.block_models[(R, M)] = m
                if M == self.cfg.block_for(R):
                    self.models[R] = m

    def _x0(self):
        """(M5 E) the launch's x0 from the tokens on device (every output preallocated: trace safe)"""
        if self.x0_t is None:
            return
        ttnn.embedding(self._tok_v, self._emb_t, layout=ttnn.TILE_LAYOUT, output_tensor=self._emb_o)
        ttnn.multiply(self._emb_o, self._emb_scale, output_tensor=self._emb_sc)
        ttnn.experimental.all_gather_async(
            ttnn.reshape(self._emb_sc, ttnn.Shape([1, 1, ROWS, HIDDEN // self.mesh.get_num_devices()])),
            persistent_output_buffer=self.x0_t, dim=3, cluster_axis=1,
            multi_device_global_semaphore=self._ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self._ccl.get_barrier_semaphore(), num_links=self._ccl.num_links,
            topology=self._ccl.topology, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _cap(self):
        ttnn.unary_chain(
            self.logits_raw,
            [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1.0 / 30.0),
             ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH, 0),
             ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 30.0)],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=self.logits,
        )

    def warmup(self):
        """every program once eagerly with all rows inert, then the traces; the counters restart at 0"""
        self.write_rows([], [], [])
        for m in self.block_models.values():
            for logits in (False, True) if self.logits is not None else (False,):
                self.set_mode("none", logits)
                m.reset()
                self._x0()
                m.launch()
                if logits:
                    self._cap()
                ttnn.synchronize_device(self.mesh)
        self._release_traces()
        for (R, M), m in self.block_models.items():
            m.reset()
            for logits in (False, True) if self.logits is not None else (False,):
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self._x0()
                ttnn.generic_op(m.io, m.mesh_pd)
                if logits:
                    self._cap()
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                self.traces[(R, M, logits)] = tid
        ttnn.synchronize_device(self.mesh)
        st0 = torch.zeros(1, 1, 1, ROWS, dtype=torch.int32)
        st0[..., ST_TAPS_OFF] = int(self._taps_off)  # (m5_13) the tap mask survives the warmup's reset
        self._write(st0, self.status)
        self.set_mode("none", False)
        self.launches = 0

    def blocks_for(self, R):
        """(M5 A2, M1_INTERFACE_M5_ADDENDUM_2) the user-block sizes M R's programs were built for; launch(R, block=M)
        takes one of them, by default the first"""
        return self.cfg.blocks_of(R)

    def _k2a(self, R, M):
        """(M5 D16) whether (R, M) is a K2a program: R = 32, or (model kw k2a_r8) R = 8 with blocks of M < 8 rows"""
        return R == 32 or (R == 8 and M < 8 and bool(getattr(self, "model_kw", {}).get("k2a_r8", False)))

    def users_for(self, R, block=None):
        """(M5 A1) users one launch of R serves: R // M on a K2a program (R = 32; D16: R = 8 with M < 8; M = `block`,
        else blocks_for(R)[0]), one (user block 0, any slot) otherwise"""
        M = self.cfg.block_for(R) if block is None else int(block)
        return R // M if self._k2a(R, M) else 1

    def verify_cap(self, R, p, block=None):
        """(m5_14 F1, M131) the rows user block `block` (default blocks_for(R)[0]) of R's program may use at position p:
        verify_cap(p, M) on a K2a program, M otherwise (the M1L programs have no merge switch)"""
        M = self.cfg.block_for(R) if block is None else int(block)
        return verify_cap(int(p), M) if self._k2a(R, M) else M

    def capacity_for(self, R, block=None):
        """(M5 A3) the highest position + 1 R's program serves: K2a (R = 32; D16: R = 8 with M < 8) caps it at
        k2a.MAX_POS until K2b"""
        from models.demos.gemma4_31b_qb2.resident_m5_14.k2a import MAX_POS

        M = self.cfg.block_for(R) if block is None else int(block)
        return min(self.cfg.capacity, MAX_POS) if self._k2a(R, M) else self.cfg.capacity

    def _unservable(self, R, pos, sl, M=None):
        """(M5 host guard, K2A_DESIGN_v2 addendum 1 s2) why R's program cannot serve these rows, or None. Every user
        block of block_for(R) rows (user u: rows [u M, (u + 1) M)) serves one slot: its live rows share the block's
        first row's slot and continue its positions. The R <= 8 attention serves user 0 only (a user u >= 1 with a
        valid block is refused); the R = 32 program (K2a) serves every block. test_allow_users lifts the user check."""
        M = self.cfg.block_for(R) if M is None else int(M)
        for u in range(R // M):
            p0, s0 = int(pos[u * M]), int(sl[u * M])
            for m in range(1, M):
                p, s = int(pos[u * M + m]), int(sl[u * M + m])
                if p >= 0 and (p0 < 0 or s != s0 or p != p0 + m):
                    return (f"row {u * M + m} (slot {s}, position {p}) is not row {m} of user {u}'s block "
                            f"(slot {s0}, position {p0}): a block of {M} rows serves one slot")
        if self._k2a(R, M):  # (M5 A3) K2a serves every block, positions below capacity_for (D16: R = 8, M < 8 too)
            cap = self.capacity_for(R, M)
            for r in range(R):
                if int(pos[r]) >= cap:
                    return f"row {r} (position {int(pos[r])}) is past the R = {R} K2a capacity {cap} (capacity_for)"
            # (K2b step 3, ADDENDUM_1 D2-4) the unit-count bound (the device's global fit flag F_SCHED is the backstop):
            # greedy by load gives <= ceil(chunks / 60) + 1 full units per core (+ 3 slack, measured), the sliding ones
            # <= ceil(users * 4 * 5 / 60) + 1 + 3; their sum must fit kMaxUnits = 40. (m5_14 F1) a block straddling a
            # chunk boundary has one more unit: full chunks count to its last live row ((p + ml - 1) // 2048, ml = the
            # rows to the last live one, as the unit cores' schedule), and a user straddling 256 c counts 4 x 6 sliding
            # units; both equal m5_13's counts for every other full block
            live = []
            for u in range(R // M):
                if int(pos[u * M]) >= 0:
                    ml = 1 + max(m for m in range(M) if int(pos[u * M + m]) >= 0)
                    live.append((int(pos[u * M]), ml))
                    # (m5_14 F1, M131) the backstop of verify_cap: live rows across a merge boundary would put the
                    # whole block on the boundary's far side (16,384: the full layers' group-node tree);
                    # test_allow_straddle lifts it for a probe's positive control (W1PROBE v4 part (e), cap OFF)
                    if not getattr(self, "test_allow_straddle", False) and verify_cap(live[-1][0], ml) < ml:
                        return (f"user {u}'s live rows {live[-1][0]}..{live[-1][0] + ml - 1} straddle a K2a merge "
                                f"boundary {K2A_MERGE_BOUNDARIES}: cap the block with verify_cap")
            chunks = sum((p + ml - 1) // 2048 + 1 for p, ml in live) * (2 if M == 8 else 1)
            sliding = sum(4 * (6 if p // 256 != (p + ml - 1) // 256 else 5) for p, ml in live)
            bound = -(-chunks // 60) + 4 + -(-sliding // 60) + 4
            if bound > 40:
                return f"{len(live)} users with {chunks} full chunks may exceed 40 attention units on a core"
            return None
        if getattr(self, "test_allow_users", False):
            return None
        for u in range(1, R // M):
            p, s = int(pos[u * M]), int(sl[u * M])
            if p >= 0 and 0 <= s < self.cfg.max_users and p + M <= self.cfg.capacity:
                return (f"user {u} (rows {u * M}..{u * M + M - 1}, slot {s}, position {p}) has a valid block, but the "
                        f"R = {R} attention serves only user 0 (R = 32 serves every user)")
        # rows past the R launch rows are not read; a slot change inside the one block is caught above
        return None

    def _guard_users(self, R, M=None):
        """launch(R)'s check of the last write_rows (before anything reaches the device)"""
        _, pos, sl = self._rows
        why = self._unservable(R, pos, sl, M)
        if why is not None:
            raise EngineError(f"launch({R}): {why}")

    def launch(self, R, block=None):
        """(M5 A2) block: the user-block rows M of this launch (one of cfg.blocks_of(R); default the first)"""
        M = self.cfg.block_for(R) if block is None else int(block)
        tid = self.traces.get((R, M, self.write_logits))
        if tid is None:
            raise EngineError(f"no trace for R={R} block {M} (rows {self.cfg.rows}, blocks {self.cfg.blocks_of(R)}); "
                              f"call warmup() first")
        self._guard_users(R, M)
        if self._last_prog is not None and self._last_prog != (R, M):
            ttnn.synchronize_device(self.mesh)
            self.switch_syncs += 1
        self._last_prog = (R, M)
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        self.launches += 1

    def cb_top(self):
        """the highest L1 address any engine CB occupies (CBs are laid out from the allocator's L1 base)"""
        return max(self.cb_tops().values())

    def cb_tops(self):
        """per R, the highest L1 address its CBs occupy. l1_top bounds the streamers' ring only: at R = 8 the
        attention leader's CBs (1,382,800 at capacity 32768) sit above the streamers and do not move with l1_top"""
        base = ttnn.get_allocator_base_address(self.mesh, ttnn.BufferType.L1)
        tops = {}
        for (R, M), m in self.block_models.items():
            top = 0
            for prog in m.programs:
                per_core = {}
                for d in prog.cbs:
                    for cr in d.core_ranges.ranges():
                        for x in range(cr.start.x, cr.end.x + 1):
                            for y in range(cr.start.y, cr.end.y + 1):
                                per_core[(x, y)] = per_core.get((x, y), 0) + -(-d.total_size // 16) * 16
                top = max(top, max(per_core.values()))
            tops[R] = max(tops.get(R, 0), base + top)
        return tops

    def _release_traces(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
        self.traces = {}

    def rebuild(self, l1_top):
        """l1_top: an int (every R) or {R: int}"""
        cfg = EngineConfig(**{**self.cfg.__dict__, "l1_top": _l1_top_value(l1_top)})
        cfg.validate(self.MAX_CAPACITY)
        ttnn.synchronize_device(self.mesh)
        self._release_traces()
        self.cfg = cfg
        self._build()
        self.warmup()

    def drop_rows(self, rows):
        """remove the programs and traces of the given R (e.g. (8,) when the L1 guard cannot fit R = 8): no DRAM
        allocation, the shared weights and the other R's programs and traces stay as they are"""
        rows = {int(R) for R in rows}
        keep = tuple(R for R in self.cfg.rows if R not in rows)
        if not keep:
            raise EngineError("drop_rows would leave no rows; use close()")
        ttnn.synchronize_device(self.mesh)
        for key in [k for k in self.traces if k[0] in rows]:
            ttnn.release_trace(self.mesh, self.traces.pop(key))
        for R in rows:
            self.models.pop(R, None)  # (the weight owner stays referenced by self._owner)
            for key in [k for k in self.block_models if k[0] == R]:
                self.block_models.pop(key)
        l1 = {R: t for R, t in self.cfg.l1_top.items() if R in keep} if isinstance(self.cfg.l1_top, dict) else self.cfg.l1_top
        self.cfg = EngineConfig(**{**self.cfg.__dict__, "rows": keep, "l1_top": l1})

    def close(self):
        """release the traces and programs and the weight owner: the weights are freed with the last reference to
        the models (M1_INTERFACE §12)"""
        ttnn.synchronize_device(self.mesh)
        self._release_traces()
        self.models, self.block_models = {}, {}
        self._owner = None
