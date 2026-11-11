# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The resident engine's serving API (bench/resident/M1_INTERFACE.md, INTERFACE_VERSION 1). resident_m3_2: the frozen
resident_m3 (the M1L engine, capacity 262,144, narrow / wide full layers, with resident_m2's API deltas:
EngineConfig.l1_top per R, cb_tops(), drop_rows(), close() releasing the weight owner; bench/resident/M3_MERGE.md)
plus M2b stage 1: F1 (block-copy K/V splice) and F9 (per-R seam split); bench/resident/M3_1_DELTA.md. resident_m3_2:
resident_m3_1 plus F2 (the slices' h multicast from two RISCs) and F5v (the merges' SFPU ops on the faces they read);
bench/resident/M3_2_DELTA.md.

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
PAGE = 128
FEEDS = ("argmax", "external", "none")
SLIDING = (4, 128, 256)  # per-chip page geometry [kv heads, rows, head_dim]
FULL = (1, 128, 512)

# status words (M1_INTERFACE.md s5)
ST_LAUNCHES, ST_FLAGS, ST_ROW, ST_POS, ST_PAGE, ST_ENTRY = range(6)
F_POSITION, F_SLOT, F_TABLE_COLUMN, F_PAGE = 0x1, 0x2, 0x4, 0x8


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

    def l1_top_for(self, R):
        """the absolute L1 address where R's streamer CBs must end"""
        return int(self.l1_top[R]) if isinstance(self.l1_top, dict) else int(self.l1_top)

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
        if self.embedding not in ("resident",):
            raise EngineError(f"embedding {self.embedding!r} is not available yet (M5 adds 'ttnn')")


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
        pages = {}
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
        self._tables_host = torch.zeros(cfg.max_users, GROUPS, self.max_pages, dtype=torch.int32)
        for g in range(GROUPS):
            if g in self.pool_pages:
                self._tables_host[:, g, :] = torch.tensor(pools.scratch[g][: cfg.max_users], dtype=torch.int32)[:, None]
        self.tables = rm(self._tables_host.clone(), ttnn.int32)
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
        self._write(self._tables_host, self.tables)

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
        self._write(s, self.status)

    # ---- device -> host
    def _read0(self, t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1)

    def read_tokens(self, n):
        return self._read0(self.tokens)[:n].to(torch.int64) & 0xFFFFFFFF

    def read_logits(self, n):
        if self.logits is None:
            raise EngineError("no logits output (EngineConfig(logits=False))")
        return ttnn.to_torch(self.logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3)).float()[0, 0, :n]

    def read_status(self):
        w = self._read0(self.status).to(torch.int64) & 0xFFFFFFFF
        signed = lambda v: v - (1 << 32) if v >= 1 << 31 else v
        return EngineStatus(int(w[ST_LAUNCHES]), int(w[ST_FLAGS]), signed(int(w[ST_ROW])), signed(int(w[ST_POS])),
                            signed(int(w[ST_PAGE])), signed(int(w[ST_ENTRY])))

    def taps_buffer(self, k):
        if not self._taps:
            raise EngineError("no taps (EngineConfig(tap_layers=()))")
        return self._taps[k % 2]

    # ---- lifecycle
    def release_slot(self, slot):
        if not 0 <= slot < self.cfg.max_users:
            raise EngineError(f"slot {slot} outside max_users")
        t, pos, sl = (x.clone() for x in self._rows)
        pos[(sl == slot) & (pos >= 0)] = -1
        self._rows = (t, pos, sl)
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

    def __init__(self, mesh, cfg, pools, *, layers=60, **model_kw):
        from models.demos.gemma4_31b_qb2.resident_m3_2.common import is_full

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
        self.io = ServeIO(
            tokens=self.tokens, positions=self.positions, slots=self.slots, control=self.control, status=self.status,
            tables=self.tables, scratch_t=self.scratch_t, logits_raw=self.logits_raw, taps=self._taps,
            tap_layers=tuple(cfg.tap_layers), capacity=cfg.capacity, max_users=cfg.max_users,
            layer_group=list(pools.layer_group), pool_pages=dict(self.pool_pages), kv=list(pools.kv), rope=pools.rope,
        )
        self.layers, self.model_kw = layers, model_kw
        self.models, self.traces = {}, {}
        self._owner = None
        self._build()

    def _build(self):
        from models.demos.gemma4_31b_qb2.resident_m3_2.model import ResidentGemma

        self.models = {}
        for R in sorted(self.cfg.rows):
            m = ResidentGemma(self.mesh, self.cfg.weight_cache, self.io, layers=self.layers, rows=R,
                              taps=self.cfg.tap_layers, shared=self._owner, l1_top=self.cfg.l1_top_for(R),
                              **self.model_kw)
            if self._owner is None:
                self._owner = m  # holds the one weight copy for every later program
            self.models[R] = m

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
        for m in self.models.values():
            for logits in (False, True) if self.logits is not None else (False,):
                self.set_mode("none", logits)
                m.reset()
                m.launch()
                if logits:
                    self._cap()
                ttnn.synchronize_device(self.mesh)
        self._release_traces()
        for R, m in self.models.items():
            m.reset()
            for logits in (False, True) if self.logits is not None else (False,):
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                ttnn.generic_op(m.io, m.mesh_pd)
                if logits:
                    self._cap()
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                self.traces[(R, logits)] = tid
        ttnn.synchronize_device(self.mesh)
        self._write(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), self.status)
        self.set_mode("none", False)
        self.launches = 0

    def launch(self, R):
        tid = self.traces.get((R, self.write_logits))
        if tid is None:
            raise EngineError(f"no trace for R={R} (rows {self.cfg.rows}); call warmup() first")
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
        for R, m in self.models.items():
            top = 0
            for prog in m.programs:
                per_core = {}
                for d in prog.cbs:
                    for cr in d.core_ranges.ranges():
                        for x in range(cr.start.x, cr.end.x + 1):
                            for y in range(cr.start.y, cr.end.y + 1):
                                per_core[(x, y)] = per_core.get((x, y), 0) + -(-d.total_size // 16) * 16
                top = max(top, max(per_core.values()))
            tops[R] = base + top
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
        l1 = {R: t for R, t in self.cfg.l1_top.items() if R in keep} if isinstance(self.cfg.l1_top, dict) else self.cfg.l1_top
        self.cfg = EngineConfig(**{**self.cfg.__dict__, "rows": keep, "l1_top": l1})

    def close(self):
        """release the traces and programs and the weight owner: the weights are freed with the last reference to
        the models (M1_INTERFACE §12)"""
        ttnn.synchronize_device(self.mesh)
        self._release_traces()
        self.models = {}
        self._owner = None
