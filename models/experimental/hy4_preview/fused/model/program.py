# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole Hy4-preview decoder layers for 8 token rows on the 8x4 mesh, several of them under one trace (PLAN.md has
the plan and the numbers; this builds it). A step of N layers is 3 N + 1 programs, every one a program of
fused/layer_attn or fused/layer_mlp, built by that piece's own code and unmodified:

    per layer   P1    root site: post-mix of the sublayer before (from the second layer on) + pre-mix and norm of the
                      attention site; X1, pre projections, Xa, query, X3
                P2    attention, X4, post projections, X5: the attention output F lands on the root's site cores
                MLP   root site: post-mix of the attention output + pre-mix and norm of the MLP site; X1', router and
                      experts (the dense FFN on layer 0), X6: the MLP output F lands on the root's site cores
    then        POST  root site alone: the post-mix of the last MLP output

What this module adds is the state the programs share and the per-layer state they are launched with:
  one iHC site     one IhcSite object, one streams tensor, one F, y and xn tensor for both halves and all layers; one
                   DRAM table of all sites (the site is a runtime argument)
  one arena        the attention's L1 arena also carries the MLP half's transport (staging and landings) and the MoE
                   hub's x and output buffers (on the hub core, which has no arena role in P1 / P2)
  per layer        attention: its three projection groups (streams in DRAM, 64 B of go words each), its sink tensor,
                   its gamma tensor, its site index and cache slot, swapped into the one AttnLayer before the layer's
                   descriptors are built: P1 and P2 are ONE program each for all layers, launched with each layer's
                   runtime arguments and buffer addresses (generic_op hashes neither).
                   MLP: an MlpLayer object of its own (two RoutedExperts: their sync words hold the stream addresses
                   and the sync tensor's address is a define, so it is one program per layer), sharing every hub
                   buffer but the router bias (and the dense layer's meta page).

The pieces are not edited. While an MlpLayer is constructed three names it uses are replaced (Stack._building):
layer_mlp.program.tp (so that every MlpLayer gets the one arena transport), layer_mlp.program.IhcSite (the shared
site) and moe.program.hub_tensor (the pool of hub buffers).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.fused.layer_common.sender import Sender
from models.experimental.hy4_preview.fused.layer_mlp import program as M
from models.experimental.hy4_preview.fused.model import hostwrites
from models.experimental.hy4_preview.fused.moe import layout as ML

KERNEL_DIR = Path(__file__).parent / "kernels"
ROWS = ML.ROWS
# the canary's words on the MLP half's sender core of a head (kernels/canary.cpp): row 0 the check inside P2, row 1
# the check inside the MLP program, rows 2 and 3 their scratch
CANARY_CORE = M.SENDER
C_CHECKS, C_FOUND, C_SEEN, C_TIMEOUTS, C_POLLS, C_RUNS = range(6)
HIDDEN = A.HIDDEN
PAYLOAD = A.PAYLOAD  # x and F: [8, 6144] BF16
# arena ranges the MLP half uses (the attention's own are in layer_attn/program.py)
MLP_STAGING, MLP_LANDING1, MLP_LANDING2 = A.A_STAGING, A.A_LANDING1, A.A_LANDING2  # sender / reduce cores
HUB_X, HUB_OUT = 0, PAYLOAD  # the MoE hub core


def is_mock() -> bool:
    return tp.is_mock()


class Alias:
    """A buffer that is a range of the arena on one core: what a piece asks of a hub tensor when it only builds
    descriptors is its address."""

    def __init__(self, address: int, nbytes: int):
        self.address, self.nbytes = int(address), int(nbytes)

    def buffer_address(self) -> int:
        return self.address


def sanity_program(mesh_device, buffer):
    """A program that cannot stall anything: the canary kernel alone on one core of every chip, with a condition that
    is met at once and no counter to read (data movement, one NoC read inside the chip). ``buffer``: a 256-byte uint32
    L1 buffer on CANARY_CORE (tp.core_buffer(mesh, [CANARY_CORE], 256, ttnn.uint32, width=16)). (io, descriptor).
    After it every chip's word 5 (runs) and word 0 (checks) of the buffer's row 2 have gone up by one."""
    base = tp.address(buffer)
    noc = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*CANARY_CORE))
    mesh_desc = ttnn.MeshProgramDescriptor()
    x, y = CANARY_CORE
    rows, cols = tuple(mesh_device.shape)
    for coord in [(r, c) for r in range(rows) for c in range(cols)]:
        rt = ttnn.RuntimeArgs()
        rt[x][y] = [noc.x | (noc.y << 16), base + 192, 0, 4, base + 128, base + 224, 0]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "canary.cpp"),
            core_ranges=tp.core_set([CANARY_CORE]),
            compile_time_args=[],
            runtime_args=rt,
            defines=[],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        desc = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
    return [buffer, buffer], mesh_desc  # (generic_op wants an input and an output tensor)


def sanity_counts(buffer):
    """Per chip (checks, runs) of the sanity program's words."""
    out = []
    for t in ttnn.get_device_tensors(buffer):
        words = (ttnn.to_torch(t).to(torch.int64) & 0xFFFFFFFF).reshape(-1, 16)
        out.append((int(words[2][0]), int(words[2][5])))
    return out


@dataclass
class LayerState:
    """What one layer's three programs are launched with."""

    key: object
    attn: object  # AttnLayer.layer_set(): its projection groups, sink peer, site index, cache slot
    site_attn: int
    site_mlp: int
    cache_layer: int
    mlp: object  # the MlpLayer of the layer's kind
    mlp_state: dict  # MlpLayer.layer_state()
    sel: object = None  # the layer's own selection table (DRAM; None: the one set_cache gave)


class Stack:
    """The shared state and the layers. ``chips`` / ``virtual``: as AttnLayer's (bring-up on one chip). ``attn``: an
    AttnLayer that exists already (a test case of fused/layer_attn built it); else one is made with ``attn_kwargs``."""

    def __init__(
        self,
        mesh_device,
        chips=None,
        attn=None,
        fidelity="recommended",
        moe_fidelity="p02",
        timing=False,
        wait_limit=30_000_000,
        attn_kwargs=None,
        alias_hub=True,
        canary=True,
        parse_only=False,
    ):
        """canary: the heads but the root carry kernels/canary.cpp in P2 and in the MLP program (read_canary).
        parse_only: every sender of the three programs is built parse-only (it writes its argument table as it read
        it to a page and sends nothing; the other kernels run into their bounded waits): ``attn_kwargs`` must then
        hold parse_only=True for the attention half; check_parsed compares the pages."""
        assert M.SITE == A.SITE, "the two halves place the root's site differently"
        assert M.PAYLOAD == A.PAYLOAD
        self.mesh = mesh_device
        self.moe_fidelity, self.timing, self.wait_limit = moe_fidelity, timing, int(wait_limit)
        # every host write into an L1 tensor from here on is registered (hostwrites.py): the restore list after
        # something else wrote over the step's L1. (An AttnLayer handed in was built before: its tensors show up as
        # unregistered.)
        self.hw = hostwrites.install(mesh_device)
        if attn is None:
            # (sink_args: a layer's sink peer is 8 runtime arguments of the merge cores' kernel, no L1 tensor.
            # fixed_order: X4's merge cores take their partners' partials in index order, so that a replay of the
            # same trace gives the same bits: the configuration of record; HY4_FIXED_ORDER=0 for the arrival order)
            import os

            defaults = dict(sink_args=True, fixed_order=os.environ.get("HY4_FIXED_ORDER", "1") != "0")
            attn = A.AttnLayer(
                mesh_device,
                chips=chips,
                fidelity=fidelity,
                timing=timing,
                wait_limit=wait_limit,
                **dict(defaults, **(attn_kwargs or {})),
            )
        self.a = attn
        # the arena is scratch (fused/layer_attn PLAN.md: every data range is written by a step before the step reads
        # it; this model's PLAN.md section 26, run A2, is the evidence) but the three groups' status words, which must be
        # zero at rest
        from models.experimental.hy4_preview.fused.projections import program as PJ

        if attn.arena.is_allocated() and not self.hw.unregistered({"arena": attn.arena}):
            self.hw.scratch(
                attn.arena,
                [
                    (hub, off, 4 * PJ.STATUS_WORDS)
                    for hub, off in (
                        (A.PRE_HUB, A.A_PRE_STATUS),
                        (A.QUERY_HUB, A.A_Q_STATUS),
                        (A.POST_HUB, A.A_POST_STATUS),
                    )
                ],
                "fused/model/PLAN.md section 26",
            )
        self.chips = None if attn.topo.whole_mesh else list(attn.topo.chips)
        self.topo = attn.topo
        self.site = attn.site
        self.alias_hub = alias_hub
        self.hub = None  # the MoE hub core (the same on every chip and for both MlpLayers)
        self.mt = None  # the MLP half's transport, on the arena
        self.table = None
        self.layers = []  # LayerState, in the order they run
        self.parse_only = parse_only
        assert parse_only == bool(attn.parse_only), "parse_only: both halves or neither"
        self.mparse = (
            tp.core_buffer(mesh_device, [M.SENDER], Sender.PARSE_PAGE_BYTES, ttnn.uint32) if parse_only else None
        )
        heads = [c for c in self.topo.chips if self.topo.is_head(c) and c != self.topo.root]
        self.canary = (
            tp.core_buffer(mesh_device, [CANARY_CORE], 256, ttnn.uint32, width=16) if canary and heads else None
        )
        self.canary_limit = max(1000, self.wait_limit // 50)  # polls of about 0.5 us
        ttnn.synchronize_device(mesh_device)
        self.mlps = {}  # dense -> the MlpLayer of that kind: one object for all layers of the kind
        self._go = {}
        self._current = None
        ttnn.synchronize_device(mesh_device)
        self._check_ranges()

    # ------------------------------------------------------------------------------------------ construction
    def _check_ranges(self):
        """The arena ranges of the MLP half against the arena and against each other (PLAN.md section 2)."""
        rows, cols = self.topo.mesh_shape
        parts = len(M.REDUCE)
        part = PAYLOAD // parts
        assert MLP_STAGING + PAYLOAD <= A.ARENA_BYTES
        assert MLP_LANDING1 + cols * part <= MLP_LANDING2 and MLP_LANDING2 + rows * part <= A.ARENA_BYTES
        assert HUB_X + PAYLOAD <= HUB_OUT and HUB_OUT + PAYLOAD <= A.ARENA_BYTES
        mlp_cores = {M.SENDER} | set(M.REDUCE) | set(M.ROOT_REDUCE)
        assert len(mlp_cores) == 1 + 2 * parts, "the MLP half's sender and reduce cores are not distinct"

    def _transport(self, mesh_device, topo, hub, x):
        """The MLP half's transport: ONE, on the attention's arena (MlpLayer(transport=...))."""
        a = self.a
        assert self.hub in (None, tuple(hub)), f"MoE hub core {hub}, the other MlpLayer's is {self.hub}"
        self.hub = tuple(hub)
        assert self.hub not in (
            {M.SENDER} | set(M.REDUCE) | set(M.ROOT_REDUCE) | {A.SENDER} | set(A.MERGE) | set(A.REDUCE)
        )
        if self.mt is None:
            self.mt = tp.Transport(
                mesh_device,
                topo,
                hub,
                M.SENDER,
                M.REDUCE,
                M.PAYLOAD,
                root_reduce_cores=M.ROOT_REDUCE,
                wait_limit=self.wait_limit,
                timing=self.timing,
                arena=(a.arena, {"staging": MLP_STAGING, "landing1": MLP_LANDING1, "landing2": MLP_LANDING2}),
                x_address=x.buffer_address(),
                parse_page=tp.address(self.mparse) if self.parse_only else None,
            )
        assert x.buffer_address() == self.mt.x_address
        return self.mt

    def _mlp(self, dense: bool):
        """The MlpLayer of a kind (MoE / dense FFN): one object for every layer of that kind (layer_args), on the
        shared site state, the arena transport and the hub's x / output as arena ranges."""
        if dense not in self.mlps:
            a = self.a
            kwargs = {} if self.moe_fidelity in (None, "custom_mm") else {"fidelity": self.moe_fidelity}
            hub = {}
            if self.alias_hub:
                hub = {"x": Alias(a.at(HUB_X), PAYLOAD), "out": Alias(a.at(HUB_OUT), PAYLOAD)}
            self.mlps[dense] = M.MlpLayer(
                self.mesh,
                chips=self.chips,
                dense=dense,
                timing=self.timing,
                wait_limit=self.wait_limit,
                moe_kwargs=kwargs,
                layer_args=True,
                site=self.site,
                site_tensors=dict(streams=a.streams, f=a.f, y=a.y, xn=a.xn),
                transport=self._transport,
                hub_tensors=hub,
            )
        return self.mlps[dense]

    def set_sites(self, sites, gammas):
        """Every iHC site of the layers, [(fn, base, scale), ...], and per site the weight of the norm that follows
        its pre-mix: one DRAM table; a layer names its two sites by index."""
        self.table = self.site.site_table(sites, gammas=gammas)
        ttnn.synchronize_device(self.mesh)

    def add_layer(
        self,
        key,
        lw,
        sink,
        site_attn,
        site_mlp,
        cache_layer=0,
        dense=False,
        router_weight=None,
        router_bias=None,
        streams=None,
        moe_layer=None,
    ):
        """One layer: ``lw`` the attention's projections (projections/layout.py layer_weights, gamma not folded),
        ``sink`` [64], the layer's two site indices (into set_sites' table: sites and norm weights), its slot in the
        KV storage; the MLP half: the dense FFN or router weight [256, 6144] and bias [256], and the expert streams
        (default: the prepared files of checkpoint layer ``moe_layer``). ``lw`` / ``sink`` None: what the AttnLayer
        holds already (a test case of fused/layer_attn set it) is this layer's.
        Nothing of a layer stays in L1: its projection streams, expert streams and router weight are DRAM tensors
        whose addresses go into the launches' runtime arguments, its bias and sink peer are runtime arguments, its
        norms' weights are in the site table."""
        assert self.table is not None, "set_sites first"
        a = self.a
        if lw is not None:
            a.set_weights(lw, free=False)  # (the layer before's groups and kv norm weight belong to its set)
            for name in ("pre", "query", "post"):  # one set of go words per group for all layers
                prog = getattr(a, name)
                if name in self._go:
                    ttnn.deallocate(prog.go)
                    prog.go = self._go[name]
                else:
                    self._go[name] = prog.go
        if sink is not None:
            # (the argument form keeps only the peer's values; the tensor form gets a tensor per layer)
            a.set_sink(sink, new=not a.sink_args and len(self.layers) > 0)
        m = self._mlp(dense)
        m.set_layer(
            key if moe_layer is None else moe_layer,
            None,
            site_mlp,
            None,
            router_weight,
            router_bias,
            streams=streams,
            table=self.table,
        )
        a.gamma, a.table, a.site_index, a.cache_layer = (
            None,
            self.table,
            site_attn,
            int(cache_layer),
        )  # (gamma: site table)
        state = LayerState(key, a.layer_set(), site_attn, site_mlp, int(cache_layer), m, m.layer_state())
        self.layers.append(state)
        ttnn.synchronize_device(self.mesh)
        return state

    def set_cache(self, cfg, kv, sel, keep_out=False):
        """The KV storage and the selection table of every layer. A layer whose selection is its own (a context above
        the top-k: the picks differ between full-indexer layers) carries it as LayerState.sel."""
        self.a.set_cache(cfg, kv, sel, 0, keep_out=keep_out)
        self._sel = sel

    # ------------------------------------------------------------------------------------------ programs
    def _use(self, state):
        a = self.a
        for k in A.LayerSet.FIELDS:  # the layer's set as the AttnLayer's current one (AttnLayer.layer_set())
            setattr(a, k, getattr(state.attn, k))
        if getattr(self, "_sel", None) is not None:  # (the table's address is the descriptor's: no new program)
            a.sel = state.sel if state.sel is not None else self._sel
        state.attn.programs = {}
        a._programs = {}
        state.mlp.use(state.mlp_state)
        self._current = state

    def layer_programs(self, state, post: bool, audit: bool = False):
        """[(name, io tensors, MeshProgramDescriptor)] of the layer's P1, P2 and MLP program. post: P1's leading
        pass also does the post-mix of the F on the site cores (a layer before left it)."""
        a = self.a
        self._use(state)
        a._p2_chip = self._p2_chip
        try:
            p1 = a.p1(post=post)
            p2 = a.p2()
        finally:
            del a._p2_chip
        if audit:
            self.audits = getattr(self, "audits", [])
            for coord in self._audit_coords():
                self.audits.append((state.key, "p1", coord, a.feed_audit("p1", coord)))
                self.audits.append((state.key, "p2", coord, a.feed_audit("p2", coord)))
        a._programs = {}
        m = state.mlp
        m._chip_program = lambda coord, post_: self._mlp_chip(m, coord, post_)
        try:
            io, desc = m.program(post=True)
        finally:
            del m._chip_program
        io = [t for t in io if not isinstance(t, Alias)]
        return [("p1", *p1), ("p2", *p2), ("mlp", io, desc)]

    # the two per-chip programs that carry a canary on the heads but the root
    def _canary_kernel(self, kind):
        a, mt = self.a, self.mt
        base = tp.address(self.canary)
        if kind == "p2":  # finished: this chip's sender has served its 4 X5 ops (P2's last read of the reused ranges)
            done = [a.noc_word(A.SENDER), a.aword(A.W_SERVED), (1 << len(A.REDUCE)) - 1]
            counters = [(a.noc_word(c), mt.word_address(tp.W_S1)) for c in M.REDUCE]  # the MLP's stage-1 arrivals
            words, scratch, processor, noc = base, base + 128, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0
        else:  # finished: the reduce cores have staged the row sum (the MLP's last read of its stage-1 landing)
            done = [a.noc_word(M.SENDER), mt.word_address(tp.W_READY), sum(Sender.bit(k) for k in range(len(M.REDUCE)))]
            overlap = [c for c in A.MERGE if c in M.REDUCE]  # landing cores of the next P1's X3
            counters = [(a.noc_word(c), a.aword(A.W_QARR)) for c in overlap]
            words, scratch, processor, noc = base + 64, base + 192, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1
        assert scratch % 16 == 0
        rt = ttnn.RuntimeArgs()
        x, y = CANARY_CORE
        rt[x][y] = done + [self.canary_limit, words, scratch, len(counters)] + [v for pair in counters for v in pair]
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "canary.cpp"),
            core_ranges=tp.core_set([CANARY_CORE]),
            compile_time_args=[],
            runtime_args=rt,
            defines=[],
            config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
        )

    def sanity_program(self):
        """sanity_program on this Stack's canary words: (io, descriptor)."""
        assert self.canary is not None
        return sanity_program(self.mesh, self.canary)

    def sanity_counts(self):
        return sanity_counts(self.canary)

    def _has_canary(self, coord) -> bool:
        coord = tuple(coord)
        return self.canary is not None and self.mt is not None and self.topo.is_head(coord) and coord != self.topo.root

    def _p2_chip(self, coord, stage="all"):
        desc, io = type(self.a)._p2_chip(self.a, coord, stage)
        if stage == "all" and self._has_canary(coord):
            desc.kernels = list(desc.kernels) + [self._canary_kernel("p2")]
            io = io + [self.canary]
        return desc, io

    def _mlp_chip(self, m, coord, post):
        desc, io = type(m)._chip_program(m, coord, post)
        if self._has_canary(coord):
            desc.kernels = list(desc.kernels) + [self._canary_kernel("mlp")]
            io = [self.canary] + io
        return desc, io

    def read_canary(self) -> dict:
        """{head coord: {"p2": [checks, found, seen, timeouts, polls of the last check, runs], "mlp": the same}}."""
        if self.canary is None:
            return {}
        ttnn.synchronize_device(self.mesh)
        tensors = ttnn.get_device_tensors(self.canary)
        out = {}
        for coord in self.topo.chips:
            if self._has_canary(coord):
                words = (ttnn.to_torch(tensors[self.topo.chip(coord)]).to(torch.int64) & 0xFFFFFFFF).reshape(-1, 16)
                out[coord] = {"p2": [int(v) for v in words[0][:6]], "mlp": [int(v) for v in words[1][:6]]}
        return out

    def canary_summary(self) -> dict:
        """Totals over the heads: checks, checks that found an arrival, polls that ran out, runs, per program."""
        out = {}
        for kind in ("p2", "mlp"):
            rows = [v[kind] for v in self.read_canary().values()]
            out[kind] = {
                "checks": sum(r[C_CHECKS] for r in rows),
                "found": sum(r[C_FOUND] for r in rows),
                "timeouts": sum(r[C_TIMEOUTS] for r in rows),
                "runs": sum(r[C_RUNS] for r in rows),
                "max_polls": max([r[C_POLLS] for r in rows], default=0),
            }
        return out

    def check_parsed(self, name, coords=None) -> list:
        """After a parse-only run of program ``name`` ("p1", "p2", "mlp"): the chips whose sender page differs from
        the table the host built, as [(coord, [(index, read, wanted), ...])]."""
        a, wrong = self.a, []
        ttnn.synchronize_device(self.mesh)
        for c in coords or self.topo.chips:
            if name == "mlp":
                want = self.mt.parsed[c]
                page = (
                    ttnn.to_torch(ttnn.get_device_tensors(self.mparse)[self.topo.chip(c)]).to(torch.int64) & 0xFFFFFFFF
                )
                page = [int(v) for v in page.flatten()[: len(want)]]
            else:
                want = a.parsed.get((name, c))
                if want is None:
                    continue
                page = a.read_parsed(c, len(want))
            if page != want:
                wrong.append((c, [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(page, want)) if x != y][:8]))
        return wrong

    def reset(self):
        """After a run that left a status (a parse-only run does): every hand-over word of both halves to zero."""
        a = self.a
        for state in self.layers[:1]:  # (the go words are one set for all layers)
            a.pre, a.query, a.post = state.attn.pre, state.attn.query, state.attn.post
            a.reset()
        for m in self.mlps.values():
            m.reset()
        if a.sm is not None:
            a.sm.reset_semaphores()
        ttnn.synchronize_device(self.mesh)

    def make_restore(self, chips=None, hooks=()):
        """The restore of the step's L1 as a program (restore.py Restore), remembered by the Stack: a host write this
        Stack makes into a stored tensor afterwards (set_frame_tables) refreshes its image in the same call."""
        from models.experimental.hy4_preview.fused.model.restore import Restore

        self.restores = getattr(self, "restores", [])
        self.restores.append(Restore(self.mesh, self.hw, chips=chips, hooks=hooks))
        return self.restores[-1]

    def set_frame_tables(self, table):
        """The frame tables of the 8 decode rows (fused/kv_store KvStore.table of the latent pool's geometry: one flat
        sequence of 8 x 128 frame bases) into BOTH of the attention's table tensors (sparse_mla's row readers', the
        append's), and the image of every restore program of this Stack refreshed: after this call the registry and
        the restore programs hold the new tables as the contents at rest. The one way a binding reaches the step."""
        a = self.a
        table = list(table)
        assert a.sm is not None and a.sm.frame_tables is not None, "set_cache with AttnLayer(reader_frames=True) first"
        a.sm.frame_tables.write(table)
        if a.frames is not None:
            a.frames.write(table)
        ttnn.synchronize_device(self.mesh)
        for r in getattr(self, "restores", []):
            r.refresh()

    def restore(self):
        """After a program that is not the step's wrote over the step's L1 (a prefill pass by trace replay): every L1
        tensor the host ever wrote gets its last host-written contents again, the global semaphores their initial
        values. Returns (tensors written, bytes per chip). The per-step inputs (hw.mark_per_step) are the caller's."""
        return self.hw.restore(self.mesh)

    def _audit_coords(self):
        chips = self.topo.chips
        picks = [self.topo.root] + [c for c in chips if self.topo.is_head(c) and c != self.topo.root][:1]
        return picks + [c for c in chips if not self.topo.is_head(c)][:1]

    def cb_audit(self, state, post: bool):
        """layer_attn's cb_audit (one producer and one consumer processor per circular buffer) on the per-chip
        programs of the layer as this module assembles them: root, a head, another chip. Returns the table rows."""
        a = self.a
        self._use(state)
        rows = []
        for coord in self._audit_coords():
            for name, desc in (
                ("p1", a._p1_chip(coord, post)[0]),
                ("p2", self._p2_chip(coord, "all")[0]),
                ("mlp", self._mlp_chip(state.mlp, coord, True)[0]),
            ):
                rows += [(state.key, name, coord) + row for row in A.cb_audit(desc)]
                # no buffer id is declared twice on a core inside one program
                seen = {}
                for cb in desc.cbs:
                    for fmt in cb.format_descriptors:
                        for core in ttnn.corerange_to_cores(cb.core_ranges):
                            k = (core.x, core.y, fmt.buffer_index)
                            assert (
                                k not in seen
                            ), f"{name} {coord}: circular buffer {fmt.buffer_index} twice on core {(core.x, core.y)}"
                            seen[k] = True
        a._programs = {}
        return rows

    def post_program(self):
        """The post-mix of the last MLP output, alone on the root: (io, MeshProgramDescriptor)."""
        return self.layers[-1].mlp.post_program()

    def step_programs(self, first_post=False, last_post=True, audit=False):
        """Every program of a step over all layers, in order."""
        out = []
        for n, state in enumerate(self.layers):
            for name, io, desc in self.layer_programs(state, post=first_post or n > 0, audit=audit):
                out.append((f"{name}[{state.key}]", io, desc))
        if last_post:
            out.append(("post", *self.post_program()))
        return out

    @staticmethod
    def launch(programs):
        for _, io, desc in programs:
            (ttnn.experimental.prepare_generic_op if is_mock() else ttnn.generic_op)(io, desc)

    # ------------------------------------------------------------------------------------------ host side
    def set_streams(self, x):
        self.a.set_streams(x)

    def set_rope(self, cos, sin):
        self.a.set_rope(cos, sin)

    def read_streams(self):
        return self.a.read_streams()

    def read_root(self):
        return self.a.read_root()

    def read_mlp_chip(self, state, coord, image=None):
        """A chip's MoE hub buffers after the layer's MLP program: x as it arrived [8, 6144], the chip's partial
        output [8, 6144], and on a MoE layer the router's scores [8, 256], picks [8, 8] and weights [8, 8]."""
        a, op = self.a, state.mlp.op
        chip = self.topo.chip(coord)
        image = a.read_arena(coord) if image is None else image
        tiles = lambda off: ML.from_row_tiles(
            A.bytes_bf16(a.arena_bytes(image, self.hub, off, PAYLOAD), (PAYLOAD // 512, 256))
        )
        dev = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[chip])
        if self.alias_hub:
            out = {"x": tiles(HUB_X), "out": tiles(HUB_OUT)}
        else:
            out = {"x": ML.from_row_tiles(dev(op.x).float()), "out": ML.from_row_tiles(dev(op.out).float())}
        if state.mlp.dense:
            return out
        picks = dev(op.picks).to(torch.int32) & 0xFFFF
        return {
            **out,
            "scores": ML.from_row_tiles(dev(op.scores).float()),
            "ids": picks[:, :8],
            "weights": picks[:, 8:].contiguous().to(torch.int16).view(torch.bfloat16).float(),
        }

    def status(self) -> dict:
        """Everything that is not at rest in either half after a step. Empty when the step completed."""
        out = {("attn",) + tuple(k if isinstance(k, tuple) else (k,)): v for k, v in self.a.status().items()}
        for dense, m in self.mlps.items():
            for k, v in m.status().items():
                out[("mlp", "dense" if dense else "moe") + tuple(k if isinstance(k, tuple) else (k,))] = v
        return out

    def l1_bytes(self) -> int:
        return ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1).total_bytes_allocated_per_bank

    def l1_report(self, programs_of=None) -> dict:
        """Persistent L1 tensors per core, and per kind of chip the largest set of program-allocated circular
        buffers of each of the layer's programs (tensor-backed ones take nothing more)."""
        view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1)
        out = {
            "tensors": view.total_bytes_allocated_per_bank,
            "core": view.total_bytes_per_bank,
            "largest_free": getattr(view, "largest_contiguous_bytes_free_per_bank", None),
        }
        if programs_of is None:
            return out
        a = self.a
        self._use(programs_of)
        for coord in self._audit_coords():
            for name, desc in (
                ("p1", a._p1_chip(coord, True)[0]),
                ("p2", self._p2_chip(coord, "all")[0]),
                ("mlp", self._mlp_chip(programs_of.mlp, coord, True)[0]),
            ):
                per_core = {}
                for cb in desc.cbs:
                    backed = cb.has_buffer() if callable(cb.has_buffer) else cb.has_buffer
                    if backed:
                        continue
                    for c in ttnn.corerange_to_cores(cb.core_ranges):
                        per_core[(c.x, c.y)] = per_core.get((c.x, c.y), 0) + cb.total_size
                worst = max(per_core, key=per_core.get)
                out[(name, coord)] = (per_core[worst], worst, len(desc.kernels), len(desc.cbs))
        a._programs = {}
        return out
