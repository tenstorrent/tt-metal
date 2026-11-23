# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Cross-op overlap for decode: while the DRAM-light programs between two uses of a projection run (residual,
message gates, attention prep / out, window), the next app's weights of that projection stream from DRAM into L1
(tt/mk/stage.py), so the next app's projection program reads them from L1.

    ov = install(model)   # before capture; ov.stats() after steps; ov.uninstall(model)

How: for each staged projection (attention w_all and / or w_o), every layer's BankLinear is re-split over more cores
(same weight tensor, placed off the hosting programs' rows) and gets a staged reader over ONE staging tensor shared
by the 24 layers. ``ttnn.generic_op`` is wrapped: after app i's projection ran, app i+1's fill for it is queued;
each later DRAM-light program in HOST_KB carries the next piece of it, sized to end inside that program (budget mode,
the default; M2 mode: ``parts`` equal pieces on programs of at most ``max_cores`` cores). Fill and host cores are
disjoint (asserted). App i+1's projection reads whatever prefix was filled. Bytes into every CB slot are unchanged,
so outputs are bit-equal.

App order: model.APPS (one pointer per projection, advanced by every call): decode steps only.
"""

import functools
import os

import ttnn
from models.demos.inkling2_mini.tt import attention_fast as AF
from models.demos.inkling2_mini.tt import model as MD
from models.demos.inkling2_mini.tt.mk.gate_stage import GateStage
from models.demos.inkling2_mini.tt.mk.stage import StagedBankLinear, coalesce, program_cores, range_bytes, resplit

_orig_generic_op = ttnn.generic_op
_STREAMS = "bank_"  # bank_linear_*, moe_bank_*: weight streams

# Budget mode (Overlap(budget=...), the default): DRAM-light programs that may carry a fill, with the bytes per core
# one piece may fill there. Keyed by (sorted kernel basenames, core count). Sized from the M3 device profiles (4K,
# integrate c99577bf5b7): ~7.5 KB/us per core (2 KB x 1 read in flight on each RISC) x the program's time, then cut
# where the profile showed the host stretched more than the staged bytes saved. Measured and left out: attn_bias
# (32 cores, 14.5 us: 90 KB stretched it to 20.2 us; 30 KB at budget 0.8: +0.08 ms), attn_near / attn_comb (global
# layers; +0.02 ms). The router (rows 2..5, DRAM idle while its collector runs) takes the largest piece: with it,
# budget 0.8 on all hosts is best (router 150 KB: 16.70 ms vs 16.93 without; budget 1.0: 16.98; 0.75 and router
# 180-200 KB worse), then router 130 KB with window / out pieces x1.5: 16.63 (x2: 16.65; attn-res x0.85 or conv
# x1.3: worse).
HOST_KB = {
    ("conv_compute", "conv_reader", "conv_writer", 1): 89,  # 10.6 us -> 11.5 with a fill
    ("attnres_compute", "attnres_reader", "attnres_writer", 1): 89,  # 10.8 -> 13.3
    ("attnres_compute", "attnres_reader", "rownorm_writer", 1): 89,
    ("attnres_compute", "attnres_reader", "rownorm_writer", 3): 37,  # 5.8 -> 8.7
    ("attnres_compute", "attnres_reader", "attnres_writer", 12): 60,
    ("attn_window", 8): 45,  # 5.4 -> 5.9 (30 KB); 45 with the router host
    ("attn_out_compute", "attn_out_reader", "attn_out_writer", 8): 50,  # 6.6 -> 7.1 (33 KB); 50 with the router host
    ("gatez_compute", "gatez_reader", "gatez_writer", 18): 60,  # 14.3 -> 16.1 (mlp gate 8.7, QKV gate 18.8)
    (
        "router_compute",
        "router_reader",
        "router_writer",
        36,
    ): 100,  # 130 before its weights were staged (router_stage.py)
}
BUDGET = 0.8
# KB per staging core one piece may fill inside a local sdpa call (sdpa_decode extra_kernels; 0 = off)
SDPA_KB = 200  # 150 / 200 / 250 / 300: 16.18 / 16.09 / 16.24 / 16.41 ms at 4K (11 us delay); 0 / no hook: 16.29
SDPA_DELAY_US = 11.0  # fill start inside the sdpa call: after its K / V reads (~11 us of 31, G3), DRAM idle after


def sdpa_extra_kernels():
    """Whether this ttnn build's sdpa_decode takes extra_kernels (R9). Checked before any fill piece is taken: a call
    that then failed would leave the staged consumer reading bytes that were never filled."""
    return "extra_kernels" in (ttnn.transformer.scaled_dot_product_attention_decode.__doc__ or "")


def host_key(prog, n_cores):
    return tuple(sorted({os.path.basename(k.kernel_source).rsplit(".", 1)[0] for k in prog.kernels})) + (n_cores,)


class _Proxy:
    """Stands in for one layer's BankLinear in _DramShardedLinear.bank."""

    def __init__(self, track, L, st):
        self.track, self.L, self.st = track, L, st

    def __getattr__(self, name):
        return getattr(self.st.bl, name)

    def __call__(self, x, out=None):
        return self.track.ov.consume(self, x, out)


class _Track:
    """One staged projection (e.g. w_all) over all layers."""

    def __init__(self, ov, model, name, split, exclude, staged_kb):
        self.ov, self.name = ov, name
        self.lins = {L: getattr(lay.attn, name) for L, lay in enumerate(model.layers)}
        self.orig = {L: lin.bank for L, lin in self.lins.items()}
        assert all(b is not None for b in self.orig.values()), f"{name}: needs the bank-streamed projection"
        self.banks = {L: resplit(b, split, exclude) for L, b in self.orig.items()}
        top = max(range_bytes(b) for b in self.banks.values())
        self.staged = top if staged_kb is None else min(staged_kb << 10, top)
        self.cores = {c for c, _ in self.banks[0].cores}
        self.ptr, self.filled, self.filled_for, self.n, self.staged_sum = 0, 0, None, 0, 0

    def bind(self, stage):
        """Proxies over the shared staging tensor (this track's cores are a subset of its grid)."""
        self.proxies = {
            L: _Proxy(self, L, StagedBankLinear(b, self.staged, stage=stage)) for L, b in self.banks.items()
        }
        for L, lin in self.lins.items():
            lin.bank = self.proxies[L]

    def restore(self):
        for L, lin in self.lins.items():
            lin.bank = self.orig[L]


class Overlap:
    def __init__(
        self,
        model,
        weights=("w_all",),
        parts=6,
        max_cores=1,
        staged_kb=None,
        split=3,
        exclude_rows=(0, 1, 2, 3, 4, 5),
        gates=None,
        budget=BUDGET,
        host_kb=None,
        host_scale=None,
        sdpa_kb=None,
    ):
        """max_cores: hosts have at most this many cores (1 = the single-core residual programs, ~22 us each with DRAM
        idle; wider small programs are short, so a fill piece makes them longer instead of hiding). gates: also stage
        the message gates' F / G (tt/mk/gate_stage.py) on their combine cores outside row 0 and outside the
        projections' staging cores; default INKLING_STAGE_GATES (off).
        budget: None = M2 mode (single-core hosts, ``parts`` equal pieces). A number (default BUDGET) = budget mode:
        every program in ``host_kb`` (default HOST_KB) carries fill bytes up to budget x its KB per core, oldest
        queued item first, then the next items on cores disjoint from the ones it filled."""
        self.parts, self.max_cores = parts, max_cores
        self.budget = budget
        # host_scale: {"<kernel basename>[:<cores>]": factor} for the hosts with that kernel (and core count), on top of
        # budget (0 = not a host); for A/B runs
        self.host_kb = dict(HOST_KB if host_kb is None else host_kb)
        for k in list(self.host_kb):
            for name, f in (host_scale or {}).items():
                n, _, c = name.partition(":")
                if n in k[:-1] and (not c or int(c) == k[-1]):
                    self.host_kb[k] *= f
            if self.host_kb[k] <= 0:
                del self.host_kb[k]
        self.order = [L for L, _ in MD.APPS]
        gx = model.device.compute_with_storage_grid_size().x
        # "w_all@480": stage at most 480 KB per core of that projection (the staging tensors take their L1 range on
        # EVERY core: the L1 allocator is lock-step), else staged_kb
        specs = [(w.split("@")[0], int(w.split("@")[1]) if "@" in w else staged_kb) for w in weights]
        # each projection on its own block of cores, so ONE staging tensor (one L1 address range, the allocator is
        # lock-step) holds all of them: the first outside exclude_rows, the next ones on rows 1.. not used yet
        self.tracks, taken = [], set()
        for i, (w, kb) in enumerate(specs):
            rows = set(exclude_rows) if i == 0 else {0} | {y for _, y in taken}
            ex = [(x, y) for y in rows for x in range(gx)] + sorted(taken)
            t = _Track(self, model, w, split, ex, kb)
            assert not (t.cores & taken)
            taken |= t.cores
            self.tracks.append(t)
        gates = os.environ.get("INKLING_STAGE_GATES", "0") if gates is None else gates
        self.gates_tail = gates == "tail"  # gate pieces after the w_all pieces (else ahead of them)
        gates = gates not in (False, "0", None)
        self.gorder, self.gstages = [], {}
        if gates:  # call order of the gates: apps with messages (loop layers, passes > 1), QKV gate then mlp gate
            for L, p in MD.APPS:
                lay = model.layers[L]
                if lay.qkv_gate is not None and p > 1:
                    self.gorder += [lay.qkv_gate, lay.mlp_gate]
            forbidden = {(x, 0) for x in range(gx)} | taken  # row 0: the hosts' cores
            for g in dict.fromkeys(self.gorder):
                self.gstages[id(g)] = GateStage(g, forbidden)
            gcores = set().union(*(gs.stage_cores for gs in self.gstages.values())) if self.gstages else set()
            taken = taken | gcores
        self.gptr, self.gfilled, self.gn, self.gstaged = 0, None, 0, 0
        self.stage_cores = taken
        words = max([t.staged for t in self.tracks] + [gs.bytes for gs in self.gstages.values()]) // 4
        spec = ttnn.ShardSpec(coalesce(sorted(taken)), (1, words), ttnn.ShardOrientation.ROW_MAJOR)
        self.stage = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, len(taken), words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, model.device,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec),
        )  # fmt: skip
        for t in self.tracks:
            t.bind(self.stage)
        for g in dict.fromkeys(self.gorder):
            gs = self.gstages[id(g)]
            gs.stage = self.stage
            g._combine = functools.partial(self._consume_gate, gs)  # instance attribute over the class method
        self.sdpa_kb = SDPA_KB if sdpa_kb is None else sdpa_kb
        AF.SDPA_HOST = self.sdpa_piece if (AF.LOCAL_GRID is not None and sdpa_extra_kernels()) else None
        self.queue = []  # [(proxy, part index)]: pieces still to attach, oldest first
        self.in_own = False
        self.seen = {"programs": 0, "too_wide": 0, "clash": 0, "attached": 0}
        ttnn.generic_op = self._generic_op

    def uninstall(self, model):
        AF.SDPA_HOST = None
        ttnn.generic_op = _orig_generic_op
        for t in self.tracks:
            t.restore()
        for g in dict.fromkeys(self.gorder):
            del g._combine
        ttnn.deallocate(self.stage)

    def consume(self, proxy, x, out):
        t = proxy.track
        assert self.order[t.ptr] == proxy.L, (t.name, "app order out of step", t.ptr, proxy.L)
        # pieces of this proxy still queued were not attached in time: dropped (the prefix stays shorter)
        self.queue = [q for q in self.queue if q[0] is not proxy]
        frac = t.filled if self.budget is not None else t.filled / self.parts
        staged = proxy.st.part_bytes(frac) if t.filled_for is proxy else 0
        self.in_own = True
        try:
            y = proxy.st(x, out, staged=staged)
        finally:
            self.in_own = False
        t.n += 1
        t.staged_sum += staged
        t.ptr = (t.ptr + 1) % len(self.order)
        nxt = t.proxies[self.order[t.ptr]]
        t.filled, t.filled_for = 0, nxt
        if self.budget is not None:
            self.queue.append((nxt, None))  # filled in host-sized pieces (t.filled = fraction done)
        else:
            self.queue += [(nxt, k) for k in range(self.parts)]
        return y

    def _consume_gate(self, gs, z, h, m):
        assert self.gorder[self.gptr] is gs.gate, ("gate order out of step", self.gptr)
        self.queue = [q for q in self.queue if q[0] is not gs]
        staged = self.gfilled is gs
        self.in_own = True
        try:
            y = gs.combine(z, h, m, staged)
        finally:
            self.in_own = False
        self.gn += 1
        self.gstaged += int(staged)
        self.gptr = (self.gptr + 1) % len(self.gorder)
        self.gfilled = None
        nxt = (self.gstages[id(self.gorder[self.gptr])], None if self.budget is not None else 0)
        if self.gates_tail:
            self.queue.append(nxt)  # only hosts the w_all pieces leave
        else:
            self.queue.insert(0, nxt)  # urgent: its consumer comes soon
        return y

    def _generic_op(self, io, prog):
        if self.in_own:
            return _orig_generic_op(io, prog)
        cores = program_cores(prog)
        self.seen["programs"] += 1
        if self.budget is not None:
            return self._budget_op(io, prog, cores)
        if len(cores) > self.max_cores:
            self.seen["too_wide"] += 1
            return _orig_generic_op(io, prog)
        if cores & self.stage_cores:
            self.seen["clash"] += 1
            return _orig_generic_op(io, prog)
        if any(_STREAMS in k.kernel_source for k in prog.kernels):  # DRAM-heavy itself: no room to hide a fill
            self.seen["streams"] = self.seen.get("streams", 0) + 1
            return _orig_generic_op(io, prog)
        if not self.queue:
            return _orig_generic_op(io, prog)
        proxy, k = self.queue.pop(0)
        if isinstance(proxy, GateStage):
            prog = proxy.attach(prog)  # asserts disjoint cores
            self.gfilled = proxy
            self.seen["attached"] += 1
            return _orig_generic_op(list(io[:-1]) + proxy.fill_io + [io[-1]], prog)
        prog = proxy.st.attach(prog, (k / self.parts, (k + 1) / self.parts))  # asserts disjoint cores
        proxy.track.filled = k + 1  # a proxy's pieces attach in order, so its filled prefix is k + 1 parts
        self.seen["attached"] += 1
        return _orig_generic_op(list(io[:-1]) + proxy.st.fill_io + [io[-1]], prog)

    def _budget_op(self, io, prog, cores):
        kb = self.host_kb.get(host_key(prog, len(cores)))
        if kb is None or not self.queue:
            self.seen["not_host" if kb is None else "idle_host"] = (
                self.seen.get("not_host" if kb is None else "idle_host", 0) + 1
            )
            return _orig_generic_op(io, prog)
        pieces = self._pieces(cores, kb)
        if pieces is None:
            self.seen["clash"] += 1
            return _orig_generic_op(io, prog)
        kernels, extra = pieces
        self.seen["attached"] += 1
        prog = ttnn.ProgramDescriptor(
            kernels=list(prog.kernels) + kernels, semaphores=list(prog.semaphores), cbs=list(prog.cbs)
        )
        return _orig_generic_op(list(io[:-1]) + extra + [io[-1]], prog)

    def _pieces(self, cores, kb, delay=0):
        """Fill kernels (and their tensors) for a host program on `cores`: up to budget x kb KB per staging core,
        oldest queued item first, then the next items on cores disjoint from the ones filled. None if none fit."""
        left, used, kernels, extra, take = kb * self.budget * 1024, set(cores), [], [], []
        for item in list(self.queue):
            obj = item[0]
            if left < 4096:
                break
            gate = isinstance(obj, GateStage)
            icores = obj.stage_cores if gate else obj.st.cores
            if icores & used:
                continue
            if gate:
                if obj.bytes > left:  # whole F + G in one piece
                    continue
                kernels += obj.fill_kernels()
                extra += obj.fill_io
                self.gfilled = obj
                left -= obj.bytes
                take.append(item)
            else:
                t = obj.track
                lo = t.filled if t.filled_for is obj else 0.0
                hi = min(1.0, lo + left / obj.st.staged)
                if obj.st.part_bytes(hi) <= obj.st.part_bytes(lo):
                    continue
                kernels += obj.st.fill_kernels((lo, hi), delay=delay)
                extra += obj.st.fill_io
                left -= (hi - lo) * obj.st.staged
                t.filled = hi
                if hi >= 1.0:
                    take.append(item)
            used |= icores
        if not kernels:
            return None
        self.queue = [q for q in self.queue if not any(q is x for x in take)]
        return kernels, extra

    def sdpa_piece(self, cores):
        """attention_fast.SDPA_HOST: fill kernels riding inside a local sdpa call on `cores` (its ~20 us of exact math
        after the KV reads leave DRAM idle), as a ttnn.ProgramDescriptor (the tensors they touch are persistent
        weights and the staging tensor)."""
        if self.sdpa_kb <= 0 or not self.queue:
            return None
        pieces = self._pieces(cores, self.sdpa_kb, delay=int(SDPA_DELAY_US * 1350))
        if pieces is None:
            return None
        self.seen["sdpa_hosted"] = self.seen.get("sdpa_hosted", 0) + 1
        return ttnn.ProgramDescriptor(kernels=pieces[0], semaphores=[], cbs=[])

    def stats(self):
        out = dict(self.seen)
        for t in self.tracks:
            out[t.name] = {"calls": t.n, "staged_kb_avg": round(t.staged_sum / max(1, t.n) / 1024, 1),
                           "staged_kb_max": t.staged // 1024, "cores": len(t.cores)}  # fmt: skip
        if self.gorder:
            out["gates"] = {"calls": self.gn, "staged": self.gstaged,
                            "stage_cores": len(set().union(*(g.stage_cores for g in self.gstages.values())))}  # fmt: skip
        return out


def install(model, **kw):
    return Overlap(model, **kw)
