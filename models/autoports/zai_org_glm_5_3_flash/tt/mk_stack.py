# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14b (PRODUCTION_PATH.md M5): N decoder layers in ONE persistent program per chip.

The single-layer megakernel (7a's MkLayer, tt/mk_layer.py) is built once per layer in STACK mode (the residual stays in
the stacked [t,1,4,4096] fp32 L1 layout: P1 reads it, P6 takes it as the attention residual, P10 overwrites it with the
layer output) with ONE shared set of intermediates (every ttnn.empty of layer 0's build is replayed for the others),
its ProgramDescriptor captured instead of launched. The N descriptors must be identical up to runtime args (checked);
the differing args (weights, KDA state / conv history, mHC / KDA parameter addresses) go into a DRAM argument table,
one row per (kernel, core, layer) = that kernel's complete runtime-arg vector for the layer. The launched program is
layer 0's descriptor with MK_LAYERS = N on every kernel (kernels/mk/stack.hpp): every RISC runs its phase sequence N
times; a data-movement kernel with per-layer args points its runtime-arg base at the layer's table row (one 64-B
aligned NoC read per layer into the MK_ARG_CB scratch); compute kernels' args are layer-invariant (checked).

Layer boundary: the owner NCRISC raises its own MK_X_SEM after P10's residual writes landed, the owner BRISC waits for
it before layer l + 1's P1 reads the same columns (the P6 -> P7 s1 hand-off, one layer later); every RISC rewinds its
CBs to their bases (all empty there). The readers' weight streamer never waits for activations: after layer l's down
blocks it streams layer l + 1's qkv blocks into the ring while layer l's swiglu / down / mhc_post and layer l + 1's
mhc_pre run (the rings carry over across layers).
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import cb_guard
from models.autoports.zai_org_glm_5_3_flash.tt.rt_guard import checked

ARG_CB = 30  # scratch CB of the per-layer argument rows (no phase uses index 30)
OWNER_DM = ("owner_dm0.cpp", "owner_dm1.cpp")
# standalone kernels that run as ONE phase per launch -> their layer-loop wrappers (kernels/mk/rtail_*.cpp)
LOOP_WRAPPERS = {"router_tail/reader.cpp": "mk/rtail_r.cpp", "router_tail/compute.cpp": "mk/rtail_c.cpp",
                 "router_tail/writer.cpp": "mk/rtail_w.cpp"}  # fmt: skip
STACKABLE = {
    "owner_dm0.cpp",
    "owner_dm1.cpp",
    "owner_c.cpp",
    "root_dm0.cpp",
    "root_dm1.cpp",
    "root_c.cpp",
    "reader_w.cpp",
    "reader_a.cpp",
    "reader_c.cpp",
    "reader_moe_w.cpp",
    "reader_moe_a.cpp",
    "reader_moe_c.cpp",
    "rtail_r.cpp",
    "rtail_c.cpp",
    "rtail_w.cpp",
}  # kernels with the layer loop (MK_LAYERS)


class _Capture:
    """Intercept ttnn.generic_op while a megakernel class builds its program: -> calls [(io, ProgramDescriptor)]."""

    def __enter__(self):
        self.orig, self.calls = ttnn.generic_op, []
        ttnn.generic_op = lambda io, desc: self.calls.append((list(io), desc)) or io[-1]
        return self

    def __exit__(self, *exc):
        ttnn.generic_op = self.orig
        return False


class AllocPool:
    """ttnn.empty during the layer builds of one or more stacks: request i of a given (shape, dtype, layout, memory
    config) in ANY build returns the same tensor (allocated on first use) -> every layer of a stack gets the same
    intermediates (its descriptors then differ in weights / state args only), and the stacks of a step (layer kinds, row
    chunks; launched one after another, never concurrently) share one L1 set. Persistent: kept by the stacks."""

    def __init__(self):
        self.orig, self.lists, self.counts, self._child = ttnn.empty, {}, None, None
        # alias (StackBackend native-t width buckets): a request is keyed by its TILE-PADDED shape and served by the
        # tensor the full-width (t = 32) build allocated, as a view with the requested logical rows
        # (ttnn.reshape(t, logical, padded): same buffer) -> every width's stacks run on ONE L1 set of intermediates
        self.alias = False
        self.views = {}
        self.used = set()  # the pool tensors one layer build has been handed (all live at once in its program)
        self.prefixed = set()  # ... of them, the bases handed as a prefix view (a later key match on one = a collision)
        self.real = {}  # per key: the real tensors this build has taken (a prefix-served request takes none)
        self.prefix_slots, self.prefix_base = {}, {}  # (key, request index, shape) -> its prefix view's key -> base

    def child(self) -> "AllocPool":
        """a separate pool (the layers' persistent hand-off buffers: never aliased with a build's intermediates)"""
        if self._child is None:
            self._child = AllocPool()
        self._child.alias = self.alias
        return self._child

    @staticmethod
    def _padded(shape, a, kw) -> tuple:
        layout = a[1] if len(a) > 1 else kw.get("layout")
        shp = tuple(int(v) for v in shape)
        if layout != ttnn.TILE_LAYOUT or len(shp) < 2:
            return shp
        r = lambda v: -(-v // 32) * 32  # noqa: E731
        return shp[:-2] + (r(shp[-2]), r(shp[-1]))

    @staticmethod
    def _key(shape, a, kw):
        return (tuple(int(v) for v in shape),) + tuple(str(v) for v in a[:2]) + tuple(
            str(kw.get(k)) for k in ("dtype", "layout") if k in kw) + (str(a[3]) if len(a) > 3 else str(kw.get(
                "memory_config")),)  # fmt: skip

    def __call__(self, shape, *a, **kw):
        if self.alias:
            return self._aliased(shape, a, kw)
        k = self._key(shape, a, kw)
        i = self.counts.get(k, 0)
        self.counts[k] = i + 1
        lst = self.lists.setdefault(k, [])
        if i == len(lst):
            lst.append(self.orig(shape, *a, **kw))
        self._take(lst[i])
        return lst[i]

    def _take(self, base):
        if id(base) in self.prefixed:  # never two live intermediates on one buffer: loud, not silent
            raise RuntimeError(f"AllocPool: a prefix-aliased tensor {tuple(base.shape)} requested again by its own key")
        self.used.add(id(base))

    def _aliased(self, shape, a, kw):
        want = tuple(int(v) for v in shape)
        pad = self._padded(shape, a, kw)
        k = self._key(pad, a, kw)  # the full-width build's key (its shapes are their own padded shapes)
        i = self.counts.get(k, 0)
        self.counts[k] = i + 1
        pk = (k, i, want)
        if pk in self.prefix_slots:  # this request was served by a prefix view in an earlier build: the same one
            return self._take_prefix(self.prefix_slots[pk])
        lst = self.lists.setdefault(k, [])
        j = self.real.get(k, 0)  # the request's index among this key's REAL tensors (prefix-served ones skip it)
        if j == len(lst):  # nothing of this padded shape left to alias
            vk = self._prefix_alias(want, pad, k)
            if vk is not None:
                self.prefix_slots[pk] = vk
                return self._take_prefix(vk)
            lst.append(self.orig(pad if pad != want else shape, *a, **kw))  # a tensor of its own
        self.real[k] = j + 1
        base = lst[j]
        self._take(base)
        if tuple(int(v) for v in base.shape) == want:
            return base
        vk = (k, j, want)
        if vk not in self.views:
            self.views[vk] = ttnn.reshape(base, ttnn.Shape(list(want)), ttnn.Shape(list(base.padded_shape)))
        return self.views[vk]

    def _take_prefix(self, vk):
        base = self.prefix_base[vk]
        if id(base) in self.used:  # never two live intermediates on one buffer: loud, not silent
            raise RuntimeError(f"AllocPool: prefix base {tuple(base.shape)} already handed in this build")
        self.used.add(id(base))
        self.prefixed.add(id(base))
        return self.views[vk]

    def _prefix_alias(self, want, pad, k):
        """a request no full-width tensor of its padded shape serves (a narrow width's stacked-mode shapes), served as a
        prefix view of another pool tensor this build has not been handed: same dtype / layout / memory config, ONE tile
        row and unit leading dims on both (then the request's tiles are the first tiles of the base, in order), at least
        as many columns. Stage 14b: the t = 1 stack's packed mhc_post (1, 1, 32, 4096) fp32 over the unused flat
        (1, 1, 32, 16384) -> no L1 allocation below the full width's compiled programs (whose static CBs it would overlap).
        Returns the view's key (views / prefix_base); None: no candidate or ttnn refuses the view -> the caller allocates.
        """
        if len(pad) < 2 or any(v != 1 for v in pad[:-2]) or pad[-2] != 32:
            return None
        best = None
        for bk, blst in self.lists.items():
            if bk[1:] != k[1:]:
                continue
            for base in blst:
                bp = tuple(int(v) for v in base.padded_shape)
                if id(base) in self.used or len(bp) != len(pad) or any(v != 1 for v in bp[:-2]) or bp[-2] != 32:
                    continue
                if bp[-1] >= pad[-1] and (best is None or bp[-1] < best[1]):
                    best = (base, bp[-1])
        if best is None:
            return None
        base = best[0]
        vk = ("prefix", id(base), want)
        if vk not in self.views:
            try:
                self.views[vk] = ttnn.reshape(base, ttnn.Shape(list(want)), ttnn.Shape(list(base.padded_shape)))
            except Exception:  # noqa: BLE001 -- ttnn refuses a wider padded last dim: allocate instead
                return None
            self.prefix_base[vk] = base
        return vk

    def tensors(self) -> list:
        return [t for lst in self.lists.values() for t in lst]

    def __enter__(self):  # one layer build
        self.counts, self.real = {}, {}
        self.used, self.prefixed = set(), set()
        ttnn.empty = self
        return self

    def __exit__(self, *exc):
        ttnn.empty = self.orig
        return False


def _crs_cores(crs) -> set:
    return {(x, y) for r in crs.ranges() for x in range(int(r.start.x), int(r.end.x) + 1)
            for y in range(int(r.start.y), int(r.end.y) + 1)}  # fmt: skip


# MK_X_SEM candidates: ids no owner-signalling phase uses (12 = MkLayer's park / the experts' MW1 on the READERS, 14 /
# 15 = reader-only expert / tail ids, 9 / 7 / 4 = reader hand-offs); the first not declared on the owners is taken
X_SEM_CANDIDATES = (12, 14, 15, 9, 7, 4)


def _cores(kd) -> list:
    out = []
    for r in kd.core_ranges.ranges():
        for x in range(int(r.start.x), int(r.end.x) + 1):
            for y in range(int(r.start.y), int(r.end.y) + 1):
                out.append((x, y))
    return sorted(out)


def _src(kd) -> str:
    return os.path.basename(str(kd.kernel_source).replace("\\", "/"))


def _relsrc(kd) -> str:
    """the kernel source relative to the kernels dir (router_tail/reader.cpp, mk/owner_dm0.cpp, ...)"""
    p = str(kd.kernel_source).replace("\\", "/").split("/tt/kernels/")
    return p[-1]


def _is_compute(kd) -> bool:
    return type(kd.config).__name__ == "ComputeConfigDescriptor"


def _kkey(kd):
    return (str(kd.kernel_source), str(kd.core_ranges), tuple(int(a) for a in kd.compile_time_args),
            tuple(sorted((str(a), str(b)) for a, b in kd.defines)), type(kd.config).__name__)  # fmt: skip


def _cbkey(cb):
    return (int(cb.total_size), str(cb.core_ranges), int(cb.address_offset), bool(cb.has_buffer()),
            tuple((int(f.buffer_index), int(f.page_size)) for f in cb.format_descriptors))  # fmt: skip


def _semkey(s):
    return (int(s.id), str(s.core_ranges), int(s.initial_value))


class MkStack:
    """N MkLayer instances (each with its own weights / state; ONE kind, same t) as ONE program per chip."""

    def __init__(self, device, layers: list, *, t: int, probe: bool = False, rev: str | None = None, res=None,
                 pool: AllocPool | None = None, live=None):  # fmt: skip
        """res: an existing stacked residual tensor (several stacks of different layer kinds run back to back on one
        residual: one launch each, one trace per step; or a DRAM residual per row chunk); default: the stack allocates
        its own (L1). pool: an AllocPool shared with the other stacks of the step (kinds, row chunks: one L1 set of
        intermediates for all of them). live: the step's position page (int32 [1, 32]) handed to every layer's routed
        experts (MkSparseLayer: idle rows route to no expert, ReaderExperts MKX_LIVE); None = unchanged programs."""
        assert layers, "MkStack: no layers"
        self.device, self.layers, self.t, self.n = device, list(layers), int(t), len(layers)
        self.rev = rev
        L0 = self.layers[0]
        for L in self.layers[1:]:  # one set of persistent hand-off buffers (on, m) for every layer of a kind
            if type(L) is type(L0):
                L.persist = L0.persist
        self.pool = pool if pool is not None else AllocPool()
        for (
            L
        ) in self.layers:  # allocated before the builds, from the pool's child (shared across kinds when shapes match)
            with self.pool.child():
                L.persistent_buffers(self.t)
        self.probe = None
        if probe:  # one probe tensor: layer l's marks in rows [256 l, 256 (l + 1)) (kernels/mk/probe.hpp)
            self.probe = ttnn.from_torch(torch.zeros(L0.PROBE_ROWS * self.n, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        for L in self.layers:
            L.probe = self.probe
        self.res = res
        self.live = live
        self.program = None  # (io, ProgramDescriptor), built on the first call (eager), reused by every later launch
        self.info = {}

    def residual(self):
        """the persistent residual (fp32, L1): the stack's input (written before a launch) and output — t <= 4 the
        stacked [t,1,4,4096] layout (the packed mhc_post), t > 4 flat [1,1,t,16384] (7a's native-t layer programs)"""
        if self.res is None:
            # t > 4: DRAM (the owners' per-tile mhc_post CB region leaves ~56 KB / bank of L1 tensors at t = 8)
            shape = (self.t, 1, 4, 4096) if self.t <= 4 else (1, 1, self.t, 16384)
            mem = ttnn.L1_MEMORY_CONFIG if self.t <= 4 else ttnn.DRAM_MEMORY_CONFIG
            self.res = ttnn.empty(shape, ttnn.float32, ttnn.TILE_LAYOUT, self.device, mem)
        return self.res

    # ------------------------------------------------------------------ build
    def _capture_layers(self, hists, active, states):
        res = self.residual()
        caps = []
        for i, L in enumerate(self.layers):
            kw = {"live": self.live} if self.live is not None else {}
            with self.pool, _Capture() as cap:
                L(res, hists[i], active, t=self.t, state=states[i], stack=True, **kw)
            assert len(cap.calls) == 1, len(cap.calls)
            io_i, d_i = cap.calls[0]
            caps.append((io_i, self.layer_hook(d_i)))
        self.shared = self.pool.tensors()
        return caps

    def build(self, hists, active, states):
        caps = self._capture_layers(hists, active, states)
        io0, d0 = caps[0]
        k0 = list(d0.kernels)
        ks = [list(d.kernels) for _, d in caps]
        for i, (_, d) in enumerate(caps[1:], 1):  # identical programs up to runtime args
            if [_kkey(k) for k in ks[i]] != [_kkey(k) for k in k0]:
                diff = [(j, [n for n, (x, y) in enumerate(zip(_kkey(a), _kkey(b))) if x != y], _kkey(a)[0].split("/")[-1],
                         [(n_, x_, y_) for n_, (x_, y_) in enumerate(zip(_kkey(a)[2], _kkey(b)[2])) if x_ != y_][:6],
                         sorted(set(_kkey(a)[3]) ^ set(_kkey(b)[3]))[:6])
                        for j, (a, b) in enumerate(zip(k0, ks[i])) if _kkey(a) != _kkey(b)]  # fmt: skip
                _a, _b = k0[diff[0][0]], ks[i][diff[0][0]]
                _defs = sorted((str(x), str(y)) for x, y in _a.defines if "_CT" in str(x))
                print(
                    "MKSTACK-DIFF defs",
                    _defs,
                    "ct0[40:70]",
                    list(_kkey(_a)[2])[40:70],
                    "ct1[40:70]",
                    list(_kkey(_b)[2])[40:70],
                    flush=True,
                )
                raise RuntimeError(
                    f"MkStack t={self.t}: layer {i}'s kernels differ from layer 0's beyond runtime args "
                    f"(kernels {len(k0)} vs {len(ks[i])}; [(kernel, fields, source, ct diffs, define diffs)] "
                    f"{diff[:4]})"
                )
            if [_cbkey(c) for c in d.cbs] != [_cbkey(c) for c in d0.cbs]:
                raise RuntimeError(f"MkStack: layer {i}'s CBs differ from layer 0's")
            if [_semkey(s) for s in d.semaphores] != [_semkey(s) for s in d0.semaphores]:
                raise RuntimeError(f"MkStack: layer {i}'s semaphores differ from layer 0's")
        if any(c.has_buffer() for c in d0.cbs):
            raise RuntimeError("MkStack: a CB bound to a buffer (its address would be per layer)")
        # runtime args per kernel / core / layer
        rts = [[{xy: [int(v) for v in ks[i][k].runtime_args[xy[0]][xy[1]]] for xy in _cores(k0[k])}
                for i in range(self.n)] for k in range(len(k0))]  # fmt: skip
        varying = [k for k in range(len(k0)) if any(rts[k][i] != rts[k][0] for i in range(1, self.n))]
        for k in varying:
            if _is_compute(k0[k]):
                raise RuntimeError(f"MkStack: compute kernel {_src(k0[k])} has per-layer runtime args")
        # argument-row slots in MK_ARG_CB: two varying kernels on one core (its BRISC and NCRISC) get different slots
        slot, owner_of = {}, {}
        for k in varying:
            cores = set(rts[k][0])
            taken = {slot[j] for j in owner_of if owner_of[j] & cores}
            slot[k] = min(s_ for s_ in range(4) if s_ not in taken)
            owner_of[k] = cores
        n_slots = max(slot.values(), default=0) + 1
        n_rt = {}
        for k in varying:  # MK_LARGS_RT is one index per kernel: shorter per-core vectors are zero-padded to it
            for xy in rts[k][0]:
                if len({len(rts[k][i][xy]) for i in range(self.n)}) != 1:
                    raise RuntimeError(f"MkStack: {_src(k0[k])} core {xy}: the arg count differs between layers")
            n_rt[k] = max(len(v) for v in rts[k][0].values())
        page_words = -(-(max(n_rt.values(), default=0) + 3) // 16) * 16
        page = 4 * page_words
        # argument table: kernel k (varying), core c (sorted), layer i -> row base_k + i * n_cores_k + c
        bases, rows = {}, 0
        for k in varying:
            bases[k] = rows
            rows += self.n * len(rts[k][0])
        rows = max(rows, 1)
        self.table = ttnn.from_torch(torch.zeros(rows, max(page_words, 16), dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        tbl_addr = self.table.buffer_address()
        host = torch.zeros(rows, max(page_words, 16), dtype=torch.int64)
        for k in varying:
            cores = sorted(rts[k][0])
            for c, xy in enumerate(cores):
                extra = [tbl_addr, bases[k] + c, len(cores)]
                for i in range(self.n):
                    v = rts[k][i][xy] + [0] * (n_rt[k] - len(rts[k][i][xy])) + extra
                    host[bases[k] + i * len(cores) + c, : len(v)] = torch.tensor(v, dtype=torch.int64)
        host = torch.where(host >= 2**31, host - 2**32, host).to(torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                        self.table)  # fmt: skip
        # owner semaphore: the residual hand-off P10 (NCRISC) -> next layer's P1 (BRISC); semaphore ids are per core
        # (the same id serves disjoint core sets), so the id only has to be free on the owners
        owner_k = next(kd for kd in k0 if _src(kd) == "owner_dm0.cpp")
        own = _crs_cores(owner_k.core_ranges)
        on_owners = {int(s_.id) for s_ in d0.semaphores if _crs_cores(s_.core_ranges) & own}
        on_owners |= set(self.reserved_sems())
        sem_x = next(i for i in X_SEM_CANDIDATES if i not in on_owners)
        owner_cores = None
        kernels = []
        for k, kd in enumerate(k0):
            rel = _relsrc(kd)
            if rel in LOOP_WRAPPERS:  # a standalone phase kernel -> its layer-loop wrapper (same args)
                kd.kernel_source = str(kd.kernel_source).replace(rel, LOOP_WRAPPERS[rel])
            if _src(kd) not in STACKABLE:
                raise RuntimeError(f"MkStack: kernel {_src(kd)} has no layer loop")
            defs = list(kd.defines) + [("MK_LAYERS", str(self.n))] + ([("S14B_REV", str(self.rev))] if self.rev else [])
            if _src(kd) in OWNER_DM:
                defs.append(("MK_X_SEM", str(sem_x)))
            # Stage 14 (GLM53_MK_PRE_PREFETCH=1, set by the layer builds; GLM53_STACK_P1_PREFETCH=1 = the stack alone): P1 / P7
            # issue their residual-independent reads (fn^T, norm weight, x zero fill) before the residual hand-off
            # (layer l + 1's P1: layer l's P10 writes; P7: P6's s1) (kernels/mk/owner_dm0.cpp)
            if (_src(kd) == "owner_dm0.cpp" and os.environ.get("GLM53_STACK_P1_PREFETCH") == "1"
                    and ("MK_PRE_PREFETCH", "1") not in defs):  # fmt: skip
                defs.append(("MK_PRE_PREFETCH", "1"))
            if _src(kd) == "owner_dm0.cpp":  # every owner (the NCRISC / compute kernels split leaders / members)
                owner_cores = kd.core_ranges
            if k in varying:
                defs += [("MK_LARGS_RT", str(n_rt[k])), ("MK_LARGS_OFF", str(slot[k] * page)),
                         ("MK_LARGS_PAGE", str(page)), ("MK_ARG_CB", str(ARG_CB))]  # fmt: skip
                cores = sorted(rts[k][0])
                for c, (x, y) in enumerate(cores):
                    pad = [0] * (n_rt[k] - len(rts[k][0][(x, y)]))
                    kd.runtime_args[x][y].extend(pad + [tbl_addr, bases[k] + c, len(cores)])
            kd.defines = defs
            kernels.append(kd)
        assert owner_cores is not None
        cbs = list(d0.cbs)
        if varying:
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges

            arg_cores = merged_ranges({xy for k in varying for xy in _cores(k0[k])})
            # FIRST in the list (at the CB base of its cores): tt-metal places CBs per unique core range, and this CB's
            # rectangles span readers + owners + root -> appended last it could land below another CB's end on some
            # cores (2026-10-20 wm7 / pw4: inside the readers' 3rd weight ring at t=8 -> argument rows overwritten)
            cbs.insert(0, ttnn.CBDescriptor(total_size=n_slots * page + 64, core_ranges=arg_cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=ARG_CB, data_format=ttnn.uint32, page_size=n_slots * page + 64)]))  # fmt: skip
        cb_guard.check(cbs, f"MkStack ({self.n} layers, t={self.t})")
        sems = list(d0.semaphores) + [ttnn.SemaphoreDescriptor(id=sem_x, core_ranges=owner_cores, initial_value=0)]
        io, seen = [], set()
        for io_i, _ in caps:
            for t_ in io_i:
                if id(t_) not in seen:
                    seen.add(id(t_))
                    io.append(t_)
        io.append(self.table)
        self.info = dict(layers=self.n, t=self.t, varying_kernels=[_src(k0[k]) for k in varying],
                         per_layer_args={_src(k0[k]): n_rt[k] for k in varying}, table_rows=rows, page_bytes=page,
                         sem_x=sem_x, shared_intermediates=len(self.shared),
                         kernels=[_src(k) for k in k0])  # fmt: skip
        self.program = self.finalize(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return self.info

    # ------------------------------------------------------------------ hooks (tt/tp_stack.py: the TP / fabric stack)
    def layer_hook(self, desc):
        """every layer's captured ProgramDescriptor before the stack is composed (identical changes per layer)"""
        return desc

    def reserved_sems(self) -> list:
        """semaphore ids the finalized program adds (MK_X_SEM must avoid them)"""
        return []

    def finalize(self, io, desc):
        """the composed single-chip ProgramDescriptor -> (io, the launched program)"""
        return io, desc

    def __call__(self, hists, active, states):
        """one launch: every layer, in order, on the residual (self.residual(): input before, output after)"""
        if self.program is None:
            self.build(hists, active, states)
        io, desc = self.program
        ttnn.generic_op(io, checked(desc) if isinstance(desc, ttnn.ProgramDescriptor) else desc)
        return self.res

    def probe_marks(self, layer: int):
        """layer `layer`'s decoded per-phase marks (MkLayer.decode_probe on its row block)"""
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import decode_probe

        L0 = self.layers[0]
        p = ttnn.to_torch(self.probe).reshape(self.n, L0.PROBE_ROWS, 16)
        return p, decode_probe(p[layer], L0.readers.assign, k_cores=len(L0.place["k"]))
