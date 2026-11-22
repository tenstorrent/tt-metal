# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Cross-op overlap for decode (M2): while the DRAM-light programs between two uses of a projection run (residual,
message gates, attention prep / out, window), the next app's weights of that projection stream from DRAM into L1
(tt/mk/stage.py), so the next app's projection program reads them from L1.

    ov = install(model, weights=("w_all",), parts=4)   # before capture; ov.stats() after steps; ov.uninstall(model)

How: for each staged projection (attention w_all and / or w_o), every layer's BankLinear is re-split over more cores
(same weight tensor, placed off the hosting programs' rows) and gets a staged reader over ONE staging tensor shared
by the 24 layers. ``ttnn.generic_op`` is wrapped: after app i's projection ran, app i+1's fill for it is queued in
``parts`` pieces; each later eligible program carries the oldest queued piece. Eligible = at most ``max_cores``
cores, none of them a staging core (asserted again per attach), no weight-streaming kernel (bank / MoE streams). App i+1's projection
reads whatever prefix was filled. Bytes into every CB slot are unchanged, so outputs are bit-equal.

App order: model.APPS (one pointer per projection, advanced by every call): decode steps only.
"""

import ttnn
from models.demos.inkling2_mini.tt import model as MD
from models.demos.inkling2_mini.tt.mk.stage import StagedBankLinear, coalesce, program_cores, range_bytes, resplit

_orig_generic_op = ttnn.generic_op
_STREAMS = "bank_"  # bank_linear_*, moe_bank_*: weight streams


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
        self, model, weights=("w_all",), parts=4, max_cores=1, staged_kb=None, split=3, exclude_rows=(0, 1, 2, 3, 4, 5)
    ):
        """max_cores: hosts have at most this many cores (1 = the single-core residual programs, ~22 us each with DRAM
        idle; wider small programs are short, so a fill piece makes them longer instead of hiding)."""
        self.parts, self.max_cores = parts, max_cores
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
        self.stage_cores = taken
        words = max(t.staged for t in self.tracks) // 4
        spec = ttnn.ShardSpec(coalesce(sorted(taken)), (1, words), ttnn.ShardOrientation.ROW_MAJOR)
        self.stage = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, len(taken), words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, model.device,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec),
        )  # fmt: skip
        for t in self.tracks:
            t.bind(self.stage)
        self.queue = []  # [(proxy, part index)]: pieces still to attach, oldest first
        self.in_own = False
        self.seen = {"programs": 0, "too_wide": 0, "clash": 0, "attached": 0}
        ttnn.generic_op = self._generic_op

    def uninstall(self, model):
        ttnn.generic_op = _orig_generic_op
        for t in self.tracks:
            t.restore()
        ttnn.deallocate(self.stage)

    def consume(self, proxy, x, out):
        t = proxy.track
        assert self.order[t.ptr] == proxy.L, (t.name, "app order out of step", t.ptr, proxy.L)
        # pieces of this proxy still queued were not attached in time: dropped (the prefix stays shorter)
        self.queue = [q for q in self.queue if q[0] is not proxy]
        staged = proxy.st.part_bytes(t.filled / self.parts) if t.filled_for is proxy else 0
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
        self.queue += [(nxt, k) for k in range(self.parts)]
        return y

    def _generic_op(self, io, prog):
        if self.in_own:
            return _orig_generic_op(io, prog)
        cores = program_cores(prog)
        self.seen["programs"] += 1
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
        prog = proxy.st.attach(prog, (k / self.parts, (k + 1) / self.parts))  # asserts disjoint cores
        proxy.track.filled = k + 1  # a proxy's pieces attach in order, so its filled prefix is k + 1 parts
        self.seen["attached"] += 1
        return _orig_generic_op(list(io[:-1]) + proxy.st.fill_io + [io[-1]], prog)

    def stats(self):
        out = dict(self.seen)
        for t in self.tracks:
            out[t.name] = {"calls": t.n, "staged_kb_avg": round(t.staged_sum / max(1, t.n) / 1024, 1),
                           "staged_kb_max": t.staged // 1024, "cores": len(t.cores)}  # fmt: skip
        return out


def install(model, **kw):
    return Overlap(model, **kw)
