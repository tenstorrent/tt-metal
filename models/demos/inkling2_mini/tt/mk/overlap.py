# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Cross-op overlap for decode (M2): while the DRAM-light programs of app i run (residual, message gates, attention
prep / out), the next app's w_all streams from DRAM into L1 (tt/mk/stage.py), so app i+1's w_all program reads
that prefix from L1.

    ov = install(model, parts=4, staged_kb=None)   # before capture; ov.stats() after a step

How: every layer's w_all BankLinear gets a StagedBankLinear over ONE shared staging tensor (all w_all linears use
the same cores; asserted). ``ttnn.generic_op`` is wrapped: after app i's w_all ran, the next ``parts`` eligible
programs each carry 1/parts of app i+1's fill. Eligible = a program with at most ``max_cores`` cores, none of them a
staging core (asserted per attach), and not a bank linear itself. App i+1's w_all then reads the filled prefix from
L1. Bytes into every CB slot are unchanged, so outputs are bit-equal; a partly filled prefix is just shorter.

The order of apps is model.APPS (one pointer, advanced by every w_all call): decode steps only.
"""

import ttnn
from models.demos.inkling2_mini.tt import model as MD
from models.demos.inkling2_mini.tt.mk.stage import StagedBankLinear, program_cores, range_bytes, resplit

_orig_generic_op = ttnn.generic_op


class _Proxy:
    """Stands in for a layer's BankLinear in _DramShardedLinear.bank."""

    def __init__(self, ov, L, st):
        self.ov, self.L, self.st = ov, L, st

    def __getattr__(self, name):
        return getattr(self.st.bl, name)

    def __call__(self, x, out=None):
        return self.ov.consume(self, x, out)


class Overlap:
    def __init__(self, model, parts=4, max_cores=36, staged_kb=None, split=3, exclude_rows=(0,)):
        self.parts, self.max_cores = parts, max_cores
        self.order = [L for L, _ in MD.APPS]
        self.ptr = 0
        self.pending = None  # (proxy of the next w_all, parts attached)
        self.in_own = False
        self.n_attached = self.n_consumed = self.staged_sum = 0
        self.seen = {"programs": 0, "too_wide": 0, "clash": 0}
        self.clash_cores = {}
        banks = {}
        for L, lay in enumerate(model.layers):
            assert lay.attn.w_all.bank is not None, "needs the bank-streamed w_all (INKLING_PROJ_BANK=1)"
            gx = model.device.compute_with_storage_grid_size().x
            excl = [
                (x, y) for y in exclude_rows for x in range(gx)
            ]  # the small programs' cores (core_list from (0, 0))
            banks[L] = resplit(lay.attn.w_all.bank, split, excl)
        top = max(range_bytes(b) for b in banks.values())
        staged = top if staged_kb is None else min(staged_kb << 10, top)
        big = max(banks, key=lambda L: range_bytes(banks[L]))
        first = StagedBankLinear(banks[big], staged)  # allocates the shared staging tensor
        self.proxies = {L: _Proxy(self, L, StagedBankLinear(b, staged, stage=first.stage)) for L, b in banks.items()}
        self.stage_cores = self.proxies[0].st.cores
        self.orig = {L: lay.attn.w_all.bank for L, lay in enumerate(model.layers)}
        for L, lay in enumerate(model.layers):
            lay.attn.w_all.bank = self.proxies[L]
        self.lin_cores = [set(p.st.cores) for p in self.proxies.values()]
        ttnn.generic_op = self._generic_op

    def uninstall(self, model):
        ttnn.generic_op = _orig_generic_op
        for L, lay in enumerate(model.layers):
            lay.attn.w_all.bank = self.orig[L]
        ttnn.deallocate(self.proxies[0].st.stage)

    def consume(self, proxy, x, out):
        assert self.order[self.ptr] == proxy.L, ("app order out of step", self.ptr, proxy.L)
        k = self.pending[1] if self.pending is not None and self.pending[0] is proxy else 0
        staged = proxy.st.part_bytes(k / self.parts)
        self.in_own = True
        try:
            y = proxy.st(x, out, staged=staged)
        finally:
            self.in_own = False
        self.n_consumed += 1
        self.staged_sum += staged
        self.ptr = (self.ptr + 1) % len(self.order)
        self.pending = (self.proxies[self.order[self.ptr]], 0)  # next app's w_all: fill from here on
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
            for c in cores & self.stage_cores:
                self.clash_cores[c] = self.clash_cores.get(c, 0) + 1
            return _orig_generic_op(io, prog)
        if self.pending is None or self.pending[1] >= self.parts:
            return _orig_generic_op(io, prog)
        proxy, k = self.pending
        prog = proxy.st.attach(prog, (k / self.parts, (k + 1) / self.parts))  # asserts disjoint cores
        self.pending = (proxy, k + 1)
        self.n_attached += 1
        return _orig_generic_op(list(io[:-1]) + proxy.st.fill_io + [io[-1]], prog)

    def stats(self):
        return {"consumed": self.n_consumed, "attached": self.n_attached,
                "staged_kb_avg": round(self.staged_sum / max(1, self.n_consumed) / 1024, 1),
                "staged_kb_max": self.proxies[0].st.staged // 1024, "stage_cores": len(self.stage_cores), **self.seen,
                "top_clash": sorted(self.clash_cores.items(), key=lambda kv: -kv[1])[:6],
                "stage_core_list": sorted(self.stage_cores)}  # fmt: skip


def install(model, **kw):
    return Overlap(model, **kw)
