# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Where a routed-expert slot's bytes sit in the flat expert's bank layout, and the run tables of
ttnn.bringup.moe_block_main.expert_slot_copy (MOE_REPLICAS=runtime, P.47).

The flat plan lays a chip's E slots out per core region (flat_routed_expert_weights.flat_weight_regions +
bank_host_tensor; the plan's region_bytes / wd_rot_region in flat_routed_expert_plan.cpp):
- w_gu: region r (n_rd of them) in bank r % banks at (r // banks) x E x gu tiles; slot s's gu tiles of it at s x gu.
  One contiguous run a region (n_rd runs a slot).
- w_d: down core d's blocks k = s nb + c (nb = It / kd K-blocks a slot, kd x pcds[d] tiles a block) in bank
  (d + k) % banks at d x sub + (k // banks) blocks (sub: the widest core's tiles a bank, wd_rot_region). A slot's blocks
  c = c0, c0 + banks, ... sit back to back in one bank: one run per (d, c0), nd x banks runs a slot; which bank a run
  is in depends on s nb (the bank rotation differs between slots, e.g. an odd slot and the 13th).
Units are listed in a canonical order (gu by region, then d by (core, c0)), the same for every slot of every plan with
the same per-slot facts, so slot s of one layout maps unit for unit onto slot t of another.

A run: (weight 0 gu / 1 d, bank, byte offset in the bank's shard). Tables: one page of the runs tensor each, word 0 the
runs n, then n x (weight, src bank, src offset, dst bank, dst offset, bytes).
"""

from __future__ import annotations

import torch

import ttnn

RUN_WORDS = 6
_FACT_KEYS = ("banks", "col0s", "n_rd", "n_rd_sg", "nd", "nk_gu", "np", "nu", "pcds", "rg")  # (per-slot layout facts)


def _fw():
    from .moe_block_main_adapter import _flat_weights_module

    return _flat_weights_module()


def tile_bytes(dtype) -> int:
    return {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[dtype]


def _kd_of(pcd: int, it: int) -> int:
    return max(k for k in (8, 4, 2, 1) if k * pcd <= 16 and it % k == 0)


class SlotLayout:
    """A plan's slot geometry: E slots a chip, tile bytes tb, for hidden H / intermediate I (a chip's)."""

    def __init__(self, lay: dict, E: int, H: int, I: int, tb: int):
        self.lay, self.E, self.H, self.I, self.tb = lay, int(E), int(H), int(I), int(tb)
        self.banks = int(lay["banks"])
        shapes1 = _fw().flat_weight_shapes(H, I, 1, lay)
        shapes = _fw().flat_weight_shapes(H, I, E, lay)
        self.gu = int(shapes1["w_gu"][1])  # gate/up tiles a slot a region
        assert int(shapes["w_gu"][1]) == E * self.gu, "w_gu regions are not E slots of equal size"
        self.n_rd = int(shapes["w_gu"][0])
        assert self.n_rd == int(lay["n_rd"])
        self.it = I // 32
        self.nd = int(lay["nd"])
        self.sub = int(shapes["w_d"][1])  # tiles a down region (a core's share of a bank)
        assert int(shapes["w_d"][0]) == self.nd * self.banks
        sub = max(
            -(-E * (self.it // _kd_of(p, self.it)) // self.banks) * _kd_of(p, self.it) * p
            for p in lay["pcds"][: self.nd]
        )
        assert sub == self.sub, f"down region {self.sub} tiles, the plan's rule says {sub}"
        assert int(lay.get("wd_rot_tiles", sub)) == sub, "the plan's wd_rot_tiles differs from the layout's"

    def facts(self) -> tuple:
        return tuple((k, repr(self.lay.get(k))) for k in _FACT_KEYS) + (("tb", self.tb), ("H", self.H), ("I", self.I))

    def units(self, s: int) -> list:
        """Slot s's runs in canonical order: [(weight, bank, byte offset, bytes)]."""
        assert 0 <= s < self.E
        tb, banks = self.tb, self.banks
        out = []
        for r in range(self.n_rd):
            out.append((0, r % banks, ((r // banks) * self.E * self.gu + s * self.gu) * tb, self.gu * tb))
        for d in range(self.nd):
            p = int(self.lay["pcds"][d])
            kd = _kd_of(p, self.it)
            nb, blk = self.it // kd, kd * p * tb
            for c0 in range(banks):
                cnt = len(range(c0, nb, banks))
                if cnt == 0:
                    continue
                k0 = s * nb + c0
                out.append((1, (d + k0) % banks, d * self.sub * tb + (k0 // banks) * blk, cnt * blk))
        return out

    def slot_bytes(self) -> int:
        return sum(u[3] for u in self.units(0))


def copy_runs(src: SlotLayout, s: int, dst: SlotLayout, t: int) -> list:
    """Slot s of src onto slot t of dst, unit for unit: [(w, src bank, src off, dst bank, dst off, bytes)]."""
    assert src.facts() == dst.facts(), f"the two plans lay a slot out differently: {src.facts()} vs {dst.facts()}"
    a, b = src.units(s), dst.units(t)
    assert [(u[0], u[3]) for u in a] == [(u[0], u[3]) for u in b], "slot units differ"
    return [(wa, ba, oa, bb, ob, n) for (wa, ba, oa, n), (_, bb, ob, _) in zip(a, b)]


def merge(runs: list) -> list:
    """Runs merged where both sides continue in the same banks (a regroup's consecutive slots)."""
    out = []
    for r in sorted(runs, key=lambda r: (r[0], r[1], r[2])):
        if out:
            w, sb, so, db, do, n = out[-1]
            if r[0] == w and r[1] == sb and r[3] == db and r[2] == so + n and r[4] == do + n:
                out[-1] = (w, sb, so, db, do, n + r[5])
                continue
        out.append(r)
    return out


def regroup_runs(src: SlotLayout, dst: SlotLayout, slots: int) -> list:
    """Slots 0 .. slots - 1 of src onto the same slots of dst (a load-time re-layout, every chip locally)."""
    return merge([r for s in range(slots) for r in copy_runs(src, s, dst, s)])


def check_inside(runs: list, lay: SlotLayout, slot: int, side: str = "dst") -> None:
    """Every run's destination (side "dst") or source ("src") bytes lie inside slot `slot` of lay (asserted)."""
    have: dict = {}
    for w, b, o, n in lay.units(slot):
        have.setdefault((w, b), []).append((o, o + n))
    for r in runs:
        w, b, o, n = (r[0], r[3], r[4], r[5]) if side == "dst" else (r[0], r[1], r[2], r[5])
        assert any(
            lo <= o and o + n <= hi for lo, hi in have.get((w, b), [])
        ), f"run {r}: {side} bytes outside slot {slot}"


def tables_tensor(tables: list, mesh) -> ttnn.Tensor:
    """Run tables -> the op's runs tensor: uint32 [1, 1, len(tables), W] ROW_MAJOR DRAM interleaved, replicated (W: the
    longest table's words, rounded up to 16)."""
    width = max(1 + RUN_WORDS * len(t) for t in tables)
    width = -(-width // 16) * 16
    host = torch.zeros(len(tables), width, dtype=torch.int64)
    for i, t in enumerate(tables):
        host[i, 0] = len(t)
        if t:
            host[i, 1 : 1 + RUN_WORDS * len(t)] = torch.tensor(t, dtype=torch.int64).reshape(-1)
    assert int(host.max()) < 2**32 and int(host.min()) >= 0
    return ttnn.from_torch(
        host.to(torch.int32).reshape(1, 1, len(tables), width),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def read_tables(runs: ttnn.Tensor) -> list:
    """The runs tensor read back (one chip's copy; debug): [[(w, sb, so, db, do, n)]]."""
    host = ttnn.to_torch(ttnn.get_device_tensors(runs)[0]).reshape(-1, runs.shape[-1]).to(torch.int64) & 0xFFFFFFFF
    out = []
    for row in host:
        n = int(row[0])
        out.append([tuple(int(v) for v in row[1 + RUN_WORDS * i : 1 + RUN_WORDS * (i + 1)]) for i in range(n)])
    return out


class SlotTables:
    """A block's copy tables (one a weight layout): pages 0 .. epc - 1 fill the spare slot (slot epc) from home slot s
    (fill_table(s) = s), page epc the load-time regroup from the epc-slot index cache (regroup_table)."""

    def __init__(self, lay_spc: dict, lay_epc: dict, epc: int, H: int, I: int, tb: int, mesh):
        self.epc = int(epc)
        self.lay_spc = SlotLayout(lay_spc, epc + 1, H, I, tb)
        self.lay_epc = SlotLayout(lay_epc, epc, H, I, tb)
        tables = []
        for s in range(self.epc):
            t = copy_runs(self.lay_spc, s, self.lay_spc, self.epc)
            check_inside(t, self.lay_spc, self.epc, "dst")
            check_inside(t, self.lay_spc, s, "src")
            tables.append(t)
        self.regroup = regroup_runs(self.lay_epc, self.lay_spc, self.epc)
        tables.append(self.regroup)
        self.host = tables
        self.runs = tables_tensor(tables, mesh)

    def fill_table(self, s: int) -> int:
        assert 0 <= s < self.epc, f"home slot {s}: a source is always a home slot (0 .. {self.epc - 1})"
        return s

    @property
    def regroup_table(self) -> int:
        return self.epc

    def verify(self) -> None:
        """Debug readback: the device's tables equal the host's, and every fill writes only inside the spare slot."""
        dev = read_tables(self.runs)
        assert dev == [list(t) for t in self.host], "the device's run tables differ from the host's"
        for s in range(self.epc):
            check_inside(dev[s], self.lay_spc, self.epc, "dst")
