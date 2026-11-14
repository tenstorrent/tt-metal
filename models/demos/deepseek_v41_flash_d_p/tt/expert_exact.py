# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MOE_REPLICAS=exact (P.55, owner 2026-11-13): the host side of the streamed expert copy (P.55b, step 2).

The copy runs inside a layer's own routed_experts_combine (ttnn.bringup.moe_block_main.set_combine_fill_exact, the
port writer's REC_FX path). Terms:
* piece: one fabric packet of a slot copy (expert_slot_copy's chunk, 10880 B on the 8x4); a slot (one expert's
  w_gu + w_d, bfp4) is ~1,830 of them, numbered across the run table's runs.
* path: one link plane (link 0 or 1) of the first hop from the sender toward the receiver; a copy edge sends half the
  pieces on each of the two paths (MOE_EXACT paths = 2), so the two links carry it at once.
* block / ready flag: every MOE_EXACT_BLOCK pieces of a path (and at its end) the sender's port sends a flushed atomic
  increment of the receiver's flag word for that path: the block has landed in the receiver's DRAM.
* tree: a hot expert with several receivers goes out as a tree; the home sends to its children, each receiver relays
  the blocks it has received to its own children (a relay job is gated on its incoming flags, block by block), at most
  MOE_EXACT_FANOUT children a chip. Every receiver holds one replica (the spare slot).
* jobs tensor: per chip COMBINE_FX_PAGES rows of COMBINE_FX_PAGE_WORDS uint32 (the kernel's layout,
  combine_port_writer.cpp REC_FX): row dir x links + link = that port's jobs, row COMBINE_FX_CHIP_PAGE = the chip's
  waits (the receiver's flag words and their targets). Built here on the host for step 2 (the microbenchmark); P.55c's
  device planner writes the same rows.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import ttnn

from .expert_slots import SlotLayout, check_inside, copy_runs, tables_tensor

AICLK_MHZ = 1350.0  # the wall clock the kernel stamps in (cycles a us)


def _mb():
    return ttnn.bringup.moe_block_main


class ExactTables:
    """Run tables of the exact copy: pages 0 .. epc - 1 home slot s -> the spare slot (as SlotTables), page epc the
    relay (the spare slot -> the spare slot: what a receiver forwards). All in the same canonical run order and run
    lengths, so piece p of a fill and piece p of a relay are the same bytes of the expert."""

    def __init__(self, lay: SlotLayout, epc: int, chunk: int, mesh):
        self.epc, self.chunk = int(epc), int(chunk)
        tables = []
        for s in range(self.epc):
            t = copy_runs(lay, s, lay, self.epc)
            check_inside(t, lay, self.epc, "dst")
            tables.append(t)
        relay = copy_runs(lay, self.epc, lay, self.epc)
        check_inside(relay, lay, self.epc, "src")
        check_inside(relay, lay, self.epc, "dst")
        shape = [(r[0], r[5]) for r in relay]
        assert all([(r[0], r[5]) for r in t] == shape for t in tables), "a fill and the relay differ in their runs"
        tables.append(relay)
        self.host = tables
        self.runs = tables_tensor(tables, mesh)
        self.pieces = sum(-(-r[5] // self.chunk) for r in relay)

    def fill_table(self, s: int) -> int:
        assert 0 <= s < self.epc
        return s

    @property
    def relay_table(self) -> int:
        return self.epc


def _ring(a: int, b: int, n: int) -> int:
    d = (b - a) % n
    return min(d, n - d)


def hops(a: int, b: int, shape: tuple, torus: bool = True) -> int:
    R, C = shape
    (ra, ca), (rb, cb) = divmod(a, C), divmod(b, C)
    if torus:
        return _ring(ra, rb, R) + _ring(ca, cb, C)
    return abs(ra - rb) + abs(ca - cb)


def plan_tree(home: int, receivers: list, shape: tuple, fanout: int, dir_of=None) -> list:
    """Edges (parent, child, depth) of one hot expert's copy tree: receivers nearest the home first; each one's parent
    is a chip already in the tree with fewer than `fanout` children, preferring one whose port toward it carries no
    other child yet (two children through one port go one after the other: a whole copy late; a relay level costs only
    a block), then the nearest, then the shallowest."""
    order = sorted(receivers, key=lambda r: (hops(home, r, shape), r))
    depth = {home: 0}
    kids = {home: 0}
    used: dict = {home: set()}
    edges = []
    for r in order:
        cand = [p for p in depth if kids[p] < fanout]

        def busy(p):
            return dir_of is not None and int(dir_of(p, r)) in used[p]

        p = min(cand, key=lambda p: (busy(p), hops(p, r, shape), depth[p], p))
        edges.append((p, r, depth[p] + 1))
        depth[r] = depth[p] + 1
        kids[p] += 1
        kids[r] = 0
        used[r] = set()
        if dir_of is not None:
            used[p].add(int(dir_of(p, r)))
    return edges


def build_jobs(
    hot: list,
    n: int,
    shape: tuple,
    links: int,
    dir_of,
    pieces: int,
    block: int,
    cap: int,
    relay_table: int,
    paths: int = 2,
    fanout: int = 2,
) -> tuple:
    """hot: [(home chip, home slot, [receiver chips])] (one replica a receiver), or (home, slot, [r], via): one receiver,
    half the pieces straight (both links), half through chip via's spare slot (relayed on its two links: four links). dir_of(a, b): the direction of chip a's
    port toward b's first hop (combine_fill_port). -> (uint32 [n, PAGES, PAGE_WORDS] numpy, info dict: edges
    [(home, slot, parent, child, depth)], waits {chip: [(word, target)]}, jobs a port)."""
    mb = _mb()
    PAGES, CHIP, PW = mb.COMBINE_FX_PAGES, mb.COMBINE_FX_CHIP_PAGE, mb.COMBINE_FX_PAGE_WORDS
    MAXJ = mb.COMBINE_FX_MAX_JOBS
    assert 1 <= paths <= links and 4 * links <= CHIP, (paths, links)
    out = np.zeros((n, PAGES, PW), dtype=np.uint32)
    per_port: dict = {}
    waits: dict = {}
    edges_all = []
    taken = set()
    cut = [pieces * i // paths for i in range(paths + 1)]

    def edge(src, dst, table, p0, p1, li, word, gate_word, dep):
        """One path of a copy edge: pieces [p0, p1) on src's port toward dst on link li, flags on dst's word `word`;
        gate_word: src relays what it received on its own word (None: src holds the slot)."""
        gate = 0 if gate_word is None else (gate_word + 1) | block << 16
        per_port.setdefault((src, int(dir_of(src, dst)) * links + li), []).append((dep, table, dst, p0, p1, word, gate))
        waits.setdefault(dst, []).append((word, -(-(p1 - p0) // block)))

    for entry in hot:
        home, slot, rcv = entry[:3]
        via = entry[3] if len(entry) > 3 else None
        assert home not in rcv and len(set(rcv)) == len(rcv)
        for r in rcv:
            assert r not in taken, f"chip {r} receives two replicas (one spare slot a chip)"
            taken.add(r)
        if via is not None:  # (a detour: half straight on both links, half through chip `via`'s spare slot)
            assert len(rcv) == 1 and via not in taken and via != home, "a detour: one receiver, a free relay chip"
            taken.add(via)
            r = rcv[0]
            q = [pieces * i // (2 * paths) for i in range(2 * paths + 1)]
            for li in range(paths):
                edge(home, r, slot, q[li], q[li + 1], li, li, None, 1)
                edge(home, via, slot, q[paths + li], q[paths + li + 1], li, li, None, 1)
                edge(via, r, relay_table, q[paths + li], q[paths + li + 1], li, paths + li, li, 2)
            edges_all += [(home, slot, home, r, 1), (home, slot, home, via, 1), (home, slot, via, r, 2)]
            continue
        for p, c, dep in plan_tree(home, rcv, shape, fanout, dir_of):
            edges_all.append((home, slot, p, c, dep))
            for li in range(paths):
                if p == home:
                    edge(p, c, slot, cut[li], cut[li + 1], li, li, None, dep)
                else:
                    edge(p, c, relay_table, cut[li], cut[li + 1], li, li, li, dep)
    for (chip, page), jobs in per_port.items():
        assert len(jobs) <= MAXJ, f"chip {chip} port page {page}: {len(jobs)} jobs > {MAXJ}"
        jobs.sort(key=lambda j: (j[0], j[2]))  # (shallow edges first: a deep subtree waits on them anyway)
        out[chip, page, 0] = len(jobs)
        out[chip, page, 1] = cap
        for i, (_, table, c, p0, p1, word, gate) in enumerate(jobs):
            out[chip, page, 16 + 8 * i : 16 + 8 * i + 6] = [table | c << 8, p0, p1, 0, word | block << 16, gate]
    for chip, ws in waits.items():
        assert len(ws) <= mb.COMBINE_FX_WAITS
        out[chip, CHIP, 2] = len(ws)
        for w, (word, target) in enumerate(ws):
            out[chip, CHIP, 8 + 2 * w] = word
            out[chip, CHIP, 9 + 2 * w] = target
    info = {
        "edges": edges_all,
        "waits": waits,
        "jobs_max_port": max((len(v) for v in per_port.values()), default=0),
        "depth_max": max((e[4] for e in edges_all), default=0),
    }
    return out, info


def jobs_tensor(host: np.ndarray, mesh) -> ttnn.Tensor:
    """[n, PAGES, PAGE_WORDS] -> the jobs tensor: per chip uint32 [1, 1, PAGES, PAGE_WORDS] ROW_MAJOR DRAM (a row a
    page), chip d's rows on device d."""
    n, P, W = host.shape
    t = torch.from_numpy(host.astype(np.int64)).to(torch.int32).reshape(n, 1, P, W)
    return ttnn.from_torch(
        t,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def read_words(res) -> np.ndarray:
    """The exact words of every chip: uint32 [n, COMBINE_FX_WORDS] (flags 2 x 32, stamps 8 x 16 at 64, arrivals
    4 x 64 at 192)."""
    t = _mb().combine_fill_exact_words(res)
    rows = [ttnn.to_torch(d).reshape(-1).to(torch.int64).numpy() & 0xFFFFFFFF for d in ttnn.get_device_tensors(t)]
    return np.stack([r[: _mb().COMBINE_FX_WORDS] for r in rows]).astype(np.uint64)


def stamps(words: np.ndarray, chip: int, links: int) -> dict:
    """One chip's stamps of the last call, in us from its waiter port's barrier (page of fill port 0)."""
    w = words[chip]
    pages = {}
    for p in range(4 * links):
        s = w[64 + 16 * p : 64 + 16 * p + 16]
        if s[0] == 0 and s[7] == 0:
            continue
        pages[p] = s
    waiter = [p for p, s in pages.items() if s[9] != 0]
    ref = pages[waiter[0]][0] if waiter else min(int(s[0]) for s in pages.values())

    def us(t):
        return ((int(t) - int(ref)) & 0xFFFFFFFF) / AICLK_MHZ if t else float("nan")

    out = {
        "status": max(int(s[4]) for s in pages.values()) if pages else 0,
        "pieces": sum(int(s[5]) for s in pages.values()),
        "flags": sum(int(s[6]) for s in pages.values()),
        "last_row_us": max(us(int(s[1]) & ~1) for s in pages.values()) if pages else float("nan"),
        "loop_end_us": max(us(s[2]) for s in pages.values()) if pages else float("nan"),
        "fill_end_us": max(us(s[3]) for s in pages.values()) if pages else float("nan"),
        "exit_us": max(us(s[7]) for s in pages.values()) if pages else float("nan"),
    }
    if waiter:
        s = pages[waiter[0]]
        out["land_us"] = us(s[8])
        out["wait_ok"] = int(s[9]) == 1
        out["wait_final"] = [int(v) for v in s[10:14]]
        arr = []
        for k in range(4):
            a = w[192 + 64 * k : 192 + 64 * k + 64]
            arr.append([us(v) for v in a if v])
        out["arrivals_us"] = arr
    return out


def aiclk_mhz() -> float:
    """The cards' AICLK (sysfs; the wall clock the stamps count), else 1350."""
    import glob

    vals = []
    for f in glob.glob("/sys/class/tenstorrent/*/tt_aiclk"):
        try:
            vals.append(float(open(f).read().split()[0]))
        except (OSError, ValueError, IndexError):
            pass
    return float(np.median(vals)) if vals else AICLK_MHZ


def ms_cycles(ms: float) -> int:
    return int(min(ms * AICLK_MHZ * 1000.0, 2**31 - 1))


def blocks_of(pieces: int, block: int, paths: int) -> list:
    cut = [pieces * i // paths for i in range(paths + 1)]
    return [math.ceil((cut[i + 1] - cut[i]) / block) for i in range(paths)]
