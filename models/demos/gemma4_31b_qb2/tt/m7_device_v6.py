# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""M7 device v6 (LEAD_DECISIONS_M7 M270, M273, M275; bench/resident/m7/hsplit_gates/HSPLIT_GATES_PREREG.md §2, §5): v5
(tt/m7_device_v5.py, frozen in class v5.5's package ac09673a) with D v10 (resident_m7/d_model_v10.py: v8 with the heads'
O and sem_o delivered as NoC1 unicasts, R4; v10 = v8 bitwise, gate G1) instead of D v8, at both imports of the D model.
Nothing else changed: set_launch's unconditional sync (v5 l.178; l.183 here), the ingest, the ring and the selector
are v5's. v5's docstring follows (its "D v8" reads "D v10").
M7's drafter-side device work for the batch class (bench/resident/m7/M7_BUILD_PREREG_v3.md §2.1-2.2, §3, §5).
Lineage: the H5 driver's backend (bench/resident/m7/h5/v3/h5_accept.py, Device; critic GO 10295c72), with the served
programs: D v8 (`rope` aliased to the engine's sliding RoPE pair, capacity = its rows), ingest v5, the drafter ring,
the batched selector. One M7Device per process, built once after the engine and before any trace (I3).

Order rules it keeps (m5-kernels' interfaces, as the host fakes bench/resident/m7/m7class/v5/fakes_m7.py model them):
- no set_* on an ingest or D instance with its own launch in flight: a sync first (the real set_* overwrite tensors
  that launch reads). Ingest v5's set_rows / set_seed_rows end with a device-wide sync themselves (F3);
- an ingest of taps launch k runs before engine launch k + 2 (taps_live; the class keeps it, set_rows asserts it);
- a D launch reads ring positions [p - 2048, p) of each user's slot, so the committed rows it needs are ingested first
  (command-queue order: the ingest launch is enqueued before the D launch);
- the D program's att_status words are sticky (word 1 ORs every launch's flags): flags() diffs them against the last
  read, and the class stops speculating on any new flag;
- no reset between launches (monotonic semaphores); idle_reset() only while idle, rarely (WRAP_GUARD).

v2 (for H4b, M7_BUILD_PREREG_v3_ADDENDUM_1 D7; m7_device.py v1 ff350c81 is frozen under review): the D step split in
two callables, the served path exactly, so H4b can import and time them unchanged:
- set_launch(block, jobs): the users' rows and x0 (set_users, set_x0), then the launch;
- read_select(block, jobs) -> (drafts, flags): chip 0's hub top-16 (the first blocking read, so it includes waiting for
  the launch), the final h rows, the batched selector, and the launched program's att_status flags. Each call leaves
  its phases' seconds in `timing` (set, launch; wait_top16, final_h, select, status, read_select).
draft() = set_launch + read_select, returning (drafts, flags): the class checks the flags with the drafts.

v5 (LEAD_DECISIONS_M7 M251, M254, M255; m5-kernels' m71/GT_B2_HANG_DIAGNOSIS.md f1704fe9 and placement review
m71/M7_DEVICE_V4_REVIEW.md b7e3d225 R1; prereg v3 addendum 15.2 F-SYNC / F-ALL; m7_device_v3.py c5862575 frozen;
m7_device_v4.py 674d1ea7 retired with class v5.4): every D launch starts from an idle mesh. set_launch syncs the mesh
(the class's ttnn.synchronize_device) UNCONDITIONALLY before its set_users / set_x0 / launch. v3 synced only when that
same D block was in flight, so a D launch could enter its fabric all-reduce while another chip still ran the ingest
(G-T v3's B2 device hang: chips 0, 2, 3 in D's all-reduce, chip 1's NoCs hung) or, in a step that launched no ingest,
the engine's verify (in_flight never tracks the engine; its reads are chip 0's only). v4's `if self.in_flight` missed
that second case (R1). Counters (sync_counts, M255): d_launches, d_after_mesh_sync (== d_launches by construction),
d_synced_after_ingest (the D launches whose sync also ended an in-flight ingest). Nothing else changed: ingest() keeps
v3's order (each set_rows ends with its own mesh-wide sync, d_ingest_v5.py:567), and read_select's flags() still reads
every chip after each D launch (D -> next engine launch).
v3 (m5b-planner's H4b mock leg on v2 6638a952: build() raised AttributeError): D v8's DrafterProgram has no
`capacity`; its F_POSITION bound is its attention's, prog.att.capacity (d_model_v8.py:326-328: the aliased RoPE
tables' rows). build()'s log reads that, and capacity() gives the class its spec limit from it. The host fakes carried
`capacity` on the program, so v2's host tests passed; the harness now gives the fakes v8's shape (an `att`).
"""

import os
import time
from collections import Counter

import torch
from loguru import logger

SYNCED_KINDS = ("verify", "ingest", "D", "switch")  # the per-1K line's kinds (prereg §2.3)


class M7Device:
    """The drafter ring, ingest program and D programs ({block: program}) on `mesh`, plus the selector's host inputs.
    `sync` is ttnn.synchronize_device bound to the mesh. Host tests build it on fakes and override _candidates and
    _final_h (the device reads)."""

    def __init__(self, *, ring, ingest, progs, embed, mask_id, whp, pred_cb, succ_cb, sync, engine):
        self.ring, self.ingest_prog, self.progs, self.embed, self.mask_id = ring, ingest, dict(progs), embed, mask_id
        self.whp, self.pred_cb, self.succ_cb, self.sync, self.engine = whp, pred_cb, succ_cb, sync, engine
        self.in_flight = set()  # "ingest" / blocks whose last launch has not been synced
        self.status_seen = {b: None for b in self.progs}  # last att_status words per program (sticky)
        self.mask_x = None
        self.timing = {}  # seconds per phase of the last set_launch / read_select
        # v5: D launches, each after a mesh-wide sync (M255: d_after_mesh_sync == d_launches), and those whose sync
        # also ended an in-flight ingest (synced ingest -> D transitions)
        self.sync_counts = Counter(d_launches=0, d_after_mesh_sync=0, d_synced_after_ingest=0)

    # ------------------------------------------------------------------ construction (served)

    @classmethod
    def build(cls, mesh, engine, *, blocks, staging_slots, rings, rope):  # (v3: the log reads p.att.capacity)
        """The served construction, in the H5 driver's order (h5_accept.main): drafter and ingest weights from the
        drafter cache (GEMMA4_M7_DRAFTER_CACHE, convert()'s layout) and the head from the engine's raw cache, the ring,
        ingest v5 on the engine's two taps buffers, one D v8 program of 32 rows per draft block with `rope` aliased, the
        selector's codebooks and W_hp, the drafter's mask token, and the target's raw embedding rows (mmap)."""
        import ttnn
        from safetensors import safe_open

        from models.demos.gemma4_31b_qb2.resident_m7 import d_model_v10 as dmod  # (v6) D v10, R4
        from models.demos.gemma4_31b_qb2.resident_m7.d_ingest_v5 import DrafterRing, IngestProgram, IngestWeightsOnDevice
        from models.demos.gemma4_31b_qb2.resident_m7.drafter_common import D_HP
        from models.demos.gemma4_31b_qb2.resident_m7.drafter_weights import dequantized
        from models.demos.gemma4_31b_qb2.resident_r64b.weights import PREFIX, Checkpoint
        from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import RAW_WEIGHT_CACHE

        dcache = os.environ.get("GEMMA4_M7_DRAFTER_CACHE")
        if not dcache or not os.path.isfile(os.path.join(dcache, "selector.pt")):
            raise ValueError(f"GEMMA4_M7_DRAFTER_CACHE={dcache!r}: M7 needs the resident drafter cache "
                             "(resident_m7.drafter_weights.convert's layout: L*/, global/, vectors.pt, selector.pt)")
        head_cache = os.environ.get("GRES_CACHE", RAW_WEIGHT_CACHE)
        vectors = torch.load(os.path.join(dcache, "vectors.pt"))
        ring = DrafterRing(mesh, slots=rings)
        ingest = IngestProgram(mesh, IngestWeightsOnDevice(mesh, dcache), vectors, ring,
                               taps=(engine.taps_buffer(0), engine.taps_buffer(1)), n_taps=len(engine.cfg.tap_layers),
                               dump=False, staging_slots=staging_slots)
        weights = dmod.DrafterWeightsOnDevice(mesh, dcache, head_cache)
        progs = {b: dmod.DrafterProgram(mesh, weights, vectors, rows=32, block=b, ring=ring, rope=rope)
                 for b in sorted(blocks)}
        sel = torch.load(os.path.join(dcache, "selector.pt"))
        ck = Checkpoint()
        name = PREFIX + "embed_tokens.weight"
        handle = safe_open(ck.path / ck.index[name], framework="pt")  # kept open: mmap'd rows, read per launch
        table = handle.get_slice(name)

        def embed(ids, handle=handle):
            return torch.stack([table[int(t): int(t) + 1][0] for t in ids]).float()

        device = cls(ring=ring, ingest=ingest, progs=progs, embed=embed, mask_id=drafter_mask_token(),
                     whp=dequantized(dcache, D_HP)[0].float(), pred_cb=sel["pred_cb"].float(),
                     succ_cb=sel["succ_cb"].float(), sync=lambda: ttnn.synchronize_device(mesh), engine=engine)
        device.out_words = dmod.OUT_WORDS
        logger.info("M7 drafter built: D v8 blocks {}, ring slots {}, staging slots {}, D capacity {}", sorted(blocks),
                    rings, staging_slots, {b: p.att.capacity for b, p in progs.items()})
        return device

    # ------------------------------------------------------------------ order helpers

    def capacity(self):
        """the positions every D program can draft at: min over the programs of their attention's capacity (a block at
        p is flagged F_POSITION and skipped when p + M exceeds it)"""
        return min(int(p.att.capacity) for p in self.progs.values())

    def _idle(self, what):
        if what in self.in_flight:
            self.sync()
            self.in_flight.clear()

    def synced(self):
        """the caller synced the device (a blocking read, a program switch)"""
        self.in_flight.clear()

    # ------------------------------------------------------------------ operations

    def zero(self, ring):
        """C9': clear a ring slot before any seed or draft (DrafterRing.zero_slot syncs)"""
        self.ring.zero_slot(ring)
        self.in_flight.clear()

    def ingest(self, rows, taps_launch):
        """rows [(launch row, position, ring slot)] of engine launch `taps_launch`'s taps into the rings, 32 per ingest
        launch (each set_rows syncs at its end, F3; the launches stay queued before whatever the caller enqueues
        next)"""
        for c in range(0, len(rows), 32):
            chunk = rows[c: c + 32]
            self._idle("ingest")
            self.ingest_prog.set_rows([r for r, _, _ in chunk], [p for _, p, _ in chunk], [s for _, _, s in chunk],
                                      taps_launch, self.engine.launches)
            self.in_flight.clear()  # set_rows synced the device
            self.ingest_prog.launch()
            self.in_flight.add("ingest")
        return -(-len(rows) // 32)

    def draft(self, block, jobs):
        """jobs [(user index u, ring slot, anchor id, anchor position p)] on the block's D program (32 rows, user u =
        rows [u block, (u + 1) block)): set_launch, then read_select. Returns (drafts [len(jobs)][block - 1] in job
        order, {block: new flag bits} or {})."""
        self.set_launch(block, jobs)
        return self.read_select(block, jobs)

    def set_launch(self, block, jobs):
        """the D program's users (set_users: each user block's first position and ring slot) and x0 (row u block: the
        anchor's embedding; the user's other rows: the mask token's), then its launch"""
        t0 = time.perf_counter()
        prog = self.progs[block]
        U = 32 // block
        positions, slots = [None] * U, [0] * U
        for u, ring, _, p in jobs:
            positions[u], slots[u] = p, ring
        if self.mask_x is None:
            self.mask_x = self.embed([self.mask_id])[0]
        x0 = self.mask_x.expand(32, -1).clone()
        x0[[u * block for u, _, _, _ in jobs]] = self.embed([a for _, _, a, _ in jobs])
        # v5 (M251, M254; m5-kernels R1): every D launch starts from an idle mesh, whatever ran before it on any chip
        # (an ingest, another D block, or the engine's verify / decode, which in_flight does not track and whose
        # reads are chip 0's only); v3: _idle(block) only
        self.sync_counts["d_synced_after_ingest"] += "ingest" in self.in_flight
        self.sync()
        self.in_flight.clear()
        self.sync_counts["d_after_mesh_sync"] += 1
        prog.set_users(positions, slots)
        prog.set_x0(x0)
        t1 = time.perf_counter()
        prog.launch()
        self.sync_counts["d_launches"] += 1
        self.in_flight.add(block)
        self.timing = {"set": t1 - t0, "launch": time.perf_counter() - t1}

    def read_select(self, block, jobs):
        """after set_launch(block, jobs): chip 0's hub top-16 (a blocking read: the launch and everything queued before
        it are done), the final h rows, the batched selector, then the launched program's att_status flags. Returns
        (drafts [len(jobs)][block - 1], {block: new flag bits} or {}); the phases' seconds go to `timing`."""
        from models.demos.gemma4_31b_qb2.resident_m7.batched_selector import hidden_projection, select_paths

        t0 = time.perf_counter()
        prog, K = self.progs[block], block - 1
        live = [u for u, _, _, _ in jobs]
        unary, cand = self._candidates(prog, live, block)
        self.in_flight.clear()
        t1 = time.perf_counter()
        fh = self._final_h(prog)
        t2 = time.perf_counter()
        idx = torch.tensor(live)[:, None] * block + torch.arange(1, block)[None, :]
        hp = hidden_projection(fh[idx.reshape(-1)], self.whp).view(len(live), K, -1)
        anchors = torch.tensor([a for _, _, a, _ in jobs], dtype=torch.int64)
        drafts = select_paths(unary, cand, hp, anchors, self.pred_cb, self.succ_cb).tolist()
        t3 = time.perf_counter()
        flags = self.flags((block,))
        t4 = time.perf_counter()
        self.timing.update(wait_top16=t1 - t0, final_h=t2 - t1, select=t3 - t2, status=t4 - t3, read_select=t4 - t0)
        return drafts, flags

    def flags(self, blocks=None):
        """{block: new flag bits} since the last call, from each program's sticky att_status (word 1 and the user
        nibbles, every chip) and its schedule-miss cores; empty when nothing new. A blocking read, of the programs in
        `blocks` (default: all; a program not launched since its last read has nothing new)."""
        new = {}
        for block, prog in self.progs.items():
            if blocks is not None and block not in blocks:
                continue
            words = [(list(st[:12]), miss) for st, miss in prog.att_status()]
            seen, self.status_seen[block] = self.status_seen[block], words
            for chip, (st, miss) in enumerate(words):
                old = seen[chip][0] if seen is not None else [0] * 12
                bits = st[1] & ~old[1]
                users = any(st[i] & ~old[i] for i in range(8, 12))  # a user nibble newly set (word 1 may not show it)
                if bits or users or miss:
                    new[block] = new.get(block, 0) | bits | (users << 31) | (bool(miss) << 30)
        self.in_flight.clear()
        return new

    def idle_reset(self):
        """zero the ingest's and D programs' monotonic semaphore counts, while idle (far from uint32 wrap)"""
        self.sync()
        self.in_flight.clear()
        self.ingest_prog.reset()
        for prog in self.progs.values():
            prog.reset()

    # ------------------------------------------------------------------ device reads (host tests override)

    def _candidates(self, prog, live, block):
        """chip 0's out tensor words -> (unary [U, K, 16], cand [U, K, 16]) (h5_accept Device.draft)"""
        import ttnn

        from models.demos.gemma4_31b_qb2.resident_m7.batched_selector import candidates

        words = ttnn.to_torch(ttnn.get_device_tensors(prog.out_t)[0]).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        return candidates(words, self.out_words, 32, live, block)

    def _final_h(self, prog):
        """chip 0's final h rows [M, H]: dump row 6 x layers, sliced on device (h5_accept Device.final_h)"""
        import ttnn

        from models.demos.gemma4_31b_qb2.resident_m7 import d_model_v10 as dmod  # (v6) D v10, R4
        from models.demos.gemma4_31b_qb2.resident_m7.model import tile_unorder

        row, w = 6 * prog.layers, prog.dump_t.shape[-1]
        s = ttnn.slice(prog.dump_t, [row, 0], [row + 1, w])
        d = ttnn.to_torch(ttnn.get_device_tensors(s)[0]).reshape(prog.K, prog.V * 1024)
        ttnn.deallocate(s)
        flat = torch.cat([d[k, : prog.T * dmod.TB // 2] for k in range(prog.K)])
        return tile_unorder(flat, 32)[: prog.M].float()


def drafter_mask_token():
    """DFlash2's mask token id (its config.json), as the H5 driver reads it (h5_accept.drafter_cfg)"""
    import json

    from models.demos.gemma4_31b_qb2.tt.dflash import drafter_checkpoint
    from models.demos.gemma4_31b_qb2.tt.dflash2 import DFLASH2_ID, DFLASH2_REVISION

    path = os.environ.get("GEMMA4_RESIDENT_DRAFTER_CHECKPOINT") or drafter_checkpoint(DFLASH2_ID, DFLASH2_REVISION)
    with open(os.path.join(path, "config.json")) as f:
        return int(json.load(f)["mask_token_id"])
