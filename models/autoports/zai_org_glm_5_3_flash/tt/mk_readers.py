# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel: the reader cores' part of the layer program — every weight-matmul phase of the layer on the 16
bank-adjacent DS readers, back to back in ONE set of kernels (kernels/mk/reader_{w,a,c}.cpp):
  BRISC streams the weights of all phases in order (DSLinear's DRAM-sharded tensors in place), into per-format rings
  whose slot position carries over between phases of the same block size, so the next phase's first blocks land while
  the current phase computes and while the glue phases between run;
  NCRISC per phase waits for the phase's producers (hand-off semaphore), pulls the activation blocks, writes the
  output tiles and signals the consumers;
  TRISC per phase runs the ds_mm matmul (packer-L1-accumulated fp32 partials, CB rewound between blocks / phases).
Used standalone (`ReaderPhases.__call__`, inputs already in place: no waits) for validation, and as the reader part of
the layer program (`parts`)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.ds_mm import TILE_BYTES, _tiles_per_txn

KERNEL_DIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "mk")
TILE = 32
BF16, FP32 = ttnn.bfloat16, ttnn.float32
L1 = ttnn.L1_MEMORY_CONFIG
RING_SLOTS = 3
RING_BYTES_MAX = 160 * 1024  # a ring whose 3 slots would exceed this gets 2 (the reader cores' CB region also bounds
# every L1-interleaved tensor's address on ALL cores)
# deep rings (GLM53_MK_RING_KB > 0): every ring gets as many slots as fit in that many KB (>= 2): the weight streamer
# prefetches the next phase's blocks while the glue phases run (the stream never waits for activations)
# CB indices of the reader cores (the owner phases use their own indices on other cores)
CB_IN0, CB_PART, CB_OUT = 40, 41, 42
CB_RING0 = 43  # rings 43, 44, 45, ...
PROBE_CB = 31  # phase-probe scratch (kernels/mk/probe.hpp)
SKIP = 0xFFFFFFFF  # the activation mover's n0 of a reader that sits a phase out
CB_OUT32 = 38  # fp32 output tiles of fp32-output phases (the router's logits)
CB_STAGE = 39  # forwarding ring: one block of staging on the readers (DRAM -> here -> an owner's park slot)
CB_INVH = 36  # Stage 14 deferred 1/rms(h): the root's per-row 1/rms(h) tile (fp32, exact unpack) for the output pack
# reader kernel revision: part of the kernels' compile signature, so a long-lived process (serve.py) rebuilds them
# after a kernel edit (a kernel already built in-process under the same signature is not rebuilt). 2: the weight
# streamer reserves two blocks at phase start (shared-ring race).
MK_READER_REV = 4  # 4: phase-relative args, MK_SPLIT hook, per-phase output CB


def reader_assignments(device, readers: int, placement_name: str) -> list:
    """[(logical core, bank, index in bank)]: ttnn's bank-adjacent DS reader placement (1..3 per bank); for more, the 3
    placed ones plus, per bank, the free cores nearest (logical Manhattan distance) to that bank's readers, taken round
    robin over the banks so every bank's extra readers stay as close as the grid allows."""
    # tenstorrent/tt-metal has no DramReaderPlacement / dram_bank_reader_assignments (the fork's DRAM-streaming
    # matmul): the bank-adjacent core per bank comes from the public optimal assignment (one per bank, NOC 0) and the
    # remaining readers are the nearest free cores, as below; `placement_name` is kept for the call sites
    del placement_name
    bank_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    out = [(ttnn.CoreCoord(int(c.x), int(c.y)), int(b), 0) for b, c in enumerate(bank_workers)]
    if readers <= 1:
        return out
    grid = device.compute_with_storage_grid_size()
    taken = {(int(c.x), int(c.y)) for c, _, _ in out}
    free = [(x, y) for x in range(int(grid.x)) for y in range(int(grid.y)) if (x, y) not in taken]
    banks = sorted({b for _, b, _ in out})
    mine = {b: [(int(c.x), int(c.y)) for c, bb, _ in out if bb == b] for b in banks}
    for idx in range(1, readers):
        for b in banks:
            d = lambda xy: min(abs(xy[0] - x) + abs(xy[1] - y) for x, y in mine[b])
            best = min(free, key=d)
            free.remove(best)
            mine[b].append(best)
            out.append((ttnn.CoreCoord(*best), b, idx))
    return out


@dataclass
class MMPhase:
    ds: object  # DSLinear (weight, k, n_pad, readers, block_w, ckc)
    kb: int
    wait: tuple | None = None  # (semaphore id, producer count)
    signal: int | None = None  # semaphore id incremented on the consumers after the output landed
    math_fidelity: object = None  # None = the DSLinear's (parity: HiFi4 low-rank, HiFi2 the others)
    ring_kb: int | None = None  # deep ring budget (KB) of this phase's ring; None = GLM53_MK_RING_KB if it waits
    # Stage 14 K-split (GLM53_MK_ROUTER_KSPLIT): (chunk_tiles, k_tiles) — the weight's N holds the K-chunk partials side
    # by side (chunk c = output tiles [c chunk_tiles, (c + 1) chunk_tiles)); a reader of output tile n reads only
    # activation K tiles [c k_tiles, (c + 1) k_tiles) with c = n // chunk_tiles (padding tiles: chunk 0)
    k_split: tuple | None = None
    # 7a: a nonzero tag gives the phase a ring of its own even when another phase has the same format and block size
    # (mk_sparse GLM53_MK_QKV_ALIAS: the qkv ring aliases the experts' CB_X and must not carry another phase's stream)
    ring_tag: int = 0
    # 7a reader range per phase (Stage 14's ask): the phase runs on readers [r0, r0 + rp) of every bank, rp = the largest
    # divisor of its SHARD_W <= min(readers - r0, rp_max); the other readers sit it out (and still take part in its
    # hand-offs). r0 = 0 / rp_max = None: today's layout (readers 0 .. rp - 1)
    r0: int = 0
    rp_max: int | None = None

    def fidelity(self):
        return self.math_fidelity if self.math_fidelity is not None else self.ds.ckc.math_fidelity


class ReaderPhases:
    def __init__(self, device, phases: list, *, no_dram: bool = False, readers_per_bank: int | None = None):
        self.device = device
        self.phases = phases
        # Stage 14 deferred 1/rms(h) (mhc_pre MHC_DEFER_INV): dict(groups=[(phase indices, invh tensor)], sem=id) ->
        # those phases' outputs are scaled per row by the group's 1/rms(h); None = off (set per call by the owner)
        self.scale = None
        self.no_dram = no_dram  # timing only: the weight streamer skips its DRAM reads (compute-bound phase time)
        ds0 = phases[0].ds
        self.banks = device.dram_grid_size().x
        # readers per DRAM bank: the DS shard of a bank is row-major tiles [K, SHARD_W], so any count that divides a
        # phase's SHARD_W reads it in place (reader i: columns [i NR, (i + 1) NR)); a phase whose SHARD_W the count
        # does not divide runs on the largest divisor RP of readers 0..RP-1 of every bank (the others skip it but still
        # take part in its hand-offs). Default: the DSLinears' own count (GLM53_MK_READERS overrides).
        self.readers = int(readers_per_bank or int(os.environ.get("GLM53_MK_READERS", "0")) or ds0.readers)
        self.assign = reader_assignments(device, self.readers, ds0.ocfg.ds_placement)
        from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges

        self.cores = merged_ranges([(c.x, c.y) for c, _, _ in self.assign])
        # per phase: NB, KB, NR, TB, TPT, SHARD_W, ring
        self.geo, rings = [], {}
        for p in phases:
            ds = p.ds
            kt = ds.k // TILE
            shard_w = ds.n_pad // TILE // self.banks
            assert 0 <= p.r0 < self.readers, (p.r0, self.readers)
            lim = min(self.readers - p.r0, p.rp_max or self.readers)
            rp = max(r for r in range(1, lim + 1) if shard_w % r == 0)
            nr = shard_w // rp
            if kt % p.kb or (2 * max(q.kb for q in phases)) % p.kb:
                raise ValueError(f"{ds.role}: K block {p.kb}")
            tb = TILE_BYTES[ds.weight.dtype]
            key = (ds.weight.dtype, p.kb * nr, p.ring_tag)  # same format + same block size -> same ring
            if key not in rings:
                rings[key] = CB_RING0 + len(rings)
            self.geo.append(dict(nb=kt // p.kb, kb=p.kb, nr=nr, tb=tb, tpt=_tiles_per_txn(nr, tb), shard_w=shard_w,
                                 ring=rings[key], subw=max(s for s in (4, 2, 1) if nr % s == 0),
                                 w_dtype=ds.weight.dtype, rp=rp, r0=p.r0))  # fmt: skip
        self.rings = {cb: (dt, block) for (dt, block, _), cb in rings.items()}
        self.ring_slots = {cb: RING_SLOTS if RING_SLOTS * block * TILE_BYTES[dt] <= RING_BYTES_MAX else 2
                           for cb, (dt, block) in self.rings.items()}  # fmt: skip
        ring_kb = int(os.environ.get("GLM53_MK_RING_KB", "0"))  # read per program (long-lived process)
        # deep rings: a phase's own budget, else the default for waiting phases; a ring shared by several phases (same
        # format and block size: one continuous stream) gets the largest of their budgets
        want = {}
        for p, g in zip(phases, self.geo):
            kb_ = p.ring_kb if p.ring_kb is not None else (ring_kb if p.wait else 0)
            if kb_ > 0:
                dt, block = self.rings[g["ring"]]
                want[g["ring"]] = max(want.get(g["ring"], 0), 2, kb_ * 1024 // (block * TILE_BYTES[dt]))
        self.ring_slots.update(want)
        self.kb_max = max(p.kb for p in phases)
        self.nr_max = max(g["nr"] for g in self.geo)
        # forwarding ring (set by the megakernel before parts()): dict(owners=[(noc x, y)], slots=<park slots per
        # owner>, sem=<owner semaphore holding the park address>, mask=<bit p: phase p parks>, block_bytes=<B>)
        self.park = None
        self.calls = 0

    def l1_bytes(self) -> int:
        rings = sum(self.ring_slots[cb] * block * TILE_BYTES[dt] for cb, (dt, block) in self.rings.items())
        return rings + 2 * self.kb_max * 2048 + self.nr_max * (4096 + 2048)

    def noc_coords(self):
        return [(int(w.x), int(w.y)) for w in (self.device.worker_core_from_logical_core(c) for c, _, _ in self.assign)]

    @staticmethod
    def _out_dtype(p):
        return FP32 if getattr(p.ds, "dtype", BF16) == FP32 else BF16

    def _out_cb(self, p):
        return CB_OUT32 if self._out_dtype(p) == FP32 else CB_OUT

    def outputs(self, m: int, out_memory_config=L1):
        return [ttnn.empty((1, 1, m, p.ds.n_pad), self._out_dtype(p), ttnn.TILE_LAYOUT, self.device, out_memory_config)
                for p in self.phases]  # fmt: skip

    def parts(self, xs: list, ys: list, *, consumers: list | None = None, mk_probe: tuple | None = None):
        """(kernels, cbs): parts_raw's pieces as the three kernel descriptors."""
        raw, cbs = self.parts_raw(xs, ys, consumers=consumers, mk_probe=mk_probe)
        kernels = [self.kernel(*raw[r]) for r in ("w", "a", "c")]
        return kernels, cbs

    def kernel(self, name, ct, rt: dict, defs, cfg):
        rta = ttnn.RuntimeArgs()
        for (x, y), args in rt.items():
            rta[x][y] = [int(a) for a in args]
        return ttnn.KernelDescriptor(kernel_source=os.path.join(KERNEL_DIR, name),
                                     source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                     compile_time_args=ct, runtime_args=rta, config=cfg, defines=list(defs))  # fmt: skip

    def parts_raw(self, xs: list, ys: list, *, consumers: list | None = None, mk_probe: tuple | None = None):
        """({"w" / "a" / "c": (kernel file, compile-time args, {(x, y): runtime args}, defines, config)}, cbs) on the
        reader cores (a composite kernel concatenates them with another phase family's); xs / ys = per-phase input / output tensors; consumers = (x, y) NoC
        coordinates signalled after every phase with a `signal` semaphore; mk_probe = (probe tensor address, first
        slot): reader i's weight streamer -> slot + 2 i, its activation mover -> slot + 2 i + 1 (kernels/mk/probe.hpp).
        """
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        cons = consumers or []
        nph = len(self.phases)
        # weight streamer
        w_ct = [nph]
        for g in self.geo:
            w_ct += [
                g["nb"],
                g["kb"],
                g["nr"],
                g["tb"],
                g["tpt"],
                g["shard_w"],
                g["ring"],
                self.ring_slots[g["ring"]],
                (g["r0"] << 8) | g["rp"],
            ]
        w_rt = {}
        pr = lambda i, off: [mk_probe[0], mk_probe[1] + 2 * i + off] if mk_probe is not None else []
        for i, (c, bank, idx) in enumerate(self.assign):
            w_rt[(c.x, c.y)] = [bank, idx, bank & 3] + [p.ds.weight.buffer_address() for p in self.phases] + pr(i, 0)
            if self.park is not None:  # park args: n owners, (x, y, slots) each, the mask of the parking phases
                own = self.park["owners"][i :: len(self.assign)][:4]
                w_rt[(c.x, c.y)] = (
                    list(w_rt[(c.x, c.y)])
                    + [len(own)]
                    + [v for xy in own for v in (*xy, self.park["slots"])]
                    + [self.park["mask"]]
                )
        # activation mover
        a_ct = [nph, CB_IN0, CB_OUT, len(cons)]
        tail = []
        offs = []
        base = 4 + 9 * nph
        for x, y in zip(xs, ys):
            xo = base + len(tail)
            tail += acc(x)
            yo = base + len(tail)
            tail += acc(y)
            offs.append((xo, yo))
        for p, g, (xo, yo) in zip(self.phases, self.geo, offs):
            ws, wc = p.wait if p.wait else (0, 0)
            a_ct += [
                g["nb"],
                g["kb"],
                g["nr"],
                ws,
                wc,
                p.signal if p.signal is not None else 0,
                xo,
                yo,
                self._out_cb(p),
            ]
        a_ct += tail
        a_rt = {}
        flat_cons = [v for xy in cons for v in xy] if any(p.signal is not None for p in self.phases) else []
        if not flat_cons:
            a_ct[3] = 0
        for i, (c, bank, idx) in enumerate(self.assign):
            args = []
            for x, y, g, p_ in zip(xs, ys, self.geo, self.phases):
                inr = g["r0"] <= idx < g["r0"] + g["rp"]
                n0 = bank * g["shard_w"] + (idx - g["r0"]) * g["nr"] if inr else SKIP  # SKIP: not in this phase
                if p_.k_split is not None and n0 != SKIP:  # (koff << 16) | n0: this reader's K chunk
                    ct_, kt_ = p_.k_split
                    c_ = n0 // ct_
                    n_chunks = (p_.ds.n // TILE) // ct_ if hasattr(p_.ds, "n") else None
                    if n_chunks is not None and c_ >= n_chunks:
                        c_ = 0  # padding columns: any valid chunk (their output is never read)
                    n0 = ((c_ * kt_) << 16) | n0
                args += [x.buffer_address(), y.buffer_address(), n0]
            a_rt[(c.x, c.y)] = args + flat_cons + pr(i, 1)
        # compute
        c_ct = [nph, CB_IN0, CB_PART, CB_OUT]
        for p, g in zip(self.phases, self.geo):
            c_ct += [g["nb"], g["kb"], g["nr"], g["subw"], g["ring"], p.fidelity().value, self._out_cb(p)]
        c_rt = {}  # per phase: 1 = this reader computes it
        for c, bank, idx in self.assign:
            c_rt[(c.x, c.y)] = [int(g["r0"] <= idx < g["r0"] + g["rp"]) for g in self.geo]
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[CB_PART] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        kd = lambda name, ct, rt, cfg, defs=(): (
            name,
            ct,
            rt,
            list(defs) + [("MK_READER_REV", str(MK_READER_REV))],
            cfg,
        )
        pw = pa = ()
        if mk_probe is not None:
            pw = [("MK_PROBE_ARG", str(3 + nph)), ("MK_PROBE_CB", str(PROBE_CB))]
            pa = [("MK_PROBE_ARG", str(3 * nph + len(flat_cons))), ("MK_PROBE_CB", str(PROBE_CB))]
        if self.no_dram:
            pw = list(pw) + [("MK_NO_DRAM", "1")]
        if any(p.k_split is not None for p in self.phases):  # Stage 14: some phase's readers read a K chunk only
            pa = list(pa) + [("MK_KSPLIT", "1")]
        if self.park is not None:
            pw = list(pw) + [("MK_PARK_ARG", str(3 + nph + (2 if mk_probe is not None else 0))),
                             ("MK_PARK_SEM", str(self.park["sem"])), ("MK_STAGE_CB", str(CB_STAGE))]  # fmt: skip
        sdefs = []
        if self.scale is not None:  # kernels/mk/reader_a.cpp fetch_inv + reader_c.cpp's scaled pack
            groups = self.scale["groups"]
            bit = lambda ps: sum(1 << q for q in ps)  # noqa: E731
            mask, first, last = bit([q for ps, _ in groups for q in ps]), bit([min(ps) for ps, _ in groups]), \
                bit([max(ps) for ps, _ in groups])  # fmt: skip
            assert [min(ps) for ps, _ in groups] == sorted(min(ps) for ps, _ in groups), "groups in phase order"
            sct = len(a_ct)
            a_ct = a_ct + acc(groups[0][1])  # every invh tensor has the same [1,1,32,32] fp32 interleaved layout
            srt = len(next(iter(a_rt.values())))
            a_rt = {core: list(args) + [t.buffer_address() for _, t in groups] for core, args in a_rt.items()}
            sdefs = [("MK_SCALE_MASK", str(mask)), ("MK_SCALE_FIRST", str(first)), ("MK_SCALE_LAST", str(last)),
                     ("MK_INVH_CB", str(CB_INVH))]  # fmt: skip
            pa = list(pa) + sdefs + [("MK_INVH_SEM", str(self.scale["sem"])), ("MK_SCALE_CT", str(sct)),
                                     ("MK_SCALE_RT", str(srt))]  # fmt: skip
            modes[CB_INVH] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        # Stage 14 knob GLM53_MK_W_DELAY_US: the weight streamers start that late (the first phase's ring prefetch
        # otherwise competes for DRAM with mhc_pre's x / fn reads, which are on the critical path)
        if os.environ.get("GLM53_MK_W_DELAY_US"):
            pw = list(pw) + [("MK_W_DELAY_CYCLES", str(int(float(os.environ["GLM53_MK_W_DELAY_US"]) * 1350)))]
        raw = dict(
            w=kd("reader_w.cpp", w_ct, w_rt, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default), pw),
            a=kd("reader_a.cpp", a_ct, a_rt, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default), pa),
            c=kd("reader_c.cpp", c_ct, c_rt, cc, sdefs),
        )
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(
            total_size=page * n, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )  # fmt: skip
        cbs = [cb(CB_IN0, BF16, 2048, 2 * self.kb_max), cb(CB_PART, FP32, 4096, self.nr_max),
               cb(CB_OUT, BF16, 2048, self.nr_max)]  # fmt: skip
        nr32 = max((g["nr"] for p, g in zip(self.phases, self.geo) if self._out_dtype(p) == FP32), default=0)
        if nr32:  # fp32-output phases (the router's logits) pack into their own CB
            cbs.append(cb(CB_OUT32, FP32, 4096, nr32))
        for ring, (dt, block) in self.rings.items():
            cbs.append(cb(ring, dt, TILE_BYTES[dt], self.ring_slots[ring] * block))
        if mk_probe is not None:  # 2 x 64-B mark rows (weight streamer, activation mover)
            cbs.append(cb(PROBE_CB, ttnn.uint32, 128, 1))
        if self.park is not None:
            cbs.append(cb(CB_STAGE, ttnn.uint32, self.park["block_bytes"], 1))
        if self.scale is not None:
            cbs.append(cb(CB_INVH, FP32, 4096, 1))
        return raw, cbs

    def __call__(self, xs: list, *, out_memory_config=L1):
        """Standalone: every phase's input already in place (no waits / signals) -> the per-phase outputs."""
        ys = self.outputs(int(xs[0].shape[-2]), out_memory_config)
        kernels, cbs = self.parts(xs, ys)
        io = list(xs) + [p.ds.weight for p in self.phases] + ys
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        self.calls += 1
        return ys
