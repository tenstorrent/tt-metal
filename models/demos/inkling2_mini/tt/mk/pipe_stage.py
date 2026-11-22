# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Weight staging by the DRAM cores: the tensor prefetcher (DRISC kernels on the DRAM cores, tt-metal
``start_tensor_prefetcher``) streams the next weights of bank-streamed linears into an L1 ring on the linears' own
cores while other programs run. No Tensix core and no host program carries the copy, so it overlaps everything,
including the ttnn ops (SDPA, norms) that cannot carry tt/mk/stage.py's fill kernels.

    ps = PipeStage(device, {key: bank_linear})   # one ring per core, one DRAM copy per key
    ps.start()                                    # starts the prefetcher (once per device)
    ps.request([key, ...], capture=...)           # slabs, in the order the linears will run
    y = ps(key, x)                                # = that BankLinear's output, weight from the ring
    ps.stop()

How: every core of a (re-split, tt/mk/stage.resplit) BankLinear reads one contiguous range of its bank. A key's
copy lays those ranges out as a receiver-contiguous prefetcher weight: shard m (bank m % 7, slab m // 7) is ring
position m's range, padded to whole entries, so the prefetcher delivers each core its range in order. The bytes are
those of the original weight (the same bf16 values tilized to the same dtype), so every cb_w slot gets the same
bytes and the output is bit-equal. The reader pops its slab when done, which frees that part of the ring for the
slabs behind it: the ring runs up to one ring ahead of its readers. All keys share the cores (one ring per core)
and the entry size, so slabs of different keys follow each other in one ring.

Cost: the ring's bytes of L1 on every core (the L1 allocator is lock-step) and a second DRAM copy of each weight.
"""

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.attn_wo import AttnWo
from models.demos.inkling2_mini.tt.fused.bank_linear import DST_TILES, TILE, BankLinear
from models.demos.inkling2_mini.tt.mk.stage import MKDIR

BIG = 1 << 30  # "staged" bytes: the whole range from the ring


def _core_list(crs):
    out = []
    for r in crs.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                out.append((x, y))
    return out


def range_tiles(bl, i):
    return sum(ncq for ncq, _ in bl.items[i][2]) * bl.nblk * bl.KB


def place_like(bl, ref):
    """``bl`` (a resplit copy) moved onto ``ref``'s cores: its work items of bank b on ref's first cores of bank b,
    so the two linears share their ring cores. Placement does not change the output."""
    import copy

    from models.demos.inkling2_mini.tt.mk.stage import coalesce

    free = {}
    for c, i in ref.cores:
        free.setdefault(ref.items[i][0], []).append(c)
    out = copy.copy(bl)
    out.cores, seen = [], {}
    for i, (b, _, _) in enumerate(bl.items):
        out.cores.append((free[b][seen.get(b, 0)], i))
        seen[b] = seen.get(b, 0) + 1
    out.grid = coalesce([c for c, _ in out.cores])
    return out


def rebuild(bl, ckc, **kw):
    """A BankLinear over the same matrix as ``bl`` (read back from its device weight, exact) with other options
    (e.g. pass_cols, so that resplit can cut every work item over more cores). Bit-equal to ``bl``."""
    nb = bl.nb
    pages = ttnn.to_torch(bl.weight).bfloat16().reshape(-1, TILE, TILE)  # page = slot * nb + bank
    nt = bl.chunk * bl.n_chunks // TILE
    W = torch.zeros(bl.Kt, nt, TILE, TILE, dtype=torch.bfloat16)
    for (b, rot, part), off in zip(bl.items, bl.offsets):
        slot = off
        for ncq, g0 in part:
            for p in range(bl.nblk):
                j = (p + rot) % bl.nblk
                for r in range(bl.KB):
                    W[j * bl.KB + r, g0 : g0 + ncq] = pages[(torch.arange(slot, slot + ncq) * nb + b).long()]
                    slot += ncq
    w = W.permute(0, 2, 1, 3).reshape(bl.Kt * TILE, nt * TILE)
    return BankLinear(bl.device, w, bl.dtype, bl.chunk, bl.KB, kw.pop("cores_per_bank"), ckc, out_cols=bl.out_cols,
                      out_dtype=bl.out_dtype, **kw)  # fmt: skip


class PipeStage:
    def __init__(self, device, banks, entry_tiles=32):
        """banks: {key: BankLinear}, all on the same cores with the same bank per core and the same dtype
        (tt/mk/stage.resplit copies). entry_tiles: tiles per entry (the reader starts on a block once its entries
        landed; every key's per-core range is padded to whole entries)."""
        self.device = device
        self.banks = banks
        b0 = next(iter(banks.values()))
        self.nb, self.dtype = b0.nb, b0.dtype
        self.tb = ttnn.Tile([TILE, TILE]).get_tile_size(self.dtype)
        cbank = {c: b0.items[i][0] for c, i in b0.cores}  # the ring cores (the first key's, all its cores)
        self.idle = {}  # key -> ring cores its linear does not use (they only pop its empty slab)
        for k, bl in banks.items():
            cb = {c: bl.items[i][0] for c, i in bl.cores}
            assert all(cbank.get(c) == b for c, b in cb.items()) and bl.dtype == self.dtype, k
            self.idle[k] = sorted(set(cbank) - set(cb))
        self.entry_tiles = entry_tiles
        self.entry = entry_tiles * self.tb
        self.n_entries = {k: -(-max(range_tiles(bl, i) for _, i in bl.cores) // entry_tiles) for k, bl in banks.items()}
        self.ring_entries = max(self.n_entries.values())
        self.ring_bytes = self.ring_entries * self.entry
        per_bank = {b: [c for c in cbank if cbank[c] == b] for b in range(self.nb)}
        R = len(per_bank[0])
        assert all(len(v) == R for v in per_bank.values()), {b: len(v) for b, v in per_bank.items()}
        self.R = R
        crs = lambda cs: ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cs]
        )  # noqa: E731
        self.space = ttnn.experimental.create_prefetcher_pipe_space(
            device, sender_cores=ttnn.CoreRangeSet(set()), receiver_domain=crs(list(cbank)),
            ring_size=self.ring_bytes, max_receivers_per_pipe=R, num_dram_senders=2 * self.nb,
        )  # fmt: skip
        # one pipe (sender) per bank, so receiver r of a pipe is bank-local slab r
        self.pipes = ttnn.experimental.create_prefetcher_pipes_for_tensor_prefetcher(
            self.space, [(b, crs(per_bank[b])) for b in range(self.nb)], support_multi_receiver_shards=True
        )
        assert self.space.ring_size() == self.ring_bytes, (self.space.ring_size(), self.ring_bytes)
        # ring position of every core: receiver r of bank b's pipe gets bank-local slab r = shard b + r * nb
        self.pos = {}
        for p in self.pipes:
            rc = _core_list(p.receiver_cores())
            b = next(b for b, v in per_bank.items() if set(v) == set(rc))
            for r, c in enumerate(rc):
                self.pos[c] = b + r * self.nb
        assert sorted(self.pos.values()) == list(range(self.nb * R)), self.pos
        self.ring, self.cfg = self.space.buffer_address(), self.space.config_address()
        self.copies = {k: self._copy(k, bl) for k, bl in banks.items()}
        self.running = False

    def _copy(self, k, bl):
        """The receiver-contiguous DRAM copy of key k's weight: column tile m = ring position m's range."""
        pages = ttnn.to_torch(bl.weight).bfloat16().reshape(-1, TILE, TILE)  # page = slot * nb + bank
        n_pos, S = self.nb * self.R, self.n_entries[k] * self.entry_tiles
        out = torch.zeros(S, n_pos, TILE, TILE, dtype=torch.bfloat16)
        for c, i in bl.cores:
            b, n = bl.items[i][0], range_tiles(bl, i)
            out[:n, self.pos[c]] = pages[(torch.arange(bl.offsets[i], bl.offsets[i] + n) * self.nb + b).long()]
        out = out.permute(0, 2, 1, 3).reshape(1, 1, S * TILE, n_pos * TILE)
        spec = ttnn.NdShardSpec(
            ttnn.Shape([S * TILE, TILE]),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.nb - 1, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )  # fmt: skip
        return ttnn.from_torch(out, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device,
                               memory_config=ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec))  # fmt: skip

    def start(self, **mpfe):
        """mpfe: start_tensor_prefetcher's MPFE weights (DRAM arbitration of the prefetcher's senders against
        ordinary reads)."""
        if not self.running:
            ttnn.experimental.start_tensor_prefetcher(self.device, **mpfe)
            self.running = True

    def stop(self):
        if self.running:
            ttnn.experimental.stop_tensor_prefetcher(self.device)
            ttnn.synchronize_device(self.device)
            self.running = False

    def drain(self, k):
        """Read (and drop) one queued slab of key k: stop() only returns once every slab it queued was popped."""
        bl = self.banks[k]
        x = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, bl.K]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device,
                                           ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        ttnn.deallocate(self(k, x))
        ttnn.deallocate(x)

    def request(self, keys, capture=False):
        """Queue the slabs of `keys`, in the order their linears will run."""
        ttnn.experimental.queue_tensor_prefetcher_request(
            self.device, [(self.copies[k], self.n_entries[k]) for k in keys], prefetcher_pipes=self.pipes,
            capture_into_trace=capture,
        )  # fmt: skip

    def reader(self, k, acc, named, r_args, staged=None):
        """The weight reader kernels (cb 1) of key k's linear program, reading the ring (plus a pop-only kernel on
        the ring cores the linear does not use). r_args: {core: the bank_linear_reader.cpp runtime args}."""
        rnamed = dict(named, entry=self.entry, n_entries=self.n_entries[k], ring_bytes=self.ring_bytes)
        st = BIG if staged is None else staged
        dm = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
        out = [ttnn.KernelDescriptor(
            kernel_source=f"{MKDIR}/pipe_linear_reader.cpp", core_ranges=self.banks[k].grid, compile_time_args=acc,
            named_compile_time_args=list(rnamed.items()),
            runtime_args=[(c, v[:3] + [self.cfg, self.ring, st] + v[3:]) for c, v in r_args.items()], config=dm,
        )]  # fmt: skip
        if self.idle[k]:
            cs = [ttnn.CoreCoord(*c) for c in self.idle[k]]
            out.append(ttnn.KernelDescriptor(
                kernel_source=f"{MKDIR}/pipe_pop.cpp", core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs]),
                named_compile_time_args=[("entry", self.entry), ("n_entries", self.n_entries[k])],
                runtime_args=[(c, [self.cfg]) for c in cs], config=dm,
            ))  # fmt: skip
        return out

    def __call__(self, k, x, out=None, staged=None):
        """Key k's BankLinear(x, out) with the weight from the ring (staged: bytes per core taken from the ring,
        default the whole range; the slab is popped either way)."""
        bl = self.banks[k]
        tiles_out = out is None
        if tiles_out:
            shape = ttnn.Shape([1, 1, x.shape[-2], bl.out_cols])
            out = ttnn.allocate_tensor_on_device(shape, bl.out_dtype, ttnn.TILE_LAYOUT, bl.device,
                                                 ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        else:
            assert bl.out_dtype == ttnn.float32 and x.shape[-2] == 1
        otb = ttnn.Tile([TILE, TILE]).get_tile_size(bl.out_dtype)
        named = dict(kt=bl.Kt, kb=bl.KB, nblk=bl.nblk, n_banks=bl.nb, out_cols=bl.out_cols,
                     out_page=otb if tiles_out else out.shape[-1] * 4, max_ncq=bl.max_ncq, tiles_out=int(tiles_out),
                     out_tile=otb)  # fmt: skip
        r_args, c_args, w_args = {}, [], []
        for c, i in bl.cores:
            b, rot, part = bl.items[i]
            core = ttnn.CoreCoord(*c)
            r_args[core] = [bl.weight.buffer_address(), b, bl.offsets[i], len(part)] + [n for n, _ in part]
            c_args.append((core, [rot, len(part)] + [n for n, _ in part]))
            w_args.append(
                (core, [x.buffer_address(), rot, out.buffer_address(), len(part)] + [v for p_ in part for v in p_])
            )
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        blk = bl.KB * bl.max_ncq
        kernels = [
            *self.reader(k, acc(bl.weight), named, r_args, staged),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_compute.cpp", core_ranges=bl.grid,
                named_compile_time_args=list(named.items()), runtime_args=c_args, config=bl.compute,
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_writer.cpp", core_ranges=bl.grid, compile_time_args=acc(x) + acc(out),
                named_compile_time_args=list(named.items()), runtime_args=w_args,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
        ]  # fmt: skip
        t32 = ttnn.TileDescriptor(ttnn.Tile([TILE, TILE]))

        def cb(i, fmt, page, n):
            return ttnn.CBDescriptor(total_size=page * n, core_ranges=bl.grid,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page, tile=t32)])  # fmt: skip

        cbs = [
            cb(0, ttnn.bfloat16, 2048, bl.Kt),
            cb(1, bl.dtype, self.tb, 4 * blk),
            cb(16, bl.out_dtype, otb, 2 * DST_TILES),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
        return ttnn.generic_op([x, bl.weight, out], prog)


class PipeAttnWo(AttnWo):
    """AttnWo (o-projection + post-attention conv) whose weight comes from a PipeStage ring."""

    def __init__(self, bank, cols, ps, key, on_call):
        super().__init__(bank, cols)
        self.ps, self.key, self.on_call = ps, key, on_call

    def __call__(self, x, conv):
        self.on_call(self.key)
        return super().__call__(x, conv)

    def _reader(self, acc, named, r_args):
        return self.ps.reader(self.key, acc, named, r_args)


class _Proxy:
    """Stands in for one layer's BankLinear in _DramShardedLinear.bank."""

    def __init__(self, ov, key, bl):
        self.ov, self.key, self.bl = ov, key, bl

    def __getattr__(self, name):
        return getattr(self.bl, name)

    def __call__(self, x, out=None):
        self.ov.on_call(self.key)
        return self.ov.ps(self.key, x, out)


class PipeOverlap:
    """Decode weight staging by the DRAM cores for the attention projections of every layer: w_all (and with
    wo=True the o-projection + conv program, A4's AttnWo) read their weights from one PipeStage ring on the same 42
    cores (each layer's BankLinear re-split over `split` x its cores off rows `exclude_rows`, as tt/mk/overlap.py;
    w_o is rebuilt with narrower passes first so it splits as far). Per decode step one request queues the slabs in
    call order, starting behind app 0's w_all: [w_o 0, w_all 1, w_o 1, ..., w_all 0 of the next step]; captured
    into the trace, so every replay re-sends it. Install queues the first w_all 0. Decode steps only; uninstall()
    after whole steps.

        ov = PipeOverlap(model)   # before capture, then two eager steps; ov.uninstall(model)
    """

    def __init__(self, model, wo=False, split=3, exclude_rows=(0, 1, 2, 3, 4, 5), entry_tiles=32, mpfe=None,
                 wo_pass_cols=2, wo_split=None):  # fmt: skip
        from models.demos.inkling2_mini.tt import model as MD
        from models.demos.inkling2_mini.tt.mk.stage import resplit

        self.apps = [L for L, _ in MD.APPS]
        gx = model.device.compute_with_storage_grid_size().x
        ex = [(x, y) for y in exclude_rows for x in range(gx)]
        self.attn = {L: lay.attn for L, lay in enumerate(model.layers)}
        self.orig_all = {L: a.w_all.bank for L, a in self.attn.items()}
        assert all(b is not None for b in self.orig_all.values()), "w_all: needs the bank-streamed projection"
        banks = {("w_all", L): resplit(b, split, ex) for L, b in self.orig_all.items()}
        self.orig_wo = None
        if wo:
            self.orig_wo = {L: a.wo for L, a in self.attn.items()}
            assert all(w is not None for w in self.orig_wo.values()), "w_o: needs the fused o-projection program"
            for L, a in self.attn.items():
                nb = rebuild(a.w_o.bank, a.w_o.bank_ckc, cores_per_bank=2, pass_cols=wo_pass_cols)
                banks[("w_o", L)] = place_like(
                    resplit(nb, wo_split or split, ex), banks[("w_all", L)]
                )  # its own DRAM weight stays: the program binds it
        self.ps = PipeStage(model.device, banks, entry_tiles)
        self.order = []
        for L in self.apps:
            self.order += [("w_all", L)] + ([("w_o", L)] if wo else [])
        self.ptr, self.n, self.steps = 0, 0, 0
        self.ps.start(**(mpfe or {}))
        self.ps.request(self.order[:1])
        for L, a in self.attn.items():
            a.w_all.bank = _Proxy(self, ("w_all", L), banks[("w_all", L)])
            if wo:
                a.wo = PipeAttnWo(banks[("w_o", L)], a.wo.cols, self.ps, ("w_o", L), self.on_call)

    def on_call(self, key):
        assert self.order[self.ptr] == key, ("call order out of step", self.ptr, key, self.order[self.ptr])
        if self.ptr == 0:  # sent now, or captured when a trace is being captured
            self.ps.request(self.order[1:] + self.order[:1], capture=True)
            self.steps += 1
        self.ptr = (self.ptr + 1) % len(self.order)
        self.n += 1

    def uninstall(self, model):
        assert self.ptr == 0, "uninstall between decode steps only (the prefetcher waits for the step's readers)"
        self.ps.drain(self.order[0])  # the next step's first slab is queued: stop() waits until it is read
        for L, a in self.attn.items():
            a.w_all.bank = self.orig_all[L]
            if self.orig_wo:
                a.wo = self.orig_wo[L]
        self.ps.stop()
        for t in self.ps.copies.values():
            ttnn.deallocate(t)
        self.ps = None

    def stats(self):
        out = {"pipe_calls": self.n, "pipe_steps": self.steps}
        if self.ps:
            out.update(ring_kb=self.ps.ring_bytes // 1024, entries={k[0]: v for k, v in self.ps.n_entries.items()})
        return out
