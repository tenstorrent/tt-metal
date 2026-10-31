# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device side of the device-resident block-paged APC (env ``DSV41_APC_PAGED=1``, default off; design: APC_PAGED_DESIGN.md, interface: INTERFACE_DEVICE.md of the dev log dir).

Two persistent DRAM stores, both allocated at model build BEFORE the first trace capture and only ever written IN PLACE (trace-layout hazard, OPTIMIZATIONS.md / SPEC_ADAPT_NOTES.md):

  * ``KeyStore``  : the index keys of one 512-token chunk block (the 4 index-key owners, 40 KiB per chip per block) per store slot. ``key_save`` copies the chunk's keys (scratch rows of the single-trace
                    key slab) to a slot, ``key_load`` copies the slots of a cached prefix to the history rows of the slab. Both are JIT ``generic_op`` data-movement kernels (tt/paged_kernels), tensor driven.
  * ``HaloStore`` : the window halos (``[U,1,128,512]`` bf16 per layer, 128 KiB) of a chunk end, part A = layers 0..20 (always) and part B = layers 21..39 (only inside the bounded-replay zone). ``halo_copy``
                    saves the persistent per-layer halo tensors of a user into a slot / restores them from it (device local copy).

No CCL anywhere: every device copies between its own tensors (SPMD, the slot ids are the same on the 8 columns of a mesh row and differ per mesh row).
Slot ids reach the kernels through small uint32 ROW_MAJOR tables ``[rows * U, 32]`` (row = mesh_row * U + slot; element 0 = store slot, ``SKIP`` = untouched) refreshed with ``h2d`` (recordable in the
prefetch thread, like ``kofs``): a captured trace reads them at replay time.
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P
from models.demos.blackhole.deepseek_v41_flash.tt.h2d import h2d

SKIP = 0xFFFFFFFF
TILE_BYTES = 2048  # bf16 32x32 tile
WINDOW = 128
HEAD_DIM = 512
IDIM = 128
HALO_PAGES = (WINDOW // 32) * (HEAD_DIM // 32)  # 64 tiles = 128 KiB per (user, layer)
N_LAYERS = 40
PART_A = tuple(range(0, 21))  # halo layers that are valid on every chunk end
PART_B = tuple(range(21, 40))  # halo layers that only run inside the bounded-replay zone
GRID = 512  # hit grid (tokens): one chunk
PAGE_TOKENS = 128

DEFAULT_HALO_A = 128
DEFAULT_HALO_B = 48
DEFAULT_KEY_BLOCKS = 2048


def enabled():
    return os.environ.get("DSV41_APC_PAGED", "0") == "1"


def env_sizes():
    """(halo A slots, halo B slots, key blocks) per mesh row from DSV41_APC_PAGED_HALO_A / _HALO_B / _KEYBLOCKS"""
    g = lambda k, d: int(os.environ.get(k, str(d)))
    return (
        g("DSV41_APC_PAGED_HALO_A", DEFAULT_HALO_A),
        g("DSV41_APC_PAGED_HALO_B", DEFAULT_HALO_B),
        g("DSV41_APC_PAGED_KEYBLOCKS", DEFAULT_KEY_BLOCKS),
    )


# ------------------------------------------------------------------------------------------------------------------ CPU: geometry and slot accounting
class KeyGeom:
    """index math of the key blocks for one compress ratio (CPU only, no device)"""

    def __init__(self, chunk, ratio, cols):
        assert chunk % GRID == 0 and chunk % ratio == 0
        self.chunk, self.ratio, self.cols = int(chunk), int(ratio), int(cols)
        self.Cc = chunk // ratio  # entries per chunk
        assert self.Cc % (cols * 32) == 0, "a chunk's entries must split into whole 32-row tiles per column"
        self.Cc8 = self.Cc // cols  # local rows per chunk per column
        self.nrb = self.Cc8 // 32  # 32-row blocks per chunk per column (1 for ratio 2, 2 for ratio 1 at C = 512)
        self.np = self.nrb * 4  # tile pages per chunk block per column (4 tile columns of 32 features)
        self.block_bytes = self.np * TILE_BYTES

    def slab_rows(self, max_ctx):
        """local rows of the key slab ([scratch Cc8 | history max_ctx / ratio / cols] + 32 pad rows) as ``alloc_single``"""
        Tl = (self.Cc + max_ctx // self.ratio) // self.cols
        return Tl + 32

    def hist_row_block(self, i):
        """first 32-row block of the slab rows that hold chunk block i (history entry j at global slot Cc + j) = the ``kofs`` value of a chunk starting at i * GRID"""
        return self.nrb * (1 + i)

    def hist_page0(self, i):
        return self.hist_row_block(i) * 4

    def max_blocks(self, rows):
        """chunk blocks that fit the history of a slab with ``rows`` local rows"""
        return (rows // 32 - self.nrb) // self.nrb


def block_of(tokens):
    """chunk-block index of a token count on the 512 grid"""
    assert tokens % GRID == 0
    return tokens // GRID


def halo_part_layers(part):
    return PART_A if part == "A" else PART_B


def halo_slot_bytes(part):
    """per chip"""
    return len(halo_part_layers(part)) * HALO_PAGES * TILE_BYTES


def store_bytes_per_chip(halo_a, halo_b, key_blocks, ratios=(2, 2, 2, 1), chunk=GRID, cols=8):
    ks = sum(KeyGeom(chunk, r, cols).block_bytes for r in ratios)
    return {
        "halo_a": halo_a * halo_slot_bytes("A"),
        "halo_b": halo_b * halo_slot_bytes("B"),
        "keys": key_blocks * ks,
        "key_block_bytes": ks,
    }


class SlotPool:
    """free list of store slots of ONE mesh row (CPU). ``alloc`` -> slot or None, ``free``; double free / foreign slot raise. The owner of the block index decides who holds a slot."""

    def __init__(self, n):
        self.n = int(n)
        self._free = list(range(self.n - 1, -1, -1))  # LIFO, slot 0 first
        self._used = set()

    def alloc(self):
        if not self._free:
            return None
        s = self._free.pop()
        self._used.add(s)
        return s

    def free(self, s):
        if s not in self._used:
            raise ValueError(f"slot {s} is not allocated")
        self._used.remove(s)
        self._free.append(s)

    @property
    def free_count(self):
        return len(self._free)

    @property
    def used_count(self):
        return len(self._used)


class RowSlots:
    """one ``SlotPool`` per mesh row for halo A, halo B and key blocks"""

    def __init__(self, rows, halo_a=None, halo_b=None, key_blocks=None):
        a, b, k = env_sizes()
        a = a if halo_a is None else halo_a
        b = b if halo_b is None else halo_b
        k = k if key_blocks is None else key_blocks
        self.rows = rows
        self.pools = {
            "A": [SlotPool(a) for _ in range(rows)],
            "B": [SlotPool(b) for _ in range(rows)],
            "K": [SlotPool(k) for _ in range(rows)],
        }

    def alloc(self, kind, row):
        return self.pools[kind][row].alloc()

    def free(self, kind, row, slot):
        self.pools[kind][row].free(slot)


# ------------------------------------------------------------------------------------------------------------------ slot tables (host -> device)
def table_host(md, vals, width=32):
    """host tensor of a slot table: ``vals`` = rows * U entries (flat, row-major over (mesh row, slot)); an entry is a slot id / None (SKIP) for ``width`` == 32 tables, or a list of slot ids for
    wider tables (the list ends at the first SKIP). Sharded over the mesh rows."""
    rows, cols = tuple(md.shape)
    t = torch.full((len(vals), width), SKIP, dtype=torch.int64)
    for i, v in enumerate(vals):
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            assert len(v) <= width, (len(v), width)
            for j, s in enumerate(v):
                t[i, j] = SKIP if (s is None or int(s) < 0) else int(s)  # (-1 = none, as the BlockIndex ids)
        elif int(v) >= 0:
            t[i, 0] = int(v)
    return ttnn.from_torch(
        t.to(torch.int64).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )


def table_device(md, n, width=32):
    """persistent table tensor, all SKIP (allocated at build)"""
    return ttnn.from_torch(
        torch.full((n, width), SKIP, dtype=torch.int64).to(torch.int32),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
    )


def set_table(md, dev, vals, width=32):
    """refresh a persistent table in place (``h2d``: recordable, no allocation)"""
    h2d(table_host(md, vals, width), dev)


# ------------------------------------------------------------------------------------------------------------------ kernels
def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _one(core_ranges_cores, total, ct, rt, kernel, common, tensors, tag):
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_ranges_cores])
    cb = ttnn.CBDescriptor(
        total_size=total,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=total)],
    )
    rta = ttnn.RuntimeArgs()
    for c, a in zip(core_ranges_cores, rt):
        rta[c.x][c.y] = a
    k = ttnn.KernelDescriptor(
        kernel_source=f"{P.KDIR}/{kernel}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rta,
        common_runtime_args=common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    prog.custom_program_hash = (tag << 40) | (hash(tuple(ct)) & ((1 << 40) - 1))
    ttnn.generic_op(tensors, prog)


def key_save(slab, sid, store, users, nrb):
    """store[sid[u]] = scratch rows [0, nrb * 32) of slab[u] for every user u with sid[u] != SKIP (IN PLACE, tile layout, one core per user).
    slab [U,1,T,128] bf16 tile, sid [rows*U, 32] uint32 RM (per device [U, 32]), store [K,1,nrb*32,128] bf16 tile."""
    NP = nrb * 4
    DP = (int(slab.shape[2]) // 32) * 4
    K = int(store.shape[0])
    assert int(store.shape[2]) == nrb * 32 and int(store.shape[3]) == IDIM
    ct = [NP, DP, TILE_BYTES, 0, K] + _acc(slab) + _acc(sid) + _acc(store)
    cores = [ttnn.CoreCoord(u, 0) for u in range(users)]
    _one(
        cores,
        128 + NP * TILE_BYTES,
        ct,
        [[u] for u in range(users)],
        "key_save.cpp",
        [slab.buffer_address(), sid.buffer_address(), store.buffer_address()],
        [slab, sid, store],
        0x9C1,
    )


def key_load(store, ltab, slab, users, nrb):
    """slab history chunk blocks i = 0, 1, ... of user u = store[ltab[u][i]] up to the first SKIP (IN PLACE). ltab [rows*U, NBMAX] uint32 RM (per device [U, NBMAX])."""
    NP = nrb * 4
    DP = (int(slab.shape[2]) // 32) * 4
    K = int(store.shape[0])
    NBMAX = int(ltab.shape[-1])
    MAXI = (DP - NP) // NP
    ct = [NP, DP, TILE_BYTES, 0, K, NBMAX, MAXI] + _acc(store) + _acc(ltab) + _acc(slab)
    cores = [ttnn.CoreCoord(u, 0) for u in range(users)]
    _one(
        cores,
        NBMAX * 4 + NP * TILE_BYTES,
        ct,
        [[u] for u in range(users)],
        "key_load.cpp",
        [store.buffer_address(), ltab.buffer_address(), slab.buffer_address()],
        [store, ltab, slab],
        0x9C2,
    )


def halo_copy(tab, store, halos, users, direction, ncores=8):
    """direction 'save': store[slot tab[u], layer l] = halos[l][u]; 'restore': halos[l][u] = store[slot tab[u], layer l]; users with tab[u] == SKIP are untouched (IN PLACE).
    halos: list of NL tensors [U,1,128,512] bf16 tile, store [K * NL, 1, 128, 512] bf16 tile, tab [rows*U, 32] uint32 RM.
    """
    NL = len(halos)
    assert int(store.shape[0]) % NL == 0
    K = int(store.shape[0]) // NL
    assert HALO_PAGES % ncores == 0
    PPC = HALO_PAGES // ncores
    d = {"save": 0, "restore": 1}[direction]
    ct = [NL, users, PPC, d, TILE_BYTES, 0, K] + _acc(tab) + _acc(store) + _acc(halos[0])
    cores = [ttnn.CoreCoord(c, 0) for c in range(ncores)]
    common = [tab.buffer_address(), store.buffer_address()] + [h.buffer_address() for h in halos]
    tensors = [tab] + list(halos) + [store] if d == 0 else [tab, store] + list(halos)
    _one(cores, 128 + PPC * TILE_BYTES, ct, [[c] for c in range(ncores)], "halo_copy.cpp", common, tensors, 0x9C3)


# ------------------------------------------------------------------------------------------------------------------ persistent stores
class KeyStore:
    """key store of the index-key owners. ``owners``: {name: ratio}. One tensor per owner, per device ``[K, 1, nrb*32, 128]`` bf16 tile (zeros at build); the slot id is shared by the owners (a slot = one chunk
    block of all owners). Tables: ``sid`` (save) and ``ltab`` (load, up to ``nbmax`` blocks), both ``[rows*U, *]`` uint32, all SKIP at build. Allocate before the first trace capture.
    """

    def __init__(self, md, U, owners, chunk=GRID, K=None, nbmax=None):
        self.md, self.U = md, int(U)
        self.rows, self.cols = tuple(md.shape)
        self.K = env_sizes()[2] if K is None else int(K)
        self.nbmax = self.K if nbmax is None else int(nbmax)
        assert self.nbmax % 32 == 0, "ltab rows are padded to 128 B"
        self.geom = {n: KeyGeom(chunk, r, self.cols) for n, r in owners.items()}
        self.store = {
            n: ttnn.zeros([self.K, 1, g.nrb * 32, IDIM], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
            for n, g in self.geom.items()
        }
        n = self.rows * self.U
        self.sid = table_device(md, n, 32)
        self.ltab = table_device(md, n, self.nbmax)

    def set_save(self, vals):
        """vals: rows*U entries, store slot or None"""
        set_table(self.md, self.sid, vals, 32)

    def set_load(self, vals):
        """vals: rows*U entries, list of store slots (chunk block 0, 1, ...) or None"""
        set_table(self.md, self.ltab, vals, self.nbmax)

    def save(self, slabs):
        """slabs: {owner: key slab tensor}; call after the chunk's keys are in the scratch rows (``key_append`` with offset 0)"""
        for n, g in self.geom.items():
            key_save(slabs[n], self.sid, self.store[n], self.U, g.nrb)

    def load(self, slabs):
        for n, g in self.geom.items():
            key_load(self.store[n], self.ltab, slabs[n], self.U, g.nrb)

    def warm(self, slabs):
        """compile both programs with all-SKIP tables (before the first trace capture); touches no data"""
        n = self.rows * self.U
        self.set_save([None] * n)
        self.set_load([None] * n)
        self.save(slabs)
        self.load(slabs)


class HaloStore:
    """halo store: part A (layers 0..20) and B (layers 21..39). Per device ``[K * NL, 1, 128, 512]`` bf16 tile per part. Tables per part: ``save[part]`` and ``restore[part]`` ``[rows*U, 32]`` (all SKIP at build).
    Allocate before the first trace capture; the halos passed to save / restore are the persistent per-layer ``halo`` tensors (written in place by restore).
    """

    def __init__(self, md, U, halo_a=None, halo_b=None):
        self.md, self.U = md, int(U)
        self.rows, self.cols = tuple(md.shape)
        a, b, _ = env_sizes()
        self.K = {"A": a if halo_a is None else int(halo_a), "B": b if halo_b is None else int(halo_b)}
        self.layers = {"A": PART_A, "B": PART_B}
        self.store = {
            p: ttnn.zeros(
                [self.K[p] * len(self.layers[p]), 1, WINDOW, HEAD_DIM],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=md,
            )
            for p in "AB"
            if self.K[p] > 0
        }
        n = self.rows * self.U
        self.save_tab = {p: table_device(md, n) for p in self.store}
        self.restore_tab = {p: table_device(md, n) for p in self.store}

    def set_save(self, part, vals):
        set_table(self.md, self.save_tab[part], vals)

    def set_restore(self, part, vals):
        set_table(self.md, self.restore_tab[part], vals)

    def _halos(self, part, halos):
        """``halos``: {layer: tensor} or a list of 40 per-layer tensors"""
        return [halos[l] for l in self.layers[part]]

    def save(self, part, halos):
        halo_copy(self.save_tab[part], self.store[part], self._halos(part, halos), self.U, "save")

    def restore(self, part, halos):
        halo_copy(self.restore_tab[part], self.store[part], self._halos(part, halos), self.U, "restore")

    def warm(self, halos):
        """compile save and restore of every part with all-SKIP tables (before the first trace capture); touches no data"""
        for p in self.store:
            self.set_save(p, [None] * (self.rows * self.U))
            self.set_restore(p, [None] * (self.rows * self.U))
            self.save(p, halos)
            self.restore(p, halos)
