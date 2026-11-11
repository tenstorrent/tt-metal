# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Target tap rows for the resident speculative adapter (bench/resident/M2_DESIGN_v4.md §4).

Every resident launch writes the post-layer residual of the drafter's tap layers for its rows into
engine.taps_buffer(s % 2), [1, 1, 32, n * 5376], row r = position p + r. The drafter needs those rows twice:
  - at a draft, the rows the last verify committed, as rows 0..c-1 of drafter.taps (one launch: §4.1);
  - at an entry (first draft, re-entry), a linear 2,080-row window ending at ceil32(A) for seed().

Three buffers carry them, all zero-initialised and updated in place only:
  staging  [1, 1, 32, n * 5376]  where(rowmask[c], launch taps, 0): rows >= c are exact zeros, never NaN
  ring     [40, n, 32, 5376]     one slot per launch (the staging tile, viewed as n heads of width 5376)
  history  n x [2P, 1, 32, 5376] logical page g (positions 32g..32g+31) at slots g mod P and g mod P + P, so any
                                 P-page window is one contiguous slice; P = 65 = the seed window

Host bookkeeping (TapBook, pure, unit-tested) decides where every position lives; TapBuffers owns the tensors and
issues only fixed-shape programs: the ring insert and page flush at one shape, the window read through the device
tensor-args slice (one program for every offset), prefill parts at one of P row-count classes (warmed at startup).
"""

import torch

import ttnn

TAP_WIDTH = 5376
TILE = 32
RING_SLOTS = 40
HISTORY_PAGES = 65  # P: the fixed seed window, SEED_ROWS 2080 = 65 pages
WINDOW_ROWS = HISTORY_PAGES * TILE
TABLE_WIDTH = 80  # page-table row: >= P entries, a multiple of 16 int32 (64-byte DRAM rows)


def page_of(position):
    return position // TILE


def mirror_slots(page, pages=HISTORY_PAGES):
    """The two history slots that hold logical page `page`."""
    slot = page % pages
    return slot, slot + pages


def window(anchor_position, pages=HISTORY_PAGES):
    """The seed window for a draft anchored at position A: positions < A are valid.

    Returns (first_page, start_position, valid). The window is `pages` pages ending at the page holding A - 1
    (end = ceil32(A)), or pages [0, pages) while A is shorter than the window; rows >= valid are selected to 0
    by the seed."""
    if anchor_position <= 0:
        raise ValueError(f"A seed window needs at least one committed position, anchor at {anchor_position}")
    last = page_of(anchor_position - 1)
    first = max(0, last - (pages - 1))
    start = first * TILE
    return first, start, anchor_position - start


def window_slot(first_page, pages=HISTORY_PAGES):
    """First mirrored slot of the window starting at `first_page`: slots s..s+pages-1 are contiguous."""
    return first_page % pages


def dram_free_per_chip(mesh):
    """Free DRAM bytes on each chip (the mesh allocator is lock-step: one view describes every chip)."""
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return int(view.num_banks) * int(view.total_bytes_free_per_bank)


def plain_page_entries(first_slot, slots=RING_SLOTS):
    """A history page's ring entries in the plain regime: 32 launches of one committed row each, in consecutive ring
    slots from `first_slot` (TapBook.insert takes the next slot per launch), row 0 of each."""
    return tuple((r, (first_slot + r) % slots, 0) for r in range(TILE))


def selection_matrix(entries, layers, slots=RING_SLOTS):
    """0/1 [layers * 32, slots * layers * 32] bf16: output row 32h + r takes ring row (slot * layers + h) * 32 + row
    for every (r, slot, row) in `entries`. Rows without an entry are 0."""
    sel = torch.zeros(layers * TILE, slots * layers * TILE, dtype=torch.bfloat16)
    for r, slot, row in entries:
        for h in range(layers):
            sel[h * TILE + r, (slot * layers + h) * TILE + row] = 1
    return sel


class TapBook:
    """Where each committed position's tap rows live, for one request.

    Positions below `prefill_end` with a whole page below it were written to the history by the prefill. The page
    holding prefill_end (when it is not page-aligned) and every decode position go through the launch-tap ring: a
    launch's committed rows are recorded as (slot, row); a page flushes from the ring once all its positions are
    recorded. `insert` refuses to overwrite a slot still holding an unflushed row."""

    def __init__(self, slots=RING_SLOTS, pages=HISTORY_PAGES):
        self.slots, self.pages = slots, pages
        self.reset()

    def reset(self):
        self.prefill_end = 0
        self.committed_end = 0  # positions < committed_end are committed
        self.pending = {}  # position -> (slot, row), recorded but not yet flushed
        self.flushed_pages = set()  # pages written from the ring since the last reset
        self.inserts = 0

    def note_prefill(self, end):
        """A prefill call wrote positions < end to the history (its window part). An earlier call's partial page
        is in that part again, so its ring entries are dropped."""
        if end < self.committed_end:
            raise ValueError(f"Prefill end {end} is behind the committed position {self.committed_end}")
        self.prefill_end = self.committed_end = end
        self.pending.clear()

    def check_write(self, page):
        """The overwrite rule: never write a page past the one holding the last committed position."""
        if self.committed_end <= 0 or page > page_of(self.committed_end - 1):
            raise ValueError(f"History page {page} is past the last committed position {self.committed_end - 1}")

    def insert(self, first_position, rows):
        """Record a launch's committed rows (positions first_position..+rows-1) in the next ring slot."""
        slot = self.inserts % self.slots
        held = sorted(p for p, (s, _) in self.pending.items() if s == slot)
        if held:
            raise ValueError(f"Ring slot {slot} still holds unflushed positions {held[:4]}")
        if first_position != self.committed_end:
            raise ValueError(f"Launch rows start at {first_position}, expected the next position {self.committed_end}")
        for r in range(rows):
            self.pending[first_position + r] = (slot, r)
        self.committed_end = first_position + rows
        self.inserts += 1
        return slot

    def insert_page(self, page, rows):
        """Record a prefill page's first `rows` rows (positions 32 * page + r) in the next ring slot."""
        slot = self.inserts % self.slots
        if any(s == slot for s, _ in self.pending.values()):
            raise ValueError(f"Ring slot {slot} still holds unflushed positions")
        for r in range(rows):
            self.pending[page * TILE + r] = (slot, r)
        self.inserts += 1
        return slot

    def complete_pages(self):
        """Pages whose 32 positions are all recorded, oldest first."""
        pages = sorted({page_of(p) for p in self.pending})
        return [g for g in pages if all(g * TILE + r in self.pending for r in range(TILE))]

    def entries(self, page):
        """(row in page, slot, row in slot) for every recorded position of `page`."""
        return [(r, *self.pending[page * TILE + r]) for r in range(TILE) if page * TILE + r in self.pending]

    def flushed(self, page, partial=False):
        """`page` was written to the history; a complete page's positions leave the ring."""
        self.flushed_pages.add(page)
        if not partial:
            for r in range(TILE):
                self.pending.pop(page * TILE + r, None)


class TapBuffers:
    """The tap tensors of one request slot and their fixed-shape device ops (module docstring)."""

    def __init__(self, mesh, layers):
        self.mesh, self.layers = mesh, int(layers)
        self.sealed = False  # set by seal() once a trace is captured: no persistent buffer may be allocated after
        rep = ttnn.ReplicateTensorToMesh(mesh)
        dram = self.dram = ttnn.DRAM_MEMORY_CONFIG

        def zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            host = torch.zeros(shape, dtype=torch.int32 if dtype in (ttnn.uint32, ttnn.int32) else torch.bfloat16)
            return ttnn.from_torch(host, dtype=dtype, layout=layout, device=mesh, mesh_mapper=rep, memory_config=dram)

        n, width = self.layers, self.layers * TAP_WIDTH
        self.masks = []
        for c in range(TILE + 1):
            host = torch.zeros(1, 1, TILE, width, dtype=torch.bfloat16)
            host[:, :, :c] = 1
            self.masks.append(
                ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep,
                                memory_config=dram)
            )
        self.staging = zeros((1, 1, TILE, width))
        self.staging_heads = ttnn.experimental.view(self.staging, [1, n, TILE, TAP_WIDTH])
        self.ring = zeros((RING_SLOTS, n, TILE, TAP_WIDTH))
        self.ring_rows = ttnn.experimental.view(self.ring, [1, 1, RING_SLOTS * n * TILE, TAP_WIDTH])
        self.sel = zeros((1, 1, n * TILE, RING_SLOTS * n * TILE))
        self.history = [zeros((2 * HISTORY_PAGES, 1, TILE, TAP_WIDTH)) for _ in range(n)]
        self.history_rows = [ttnn.experimental.view(h, [1, 1, 2 * HISTORY_PAGES * TILE, TAP_WIDTH]) for h in self.history]
        self.windows = [zeros((1, 1, WINDOW_ROWS, TAP_WIDTH)) for _ in range(n)]
        self.pages = [zeros((1, 1, TILE, TAP_WIDTH)) for _ in range(n)]
        # Page tables (data, so the programs are fixed): the ring slot; the two mirrors of a history write.
        self.slot_table = zeros((1, TABLE_WIDTH), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.mirror_tables = [zeros((1, TABLE_WIDTH), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for _ in range(2)]

        # Device-resident constants (M2 item 1, the declined step's cost): built here, before any trace capture (a
        # buffer allocated after a capture can be overwritten by its replay), so the per-step ring insert and the
        # per-page flush upload nothing. One page-table row per ring slot and per mirror slot of a history page, and
        # the selection matrix of every plain-regime page (any first slot: earlier verifies leave the slot sequence
        # anywhere), 40 x 2 MB per chip. A page of any other shape (a verify's multi-row commits, the page holding the
        # prefill end) still uploads its matrix into `sel`.
        def const_table(pages):
            assert not self.sealed, "TapBuffers allocates its persistent buffers before any trace capture"
            row = torch.zeros(1, TABLE_WIDTH, dtype=torch.int32)
            row[0, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
            return ttnn.from_torch(row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep,
                                   memory_config=dram)

        self.slot_tables = [const_table([slot]) for slot in range(RING_SLOTS)]
        self.page_tables = [[const_table([mirror_slots(g)[m]]) for g in range(HISTORY_PAGES)] for m in range(2)]
        self.plain_sel = {}
        free_before = dram_free_per_chip(mesh)
        for first in range(RING_SLOTS):
            entries = plain_page_entries(first)
            self.plain_sel[entries] = ttnn.from_torch(
                selection_matrix(entries, n).reshape(1, 1, n * TILE, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh, mesh_mapper=rep, memory_config=dram)
        # measured, for the DRAM headroom report (lead, 2026-11-11): what the resident matrices took on each chip
        self.dram_stats = {"free_before_plain_sel": free_before, "free_after_plain_sel": dram_free_per_chip(mesh)}
        # Device start / end of the tensor-args slices (the reader takes the start; the size is fixed).
        self.slice_start = zeros((4,), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.slice_end = zeros((4,), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.hifi4 = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def seal(self):
        """A trace was captured: from now on a persistent allocation here would be a buffer a replay can overwrite
        (the TTSampling-trace clobber). Every persistent buffer is built in __init__; seal() makes that enforced."""
        self.sealed = True

    @staticmethod
    def _refresh(host, dest):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host.contiguous(), dtype=dest.dtype, layout=dest.layout), dest)

    def _table(self, dest, pages):
        row = torch.zeros(1, TABLE_WIDTH, dtype=torch.int32)
        row[0, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
        self._refresh(row, dest)

    # ------------------------------------------------------------------ per step

    def stage(self, launch_taps, rows):
        """staging = launch taps with rows >= `rows` selected to 0 (a select: NaN in those rows cannot leak)."""
        ttnn.where(self.masks[rows], launch_taps, 0.0, output_tensor=self.staging)

    def ring_insert(self, slot):
        ttnn.experimental.paged_fill_cache(self.ring, self.staging_heads, self.slot_tables[slot], batch_idx=0)

    def commit(self, drafter_taps, rows):
        """Rows 0..rows-1 of the staging tile into each drafter.taps[i] ([1,1,block,5376]); rows >= the staged
        count are already 0."""
        for h, dest in enumerate(drafter_taps):
            part = self.staging[:, :, : dest.shape[2], h * TAP_WIDTH : (h + 1) * TAP_WIDTH]
            ttnn.copy(part, dest)

    # ------------------------------------------------------------------ history

    def _write_history(self, layer, part, first_page):
        """Write `part` ([1,1,32r,5376], page-aligned rows of positions 32 * first_page..) to both mirrors."""
        count = part.padded_shape[2] // TILE
        for mirror, table in enumerate(self.mirror_tables):
            if count == 1:  # a flushed page: its constant table, no upload
                table = self.page_tables[mirror][first_page % HISTORY_PAGES]
            else:
                self._table(table, [mirror_slots(first_page + i)[mirror] for i in range(count)])
            ttnn.experimental.paged_fill_cache(self.history[layer], part, table, batch_idx=0)

    def write_prefill(self, parts, first_page):
        """A prefill call's window part: {tap index: [1,1,32r,5376]} of positions 32 * first_page.."""
        for layer, part in parts.items():
            self._write_history(layer, part, first_page)

    def flush(self, page, entries):
        """History page `page` from the ring: entries are TapBook.entries(page). A plain-regime page uses its
        resident selection matrix; any other uploads its own into `sel`."""
        sel = self.plain_sel.get(tuple(entries))
        if sel is None:
            self._refresh(selection_matrix(entries, self.layers).reshape(1, 1, self.layers * TILE, -1), self.sel)
            sel = self.sel
        out = ttnn.matmul(
            sel, self.ring_rows, compute_kernel_config=self.hifi4, dtype=ttnn.bfloat16, memory_config=self.dram
        )
        for h in range(self.layers):
            self._write_history(h, out[:, :, h * TILE : (h + 1) * TILE, :], page)
        del out

    def read_window(self, first_page):
        """The P-page window starting at `first_page` into self.windows (one program for every offset)."""
        start = window_slot(first_page) * TILE
        self._refresh(torch.tensor([0, 0, start, 0], dtype=torch.int32), self.slice_start)
        self._refresh(torch.tensor([1, 1, start + WINDOW_ROWS, TAP_WIDTH], dtype=torch.int32), self.slice_end)
        for src, dest in zip(self.history_rows, self.windows):
            ttnn.slice(src, self.slice_start, self.slice_end, slice_dim=2, num_devices=2, output_tensor=dest)
        return self.windows

    def read_page(self, page):
        """History page `page` of every layer into self.pages, as one [1, n, 32, 5376] tile-row."""
        start = mirror_slots(page)[0] * TILE
        self._refresh(torch.tensor([0, 0, start, 0], dtype=torch.int32), self.slice_start)
        self._refresh(torch.tensor([1, 1, start + TILE, TAP_WIDTH], dtype=torch.int32), self.slice_end)
        for src, dest in zip(self.history_rows, self.pages):
            ttnn.slice(src, self.slice_start, self.slice_end, slice_dim=2, num_devices=2 * HISTORY_PAGES,
                       output_tensor=dest)
        return ttnn.concat(self.pages, dim=1, memory_config=self.dram)

    def insert_page(self, page, rows, slot):
        """The prefill rows < `rows` of history page `page` into ring slot `slot` (the page holding the prefill
        end: its later rows come from decode launches, and its flush must keep these)."""
        heads = self.read_page(page)
        mask = ttnn.experimental.view(self.masks[rows], [1, self.layers, TILE, TAP_WIDTH])
        ttnn.where(mask, heads, 0.0, output_tensor=self.staging_heads)
        self.ring_insert(slot)
        del heads

    def warm(self):
        """Every fixed program once, on zero data, before any trace capture: the ring insert, the flush, the
        window and page reads, and the P history-fill row-count classes (a part of r pages compiles its own
        paged_fill_cache writer). Leaves every buffer zero."""
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        self.stage(self.masks[0], 0)
        self.ring_insert(0)
        self.flush(0, [])
        self.read_window(0)
        self.insert_page(0, 0, 0)
        for count in range(1, HISTORY_PAGES + 1):
            zero = ttnn.from_torch(
                torch.zeros(1, 1, count * TILE, TAP_WIDTH, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=rep, memory_config=self.dram,
            )
            self._write_history(0, zero, 0)
            del zero
        ttnn.synchronize_device(self.mesh)
