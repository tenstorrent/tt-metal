# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The ends of a decode step as programs of this piece (PLAN.md section 46), in place of the ttnn modules of ends.py:

    EndsIn    (root chip only, before layer 0's P1) token ids -> their embedding rows as the iHC site's F tensor, the
              streams zeroed, the coefficient page "post = 1": P1(post=True) of layer 0 then leaves the 4 streams
              equal to the embedding row. kernels/ends_in.cpp on the root's site cores; data movement only.

Token ids live in DRAM (L1 does not survive a prefill pass): 16 uint32 per tensor, words 0..7 the 8 rows. A launch
names the tensor it reads (``ids``) and a second one of forced ids (``force``: word r = id | 1 << 31 takes row r's
token from the host instead of the fed-back one; 0: not forced).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.experimental.hy4_preview.fused.ihc_site import layout as ihc_layout
from models.experimental.hy4_preview.fused.layer_common import transport as tp

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
ROWS = 8
PIECE = 32  # bytes: 16 BF16 values, one face-row half of an [8, 32] row tile
PAGE = 4096  # an fp32 page of the site
ID_WORDS = 16  # a 64-byte page
FORCED = 1 << 31
CB_SCRATCH = 0


def id_tensor(mesh_device, ids=None):
    """A DRAM tensor of 16 uint32 on every chip (the root's is the one that is used): words 0..7 the rows' ids."""
    t = ttnn.from_torch(
        torch.zeros(1, ID_WORDS, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if ids is not None:
        write_ids(t, ids)
    return t


def write_ids(tensor, ids, forced=False, vocab=None):
    """ids: up to 8 ints (None: leave the word 0). forced: mark every given id as a forced one. vocab: the table's
    size: every id is asserted inside it before the write (EndsIn.write_ids passes it: the kernel's clamp of an id
    outside the table is then reached only by a corrupted tensor)."""
    words = torch.zeros(1, ID_WORDS, dtype=torch.int64)
    for r, v in enumerate(ids):
        if v is not None:
            assert 0 <= int(v) < FORCED and (
                vocab is None or int(v) < vocab
            ), f"row {r}: token id {v} outside the vocabulary of {vocab}"
            words[0, r] = int(v) | (FORCED if forced else 0)
    words = torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(words, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), tensor)


def read_ids(tensor, chip=0):
    words = ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip]).to(torch.int64).flatten() & 0xFFFFFFFF
    return [int(v) for v in words[:ROWS]]


class EndsIn:
    """``stack``: the Stack whose site this writes (its F, streams and coefficient page). ``table``: the embedding
    table [vocab, hidden], BF16 values (a torch tensor; any float dtype), or a device tensor made by ``table_tensor``.
    """

    def __init__(self, stack, table):
        self.st, self.mesh = stack, stack.mesh
        site = stack.site
        self.site = site
        K, C = site.K, site.C
        assert K % 2 == 0 and (K // 2) % ROWS == 0, f"{K} site cores: each half must hold the 8 token rows"
        self.tiles = C // 32 // K  # 32-column tiles of an F shard
        self.shard = self.tiles * 2 * ROWS * PIECE  # bytes of a core's F shard
        self.half = K // 2
        self.slice = self.half * self.tiles * 2 * PIECE  # bytes of a token's row one core moves
        self.row_bytes = 2 * C
        assert self.slice * 2 == self.row_bytes
        self.table = table if isinstance(table, ttnn.Tensor) else self.table_tensor(self.mesh, table)
        self.vocab = int(self.table.shape[0])
        assert int(self.table.shape[1]) == C
        # the constants: X_PAGES zero pages, then the coefficient page with post = 1 for every row and stream
        coef = ihc_layout.coef_to_page(torch.zeros(ROWS, 4), torch.ones(ROWS, 4), site.lanes)
        consts = torch.cat([torch.zeros(32, 32 * site.px), coef], dim=1)
        self.consts = ttnn.from_torch(
            consts.reshape(1, 1, 32, -1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.no_force = id_tensor(self.mesh)
        ttnn.synchronize_device(self.mesh)

    @staticmethod
    def table_tensor(mesh_device, table: torch.Tensor):
        """[vocab, hidden] -> BF16 row-major in DRAM: one page per token (its row whole, in one bank)."""
        return ttnn.from_torch(
            table,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def write_ids(self, tensor, ids, forced=False):
        """The host's way to put token ids into an id tensor of this step (fed ids of a first step, forced ids of
        rows that start a request): asserted inside the table before the write."""
        write_ids(tensor, ids, forced=forced, vocab=self.vocab)

    def assignment(self):
        """Per site core k: (token row, byte offset of its slice in the token's table row, the site cores it writes
        to in the order of their hidden columns). Core k moves the columns of the half it is not in."""
        out = []
        for k in range(self.site.K):
            other = 0 if k >= self.half else self.half
            out.append((k % ROWS, other * self.tiles * 2 * PIECE, list(range(other, other + self.half))))
        return out

    def program(self, ids, force=None):
        """(io tensors, MeshProgramDescriptor): the root alone. ids / force: id tensors (id_tensor)."""
        site, st = self.site, self.st
        force = force if force is not None else self.no_force
        streams, coef, f = st.a.streams, site.coef, st.a.f
        ct = list(ttnn.TensorAccessorArgs(self.table).get_compile_time_args())
        ct += list(ttnn.TensorAccessorArgs(ids).get_compile_time_args())
        ct += list(ttnn.TensorAccessorArgs(self.consts).get_compile_time_args())
        ct += [CB_SCRATCH, site.px, self.slice, self.half, self.tiles, self.row_bytes, self.vocab]
        rt = ttnn.RuntimeArgs()
        xy = [n.x | (n.y << 16) for n in site.noc]
        for k, (core, (row, offset, peers)) in enumerate(zip(site.cores, self.assignment())):
            rt[core.x][core.y] = [
                row,
                offset,
                self.table.buffer_address(),
                ids.buffer_address(),
                force.buffer_address(),
                self.consts.buffer_address(),
                streams.buffer_address(),
                coef.buffer_address(),
                f.buffer_address(),
            ] + [xy[p] for p in peers]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "ends_in.cpp"),
            core_ranges=site.grid,
            compile_time_args=ct,
            runtime_args=rt,
            defines=[],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        scratch = ttnn.CBDescriptor(
            total_size=64 + 2 * 4 * ID_WORDS + self.slice,
            core_ranges=site.grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCRATCH, data_format=ttnn.uint32, page_size=64 + 2 * 4 * ID_WORDS + self.slice
                )
            ],
        )
        desc = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[scratch])
        mesh_desc = ttnn.MeshProgramDescriptor()
        root = ttnn.MeshCoordinate(*st.topo.root)
        mesh_desc[ttnn.MeshCoordinateRange(root, root)] = desc
        return [self.table, ids, force, self.consts, streams, coef, f], mesh_desc

    def launch(self, ids, force=None):
        io, desc = self.program(ids, force)
        (ttnn.experimental.prepare_generic_op if tp.is_mock() else ttnn.generic_op)(io, desc)

    def host_f(self, table: torch.Tensor, tokens) -> torch.Tensor:
        """What F must hold after a launch: the rows of ``tokens`` [8, hidden] (BF16 values as fp32)."""
        return table[torch.as_tensor(list(tokens), dtype=torch.long)].float()


# ---------------------------------------------------------------------------------------------- HEAD
from models.experimental.hy4_preview.fused.ihc_site.op import NormOut, SiteRef  # noqa: E402
from models.experimental.hy4_preview.fused.layer_common.sender import Sender, unicast  # noqa: E402
from models.experimental.hy4_preview.fused.moe import layout as ML  # noqa: E402
from models.experimental.hy4_preview.fused.moe import program as MP  # noqa: E402

HIDDEN = ML.HIDDEN
K_TILES = ML.K_TILES  # 192 K tiles of the hidden axis
BANKS = ML.BANKS
BF16_TILE = 2048
KB = 8  # K tiles per streamed block: 16 KB, one NOC packet
ROW_TILE = 2 * 32 * ROWS  # an [8, 32] BF16 row tile
ROW_TILE_F32 = 2 * ROW_TILE
ITEM = K_TILES * BF16_TILE  # one tile column of the head: 393,216 B
VOCAB_PER_CHIP = 3776
COLS = 120  # tile columns of a chip's stream: 118 of its 3,776 ids and 2 of padding
COLS_PER_BANK = COLS // BANKS  # 15
UNITS_PER_BANK = 8  # 7 units of 2 columns and one of 1
UNITS = BANKS * UNITS_PER_BANK
PAIR = 8  # bytes of a (key, id) entry
CB_X, CB_META, CB_CTL, CB_W, CB_OUT = 0, 1, 4, 5, 16
CFG_COLS, CFG_UNIT, CFG_COL0 = 12, 13, 14  # sync words of a unit core (kernels/head_unit.cpp)
# the hub core's arena ranges in the HEAD program (x is the MLP half's hub x buffer at offset 0)
A_LOGITS = 98_304  # 120 FP32 row tiles
A_SLOTS = A_LOGITS + COLS * ROW_TILE_F32  # the units' lists
MODE_CHIP, MODE_HEAD, MODE_ROOT = 0, 1, 2


def head_site(fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor):
    """The checkpoint's hc_head (fn [4, 4 C], base [4], scale [1]) as an entry of the Stack's site table: a site
    whose post rows are zero. Its pre-mix is reference/ihc.py IHCHead; the norm that follows it (the site's gamma)
    is the model's final norm."""
    n = fn.shape[0]
    return (
        torch.cat([fn.float(), torch.zeros_like(fn.float())], dim=0),
        torch.cat([base.float().reshape(n), torch.zeros(n)]),
        torch.cat([scale.float().reshape(1), torch.ones(1)]),
    )


def head_stream_host(shard: torch.Tensor):
    """One chip's shard of the head [ids <= 3,776, hidden] -> its host stream tensor: [15 x 6144, 256] BF16 tiles,
    bank b's columns [32 b, 32 b + 32): the tile columns b * 15 + i of W^T, i = 0..14, one after the other (each
    192 K tiles: one streamed item)."""
    ids, hidden = shard.shape
    assert hidden == HIDDEN and ids <= COLS * 32
    wt = torch.zeros(HIDDEN, COLS * 32, dtype=shard.dtype)
    wt[:, :ids] = shard.T
    stream = (
        wt.reshape(HIDDEN, BANKS, COLS_PER_BANK, 32).permute(2, 0, 1, 3).reshape(COLS_PER_BANK * HIDDEN, BANKS * 32)
    )
    return ttnn.from_torch(stream.contiguous()[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def head_stream(mesh_device, shards):
    """shards: one [ids, hidden] tensor (every chip the same: bring-up) or a function chip -> [ids, hidden]."""
    rows = COLS_PER_BANK * HIDDEN
    if callable(shards):
        hosts = [head_stream_host(shards(c)) for c in range(mesh_device.get_num_devices())]
        host = ttnn.from_host_shards(hosts, mesh_device.shape)
        return ttnn.to_device(host, mesh_device, memory_config=ML.stream_memory_config(rows))
    one = head_stream_host(shards)
    host = ttnn.from_host_shards([one] * mesh_device.get_num_devices(), mesh_device.shape)
    return ttnn.to_device(host, mesh_device, memory_config=ML.stream_memory_config(rows))


def logits_tensor(mesh_device):
    """A kept copy of a step's logits: per chip 120 pages of one FP32 [8, 32] row tile (DRAM)."""
    return ttnn.from_torch(
        torch.zeros(1, 1, COLS, ROW_TILE_F32 // 4),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def read_logits(kept, chips=None) -> torch.Tensor:
    """[chips, 8, 3776] fp32 of the kept tensor (chips: linear indices, default all)."""
    tensors = ttnn.get_device_tensors(kept)
    out = []
    for c in range(len(tensors)) if chips is None else chips:
        tiles = ttnn.to_torch(tensors[c]).float().reshape(COLS, ROW_TILE_F32 // 4)
        out.append(ML.from_row_tiles(tiles)[:, :VOCAB_PER_CHIP])
    return torch.stack(out)


def unit_cores(mesh_device, exclude_columns=(5, 6)):
    """[(core (x, y), bank, index in the bank)] of the 64 unit cores, the same on every chip: each DRAM bank takes
    UNITS_PER_BANK cores in turn, nearest its NOC0 endpoint first (the MoE piece's "side" rule), outside the
    columns of the site, the hub and the sender."""
    grid = mesh_device.compute_with_storage_grid_size()
    anchors = MP.bank_anchors(mesh_device)
    cols = sorted({a.x for a in anchors})

    def cost(b, x, y):
        ax, ay = anchors[b].x, anchors[b].y
        later = [c for c in cols if c > ax]
        segment = range(ax, later[0] if later else grid.x)
        return (0 if x in segment else 1, (y - ay) % grid.y, (x - ax) % grid.x)

    cells = [(x, y) for x in range(grid.x) for y in range(grid.y) if x not in exclude_columns]
    order = {b: sorted(cells, key=lambda c: cost(b, *c)) for b in range(BANKS)}
    used, out = set(), {b: [] for b in range(BANKS)}
    for _ in range(UNITS_PER_BANK):
        for b in range(BANKS):
            c = next(c for c in order[b] if c not in used)
            used.add(c)
            out[b].append(c)
    return [(c, b, j) for b in range(BANKS) for j, c in enumerate(out[b])]


class EndsHead:
    """The HEAD program (module docstring of kernels/head_hub.cpp, head_unit.cpp, head_compute.cpp).

    ``stack``: a Stack with at least one layer (its MLP half's transport, hub core and site are used as they are).
    ``site_index``: the head's entry in the Stack's site table (head_site; its gamma the final norm's weight).
    ``stream``: head_stream(...). ``K``: candidates per row (1: greedy).
    ``virtual``: bring-up on ONE chip with partners written by the host: {"mode": MODE_HEAD | MODE_ROOT, "n1", "idx1",
    "n2", "idx2"}: the hub merges landings of that size; a head's op to the root goes into slot idx2 of ITS OWN
    stage-2 landing (an own copy: no fabric connection)."""

    def write_partners(self, coord, land1=None, land2=None):
        """Bring-up: the host writes partners' lists into the hub's landings of the chip at ``coord`` and their
        arrival counts into its words. land1 / land2: {slot index: (keys [8, K], ids [8, K])}."""
        import numpy as np

        st, mt = self.st, self.mt
        updates = []
        for base, slots in ((self.a_land1, land1 or {}), (self.a_land2, land2 or {})):
            for i, (keys, ids) in slots.items():
                pairs = (
                    torch.stack([keys.to(torch.int64), ids.to(torch.int64)], dim=-1)
                    .reshape(-1)
                    .numpy()
                    .astype(np.uint32)
                )
                updates.append((base + i * self.list_bytes, pairs.tobytes()))
        st.a.write_arena({(tuple(coord), self.hub): updates})
        gx = self.grid[0]
        shards = []
        for chip in range(self.mesh.get_num_devices()):
            words = torch.zeros(self.grid[0] * self.grid[1], tp.WORDS, dtype=torch.int32)
            if chip == self.topo.chip(coord):
                words[self.hub[1] * gx + self.hub[0], tp.W_S1] = len(land1 or {})
                words[self.hub[1] * gx + self.hub[0], tp.W_S2] = len(land2 or {})
            shards.append(ttnn.from_torch(words, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT))
        ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(shards, self.mesh.shape), mt.words)
        ttnn.synchronize_device(self.mesh)

    def read_canary(self) -> dict:
        """{coord: (checks, failed, code of the last failure)} of every hub's canary on the landings
        (kernels/head_hub.cpp): 1 the x count was not 1, 2 / 4 a stage count was not what the step delivers, 3 / 5 a
        stage word was not zero when the hub ended."""
        ttnn.synchronize_device(self.mesh)
        tensors = ttnn.get_device_tensors(self.sync)
        row = self.hub[1] * self.grid[0] + self.hub[0]
        out = {}
        for coord in self.topo.chips:
            words = ttnn.to_torch(tensors[self.topo.chip(coord)]).to(torch.int64).reshape(-1, 16)[row] & 0xFFFFFFFF
            out[coord] = (int(words[8]), int(words[9]), int(words[10]))
        return out

    def canary_summary(self) -> dict:
        rows = list(self.read_canary().values())
        return {
            "checks": sum(r[0] for r in rows),
            "failed": sum(r[1] for r in rows),
            "codes": sorted({r[2] for r in rows if r[1]}),
        }

    def read_parsed(self, coord):
        """After a parse-only run: (what the chip's sender wrote to its page, what the host built)."""
        want = self.parsed[tuple(coord)]
        page = (
            ttnn.to_torch(ttnn.get_device_tensors(self.st.mparse)[self.topo.chip(coord)]).to(torch.int64) & 0xFFFFFFFF
        )
        return [int(v) for v in page.flatten()[: len(want)]], list(want)

    def __init__(self, stack, site_index, stream, K=1, virtual=None, wait_limit=None):
        self.st, self.mesh = stack, stack.mesh
        st = stack
        assert st.mt is not None and st.hub is not None, "add a layer first: the HEAD uses the MLP half's transport"
        assert 1 <= K <= 16
        self.K, self.site_index, self.stream, self.virtual = int(K), int(site_index), stream, virtual
        self.list_bytes = ROWS * K * PAIR
        self.topo, self.mt, self.hub = st.topo, st.mt, tuple(st.hub)
        self.wait_limit = int(wait_limit or st.wait_limit)
        from models.experimental.hy4_preview.fused.layer_mlp import program as M
        from models.experimental.hy4_preview.fused.layer_attn import program as A

        self.sender_core = M.SENDER
        rows, cols = self.topo.mesh_shape
        # arena ranges on the hub core
        self.a_slots = A_SLOTS
        self.a_land1 = self.a_slots + UNITS * self.list_bytes
        self.a_land2 = self.a_land1 + cols * self.list_bytes
        self.a_scratch = self.a_land2 + rows * self.list_bytes
        self.a_end = self.a_scratch + 3 * self.list_bytes + 64
        assert (
            self.a_end <= A.ARENA_BYTES and st.alias_hub and self.mt.x_address == st.a.at(0)
        ), "the hub's arena ranges"
        self.units = unit_cores(self.mesh)
        taken = {(c.x, c.y) for c in st.site.cores} | {self.hub, tuple(self.sender_core)}
        assert not taken & {c for c, _, _ in self.units}, "a unit core has another role"
        grid = self.mesh.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        self.grid_crs = tp.grid_set(self.mesh)
        self.unit_crs = tp.core_set([c for c, _, _ in self.units])
        self.sync = tp.core_buffer(self.mesh, self.grid_crs, 64, ttnn.uint32, width=16)
        self._write_sync()
        self._like = logits_tensor(self.mesh)  # (its accessor arguments stand in when a launch keeps no logits)
        self.parsed = {}
        ttnn.synchronize_device(self.mesh)

    # ------------------------------------------------------------------------------------------ state
    def sync_words(self):
        """[cores, 16]: 0..7 the hand-over words (zero), 8..11 the streamer's constants (moe_common.hpp CFG_*),
        12..14 the unit's (its columns, its index, its first tile column)."""
        gx = self.grid[0]
        words = torch.zeros(self.grid[0] * self.grid[1], 16, dtype=torch.int64)
        address = self.stream.buffer_address()
        assert address % 32 == 0
        vcs = []
        for u, ((x, y), bank, j) in enumerate(self.units):
            vc = (bank + j) & 0x3
            for (ox, oy), ovc in vcs:
                if oy == y and ovc == vc:
                    vc = (vc + 1) & 0x3
            vcs.append(((x, y), vc))
            n = 2 if j < UNITS_PER_BANK - 1 else 1
            words[y * gx + x, 8:15] = torch.tensor([bank, j, vc, address >> 5, n, u, bank * COLS_PER_BANK + 2 * j])
        return words

    def _write_sync(self):
        host = ttnn.from_torch(
            self.sync_words().to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, self.sync)

    def reset(self):
        """After a run that left a status: the units' hand-over words to zero (the transport's words: Stack.reset)."""
        self._write_sync()
        ttnn.synchronize_device(self.mesh)

    def feed_audit(self):
        """Per unit and per column of it, from the numbers the kernels are built with: tiles the compute kernel
        waits for or pops against tiles its producers push. [(unit, bank, index, column of the unit, tile column,
        {buffer: (pushed, consumed)})]. x is HELD: pushed once by the unit's BRISC (192 row tiles), waited for once
        by head_compute.cpp and never popped; every column's matmuls index the same 192 tiles."""
        item_blocks, block_tiles = K_TILES // KB, KB  # the streamer's compile-time arguments 5 and 6
        rows = []
        words = self.sync_words()
        for u, ((x, y), bank, j) in enumerate(self.units):
            w = words[y * self.grid[0] + x]
            cols = int(w[CFG_COLS])
            assert int(w[CFG_UNIT]) == u and int(w[9]) == j and int(w[8]) == bank
            for c in range(cols):
                pushed_w = item_blocks * block_tiles  # one item of the streamer = this column
                popped_w = (K_TILES // KB) * KB  # head_compute.cpp: x_tiles / kb rounds of kb
                rows.append(
                    (
                        u,
                        bank,
                        j,
                        c,
                        int(w[CFG_COL0]) + c,
                        {"x": (K_TILES, K_TILES), "w": (pushed_w, popped_w), "out": (1, 1)},
                    )
                )
                assert pushed_w == popped_w == K_TILES
        return rows

    def columns(self):
        """Tile column -> unit index (which unit computes it)."""
        out = {}
        for u, (_, bank, j) in enumerate(self.units):
            for c in range(2 if j < UNITS_PER_BANK - 1 else 1):
                out[bank * COLS_PER_BANK + 2 * j + c] = u
        assert sorted(out) == list(range(COLS))
        return out

    # ------------------------------------------------------------------------------------------ the program
    def _noc(self, core):
        return self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(*core))

    def _roles(self, coord):
        """(mode, n1, idx1, n2, idx2) of the chip at ``coord``."""
        topo = self.topo
        if self.virtual is not None:
            v = self.virtual
            return v["mode"], v["n1"], v["idx1"], v["n2"], v["idx2"]
        mode = MODE_ROOT if coord == topo.root else (MODE_HEAD if topo.is_head(coord) else MODE_CHIP)
        return mode, topo.n1(coord), topo.idx1(coord), topo.n2, topo.idx2(coord)

    def _chip_program(self, coord, ids_out, kept):
        st, mt, a, topo = self.st, self.mt, self.st.a, self.topo
        coord = tuple(coord)
        root = coord == topo.root
        mode, n1, idx1, n2, idx2 = self._roles(coord)
        hub_noc = self._noc(self.hub)
        lo, hi = self._noc((0, 0)), self._noc((self.grid[0] - 1, self.grid[1] - 1))
        mc = [lo.x, lo.y, hi.x, hi.y, self.grid[0] * self.grid[1] - 1]
        defines = [
            ("ROWS", str(ROWS)),
            ("HY4_SYNC", f"{self.sync.buffer_address()}u"),
            ("HY4_SPIN_LIMIT", f"{self.wait_limit}u"),
        ]
        dm = ttnn.DataMovementConfigDescriptor
        writer_cfg = dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
        reader_cfg = dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
        hifi4 = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        includes = [str(MP.KERNEL_DIR), str(tp.KERNEL_DIR)]

        def cb(index, cores, dtype, page, pages, row_tile=False):
            kwargs = dict(buffer_index=index, data_format=dtype, page_size=page)
            if row_tile:
                kwargs["tile"] = ttnn.TileDescriptor(ttnn.Tile([ROWS, 32]))
            return ttnn.CBDescriptor(
                total_size=page * pages, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(**kwargs)]
            )

        ctl_bytes = max(1024, 64 * -(-(16 + self.list_bytes) // 64))
        cbs = [
            cb(CB_X, self.grid_crs, ttnn.bfloat16, ROW_TILE, K_TILES, row_tile=True),  # the same address on every core
            cb(CB_META, self.unit_crs, ttnn.uint32, 64, 1),
            cb(CB_CTL, self.unit_crs, ttnn.uint32, ctl_bytes, 1),
            cb(CB_W, self.unit_crs, ttnn.bfloat16, BF16_TILE, 4 * KB),
            cb(CB_OUT, self.unit_crs, ttnn.float32, ROW_TILE_F32, 2, row_tile=True),
        ]
        hx, hy = self.hub
        sx, sy = self.sender_core
        own_slot2 = self.virtual is not None and mode == MODE_HEAD
        hub_args = [
            mt.x_address,
            a.at(self.a_slots),
            a.at(self.a_scratch),
            topo.chip(coord) * VOCAB_PER_CHIP,
            mode,
            a.at(self.a_land1),
            n1,
            idx1,
            a.at(self.a_land2),
            n2,
            idx2,
            mt.noc_word(self.sender_core),
            mt.staging_address,
            mt.word_address(tp.W_READY),
            Sender.bit(0),
            ids_out.buffer_address(),
        ]
        if own_slot2:  # (bring-up: the one chip as a head: its list is the sender's second op, see below)
            hub_args[12], hub_args[14] = mt.staging_address + mt.payload_bytes, Sender.bit(1)
        hub_rt = ttnn.RuntimeArgs()
        hub_rt[hx][hy] = hub_args
        hub_ct = (
            [CB_X, K_TILES * ROW_TILE]
            + mc
            + [UNITS, self.K]
            + list(ttnn.TensorAccessorArgs(ids_out).get_compile_time_args())
        )
        unit_rt = ttnn.RuntimeArgs()
        for (x, y), _, _ in self.units:
            unit_rt[x][y] = [
                a.at(A_LOGITS),
                a.at(self.a_slots),
                mt.word_address(tp.W_GATHER),
                kept.buffer_address() if kept is not None else 0,
            ]
        keep_like = kept if kept is not None else self._keep_like()
        unit_ct = [CB_X, CB_CTL, CB_OUT, K_TILES, hub_noc.x, hub_noc.y, self.K, VOCAB_PER_CHIP]
        unit_ct += list(ttnn.TensorAccessorArgs(keep_like).get_compile_time_args())
        reader_ct = [CB_W, CB_CTL, CB_META, 1, ITEM, K_TILES // KB, KB, KB * BF16_TILE, 1, 4, 2, ITEM, 1, 0, 2 * ITEM]

        def kernel(source, cores, ct, rt, config, extra=()):
            names = {k for k, _ in extra}
            return ttnn.KernelDescriptor(
                kernel_source=str(source),
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
                defines=[d for d in defines if d[0] not in names] + list(extra),
                compiler_include_paths=includes,
            )

        kernels = [
            kernel(KERNEL_DIR / "head_hub.cpp", tp.core_set([self.hub]), hub_ct, hub_rt, writer_cfg, mt.defines),
            kernel(
                MP.KERNEL_DIR / "stream_reader.cpp",
                self.unit_crs,
                reader_ct,
                ttnn.RuntimeArgs(),
                reader_cfg,
                [("STREAM_FIXED", "1")],
            ),
            # (a unit waits for its release through the site's pass, X1' and the hub's own bounded wait: four times
            # the hub's bound, so that a unit never gives up before a hub that is still going to release it)
            kernel(
                KERNEL_DIR / "head_unit.cpp",
                self.unit_crs,
                unit_ct,
                unit_rt,
                writer_cfg,
                [("HY4_SPIN_LIMIT", f"{4 * self.wait_limit}u")],
            ),
            kernel(
                KERNEL_DIR / "head_compute.cpp",
                self.unit_crs,
                [CB_X, CB_W, CB_OUT, K_TILES, KB],
                ttnn.RuntimeArgs(),
                hifi4,
            ),
        ]
        desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        io = [self.stream, self.sync, ids_out] + ([kept] if kept is not None else [])
        if root:
            sender_noc, staging, ready, bit = mt.x_source_args()
            norm = NormOut(
                gamma=None,
                xn=a.xn,
                sender=sender_noc,
                staging=staging,
                ready=ready,
                ready_bit=bit,
                gather=mt.word_address(tp.W_GATHER),
                status=mt.word_address(tp.W_STATUS),
                wait_limit=mt.wait_limit,
            )
            s_io, s_desc = st.site.program(
                a.streams, f=a.f, params=SiteRef(st.table, self.site_index), y=a.y, norm_out=norm
            )
            desc.kernels = list(desc.kernels) + list(s_desc.kernels)
            desc.cbs = list(desc.cbs) + list(s_desc.cbs)
            desc.semaphores = list(desc.semaphores) + list(s_desc.semaphores)
            io = io + s_io
        # the chip's sender: the root's broadcast of x; another chip's one op with its list
        sender = Sender(
            self.mesh,
            coord,
            self.sender_core,
            mt.word_address(tp.W_READY),
            self.wait_limit,
            parse_page=tp.address(st.mparse) if st.parse_only else None,
        )
        chunk = min(self.list_bytes, mt.chunk)
        if root:
            mt.broadcast_op(sender)
            if own_slot2:
                # bring-up: the one chip as a head: after the broadcast (its own copy of x) a second op takes its
                # list into slot idx2 of its OWN stage-2 landing (staging behind the broadcast's)
                op = sender.op(
                    mt.staging_address + mt.payload_bytes,
                    self.list_bytes,
                    chunk,
                    self.hub,
                    a.at(self.a_land2) + idx2 * self.list_bytes,
                    mt.word_address(tp.W_S2),
                    [],
                    own=True,
                )
                assert op == 1
        else:
            head = topo.is_head(coord)
            target = topo.root if head else topo.head(coord)
            slot = (
                (a.at(self.a_land2) + topo.idx2(coord) * self.list_bytes)
                if head
                else (a.at(self.a_land1) + topo.idx1(coord) * self.list_bytes)
            )
            sender.op(
                mt.staging_address,
                self.list_bytes,
                chunk,
                self.hub,
                slot,
                mt.word_address(tp.W_S2 if head else tp.W_S1),
                [unicast(target)],
                own=False,
            )
        sender.finish(desc)
        self.parsed[coord] = sender.parsed
        return desc, io

    def _keep_like(self):
        return self._like

    def program(self, ids_out, kept=None):
        """(io tensors, MeshProgramDescriptor). ids_out: the id tensor the root writes the rows' tokens to; kept: a
        logits_tensor the units also write their tiles to (None: the hub's L1 range only)."""
        mesh_desc = ttnn.MeshProgramDescriptor()
        io = list(self.mt.io())
        for coord in self.topo.chips:
            desc, chip_io = self._chip_program(coord, ids_out, kept)
            io += chip_io
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
        unique = []
        for t in io:
            if isinstance(t, ttnn.Tensor) and not any(t is u for u in unique):
                unique.append(t)
        return unique, mesh_desc

    def launch(self, ids_out, kept=None):
        io, desc = self.program(ids_out, kept)
        (ttnn.experimental.prepare_generic_op if tp.is_mock() else ttnn.generic_op)(io, desc)

    # ------------------------------------------------------------------------------------------ host side
    def read_hub(self, coord, image=None):
        """A chip's hub ranges after a run: x [8, 6144] as it arrived, the logits [8, 3776] the units wrote, the units'
        lists [units, 8, K, 2], the scratch lists [3, 8, K, 2] (chip, row, all) as (key, id) int64."""
        import numpy as np

        a = self.st.a
        image = a.read_arena(coord) if image is None else image
        raw = lambda off, n: a.arena_bytes(image, self.hub, off, n)
        from models.experimental.hy4_preview.fused.layer_attn.program import bytes_bf16, bytes_f32

        x = ML.from_row_tiles(bytes_bf16(raw(0, K_TILES * ROW_TILE), (K_TILES, ROW_TILE // 2)))
        logits = ML.from_row_tiles(bytes_f32(raw(A_LOGITS, COLS * ROW_TILE_F32)).reshape(COLS, ROW_TILE_F32 // 4))[
            :, :VOCAB_PER_CHIP
        ]
        pairs = lambda off, n: torch.from_numpy(
            np.frombuffer(raw(off, n * self.list_bytes), dtype=np.uint32).astype(np.int64)
        ).reshape(n, ROWS, self.K, 2)
        return {"x": x, "logits": logits, "slots": pairs(self.a_slots, UNITS), "lists": pairs(self.a_scratch, 3)}


def sort_key(values: torch.Tensor) -> torch.Tensor:
    """The kernels' sort key of fp32 values (int64): larger key <=> larger value."""
    bits = values.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(bits >> 31 == 1, (~bits) & 0xFFFFFFFF, bits | 0x80000000)


def host_merge(lists, K: int = 1):
    """lists: [(keys [8, k], ids [8, k]), ...] -> (keys [8, K], ids [8, K]): per row the K entries with the largest
    keys, of equal keys the lower id first (kernels/head_hub.cpp merge)."""
    keys = torch.cat([k.to(torch.int64) for k, _ in lists], dim=1)
    ids = torch.cat([i.to(torch.int64) for _, i in lists], dim=1)
    out_k, out_i = [], []
    for r in range(keys.shape[0]):
        order = sorted(range(keys.shape[1]), key=lambda n: (-int(keys[r, n]), int(ids[r, n])))[:K]
        out_k.append(keys[r, order])
        out_i.append(ids[r, order])
    return torch.stack(out_k), torch.stack(out_i)


def host_best(logits: torch.Tensor, K: int = 1):
    """logits [rows, ids] -> (keys [rows, K], ids [rows, K]): the K largest, of equal values the lower id first."""
    keys = sort_key(logits)
    order = torch.argsort(
        keys * (logits.shape[-1] + 1) + (logits.shape[-1] - torch.arange(logits.shape[-1])), dim=-1, descending=True
    )[:, :K]
    return torch.gather(keys, 1, order), order
