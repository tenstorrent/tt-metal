# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The cross-chip log-sum-exp merge of the attention partials (step X4 of the decode layer), with the sink as one
more peer. Piece 6 of fused/README.md.

Input, per chip: what fused/sparse_mla produces (its ``out`` tensor, layout.py "Partial contract"): per user u and
head group g (8 heads) the page ``u * 8 + g`` = O [8 heads, 512] unnormalised | max, sum per head. Output, per
chip: the merged, normalised attention output in latent space for the heads THAT CHIP OWNS in the post stage,
``ctx`` [8 users, 512] per owned head as 16 tiles of [8 users, 32] (the row tiles fused/projections reads kv_b's
value side with), resident in L1 on one core of the chip.

Who needs which heads, and what therefore moves
-----------------------------------------------
fused/projections/layout.py: chip c = 4 * row + col owns heads 2c and 2c + 1, so head group g (heads 8g .. 8g + 7,
what one attention core holds) belongs to the four chips of mesh row g, two heads per column. A chip's partial of
a head it does not own has to leave the chip once (62 of 64 heads: 508 KB of the 524 KB per chip): that is the
floor for any merge that leaves ctx where the post stage runs, and it is what moves here.

    column stage   chip (r, c) sends its partial of head group g to chip (g, c), for the 7 groups g != r: 56
                   packets of 8.7 KB over the north / south links (a personalised all-to-all on the column's ring,
                   kernels/lse_sender.cpp). Chip (g, c) merges the 8 partials of group g of its column.
    row stage      the 4 chips of row g all-gather their column results (8 users x 8.7 KB each, the collective's
                   multicast over the east / west links) and merge them: 3 more merges.
    sink + divide  one last merge with the sink peer (max = sink / scale, sum = 1, O = 0), normalised.
    output         every chip writes the rows of its 2 heads into its ctx tiles.

Bytes per chip: 487 KB out and 487 KB in on the column stage, 70 KB injected twice (two multicast walks) and
209 KB in on the row stage. The row stage sends all 8 heads of the group where 2 per neighbour would do (51 KB):
the east / west links carry nothing else during X4, and whole partials need no row slicing before the send.
Sending the scalars first and scaling O before it travels would turn the merges into plain sums but adds a
fabric pass; merging at the 2-head owner directly needs 31 unicast routes and a chain of 32 merges per core.

The merges are sdpa_tail (api/compute/experimental/sdpa.h), the LLK fused/sparse_mla merges its S-blocks with:
per user a chain of 7 + 3 + 1 pairwise merges on one merge core, in slot order, so every chip of a row computes
the same bits from the same landing.

What can start before the attention has finished: the fabric connections of both senders open at program start;
the column sender can send the partial of (u, g) as soon as attention core (u, g) has it (the standalone form
reads all 56 from DRAM first); a merge core starts its chain when its column landing is complete; the row stage
needs the column stage. Nothing here waits for the last user of the chip except through the landing counters.

As a piece of a layer program
-----------------------------
cores       per chip: one column sender core (BRISC: kernels/lse_sender.cpp), ``users`` merge cores in a logical
            row (NCRISC: lse_merge_dm.cpp, TRISCs: lse_merge_compute.cpp), one row sender core (the collective's
            sender), and the core that holds ctx (any core of the post stage).
fabric      the column sender holds link ``link`` north and south, the row sender link ``link`` east and west, for
            the whole program (default link 0, the one fused/collective and the top-k merge are proven on). The
            only board run of this program so far used link 1, which the MOCK lists
            (ttnn.get_forwarding_link_indices: [0, 1]) and nothing had verified on the board, and it hit the
            operation timeout (log mk6_lse_pairv): whether a second link per direction can be used, so that this
            merge and the top-k merge's sender share a program, is open.
semaphores  none of its own; the collective's Fragment takes one program semaphore (ready word) on the row
            sender and the merge cores, and two per fabric direction on each sender core. All waits on other
            cores are on persistent L1 counters (collective.hpp), bounded.
CBs         merge cores: 10 ids (running partial 1 + 16 tiles, send 1 + 16, final 16, the sink and the four
            landing parities laid over L1 tensors). Column sender: a staging CB of 7 x users x 8.7 KB (487 KB on
            that core only) and a 64-byte scratch.
L1          landings: column 2 parities x 8 slots x 8.7 KB = 139 KB per merge core, row 2 x 4 x 8.7 KB = 70 KB;
            row staging 70 KB on the row sender core; sink 10 KB, state 4 KB, ctx 16 KB. L1 buffers take their
            bytes on every core of the chip: 310 KB in all.
inputs      attention core (u, g) -> the column sender's staging slot (g != row) or merge core u's running-partial
            CB (g == row), where the standalone form reads DRAM; the layer's sink per head group -> the sink
            tile (a per-layer constant, 512 bytes per merge core).

Precision: BF16 partials in, BF16 pack after every merge (11 in a chain), FP32 exp and sums inside a merge;
``fidelity`` is the FPU fidelity of the O x weight multiplies (HiFi4 by default).
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl
from models.experimental.hy4_preview.fused.merges.mesh_site import MeshSite

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
TINY_ROWS = 8
TINY_VALUES = TINY_ROWS * TILE  # 256 BF16 values, 512 bytes
TINY_BYTES = 2 * TINY_VALUES
LATENT = 512
O_TILES = LATENT // TILE
PART_TILES = 1 + O_TILES
PART_BYTES = PART_TILES * TINY_BYTES
HEADS, HEAD_GROUPS, GROUP = 64, 8, 8
PAGE_VALUES = PART_TILES * TINY_VALUES + 32  # sparse_mla's output page: O | max, sum | status
EMPTY_MAX = struct.unpack("<f", struct.pack("<I", 0xF149 << 16))[0]  # sparse_mla layout.EMPTY_MAX
(CB_ACC_O, CB_ACC_MS, CB_SEND_MS, CB_SEND_O, CB_SINK, CB_FINAL, CB_COL0, CB_COL1, CB_ROW0, CB_ROW1) = range(10)
CB_STAGING, CB_SCRATCH = 0, 1  # on the column sender core
SENDER_CORE, MERGE_CORE, ROW_SENDER_CORE, CTX_CORE = (0, 1), (1, 1), (9, 1), (10, 1)
FIDELITY = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi3": ttnn.MathFidelity.HiFi3,
    "hifi4": ttnn.MathFidelity.HiFi4,
}
STAMPS = ("start", "local", "col_landed", "col_merged", "row_handed", "row_landed", "final", "written")


# ---------------------------------------------------------------------------------------------
# Host side: tile memory, pages, the golden
# ---------------------------------------------------------------------------------------------
def _tile_word():
    r = torch.arange(TILE)[:, None].expand(TILE, TILE)
    c = torch.arange(TILE)[None, :].expand(TILE, TILE)
    return (((((r >> 4) << 1) | (c >> 4)) << 8) | ((r & 15) << 4) | (c & 15)).reshape(-1)


def raw_to_tiles(raw: torch.Tensor) -> torch.Tensor:
    """[..., tiles * 1024] values in memory order -> [..., tiles * 32, 32] as a 32 x 32 tile tensor holds them."""
    tiles = raw.shape[-1] // (TILE * TILE)
    return raw.reshape(*raw.shape[:-1], tiles, TILE * TILE)[..., _tile_word()].reshape(
        *raw.shape[:-1], tiles * TILE, TILE
    )


def tiles_to_raw(tiles: torch.Tensor) -> torch.Tensor:
    """The inverse of raw_to_tiles: [..., tiles * 32, 32] -> [..., tiles * 1024] in memory order."""
    n = tiles.shape[-2] // TILE
    flat = tiles.reshape(*tiles.shape[:-2], n, TILE * TILE)
    raw = torch.empty_like(flat)
    raw[..., _tile_word()] = flat
    return raw.reshape(*tiles.shape[:-2], n * TILE * TILE)


def pack_tiny(x: torch.Tensor) -> torch.Tensor:
    """[..., 8, W] -> [..., W / 32 tiles * 256] in tiny-tile memory order (two [8, 16] faces per tile)."""
    w = x.shape[-1]
    return (
        x.reshape(*x.shape[:-2], TINY_ROWS, w // TILE, 2, 16)
        .transpose(-4, -3)
        .transpose(-3, -2)
        .reshape(*x.shape[:-2], -1)
    )


def unpack_tiny(flat: torch.Tensor) -> torch.Tensor:
    """The inverse of pack_tiny: [..., tiles * 256] -> [..., 8, tiles * 32]."""
    tiles = flat.shape[-1] // TINY_VALUES
    x = flat.reshape(*flat.shape[:-1], tiles, 2, TINY_ROWS, 16)
    return x.transpose(-3, -2).transpose(-4, -3).reshape(*flat.shape[:-1], TINY_ROWS, tiles * TILE)


def partial_pages(o: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """O [..., users, 64, 512], max / sum [..., users, 64] -> sparse_mla's output pages
    [..., users * 8, PAGE_VALUES]: page u * 8 + g = O tiles | max, sum tile | status (zeros)."""
    lead, users = o.shape[:-3], o.shape[-3]
    og = o.reshape(*lead, users, HEAD_GROUPS, GROUP, LATENT)
    ms = torch.zeros(*lead, users, HEAD_GROUPS, GROUP, TILE)
    ms[..., 0] = m.reshape(*lead, users, HEAD_GROUPS, GROUP)
    ms[..., 1] = s.reshape(*lead, users, HEAD_GROUPS, GROUP)
    pages = torch.cat([pack_tiny(og), pack_tiny(ms), torch.zeros(*lead, users, HEAD_GROUPS, 32)], dim=-1)
    return pages.reshape(*lead, users * HEAD_GROUPS, PAGE_VALUES)


def merge_golden(o, m, s, sink, scale: float) -> torch.Tensor:
    """sparse_mla layout.merge over peers on dim 0: O [peers, ..., 512], m, s [peers, ...], sink [...] (a logit)
    -> ctx [..., 512] in float64."""
    o, m, s, sink = o.double(), m.double(), s.double(), sink.double()
    top = torch.maximum(m.max(dim=0).values, sink / scale)
    w = torch.exp(scale * (m - top))
    total = (w * s).sum(0) + torch.exp(sink - scale * top)
    return (w[..., None] * o).sum(0) / total[..., None]


# ---------------------------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------------------------
class LseMerge:
    """The merge on the rectangle ``rows`` x ``cols`` of the mesh (default all of it). The chip at (rows[i],
    cols[j]) owns head group i and, of it, heads ``8 i + j * heads_out .. + heads_out - 1`` with
    ``heads_out = 8 / len(cols)``: on the whole 8x4 mesh chip c owns heads 2c and 2c + 1, as
    fused/projections has it. Smaller rectangles merge the head groups 0 .. len(rows) - 1 only.

    ``partials``: sparse_mla's output tensor per chip (bf16 ROW_MAJOR [1, 1, users * 8, PAGE_VALUES] in DRAM);
    made here when not given (``set_partials`` fills it).
    """

    def __init__(
        self,
        mesh_device,
        rows=None,
        cols=None,
        users: int = 8,
        scale: float = 1.0 / 16.0,
        fidelity: str = "hifi4",
        sender_core=SENDER_CORE,
        merge_core=MERGE_CORE,
        row_sender_core=ROW_SENDER_CORE,
        ctx_core=CTX_CORE,
        link: int = 0,
        partials=None,
        wait_limit: int = cl.DEFAULT_WAIT_LIMIT,
        timing: bool = False,
        dry_sender_groups=(),
    ):
        """``dry_sender_groups``: with a rectangle of one row (no column stage), run the column sender all the
        same, without its fabric calls, as if it had to send these head groups: its DRAM reads and staging on a
        chip before it ever opens a connection (``read()["dry"]``)."""
        mesh_rows, mesh_cols = tuple(mesh_device.shape)
        self.dry_sender_groups = list(dry_sender_groups)
        self.mesh_device, self.users, self.scale, self.fidelity = mesh_device, users, scale, fidelity
        self.rows = list(range(mesh_rows)) if rows is None else list(rows)
        self.cols = list(range(mesh_cols)) if cols is None else list(cols)
        self.R, self.C = len(self.rows), len(self.cols)
        assert 1 <= self.R <= HEAD_GROUPS and self.C in (1, 2, 4, 8)  # 1 x 1: one chip, its partial and the sink
        self.heads_out = GROUP // self.C
        self.coords = [(r, c) for r in self.rows for c in self.cols]
        self.device_index = {(r, c): r * mesh_cols + c for r in range(mesh_rows) for c in range(mesh_cols)}
        self.sender_core, self.merge_core = tuple(sender_core), tuple(merge_core)
        self.row_sender_core, self.ctx_core = tuple(row_sender_core), tuple(ctx_core)
        self.merge_cores = cl.row_cores(self.merge_core, users)
        self.link, self.wait_limit, self.timing = link, wait_limit, timing
        placed = [self.sender_core, self.row_sender_core, self.ctx_core] + self.merge_cores
        assert len(set(placed)) == len(placed), "the cores of the merge must be distinct"

        self.col_site = self.row_site = None
        if self.R > 1:
            self.col_site = MeshSite(
                mesh_device,
                [[(r, c) for r in self.rows] for c in self.cols],
                PART_BYTES,
                parts=users,
                landing_core=self.merge_core,
                sender_core=self.sender_core,
                name="lse_col",
                staging=False,
            )
        if self.C > 1:
            self.row_site = MeshSite(
                mesh_device,
                [[(r, c) for c in self.cols] for r in self.rows],
                PART_BYTES,
                parts=users,
                landing_core=self.merge_core,
                sender_core=self.row_sender_core,
                name="lse_row",
            )
        self.sites = [s for s in (self.col_site, self.row_site) if s is not None]

        chips = mesh_device.get_num_devices()
        self.partials = partials
        if partials is None:
            self.partials = ttnn.from_torch(
                torch.zeros(chips, 1, users * HEAD_GROUPS, PAGE_VALUES),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )
        self.sink_tiles = -(-PART_BYTES // 2048)
        self.sink = cl.tile_column(mesh_device, self.merge_core, self.sink_tiles, ttnn.bfloat16, cores=users)
        self.ctx_tiles = self.heads_out * O_TILES * TINY_BYTES // 2048
        self.ctx = cl.tile_column(mesh_device, self.ctx_core, self.ctx_tiles, ttnn.bfloat16)
        self.state = cl.tile_column(mesh_device, self.merge_core, 1, ttnn.uint32, cores=users)
        ttnn.synchronize_device(mesh_device)
        self._descriptors = {}

    # inputs -------------------------------------------------------------------------------------
    def heads_of(self, slot_row: int, slot_col: int) -> list:
        """The heads the chip at (rows[slot_row], cols[slot_col]) owns."""
        return [GROUP * slot_row + slot_col * self.heads_out + j for j in range(self.heads_out)]

    def set_partials(self, o: torch.Tensor, m: torch.Tensor, s: torch.Tensor):
        """o [R * C, users, 64, 512], m, s [R * C, users, 64] (BF16 values), member chips in row-major order of
        the rectangle. A host write."""
        chips = self.mesh_device.get_num_devices()
        pages = torch.zeros(chips, 1, self.users * HEAD_GROUPS, PAGE_VALUES)
        for k, coord in enumerate(self.coords):
            pages[self.device_index[coord], 0] = partial_pages(o[k], m[k], s[k])
        host = ttnn.from_torch(
            pages,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
        )
        ttnn.synchronize_device(self.mesh_device)
        ttnn.copy_host_to_device_tensor(host, self.partials)

    def set_sink(self, sink: torch.Tensor):
        """sink [64]: the layer's per-head sink logit. Every merge core of a chip gets the peer of the chip's
        head group: max = BF16(sink / scale), sum = 1, O = 0. A host write."""
        chips = self.mesh_device.get_num_devices()
        peer = (sink.float() / self.scale).to(torch.bfloat16).float()
        raw = torch.zeros(chips, self.users, self.sink_tiles * TILE * TILE)
        for i, r in enumerate(self.rows):
            ms = torch.zeros(GROUP, TILE)
            ms[:, 0], ms[:, 1] = peer[GROUP * i : GROUP * (i + 1)], 1.0
            for c in self.cols:
                raw[self.device_index[(r, c)], :, :TINY_VALUES] = pack_tiny(ms)
        cl.write_tile_column(self.mesh_device, self.sink, raw_to_tiles(raw).reshape(chips, -1, TILE))
        self.sink_peer = peer

    def reset(self):
        for site in self.sites:
            site.reset()

    # program ------------------------------------------------------------------------------------
    def _tiny_cb(self, cores, index, pages):
        tile = ttnn.Tile((TINY_ROWS, TILE))
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=index, data_format=ttnn.bfloat16, page_size=TINY_BYTES, tile=ttnn.TileDescriptor(tile)
        )
        return ttnn.CBDescriptor(total_size=TINY_BYTES * pages, core_ranges=cores, format_descriptors=[fmt])

    def _tiny_cb_over(self, index, tensor, offset, size):
        cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor, address_offset=offset, total_size=size)
        cb.format_descriptors[0].tile = ttnn.TileDescriptor(ttnn.Tile((TINY_ROWS, TILE)))
        cb.format_descriptors[0].page_size = TINY_BYTES
        return cb

    def _column_routes(self, coord):
        """The column sender's routes of chip ``coord``: (connection nodes, [(group, chip id, connection)])."""
        mesh = self.mesh_device
        node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))
        src = node(coord)
        directions, nodes, routes = [], [], []
        for group, r in enumerate(self.rows):
            if r == coord[0]:
                continue
            dst = node((r, coord[1]))
            direction = ttnn.get_eth_forwarding_direction(src, dst)
            assert direction is not None, f"no route from {coord} to {(r, coord[1])}"
            if direction not in directions:
                directions.append(direction)
                nodes.append(dst)
            routes.append((group, dst.chip_id, directions.index(direction)))
        return nodes, routes

    def append_to(self, desc, coord, nop: bool = False):
        """Append the merge to ``desc``, the program of chip ``coord``. Returns the collective fragment of the
        row stage (finished), or None when the rectangle has one column."""
        mesh, users = self.mesh_device, self.users
        coord = tuple(coord)
        device = self.device_index[coord]
        on_chip = lambda t: ttnn.get_device_tensors(t)[device]
        slot_row, slot_col = self.rows.index(coord[0]), self.cols.index(coord[1])
        merge = cl.row_range(self.merge_core, users)
        noc = lambda core: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*core))
        xy = lambda core: noc(core).x | (noc(core).y << 16)
        nop_defines = [("HY4_MERGE_NOP", "1")] if nop else []

        frag = None
        defines = cl.kernel_defines(self.wait_limit, self.timing)
        if self.C > 1:
            frag = cl.Fragment(
                mesh,
                coord,
                desc,
                self.row_sender_core,
                self.merge_cores,
                self.wait_limit,
                link=self.link,
                timing=self.timing,
            )
        cbs = [
            self._tiny_cb(merge, CB_ACC_O, O_TILES),
            self._tiny_cb(merge, CB_ACC_MS, 1),
            self._tiny_cb(merge, CB_SEND_MS, 1),
            self._tiny_cb(merge, CB_SEND_O, O_TILES),
            self._tiny_cb(merge, CB_FINAL, O_TILES),
            self._tiny_cb_over(CB_SINK, on_chip(self.sink), 0, PART_BYTES),
        ]
        zero_use = [0] * cl.USE_ARGS
        col_use = {core: zero_use for core in self.merge_cores}
        row_use = {core: zero_use for core in self.merge_cores}
        if self.col_site is not None:
            site = self.col_site
            for parity, cb in enumerate((CB_COL0, CB_COL1)):
                cbs.append(
                    self._tiny_cb_over(cb, on_chip(site.landing), parity * site.parity_stride, site.parity_stride)
                )
            use = [
                site.ctrl_address,
                site.landing_address,
                site.parity_stride,
                site.slot_stride,
                self.R - 1,
                PART_BYTES,
            ]
            col_use = {core: use + [0, 0, 0, 0] for core in self.merge_cores}
        if self.row_site is not None:
            site = self.row_site
            for parity, cb in enumerate((CB_ROW0, CB_ROW1)):
                cbs.append(
                    self._tiny_cb_over(cb, on_chip(site.landing), parity * site.parity_stride, site.parity_stride)
                )
            if not nop:
                row_use = frag.add(site).core_args
        desc.cbs = list(desc.cbs) + cbs

        cb_ids = [CB_ACC_MS, CB_ACC_O, CB_SEND_MS, CB_SEND_O, CB_SINK, CB_FINAL, CB_COL0, CB_COL1, CB_ROW0, CB_ROW1]
        accessor = list(ttnn.TensorAccessorArgs(on_chip(self.partials)).get_compile_time_args())
        dm_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        ctx = noc(self.ctx_core)
        for user, (x, y) in enumerate(self.merge_cores):
            dm_rt[x][y] = (
                [
                    user,
                    slot_row,
                    slot_col,
                    on_chip(self.partials).buffer_address(),
                    ctx.x,
                    ctx.y,
                    on_chip(self.ctx).buffer_address(),
                    on_chip(self.state).buffer_address(),
                ]
                + list(col_use[(x, y)])
                + list(row_use[(x, y)])
            )
            compute_rt[x][y] = [slot_row, slot_col]
        include = [str(KERNEL_DIR), str(cl.KERNEL_DIR)]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "lse_merge_dm.cpp"),
                core_ranges=merge,
                compile_time_args=[self.R, self.C] + cb_ids + accessor,
                defines=defines + nop_defines,
                runtime_args=dm_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
                compiler_include_paths=include,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "lse_merge_compute.cpp"),
                core_ranges=merge,
                compile_time_args=[self.R, self.C] + cb_ids + [int.from_bytes(struct.pack("<f", self.scale), "little")],
                defines=nop_defines,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=FIDELITY[self.fidelity], fp32_dest_acc_en=False, math_approx_mode=False
                ),
                compiler_include_paths=include,
            ),
        ]
        desc.kernels = list(desc.kernels) + kernels

        if self.col_site is None and self.dry_sender_groups and not nop:
            sender = cl.core_set([self.sender_core])
            groups = self.dry_sender_groups
            desc.cbs = list(desc.cbs) + [
                ttnn.CBDescriptor(
                    total_size=size,
                    core_ranges=sender,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=size)
                    ],
                )
                for index, size in ((CB_STAGING, len(groups) * users * PART_BYTES), (CB_SCRATCH, 64))
            ]
            rt = ttnn.RuntimeArgs()
            x, y = self.sender_core
            # the arguments of the real sender, with nothing to route to, then the state tile's address
            rt[x][y] = (
                [
                    on_chip(self.partials).buffer_address(),
                    slot_row,
                    xy(self.merge_cores[0]),
                    on_chip(self.state).buffer_address(),
                    0,
                    0,
                    PART_BYTES,
                ]
                + [xy(core) for core in self.merge_cores]
                + [0]
                + [a for g in groups for a in (g, 0, 0)]
                + [on_chip(self.state).buffer_address()]
            )
            desc.kernels = list(desc.kernels) + [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "lse_sender.cpp"),
                    core_ranges=sender,
                    compile_time_args=[users, len(groups), CB_STAGING, CB_SCRATCH] + accessor,
                    defines=defines + [("HY4_LSE_DRY", "1")],
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                    ),
                    compiler_include_paths=include,
                )
            ]
        if self.col_site is not None:
            site = self.col_site
            sender = cl.core_set([self.sender_core])
            staging = (self.R - 1) * users * PART_BYTES
            desc.cbs = list(desc.cbs) + [
                ttnn.CBDescriptor(
                    total_size=size,
                    core_ranges=sender,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=size)
                    ],
                )
                for index, size in ((CB_STAGING, staging), (CB_SCRATCH, 64))
            ]
            nodes, routes = self._column_routes(coord)
            head = (
                [
                    on_chip(self.partials).buffer_address(),
                    slot_row,
                    xy(self.merge_cores[0]),
                    site.ctrl_address,
                    site.landing_address,
                    site.parity_stride,
                    site.slot_stride,
                ]
                + [xy(core) for core in self.merge_cores]
                + [len(nodes)]
                + [a for route in routes for a in route]
            )
            desc.kernels = list(desc.kernels) + [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "lse_sender.cpp"),
                    core_ranges=sender,
                    compile_time_args=[users, len(routes), CB_STAGING, CB_SCRATCH] + accessor,
                    defines=defines + nop_defines,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                    ),
                    compiler_include_paths=include,
                )
            ]
            index = len(desc.kernels) - 1
            fabric_args = []
            if not nop:
                src = mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
                fabric_args = list(
                    ttnn.setup_routing_plane_connection(
                        mesh, src, nodes, [self.link], desc, index, ttnn.CoreCoord(*self.sender_core)
                    )
                )
            x, y = self.sender_core
            desc.kernels[index].runtime_args[x][y] = head + fabric_args
        if frag is not None and not nop:
            frag.finish()
        return frag

    def descriptor(self, nop: bool = False):
        """One program per member chip. ``nop``: the same cores and buffers with kernels that return at once
        (no row sender kernel: the launch floor is short of one data-movement kernel on one core)."""
        mesh_desc = ttnn.MeshProgramDescriptor()
        for coord in self.coords:
            desc = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
            self.append_to(desc, coord, nop)
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
        return mesh_desc

    def io_tensors(self):
        persistent = [t for site in self.sites for t in (site.landing, site.ctrl, site.staging)]
        return [self.partials, self.sink] + persistent + [self.state, self.ctx]

    def compile(self, nop: bool = False):
        ttnn.experimental.prepare_generic_op(self.io_tensors(), self.descriptor(nop))

    def __call__(self, nop: bool = False):
        if nop not in self._descriptors:
            self._descriptors[nop] = self.descriptor(nop)
        return ttnn.generic_op(self.io_tensors(), self._descriptors[nop])

    # results ------------------------------------------------------------------------------------
    def read(self) -> dict:
        """ctx [members, users, heads_out, 512] (member chips row-major over the rectangle; the heads of member
        k are ``heads_of(k // C, k % C)``), status [members, users] (0x100: done, no wait ran out) and stamps
        [members, users, 8]."""
        ttnn.synchronize_device(self.mesh_device)
        ctx, state = ttnn.get_device_tensors(self.ctx), ttnn.get_device_tensors(self.state)
        out = {"ctx": [], "status": [], "stamps": [], "dry": []}
        for coord in self.coords:
            d = self.device_index[coord]
            raw = tiles_to_raw(ttnn.to_torch(ctx[d]).float().reshape(self.ctx_tiles * TILE, TILE))
            per_head = unpack_tiny(raw.reshape(self.heads_out, O_TILES * TINY_VALUES))  # [heads_out, 8 users, 512]
            out["ctx"].append(per_head.transpose(0, 1)[: self.users])
            words = ttnn.to_torch(state[d]).reshape(self.users, TILE, TILE).to(torch.int64) & 0xFFFFFFFF
            out["status"].append(words[:, 0, 0])
            out["stamps"].append(words[:, 0, 4:12])
            out["dry"].append(words[0, 1, 0:4])  # merge core 0, words 16..19: the dry sender's report
        return {k: torch.stack(v) for k, v in out.items()}

    def golden(self, o, m, s, sink) -> torch.Tensor:
        """What ``read()["ctx"]`` must hold for the partials given to set_partials and the sink given to
        set_sink (its BF16 peer): float64 [members, users, heads_out, 512]."""
        peer = (sink.float() / self.scale).to(torch.bfloat16).double() * self.scale
        merged = merge_golden(o, m, s, peer[None].expand(self.users, HEADS), self.scale)  # [users, 64, 512]
        return torch.stack([merged[:, self.heads_of(k // self.C, k % self.C)] for k in range(len(self.coords))])

    def bytes_moved(self) -> dict:
        """Fabric payload bytes per chip: column stage out / in, row stage injected / in, packets injected."""
        col = (self.R - 1) * self.users * PART_BYTES
        row = (
            self.row_site.bytes_per_use() if self.row_site is not None else {"injected": 0, "received": 0, "packets": 0}
        )
        return {
            "column_sent": col,
            "column_received": col,
            "row_injected": row["injected"],
            "row_received": row["received"],
            "packets": (self.R - 1) * self.users + row["packets"],
        }

    def l1_bytes(self) -> int:
        return sum(s.l1_bytes() for s in self.sites) + (self.sink_tiles + self.ctx_tiles) * 2048 + 4096

    def release(self):
        for tensor in (self.sink, self.ctx, self.state):
            ttnn.deallocate(tensor)
        for site in self.sites:
            site.release()
