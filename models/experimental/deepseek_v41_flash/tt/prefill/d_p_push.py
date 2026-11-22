# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 of the fabric push (``PREFILL_D_P_FABRIC_PUSH.md``): metal_2's prefill state written from its caches
straight into a decode session's DRAM over TT-Fabric, no host copy of the compressed entries.

:func:`push` takes the targets a decode model's ``remote_commit_targets`` returned (where the session's rings, entry
pages and key blocks live on every decode chip) and writes, from the 8 x 4 prefill mesh:

* compressed KV entries from the slab of the chip that owns them (one 1 KiB row a page, dedup: one owner per entry),
* index-key blocks (64 entries, eight 2 KiB tiles) from a device tilize of the owner row's slab, by one of that
  row's chips (TP-replicated: any column; they take turns),
* the window rings (the last ``window`` rows of each layer, read on the host from the chips owning those ring blocks
  and tilized there: 128 KiB a layer) from a staging tensor every chip holds, the tiles spread over every chip.

Every sending chip runs one core: a work list in DRAM of (source, page count, destination chip, first source page,
destination buffer, first destination page) runs, one a KV cache block (or the part of one in an owner stripe), a
key block or a ring, read ``K`` at a time; a run's pages are staged in L1 ``K`` at a time and sent as fabric unicast
writes. A remote mesh
is reached through this mesh's exit chip, so one connection a chip carries every destination: the route is rewritten
when the destination changes. A destination page goes to its interleaved bank (``page % banks``) at the bank's
offset from the targets' bank table.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

import ttnn

CORE = ttnn.CoreCoord(0, 0)
K = 32  # work-list runs a table page, and pages read into L1 a NoC barrier
SLOT = 2048  # L1 bytes a staged page (the largest page: a tile)
ENTRY_WORDS = (
    4  # (source | count << 8 | destination << 16, first source page, destination buffer, first destination page)
)

_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#if !defined(FABRIC_2D)
#error "the push needs a 2D fabric: a remote mesh is reached through its exit chip"
#endif
using namespace tt::tt_fabric::mesh::experimental;

constexpr uint32_t NB = get_compile_time_arg_val(0);    // a destination chip's DRAM banks
constexpr uint32_t K = get_compile_time_arg_val(1);     // work-list entries a table page
constexpr uint32_t SLOT = get_compile_time_arg_val(2);  // L1 bytes a staged page
constexpr uint32_t NSRC = get_compile_time_arg_val(3);  // source buffers
constexpr uint32_t DST_BYTES = get_compile_time_arg_val(4);  // the destination table
constexpr uint32_t DST_WORDS = 2 + 3 * NB;  // a destination: fabric chip, mesh, then (x, y, offset) a bank
constexpr auto args = TensorAccessorArgs<5>();  // every buffer read here: DRAM interleaved

void kernel_main() {
    size_t a = 0;
    uint32_t src_addr[NSRC], src_page[NSRC];
    for (uint32_t s = 0; s < NSRC; ++s) {
        src_addr[s] = get_arg_val<uint32_t>(a++);
        src_page[s] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t tbl = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t dsts = get_arg_val<uint32_t>(a++);

    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t tbuf = buf + K * SLOT;
    const uint32_t dbuf = tbuf + K * 16;
    noc_async_read(TensorAccessor(args, dsts, DST_BYTES).get_noc_addr(0), dbuf, DST_BYTES);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbuf);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbuf);
    const auto table = TensorAccessor(args, tbl, K * 16);

    tt::tt_fabric::RoutingPlaneConnectionManager conn;
    open_connections(conn, 1, a);
    auto& slot = conn.get(0);
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    uint32_t cur = 0xffffffffu;

    for (uint32_t i0 = 0; i0 < n; i0 += K) {
        const uint32_t nr = (n - i0) < K ? (n - i0) : K;
        noc_async_read(table.get_noc_addr(i0 / K), tbuf, K * 16);
        noc_async_read_barrier();
        for (uint32_t r = 0; r < nr; ++r) {
            const uint32_t w = t[4 * r], s = w & 0xff, count = (w >> 8) & 0xff, j = w >> 16;
            const uint32_t sp = t[4 * r + 1], base = t[4 * r + 2], dp = t[4 * r + 3];
            const uint32_t page = src_page[s];
            const auto src = TensorAccessor(args, src_addr[s], page);
            volatile tt_l1_ptr uint32_t* dst = d + j * DST_WORDS;
            if (j != cur) {
                fabric_set_unicast_route(hdr, dst[0], dst[1]);
                cur = j;
            }
            for (uint32_t q0 = 0; q0 < count; q0 += K) {
                const uint32_t m = (count - q0) < K ? (count - q0) : K;
                for (uint32_t q = 0; q < m; ++q) {
                    noc_async_read(src.get_noc_addr(sp + q0 + q), buf + q * SLOT, page);
                }
                noc_async_read_barrier();
                for (uint32_t q = 0; q < m; ++q) {
                    const uint32_t p = dp + q0 + q, bank = p % NB;
                    const uint64_t addr =
                        get_noc_addr(dst[2 + 3 * bank], dst[3 + 3 * bank], dst[4 + 3 * bank] + base + (p / NB) * page);
                    fabric_unicast_noc_unicast_write<decltype(slot.sender), false>(
                        &slot.sender, hdr, dst[0], dst[1], buf + q * SLOT, page,
                        tt::tt_fabric::NocUnicastCommandHeader{addr});
                    noc_async_writes_flushed();  // the header and the staged page left L1 before either is rewritten
                }
            }
        }
    }
    close_connections(conn);
}
"""


def _int32(x: torch.Tensor) -> torch.Tensor:
    """uint32 values held in int64 -> the same bits as int32."""
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)


def _owner_chips(c, owner: torch.Tensor, spread: torch.Tensor, cols: int) -> torch.Tensor:
    """The sending chip (row-major mesh index) of entries whose owner stripe is ``owner`` in metal_2's served cache
    ``c``: a dedup cache's owners are the chips themselves; a TP-replicated one's are mesh rows, any of whose columns
    holds the rows (``spread`` picks one)."""
    if c.tp_axis is not None:
        return owner
    return owner * cols + spread % cols


def _slab_rows(c, b0: int, b1: int) -> ttnn.Tensor:
    """Local rows of owner-stripe blocks ``[b0, b1)`` of metal_2's served cache ``c`` (every chip's own), DRAM
    interleaved ROW_MAJOR."""
    slab = c.slab if c.slab is not None else c.served
    hs = c.stripe // c.ratio
    if b1 * hs > c.crows:
        raise ValueError(f"owner blocks up to {b1} of {hs} rows: past the cache's {c.crows} local rows")
    return slab.slice_rows(b0 * hs, b1 * hs)


def push(d_p, targets: dict, slot: int = 0, times: Optional[dict] = None) -> dict:
    """Write cache slot ``slot`` of ``d_p`` (a :class:`~.d_p_bridge.DPPrefill` whose slot holds a prompt of
    ``targets["seq_len"]`` tokens) into the decode session ``targets`` describes. Returns ``{"bytes", "pages",
    "seconds"}`` (``seconds``: the device program, launch included); ``times``: host seconds per part, added up.
    The writes have left this mesh when it returns; the receiver commits after a reply over TCP."""
    from models.demos.deepseek_v41_flash_d_p.tt.served import dealloc

    from .d_p_bridge import _window_rows

    times = times if times is not None else {}
    t0 = time.perf_counter()
    mesh, cfg = d_p.mesh_device, d_p.cfg
    rows_m, cols = tuple(mesh.shape)
    chips = rows_m * cols
    T, start, block, window = (int(targets[k]) for k in ("seq_len", "start", "block", "window"))
    if window != cfg.window:
        raise ValueError(f"decode's {window}-row window is not this prefill's {cfg.window}")
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    if max(targets["page_bytes"].values()) > payload:
        raise ValueError(f"pages of {targets['page_bytes']} B exceed the {payload} B fabric payload")
    st = d_p.runtime.state(d_p.slots[slot], 0)
    skipped = set(d_p.runtime.model.decoder)
    rows_of = {r["ratio"]: torch.tensor(r["blocks"], dtype=torch.int64) for r in targets["page_rows"]}

    dests, first_dst = [], {}  # every decode chip; stage -> index of its first chip in ``dests``
    for stage in targets["stages"]:
        first_dst[stage["stage"]] = len(dests)
        dests += stage["chips"]
    banks = {len(chip["banks"]) for chip in dests}
    if len(banks) != 1:
        raise ValueError(f"decode chips with {sorted(banks)} DRAM banks")
    nb = banks.pop()

    sources: list[tuple[ttnn.Tensor, int]] = []  # (DRAM interleaved buffer, page bytes); a work entry's source index
    # (sender, source, first source page, count, destination, destination buffer, first destination page) columns
    work: list[torch.Tensor] = []
    free = []

    def add(
        sender: torch.Tensor, dst: int, base: int, src_page: torch.Tensor, count: torch.Tensor, page: torch.Tensor
    ) -> None:
        """Runs of ``count`` pages of the newest source from ``src_page`` on to pages ``page`` on of buffer ``base``
        on decode chip ``dst``."""
        full = torch.ones_like(page)
        work.append(torch.stack([sender, full * (len(sources) - 1), src_page, count, full * dst, full * base, page]))

    for source in targets["sources"]:
        layer, ratio, e0 = int(source["layer"]), int(source["ratio"]), int(source["first_entry"])
        n = T // ratio
        if e0 != start // ratio:
            raise ValueError(f"source {layer}: first entry {e0} for a commit from position {start}")
        if n <= e0:
            continue
        row = rows_of[ratio]
        pools = [
            (first_dst[stage["stage"]] + k, pool)
            for stage in targets["stages"]
            for pool in stage["pools"]
            if int(pool["source"]) == layer
            for k in range(len(stage["chips"]))
        ]
        comp = st.compress[layer]

        c = comp.ckv
        hs = c.stripe // ratio
        period = c.owners * hs
        b0, b1 = e0 // period, -(-n // period)
        sl = _slab_rows(c, b0, b1)
        free.append(sl)
        if int(sl.shape[-1]) * 2 != targets["page_bytes"]["kv"]:
            raise ValueError(f"source {layer}: {sl.shape[-1]}-wide entries, decode pages of {targets['page_bytes']}")
        sources.append((sl, int(sl.shape[-1]) * 2))
        # A run is the entries of one cache block within one owner stripe: consecutive pages on both sides.
        e = torch.arange(e0, n, dtype=torch.int64)
        e = e[(e % block == 0) | (e % hs == 0) | (e == e0)]
        count = torch.diff(e, append=torch.tensor([n]))
        owner, local = (e % period) // hs, (e // period - b0) * hs + e % hs
        page = row[e // block] * block + e % block
        run = torch.arange(e.numel(), dtype=torch.int64)
        for dst, pool in pools:
            add(_owner_chips(c, owner, run + dst, cols), dst, int(pool["kv"]), local, count, page)

        c = comp.index_k
        if c.fp8:
            raise ValueError("index keys in fp8_row192: the push needs DS41_INDEX_K_SERVED=bf16")
        hs = c.stripe // ratio
        period = c.owners * hs
        if hs % block:
            raise ValueError(f"source {layer}: {hs} key rows an owner stripe split {block}-entry blocks")
        b0, b1 = e0 // period, -(-n // period)
        sl = _slab_rows(c, b0, b1)
        keys = ttnn.to_layout(sl, ttnn.TILE_LAYOUT)
        dealloc(sl)
        free.append(keys)
        sources.append((keys, targets["page_bytes"]["keys"]))
        tile_cols = int(keys.shape[-1]) // ttnn.TILE_SIZE
        per_block = block // ttnn.TILE_SIZE * tile_cols
        b = torch.arange(e0 // block, -(-n // block), dtype=torch.int64)
        eb = b * block
        owner, local = (eb % period) // hs, (eb // period - b0) * hs + eb % hs
        # A block's 64 rows are block / 32 whole tile rows of the tilized slab: per_block consecutive tiles.
        src_page = (local // ttnn.TILE_SIZE) * tile_cols
        page = row[b] * per_block
        count = torch.full_like(b, per_block)
        for dst, pool in pools:
            add(_owner_chips(c, owner, b + dst, cols), dst, int(pool["keys"]), src_page, count, page)
    t1 = time.perf_counter()
    times["compressed"] = times.get("compressed", 0.0) + t1 - t0

    ring_layers = sorted({int(r["layer"]) for stage in targets["stages"] for r in stage["rings"]})
    if ring_layers:
        p0 = max(0, T - window)
        tails = torch.zeros(len(ring_layers), window, cfg.head_dim, dtype=torch.bfloat16)
        for k, li in enumerate(ring_layers):
            if li not in skipped:
                tails[k, window - (T - p0) :] = _window_rows(st.window[li], p0, T)
        staged = ttnn.from_torch(
            tails.reshape(1, 1, -1, cfg.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        free.append(staged)
        sources.append((staged, targets["page_bytes"]["ring"]))
        per_layer = window * cfg.head_dim // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)
        first = torch.tensor([int(targets["slot"]) * per_layer])
        count = torch.tensor([per_layer])
        index = {li: k for k, li in enumerate(ring_layers)}
        for stage in targets["stages"]:
            for ring in stage["rings"]:
                k_layer = index[int(ring["layer"])]
                src_page = torch.tensor([k_layer * per_layer])
                for k in range(len(stage["chips"])):
                    dst = first_dst[stage["stage"]] + k
                    add(torch.tensor([(k_layer + dst) % chips]), dst, int(ring["addr"]), src_page, count, first)
    t2 = time.perf_counter()
    times["window"] = times.get("window", 0.0) + t2 - t1
    sender, src, src_page, count, dst, base, page = torch.cat(work, dim=1)
    if len(sources) > 255 or len(dests) > 2**16 or int(count.max()) > 255:
        raise ValueError(
            f"{len(sources)} sources / {len(dests)} destinations / runs of {int(count.max())} pages: past the work "
            f"entry's fields"
        )
    for t, _ in sources:
        mc = t.memory_config()
        assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED and mc.buffer_type == ttnn.BufferType.DRAM, mc

    order = torch.argsort(sender, stable=True)
    sender, src, src_page, count, dst, base, page = (x[order] for x in (sender, src, src_page, count, dst, base, page))
    runs = torch.bincount(sender, minlength=chips)
    offsets = torch.cumsum(runs, 0) - runs
    rank = torch.arange(sender.numel(), dtype=torch.int64) - offsets[sender]
    n_tbl = max(1, -(-int(runs.max()) // K))
    table = torch.zeros(chips, n_tbl * K, ENTRY_WORDS, dtype=torch.int32)
    table[sender, rank] = _int32(torch.stack([src | (count << 8) | (dst << 16), src_page, base, page], dim=1))
    table_t = ttnn.from_torch(
        table.reshape(rows_m, cols, n_tbl, K * ENTRY_WORDS),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows_m, cols)),
    )
    dst_words = 2 + 3 * nb
    dst_table = torch.zeros(-(-len(dests) * dst_words // 16) * 16, dtype=torch.int64)
    for j, chip in enumerate(dests):
        dst_table[j * dst_words : (j + 1) * dst_words] = torch.tensor(
            [chip["chip_id"], targets["mesh_id"]] + [v for bank in chip["banks"] for v in bank], dtype=torch.int64
        )
    dst_t = ttnn.from_torch(
        _int32(dst_table).reshape(1, 1, 1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    free += [table_t, dst_t]

    dst_nodes = [ttnn.FabricNodeId(ttnn.MeshId(int(targets["mesh_id"])), int(chip["chip_id"])) for chip in dests]
    ct = [nb, K, SLOT, len(sources), int(dst_table.numel()) * 4]
    ct += list(ttnn.TensorAccessorArgs(table_t).get_compile_time_args())
    common = [v for t, p in sources for v in (int(t.buffer_address()), p)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
    size = K * SLOT + K * 16 + int(dst_table.numel()) * 4 + 64
    md = ttnn.MeshProgramDescriptor()
    for linear in range(chips):
        n = int(runs[linear])
        if not n:
            continue
        coord = ttnn.MeshCoordinate(*divmod(linear, cols))
        src_node = mesh.get_fabric_node_id(coord)
        mine = sorted(set(dst[sender == linear].tolist()))
        directions = {ttnn.get_eth_forwarding_direction(mesh, src_node, dst_nodes[j]) for j in mine}
        if len(directions) != 1 or None in directions:
            raise RuntimeError(f"prefill chip {linear}: decode chips {mine} leave by directions {directions}")
        links = list(ttnn.get_forwarding_link_indices(mesh, src_node, dst_nodes[mine[0]]))
        if not links:
            raise RuntimeError(f"no fabric route from {src_node} to {dst_nodes[mine[0]]}")
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=size, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)]
            )
        ]
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        ]
        fabric = ttnn.setup_routing_plane_connection(mesh, src_node, [dst_nodes[mine[0]]], [links[0]], prog, 0, CORE)
        own = common + [int(table_t.buffer_address()), n, int(dst_t.buffer_address())]
        prog.kernels[0].runtime_args[CORE.x][CORE.y] = own + list(fabric)
        md[ttnn.MeshCoordinateRange(coord, coord)] = prog
    t3 = time.perf_counter()
    times["plan"] = times.get("plan", 0.0) + t3 - t2

    ttnn.generic_op([*(t for t, _ in sources), table_t, dst_t], md)
    ttnn.synchronize_device(mesh)
    seconds = time.perf_counter() - t3
    times["device"] = times.get("device", 0.0) + seconds
    page_of = torch.tensor([p for _, p in sources], dtype=torch.int64)
    nbytes = int((page_of[src] * count).sum())
    for t in free:
        dealloc(t)
    return {"bytes": nbytes, "pages": int(count.sum()), "runs": int(sender.numel()), "seconds": seconds}


class FabricPush:
    """The push as the ``KvFabricPush`` C++ op, a launch per prefill chunk: :meth:`begin` with a request's targets
    before its prefill, :meth:`chunk` after each chunk is enqueued (pushes the entries it completed while the device
    runs the next), :meth:`finish` after the last (the rest, and the window rings). The op reads metal_2's slabs in
    place (no slices, the index keys tilized in its kernel) and takes each launch's runs from an H2D socket a chip, so
    planning and streaming them overlaps the device. One per :class:`~.d_p_bridge.DPPrefill` (it holds the sockets)."""

    def __init__(self, d_p, fifo_bytes: int = 1 << 20):
        from ttnn._ttnn.operations.experimental import KvFabricPush

        self.d_p = d_p
        self.op = KvFabricPush(d_p.mesh_device, fifo_bytes)
        self.targets: Optional[dict] = None
        self.slot = 0
        self.times: dict = {}
        self.stats = {"bytes": 0, "pages": 0, "runs": 0, "launches": 0}

    def _launch(self, end: int, last: bool, rings=None, ring_runs=()) -> None:
        r = self.op.push(end, last, rings, list(ring_runs))
        for k in ("bytes", "pages", "runs"):
            self.stats[k] += int(getattr(r, k))
        self.stats["launches"] += 1
        for k in ("plan", "enqueue", "stream"):
            self.times[k] = self.times.get(k, 0.0) + float(getattr(r, f"{k}_seconds"))

    def begin(self, targets: dict, slot: int = 0) -> None:
        """A request: decode's ``remote_commit_targets`` for cache slot ``slot`` of ``d_p`` (the prompt goes in from
        ``targets["start"]``, whatever the prefill resumes from: entries before it are in the slot already)."""
        from ttnn._ttnn.operations.experimental import KvPushCache, KvPushLayer

        t0 = time.perf_counter()
        d_p, cfg = self.d_p, self.d_p.cfg
        T, start, block, window = (int(targets[k]) for k in ("seq_len", "start", "block", "window"))
        if window != cfg.window:
            raise ValueError(f"decode's {window}-row window is not this prefill's {cfg.window}")
        payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        if max(targets["page_bytes"].values()) > payload:
            raise ValueError(f"pages of {targets['page_bytes']} B exceed the {payload} B fabric payload")
        st = d_p.runtime.state(d_p.slots[slot], 0)
        rows_of = {int(r["ratio"]): [int(b) for b in r["blocks"]] for r in targets["page_rows"]}
        dests, first_dst = [], {}
        for stage in targets["stages"]:
            first_dst[stage["stage"]] = len(dests)
            dests += stage["chips"]

        sources: list = []
        index: dict = {}

        def cache(c, n: int, width_bytes: int) -> KvPushCache:
            """The op's view of metal_2's served cache ``c`` for ``n`` entries (its slab's tensor a source)."""
            slab = c.slab if c.slab is not None else c.served
            hs = c.stripe // c.ratio
            if -(-n // (c.owners * hs)) * hs > c.crows:
                raise ValueError(f"{n} entries of {hs}-row owner stripes: past the cache's {c.crows} local rows")
            if slab.width * 2 != width_bytes:
                raise ValueError(f"{slab.width}-wide rows for {width_bytes} B decode pages")
            key = int(slab.tensor.buffer_address())
            if key not in index:
                index[key] = len(sources)
                sources.append(slab.tensor)
            v = KvPushCache()
            v.source, v.first_page, v.stripe_rows, v.owners = index[key], slab.batch * slab.rows, hs, c.owners
            v.owners_are_chips = c.tp_axis is not None
            return v

        layers = []
        for source in targets["sources"]:
            li, ratio, e0 = int(source["layer"]), int(source["ratio"]), int(source["first_entry"])
            if e0 != start // ratio:
                raise ValueError(f"source {li}: first entry {e0} for a commit from position {start}")
            comp = st.compress[li]
            if comp.index_k.fp8:
                raise ValueError("index keys in fp8_row192: the push needs DS41_INDEX_K_SERVED=bf16")
            layer = KvPushLayer()
            layer.ratio, layer.first_entry = ratio, e0
            layer.kv = cache(comp.ckv, T // ratio, int(targets["page_bytes"]["kv"]))
            keys = comp.index_k.slab if comp.index_k.slab is not None else comp.index_k.served
            layer.keys = cache(comp.index_k, T // ratio, keys.width * 2)
            layer.key_tile_cols = keys.width // ttnn.TILE_SIZE
            layer.blocks = rows_of[ratio]
            layer.pools = [
                (first_dst[stage["stage"]] + k, int(pool["kv"]), int(pool["keys"]))
                for stage in targets["stages"]
                for pool in stage["pools"]
                if int(pool["source"]) == li
                for k in range(len(stage["chips"]))
            ]
            layers.append(layer)
        self.op.begin(
            sources,
            int(targets["mesh_id"]),
            [int(chip["chip_id"]) for chip in dests],
            [[int(v) for bank in chip["banks"] for v in bank] for chip in dests],
            block,
            layers,
        )
        self.targets, self.slot = targets, slot
        self.times = {"begin": time.perf_counter() - t0}
        self.stats = {"bytes": 0, "pages": 0, "runs": 0, "launches": 0}

    def chunk(self, end: int) -> None:
        """After the prefill chunk ending at position ``end`` is enqueued: push what it completed."""
        if end < int(self.targets["seq_len"]):
            self._launch(end, False)

    def finish(self, times: Optional[dict] = None) -> dict:
        """After the prefill: push the rest and the window rings (read from the chips owning the last ``window``
        rows: this waits for the prefill), then wait until the writes have left this mesh. Returns ``{"bytes",
        "pages", "runs", "launches", "seconds"}`` (``seconds``: from here until the device is done); ``times``:
        host seconds per part, added up."""
        from ttnn._ttnn.operations.experimental import KvPushRing
        from models.demos.deepseek_v41_flash_d_p.tt.served import dealloc

        from .d_p_bridge import _window_rows

        t0 = time.perf_counter()
        d_p, cfg, targets = self.d_p, self.d_p.cfg, self.targets
        mesh = d_p.mesh_device
        T, window = int(targets["seq_len"]), int(targets["window"])
        st = d_p.runtime.state(d_p.slots[self.slot], 0)
        skipped = set(d_p.runtime.model.decoder)
        ring_layers = sorted({int(r["layer"]) for stage in targets["stages"] for r in stage["rings"]})
        staged, ring_runs = None, []
        if ring_layers:
            p0 = max(0, T - window)
            tails = torch.zeros(len(ring_layers), window, cfg.head_dim, dtype=torch.bfloat16)
            for k, li in enumerate(ring_layers):
                if li not in skipped:
                    tails[k, window - (T - p0) :] = _window_rows(st.window[li], p0, T)
            staged = ttnn.from_torch(
                tails.reshape(1, 1, -1, cfg.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            per_layer = window * cfg.head_dim // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)
            k_of = {li: k for k, li in enumerate(ring_layers)}
            dst = 0
            for stage in targets["stages"]:
                for ring in stage["rings"]:
                    for k in range(len(stage["chips"])):
                        r = KvPushRing()
                        r.destination, r.addr = dst + k, int(ring["addr"])
                        r.first_source_page, r.pages = k_of[int(ring["layer"])] * per_layer, per_layer
                        r.first_page = int(targets["slot"]) * per_layer
                        ring_runs.append(r)
                dst += len(stage["chips"])
        t1 = time.perf_counter()
        self.times["window"] = self.times.get("window", 0.0) + t1 - t0
        self._launch(T, True, staged, ring_runs)
        t2 = time.perf_counter()
        ttnn.synchronize_device(mesh)
        t3 = time.perf_counter()
        self.times["device"] = self.times.get("device", 0.0) + t3 - t2
        if staged is not None:
            dealloc(staged)
        if times is not None:
            for k, v in self.times.items():
                times[k] = times.get(k, 0.0) + v
        self.targets = None
        return {**self.stats, "seconds": t3 - t0}
