# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The sequence-parallel prefix as a relay around the SP ring, in place of all-gathering every rank's [A | B] summary.

Each rank needs only its entry state E_p (the recurrent state after every chronologically earlier rank) and the
chronologically first rank also needs the final state (its split tail's entry). Rank p (chronological position)
receives E_p from its predecessor, applies its own transform, E_{p+1} = A_p E_p + B_p, and forwards E_{p+1} to its
successor: about one state per rank instead of every rank's two summaries. The heads are independent, so each runs on
its own core and the heads stream through the ring one behind another. Column j of A_p E_p + B_p needs only column j
of E_p, so the unit a hop moves and computes is one tile column of a head's state (states are stored column-major by
tile): a rank forwards a head's first column before the head's last one has arrived.

Per device:
  * head cores (one per head): read A_p and B_p (BF16, this rank's packed summary), take E_p column by column (the
    initial state on the first rank, else the predecessor's landed columns), compute each column of A_p E_p + B_p in
    FP32 and count it on its sender's core;
  * sender cores (one per link): read each finished column out of its head core into one of several buffers (later
    columns while earlier ones' packets go out), release the head core after its last column and forward each column
    into the successor's landing buffer over the fabric. The last rank sends to the first rank's final-state landing.
    The link's throughput (about 1 us per 16 KiB column) and the per-hop latency bound the relay.
Landing buffers and the hand-off are L1 tensors sharded over the head cores, so their addresses agree on every device.

Successive calls are not fenced against each other: a rank writes into its successor's landing only after that call has
started on every rank of the line, which the KDA layer guarantees by its earlier SP-line barriers (the QKV
convolution's history exchange) between two relays.
"""

from __future__ import annotations

import ttnn

_CHRONOLOGY = (
    '#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"\n'
)

_HEAD_READER = _CHRONOLOGY + r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
    constexpr uint32_t cb_chron = get_compile_time_arg_val(3);
    constexpr uint32_t sp_rank = get_compile_time_arg_val(4);
    constexpr uint32_t sp_size = get_compile_time_arg_val(5);
    constexpr uint32_t local_rows = get_compile_time_arg_val(6);
    constexpr uint32_t kt = get_compile_time_arg_val(7);  // key tiles; the state is kt x kt tiles
    constexpr auto packed_args = TensorAccessorArgs<8>();
    constexpr auto init_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    constexpr auto start_args = TensorAccessorArgs<init_args.next_compile_time_args_offset()>();
    constexpr uint32_t state_tiles = kt * kt;

    const uint32_t packed_addr = get_arg_val<uint32_t>(0);
    const uint32_t init_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_addr = get_arg_val<uint32_t>(2);
    const uint32_t head = get_arg_val<uint32_t>(3);
    const uint32_t arrival_sem = get_arg_val<uint32_t>(4);

    cb_reserve_back(cb_chron, 1);
    const auto start = TensorAccessor(start_args, start_addr);
    noc_async_read(start.get_noc_addr(0), get_write_ptr(cb_chron), sizeof(uint32_t));
    const auto packed = TensorAccessor(packed_args, packed_addr);
    cb_reserve_back(cb_a, state_tiles);
    cb_reserve_back(cb_b, state_tiles);
    const uint32_t ab_tile = get_tile_size(cb_a);
    // [A | B] rows of this head: kt tile rows of 2 kt tiles.
    for (uint32_t r = 0; r < kt; ++r) {
        for (uint32_t c = 0; c < kt; ++c) {
            const uint32_t page = (head * kt + r) * 2 * kt + c;
            noc_async_read_page(page, packed, get_write_ptr(cb_a) + (r * kt + c) * ab_tile);
            noc_async_read_page(page + kt, packed, get_write_ptr(cb_b) + (r * kt + c) * ab_tile);
        }
    }
    noc_async_read_barrier();
    const uint32_t actual_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_chron))[0];
    const auto t = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    const uint32_t position = (t.rank + sp_size - t.first_rank) % sp_size;
    cb_push_back(cb_chron, 1);
    cb_push_back(cb_a, state_tiles);
    cb_push_back(cb_b, state_tiles);

    // The entry state is stored column-major by tile, so each tile column (all the next state's column needs) can
    // land, be computed and leave on its own.
    cb_reserve_back(cb_ein, state_tiles);
    if (position == 0) {
        const auto init = TensorAccessor(init_args, init_addr);
        const uint32_t state_tile = get_tile_size(cb_ein);
        for (uint32_t j = 0; j < kt; ++j) {
            for (uint32_t r = 0; r < kt; ++r) {
                noc_async_read_page(head * state_tiles + r * kt + j, init, get_write_ptr(cb_ein) + (j * kt + r) * state_tile);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_ein, state_tiles);
    } else {
        // The predecessor's sender writes each tile column of this head's entry state into cb_ein's storage, then
        // increments.
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_sem);
        for (uint32_t j = 0; j < kt; ++j) {
            noc_semaphore_wait_min(arrived, j + 1);
            cb_push_back(cb_ein, kt);
        }
        noc_semaphore_set(arrived, 0);
    }
}
"""

_HEAD_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// E_out = A E_in + B for one head, accumulated in the FP32 DST at the affine prefix's fidelity.
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eout = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr uint32_t state_tiles = kt * kt;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_ein, cb_eout);
    cb_wait_front(cb_a, state_tiles);
    cb_wait_front(cb_b, state_tiles);
    // Per output tile column (column-major state): B's column (BF16, exact in the FP32 DST) is copied into DST first
    // and the K-sum accumulates on top of it, so the state never leaves DST between the product and the add. Column j
    // needs only column j of E_in, so B's column is in DST before that column lands.
    for (uint32_t j = 0; j < kt; ++j) {
        cb_reserve_back(cb_eout, kt);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_b);
        copy_init(cb_b);
        for (uint32_t r = 0; r < kt; ++r) {
            copy_tile(cb_b, r * kt + j, r);
        }
        cb_wait_front(cb_ein, (j + 1) * kt);
        reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_ein);
        matmul_block_init(cb_a, cb_ein, false, 1, kt, kt);
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(cb_a, cb_ein, k, j * kt + k, 0, false, 1, kt, kt);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < kt; ++r) {
            pack_tile(r, cb_eout);
        }
        tile_regs_release();
        cb_push_back(cb_eout, kt);
    }
    cb_pop_front(cb_a, state_tiles);
    cb_pop_front(cb_ein, state_tiles);
    cb_pop_front(cb_b, state_tiles);
}
"""

_HEAD_WRITER = _CHRONOLOGY + r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_eout = get_compile_time_arg_val(0);
    constexpr uint32_t cb_chron = get_compile_time_arg_val(1);
    constexpr uint32_t sp_rank = get_compile_time_arg_val(2);
    constexpr uint32_t sp_size = get_compile_time_arg_val(3);
    constexpr uint32_t local_rows = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);
    constexpr auto entry_args = TensorAccessorArgs<6>();
    constexpr auto final_args = TensorAccessorArgs<entry_args.next_compile_time_args_offset()>();
    constexpr uint32_t state_tiles = kt * kt;

    const uint32_t entry_addr = get_arg_val<uint32_t>(0);
    const uint32_t final_addr = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);
    const uint32_t ein_addr = get_arg_val<uint32_t>(3);           // this head's entry state (cb_ein storage)
    const uint32_t final_landing = get_arg_val<uint32_t>(4);      // the last rank's final state lands here
    const uint32_t final_sem = get_arg_val<uint32_t>(5);
    const uint32_t ack_sem = get_arg_val<uint32_t>(6);
    const uint32_t notify_sem = get_arg_val<uint32_t>(7);        // this head's column count on its sender core
    const uint32_t sender_xy = get_arg_val<uint32_t>(8);         // the sender's NoC x | y << 16

    cb_wait_front(cb_chron, 1);
    const uint32_t actual_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_chron))[0];
    const auto t = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    const uint32_t position = (t.rank + sp_size - t.first_rank) % sp_size;
    cb_pop_front(cb_chron, 1);

    const auto entry = TensorAccessor(entry_args, entry_addr);
    const auto final_state = TensorAccessor(final_args, final_addr);
    const uint32_t tile_bytes = get_tile_size(cb_eout);
    // Hand each tile column to this head's sender as it is packed (a count on the sender's core): the DRAM copies
    // below are off the relay's path.
    const uint64_t notify = get_noc_addr(sender_xy & 0xFFFF, sender_xy >> 16, notify_sem);
    for (uint32_t j = 0; j < kt; ++j) {
        cb_wait_front(cb_eout, (j + 1) * kt);
        noc_semaphore_inc(notify, 1);
    }
    // The states are stored column-major by tile; the DRAM tensors are row-major.
    const auto page = [&](uint32_t i) { return head * state_tiles + (i % kt) * kt + i / kt; };
    // Compute has consumed the entry state; it stays in place until the next call's predecessor writes it.
    for (uint32_t i = 0; i < state_tiles; ++i) {
        noc_async_write_page(page(i), entry, ein_addr + i * tile_bytes);
    }
    if (position == sp_size - 1) {
        for (uint32_t i = 0; i < state_tiles; ++i) {
            noc_async_write_page(page(i), final_state, get_read_ptr(cb_eout) + i * tile_bytes);
        }
    }
    noc_async_write_barrier();
    // Wait until the sender has read the state out of cb_eout.
    volatile tt_l1_ptr uint32_t* acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_sem);
    noc_semaphore_wait_min(acked, 1);
    noc_semaphore_set(acked, 0);
    cb_pop_front(cb_eout, state_tiles);
    if (position == 0) {
        // The first rank's split tail enters at the final state, which the last rank sends here.
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(final_sem);
        noc_semaphore_wait_min(landed, kt);
        noc_semaphore_set(landed, 0);
        for (uint32_t i = 0; i < state_tiles; ++i) {
            noc_async_write_page(page(i), final_state, final_landing + i * tile_bytes);
        }
        noc_async_write_barrier();
    }
    noc_async_atomic_barrier();
}
"""

_SENDER = _CHRONOLOGY + r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

void kernel_main() {
    constexpr uint32_t cb_stage = get_compile_time_arg_val(0);
    constexpr uint32_t sp_rank = get_compile_time_arg_val(1);
    constexpr uint32_t sp_size = get_compile_time_arg_val(2);
    constexpr uint32_t local_rows = get_compile_time_arg_val(3);
    constexpr uint32_t unit_bytes = get_compile_time_arg_val(4);  // one tile column of a head's state
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(5);  // one fabric packet
    constexpr uint32_t kt = get_compile_time_arg_val(6);           // tile columns per state
    constexpr uint32_t slots = get_compile_time_arg_val(7);        // unit buffers
    constexpr auto start_args = TensorAccessorArgs<8>();
    constexpr uint32_t chunks = (unit_bytes + chunk_bytes - 1) / chunk_bytes;

    size_t a = 0;
    const uint32_t start_addr = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t done_sem = get_arg_val<uint32_t>(a++);        // on each head core
    const uint32_t eout_addr = get_arg_val<uint32_t>(a++);       // a head's finished state (same on every head core)
    const uint32_t ein_landing = get_arg_val<uint32_t>(a++);     // the successor head core's entry-state storage
    const uint32_t final_landing = get_arg_val<uint32_t>(a++);   // the first rank's final-state landing
    const uint32_t arrival_sem = get_arg_val<uint32_t>(a++);
    const uint32_t final_sem = get_arg_val<uint32_t>(a++);
    const uint32_t ack_sem = get_arg_val<uint32_t>(a++);
    const uint32_t num_heads = get_arg_val<uint32_t>(a++);       // entries: head core NoC x | y << 16
    const uint32_t heads_idx = a;
    a += num_heads;
    const uint32_t notify_idx = a;                                // then each head's count semaphore here
    a += num_heads;

    cb_reserve_back(cb_stage, 1);
    const uint32_t stage = get_write_ptr(cb_stage);
    const auto start = TensorAccessor(start_args, start_addr);
    noc_async_read(start.get_noc_addr(0), stage, sizeof(uint32_t));
    noc_async_read_barrier();
    const auto t = kda_chronology::derive(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[0], sp_rank, sp_size, local_rows);
    const uint32_t position = (t.rank + sp_size - t.first_rank) % sp_size;
    const bool last = position == sp_size - 1;
    const uint32_t landing = last ? final_landing : ein_landing;
    const uint32_t landed_sem = last ? final_sem : arrival_sem;

    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    // The relay's unit is one tile column of a head's state (column-major storage): the successor computes that
    // column of its state as soon as it lands. Units go through `slots` buffers: unit u + slots - 1 is read out of its
    // head core while earlier units' packets are on the link, and a slot (and its packet headers) is reused only
    // after its packets have left L1.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[slots][chunks];
    for (uint32_t b = 0; b < slots; ++b) {
        for (uint32_t c = 0; c < chunks; ++c) {
            hdrs[b][c] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[b][c], dst_chip_id, dst_mesh_id);
        }
    }
    conn.open();
    const uint32_t units = num_heads * kt;
    const auto head_xy = [&](uint32_t u) { return get_arg_val<uint32_t>(heads_idx + u / kt); };
    // Head k's writer counts its finished columns into notify_sems[k] on this core.
    const auto ready = [&](uint32_t u) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(notify_idx + u / kt))[0] >=
               u % kt + 1;
    };
    uint32_t fetched = 0;  // units whose reads are issued
    const auto fetch = [&](uint32_t u) {
        const uint32_t xy = head_xy(u);
        noc_async_read(
            get_noc_addr(xy & 0xFFFF, xy >> 16, eout_addr + (u % kt) * unit_bytes),
            stage + (u % slots) * unit_bytes,
            unit_bytes);
    };
    for (uint32_t u = 0; u < units; ++u) {
        // Issue every read the free slots allow, waiting only for unit u itself.
        while (fetched < units && fetched < u + slots && (fetched == u || ready(fetched))) {
            if (fetched == u) {
                while (!ready(u)) {
                }
            }
            if (fetched >= slots) {
                // Unit fetched - slots last used this slot: its packets must have left L1.
                noc_async_writes_flushed();
            }
            fetch(fetched);
            ++fetched;
        }
        noc_async_read_barrier();
        const uint32_t xy = head_xy(u);
        const uint32_t hx = xy & 0xFFFF;
        const uint32_t hy = xy >> 16;
        if (u % kt == kt - 1) {
            // The head's last column has left its head core: release the head.
            noc_semaphore_inc(get_noc_addr(hx, hy, ack_sem), 1);
        }
        const uint32_t buffer = stage + (u % slots) * unit_bytes;
        const uint32_t column = landing + (u % kt) * unit_bytes;
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t offset = c * chunk_bytes;
            const uint32_t bytes = offset + chunk_bytes <= unit_bytes ? chunk_bytes : unit_bytes - offset;
            const uint64_t dst = get_noc_addr(hx, hy, column + offset);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[u % slots][c];
            if (c + 1 == chunks) {
                hdr->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(hx, hy, landed_sem), 1, true}, bytes);
            } else {
                hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
            }
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(buffer + offset, bytes, reinterpret_cast<uint32_t>(hdr));
        }
    }
    noc_async_writes_flushed();
    // Reset the heads' counts for the next call (every head's columns have been counted).
    for (uint32_t k = 0; k < num_heads; ++k) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(notify_idx + k))[0] = 0;
    }
    conn.close();
    noc_async_atomic_barrier();
}
"""

_CB_A, _CB_B, _CB_EIN, _CB_EOUT, _CB_CHRON, _CB_STAGE = 0, 1, 2, 3, 4, 5
# Unit (tile column) buffers per sender.
_SLOTS = 4


class RingRelayPrefix:
    """Persistent state (landing buffers, semaphores) of the SP ring relay for one mesh and geometry."""

    _shared: dict = {}

    @classmethod
    def shared(cls, mesh_device, **kwargs) -> "RingRelayPrefix":
        """One relay per mesh and geometry: layers run one after another, so they share its L1 landing buffers."""
        if kwargs.get("cores") is not None:
            kwargs["cores"] = tuple((c.x, c.y) if hasattr(c, "x") else tuple(c) for c in kwargs["cores"])
        key = (id(mesh_device), tuple(sorted(kwargs.items(), key=lambda item: item[0])))
        relay = cls._shared.get(key)
        if relay is None or relay.mesh_device is not mesh_device:
            relay = cls._shared[key] = cls(mesh_device, **kwargs)
        return relay

    def __init__(
        self,
        mesh_device,
        *,
        sequence_parallel_axis,
        batch_heads,
        key_dim,
        value_dim,
        num_links,
        local_rows,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        cores=None,
    ):
        if key_dim != value_dim:
            raise ValueError("ring relay prefix requires K == V")
        self.mesh_device = mesh_device
        self.axis = sequence_parallel_axis
        self.batch_heads = batch_heads
        self.kt = key_dim // 32
        self.num_links = num_links
        self.local_rows = local_rows
        grid = mesh_device.compute_with_storage_grid_size()
        # One core per head, then one sender per link: the given cores (those another program merged into the relay
        # leaves idle), else the grid in row-major order.
        if cores is None:
            cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        cores = [c if isinstance(c, ttnn.CoreCoord) else ttnn.CoreCoord(*c) for c in cores]
        if batch_heads + num_links > len(cores):
            raise ValueError(f"ring relay prefix needs {batch_heads + num_links} cores, has {len(cores)}")
        self.head_cores = cores[:batch_heads]
        self.sender_cores = cores[batch_heads : batch_heads + num_links]
        # Merged into rectangles, which dispatch writes in one multicast each.
        merge = lambda cores: ttnn.CoreRangeSet([]).merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]))
        self.head_set = merge(self.head_cores)
        self.sender_set = merge(self.sender_cores)
        rows = self.kt * 32
        shard = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.head_set, (rows, rows), ttnn.ShardOrientation.ROW_MAJOR),
        )
        alloc = lambda: ttnn.allocate_tensor_on_device(
            ttnn.Shape([batch_heads * rows, rows]), ttnn.float32, ttnn.TILE_LAYOUT, mesh_device, shard
        )
        self.ein_landing = alloc()
        self.eout = alloc()
        self.final_landing = alloc()
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.sems = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(4)]
        # Per sender, one count per head it serves (the head's finished columns), written by the head writers.
        self.heads_per_link = -(-batch_heads // num_links)
        self.notify_sems = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(self.heads_per_link)]
        # The matmul runs at the affine prefix's fidelity, accumulating in an FP32 DST.
        self.compute_config = ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=True)
        ttnn.synchronize_device(mesh_device)

    def __call__(self, packed, initial_state, actual_start, companion=None):
        """companion: a callable returning (io_tensors, MeshProgramDescriptor) of a program on cores the relay leaves
        idle, merged into the relay's dispatch so it runs alongside. It is called after the relay's own outputs are
        allocated, so the L1 it plans around is the L1 live at the dispatch."""
        mesh = self.mesh_device
        rows, cols = tuple(mesh.shape)
        sp_size = tuple(mesh.shape)[self.axis]
        kt = self.kt
        state_tiles = kt * kt
        state_shape = ttnn.Shape([self.batch_heads, kt * 32, kt * 32])
        entry = ttnn.allocate_tensor_on_device(
            state_shape, ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        final = ttnn.allocate_tensor_on_device(
            state_shape, ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        if companion is not None:
            companion = companion()
        arrival, final_sem, ack, done = (int(ttnn.get_global_semaphore_address(s)) for s in self.sems)
        ein_addr = self.ein_landing.buffer_address()
        eout_addr = self.eout.buffer_address()
        final_landing = self.final_landing.buffer_address()
        chunk_bytes = min(3 * 4096, int(ttnn.get_tt_fabric_max_payload_size_bytes()) // 4096 * 4096)
        node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))
        virt = lambda c: mesh.worker_core_from_logical_core(c)

        desc = ttnn.MeshProgramDescriptor()
        for r in range(rows):
            for c in range(cols):
                coord = (r, c)
                sp_rank = coord[self.axis]
                succ = list(coord)
                succ[self.axis] = (sp_rank + 1) % sp_size
                succ = tuple(succ)
                program = ttnn.ProgramDescriptor()
                program.cbs = [
                    ttnn.CBDescriptor(
                        total_size=state_tiles * 2048,
                        core_ranges=self.head_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=_CB_A, data_format=ttnn.bfloat16, page_size=2048)
                        ],
                    ),
                    ttnn.CBDescriptor(
                        total_size=state_tiles * 2048,
                        core_ranges=self.head_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=_CB_B, data_format=ttnn.bfloat16, page_size=2048)
                        ],
                    ),
                    ttnn.cb_descriptor_from_sharded_tensor(_CB_EIN, self.ein_landing),
                    ttnn.cb_descriptor_from_sharded_tensor(_CB_EOUT, self.eout),
                    ttnn.CBDescriptor(
                        total_size=64,
                        core_ranges=self.head_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=_CB_CHRON, data_format=ttnn.uint32, page_size=64)
                        ],
                    ),
                    ttnn.CBDescriptor(
                        total_size=_SLOTS * kt * 4096,
                        core_ranges=self.sender_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=_CB_STAGE, data_format=ttnn.float32, page_size=_SLOTS * kt * 4096
                            )
                        ],
                    ),
                ]
                chron = [sp_rank, sp_size, self.local_rows]
                reader_ct = [_CB_A, _CB_B, _CB_EIN, _CB_CHRON, *chron, kt]
                for t in (packed, initial_state, actual_start):
                    reader_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
                writer_ct = [_CB_EOUT, _CB_CHRON, *chron, kt]
                for t in (entry, final):
                    writer_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
                sender_ct = [_CB_STAGE, *chron, kt * 4096, chunk_bytes, kt, _SLOTS] + list(
                    ttnn.TensorAccessorArgs(actual_start).get_compile_time_args()
                )
                rd, wr, sd = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                notify = [int(ttnn.get_global_semaphore_address(n)) for n in self.notify_sems]
                for h, core in enumerate(self.head_cores):
                    sender = virt(self.sender_cores[h % self.num_links])
                    rd[core.x][core.y] = [
                        packed.buffer_address(),
                        initial_state.buffer_address(),
                        actual_start.buffer_address(),
                        h,
                        arrival,
                    ]
                    wr[core.x][core.y] = [
                        entry.buffer_address(),
                        final.buffer_address(),
                        h,
                        ein_addr,
                        final_landing,
                        final_sem,
                        ack,
                        notify[h // self.num_links],
                        sender.x | (sender.y << 16),
                    ]
                src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
                program.kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=_HEAD_READER,
                        source_type=src,
                        core_ranges=self.head_set,
                        compile_time_args=reader_ct,
                        runtime_args=rd,
                        config=ttnn.ReaderConfigDescriptor(),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=_HEAD_WRITER,
                        source_type=src,
                        core_ranges=self.head_set,
                        compile_time_args=writer_ct,
                        runtime_args=wr,
                        config=ttnn.WriterConfigDescriptor(),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=_HEAD_COMPUTE,
                        source_type=src,
                        core_ranges=self.head_set,
                        compile_time_args=[_CB_A, _CB_B, _CB_EIN, _CB_EOUT, kt],
                        runtime_args=ttnn.RuntimeArgs(),
                        config=self.compute_config,
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=_SENDER,
                        source_type=src,
                        core_ranges=self.sender_set,
                        compile_time_args=sender_ct,
                        runtime_args=ttnn.RuntimeArgs(),
                        # NoC 0, away from the head writers' DRAM copies (the writer default, NoC 1): the
                        # merged output-gate matmul shares the NoCs, and the relay's hops are latency bound.
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                        ),
                    ),
                ]
                for l, score in enumerate(self.sender_cores):
                    heads = [h for h in range(self.batch_heads) if h % self.num_links == l]
                    entries = []
                    for h in heads:
                        v = virt(self.head_cores[h])
                        entries.append(v.x | (v.y << 16))
                    dst = node(succ)
                    args = (
                        [
                            actual_start.buffer_address(),
                            int(dst.mesh_id),
                            int(dst.chip_id),
                            done,
                            eout_addr,
                            ein_addr,
                            final_landing,
                            arrival,
                            final_sem,
                            ack,
                            len(entries),
                        ]
                        + entries
                        + notify[: len(entries)]
                    )
                    link = list(ttnn.get_forwarding_link_indices(node(coord), dst))[l]
                    args += list(ttnn.setup_fabric_connection(node(coord), dst, link, program, score))
                    sd[score.x][score.y] = args
                kernels = program.kernels
                kernels[3].runtime_args = sd
                program.kernels = kernels
                key = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))
                if companion is not None:
                    program = ttnn.merge_program_descriptors([program, companion[1][key]])
                desc[key] = program
        io = [packed, initial_state, actual_start, self.ein_landing, self.eout, self.final_landing]
        if companion is not None:
            io += list(companion[0])
        ttnn.generic_op(io + [entry, final], desc)
        return entry, final
