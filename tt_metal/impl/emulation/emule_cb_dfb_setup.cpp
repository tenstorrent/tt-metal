// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "emule_cb_dfb_setup.hpp"

#include <algorithm>

#include "emule_device_map.hpp"
#include "emule_kernel_defines.hpp"
#include "emule_sanitizers.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "hostdevcommon/fabric_common.h"
#include "llrt/metal_soc_descriptor.hpp"
#include "umd/device/chip/sw_emule_chip.hpp"
#include "tt_emule/tile_counter.hpp"
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::emule {

// Per-slot initialization data for a DFB tile-counter slot. wr_ptr and rd_ptr
// always start at the same position — producer-STRIDED/ALL at the per-slot
// offset, consumer-ALL at the sub-range base. The 4 DFB role combinations
// differ only in how these fields are computed (see fill_dfb_slots callers).
struct DfbSlotInit {
    uint8_t counter_id;
    uint32_t base_addr;
    uint32_t limit;
    uint32_t ptr;
};

template <typename SlotFn>
static void fill_dfb_slots(tt_emule::EmuleDFBInterface& iface, uint32_t n, SlotFn&& slot_fn) {
    const uint32_t cap = std::min<uint32_t>(n, tt_emule::MAX_TC_SLOTS_PER_DFB);
    for (uint32_t k = 0; k < cap; ++k) {
        auto& slot = iface.tc_slots[k];
        DfbSlotInit s = slot_fn(k);
        slot.neo_id = 0;
        slot.counter_id = s.counter_id;
        slot.base_addr = s.base_addr;
        slot.limit = s.limit;
        slot.wr_ptr = s.ptr;
        slot.rd_ptr = s.ptr;
    }
}

static void init_core_cb_sync(
    tt_emule::Core* core,
    detail::ProgramImpl& impl,
    const CoreCoord& logical_core,
    std::vector<uint64_t>& persistent_cb_ranges) {
    core->reset_cb_sync();
    // Record this core's globally-allocated (persistent) CB extents so Object-Intent
    // exempts kernel writes anywhere in them (§12). Separate pass so the exempt set
    // stays exactly the local CBs.
    for (auto& cb_impl : impl.circular_buffers_on_core(logical_core)) {
        if (cb_impl->globally_allocated()) {
            uint32_t start = cb_impl->address();
            persistent_cb_ranges.push_back((static_cast<uint64_t>(start) << 32) | (start + cb_impl->size()));
        }
    }

    bool configured[EMULE_NUM_CBS] = {};
    auto configure = [&](const std::shared_ptr<CircularBufferImpl>& cb_impl, const CoreCoord& lc) {
        for (uint8_t idx : cb_impl->local_buffer_indices()) {
            // Loud, not clamped: a silent skip leaves the CB's sync state uninitialised, which
            // resurfaces far away as wrong tile data.
            TT_FATAL(
                idx < EMULE_NUM_CBS,
                "CB index {} exceeds the emulated CB ceiling ({}); the host CircularBufferConfig must cap at the "
                "arch's NUM_CIRCULAR_BUFFERS.",
                idx,
                EMULE_NUM_CBS);
            if (configured[idx]) {
                continue;
            }
            uint32_t cb_addr = cb_impl->address();
            uint32_t page_size = cb_impl->page_size(idx);
            uint32_t num_pages = (page_size > 0) ? cb_impl->num_pages(idx) : 0;
            uint8_t* base = (page_size > 0) ? core->l1_ptr(cb_addr) : nullptr;
            // Face geometry is a compile-time descriptor on silicon and reaches the
            // kernel as a JIT define, not through this runtime CB state.
            core->init_cb_sync(idx, base, page_size, num_pages, cb_impl->globally_allocated());
            configured[idx] = true;
            log_debug(
                tt::LogMetal,
                "  Core({},{}) CB[{}]: addr=0x{:x} page_size={} num_pages={} base={:p}",
                lc.x,
                lc.y,
                idx,
                cb_addr,
                page_size,
                num_pages,
                (void*)base);
        }
    };
    // core_ranges-scoped, no global fill: blaze shares one cb_id across CBs on disjoint
    // grids, so binding a CB whose grid excludes this core would install the wrong
    // (addr, num_pages) for that shared cb_id.
    for (auto& cb_impl : impl.circular_buffers_on_core(logical_core)) {
        configure(cb_impl, logical_core);
    }
}

// Write semaphore initial values into L1 at the HAL-derived semaphore base.
static void init_core_semaphores(
    tt_emule::Core* core, detail::ProgramImpl& impl, const CoreCoord& logical_core, uint32_t emule_sem_base) {
    for (auto& sem : impl.semaphores()) {
        if (!sem.initialized_on_logical_core(logical_core)) {
            continue;
        }
        uint32_t sem_id = sem.id();
        uint32_t initial_value = sem.initial_value();
        uint32_t sem_addr = emule_sem_base + sem_id * EMULE_SEM_ALIGN;
        if (sem_addr + sizeof(uint32_t) > core->l1_size()) {
            continue;
        }
        auto* sem_ptr = reinterpret_cast<uint32_t*>(core->l1_ptr(sem_addr));
        *sem_ptr = initial_value;
        log_debug(
            tt::LogMetal,
            "  Core({},{}) Sem[{}]: addr=0x{:x} initial={}",
            logical_core.x,
            logical_core.y,
            sem_id,
            sem_addr,
            initial_value);
    }
}

// Allocate L1 for each DFB on a core, register CB-sync bridges, and — on Quasar —
// initialize tile counters. Returns per-DFB allocation info consumed by launch_cores.
static std::vector<DFBAllocInfo> allocate_dfbs_on_core(
    tt_emule::Core* core,
    const CoreCoord& logical_core,
    const std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>>& dfb_impls) {
    core->reset_dfb_sync();
    if (dfb_impls.empty()) {
        // Nothing to allocate, so the L1 bump allocator never grows and there's
        // nothing to reset. Skipping reset also leaves the mmap-init zeros at
        // MEM_ZEROS_BASE undisturbed for kernels that NOC-read the region.
        return {};
    }
    // Tile counters are Quasar hardware. On WH/BH a DFB is the CB the kernel-side
    // DataflowBuffer wraps, so init_cb_sync below is its whole sync state.
    // See tt-emule docs/DFB_EMULATION.md §1.
    const bool tc_backed = MetalContext::instance().hal().has_tile_counter_registers();
    // DFB fallback path: start the bump allocator at 0.  When Quasar bring-up needs
    // to protect MEM_ZEROS from bump-allocator overlap, dispatch its per-arch
    // MEM_ZEROS_BASE here.
    core->reset_l1_bump();
    if (tc_backed && !core->tile_counters()) {
        core->init_tile_counters(4);
    }

    std::vector<DFBAllocInfo> dfb_allocs;
    dfb_allocs.reserve(dfb_impls.size());
    // Compute bridge sharing: a compute-consumer input DFB and a compute-producer
    // output DFB with matching dimensions share L1 (real HW routes through the
    // register file). Independent compute-consumer inputs (e.g. matmul in0/in1)
    // must NOT share.
    constexpr uint16_t TENSIX_MASK = 0xFF00u;  // bits 8-15
    std::unordered_map<uint64_t, uint32_t> bridge_consumer_alloc;
    for (auto& dfb_impl : dfb_impls) {
        uint32_t device_slot = dfb_impl->device_slot;
        auto& cfg = dfb_impl->config;
        uint32_t total = cfg.entry_size * cfg.num_entries;
        uint64_t dim_key = (static_cast<uint64_t>(cfg.entry_size) << 32) | cfg.num_entries;
        bool compute_is_consumer = (cfg.consumer_risc_mask & TENSIX_MASK) != 0;
        bool compute_is_producer = (cfg.producer_risc_mask & TENSIX_MASK) != 0;
        // Prefer the finalize-allocated L1 offset (so host/test verification
        // hits the same offset); fall back to bump-alloc when absent. L1 offset
        // model: base_addr is a 0-based L1 offset (finalize supplies the offset
        // directly; l1_alloc returns one too). Use a found-flag, not addr != 0,
        // as the "has finalize" test — offset 0 is a valid L1 address.
        auto cl = dfb_impl->core_lookup_.find(logical_core);
        bool has_finalize = (cl != dfb_impl->core_lookup_.end());
        uint32_t finalize_addr = has_finalize ? cl->second.second : 0;  // 0-based L1 offset
        uint32_t base_addr;
        if (compute_is_producer && !compute_is_consumer) {
            auto it = bridge_consumer_alloc.find(dim_key);
            base_addr = (it != bridge_consumer_alloc.end()) ? it->second
                                                            : (has_finalize ? finalize_addr : core->l1_alloc(total));
        } else {
            base_addr = has_finalize ? finalize_addr : core->l1_alloc(total);
            if (compute_is_consumer && !compute_is_producer) {
                bridge_consumer_alloc.emplace(dim_key, base_addr);
            }
        }
        // base_addr is a 0-based L1 offset (L1 offset model); rebase onto this
        // core's L1 to get the host pointer the DFB/CB sync state stores.
        uint8_t* base = core->l1_data() + base_addr;
        // STRIDED: M = max(P, C); ALL: M = P.
        bool is_all = (cfg.cap == ::dfb::AccessPattern::ALL);
        uint32_t M = is_all ? cfg.num_producers : std::max<uint32_t>(cfg.num_producers, cfg.num_consumers);
        uint32_t capacity = cfg.num_entries / M;

        // Also populate CB sync state for this DFB so compute ops (pack_tile,
        // matmul_tiles) can reuse the same L1 buffer via cb_read_ptr/cb_write_ptr.
        // On WH/BH the slot is a CB index, so this IS the DFB's whole sync state.
        TT_FATAL(
            device_slot < EMULE_NUM_CBS,
            "DFB device slot {} exceeds the emulated CB ceiling ({}); the host assigns slots below the arch's "
            "NUM_CIRCULAR_BUFFERS ({}).",
            device_slot,
            EMULE_NUM_CBS,
            MetalContext::instance().hal().get_arch_num_circular_buffers());
        core->init_cb_sync(
            static_cast<uint8_t>(device_slot),
            base,
            cfg.entry_size,
            cfg.num_entries,
            /*globally_allocated=*/false);

        // STRIDED gets M TCs, ALL DM-DM gets P*C, spaced by MAX_TC_SLOTS_PER_DFB so DFBs cannot
        // collide. DFBSyncState belongs to the same model, so it is populated here too.
        if (tc_backed) {
            core->init_dfb_sync(device_slot, base, cfg.entry_size, cfg.num_entries, capacity);
            if (device_slot >= (tt_emule::TILE_COUNTERS_PER_NEO / tt_emule::MAX_TC_SLOTS_PER_DFB)) {
                // counter_base assigns out of NEO 0 only. Lifting this means spreading DFBs
                // across NEOs and threading neo_id through the CB->DFB bridge.
                throw std::out_of_range(
                    "Quasar DFB device slot exceeds safe TC range (max 8 DFBs per NEO with neo_id=0)");
            }
            uint8_t counter_base = static_cast<uint8_t>(device_slot * tt_emule::MAX_TC_SLOTS_PER_DFB);
            uint32_t num_tcs_to_init = is_all ? static_cast<uint32_t>(cfg.num_producers) * cfg.num_consumers : M;
            for (uint32_t tc_idx = 0; tc_idx < num_tcs_to_init; ++tc_idx) {
                auto& tc = core->tile_counters()->get(0, counter_base + static_cast<uint8_t>(tc_idx));
                tc.capacity = capacity;
                tc.posted.store(0, std::memory_order_relaxed);
                tc.acked.store(0, std::memory_order_relaxed);
            }
        }

        dfb_allocs.push_back({device_slot, base_addr, &cfg});
        log_debug(
            tt::LogMetal,
            "  Core({},{}) DFB[{}]: addr=0x{:x} entry_size={} num_entries={} total={}",
            logical_core.x,
            logical_core.y,
            device_slot,
            base_addr,
            cfg.entry_size,
            cfg.num_entries,
            total);
    }
    return dfb_allocs;
}

void setup_core_state(
    detail::ProgramImpl& impl,
    IDevice* device,
    tt::umd::SWEmuleChip* sw_emu,
    std::map<CoreCoord, std::vector<KernelInfo>>& core_kernels,
    uint32_t emule_sem_base,
    std::vector<CoreSetup>& core_setups) {
    auto& metal_ctx = MetalContext::instance(impl.get_context_id());
    const auto fabric_node = metal_ctx.get_control_plane().get_fabric_node_id_from_physical_chip_id(device->id());
    const uint32_t routing_table_base = static_cast<uint32_t>(
        metal_ctx.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::ROUTING_TABLE));
    for (auto& [logical_core, ki_list] : core_kernels) {
        if (!sw_emu) {
            continue;
        }
        auto phys = device->virtual_core_from_logical_core(logical_core, tt::CoreType::WORKER);
        tt_emule::Core* core = sw_emu->get_core(tt_xy_pair(phys.x, phys.y));
        if (!core) {
            continue;
        }
        // Fabric initialization writes this routing-table identity on silicon. Mirror it here because
        // SWEmule's launch-owned worker L1 does not retain the earlier control-plane broadcast.
        auto* routing_info = reinterpret_cast<tt::tt_fabric::routing_l1_info_t*>(core->l1_ptr(routing_table_base));
        routing_info->my_mesh_id = static_cast<uint16_t>(*fabric_node.mesh_id);
        routing_info->my_device_id = static_cast<uint16_t>(fabric_node.chip_id);
        uint8_t phys_x = static_cast<uint8_t>(phys.x);
        uint8_t phys_y = static_cast<uint8_t>(phys.y);

        std::vector<uint64_t> persistent_cb_ranges;
        init_core_cb_sync(core, impl, logical_core, persistent_cb_ranges);
        init_core_semaphores(core, impl, logical_core, emule_sem_base);

        auto dfb_impls = impl.dataflow_buffers_on_core(logical_core);
        // Quasar-only. Null on WH/BH keeps the cb_api CB->DFB bridge short-circuited, and stops
        // a slot legal up to get_arch_num_circular_buffers() indexing the MAX_DFBS-sized array.
        bool has_tc_dfbs = !dfb_impls.empty() && MetalContext::instance().hal().has_tile_counter_registers();
        std::vector<DFBAllocInfo> dfb_allocs = allocate_dfbs_on_core(core, logical_core, dfb_impls);

        uint32_t sem_region_size = tt::tt_metal::NUM_SEMAPHORES * EMULE_SEM_ALIGN;
        core_setups.push_back(
            {logical_core,
             core,
             &ki_list,
             phys_x,
             phys_y,
             std::move(dfb_allocs),
             has_tc_dfbs,
             emule_sem_base,
             sem_region_size,
             std::move(persistent_cb_ranges)});
    }
}

// ---------------------------------------------------------------------------
// Populate tile-counter slot state on a single EmuleDFBInterface for one
// (producer/consumer) × (STRIDED/ALL) role.  Pulled out of launch_cores
// because the 4-case slot-init block dominated the parent function.
// ---------------------------------------------------------------------------
static void populate_dfb_interface_slots(
    tt_emule::EmuleDFBInterface& iface, const DFBAllocInfo& alloc, uint8_t proc_id, bool is_tensix) {
    const auto& cfg = *alloc.cfg;
    const uint32_t total = cfg.entry_size * cfg.num_entries;

    // Compute proc_bit per-alloc: WH/BH ComputeKernel sets bit 2,
    // Quasar uses bits 8+ (detect by presence of high mask bits).
    uint16_t proc_bit;
    if (is_tensix) {
        bool quasar_masks = ((cfg.producer_risc_mask | cfg.consumer_risc_mask) & 0xFF00u) != 0;
        proc_bit = quasar_masks ? static_cast<uint16_t>(1u << (proc_id + ::dfb::TENSIX_RISC_OFFSET))
                                : static_cast<uint16_t>(1u << 2);
    } else {
        proc_bit = static_cast<uint16_t>(1u << proc_id);
    }
    bool is_all = (cfg.cap == ::dfb::AccessPattern::ALL);
    uint32_t M = is_all ? cfg.num_producers : std::max<uint32_t>(cfg.num_producers, cfg.num_consumers);
    uint32_t stride_size = M * cfg.entry_size;
    if (alloc.device_slot >= (tt_emule::TILE_COUNTERS_PER_NEO / tt_emule::MAX_TC_SLOTS_PER_DFB)) {
        return;
    }
    uint8_t counter_base = static_cast<uint8_t>(alloc.device_slot * tt_emule::MAX_TC_SLOTS_PER_DFB);

    bool is_producer = (cfg.producer_risc_mask & proc_bit) != 0;
    bool is_consumer = (cfg.consumer_risc_mask & proc_bit) != 0;
    if (!is_producer && !is_consumer) {
        return;
    }

    iface.active = true;
    iface.entry_size = cfg.entry_size;
    iface.stride_size = stride_size;
    iface.num_entries = cfg.num_entries;
    iface.tc_idx = 0;
    iface.broadcast_tc = false;
    iface.rd_entry_idx = 0;
    iface.wr_entry_idx = 0;

    if (is_producer) {
        uint8_t p = static_cast<uint8_t>(std::popcount(cfg.producer_risc_mask & (proc_bit - 1u)));
        if (is_all) {
            // ALL DM-DM: producer broadcasts to all consumer TCs.
            iface.broadcast_tc = true;
            iface.num_tcs_to_rr = static_cast<uint8_t>(cfg.num_consumers);
            iface.stride_size = cfg.entry_size;
            uint32_t capacity_per_p = cfg.num_entries / cfg.num_producers;
            uint32_t producer_ptr = alloc.base_addr + p * capacity_per_p * cfg.entry_size;
            fill_dfb_slots(iface, cfg.num_consumers, [&](uint32_t c) {
                return DfbSlotInit{
                    static_cast<uint8_t>(counter_base + p * cfg.num_consumers + c),
                    alloc.base_addr,
                    alloc.base_addr + total,
                    producer_ptr};
            });
        } else {
            uint32_t num_tcs = M / cfg.num_producers;
            iface.num_tcs_to_rr = static_cast<uint8_t>(num_tcs);
            fill_dfb_slots(iface, num_tcs, [&](uint32_t k) {
                uint8_t tc_idx = static_cast<uint8_t>(p + k * cfg.num_producers);
                return DfbSlotInit{
                    static_cast<uint8_t>(counter_base + tc_idx),
                    alloc.base_addr,
                    alloc.base_addr + total,
                    alloc.base_addr + tc_idx * cfg.entry_size};
            });
        }
    } else {
        uint8_t c = static_cast<uint8_t>(std::popcount(cfg.consumer_risc_mask & (proc_bit - 1u)));
        if (is_all) {
            // ALL DM-DM consumer: drain each producer's TC block fully.
            iface.num_tcs_to_rr = static_cast<uint8_t>(cfg.num_producers);
            iface.stride_size = cfg.entry_size;
            iface.drain_per_tc = true;
            uint32_t capacity_per_p = cfg.num_entries / cfg.num_producers;
            uint32_t sub_range = capacity_per_p * cfg.entry_size;
            fill_dfb_slots(iface, cfg.num_producers, [&](uint32_t p) {
                uint32_t sub_base = alloc.base_addr + p * sub_range;
                return DfbSlotInit{
                    static_cast<uint8_t>(counter_base + p * cfg.num_consumers + c),
                    sub_base,
                    sub_base + sub_range,
                    sub_base};
            });
        } else {
            uint32_t num_tcs = M / cfg.num_consumers;
            iface.num_tcs_to_rr = static_cast<uint8_t>(num_tcs);
            fill_dfb_slots(iface, num_tcs, [&](uint32_t k) {
                uint8_t tc_idx = static_cast<uint8_t>(c + k * cfg.num_consumers);
                return DfbSlotInit{
                    static_cast<uint8_t>(counter_base + tc_idx),
                    alloc.base_addr,
                    alloc.base_addr + total,
                    alloc.base_addr + tc_idx * cfg.entry_size};
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Build per-thread DFB interface arrays for one core.  Each kernel thread
// gets its own copy with independent wr/rd ptrs (matching real HW where each
// RISC has a separate LocalDFBInterface).
// ---------------------------------------------------------------------------
std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>> build_per_thread_dfb_interfaces(
    const std::vector<KernelInfo>& ki_list, const std::vector<DFBAllocInfo>& dfb_allocs) {
    std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>> per_thread_dfbs;
    per_thread_dfbs.resize(ki_list.size());
    for (size_t t = 0; t < ki_list.size(); t++) {
        per_thread_dfbs[t] = std::make_unique<tt_emule::EmuleDFBInterface[]>(tt_emule::MAX_DFBS);
        for (const auto& alloc : dfb_allocs) {
            if (alloc.device_slot >= tt_emule::MAX_DFBS) {
                continue;
            }
            populate_dfb_interface_slots(
                per_thread_dfbs[t][alloc.device_slot], alloc, ki_list[t].processor_id, ki_list[t].is_tensix);
        }
    }
    return per_thread_dfbs;
}

}  // namespace tt::tt_metal::emule
