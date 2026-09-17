// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Marshaller: flatten a tt-metal Program/IDevice into the public POD. Reads mirror
// the runner's existing extraction (emulated_program_runner.cpp) 1:1 so the POD
// carries exactly what silicon feeds. STAGE 1: fields whose reads are simple getters
// are populated; callback-driven / firmware-layout fields (source, include paths,
// named-arg namespaces, defines, Metal-2.0 bindings, Quasar procs, KernelGroup
// launch offsets, DFB finalize offset, DRAM logical channel, fabric node id) are
// left default with TODO(stage2) — the descriptor is built-and-discarded this stage,
// so this preserves behavior; consumers and the remaining fields land in Stage 2.

#include "emule_descriptor_builder.hpp"

#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "emule_device_map.hpp"  // NOC_NODE_ID_BITS

#include <tt-metalium/device.hpp>
#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>  // is_2d_fabric_config

namespace tt_emule {

using namespace tt::tt_metal;

// ── SocView: mirrors populate_bank_mapping (1184-1284) + build_worker_coord_maps
//    (1289-1319) + the HAL/arch reads scattered in build_kernel_defines / setup_core_state.
SocView build_soc_view(IDevice* device) {
    auto& ctx = MetalContext::instance();
    auto& cluster = ctx.get_cluster();
    const auto& msoc = cluster.get_soc_desc(device->id());
    const auto& hw = ctx.hal();

    SocView v;
    v.arch = static_cast<uint32_t>(cluster.arch());
    v.fabric_config = static_cast<uint32_t>(ctx.get_fabric_config());
    v.fabric_2d = tt::tt_fabric::is_2d_fabric_config(ctx.get_fabric_config());

    // DRAM views: per-view [NOC0,NOC1] preferred worker coord + address offset.
    const uint32_t num_views = static_cast<uint32_t>(msoc.get_num_dram_views());
    for (uint32_t view = 0; view < num_views; ++view) {
        DramView dv;
        auto dc0 = msoc.get_preferred_worker_core_for_dram_view(static_cast<int>(view), 0);
        auto dc1 = msoc.get_preferred_worker_core_for_dram_view(static_cast<int>(view), 1);
        dv.noc_xy[0] = (static_cast<uint32_t>(dc0.y) << NOC_NODE_ID_BITS) | static_cast<uint32_t>(dc0.x);
        dv.noc_xy[1] = (static_cast<uint32_t>(dc1.y) << NOC_NODE_ID_BITS) | static_cast<uint32_t>(dc1.x);
        dv.address_offset = static_cast<uint32_t>(msoc.get_address_offset(static_cast<int>(view)));
        // TODO(stage2): logical_channel via umd translate_coord_to (build_core_map 2303-2323).
        v.dram_views.push_back(dv);
    }

    // L1 banks: logical core + virtual noc_xy per bank (allocator distribution).
    const auto& allocator = device->allocator();
    const uint32_t nbanks = allocator->get_num_banks(BufferType::L1);
    for (uint32_t b = 0; b < nbanks; ++b) {
        auto logical = allocator->get_logical_core_from_bank_id(b);
        auto virt = device->virtual_core_from_logical_core(logical, tt::CoreType::WORKER);
        L1Bank bank;
        bank.logical_x = logical.x;
        bank.logical_y = logical.y;
        bank.noc_xy = (static_cast<uint32_t>(virt.y) << NOC_NODE_ID_BITS) | static_cast<uint32_t>(virt.x);
        v.l1_banks.push_back(bank);
    }

    // Worker logical->virtual col/row maps (padded to 64, matching build_worker_coord_maps).
    auto grid = device->compute_with_storage_grid_size();
    v.worker_grid_x = grid.x;
    v.worker_grid_y = grid.y;
    for (uint32_t lx = 0; lx < grid.x && lx < 64; ++lx) {
        v.worker_col_to_virt[lx] =
            device->virtual_core_from_logical_core(tt::tt_metal::CoreCoord(lx, 0), tt::CoreType::WORKER).x;
    }
    for (uint32_t ly = 0; ly < grid.y && ly < 64; ++ly) {
        v.worker_row_to_virt[ly] =
            device->virtual_core_from_logical_core(tt::tt_metal::CoreCoord(0, ly), tt::CoreType::WORKER).y;
    }

    v.dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    v.l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    v.arch_num_circular_buffers = tt::tt_metal::hal::get_arch_num_circular_buffers();
    v.has_tile_counter_registers = hw.has_tile_counter_registers();

    const uint32_t pct_count = hw.get_programmable_core_type_count();
    for (uint32_t pct = 0; pct < pct_count; ++pct) {
        auto ct = hw.get_programmable_core_type(pct);
        PctInfo p;
        p.core_type = static_cast<uint32_t>(ct);
        // TODO(stage2): kernel_config_addr / _size / default_unreserved_addr / routing_table_addr.
        // These dev-addr/size reads are per-pct CONDITIONAL — e.g. get_dev_size(KERNEL_CONFIG)
        // asserts for TENSIX ("start of unreserved memory") — so mirror the guarded reads in
        // prepare_program / setup_core_state / check_program_metadata_size when consumers need them.
        v.pcts.push_back(p);
    }
    // TODO(stage2): mesh_id/chip_id via get_control_plane().get_fabric_node_id_from_physical_chip_id.
    return v;
}

// ── EmuleProgramDescriptor: mirrors collect_kernels / init_core_* / semaphores reads.
EmuleProgramDescriptor build_emule_descriptor(Program& program, IDevice* device) {
    (void)device;
    auto& impl = program.impl();  // non-const: get_kernels/get_kernel_groups/get_program_config_sizes
    const auto& hw = MetalContext::instance().hal();

    EmuleProgramDescriptor pd;
    pd.config.program_id = static_cast<uint64_t>(impl.get_id());
    // TODO(stage2): context_id (ContextId strong type — extract when setup_core_state consumes it).
    pd.config.config_sizes = impl.get_program_config_sizes();

    const uint32_t pct_count = hw.get_programmable_core_type_count();
    for (uint32_t pct = 0; pct < pct_count; ++pct) {
        pd.config.sem_offset.push_back(impl.get_program_config(pct).sem_offset);

        for (auto& [handle, kptr] : impl.get_kernels(pct)) {
            if (!kptr) {
                continue;
            }
            Kernel& k = *kptr;
            KernelDescriptor kd;
            kd.id = handle;
            kd.compile_time_args = k.compile_time_args();
            kd.named_compile_time_args = k.named_compile_time_args();
            kd.common_runtime_args = k.common_runtime_args();
            kd.programmable_core_type = static_cast<uint32_t>(k.get_kernel_programmable_core_type());
            kd.processor_class = static_cast<uint32_t>(k.get_kernel_processor_class());
            kd.processor_type = static_cast<uint32_t>(k.get_kernel_processor_type(0));
            // TODO(stage2): source, include_paths, named CT/RT namespaces, defines, is_compute,
            // dm_processor / Quasar proc_ids / num_threads, per-core unique RTA, Metal-2.0 bindings.
            pd.kernels.emplace(kd.id, std::move(kd));
        }

        for (auto& kg : impl.get_kernel_groups(pct)) {
            if (!kg) {
                continue;
            }
            KernelGroupDescriptor g;
            g.pct = pct;
            g.kernel_ids.assign(kg->kernel_ids.begin(), kg->kernel_ids.end());
            for (const auto& r : kg->core_ranges.ranges()) {
                g.core_ranges.push_back(
                    {static_cast<uint32_t>(r.start_coord.x),
                     static_cast<uint32_t>(r.start_coord.y),
                     static_cast<uint32_t>(r.end_coord.x),
                     static_cast<uint32_t>(r.end_coord.y)});
            }
            // TODO(stage2): kernel_config_base[pct] + per-processor rta/crta offsets via
            // kg->launch_msg.view().kernel_config() (collect_kernels 2051-2055).
            pd.kernel_groups.push_back(std::move(g));
        }
    }

    // Per-core CB / DFB / semaphore setup (init_core_cb_sync / allocate_dfbs_on_core / init_core_semaphores).
    for (const auto& core_vec : impl.logical_cores()) {
        for (const tt::tt_metal::CoreCoord& core : core_vec) {
            CoreDescriptor cs;
            cs.logical_x = core.x;
            cs.logical_y = core.y;

            for (const auto& cb : impl.circular_buffers_on_core(core)) {
                if (!cb) {
                    continue;
                }
                CbDescriptor cd;
                cd.address = cb->address();
                cd.total_size = cb->size();
                cd.globally_allocated = cb->globally_allocated();
                for (uint8_t idx : cb->local_buffer_indices()) {
                    CbBuffer b;
                    b.index = idx;
                    b.page_size = cb->page_size(idx);
                    b.num_pages = cb->num_pages(idx);
                    b.data_format = static_cast<uint32_t>(cb->data_format(idx));
                    if (const auto& t = cb->tile(idx)) {
                        b.tile = TileGeom{
                            t->get_height(),
                            t->get_width(),
                            static_cast<bool>(t->get_partial_face()),
                            static_cast<bool>(t->get_narrow_tile())};
                    }
                    if (const auto& f = cb->unpack_face_geometry(idx)) {
                        b.unpack_face = FaceGeom{f->face_r_dim, f->num_faces};
                    }
                    cd.buffers.push_back(std::move(b));
                }
                cs.cbs.push_back(std::move(cd));
            }

            for (const auto& dfb : impl.dataflow_buffers_on_core(core)) {
                if (!dfb) {
                    continue;
                }
                const auto& c = dfb->config;
                DfbDescriptor dd;
                dd.device_slot = dfb->device_slot;
                dd.entry_size = c.entry_size;
                dd.num_entries = c.num_entries;
                dd.num_producers = c.num_producers;
                dd.num_consumers = c.num_consumers;
                dd.producer_risc_mask = c.producer_risc_mask;
                dd.consumer_risc_mask = c.consumer_risc_mask;
                dd.cap = static_cast<AccessPattern>(static_cast<uint8_t>(c.cap));
                dd.data_format = static_cast<uint32_t>(c.data_format);
                // TODO(stage2): tile / unpack_face geometry, finalize L1 offset (core_lookup_).
                cs.dfbs.push_back(std::move(dd));
            }

            for (const auto& sem : impl.semaphores()) {
                if (sem.initialized_on_logical_core(core)) {
                    cs.semaphore_ids.push_back(sem.id());
                }
            }
            pd.cores.push_back(std::move(cs));
        }
    }

    for (const auto& sem : impl.semaphores()) {
        SemaphoreDescriptor sd;
        sd.id = sem.id();
        sd.initial_value = sem.initial_value();
        pd.semaphores.push_back(sd);
    }
    return pd;
}

}  // namespace tt_emule
