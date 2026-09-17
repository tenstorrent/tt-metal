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
#include "emule_device_map.hpp"              // NOC_NODE_ID_BITS
#include "emule_kernel_defines.hpp"          // compute_proc_ids_and_thread_count, ProcIdList
#include "emule_tile_geometry.hpp"           // resolve_tile_geometry, ResolvedTileGeometry
#include "jit_build/jit_build_settings.hpp"  // NamedCTArgNamespaces, NamedRuntimeArgNamespaces
#include <tt-metalium/kernel_types.hpp>      // DataMovementConfig/ComputeConfig, DataMovementProcessor

#include <tt-metalium/device.hpp>
#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>  // is_2d_fabric_config
#include <tt_stl/assert.hpp>                           // TT_FATAL

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
    pd.config.context_id = static_cast<uint32_t>(impl.get_context_id().get());
    pd.config.config_sizes = impl.get_program_config_sizes();

    // Per (logical core) -> the kernels placed there with resolved launch offsets + unique RTA;
    // stitched into each CoreDescriptor below. Keyed by (logical_x, logical_y).
    std::map<std::pair<uint32_t, uint32_t>, std::vector<CoreKernel>> core_kernels;

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

            // source (raw KernelSource; path resolution stays in the consumer)
            const auto& ksrc = k.kernel_source();
            kd.source.is_file = (ksrc.source_type_ == KernelSource::FILE_PATH);
            kd.source.path = ksrc.path_;
            kd.source.inline_src = ksrc.source_;
            k.process_include_paths([&kd](const std::string& p) { kd.include_paths.push_back(p); });
            k.process_named_ct_arg_namespaces([&kd](const NamedCTArgNamespaces& ns) {
                for (const auto& [name, entries] : ns) {
                    kd.named_ct_arg_namespaces[name] = entries;
                }
            });
            k.process_named_runtime_args([&kd](const NamedRuntimeArgNamespaces& ns) {
                for (const auto& [name, entries] : ns) {
                    auto& out = kd.named_runtime_arg_namespaces[name];
                    for (const auto& e : entries) {
                        out.push_back(NamedRtEntry{e.field, e.index, e.length, static_cast<uint32_t>(e.dispatch)});
                    }
                }
            });
            k.process_defines([&kd](const std::string& dk, const std::string& dv) { kd.defines[dk] = dv; });
            kd.is_compute = (k.get_kernel_processor_class() == HalProcessorClassType::COMPUTE);
            {
                const auto cfg = k.config();
                if (const auto* dc = std::get_if<DataMovementConfig>(&cfg)) {
                    kd.dm_processor = static_cast<uint32_t>(dc->processor);
                }
                if (const auto* cc = std::get_if<ComputeConfig>(&cfg)) {
                    kd.has_compute_config = true;
                    kd.fp32_dest_acc_en = cc->fp32_dest_acc_en;
                    kd.dst_full_sync_en = cc->dst_full_sync_en;
                }
            }
            {
                auto* qdm = dynamic_cast<experimental::quasar::QuasarDataMovementKernel*>(&k);
                auto* qck = dynamic_cast<experimental::quasar::QuasarComputeKernel*>(&k);
                tt::tt_metal::emule::ProcIdList procs =
                    tt::tt_metal::emule::compute_proc_ids_and_thread_count(k, qdm, qck);
                kd.proc_ids.assign(procs.proc_ids.begin(), procs.proc_ids.end());
                kd.num_threads = procs.num_threads;
                kd.is_quasar_compute = kd.is_compute && (qck != nullptr);
            }
            // Metal 2.0 binding handles (mirror build_metal2_snapshot).
            kd.bindings.is_metal2 = k.is_metal2_kernel();
            kd.bindings.rta_names = k.get_runtime_arg_names();
            kd.bindings.crta_names = k.get_common_runtime_arg_names();
            k.process_dataflow_buffer_binding_handles(
                [&kd](const std::string& name, uint16_t id, bool is_relay, uint8_t pipe) {
                    kd.bindings.dfb.push_back(DfbBinding{name, id, is_relay, pipe});
                });
            k.process_semaphore_binding_handles(
                [&kd](const std::string& name, uint16_t id, auto scope, uint32_t harts) {
                    kd.bindings.sem.push_back(
                        SemBinding{name, id, static_cast<tt_emule::SemScope>(static_cast<uint8_t>(scope)), harts});
                });
            k.process_tensor_binding_handles(
                [&kd](const std::string& name, uint32_t cta_off, uint32_t addr_crta_off, uint32_t num_rt) {
                    // Emule doesn't yet model per-binding runtime CRTA words; the downstream
                    // get_common_vararg base math assumes 1 word/binding. Fail loudly on the
                    // dynamic-shape case here (the sole binding reader) rather than in a consumer.
                    TT_FATAL(
                        num_rt == 0,
                        "Emule does not yet support dynamic-shape Metal 2.0 tensor bindings "
                        "(binding '{}' has num_runtime_field_crta_words={}).",
                        name,
                        num_rt);
                    kd.bindings.tensor.push_back(TensorBinding{name, cta_off, addr_crta_off});
                });
            k.process_scratchpad_binding_handles(
                [&kd](const std::string& name, uint32_t size_bytes, uint32_t addr_crta_word) {
                    kd.bindings.scratch.push_back(ScratchBinding{name, size_bytes, addr_crta_word});
                });
            for (const auto& r : k.core_range_set().ranges()) {
                kd.core_ranges.push_back(
                    {static_cast<uint32_t>(r.start_coord.x),
                     static_cast<uint32_t>(r.start_coord.y),
                     static_cast<uint32_t>(r.end_coord.x),
                     static_cast<uint32_t>(r.end_coord.y)});
            }
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
            // KernelGroupDescriptor.kernel_config_base/proc_offsets are resolved per (kernel, core)
            // into CoreKernel below (the firmware launch_msg read), so collect_kernels needs no KG access.
            pd.kernel_groups.push_back(std::move(g));
        }

        // Resolve each (kernel, core) launch offset + unique RTA — the KernelGroup launch_msg read,
        // done here so collect_kernels needs no firmware/KG access. Mirrors collect_kernels' core loop:
        // key by (kernel, core) because a kernel on cores across KGs has distinct launch layouts.
        std::map<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>, KernelGroup*> k2kg;
        for (const auto& kg : impl.get_kernel_groups(pct)) {
            if (!kg) {
                continue;
            }
            for (const auto& cr : kg->core_ranges.ranges()) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; ++x) {
                    for (auto y = cr.start_coord.y; y <= cr.end_coord.y; ++y) {
                        for (auto kid : kg->kernel_ids) {
                            k2kg.emplace(
                                std::make_pair(
                                    static_cast<uint32_t>(kid),
                                    std::make_pair(static_cast<uint32_t>(x), static_cast<uint32_t>(y))),
                                kg.get());
                        }
                    }
                }
            }
        }
        for (auto& [kernel_id, kptr] : impl.get_kernels(pct)) {
            if (!kptr) {
                continue;
            }
            Kernel& k = *kptr;
            const uint32_t processor_index = hw.get_processor_index(
                k.get_kernel_programmable_core_type(), k.get_kernel_processor_class(), k.get_kernel_processor_type(0));
            for (const auto& cr : k.core_range_set().ranges()) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; ++x) {
                    for (auto y = cr.start_coord.y; y <= cr.end_coord.y; ++y) {
                        CoreKernel ck;
                        ck.kernel = static_cast<uint32_t>(kernel_id);
                        auto it = k2kg.find(std::make_pair(
                            static_cast<uint32_t>(kernel_id),
                            std::make_pair(static_cast<uint32_t>(x), static_cast<uint32_t>(y))));
                        if (it != k2kg.end()) {
                            auto kc = it->second->launch_msg.view().kernel_config();
                            ck.kernel_config_base = static_cast<uint32_t>(kc.kernel_config_base()[pct]);
                            auto rta = kc.rta_offset()[processor_index];
                            ck.rta_offset = rta.rta_offset();
                            ck.crta_offset = rta.crta_offset();
                        }
                        const tt::tt_metal::CoreCoord lc(x, y);
                        if (k.cores_with_runtime_args().count(lc) != 0) {
                            const auto& ra = k.runtime_args(lc);
                            ck.unique_rt_args.assign(ra.begin(), ra.end());
                        }
                        core_kernels[std::make_pair(static_cast<uint32_t>(x), static_cast<uint32_t>(y))].push_back(
                            std::move(ck));
                    }
                }
            }
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
                    const auto fmt = cb->data_format(idx);
                    b.data_format = static_cast<uint32_t>(fmt);
                    // Apply silicon's tile/face precedence here (marshaller has the live Tile);
                    // the POD carries only the resolved primitives. Mirrors build_kernel_defines.
                    const tt::tt_metal::emule::ResolvedTileGeometry g =
                        tt::tt_metal::emule::resolve_tile_geometry(cb->tile(idx), cb->unpack_face_geometry(idx));
                    b.geom = ResolvedGeom{
                        g.tile.get_tile_size(fmt),
                        g.tile.get_height(),
                        g.tile.get_width(),
                        g.face_r_dim,
                        g.num_faces,
                        g.partial_face,
                        g.narrow_tile};
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
                // Only valid-format DFBs feed the geometry tables (build_kernel_defines skips Invalid).
                if (c.data_format != tt::DataFormat::Invalid) {
                    const tt::tt_metal::emule::ResolvedTileGeometry g =
                        tt::tt_metal::emule::resolve_tile_geometry(c.tile, c.unpack_face_geometry);
                    dd.geom = ResolvedGeom{
                        g.tile.get_tile_size(c.data_format),
                        g.tile.get_height(),
                        g.tile.get_width(),
                        g.face_r_dim,
                        g.num_faces,
                        g.partial_face,
                        g.narrow_tile};
                }
                // TODO(stage2): finalize L1 offset (core_lookup_).
                cs.dfbs.push_back(std::move(dd));
            }

            for (const auto& sem : impl.semaphores()) {
                if (sem.initialized_on_logical_core(core)) {
                    cs.semaphore_ids.push_back(sem.id());
                }
            }
            auto ck_it =
                core_kernels.find(std::make_pair(static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)));
            if (ck_it != core_kernels.end()) {
                cs.kernels = std::move(ck_it->second);
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
