// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "emule_kernel_defines.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>

#include "emule_device_map.hpp"     // NUM_NOCS
#include "emule_sanitizers.hpp"     // EMULE_NUM_CBS
#include "emule_tile_geometry.hpp"  // resolve_tile_geometry, ResolvedTileGeometry
#include "impl/buffers/circular_buffer.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>  // is_2d_fabric_config
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/tile.hpp>

namespace tt::tt_metal::emule {

// Build the full defines map for a kernel: subclass-derived + arch + emulator
// constants (banking, alignments, worker maps, sem base, CB tile sizes).

std::map<std::string, std::string> build_kernel_defines(
    Kernel& kernel,
    detail::ProgramImpl& impl,
    uint32_t num_dram_channels,
    uint32_t num_l1_banks,
    const std::string& worker_col_map_str,
    const std::string& worker_row_map_str,
    uint32_t emule_sem_base) {
    std::map<std::string, std::string> defines;
    kernel.process_defines([&](const std::string& k, const std::string& v) { defines[k] = v; });

    // Opt-in deadlock-watchdog timeout. Off by default so <chrono> stays out of
    // the kernel include graph (~1s faster cold JIT compile; see
    // tt-emule include/jit_hw/emule_wait.h). Set TT_EMULE_WAIT_TIMEOUT=1 to
    // restore the bounded cv.wait_for + per-op hang diagnostic. Routed through
    // the defines map (not a bare -D) so it lands in both the wrapper and the
    // JIT cache key — toggling it invalidates stale cached .so files.
    if (std::getenv("TT_EMULE_WAIT_TIMEOUT")) {
        defines["EMULE_WAIT_TIMEOUT"] = "1";
    }

    // Opt-in deep-SFPU override. TT_EMULE_DEEP_SFPU=sqrt,sigmoid promotes those
    // shadowed SFPU ops from their layer-1 libm shadow to the deep path (the real
    // silicon ckernel_sfpu_<op>.h run on emule's faithful sfpi backend — see
    // tt-emule docs/sfpu-deep-path.md). Each comma-separated name becomes an
    // EMULE_DEEP_SFPU_<UPPER> define. Routed through the defines map so it lands
    // in the JIT cache key (toggling invalidates stale cached .so). Ops with no
    // layer-1 shadow take the deep path automatically and need no opt-in.
    if (const char* deep = std::getenv("TT_EMULE_DEEP_SFPU")) {
        const std::string list(deep);
        size_t start = 0;
        while (start <= list.size()) {
            const size_t comma = list.find(',', start);
            const size_t end = (comma == std::string::npos) ? list.size() : comma;
            std::string op;
            for (size_t i = start; i < end; ++i) {
                const char c = list[i];
                if (c == ' ' || c == '\t') {
                    continue;  // trim whitespace
                }
                op.push_back((c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c);
            }
            if (!op.empty()) {
                defines["EMULE_DEEP_SFPU_" + op] = "1";
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    }

    auto arch = MetalContext::instance().get_cluster().arch();
    if (arch == ARCH::QUASAR) {
        defines["ARCH_QUASAR"] = "1";
    } else if (arch == ARCH::WORMHOLE_B0) {
        defines["ARCH_WORMHOLE"] = "1";
    } else if (arch == ARCH::BLACKHOLE) {
        defines["ARCH_BLACKHOLE"] = "1";
    }

    {
        uint32_t num_dram = num_dram_channels ? num_dram_channels : 1;
        uint32_t num_l1 = num_l1_banks ? num_l1_banks : 1;
        defines["NUM_DRAM_BANKS"] = std::to_string(num_dram);
        defines["NUM_L1_BANKS"] = std::to_string(num_l1);
        // Mirror tt_metal/jit_build/build_env_manager.cpp:118-129. Upstream
        // `interleaved_addr_gen::get_bank_offset_index<DRAM>` chooses bit-shift
        // when banks are a power of two and a constant divisor otherwise.
        // Without these defines, non-pow2 bank counts (12 on WH-N150) silently
        // fall through to a 0-bit shift and every page lands in bank 0.
        auto is_pow2 = [](uint32_t n) { return n > 0 && (n & (n - 1)) == 0; };
        auto log2u = [](uint32_t n) {
            uint32_t l = 0;
            while ((1u << l) < n) {
                ++l;
            }
            return l;
        };
        if (is_pow2(num_dram)) {
            defines["LOG_BASE_2_OF_NUM_DRAM_BANKS"] = std::to_string(log2u(num_dram));
        } else {
            defines["IS_NOT_POW2_NUM_DRAM_BANKS"] = "1";
        }
        if (is_pow2(num_l1)) {
            defines["LOG_BASE_2_OF_NUM_L1_BANKS"] = std::to_string(log2u(num_l1));
        } else {
            defines["IS_NOT_POW2_NUM_L1_BANKS"] = "1";
        }
    }
    defines["NUM_NOCS"] = std::to_string(NUM_NOCS);
    // Fabric routing mode for the emule packet-header stamping shims. The real
    // fabric_set_line_unicast_route dispatches 1D-vs-2D on the header TYPE, but emule aliases
    // LowLatencyPacketHeader == HybridMeshPacketHeader (one 64B layout), so the shim cannot tell
    // them apart by type — it disambiguates on this build-mode define instead.
    const auto fabric_cfg = MetalContext::instance().get_fabric_config();
    if (tt::tt_fabric::is_2d_fabric_config(fabric_cfg)) {
        defines["EMULE_FABRIC_2D"] = "1";
    }
    // Upstream tensor/dspec.h gates `get_common_arg_addr` as a forward-decl
    // under KERNEL_BUILD; emule's jit_kernel_stubs.hpp provides the definition.
    // Without KERNEL_BUILD, dspec.h emits a stub that collides with emule's.
    defines["KERNEL_BUILD"] = "1";
    defines["DRAM_ALIGNMENT"] = std::to_string(hal::get_dram_alignment());
    defines["L1_ALIGNMENT"] = std::to_string(hal::get_l1_alignment());
    defines["EMULE_WORKER_COL_MAP"] = worker_col_map_str;
    defines["EMULE_WORKER_ROW_MAP"] = worker_row_map_str;
    {
        char buf[12];  // "0x" + max 8 hex digits + null
        std::snprintf(buf, sizeof(buf), "0x%x", emule_sem_base);
        defines["EMULE_SEM_BASE"] = buf;
    }
    defines["EMULE_SEM_ALIGN"] = std::to_string(EMULE_SEM_ALIGN);

    // Collect CB tile sizes + per-CB tile shape from program for the constexpr
    // get_tile_size() / get_tile_r_dim() / get_tile_c_dim() metadata. The shape
    // (height/width) is the ground truth for thin tiles (e.g. Tile([1,16])) —
    // the emulated reduce/unpack primitives bound their iteration by it instead
    // of assuming a full 32x32 tile. Default 32x32 when a CB has no Tile spec.
    const auto& core_range_set = kernel.core_range_set();
    if (!core_range_set.ranges().empty()) {
        auto first_core = core_range_set.ranges().begin()->start_coord;
        auto cb_impls = impl.circular_buffers_on_core(first_core);
        uint32_t tile_sizes[EMULE_NUM_CBS] = {};
        // Per-CB data format → emule's analog of genfiles.cpp::compute_data_formats()
        // (which bakes unpack_src_format[]/pack_dst_format[] into chlkc_descriptors.h).
        // 255 == tt::DataFormat::Invalid marks unconfigured slots (mirrors the host's
        // std::optional<DataFormat> empty state); consumers fall back to the page_size
        // heuristic for those. tile_r_dim/tile_c_dim carry the per-CB tile shape
        // (height/width) for thin tiles; default 32x32 when a CB has no Tile spec.
        uint8_t cb_formats[EMULE_NUM_CBS];
        uint32_t tile_r_dim[EMULE_NUM_CBS];
        uint32_t tile_c_dim[EMULE_NUM_CBS];
        uint32_t face_r_dim[EMULE_NUM_CBS];
        uint32_t num_faces[EMULE_NUM_CBS];
        uint32_t partial_face[EMULE_NUM_CBS];
        uint32_t narrow_tile[EMULE_NUM_CBS];
        for (uint32_t i = 0; i < EMULE_NUM_CBS; i++) {
            cb_formats[i] = static_cast<uint8_t>(tt::DataFormat::Invalid);
            tile_r_dim[i] = tt::constants::TILE_HEIGHT;
            tile_c_dim[i] = tt::constants::TILE_WIDTH;
            face_r_dim[i] = tt::constants::FACE_HEIGHT;
            num_faces[i] = tt::constants::TILE_HW / tt::constants::FACE_HW;
            partial_face[i] = 0;
            narrow_tile[i] = 0;
        }
        for (auto& cb_impl : cb_impls) {
            for (uint8_t idx : cb_impl->local_buffer_indices()) {
                TT_FATAL(
                    idx < EMULE_NUM_CBS,
                    "CB index {} exceeds the emulated CB ceiling ({}); the host CircularBufferConfig must cap "
                    "at the arch's NUM_CIRCULAR_BUFFERS.",
                    idx,
                    EMULE_NUM_CBS);
                // Same resolution silicon's JIT descriptor build uses, so the emulated
                // kernel sees the geometry a real kernel binary would be compiled against.
                const ResolvedTileGeometry geom =
                    resolve_tile_geometry(cb_impl->tile(idx), cb_impl->unpack_face_geometry(idx));
                tile_sizes[idx] = geom.tile.get_tile_size(cb_impl->data_format(idx));
                cb_formats[idx] = static_cast<uint8_t>(cb_impl->data_format(idx));
                tile_r_dim[idx] = geom.tile.get_height();
                tile_c_dim[idx] = geom.tile.get_width();
                face_r_dim[idx] = geom.face_r_dim;
                num_faces[idx] = geom.num_faces;
                partial_face[idx] = geom.partial_face;
                narrow_tile[idx] = geom.narrow_tile;
            }
        }
        // A DFB carries the same entry metadata at the same device slot, so it feeds the same
        // tables; without this ttnn::typecast reports page size 0 to its kernels.
        // See tt-emule docs/cb-dataformat.md.
        for (const auto& dfb_impl : impl.dataflow_buffers_on_core(first_core)) {
            const uint32_t slot = dfb_impl->device_slot;
            TT_FATAL(
                slot < EMULE_NUM_CBS,
                "DFB device slot {} exceeds the emulated CB ceiling ({}); the host assigns slots below the arch's "
                "NUM_CIRCULAR_BUFFERS ({}).",
                slot,
                EMULE_NUM_CBS,
                MetalContext::instance().hal().get_arch_num_circular_buffers());
            const auto& dfb_cfg = dfb_impl->config;
            // Derived like the CB pass above, not from entry_size: that is the NOC-facing entry
            // stride, which typecast deliberately aligns, and it belongs to the sync state only.
            // Invalid format is skipped for the same reason set_dfb_data_fmt_and_tile skips it.
            if (dfb_cfg.data_format == tt::DataFormat::Invalid) {
                continue;
            }
            const ResolvedTileGeometry dfb_geom = resolve_tile_geometry(dfb_cfg.tile, dfb_cfg.unpack_face_geometry);
            tile_sizes[slot] = dfb_geom.tile.get_tile_size(dfb_cfg.data_format);
            cb_formats[slot] = static_cast<uint8_t>(dfb_cfg.data_format);
            tile_r_dim[slot] = dfb_geom.tile.get_height();
            tile_c_dim[slot] = dfb_geom.tile.get_width();
            face_r_dim[slot] = dfb_geom.face_r_dim;
            num_faces[slot] = dfb_geom.num_faces;
            partial_face[slot] = dfb_geom.partial_face;
            narrow_tile[slot] = dfb_geom.narrow_tile;
        }
        std::ostringstream ts, df, tr, tc, fr, nf, pf, nt;
        for (uint32_t i = 0; i < EMULE_NUM_CBS; i++) {
            if (i) {
                ts << ',';
                df << ',';
                tr << ',';
                tc << ',';
                fr << ',';
                nf << ',';
                pf << ',';
                nt << ',';
            }
            ts << tile_sizes[i];
            df << static_cast<uint32_t>(cb_formats[i]);
            tr << tile_r_dim[i];
            tc << tile_c_dim[i];
            fr << face_r_dim[i];
            nf << num_faces[i];
            pf << partial_face[i];
            nt << narrow_tile[i];
        }
        defines["EMULE_TILE_SIZES"] = ts.str();
        defines["EMULE_CB_DATA_FORMATS"] = df.str();
        defines["EMULE_TILE_R_DIM"] = tr.str();
        defines["EMULE_TILE_C_DIM"] = tc.str();
        defines["EMULE_TILE_FACE_R_DIM"] = fr.str();
        defines["EMULE_TILE_NUM_FACES"] = nf.str();
        defines["EMULE_TILE_PARTIAL_FACE"] = pf.str();
        defines["EMULE_TILE_NARROW_TILE"] = nt.str();
    }

    // Thread the compute kernel's resolved fp32_dest_acc_en / dst_full_sync_en
    // into its TU, mirroring silicon genfiles.cpp::emit_compute_scalar_descriptors.
    // dest_helpers.hpp::DEST_AUTO_LIMIT must resolve identically in a program's
    // reader and compute kernels (e.g. multi-core H-reduce interleaves input
    // tiles in chunks of DEST_AUTO_LIMIT). The factory already injects
    // ENABLE_FP32_DEST_ACC/DST_SYNC_FULL into the reader's defines; without this
    // the compute TU falls back to the jit_kernel_stubs defaults (bf16/SyncFull
    // → 16) instead of the program's real mode, scrambling the chunked reduce.
    if (kernel.get_kernel_processor_class() == HalProcessorClassType::COMPUTE) {
        const auto kernel_config = kernel.config();
        if (const auto* cc = std::get_if<ComputeConfig>(&kernel_config)) {
            defines["DST_ACCUM_MODE"] = cc->fp32_dest_acc_en ? "1" : "0";
            defines["ENABLE_FP32_DEST_ACC"] = cc->fp32_dest_acc_en ? "1" : "0";
            defines["DST_SYNC_FULL"] = cc->dst_full_sync_en ? "1" : "0";
        }
    }
    return defines;
}

// Determine per-kernel thread count and the processor ids each thread runs as:
// - QuasarDataMovementKernel: one thread per DM processor (0..7).
// - QuasarComputeKernel: one thread per NEO engine (0..3), each running 4 TRISCs.
// - Other kernels: single thread at the kernel's processor type.
ProcIdList compute_proc_ids_and_thread_count(
    Kernel& kernel,
    experimental::quasar::QuasarDataMovementKernel* qdm,
    experimental::quasar::QuasarComputeKernel* qck) {
    ProcIdList out{};
    out.num_threads = 1;
    if (qdm && !qdm->get_dm_processors().empty()) {
        for (const auto& proc : qdm->get_dm_processors()) {
            out.proc_ids.push_back(
                static_cast<uint8_t>(static_cast<std::underlying_type_t<std::remove_cvref_t<decltype(proc)>>>(proc)));
        }
        out.num_threads = static_cast<uint32_t>(qdm->get_dm_processors().size());
    } else if (qck) {
        std::set<uint8_t> neo_ids_seen;
        for (const auto& proc : qck->get_compute_processors()) {
            uint8_t neo_id = static_cast<uint8_t>(
                static_cast<std::underlying_type_t<std::remove_cvref_t<decltype(proc)>>>(proc) /
                experimental::quasar::QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE);
            if (neo_ids_seen.insert(neo_id).second) {
                out.proc_ids.push_back(neo_id);
            }
        }
        out.num_threads = static_cast<uint32_t>(neo_ids_seen.size());
    } else {
        out.proc_ids.push_back(static_cast<uint8_t>(kernel.get_kernel_processor_type(0)));
    }
    return out;
}

// For Quasar compute kernels, scan the source for TRISC guards:
// - TRISC_UNPACK/MATH/PACK/ISOLATE_SFPU defines → compile 4 variants with each define set
// - else mentions TRISC_ID → single compile, run 4 times with different TRISC_ID
// - otherwise → single compile, single run
TriscMode detect_quasar_trisc_mode(bool is_quasar_compute, const std::string& src_path) {
    TriscMode mode;
    if (!is_quasar_compute) {
        return mode;
    }
    static const char* trisc_define_names[] = {"TRISC_UNPACK", "TRISC_MATH", "TRISC_PACK", "TRISC_ISOLATE_SFPU"};
    std::ifstream kscan(src_path);
    if (!kscan) {
        throw std::runtime_error("detect_quasar_trisc_mode: cannot read " + src_path);
    }
    std::string kcontent((std::istreambuf_iterator<char>(kscan)), std::istreambuf_iterator<char>());
    for (int t = 0; t < 4 && !mode.needs_trisc_compile; t++) {
        if (kcontent.find(trisc_define_names[t]) != std::string::npos) {
            mode.needs_trisc_compile = true;
        }
    }
    if (!mode.needs_trisc_compile && kcontent.find("TRISC_ID") != std::string::npos) {
        mode.needs_runtime_trisc = true;
    }
    return mode;
}

}  // namespace tt::tt_metal::emule
