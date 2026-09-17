// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "emule_kernel_defines.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

#include "emule_device_map.hpp"     // NUM_NOCS
#include "emule_sanitizers.hpp"     // EMULE_NUM_CBS
#include "emule_tile_geometry.hpp"  // resolve_tile_geometry, ResolvedTileGeometry
#include <tt-metalium/experimental/fabric/fabric.hpp>  // is_2d_fabric_config
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/tile.hpp>

namespace tt::tt_metal::emule {

namespace {
// Opt-in env-driven defines. Routed through the defines map (not a bare -D) so they land
// in both the JIT wrapper and the cache key — toggling one invalidates stale cached .so:
//   TT_EMULE_WAIT_TIMEOUT=1        -> EMULE_WAIT_TIMEOUT (bounded cv.wait_for + hang diagnostic)
//   TT_EMULE_DEEP_SFPU=sqrt,...    -> EMULE_DEEP_SFPU_<UPPER> (promote a shadowed SFPU op to the deep path)
void apply_env_defines(std::map<std::string, std::string>& defines) {
    if (std::getenv("TT_EMULE_WAIT_TIMEOUT")) {
        defines["EMULE_WAIT_TIMEOUT"] = "1";
    }
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
                    continue;
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
}
}  // namespace

std::map<std::string, std::string> build_kernel_defines_from_desc(
    const tt_emule::KernelDescriptor& kd,
    const tt_emule::CoreDescriptor* first_core_desc,
    const tt_emule::SocView& soc,
    uint32_t num_dram_channels,
    uint32_t num_l1_banks,
    const std::string& worker_col_map_str,
    const std::string& worker_row_map_str,
    uint32_t emule_sem_base) {
    std::map<std::string, std::string> defines;
    for (const auto& [k, v] : kd.defines) {
        defines[k] = v;
    }
    apply_env_defines(defines);

    if (soc.arch == static_cast<uint32_t>(ARCH::QUASAR)) {
        defines["ARCH_QUASAR"] = "1";
    } else if (soc.arch == static_cast<uint32_t>(ARCH::WORMHOLE_B0)) {
        defines["ARCH_WORMHOLE"] = "1";
    } else if (soc.arch == static_cast<uint32_t>(ARCH::BLACKHOLE)) {
        defines["ARCH_BLACKHOLE"] = "1";
    }

    {
        uint32_t num_dram = num_dram_channels ? num_dram_channels : 1;
        uint32_t num_l1 = num_l1_banks ? num_l1_banks : 1;
        defines["NUM_DRAM_BANKS"] = std::to_string(num_dram);
        defines["NUM_L1_BANKS"] = std::to_string(num_l1);
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
    if (soc.fabric_2d) {
        defines["EMULE_FABRIC_2D"] = "1";
    }
    defines["KERNEL_BUILD"] = "1";
    defines["DRAM_ALIGNMENT"] = std::to_string(soc.dram_alignment);
    defines["L1_ALIGNMENT"] = std::to_string(soc.l1_alignment);
    defines["EMULE_WORKER_COL_MAP"] = worker_col_map_str;
    defines["EMULE_WORKER_ROW_MAP"] = worker_row_map_str;
    {
        char buf[12];
        std::snprintf(buf, sizeof(buf), "0x%x", emule_sem_base);
        defines["EMULE_SEM_BASE"] = buf;
    }
    defines["EMULE_SEM_ALIGN"] = std::to_string(EMULE_SEM_ALIGN);

    // CB/DFB tile-size + shape tables — the marshaller already resolved the geometry
    // (silicon's precedence) into ResolvedGeom, so this only fills the per-slot arrays.
    if (first_core_desc != nullptr) {
        uint32_t tile_sizes[EMULE_NUM_CBS] = {};
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
        for (const auto& cb : first_core_desc->cbs) {
            for (const auto& b : cb.buffers) {
                const uint8_t idx = b.index;
                TT_FATAL(
                    idx < EMULE_NUM_CBS,
                    "CB index {} exceeds the emulated CB ceiling ({}); the host CircularBufferConfig must cap "
                    "at the arch's NUM_CIRCULAR_BUFFERS.",
                    idx,
                    EMULE_NUM_CBS);
                tile_sizes[idx] = b.geom.tile_size;
                cb_formats[idx] = static_cast<uint8_t>(b.data_format);
                tile_r_dim[idx] = b.geom.tile_r_dim;
                tile_c_dim[idx] = b.geom.tile_c_dim;
                face_r_dim[idx] = b.geom.face_r_dim;
                num_faces[idx] = b.geom.num_faces;
                partial_face[idx] = b.geom.partial_face;
                narrow_tile[idx] = b.geom.narrow_tile;
            }
        }
        for (const auto& dfb : first_core_desc->dfbs) {
            const uint32_t slot = dfb.device_slot;
            TT_FATAL(
                slot < EMULE_NUM_CBS,
                "DFB device slot {} exceeds the emulated CB ceiling ({}); the host assigns slots below the arch's "
                "NUM_CIRCULAR_BUFFERS ({}).",
                slot,
                EMULE_NUM_CBS,
                soc.arch_num_circular_buffers);
            if (dfb.data_format == static_cast<uint32_t>(tt::DataFormat::Invalid)) {
                continue;
            }
            tile_sizes[slot] = dfb.geom.tile_size;
            cb_formats[slot] = static_cast<uint8_t>(dfb.data_format);
            tile_r_dim[slot] = dfb.geom.tile_r_dim;
            tile_c_dim[slot] = dfb.geom.tile_c_dim;
            face_r_dim[slot] = dfb.geom.face_r_dim;
            num_faces[slot] = dfb.geom.num_faces;
            partial_face[slot] = dfb.geom.partial_face;
            narrow_tile[slot] = dfb.geom.narrow_tile;
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

    if (kd.is_compute && kd.has_compute_config) {
        defines["DST_ACCUM_MODE"] = kd.fp32_dest_acc_en ? "1" : "0";
        defines["ENABLE_FP32_DEST_ACC"] = kd.fp32_dest_acc_en ? "1" : "0";
        defines["DST_SYNC_FULL"] = kd.dst_full_sync_en ? "1" : "0";
    }
    return defines;
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
