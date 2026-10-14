// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/packet_injection/tensix_to_x280.hpp -- Tensix -> X280 packet injection
// characterization suite: Tensix placement, L1 allocation, producer kernels,
// runtime args and producer readback around the stages in common.hpp.

#ifndef X280_SUITES_PACKET_INJECTION_TENSIX_TO_X280_HPP
#define X280_SUITES_PACKET_INJECTION_TENSIX_TO_X280_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "harness/regions/release_selector.hpp"
#include "tests/device/suites/packet_injection/regions.hpp"
#include "harness/regions/regions.hpp"
#include "harness/regions/tensix_l1.hpp"
#include "harness/runner.hpp"
#include "harness/regions/release_guard.hpp"
#include "harness/metal_context.hpp"
#include "suites/packet_injection/common.hpp"

#include "kernels/packet_injection/packet_injection_abi.h"
#include "kernels/packet_injection/packet_injection_slot.hpp"
#include "rt/x280_noc_map.h"
#include "rt/x280_pinned_input.h"

#include <gtest/gtest.h>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace x280::suites::packet_injection::tensix_to_x280 {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;
namespace hmeas = x280::harness::measurement;
namespace hout = x280::harness::output;
namespace hreg = x280::harness::regions;

using namespace x280::harness::regions::packet_injection;

struct SweepPoint {
    uint32_t mode;
    uint32_t include_creation;
    const char* label;
    // Nonzero injects one worker-side validation failure and requires recovered cleanup.
    pi_error_t expected_worker_failure = PI_ERROR_NONE;
};
using Axis = SweepPoint;

inline std::vector<LogicalCoord> parse_tile_list(const std::string& str) {
    if (str.empty() || str.back() == ';') {
        throw std::runtime_error("empty tile in ordered tile list");
    }
    std::vector<LogicalCoord> tiles;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ';')) {
        if (item.empty()) {
            throw std::runtime_error("empty tile in ordered tile list");
        }
        size_t comma = item.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("invalid tile format: expected x,y got " + item);
        }
        uint32_t x = static_cast<uint32_t>(hart_::parse_u64_strict(item.substr(0, comma), "tile.x", UINT32_MAX));
        uint32_t y = static_cast<uint32_t>(hart_::parse_u64_strict(item.substr(comma + 1), "tile.y", UINT32_MAX));
        tiles.push_back(LogicalCoord{x, y});
    }
    return tiles;
}

struct ProducerKernels {
    ttm::KernelHandle brisc{};
    ttm::KernelHandle ncrisc{};
};

// Creates the Tensix producer kernels for `cfg` on `cores`, and is the only
// definition of the producer's defines and compiler include roots. `kernel_dir`
// holds packet_injection_tensix_to_x280.cpp. The returned `ncrisc` handle is
// meaningful only when `cfg.family == WorkerFamily::Paired`; it is default
// constructed otherwise. Creation records the kernel on `program`; the JIT
// compile happens later. Throws on an empty core set or on a CreateKernel
// failure.
inline ProducerKernels create_producer_kernels(
    ttm::Program& program, const Config& cfg, const std::filesystem::path& kernel_dir, const ttm::CoreRangeSet& cores) {
    if (cores.num_cores() == 0) {
        throw std::runtime_error("create_producer_kernels requires at least one core");
    }

    std::map<std::string, std::string> kernel_defines = {
        {"PACKET_INJECTION_MODE", std::to_string(cfg.mode)},
        {"PACKET_INJECTION_INCLUDE_CREATION", std::to_string(cfg.include_creation)}};

    const std::vector<std::filesystem::path> compiler_includes = {
        hart_::shared_kernel_dir(kernel_dir, ""),
        hart_::shared_kernel_dir(kernel_dir, "packet_injection"),
        hart_::shared_kernel_dir(kernel_dir, "shared"),
        hart_::shared_kernel_dir(kernel_dir, "pipeline_stage"),
        hart_::shared_kernel_dir(kernel_dir, "../include").lexically_normal(),
        kernel_dir};

    const auto source = kernel_dir / "packet_injection_tensix_to_x280.cpp";

    ProducerKernels kernels{};
    kernels.brisc = ttm::CreateKernel(
        program,
        source,
        cores,
        ttm::DataMovementConfig{
            .processor = ttm::DataMovementProcessor::RISCV_0,
            .noc = ttm::NOC::NOC_0,
            .noc_mode = ttm::NOC_MODE::DM_DEDICATED_NOC,
            .defines = kernel_defines,
            .compiler_include_paths = compiler_includes});

    if (cfg.family == WorkerFamily::Paired) {
        kernels.ncrisc = ttm::CreateKernel(
            program,
            source,
            cores,
            ttm::DataMovementConfig{
                .processor = ttm::DataMovementProcessor::RISCV_1,
                .noc = ttm::NOC::NOC_1,
                .noc_mode = ttm::NOC_MODE::DM_DEDICATED_NOC,
                .defines = kernel_defines,
                .compiler_include_paths = compiler_includes});
    }

    return kernels;
}

inline void run_axis(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, const Axis& axis) {
    auto* clock_device = f.mesh->get_device(0, 0);
    const uint64_t expected_aiclk_hz =
        x280::harness::execution::programmed_aiclk_hz(x280::harness::cluster(), *clock_device);

    // Prelaunch admission: ensure loaded image matches parameterized selector
    std::string expected_sub;
    if (axis.mode == PI_MODE_LATENCY) {
        expected_sub = axis.include_creation ? "latency-include" : "latency-exclude";
    } else if (axis.mode == PI_MODE_THROUGHPUT) {
        expected_sub = "throughput";
    } else {
        expected_sub = axis.include_creation ? "backpressure-include" : "backpressure-exclude";
    }
    const std::string default_image = "firmware-packet-injection-" + expected_sub + ".bin";
    const std::string fw_img = hart_::image_path(spec.open.image_env_var, default_image.c_str());
    if (fw_img.find(expected_sub) == std::string::npos) {
        GTEST_SKIP() << "Image " << fw_img << " does not match test selector " << (axis.label ? axis.label : "")
                     << " (expected '" << expected_sub << "')";
    }

    Config cfg{};
    cfg.mode = axis.mode;
    cfg.include_creation = axis.include_creation;

    // Parse runtime environment variables
    std::string s_val = hart_::env_str("PACKET_INJECTION_S");
    if (!s_val.empty()) {
        cfg.S = static_cast<uint32_t>(hart_::parse_u64_strict(s_val, "PACKET_INJECTION_S", UINT32_MAX));
    }

    std::string b_val = hart_::env_str("PACKET_INJECTION_B");
    if (!b_val.empty()) {
        cfg.B = static_cast<uint32_t>(hart_::parse_u64_strict(b_val, "PACKET_INJECTION_B", UINT32_MAX));
    }

    std::string c_val = hart_::env_str("PACKET_INJECTION_C");
    if (!c_val.empty()) {
        cfg.C = static_cast<uint32_t>(hart_::parse_u64_strict(c_val, "PACKET_INJECTION_C", UINT32_MAX));
    }

    std::string k_val = hart_::env_str("PACKET_INJECTION_K");
    if (!k_val.empty()) {
        cfg.K = static_cast<uint32_t>(hart_::parse_u64_strict(k_val, "PACKET_INJECTION_K", UINT32_MAX));
    }

    std::string w_val = hart_::env_str("PACKET_INJECTION_W");
    if (!w_val.empty()) {
        cfg.W = hart_::parse_u64_strict(w_val, "PACKET_INJECTION_W");
    }

    std::string t_val = hart_::env_str("PACKET_INJECTION_T");
    if (!t_val.empty()) {
        cfg.tile_count = static_cast<uint32_t>(hart_::parse_u64_strict(t_val, "PACKET_INJECTION_T", UINT32_MAX));
    }

    std::string fam_val = hart_::env_str("PACKET_INJECTION_FAMILY");
    if (!fam_val.empty()) {
        if (fam_val == "brisc") {
            cfg.family = WorkerFamily::BriscOnly;
        } else if (fam_val == "paired") {
            cfg.family = WorkerFamily::Paired;
        } else {
            throw std::runtime_error("PACKET_INJECTION_FAMILY must be 'brisc' or 'paired', got: " + fam_val);
        }
    }

    std::string n_val = hart_::env_str("PACKET_INJECTION_N");
    if (!n_val.empty()) {
        cfg.N = hart_::parse_u64_strict(n_val, "PACKET_INJECTION_N");
    } else {
        if (cfg.mode == PI_MODE_THROUGHPUT) {
            cfg.N = 100000ULL;
        } else {
            cfg.N = 4096ULL;
        }
    }

    std::string tiles_val = hart_::env_str("PACKET_INJECTION_TILES");
    if (!tiles_val.empty()) {
        cfg.ordered_tiles = parse_tile_list(tiles_val);
    } else {
        cfg.ordered_tiles.push_back(LogicalCoord{0, 0});
    }

    std::string chip_val = hart_::env_str("PACKET_INJECTION_DST_CHIP");
    if (!chip_val.empty()) {
        cfg.dst_chip =
            static_cast<uint16_t>(hart_::parse_u64_strict(chip_val, "PACKET_INJECTION_DST_CHIP", UINT16_MAX));
    }

    std::string mesh_val = hart_::env_str("PACKET_INJECTION_DST_MESH");
    if (!mesh_val.empty()) {
        cfg.dst_mesh =
            static_cast<uint16_t>(hart_::parse_u64_strict(mesh_val, "PACKET_INJECTION_DST_MESH", UINT16_MAX));
    }

    std::string poll_val = hart_::env_str("PACKET_INJECTION_POLL_CADENCE");
    if (!poll_val.empty()) {
        cfg.poll_cadence =
            static_cast<uint32_t>(hart_::parse_u64_strict(poll_val, "PACKET_INJECTION_POLL_CADENCE", UINT32_MAX));
    }

    std::string phase_to_val = hart_::env_str("PACKET_INJECTION_PHASE_TIMEOUT_MS");
    if (!phase_to_val.empty()) {
        cfg.phase_timeout_ms = hart_::parse_u64_strict(phase_to_val, "PACKET_INJECTION_PHASE_TIMEOUT_MS");
    }

    std::string host_to_val = hart_::env_str("PACKET_INJECTION_HOST_TIMEOUT_MS");
    if (!host_to_val.empty()) {
        cfg.host_timeout_ms = hart_::parse_u64_strict(host_to_val, "PACKET_INJECTION_HOST_TIMEOUT_MS");
    }

    std::string bridge_hist = hart_::env_str("PACKET_INJECTION_BRIDGE_HISTORY");
    if (bridge_hist.empty()) {
        bridge_hist = "unknown";
    }
    if (bridge_hist == "mode3_observed") {
        throw std::runtime_error("PACKET_INJECTION_BRIDGE_HISTORY=mode3_observed: ASIC recovery required before reuse");
    }
    if (bridge_hist != "unknown" && bridge_hist != "reported_no_mode3_since_reset") {
        throw std::runtime_error("invalid PACKET_INJECTION_BRIDGE_HISTORY: " + bridge_hist);
    }

    const std::string retain_test = hart_::env_str("PACKET_INJECTION_TEST_RETAIN_UNANNOUNCED_FAILURE");
    if (!retain_test.empty() && retain_test != "1") {
        throw std::runtime_error("PACKET_INJECTION_TEST_RETAIN_UNANNOUNCED_FAILURE must be '1' when set");
    }
    const bool inject_unannounced_failure = retain_test == "1";

    // The host owns both clocks: the device ABI no longer carries either.
    const uint64_t tensix_hz = expected_aiclk_hz;
    const uint64_t mtime_hz = x280::harness::measurement::kX280MtimeHz;

    Capacity cap{};
    cap.lim_backed_bytes = f.tile->inspect().lim_backed_bytes;
    cap.l1_span_limit = f.mesh->get_device(0, 0)->l1_size_per_core();
    cap.l1_window_base = f.mesh->get_device(0, 0)->allocator()->get_base_allocator_addr(ttm::HalMemType::L1);
    cap.producer_timer_hz = tensix_hz;
    cap.mtime_hz = mtime_hz;

    // Sizing admission: l1_tile_base is not known until the L1 pages exist, and
    // only that base and the 32-bit extent bound depend on it.
    const Admission sizing = validate_config(cfg, cap);
    if (!sizing.ok()) {
        throw std::runtime_error("Configuration rejected: " + sizing.describe());
    }
    const uint32_t P = sizing.layout.abi.worker_count;
    const uint32_t T = sizing.layout.abi.tile_count;

    uint32_t nonce = 0;
    uint64_t run_value = mint_run_value(nonce);
    if (run_value == 0u) {
        throw std::runtime_error("minted run value is zero, which X280 cannot distinguish from an unacked collection");
    }

    // Verify worker coordinates against mesh
    // MeshDevice, not Device: Device::worker_cores is an unconditional TT_FATAL
    // at this tt-metal revision, so the per-device spelling aborts every run.
    const auto worker_cores = f.mesh->worker_cores(ttm::HalProgrammableCoreType::TENSIX, ttm::SubDeviceId{0});
    for (uint32_t t = 0; t < T; ++t) {
        ttm::CoreCoord c{cfg.ordered_tiles[t].x, cfg.ordered_tiles[t].y};
        if (!worker_cores.contains(c)) {
            throw std::runtime_error(
                "Logical tile (" + std::to_string(c.x) + "," + std::to_string(c.y) +
                ") is not an available worker core");
        }
    }

    // One height-sharded L1 buffer over the selected tiles, one page each at a
    // common base. The set is built from singleton CoreRanges in ordered-tile
    // order: the CoreRange constructor preserves that order, whereas the
    // CoreCoord one merges and coalesces, which would silently permute the page
    // order the collection below relies on.
    std::vector<ttm::CoreRange> tile_ranges;
    tile_ranges.reserve(T);
    for (uint32_t t = 0; t < T; ++t) {
        const ttm::CoreCoord c{cfg.ordered_tiles[t].x, cfg.ordered_tiles[t].y};
        tile_ranges.push_back(ttm::CoreRange(c, c));
    }
    const ttm::CoreRangeSet tile_set(tile_ranges);
    auto tile_l1 = hreg::allocate_tile_l1_buffer(f.mesh.get(), tile_set, sizing.layout.abi.l1_tile_bytes);

    // Re-admit at the allocated base, which proves its alignment and that the
    // per-tile extent stays inside the 32-bit L1 address space.
    const Admission placed = validate_config(cfg, cap, tile_l1.l1_tile_base);
    if (!placed.ok()) {
        throw std::runtime_error("Allocated L1 base rejected: " + placed.describe());
    }
    const pi_layout_t& L = placed.layout.abi;

    // Each producer addresses the L2CPU on its own NoC, indexed by
    // pi_worker_risc_index: BRISC on NoC 0, NCRISC on NoC 1.
    const x280::TranslatedCoord l2cpu_by_risc[2] = {
        f.runtime->virtual_core_from_tile(f.tile_id, ttm::NOC::NOC_0),
        f.runtime->virtual_core_from_tile(f.tile_id, ttm::NOC::NOC_1)};

    // The X280 consumer releases the tiles through its NoC 0 TLB window.
    std::vector<hreg::ReleaseParticipant> participants;
    participants.reserve(T);
    for (const auto& tile : cfg.ordered_tiles) {
        participants.push_back({tt::CoreType::TENSIX, ttm::CoreCoord{tile.x, tile.y}});
    }
    const auto plan = hreg::select_release(*f.mesh, participants, hreg::kX280ReleaseSender, hreg::ReleaseMode::Auto, T);

    const uint32_t abi_tag = PI_ABI_TAG(cfg.mode, cfg.include_creation);
    const uint32_t assumption_mask = PI_ASSUMED_INBOUND | PI_ASSUMED_COUNTER_ATOMICITY | PI_ASSUMED_BRIDGE_COMPLETION |
                                     PI_ASSUMED_MTIME_HZ | PI_ASSUMED_LOCAL_TIMER_ORDER |
                                     (cfg.mode == PI_MODE_LATENCY ? PI_ASSUMED_PIN_EXPORT : 0u);

    const pi_x280_args_t xargs = make_x280_args(cfg, P, T, L, abi_tag, placed.x280_timeout_ticks, nonce, plan.group);
    prime_lim_arena(*f.tile, L, xargs, plan.translated);

    // Build Metal program and workload
    ttmd::MeshWorkload workload;
    ttm::Program program;

    const ProducerKernels kernels = create_producer_kernels(program, cfg, spec.open.kernel_dir, tile_l1.worker_set);

    // Set per-core runtime args
    for (uint32_t w = 0; w < P; ++w) {
        const LogicalCoord& tile = cfg.ordered_tiles[pi_worker_tile_index(&L, w)];
        ttm::CoreCoord c{tile.x, tile.y};

        pi_producer_args_t targs = make_producer_args(
            cfg, P, T, L, abi_tag, nonce, w, l2cpu_by_risc[pi_worker_risc_index(&L, w)], placed.producer_timeout_ticks);
        if (axis.expected_worker_failure != PI_ERROR_NONE && w == 0u) {
            if (inject_unannounced_failure) {
                targs.peer.x = X280_NOC_COORD_MAX + 1u;
            } else {
                targs.poll_cadence = 0u;
            }
        }

        // Same construct the kernel reconstructs through: the union's asserted
        // overlay is the serialization, so there is no second copy to keep in
        // step with the field offsets.
        pi_producer_arg_words_t block{};
        block.args = targs;
        std::vector<uint32_t> rt_args(block.words, block.words + PI_PRODUCER_ARG_WORDS);
        if (pi_worker_risc_index(&L, w) == 0) {
            ttm::SetRuntimeArgs(program, kernels.brisc, c, rt_args);
        } else {
            ttm::SetRuntimeArgs(program, kernels.ncrisc, c, rt_args);
        }
    }

    workload.add_program(ttmd::MeshCoordinateRange(f.mesh->shape()), std::move(program));

    // A multicast release must reach only its tiles: every other compute-grid
    // core holds a guard word at the release address through the run. Armed
    // before the AICLK start read and checked after the end read.
    const uint32_t release_addr = L.l1_tile_base + L.l1_release_offset;
    const std::vector<ttm::CoreCoord> release_guards =
        plan.group ? x280::harness::regions::guards_outside(*f.mesh, tile_set) : std::vector<ttm::CoreCoord>{};
    x280::harness::regions::write_guards(*f.mesh, release_guards, release_addr);

    const hexec::X280Launch launch{
        .image_path = fw_img,
        .run_value = run_value,
        .args_addr = X280_LIM_PACKET_INJECTION_ARENA_BASE + L.args_offset,
        .args_bytes = sizeof(pi_x280_args_t),
        .cleanup_timeout = std::chrono::seconds(5),
    };
    const auto reference_core = clock_device->worker_core_from_logical_core(ttm::CoreCoord{0, 0});
    const auto clock_start =
        x280::harness::execution::read_aiclk_start(x280::harness::cluster(), *clock_device, reference_core);
    ASSERT_TRUE(clock_start.errors.empty()) << "pre-run AICLK evidence is unavailable";
    hmeas::AiclkEndpoint clock_end;
    const size_t l1_tile_words = L.l1_tile_bytes / sizeof(uint32_t);
    const size_t l1_dump_words = static_cast<size_t>(T) * l1_tile_words;
    std::vector<uint32_t> host_l1_dump;
    bool resources_safe = false;
    const auto host_start = std::chrono::steady_clock::now();
    const auto host_deadline = std::chrono::milliseconds(cfg.host_timeout_ms);
    try {
        hexec::enqueue_tensix(*f.mesh, workload);
        ConsumerRun x280_run;
        enqueue_consumer(f, launch, x280_run);
        await_quiescence(f, L, P, abi_tag, run_value, host_start + host_deadline, x280_run);

        collect_within_deadline(x280_run, [&]() {
            hexec::finish_tensix(*f.mesh);
            clock_end =
                x280::harness::execution::read_aiclk_end(x280::harness::cluster(), *clock_device, reference_core);
            host_l1_dump.resize(l1_dump_words);
            ttmd::EnqueueReadMeshBuffer(
                f.mesh->mesh_command_queue(),
                host_l1_dump,
                tile_l1.buffer,
                /*blocking=*/true);
            if (host_l1_dump.size() != l1_dump_words) {
                throw std::runtime_error("short producer L1 read");
            }
        });

        const auto clock = hmeas::assess_aiclk_run(expected_aiclk_hz, clock_start, clock_end);
        const std::vector<x280::harness::regions::OverwrittenGuard> overwritten_guards =
            x280::harness::regions::overwritten_guards(*f.mesh, release_guards, release_addr);

        if (x280_run.argument_rejection) {
            reject_arguments(f, x280_run, resources_safe);
        }

        // Read valid latency export samples from LIM with length validation. The
        // export is tiled by the layout's own per-worker stride, so the host slices
        // it with that field rather than re-deriving the multiply.
        const uint64_t sample_stride = L.latency_sample_stride / sizeof(uint64_t);
        const std::vector<uint64_t> export_samples = read_latency_export(*f.tile, cfg.mode, L);

        // Read the X280-owned consumer slots from LIM: the host's second, independent
        // view of per-worker progress.
        const std::vector<pi_consumer_slot_t> consumer_slots = read_consumer_slots(*f.tile, L, P);

        // Decode the producer-owned results out of the single L1 dump. Page i of the
        // shared buffer is ordered tile i, so worker w sits at page
        // (w / risc_per_tile), slice (w % risc_per_tile). Nothing on the host can
        // prove that page order without silicon, so each slice must identify itself:
        // a permutation here would silently attribute one tile's counters to another.
        std::vector<pi_producer_result_t> producer_res(P);
        std::vector<std::vector<uint64_t>> producer_stalls(P);
        std::vector<std::vector<uint64_t>> producer_rtts(P);
        std::vector<std::vector<uint64_t>> producer_stall_cycles(P), producer_rtt_cycles(P);
        // Per-worker latency sample prefix, taken from the consumer that wrote the
        // samples. Zero outside PI_MODE_LATENCY.
        std::vector<uint64_t> sample_count(P, 0);

        for (uint32_t w = 0; w < P; ++w) {
            const uint32_t page = pi_worker_tile_index(&L, w);
            const uint32_t risc = pi_worker_risc_index(&L, w);
            const size_t worker_words = (pi_worker_l1_base(&L, w) - L.l1_tile_base) / sizeof(uint32_t);
            const uint8_t* w_l1_bytes = reinterpret_cast<const uint8_t*>(
                host_l1_dump.data() + static_cast<size_t>(page) * l1_tile_words + worker_words);

            std::memcpy(&producer_res[w], w_l1_bytes + PI_L1_RESULT_OFFSET, sizeof(pi_producer_result_t));

            if (producer_res[w].abi_tag != abi_tag || producer_res[w].worker_index != w) {
                const LogicalCoord& tile = cfg.ordered_tiles[page];
                std::ostringstream why;
                why << "producer result in L1 page " << page << " (ordered tile " << tile.x << ',' << tile.y
                    << ") risc " << risc << " expected abi_tag 0x" << std::hex << abi_tag << " worker_index "
                    << std::dec << w << ", found abi_tag 0x" << std::hex << producer_res[w].abi_tag << std::dec
                    << " worker_index " << producer_res[w].worker_index << ": ";
                if (producer_res[w].abi_tag == 0u) {
                    // A producer whose launch was malformed cannot derive its own
                    // slice address and so publishes no identity at all. That is
                    // identical for every worker, so an all-zero dump means the
                    // launch was rejected, while one zeroed slice among written
                    // ones means that single producer died before publishing.
                    uint32_t unpublished = 0;
                    for (uint32_t other = 0; other < P; ++other) {
                        uint32_t tag = 0;
                        const size_t other_worker_words =
                            (pi_worker_l1_base(&L, other) - L.l1_tile_base) / sizeof(uint32_t);
                        std::memcpy(
                            &tag,
                            reinterpret_cast<const uint8_t*>(
                                host_l1_dump.data() +
                                static_cast<size_t>(pi_worker_tile_index(&L, other)) * l1_tile_words +
                                other_worker_words) +
                                PI_L1_RESULT_OFFSET + offsetof(pi_producer_result_t, abi_tag),
                            sizeof(tag));
                        unpublished += (tag == 0u) ? 1u : 0u;
                    }
                    why << "the slice was never published, and " << unpublished << " of " << P
                        << " slices are unpublished, so "
                        << (unpublished == P
                                ? "the launch itself was rejected before any producer could derive its L1 slice"
                                : "this producer alone failed before its result write");
                } else if (producer_res[w].abi_tag != abi_tag) {
                    why << "the slice carries a foreign or stale mode/creation variant";
                } else {
                    why << "worker " << producer_res[w].worker_index << " belongs to ordered tile "
                        << pi_worker_tile_index(&L, producer_res[w].worker_index)
                        << ", so the host page order does not match the ordered tile list";
                }
                throw std::runtime_error(why.str());
            }
            check_worker_counts(cfg, w, producer_res[w], consumer_slots[w], nonce);
            if (cfg.mode == PI_MODE_LATENCY) {
                const SamplePrefix prefix = latency_sample_prefix(consumer_slots[w], producer_res[w], L, w);
                if (!prefix.ok()) {
                    throw std::runtime_error("latency sample prefix rejected: " + prefix.describe());
                }
                sample_count[w] = prefix.count;
            }

            if (cfg.mode == PI_MODE_BACKPRESSURE) {
                const uint64_t sc = producer_res[w].stall_count;
                const uint64_t rc = producer_res[w].rtt_count;
                if (L.l1_stalls_offset > L.l1_worker_stride ||
                    sc > (L.l1_worker_stride - L.l1_stalls_offset) / sizeof(pi_local_sample_t) ||
                    L.l1_rtts_offset > L.l1_worker_stride ||
                    rc > (L.l1_worker_stride - L.l1_rtts_offset) / sizeof(pi_local_sample_t)) {
                    throw std::runtime_error("backpressure sample prefix exceeds L1 worker block");
                }
                for (uint64_t i = 0; i < sc; ++i) {
                    pi_local_sample_t sample;
                    std::memcpy(&sample, w_l1_bytes + L.l1_stalls_offset + i * sizeof(sample), sizeof(sample));
                    producer_stalls[w].push_back(sample.wall);
                    producer_stall_cycles[w].push_back(sample.cpu);
                }
                for (uint64_t i = 0; i < rc; ++i) {
                    pi_local_sample_t sample;
                    std::memcpy(&sample, w_l1_bytes + L.l1_rtts_offset + i * sizeof(sample), sizeof(sample));
                    producer_rtts[w].push_back(sample.wall);
                    producer_rtt_cycles[w].push_back(sample.cpu);
                }
            }
        }

        acknowledge_collection(*f.tile, L, run_value);

        const ConsumerTerminal terminal =
            finish_consumer(f, x280_run, L, cfg.mode, abi_tag, run_value, P, resources_safe);
        const pi_run_result_t& run_res = terminal.run_res;
        const bool successful = run_res.state == PI_PHASE_DONE && run_res.failure.code == PI_ERROR_NONE &&
                                terminal.harts[0].state.result == PI_ERROR_NONE;

        if (successful) {
            check_successful_run(cfg, run_res, producer_res, consumer_slots);
        }

        check_export_accounting(cfg, L, run_res, consumer_slots);

        // Per-worker throughput only: one producer's release-to-final-credit
        // interval in its own live-timer ticks. Nothing is summed or averaged.
        const uint32_t first_bad_interval = cfg.mode == PI_MODE_THROUGHPUT ? first_invalid_interval(producer_res) : P;
        // X280's aggregate interval brackets the measured phase in every mode.
        const bool aggregate_valid = aggregate_interval_valid(run_res);
        const bool report_successful = successful && first_bad_interval == P && aggregate_valid;

        // A failed run that still collected is the case where a producer can have
        // published progress the consumer never acted on, and the arena is intact
        // here, so the slots are read once and reported both per worker and in the
        // final message. A successful run reads nothing.
        const LimSlotDump lim_dump = successful ? LimSlotDump{} : read_lim_slots(*f.tile, L, P);

        // Records: raw values only; the tabulator pools the per-worker populations.
        const Run run{
            .provenance = build_provenance(),
            .run_value = run_value,
            .cfg = cfg,
            .layout = L,
            .tile_count = T,
            .worker_count = P,
            .mtime_hz = mtime_hz,
            .result = run_res,
            .successful = successful,
            .report_successful = report_successful,
        };
        const auto identity = tensix_to_x280_identity(
            run,
            {
                .expected_worker_failure = axis.expected_worker_failure,
                .release_guards = release_guards.size(),
                .release_guards_overwritten = overwritten_guards.size(),
                .assumption_mask = assumption_mask,
                .bridge_history = bridge_hist,
            });
        const char* member = "tensix_to_x280";
        const hmeas::WarmupSelection warmup{.warmup = cfg.W};
        hout::write_jsonl(
            spec.suite_label, hout::record(run_record(member, identity, "run", {}, std::nullopt, clock, warmup)));
        hout::write_jsonl(spec.suite_label, hout::record(aggregate_record(member, identity, run, warmup)));

        for (uint32_t w = 0; w < P; ++w) {
            const uint32_t tile_index = pi_worker_tile_index(&L, w);
            TensixWorkerSamples samples;
            if (cfg.mode == PI_MODE_LATENCY) {
                samples.latency.assign(
                    export_samples.begin() + w * sample_stride,
                    export_samples.begin() + w * sample_stride + sample_count[w]);
            } else if (cfg.mode == PI_MODE_BACKPRESSURE) {
                samples.credit_stall_cpu_cycles = std::move(producer_stall_cycles[w]);
                samples.stall_triggered_rtt_cpu_cycles = std::move(producer_rtt_cycles[w]);
                samples.credit_stalls = std::move(producer_stalls[w]);
                samples.stall_triggered_rtt = std::move(producer_rtts[w]);
            }
            hout::write_jsonl(
                spec.suite_label + "_workers",
                hout::record(run_record(
                    member,
                    identity,
                    "worker",
                    tensix_to_x280_worker(
                        run,
                        {.index = w, .slot = consumer_slots[w], .lim_dump = lim_dump, .producer = producer_res[w]},
                        {plan.translated[tile_index].x(), plan.translated[tile_index].y()},
                        std::move(samples)),
                    hmeas::Counter::TensixLiveTimer,
                    clock,
                    warmup)));
        }
        EXPECT_TRUE(clock.errors.empty()) << hout::clock_json(clock);
        if (!overwritten_guards.empty()) {
            std::ostringstream why;
            why << "the release reached " << overwritten_guards.size() << " core(s) outside its tiles:";
            for (const auto& g : overwritten_guards) {
                why << " logical " << g.core.str() << " held 0x" << std::hex << g.observed << std::dec;
            }
            throw std::runtime_error(why.str());
        }
        if (!successful) {
            std::cerr << describe_lim_slots(lim_dump, L) << '\n';
            if (axis.expected_worker_failure != PI_ERROR_NONE &&
                producer_res[0].failure.code == static_cast<uint32_t>(axis.expected_worker_failure)) {
                return;
            }
            throw std::runtime_error("packet injection failed: " + describe_failure(run_res.failure));
        }
        if (axis.expected_worker_failure != PI_ERROR_NONE) {
            throw std::runtime_error("injected producer failure did not fail the run");
        }
        if (!aggregate_valid) {
            throw std::runtime_error("invalid X280 aggregate interval: " + describe_aggregate_interval(run_res));
        }
        if (first_bad_interval < producer_res.size()) {
            throw std::runtime_error(
                "invalid throughput endpoint interval for worker " + std::to_string(first_bad_interval) + ": start " +
                std::to_string(producer_res[first_bad_interval].throughput_start) + " end " +
                std::to_string(producer_res[first_bad_interval].throughput_end));
        }
        if (!report_successful) {
            throw std::runtime_error(
                "run not reported successful, yet no failure, aggregate "
                "interval or worker interval (first_bad_interval " +
                std::to_string(first_bad_interval) + " of " + std::to_string(producer_res.size()) + ") explains it");
        }
    } catch (const std::exception& e) {
        if (resources_safe) {
            throw;
        }
        std::cerr << "Packet injection ownership retained after host error: " << e.what() << '\n';
        retain_failed_run(pi_failure_t{PI_ERROR_HOST_IO, PI_PHASE_FAILED, UINT32_MAX, PI_ACTOR_HOST});
    } catch (...) {
        if (resources_safe) {
            throw;
        }
        std::cerr << "Packet injection ownership retained after non-standard host exception\n";
        retain_failed_run(pi_failure_t{PI_ERROR_HOST_IO, PI_PHASE_FAILED, UINT32_MAX, PI_ACTOR_HOST});
    }
}

}  // namespace x280::suites::packet_injection::tensix_to_x280

#endif  // X280_SUITES_PACKET_INJECTION_TENSIX_TO_X280_HPP
