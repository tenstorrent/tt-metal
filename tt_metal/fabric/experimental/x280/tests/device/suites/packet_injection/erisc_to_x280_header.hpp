// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/packet_injection/erisc_to_x280_header.hpp -- ERISC -> X280 packet
// header ingress: idle-ERISC writer selection, IDLE_ETH L1 placement and header
// seeding, producer kernels, runtime args, producer readback and resident-slot
// validation around the stages in common.hpp.

#ifndef X280_SUITES_PACKET_INJECTION_ERISC_TO_X280_HEADER_HPP
#define X280_SUITES_PACKET_INJECTION_ERISC_TO_X280_HEADER_HPP

// The ERISC producer image is throughput-only; erisc_producer.hpp asserts it.
#ifndef PACKET_INJECTION_MODE
#define PACKET_INJECTION_MODE 2
#endif
#ifndef PACKET_INJECTION_INCLUDE_CREATION
#define PACKET_INJECTION_INCLUDE_CREATION 0
#endif

#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "harness/regions/l1_window.hpp"
#include "harness/regions/metal_l1_region.hpp"
#include "tests/device/suites/packet_injection/regions.hpp"
#include "harness/regions/regions.hpp"
#include "kernels/shared/ubench_units.h"
#include "harness/regions/release_selector.hpp"
#include "harness/runner.hpp"
#include "harness/metal_context.hpp"
#include "suites/packet_injection/common.hpp"

#include "kernels/packet_injection/erisc_producer.hpp"
#include "kernels/packet_injection/packet_injection_abi.h"
#include "kernels/packet_injection/packet_injection_slot.hpp"
#include "rt/x280_noc_map.h"

#include <gtest/gtest.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace x280::suites::packet_injection::erisc_to_x280_header {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;
namespace hmeas = x280::harness::measurement;
namespace hout = x280::harness::output;
namespace hreg = x280::harness::regions;

using namespace x280::harness::regions::packet_injection;

/** One registered point: `writers` ERISCs, zero meaning every available one
 * (capped at kMaxEthernetWriters), each publishing in batches of `B`. */
struct SweepPoint {
    uint32_t writers;
    uint32_t B;
    const char* label;
};
using Axis = SweepPoint;

/* Whether X280 released every Ethernet writer by its own endpoint: a NoC
 * broadcast never reaches an Ethernet tile. */
inline bool released_by_endpoints(const pi_run_result_t& r, uint32_t P) {
    return r.released_group_dests == 0u && r.released_endpoints == P;
}

/** The pilot's warm-up and the measured count when their env vars are unset. */
inline constexpr uint64_t kDefaultW = 32u;
inline constexpr uint64_t kDefaultN = 1024u;

/** One selected writer's coordinates in every space the run uses. `virt` is
 * Metal's virtual core, the TRANSLATED coordinate the X280 window takes. */
struct WriterCoords {
    LogicalCoord logical;
    ttm::CoreCoord virt;
    tt::umd::CoreCoord translated{0, 0, tt::CoreType::ETH, tt::CoordSystem::TRANSLATED};
    tt::umd::CoreCoord noc0{0, 0, tt::CoreType::ETH, tt::CoordSystem::NOC0};
};

/**
 * The 96-byte header host setup seeds into every staging record of writer
 * `writer_index`: word k is (`run_nonce` + k) XOR a per-writer constant, so a
 * slot routed from another writer or partially written does not match.
 */
inline std::array<uint8_t, sizeof(fs_hybrid_mesh_packet_header_t)> writer_header(
    uint32_t run_nonce, uint32_t writer_index) {
    std::array<uint32_t, sizeof(fs_hybrid_mesh_packet_header_t) / sizeof(uint32_t)> words{};
    const uint32_t writer_mix = 0x9E3779B9u * (writer_index + 1u);
    for (uint32_t k = 0; k < words.size(); ++k) {
        words[k] = (run_nonce + k) ^ writer_mix;
    }
    std::array<uint8_t, sizeof(fs_hybrid_mesh_packet_header_t)> bytes{};
    std::memcpy(bytes.data(), words.data(), bytes.size());
    return bytes;
}

/**
 * The result of comparing every resident ring slot and every peer-written
 * record's padding after a run. `slots_checked` counts compared slots; the
 * first mismatch, if any, is named by writer, slot and byte offset.
 */
struct ResidentCheck {
    uint64_t slots_checked = 0;
    uint64_t mismatches = 0;
    uint32_t first_writer = UINT32_MAX;
    uint32_t first_slot = UINT32_MAX;
    uint32_t first_offset = UINT32_MAX;
    uint64_t guard_bytes_checked = 0;
    uint64_t guard_mismatches = 0;

    bool ok() const { return mismatches == 0u && guard_mismatches == 0u; }

    std::string describe() const {
        std::ostringstream out;
        out << slots_checked << " resident slots checked, " << mismatches << " mismatched; " << guard_bytes_checked
            << " guard bytes checked, " << guard_mismatches << " nonzero";
        if (first_writer != UINT32_MAX) {
            out << "; first mismatch writer " << first_writer << " slot " << first_slot << " byte " << first_offset;
        }
        return out.str();
    }
};

/**
 * Reads every writer's ring and the produced/ready slot arrays from `tile` and
 * compares them with what the run must have left: the first min(S, W+N) slots
 * of writer w hold writer_header(`run_nonce`, w) in bytes 0..95 and zeros in
 * 96..127 (the zeroed staging tail the 128-byte write carries), and the 60
 * padding bytes of each produced and ready slot stay zero. Throws on a short
 * read.
 */
inline ResidentCheck check_resident_slots(
    x280::X280Tile& tile, const pi_layout_t& L, uint32_t P, uint32_t S, uint64_t W, uint64_t N, uint32_t run_nonce) {
    ResidentCheck check{};
    const auto rings = tile.read_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + L.rings_offset, L.rings_bytes);
    if (rings.size() != L.rings_bytes) {
        throw std::runtime_error(
            "short read of rings: expected " + std::to_string(L.rings_bytes) + " got " + std::to_string(rings.size()));
    }
    const uint64_t resident = std::min<uint64_t>(S, W + N);
    for (uint32_t w = 0; w < P; ++w) {
        const auto header = writer_header(run_nonce, w);
        for (uint64_t i = 0; i < resident; ++i) {
            const uint8_t* slot = rings.data() + w * L.ring_stride + i * PI_SLOT_BYTES;
            check.slots_checked += 1;
            for (uint32_t b = 0; b < PI_SLOT_BYTES; ++b) {
                const uint8_t expected = b < header.size() ? header[b] : 0u;
                if (slot[b] != expected) {
                    if (check.mismatches == 0u) {
                        check.first_writer = w;
                        check.first_slot = static_cast<uint32_t>(i);
                        check.first_offset = b;
                    }
                    check.mismatches += 1;
                    break;
                }
            }
        }
    }

    for (const uint64_t offset : {L.produced_offset, L.ready_offset}) {
        const size_t bytes = static_cast<size_t>(P) * sizeof(pi_counter_slot_t);
        const auto slots = tile.read_available_lim(X280_LIM_PACKET_INJECTION_ARENA_BASE + offset, bytes);
        if (slots.size() != bytes) {
            throw std::runtime_error("short read of peer-written slots at arena offset " + std::to_string(offset));
        }
        for (uint32_t w = 0; w < P; ++w) {
            for (size_t b = offsetof(pi_counter_slot_t, pad); b < sizeof(pi_counter_slot_t); ++b) {
                check.guard_bytes_checked += 1;
                check.guard_mismatches += slots[w * sizeof(pi_counter_slot_t) + b] != 0u ? 1u : 0u;
            }
        }
    }
    return check;
}

/** Throws, naming `name` and `why`, when the Tensix-suite knob `name` is set:
 * this suite does not read it, so a stale environment cannot look like it
 * configured the run. */
inline void reject_unused_env(const char* name, const char* why) {
    if (!hart_::env_str(name).empty()) {
        throw std::runtime_error(std::string(name) + " is set, but " + why);
    }
}

/**
 * Runs one admitted packet-header ingress launch of `selection`'s writers at
 * `cfg`, whose W is already chosen, and validates it; throws on any failure.
 * With `warmup` set it brackets the launch with writer 0's ERISC AICLK evidence
 * and writes the run, aggregate and worker records, naming `warmup` as the
 * warm-up's source; without it the launch is a calibration pilot that records
 * nothing. Returns writer 0's measured-phase interval in its ERISC wall ticks.
 */
inline uint64_t run_once(
    x280::harness::session::Fixture& f,
    const x280::harness::RunSpec& spec,
    const Axis& axis,
    const std::string& fw_img,
    Config cfg,
    const WriterSelection& selection,
    const hmeas::WarmupSelection* warmup) {
    auto& cluster = x280::harness::cluster();
    const auto& hal = ttm::MetalContext::instance().hal();
    auto* device = f.mesh->get_device(0, 0);
    const uint32_t P = selection.requested;
    uint64_t writer0_ticks = 0;
    // ERISC ticks convert at the programmed AICLK, which also sizes the
    // per-phase timeout budget.
    const uint64_t expected_aiclk_hz = hexec::programmed_aiclk_hz(x280::harness::cluster(), *device);

    // IDLE_ETH L1: the unreserved window, whose end the producer image bounds
    // by its own MEM_ERISC_MAX_SIZE.
    const hreg::L1Window window =
        hreg::query_unreserved_window(hal, ttm::HalProgrammableCoreType::IDLE_ETH, hreg::kCacheLineBytes);
    if (window.local.base + window.local.size != ::packet_injection::DeviceEriscTransport::kL1SpanLimit) {
        throw std::runtime_error(
            "idle-ERISC unreserved window ends at " + std::to_string(window.local.base + window.local.size) +
            ", but the producer image bounds spans at " +
            std::to_string(::packet_injection::DeviceEriscTransport::kL1SpanLimit));
    }

    Capacity cap{};
    cap.lim_backed_bytes = f.tile->inspect().lim_backed_bytes;
    cap.l1_span_limit = ::packet_injection::DeviceEriscTransport::kL1SpanLimit;
    cap.l1_window_base = static_cast<uint32_t>(window.local.base);
    cap.producer_timer_hz = expected_aiclk_hz;
    cap.mtime_hz = x280::harness::measurement::kX280MtimeHz;

    const Admission placed = validate_config(cfg, cap, static_cast<uint32_t>(window.local.base));
    if (!placed.ok()) {
        throw std::runtime_error("Configuration rejected: " + placed.describe());
    }
    const pi_layout_t& L = placed.layout.abi;
    if (L.worker_count != P || L.tile_count != P || L.risc_per_tile != 1u) {
        throw std::runtime_error(
            "Ethernet layout is not one producer per tile: P " + std::to_string(L.worker_count) + " T " +
            std::to_string(L.tile_count) + " for " + std::to_string(P) + " writers");
    }
    try {
        (void)hreg::subwindow(window, L.l1_tile_base, L.l1_tile_bytes);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(
            "ERISC tile span [" + std::to_string(L.l1_tile_base) + ", +" + std::to_string(L.l1_tile_bytes) +
            ") is not inside the unreserved window [" + std::to_string(window.local.base) + ", +" +
            std::to_string(window.local.size) + "): " + e.what());
    }

    uint32_t nonce = 0;
    const uint64_t run_value = mint_run_value(nonce);

    // Coordinates: the release selector resolves each writer's TRANSLATED
    // coordinate, which the X280 outbound window takes; an Ethernet tile is
    // always an endpoint. NOC0 is recorded beside it.
    std::vector<hreg::ReleaseParticipant> participants;
    participants.reserve(P);
    for (uint32_t w = 0; w < P; ++w) {
        participants.push_back({tt::CoreType::ETH, ttm::CoreCoord{cfg.ordered_tiles[w].x, cfg.ordered_tiles[w].y}});
    }
    const auto plan = hreg::select_release(*f.mesh, participants, hreg::kX280ReleaseSender, hreg::ReleaseMode::Auto, P);
    const auto& soc = x280::harness::cluster().get_soc_desc(f.tile_id.chip_id);
    std::vector<WriterCoords> coords(P);
    for (uint32_t w = 0; w < P; ++w) {
        coords[w].logical = cfg.ordered_tiles[w];
        coords[w].virt = ttm::CoreCoord{plan.translated[w].x(), plan.translated[w].y()};
        const ttm::CoreCoord& virt = coords[w].virt;
        coords[w].translated = soc.get_coord_at(tt_xy_pair(virt.x, virt.y), tt::CoordSystem::TRANSLATED);
        if (coords[w].translated.core_type != tt::CoreType::ETH) {
            throw std::runtime_error(
                "writer " + std::to_string(w) + " at logical (" + std::to_string(coords[w].logical.x) + "," +
                std::to_string(coords[w].logical.y) + ") resolves to a non-Ethernet core at translated (" +
                std::to_string(virt.x) + "," + std::to_string(virt.y) + ")");
        }
        coords[w].noc0 = soc.translate_coord_to(coords[w].translated, tt::CoordSystem::NOC0);
    }

    const auto l2cpu_translated = f.runtime->virtual_core_from_tile(f.tile_id, ttm::NOC::NOC_0);

    const uint32_t abi_tag = PI_ABI_TAG(cfg.mode, cfg.include_creation);
    const pi_x280_args_t xargs = make_x280_args(cfg, P, P, L, abi_tag, placed.x280_timeout_ticks, nonce, plan.group);
    prime_lim_arena(*f.tile, L, xargs, plan.translated);

    const uint64_t staging0 = pi_worker_l1_base(&L, 0) + PI_L1_STAGING_OFFSET;
    const uint64_t ring0 = X280_LIM_PACKET_INJECTION_ARENA_BASE + L.rings_offset;
    if (!ubench_noc_congruent(staging0, ring0) || (L.ring_stride % UBENCH_NOC_UNIT_BYTES) != 0u) {
        throw std::runtime_error(
            "staging " + std::to_string(staging0) + " and ring " + std::to_string(ring0) + " (stride " +
            std::to_string(L.ring_stride) + ") are not congruent mod " + std::to_string(UBENCH_NOC_UNIT_BYTES));
    }

    // Seed each writer's L1 span: zeros everywhere (the producer does not
    // re-zero its consumed mirror or control line), then its header in every
    // staging record it can use, and prove the image by reading it back.
    std::vector<tt_cxy_pair> cores(P);
    for (uint32_t w = 0; w < P; ++w) {
        cores[w] = tt_cxy_pair(device->id(), coords[w].virt);
        std::vector<uint8_t> image(L.l1_tile_bytes, 0u);
        const auto header = writer_header(nonce, w);
        const uint32_t staging = pi_worker_l1_base(&L, w) - L.l1_tile_base + PI_L1_STAGING_OFFSET;
        for (uint32_t d = 0; d < L.staging_depth; ++d) {
            std::memcpy(image.data() + staging + d * PI_SLOT_BYTES, header.data(), header.size());
        }
        cluster.write_core(image.data(), static_cast<uint32_t>(image.size()), cores[w], L.l1_tile_base);
        std::vector<uint8_t> readback(image.size(), 0xFFu);
        cluster.read_core(readback.data(), static_cast<uint32_t>(readback.size()), cores[w], L.l1_tile_base);
        if (readback != image) {
            throw std::runtime_error("writer " + std::to_string(w) + " L1 seed did not read back");
        }
    }

    ttm::Program program;
    std::vector<ttm::CoreRange> ranges;
    for (uint32_t w = 0; w < P; ++w) {
        const ttm::CoreCoord c{coords[w].logical.x, coords[w].logical.y};
        ranges.push_back(ttm::CoreRange(c, c));
    }
    const auto source = spec.open.kernel_dir / "packet_injection_erisc_to_x280_header.cpp";
    const auto kernel = ttm::CreateKernel(
        program,
        source.string(),
        ttm::CoreRangeSet(ranges),
        ttm::EthernetConfig{
            .eth_mode = ttm::Eth::IDLE,
            .noc = ttm::NOC::NOC_0,
            .processor = ttm::DataMovementProcessor::RISCV_0,
            .defines =
                {{"PACKET_INJECTION_MODE", std::to_string(cfg.mode)},
                 {"PACKET_INJECTION_INCLUDE_CREATION", std::to_string(cfg.include_creation)}},
            .opt_level = ttm::KernelBuildOptLevel::Os,
        });
    for (uint32_t w = 0; w < P; ++w) {
        pi_producer_arg_words_t block{};
        block.args =
            make_producer_args(cfg, P, P, L, abi_tag, nonce, w, l2cpu_translated, placed.producer_timeout_ticks);
        ttm::SetRuntimeArgs(
            program,
            kernel,
            ttm::CoreCoord{coords[w].logical.x, coords[w].logical.y},
            std::vector<uint32_t>(block.words, block.words + PI_PRODUCER_ARG_WORDS));
    }
    ttmd::MeshWorkload workload;
    workload.add_program(ttmd::MeshCoordinateRange(f.mesh->shape()), std::move(program));

    const hexec::X280Launch launch{
        .image_path = fw_img,
        .run_value = run_value,
        .args_addr = X280_LIM_PACKET_INJECTION_ARENA_BASE + L.args_offset,
        .args_bytes = sizeof(pi_x280_args_t),
        .cleanup_timeout = std::chrono::seconds(5),
    };
    std::vector<std::vector<uint8_t>> l1_dumps(P);
    bool resources_safe = false;
    // The ERISC wall clock converts through the AICLK evidence bracketing the
    // run, taken on writer 0's own ERISC wall clock and applied to every
    // writer: all idle ERISCs on the chip are assumed to share one AICLK rate.
    const ttm::CoreCoord reference_core = coords[0].virt;
    hmeas::AiclkEndpoint clock_start;
    hmeas::AiclkEndpoint clock_end;
    if (warmup != nullptr) {
        clock_start = hexec::read_aiclk_start(x280::harness::cluster(), *device, reference_core);
        if (!clock_start.errors.empty()) {
            throw std::runtime_error("pre-run AICLK evidence is unavailable");
        }
    }
    const auto host_start = std::chrono::steady_clock::now();
    try {
        hexec::enqueue_tensix(*f.mesh, workload);
        ConsumerRun x280_run;
        enqueue_consumer(f, launch, x280_run);
        await_quiescence(
            f, L, P, abi_tag, run_value, host_start + std::chrono::milliseconds(cfg.host_timeout_ms), x280_run);

        collect_within_deadline(x280_run, [&]() {
            hexec::finish_tensix(*f.mesh);
            if (warmup != nullptr) {
                clock_end = hexec::read_aiclk_end(x280::harness::cluster(), *device, reference_core);
            }
            for (uint32_t w = 0; w < P; ++w) {
                l1_dumps[w].assign(L.l1_tile_bytes, 0u);
                cluster.read_core(
                    l1_dumps[w].data(), static_cast<uint32_t>(l1_dumps[w].size()), cores[w], L.l1_tile_base);
            }
        });
        const auto clock = warmup != nullptr ? hmeas::assess_aiclk_run(expected_aiclk_hz, clock_start, clock_end)
                                             : hmeas::AiclkRunEvidence{};

        if (x280_run.argument_rejection) {
            reject_arguments(f, x280_run, resources_safe);
        }

        const std::vector<pi_consumer_slot_t> consumer_slots = read_consumer_slots(*f.tile, L, P);
        const ResidentCheck resident = check_resident_slots(*f.tile, L, P, cfg.S, cfg.W, cfg.N, nonce);

        std::vector<pi_producer_result_t> producer_res(P);
        for (uint32_t w = 0; w < P; ++w) {
            std::memcpy(
                &producer_res[w],
                l1_dumps[w].data() + (pi_worker_l1_base(&L, w) - L.l1_tile_base) + PI_L1_RESULT_OFFSET,
                sizeof(pi_producer_result_t));
            if (producer_res[w].abi_tag != abi_tag || producer_res[w].worker_index != w) {
                std::ostringstream why;
                why << "producer result of writer " << w << " at logical (" << coords[w].logical.x << ","
                    << coords[w].logical.y << ") expected abi_tag 0x" << std::hex << abi_tag << " worker_index "
                    << std::dec << w << ", found abi_tag 0x" << std::hex << producer_res[w].abi_tag << std::dec
                    << " worker_index " << producer_res[w].worker_index
                    << (producer_res[w].abi_tag == 0u
                            ? ": never published; an all-zero result under fast dispatch is an invalid "
                              "IDLE_ETH launch, not a measurement"
                            : "");
                throw std::runtime_error(why.str());
            }
            check_worker_counts(cfg, w, producer_res[w], consumer_slots[w], nonce);
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

        const uint32_t first_bad_interval = first_invalid_interval(producer_res);
        const bool aggregate_valid = aggregate_interval_valid(run_res);
        const bool report_successful = successful && first_bad_interval == P && aggregate_valid && resident.ok() &&
                                       released_by_endpoints(run_res, P);
        const LimSlotDump lim_dump = successful ? LimSlotDump{} : read_lim_slots(*f.tile, L, P);

        // A calibration pilot records nothing.
        if (warmup != nullptr) {
            const Run run{
                .provenance = {PI_ENGINE_SHA, PI_METAL_SHA, PI_SOURCE_DIRTY != 0},
                .run_value = run_value,
                .cfg = cfg,
                .layout = L,
                .tile_count = P,
                .worker_count = P,
                .mtime_hz = x280::harness::measurement::kX280MtimeHz,
                .result = run_res,
                .successful = successful,
                .report_successful = report_successful,
            };
            EriscRunState state{
                .writers_requested = axis.writers,
                .inactive_eth_available = static_cast<uint32_t>(selection.available),
                .erisc_timeout_budget_hz = expected_aiclk_hz,
                .resident_slots_checked = resident.slots_checked,
                .resident_slot_mismatches = resident.mismatches,
                .guard_bytes_checked = resident.guard_bytes_checked,
                .guard_mismatches = resident.guard_mismatches,
            };
            for (uint32_t w = 0; w < P; ++w) {
                state.writers.push_back({coords[w].logical.x, coords[w].logical.y});
            }
            const auto identity = erisc_to_x280_header_identity(run, state);
            const char* member = "erisc_to_x280_header";
            hout::write_jsonl(
                spec.suite_label, hout::record(run_record(member, identity, "run", {}, std::nullopt, clock, *warmup)));
            hout::write_jsonl(spec.suite_label, hout::record(aggregate_record(member, identity, run, *warmup)));

            for (uint32_t w = 0; w < P; ++w) {
                const WriterPlacement at{
                    .logical = {coords[w].logical.x, coords[w].logical.y},
                    .virt = {static_cast<uint32_t>(coords[w].virt.x), static_cast<uint32_t>(coords[w].virt.y)},
                    .translated =
                        {static_cast<uint32_t>(coords[w].translated.x), static_cast<uint32_t>(coords[w].translated.y)},
                    .noc0 = {static_cast<uint32_t>(coords[w].noc0.x), static_cast<uint32_t>(coords[w].noc0.y)},
                };
                hout::write_jsonl(
                    spec.suite_label + "_workers",
                    hout::record(run_record(
                        member,
                        identity,
                        "worker",
                        erisc_to_x280_header_worker(
                            run,
                            {.index = w, .slot = consumer_slots[w], .lim_dump = lim_dump, .producer = producer_res[w]},
                            at),
                        hmeas::Counter::EriscWallClock,
                        clock,
                        *warmup)));
            }
            EXPECT_TRUE(clock.errors.empty()) << hout::clock_json(clock);
        }

        if (!successful) {
            std::cerr << describe_lim_slots(lim_dump, L) << '\n';
            throw std::runtime_error("packet-header ingress failed: " + describe_failure(run_res.failure));
        }
        if (!resident.ok()) {
            throw std::runtime_error("resident slot validation failed: " + resident.describe());
        }
        if (!released_by_endpoints(run_res, P)) {
            throw std::runtime_error(
                "X280 executed a release group of " + std::to_string(run_res.released_group_dests) + " and " +
                std::to_string(run_res.released_endpoints) + " endpoints, not " + std::to_string(P) + " endpoints");
        }
        if (!aggregate_valid) {
            throw std::runtime_error("invalid X280 aggregate interval: " + describe_aggregate_interval(run_res));
        }
        if (first_bad_interval < P) {
            throw std::runtime_error(
                "invalid throughput endpoint interval for writer " + std::to_string(first_bad_interval) + ": start " +
                std::to_string(producer_res[first_bad_interval].throughput_start) + " end " +
                std::to_string(producer_res[first_bad_interval].throughput_end));
        }
        if (!report_successful) {
            throw std::runtime_error("run not reported successful, yet no check explains it");
        }
        std::cout << "writers " << P << " B " << cfg.B << " W " << cfg.W << ": " << resident.describe() << '\n';
        writer0_ticks = producer_res[0].throughput_end - producer_res[0].throughput_start;
    } catch (const std::exception& e) {
        if (resources_safe) {
            throw;
        }
        std::cerr << "Packet-header ingress ownership retained after host error: " << e.what() << '\n';
        retain_failed_run(pi_failure_t{PI_ERROR_HOST_IO, PI_PHASE_FAILED, UINT32_MAX, PI_ACTOR_HOST});
    } catch (...) {
        if (resources_safe) {
            throw;
        }
        std::cerr << "Packet-header ingress ownership retained after non-standard host exception\n";
        retain_failed_run(pi_failure_t{PI_ERROR_HOST_IO, PI_PHASE_FAILED, UINT32_MAX, PI_ACTOR_HOST});
    }
    return writer0_ticks;
}

inline void run_axis(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, const Axis& axis) {
    const std::string fw_img = hart_::image_path(spec.open.image_env_var);
    if (fw_img.empty() || !std::filesystem::exists(fw_img)) {
        GTEST_SKIP() << "set " << spec.open.image_env_var << " to a built firmware image";
    }
    if (fw_img.find("throughput") == std::string::npos) {
        throw std::runtime_error(
            "image " + fw_img + " is not the packet-injection throughput image the ERISC member runs");
    }

    reject_unused_env("PACKET_INJECTION_B", "the batch size is this suite's gtest axis");
    reject_unused_env("PACKET_INJECTION_T", "the writer count is this suite's gtest axis");
    reject_unused_env("PACKET_INJECTION_TILES", "writers are the sorted inactive Ethernet cores");
    reject_unused_env("PACKET_INJECTION_FAMILY", "the ERISC member has one producer per Ethernet tile");

    Config cfg{};
    cfg.mode = PI_MODE_THROUGHPUT;
    cfg.include_creation = 0;
    cfg.family = WorkerFamily::Ethernet;
    cfg.B = axis.B;
    cfg.N = kDefaultN;

    std::string v = hart_::env_str("PACKET_INJECTION_S");
    if (!v.empty()) {
        cfg.S = static_cast<uint32_t>(hart_::parse_u64_strict(v, "PACKET_INJECTION_S", UINT32_MAX));
    }
    v = hart_::env_str("PACKET_INJECTION_C");
    if (!v.empty()) {
        cfg.C = static_cast<uint32_t>(hart_::parse_u64_strict(v, "PACKET_INJECTION_C", UINT32_MAX));
    }
    v = hart_::env_str("PACKET_INJECTION_K");
    if (!v.empty()) {
        cfg.K = static_cast<uint32_t>(hart_::parse_u64_strict(v, "PACKET_INJECTION_K", UINT32_MAX));
    }
    std::optional<uint64_t> explicit_w;
    v = hart_::env_str("PACKET_INJECTION_W");
    if (!v.empty()) {
        explicit_w = hart_::parse_u64_strict(v, "PACKET_INJECTION_W");
    }
    v = hart_::env_str("PACKET_INJECTION_N");
    if (!v.empty()) {
        cfg.N = hart_::parse_u64_strict(v, "PACKET_INJECTION_N");
    }
    v = hart_::env_str("PACKET_INJECTION_POLL_CADENCE");
    if (!v.empty()) {
        cfg.poll_cadence =
            static_cast<uint32_t>(hart_::parse_u64_strict(v, "PACKET_INJECTION_POLL_CADENCE", UINT32_MAX));
    }
    v = hart_::env_str("PACKET_INJECTION_PHASE_TIMEOUT_MS");
    if (!v.empty()) {
        cfg.phase_timeout_ms = hart_::parse_u64_strict(v, "PACKET_INJECTION_PHASE_TIMEOUT_MS");
    }
    v = hart_::env_str("PACKET_INJECTION_HOST_TIMEOUT_MS");
    if (!v.empty()) {
        cfg.host_timeout_ms = hart_::parse_u64_strict(v, "PACKET_INJECTION_HOST_TIMEOUT_MS");
    }

    // Writer selection: every inactive Ethernet core, sorted, as a nested prefix.
    const auto& hal = ttm::MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(ttm::HalProgrammableCoreType::IDLE_ETH)) {
        GTEST_SKIP() << "IDLE_ETH programmable cores are not enabled, so no ERISC writer exists";
    }
    auto* device = f.mesh->get_device(0, 0);
    std::vector<LogicalCoord> inactive;
    for (const auto& c : device->get_inactive_ethernet_cores()) {
        inactive.push_back(LogicalCoord{static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)});
    }
    const WriterSelection selection = select_ethernet_writers(inactive, axis.writers);
    std::cout << "inactive IDLE_ETH cores: " << selection.available << '\n';
    if (selection.outcome == WriterSelection::Outcome::Shortfall) {
        GTEST_SKIP() << "writer availability: " << selection.describe();
    }
    if (!selection.ok()) {
        throw std::runtime_error("writer selection rejected: " + selection.describe());
    }
    const uint32_t P = selection.requested;
    cfg.tile_count = P;
    cfg.ordered_tiles = selection.writers;

    // An explicit W is used as given; otherwise a default-W pilot of the same
    // point sizes it from writer 0's measured interval.
    if (explicit_w) {
        cfg.W = *explicit_w;
        const hmeas::WarmupSelection given{.warmup = cfg.W};
        run_once(f, spec, axis, fw_img, cfg, selection, &given);
        return;
    }
    cfg.W = kDefaultW;
    const uint64_t pilot_ticks = run_once(f, spec, axis, fw_img, cfg, selection, nullptr);
    auto request = hmeas::batch_warmup_calibration_request(
        pilot_ticks, cfg.N, hexec::programmed_aiclk_hz(x280::harness::cluster(), *device), kDefaultW, cfg.N);
    request.max_device_ns = cfg.host_timeout_ms * 1000000ull;
    const hmeas::WarmupSelection calibrated = hmeas::calibrate_warmup(request);
    ASSERT_FALSE(calibrated.refusal.has_value())
        << "warm-up calibration refused: " << hmeas::warmup_refusal_name(*calibrated.refusal);
    cfg.W = calibrated.warmup;
    run_once(f, spec, axis, fw_img, cfg, selection, &calibrated);
}

}  // namespace x280::suites::packet_injection::erisc_to_x280_header

#endif  // X280_SUITES_PACKET_INJECTION_ERISC_TO_X280_HEADER_HPP
