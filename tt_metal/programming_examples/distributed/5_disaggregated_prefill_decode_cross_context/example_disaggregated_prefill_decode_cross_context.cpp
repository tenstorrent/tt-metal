// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Disaggregated prefill/decode: 1 MPI rank in sub-context 0 (prefill, FABRIC_1D) and 64 ranks in
// sub-context 1 (decode stages, FABRIC_2D). Cross-sub-context traffic is only prefill local rank 0
// → decode local rank 0.
//
// Build: `cmake --build build --target example_disaggregated_prefill_decode_cross_context` (ENABLE_DISTRIBUTED).
//
// Launch (65 MPI ranks) requires matching mock cluster + rank-bindings mapping YAMLs; see aisle
// testfiles docs for concrete tt-run lines.

#include <mpi.h>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <tt_stl/span.hpp>

namespace {

constexpr std::size_t kKvSize = 128;
constexpr int kTagInterKv = 20;

constexpr int kPrefillSubcontextSize = 1;
constexpr int kDecodeSubcontextSize = 64;

const tt::tt_metal::distributed::multihost::SubcontextId kPrefillSubctx{0};
const tt::tt_metal::distributed::multihost::SubcontextId kDecodeSubctx{1};

tt::stl::Span<std::byte> as_byte_span(std::vector<float>& v) {
    return {reinterpret_cast<std::byte*>(v.data()), v.size() * sizeof(float)};
}

bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

void run_prefill_one_process_to_decode_rank0() {
    using tt::tt_metal::distributed::multihost::Rank;
    using tt::tt_metal::distributed::multihost::Tag;

    auto& metal = tt::tt_metal::MetalContext::instance();
    metal.set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    metal.initialize_fabric_config();

    const auto& ctx = metal.full_world_distributed_context();
    auto world = tt::tt_metal::distributed::multihost::DistributedContext::get_world_context();

    const int local_rank = static_cast<int>(*ctx.rank());
    if (local_rank != 0) {
        std::cerr << "Prefill sub-context must have a single rank (local rank 0)\n";
        std::abort();
    }

    std::vector<float> kv_cache(kKvSize, 0.0f);
    for (std::size_t i = 0; i < kKvSize; i++) {
        kv_cache[i] = static_cast<float>(i) * 0.01f;
    }

    const auto decode_rank0_world = world->local_to_world_rank(kDecodeSubctx, Rank{0});
    world->send(as_byte_span(kv_cache), decode_rank0_world, Tag{kTagInterKv});
}

void run_decode_stages_recv_prefill_at_rank0_only() {
    using tt::tt_metal::distributed::multihost::Rank;
    using tt::tt_metal::distributed::multihost::Tag;

    auto& metal = tt::tt_metal::MetalContext::instance();
    metal.set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    metal.initialize_fabric_config();

    const auto& ctx = metal.full_world_distributed_context();
    auto world = tt::tt_metal::distributed::multihost::DistributedContext::get_world_context();
    const int local_rank = static_cast<int>(*ctx.rank());

    std::vector<float> kv_cache(kKvSize, 0.0f);

    if (local_rank == 0) {
        const auto prefill_rank0_world = world->local_to_world_rank(kPrefillSubctx, Rank{0});
        world->recv(as_byte_span(kv_cache), prefill_rank0_world, Tag{kTagInterKv});

        for (std::size_t i = 0; i < kKvSize; i++) {
            float expected = static_cast<float>(i) * 0.01f;
            if (!near(kv_cache[i], expected)) {
                std::cerr << "Decode rank 0: KV mismatch at index " << i << "\n";
                std::abort();
            }
        }
    }
}

}  // namespace

using tt::tt_metal::distributed::multihost::DistributedContext;

int main(int argc, char** argv) {
    DistributedContext::create(argc, argv);

    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (!rtoptions.get_mock_enabled() && !cluster.is_ubb_galaxy()) {
        log_info(
            tt::LogAlways, "example_disaggregated_prefill_decode_cross_context: requires mock cluster or UBB Galaxy");
        return 0;
    }

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
    const int local_size = static_cast<int>(*distributed_context.size());

    const char* sub_env = std::getenv("TT_RUN_SUBCONTEXT_ID");
    const bool subcontext_mode = (sub_env != nullptr && sub_env[0] != '\0');

    if (!subcontext_mode) {
        log_info(
            tt::LogAlways,
            "example_disaggregated_prefill_decode_cross_context: requires tt-run with "
            "--rank-bindings-mapping (sub-context launch)");
        return 0;
    }

    const char* sub_size_env = std::getenv("TT_RUN_SUBCONTEXT_SIZE");
    if (sub_size_env == nullptr) {
        std::cerr << "TT_RUN_SUBCONTEXT_SIZE not set\n";
        return 1;
    }

    const int sub_id = std::stoi(std::string(sub_env));
    const int sub_size = std::stoi(std::string(sub_size_env));

    if (sub_id == 0) {
        if (sub_size != kPrefillSubcontextSize || local_size != kPrefillSubcontextSize) {
            std::cerr << "Prefill sub-context: expected size " << kPrefillSubcontextSize
                      << ", got sub_size=" << sub_size << " local_size=" << local_size << "\n";
            return 1;
        }
        run_prefill_one_process_to_decode_rank0();
    } else if (sub_id == 1) {
        if (sub_size != kDecodeSubcontextSize || local_size != kDecodeSubcontextSize) {
            std::cerr << "Decode sub-context: expected size " << kDecodeSubcontextSize << ", got sub_size=" << sub_size
                      << " local_size=" << local_size << "\n";
            return 1;
        }
        run_decode_stages_recv_prefill_at_rank0_only();
    } else {
        std::cerr << "Unexpected TT_RUN_SUBCONTEXT_ID " << sub_id << "\n";
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int mpi_world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
    if (mpi_world_rank == 0) {
        log_info(tt::LogAlways, "example_disaggregated_prefill_decode_cross_context: finished OK");
    }
    return 0;
}
