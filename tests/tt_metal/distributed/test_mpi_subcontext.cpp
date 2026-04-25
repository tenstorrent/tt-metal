// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MPI sub-context bring-up: dual fabric modes + intra-context DistributedContext + inter-context MPI_COMM_WORLD.
// Dispatch mirrors models/demos/deepseek_v3_b1/docs/example_dual_rankbindings_one_psd.md (PrefillDecodeDisaggregated).
//
// Launch (from repo root); join lines with shell line-continuation as needed:
//   tt-run --mock-cluster-rank-binding
//     tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/mock_galaxy_quad_2x4_four_rank_cluster_desc_mapping.yaml
//     --rank-bindings-mapping
//     tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_rank_bindings_mapping.yaml
//     --mpi-args "--allow-run-as-root --oversubscribe"
//     ./build/test/tt_metal/distributed/distributed_unit_tests
//     --gtest_filter="MpiSubContext.*"

#include <gtest/gtest.h>
#include <mpi.h>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include <cstdlib>
#include <string>
#include <vector>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::distributed {
namespace {

struct SubcontextLocalBinding {
    multihost::SubcontextId subcontext_id;
    multihost::Rank local_rank;
};

constexpr std::size_t kQuad2x4KvSize = 128;
constexpr int kTagPrefillIntra = 10;
constexpr int kTagInterKv = 20;
constexpr int kTagDecodeIntra = 30;

tt::stl::Span<std::byte> as_byte_span(std::vector<float>& v) {
    return {reinterpret_cast<std::byte*>(v.data()), v.size() * sizeof(float)};
}

const multihost::SubcontextId kPrefillSubctx{0};
const multihost::SubcontextId kDecodeSubctx{1};

void run_prefill_quad2x4_mock_galaxy() {
    using multihost::Rank;
    using multihost::Tag;

    MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    MetalContext::instance().initialize_fabric_config();

    // TODO: full_world_distributed_context() is misleading — returns the sub-context,
    // not the full world. Rename or remove.
    const auto& ctx = MetalContext::instance().full_world_distributed_context();
    auto world = multihost::DistributedContext::get_world_context();
    const int local_rank = static_cast<int>(*ctx.rank());

    std::vector<float> kv_cache(kQuad2x4KvSize, 0.0f);

    if (local_rank == 0) {
        // Prefill rank 0: generate KV data and send to prefill rank 1 (intra-context).
        for (std::size_t i = 0; i < kQuad2x4KvSize; i++) {
            kv_cache[i] = static_cast<float>(i) * 0.01f;
        }
        ctx.send(as_byte_span(kv_cache), Rank{1}, Tag{kTagPrefillIntra});
    } else {
        // Prefill rank 1: receive KV data from prefill rank 0 (intra-context).
        ctx.recv(as_byte_span(kv_cache), Rank{0}, Tag{kTagPrefillIntra});

        // Verify the intra-context transfer produced the expected data.
        for (std::size_t i = 0; i < kQuad2x4KvSize; i++) {
            float expected = static_cast<float>(i) * 0.01f;
            EXPECT_NEAR(kv_cache[i], expected, 1e-5f) << "Prefill intra-ctx mismatch at index " << i;
        }

        // Translate decode sub-context local rank 0 to a world rank via the API.
        const auto decode_rank0_world = world->local_to_world_rank(kDecodeSubctx, Rank{0});

        // Inter-context send to decode sub-context rank 0 via the world context.
        world->send(as_byte_span(kv_cache), decode_rank0_world, Tag{kTagInterKv});
    }
}

void run_decode_quad2x4_mock_galaxy() {
    using multihost::Rank;
    using multihost::Tag;

    MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    MetalContext::instance().initialize_fabric_config();

    // TODO: full_world_distributed_context() is misleading — returns the sub-context,
    // not the full world. Rename or remove.
    const auto& ctx = MetalContext::instance().full_world_distributed_context();
    auto world = multihost::DistributedContext::get_world_context();
    const int local_rank = static_cast<int>(*ctx.rank());

    std::vector<float> kv_cache(kQuad2x4KvSize, 0.0f);

    if (local_rank == 0) {
        // Translate prefill sub-context local rank 1 to a world rank via the API.
        const auto prefill_rank1_world = world->local_to_world_rank(kPrefillSubctx, Rank{1});

        // Inter-context recv from prefill sub-context rank 1 via the world context.
        world->recv(as_byte_span(kv_cache), prefill_rank1_world, Tag{kTagInterKv});

        // Verify the inter-context transfer produced the expected data.
        for (std::size_t i = 0; i < kQuad2x4KvSize; i++) {
            float expected = static_cast<float>(i) * 0.01f;
            EXPECT_NEAR(kv_cache[i], expected, 1e-5f) << "Decode inter-ctx recv mismatch at index " << i;
        }

        // Add 1.0 and forward to decode rank 1 (intra-context).
        for (auto& v : kv_cache) {
            v += 1.0f;
        }
        ctx.send(as_byte_span(kv_cache), Rank{1}, Tag{kTagDecodeIntra});
    } else {
        // Decode rank 1: receive from decode rank 0 (intra-context).
        ctx.recv(as_byte_span(kv_cache), Rank{0}, Tag{kTagDecodeIntra});

        // Verify the full pipeline: original value + 1.0f.
        for (std::size_t i = 0; i < kQuad2x4KvSize; i++) {
            float expected = static_cast<float>(i) * 0.01f + 1.0f;
            EXPECT_NEAR(kv_cache[i], expected, 1e-5f) << "Decode intra-ctx mismatch at index " << i;
        }
    }
}

SubcontextLocalBinding world_rank_to_subcontext_local(
    const multihost::DistributedContext& world, multihost::Rank world_rank) {
    const int wr = *world_rank;
    int prefix = 0;
    for (int i = 0; i < world.subcontext_count(); ++i) {
        const int sz = *world.subcontext_size(multihost::SubcontextId{i});
        if (wr >= prefix && wr < prefix + sz) {
            return SubcontextLocalBinding{multihost::SubcontextId{i}, multihost::Rank{wr - prefix}};
        }
        prefix += sz;
    }
    ADD_FAILURE() << "world rank " << wr << " not in layout";
    return SubcontextLocalBinding{multihost::SubcontextId{0}, multihost::Rank{0}};
}

}  // namespace

// Verifies that in a split launch (4 MPI ranks, 2 sub-contexts of 2 ranks each):
//  - get_current_world() returns the split subcommunicator with size 2 and rank in {0,1}
//  - get_world_context() returns the full MPI_COMM_WORLD with size 4 and rank in {0,1,2,3}
//  - subcontext_id() is set and matches the TT_RUN_SUBCONTEXT_ID env
//  - subcontext_count() == 2
//  - subcontext_sizes() == {2, 2}, and subcontext_size(i) matches for each i
//  - local_to_world_rank round-trips correctly against MPI_Comm_rank(MPI_COMM_WORLD)
//  - get_distributed_context_ptr() agrees with get_current_world()
//  - calling get_world_context() twice returns the same handle
TEST(MpiSubContext, CurrentWorldIsSplitSubcommunicator) {
    const auto& rtoptions = MetalContext::instance().rtoptions();
    const auto& cluster = MetalContext::instance().get_cluster();
    if (!rtoptions.get_mock_enabled() && !cluster.is_ubb_galaxy()) {
        log_info(tt::LogTest, "This test runs on mock cluster or UBB Galaxy");
        GTEST_SKIP();
    }

    const char* sub_env = std::getenv("TT_RUN_SUBCONTEXT_ID");
    if (sub_env == nullptr || sub_env[0] == '\0') {
        log_info(
            tt::LogTest,
            "Requires TT_RUN_SUBCONTEXT_ID (launch with tt-run --rank-bindings-mapping "
            "mock_galaxy_single_host_subcontext_rank_bindings_mapping.yaml)");
        GTEST_SKIP();
    }

    auto& metal = MetalContext::instance();

    // --- get_current_world() is the split subcommunicator ---
    // TODO: full_world_distributed_context() is a misleading name — it actually returns
    // the *sub-context* communicator (post MPI_Comm_split), NOT the full MPI_COMM_WORLD.
    // Rename or remove in favor of get_current_world() / get_distributed_context_ptr().
    const auto& current = metal.full_world_distributed_context();

    // Each sub-context has 2 ranks, so local rank must be 0 or 1.
    EXPECT_GE(*current.rank(), 0);
    EXPECT_LE(*current.rank(), 1);
    // Sub-context size is 2.
    EXPECT_EQ(*current.size(), 2);

    // --- get_world_context() is the unsplit MPI_COMM_WORLD ---
    auto job_world = multihost::DistributedContext::get_world_context();
    ASSERT_NE(job_world, nullptr);

    // World size must be 4 (2 sub-contexts * 2 ranks each).
    EXPECT_EQ(*job_world->size(), 4);
    // World rank must be in {0,1,2,3}.
    EXPECT_GE(*job_world->rank(), 0);
    EXPECT_LE(*job_world->rank(), 3);

    // Cross-check against raw MPI.
    int mpi_rank = 0;
    int mpi_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    EXPECT_EQ(*job_world->size(), mpi_size);
    EXPECT_EQ(*job_world->rank(), mpi_rank);

    // The split subcommunicator must be strictly smaller than the world.
    EXPECT_LT(*current.size(), *job_world->size());

    // --- get_distributed_context_ptr() agrees with full_world_distributed_context() ---
    auto ctx_ptr = metal.get_distributed_context_ptr();
    ASSERT_NE(ctx_ptr, nullptr);
    EXPECT_EQ(*ctx_ptr->size(), *current.size());
    EXPECT_EQ(*ctx_ptr->rank(), *current.rank());

    // --- subcontext_id(): must be present and either 0 or 1 ---
    ASSERT_TRUE(job_world->subcontext_id().has_value());
    const auto this_sub = *job_world->subcontext_id();
    EXPECT_GE(*this_sub, 0);
    EXPECT_LE(*this_sub, 1);

    // --- subcontext_count(): exactly 2 sub-contexts ---
    EXPECT_EQ(job_world->subcontext_count(), 2);

    // --- subcontext_sizes(): both sub-contexts have 2 ranks ---
    auto sizes_span = job_world->subcontext_sizes();
    ASSERT_EQ(sizes_span.size(), 2u);
    EXPECT_EQ(sizes_span[0], 2);
    EXPECT_EQ(sizes_span[1], 2);

    // subcontext_size(i) must agree with subcontext_sizes()[i].
    for (int i = 0; i < job_world->subcontext_count(); ++i) {
        EXPECT_EQ(*job_world->subcontext_size(multihost::SubcontextId{i}), sizes_span[i]);
    }

    // This process's sub-context size must equal the current subcommunicator size.
    EXPECT_EQ(sizes_span[*this_sub], *current.size());

    // Sum of all sub-context sizes must equal world size.
    int total_from_sizes = 0;
    for (int i = 0; i < job_world->subcontext_count(); ++i) {
        total_from_sizes += *job_world->subcontext_size(multihost::SubcontextId{i});
    }
    EXPECT_EQ(total_from_sizes, *job_world->size());

    // --- local_to_world_rank: translating this process's local rank must produce its MPI world rank ---
    const int local = static_cast<int>(*current.rank());
    const auto world_rank = job_world->local_to_world_rank(this_sub, multihost::Rank{local});
    EXPECT_EQ(*world_rank, mpi_rank);

    // Round-trip: world rank -> (subcontext_id, local_rank) must return to where we started.
    const auto back = world_rank_to_subcontext_local(*job_world, world_rank);
    EXPECT_EQ(*back.subcontext_id, *this_sub);
    EXPECT_EQ(*back.local_rank, local);

    // --- get_world_context() called again returns a consistent handle ---
    auto static_job = multihost::DistributedContext::get_world_context();
    ASSERT_NE(static_job, nullptr);
    EXPECT_EQ(*static_job->size(), *job_world->size());
    EXPECT_EQ(*static_job->rank(), *job_world->rank());
}

// Verifies the exact expected layout of the dual sub-context mock galaxy mapping:
//  - subcontext_id is set (0 or 1)
//  - subcontext_count == 2, sizes == {2, 2}
//  - local_to_world_rank(0, 0) == 0, local_to_world_rank(0, 1) == 1   (sub-ctx 0 occupies world ranks 0-1)
//  - local_to_world_rank(1, 0) == 2, local_to_world_rank(1, 1) == 3   (sub-ctx 1 occupies world ranks 2-3)
//  - sub-context size is 2, world size is 4
//  - sub-context local rank is in {0,1}
//  - local_to_world_rank for this process's local rank matches MPI_Comm_rank(MPI_COMM_WORLD)
//  - world_rank_to_subcontext_local round-trips correctly
TEST(MpiSubContext, LauncherMetadataAndTranslationInSplitLaunch) {
    const auto& rtoptions = MetalContext::instance().rtoptions();
    const auto& cluster = MetalContext::instance().get_cluster();
    if (!rtoptions.get_mock_enabled() && !cluster.is_ubb_galaxy()) {
        log_info(tt::LogTest, "This test runs on mock cluster or UBB Galaxy");
        GTEST_SKIP();
    }

    const char* sub_env = std::getenv("TT_RUN_SUBCONTEXT_ID");
    const bool subcontext_mode = (sub_env != nullptr && sub_env[0] != '\0');
    if (!subcontext_mode) {
        log_info(
            tt::LogTest,
            "Requires TT_RUN_SUBCONTEXT_ID (launch with tt-run --rank-bindings-mapping "
            "mock_galaxy_single_host_subcontext_rank_bindings_mapping.yaml)");
        GTEST_SKIP();
    }

    auto& metal = MetalContext::instance();
    // TODO: full_world_distributed_context() is misleading — returns the sub-context,
    // not the full world. Rename or remove.
    const auto& sub_ctx = metal.full_world_distributed_context();
    auto job_world = multihost::DistributedContext::get_world_context();
    ASSERT_NE(job_world, nullptr);

    // --- subcontext_id must be present (this is a split launch) ---
    ASSERT_TRUE(job_world->subcontext_id().has_value());
    const auto this_sub = *job_world->subcontext_id();

    // --- Exactly 2 sub-contexts ---
    EXPECT_EQ(job_world->subcontext_count(), 2);

    // --- Both sub-contexts have exactly 2 ranks ---
    auto sizes_span = job_world->subcontext_sizes();
    ASSERT_EQ(sizes_span.size(), 2u);
    EXPECT_EQ(sizes_span[0], 2);
    EXPECT_EQ(sizes_span[1], 2);

    // --- Verify the full local_to_world_rank mapping for both sub-contexts ---
    // Sub-context 0: local ranks {0,1} -> world ranks {0,1}
    EXPECT_EQ(*job_world->local_to_world_rank(multihost::SubcontextId{0}, multihost::Rank{0}), 0);
    EXPECT_EQ(*job_world->local_to_world_rank(multihost::SubcontextId{0}, multihost::Rank{1}), 1);
    // Sub-context 1: local ranks {0,1} -> world ranks {2,3}
    EXPECT_EQ(*job_world->local_to_world_rank(multihost::SubcontextId{1}, multihost::Rank{0}), 2);
    EXPECT_EQ(*job_world->local_to_world_rank(multihost::SubcontextId{1}, multihost::Rank{1}), 3);

    // --- Sub-context size is 2, world size is 4 ---
    EXPECT_EQ(*sub_ctx.size(), 2);
    EXPECT_EQ(*job_world->size(), 4);

    // --- Local rank within each sub-context is 0 or 1 ---
    const int local_rank = static_cast<int>(*sub_ctx.rank());
    EXPECT_GE(local_rank, 0);
    EXPECT_LE(local_rank, 1);

    // --- local_to_world_rank for this process must match MPI_Comm_rank(MPI_COMM_WORLD) ---
    int mpi_world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
    const auto computed_world = job_world->local_to_world_rank(this_sub, multihost::Rank{local_rank});
    EXPECT_EQ(*computed_world, mpi_world_rank);

    // --- Round-trip: world rank -> (subcontext_id, local_rank) ---
    const auto back = world_rank_to_subcontext_local(*job_world, multihost::Rank{mpi_world_rank});
    EXPECT_EQ(*back.subcontext_id, *this_sub);
    EXPECT_EQ(*back.local_rank, local_rank);

    // --- Round-trip from the computed world rank should give the same result ---
    const auto back_from_computed = world_rank_to_subcontext_local(*job_world, computed_world);
    EXPECT_EQ(*back_from_computed.subcontext_id, *back.subcontext_id);
    EXPECT_EQ(*back_from_computed.local_rank, *back.local_rank);
}

TEST(MpiSubContext, SingleGalaxySplitContext) {
    const auto& rtoptions = MetalContext::instance().rtoptions();
    const auto& cluster = MetalContext::instance().get_cluster();
    if (!rtoptions.get_mock_enabled() && !cluster.is_ubb_galaxy()) {
        log_info(tt::LogTest, "This test runs on mock cluster or UBB Galaxy");
        GTEST_SKIP();
    }

    // TODO: full_world_distributed_context() is misleading — returns the sub-context,
    // not the full world. Rename or remove.
    const auto& distributed_context = MetalContext::instance().full_world_distributed_context();
    const int local_size = static_cast<int>(*distributed_context.size());

    const char* sub_env = std::getenv("TT_RUN_SUBCONTEXT_ID");
    const bool subcontext_mode = (sub_env != nullptr && sub_env[0] != '\0');

    if (!subcontext_mode) {
        log_info(
            tt::LogTest,
            "Requires TT_RUN_SUBCONTEXT_ID (launch with tt-run --rank-bindings-mapping "
            "mock_galaxy_single_host_subcontext_rank_bindings_mapping.yaml)");
        GTEST_SKIP();
    }

    const int sub_id = std::stoi(std::string(sub_env));
    // <<<<<<< HEAD
    auto job_world = multihost::DistributedContext::get_world_context();
    ASSERT_EQ(static_cast<int>(*job_world->subcontext_size(multihost::SubcontextId{sub_id})), 2)
        << "Quad 2×4 mapping uses two ranks per sub-context";
    // =======
    //     const char* sub_size_env = std::getenv("TT_RUN_SUBCONTEXT_SIZE");
    //     ASSERT_NE(sub_size_env, nullptr);
    //     const int sub_size = std::stoi(std::string(sub_size_env));
    //     ASSERT_EQ(sub_size, 2) << "Quad 2×4 mapping uses two ranks per sub-context";
    // >>>>>>> 99c91cd142c (Add sub-context API for split MPI jobs)
    ASSERT_EQ(local_size, 2) << "MPI_Comm_split should yield communicator size 2 per sub-context";

    if (sub_id == 0) {
        run_prefill_quad2x4_mock_galaxy();
    } else {
        run_decode_quad2x4_mock_galaxy();
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace tt::tt_metal::distributed
