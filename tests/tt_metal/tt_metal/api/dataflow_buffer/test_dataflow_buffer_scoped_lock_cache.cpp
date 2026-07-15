// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar scoped_lock CACHE-OP validation (invalidate on acquire / flush on release).
// Ported into the recategorized DFB test layout (was in the monolithic test_dataflow_buffer.cpp); see #45917.
// scoped_lock's L2 cache ops execute on Quasar (the WH/BH path only records the lock tracker event), so these
// tests are Quasar-only. Technique: the DFB ring lives in cacheable L1; the same address + MEM_L1_UNCACHED_BASE
// is the non-cacheable alias (straight to TL1). The kernels build deterministic stale (L2=OLD) / fresh (TL1=NEW)
// state and verify that scoped_lock touches EXACTLY the held entries (invalidate both roles / flush producer-only).

#include "dfb_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/dataflow/dfb_scoped_lock_cache_common.h"

namespace tt::tt_metal {
namespace {

// Sentinels: NEW[s] = DFB_CACHE_NEW_BASE + s, OLD[s] = DFB_CACHE_OLD_BASE + s.
constexpr uint32_t DFB_CACHE_NEW_BASE = 0xAA00;
constexpr uint32_t DFB_CACHE_OLD_BASE = 0xBB00;
constexpr uint32_t DFB_CACHE_RESULT_OFFSET = 0x40000;  // scratch L1 region, well clear of the small ring

struct DfbCacheParams {
    uint32_t num_producers;
    uint32_t num_consumers;
    dfb::AccessPattern cap;   // consumer access pattern (STRIDED or ALL)
    uint32_t entry_size;      // bytes per entry (>= 64, 64-aligned)
    uint32_t num_entries;     // ring slots driven by the kernel
    bool active_is_producer;  // which role takes the lock (does the cache dance)
    DfbCacheTestMode mode;    // FlushOnRelease vs InvalidateOnAcquire
    uint32_t lock_n;          // entries locked
    // Multi-consumer ALL invalidate variant: the producer publishes the shared stale state and signals,
    // then all num_consumers consumers concurrently invalidate the shared held entries (each verifies into
    // its own result block). Compiled via -DDFB_CACHE_MULTI_ALL on both kernels. Use with cap=ALL.
    bool multi_all = false;
    // Producer<->consumer handshake variant: reserve_back/push_back, wait_front/pop_front over num_rounds
    // rounds, with cacheable access at the live get_write_ptr()/get_read_ptr(). Run more rounds than the
    // ring capacity so slots are reused, the consumer's invalidate must discard the stale prior-round
    // cached line so each round reads fresh.
    bool handshake = false;
    uint32_t num_rounds = 0;
    // DM producer writes via the uncached alias (write-around) so its store lands in TL1 WITHOUT updating
    // the consumer's cache, mimicking a non-snooping (e.g. Tensix) producer. This makes the DM consumer's
    // acquire-invalidate load-bearing at a wrapped slot. Compiled via -DDFB_CACHE_NONSNOOP_PRODUCER on
    // the producer kernel.
    bool nonsnoop_producer = false;
};

// stride_in_entries = max(P,C) for STRIDED, 1 for ALL — the same value the host serializes into the DFB.
uint32_t dfb_cache_stride_in_entries(const DfbCacheParams& p) {
    if (p.cap == dfb::AccessPattern::ALL) {
        return 1;
    }
    return (p.num_producers > p.num_consumers) ? p.num_producers : p.num_consumers;
}

// Expected per-slot read-back. A slot reads back NEW iff it is a held entry AND the op that touches it
// happened: flush-on-release happens only for the producer; invalidate-on-acquire happens for both roles.
std::vector<uint32_t> dfb_cache_expected(const DfbCacheParams& p) {
    const uint32_t stride = dfb_cache_stride_in_entries(p);
    std::vector<bool> held(p.num_entries, false);
    for (uint32_t k = 0; k < p.lock_n; ++k) {
        held[(k * stride) % p.num_entries] = true;  // wraps at the ring limit, like scoped_lock
    }
    const bool op_makes_fresh = (p.mode == DfbCacheTestMode::InvalidateOnAcquire) ||
                                (p.mode == DfbCacheTestMode::FlushOnRelease && p.active_is_producer);
    std::vector<uint32_t> exp(p.num_entries);
    for (uint32_t s = 0; s < p.num_entries; ++s) {
        exp[s] = (held[s] && op_makes_fresh) ? (DFB_CACHE_NEW_BASE + s) : (DFB_CACHE_OLD_BASE + s);
    }
    return exp;
}

// Builds the DFB, runs the cache-op kernel under slow dispatch, and returns the per-slot read-back.
std::vector<uint32_t> run_dfb_scoped_lock_cache_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const DfbCacheParams& p) {
    IDevice* device = mesh_device->get_devices()[0];
    const uint32_t ring_base = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t result_addr = ring_base + DFB_CACHE_RESULT_OFFSET;
    const CoreCoord core{0, 0};
    const CoreRangeSet crs(CoreRange(core, core));
    const experimental::NodeCoord node{core.x, core.y};

    const experimental::DFBSpecName DFB_NAME{"cache_dfb"};
    const experimental::KernelSpecName PRODUCER{"producer"};
    const experimental::KernelSpecName CONSUMER{"consumer"};
    const experimental::SemaphoreSpecName SEM_RELEASE{"cache_release"};  // multi-consumer-ALL publisher->waiters

    experimental::DataflowBufferSpec dfb_spec{
        .unique_id = DFB_NAME,
        .entry_size = p.entry_size,
        .num_entries = p.num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    const auto consumer_pattern = (p.cap == dfb::AccessPattern::ALL) ? experimental::DFBAccessPattern::ALL
                                                                     : experimental::DFBAccessPattern::STRIDED;

    const std::vector<std::string> rta_names = {
        "is_active", "test_mode", "lock_n", "num_entries", "ring_base", "result_addr", "new_val", "num_rounds"};

    experimental::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_scoped_lock_cache_producer.cpp",
        .num_threads = static_cast<uint8_t>(p.num_producers),
        .dfb_bindings = {experimental::ProducerOf(DFB_NAME, "out")},
        .runtime_arg_schema = {.runtime_arg_names = rta_names},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_scoped_lock_cache_consumer.cpp",
        .num_threads = static_cast<uint8_t>(p.num_consumers),
        .dfb_bindings = {{
            .dfb_spec_name = DFB_NAME,
            .accessor_name = "in",
            .endpoint_type = experimental::DFBEndpointType::CONSUMER,
            .access_pattern = consumer_pattern,
        }},
        .runtime_arg_schema = {.runtime_arg_names = rta_names},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    std::vector<experimental::SemaphoreSpec> semaphores;
    if (p.multi_all) {
        // Compile both kernels' multi-consumer-ALL path; publisher releases the waiters via a Program-scope
        // semaphore (single core: local up(1) is seen by every waiter's wait(1)).
        producer_spec.compiler_options.defines = {{"DFB_CACHE_MULTI_ALL", "1"}};
        consumer_spec.compiler_options.defines = {{"DFB_CACHE_MULTI_ALL", "1"}};
        semaphores.push_back({.unique_id = SEM_RELEASE, .target_nodes = core});
        producer_spec.semaphore_bindings = {{.semaphore_spec_name = SEM_RELEASE, .accessor_name = "release"}};
        consumer_spec.semaphore_bindings = {{.semaphore_spec_name = SEM_RELEASE, .accessor_name = "release"}};
    } else if (p.handshake) {
        producer_spec.source =
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_scoped_lock_cache_handshake_producer.cpp";
        consumer_spec.source =
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_scoped_lock_cache_handshake_consumer.cpp";
        if (p.nonsnoop_producer) {
            producer_spec.compiler_options.defines = {{"DFB_CACHE_NONSNOOP_PRODUCER", "1"}};
        }
        // Explicit flow control so the kernels can inject (un)cacheable accesses between reserve and push.
        disable_implicit_sync_for(producer_spec, DFB_NAME);
        disable_implicit_sync_for(consumer_spec, DFB_NAME);
    }

    experimental::WorkUnitSpec wu{.name = "main", .kernels = {PRODUCER, CONSUMER}, .target_nodes = crs};
    experimental::ProgramSpec spec{
        .name = "dfb_scoped_lock_cache",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {dfb_spec},
        .semaphores = semaphores,  // empty unless multi-consumer-ALL
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    auto make_node_args = [&](bool active) {
        return experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"is_active", active ? 1u : 0u},
             {"test_mode", static_cast<uint32_t>(p.mode)},
             {"lock_n", p.lock_n},
             {"num_entries", p.num_entries},
             {"ring_base", ring_base},
             {"result_addr", result_addr},
             {"new_val", DFB_CACHE_NEW_BASE},
             {"num_rounds", p.num_rounds}});
    };
    experimental::ProgramRunArgs run_args;
    experimental::ProgramRunArgs::KernelRunArgs producer_params{};
    producer_params.kernel = PRODUCER;
    producer_params.runtime_arg_values = make_node_args(p.active_is_producer);
    experimental::ProgramRunArgs::KernelRunArgs consumer_params{};
    consumer_params.kernel = CONSUMER;
    consumer_params.runtime_arg_values = make_node_args(!p.active_is_producer);
    run_args.kernel_run_args = {producer_params, consumer_params};
    experimental::SetProgramRunArgs(program, run_args);

    // Pre-fill the ring's TL1 with OLD sentinels (first word of each slot); isolation + multi-all rely on
    // this baseline, the handshake variant doesn't (its producer writes the values itself via real
    // dataflow). LaunchProgram's l1_barrier ensures these land before the kernel runs.
    const uint32_t wpe = p.entry_size / sizeof(uint32_t);
    if (!p.handshake) {
        std::vector<uint32_t> prefill(p.num_entries * wpe, 0u);
        for (uint32_t s = 0; s < p.num_entries; ++s) {
            prefill[s * wpe] = DFB_CACHE_OLD_BASE + s;
        }
        detail::WriteToDeviceL1(device, core, ring_base, prefill);
    }

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Kernels write their in-kernel verification read-back (via the non-cacheable alias so the result lands
    // in TL1) to the scratch region; the host reads it directly. Layout: handshake -> num_rounds words (one
    // per round, written by the consumer); multi-consumer-ALL -> num_consumers blocks of num_entries; else 1
    // block of num_entries.
    uint32_t result_words;
    if (p.handshake) {
        result_words = p.num_rounds;
    } else {
        result_words = (p.multi_all ? p.num_consumers : 1u) * p.num_entries;
    }
    std::vector<uint32_t> result;
    detail::ReadFromDeviceL1(device, core, result_addr, result_words * sizeof(uint32_t), result);
    return result;
}

#define DFB_CACHE_SKIP_IF_NOT_QUASAR()                                                               \
    if (devices_.at(0)->arch() != ARCH::QUASAR) {                                                    \
        GTEST_SKIP() << "scoped_lock cache ops are Quasar-only; the WH/BH path is the lock tracker"; \
    }

// ---- Flush-on-release (producer) -------------------------------------------------------
// Held entries are flushed to TL1 (read back NEW); non-held stay cache-resident (read back OLD).

// Baseline: 1P/1C STRIDED (stride==1, contiguous), lock_n=4 -> held = {0..3}.
TEST_F(MeshDeviceFixture, ScopedLockCacheFlushProducerStrided1Sx1S) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        1,
        1,
        dfb::AccessPattern::STRIDED,
        1024,
        8,
        /*producer=*/true,
        /*mode=*/DfbCacheTestMode::FlushOnRelease,
        /*lock_n=*/4};
    EXPECT_EQ(run_dfb_scoped_lock_cache_test(this->devices_.at(0), p), dfb_cache_expected(p));
}

// Same baseline but lock_n=1: only the head entry {0} is held/flushed.
TEST_F(MeshDeviceFixture, ScopedLockCacheFlushProducerStrided1Sx1S_LockOne) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        1,
        1,
        dfb::AccessPattern::STRIDED,
        1024,
        8,
        /*producer=*/true,
        /*mode=*/DfbCacheTestMode::FlushOnRelease,
        /*lock_n=*/1};
    EXPECT_EQ(run_dfb_scoped_lock_cache_test(this->devices_.at(0), p), dfb_cache_expected(p));
}

// 2 producers (stride==2): only this producer's held {0,2,4,6} flush; interleaved neighbours {1,3,5,7} untouched.
TEST_F(MeshDeviceFixture, ScopedLockCacheFlushProducerStrided2Sx1S_SkipsNeighbours) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        2,
        1,
        dfb::AccessPattern::STRIDED,
        1024,
        8,
        /*producer=*/true,
        /*mode=*/DfbCacheTestMode::FlushOnRelease,
        /*lock_n=*/4};
    EXPECT_EQ(run_dfb_scoped_lock_cache_test(this->devices_.at(0), p), dfb_cache_expected(p));
}

// ALL pattern (broadcast, stride==1) instead of STRIDED; held = {0..3}.
TEST_F(MeshDeviceFixture, ScopedLockCacheFlushProducerAll) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        1,
        2,
        dfb::AccessPattern::ALL,
        1024,
        8,
        /*producer=*/true,
        /*mode=*/DfbCacheTestMode::FlushOnRelease,
        /*lock_n=*/4};
    EXPECT_EQ(run_dfb_scoped_lock_cache_test(this->devices_.at(0), p), dfb_cache_expected(p));
}

// Wrap: held window {0,2,0} crosses the ring end -> exercises the wrap-to-base branch (idempotent); {0,2}=NEW.
TEST_F(MeshDeviceFixture, ScopedLockCacheFlushProducerWrap) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        2,
        1,
        dfb::AccessPattern::STRIDED,
        1024,
        4,
        /*producer=*/true,
        /*mode=*/DfbCacheTestMode::FlushOnRelease,
        /*lock_n=*/3};
    EXPECT_EQ(run_dfb_scoped_lock_cache_test(this->devices_.at(0), p), dfb_cache_expected(p));
}

// Consumer release must NOT flush -> every slot reads back OLD.
TEST_F(MeshDeviceFixture, ScopedLockCacheFlushConsumerDoesNotFlush) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        1,
        1,
        dfb::AccessPattern::STRIDED,
        1024,
        8,
        /*producer=*/false,
        /*mode=*/DfbCacheTestMode::FlushOnRelease,
        /*lock_n=*/4};
    EXPECT_EQ(run_dfb_scoped_lock_cache_test(this->devices_.at(0), p), dfb_cache_expected(p));
}

// ---- Invalidate-on-acquire (both roles) ------------------------------------------------
// Held entries' stale L2 lines are discarded (re-read fetches NEW from TL1); non-held keep the stale OLD.

// Baseline: 1P/1C STRIDED producer (stride==1), lock_n=4 -> held = {0..3}.
TEST_F(MeshDeviceFixture, ScopedLockCacheInvalidateProducerStrided1Sx1S) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        1,
        1,
        dfb::AccessPattern::STRIDED,
        1024,
        8,
        /*producer=*/true,
        /*mode=*/DfbCacheTestMode::InvalidateOnAcquire,
        /*lock_n=*/4};
    EXPECT_EQ(run_dfb_scoped_lock_cache_test(this->devices_.at(0), p), dfb_cache_expected(p));
}

// 2 producers (stride==2): only held {0,2,4,6} invalidated; interleaved neighbours {1,3,5,7} keep stale OLD.
TEST_F(MeshDeviceFixture, ScopedLockCacheInvalidateProducerStrided2Sx1S_SkipsNeighbours) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        2,
        1,
        dfb::AccessPattern::STRIDED,
        1024,
        8,
        /*producer=*/true,
        /*mode=*/DfbCacheTestMode::InvalidateOnAcquire,
        /*lock_n=*/4};
    EXPECT_EQ(run_dfb_scoped_lock_cache_test(this->devices_.at(0), p), dfb_cache_expected(p));
}

// Consumer also invalidates on acquire (both roles) -> held {0,1,2,3}=NEW, rest OLD.
TEST_F(MeshDeviceFixture, ScopedLockCacheInvalidateConsumer) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        1,
        1,
        dfb::AccessPattern::STRIDED,
        1024,
        8,
        /*producer=*/false,
        /*mode=*/DfbCacheTestMode::InvalidateOnAcquire,
        /*lock_n=*/4};
    EXPECT_EQ(run_dfb_scoped_lock_cache_test(this->devices_.at(0), p), dfb_cache_expected(p));
}

// 1 producer + 4 ALL consumers sharing entries: producer seeds shared-L2=OLD/TL1=NEW + signals, then all 4
// concurrently invalidate the SHARED held {0,1} on acquire and each must read held -> NEW (ALL redundant-invalidate
// path).
TEST_F(MeshDeviceFixture, ScopedLockCacheInvalidateMultiConsumerAll) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        /*num_producers=*/1,
        /*num_consumers=*/4,
        dfb::AccessPattern::ALL,
        /*entry_size=*/1024,
        /*num_entries=*/4,
        /*active_is_producer=*/false,
        /*mode=*/DfbCacheTestMode::InvalidateOnAcquire,
        /*lock_n=*/2,
        /*multi_all=*/true};
    const auto result = run_dfb_scoped_lock_cache_test(this->devices_.at(0), p);
    const auto expected = dfb_cache_expected(p);  // held {0,1} -> NEW, {2,3} -> OLD
    ASSERT_EQ(result.size(), static_cast<size_t>(p.num_consumers) * p.num_entries);
    for (uint32_t c = 0; c < p.num_consumers; ++c) {
        const std::vector<uint32_t> block(result.begin() + c * p.num_entries, result.begin() + (c + 1) * p.num_entries);
        EXPECT_EQ(block, expected) << "consumer thread " << c << " did not see a coherent (invalidated) view";
    }
}

// End-to-end over reserve/push <-> wait/pop: 4-entry ring x 12 rounds so slots wrap; the consumer's acquire-invalidate
// discards its stale prior-round line.
TEST_F(MeshDeviceFixture, ScopedLockCacheHandshakeDmToDmWrap) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        /*num_producers=*/1,
        /*num_consumers=*/1,
        dfb::AccessPattern::STRIDED,
        /*entry_size=*/1024,
        /*num_entries=*/4,             // small ring: slots wrap/reuse across rounds and stay cache-resident
        /*active_is_producer=*/false,  // unused in handshake mode
        /*mode=*/DfbCacheTestMode::InvalidateOnAcquire,  // unused in handshake mode
        /*lock_n=*/1};
    p.handshake = true;
    p.num_rounds = 12;  // > capacity (4) so each slot is reused ~3x
    const auto result = run_dfb_scoped_lock_cache_test(this->devices_.at(0), p);
    ASSERT_EQ(result.size(), p.num_rounds);
    std::vector<uint32_t> expected(p.num_rounds);
    for (uint32_t r = 0; r < p.num_rounds; ++r) {
        expected[r] = DFB_CACHE_NEW_BASE + r;  // the consumer must read each round's produced value
    }
    EXPECT_EQ(result, expected);
}

// Same handshake, but the producer writes WRITE-AROUND (uncached alias) so the store lands in TL1 without updating
// the consumer's cache (mimics a non-snooping Tensix producer) -> the consumer's acquire-invalidate is LOAD-BEARING.
TEST_F(MeshDeviceFixture, ScopedLockCacheHandshakeNonSnoopingProducerWrap) {
    DFB_CACHE_SKIP_IF_NOT_QUASAR();
    DfbCacheParams p{
        /*num_producers=*/1,
        /*num_consumers=*/1,
        dfb::AccessPattern::STRIDED,
        /*entry_size=*/1024,
        /*num_entries=*/4,
        /*active_is_producer=*/false,
        /*mode=*/DfbCacheTestMode::InvalidateOnAcquire,
        /*lock_n=*/1};
    p.handshake = true;
    p.num_rounds = 12;
    p.nonsnoop_producer = true;
    const auto result = run_dfb_scoped_lock_cache_test(this->devices_.at(0), p);
    ASSERT_EQ(result.size(), p.num_rounds);
    std::vector<uint32_t> expected(p.num_rounds);
    for (uint32_t r = 0; r < p.num_rounds; ++r) {
        expected[r] = DFB_CACHE_NEW_BASE + r;
    }
    EXPECT_EQ(result, expected);
}

#undef DFB_CACHE_SKIP_IF_NOT_QUASAR

}  // namespace
}  // namespace tt::tt_metal
