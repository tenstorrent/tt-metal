// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "device_fixture.hpp"
#include "hostdev/runtime_reload_abi.h"
#include "impl/program/program_impl.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {
namespace {

class L1AddressPool {
public:
    explicit L1AddressPool(IDevice* device) :
        next_(align(device->allocator()->get_base_allocator_addr(HalMemType::L1))),
        limit_(device->l1_size_per_core()) {}

    // Match the 64-byte DRAM alignment used for reload images, weights, and taps.
    static uint32_t align(uint32_t bytes) { return (bytes + 63) & ~63u; }

    uint32_t allocate(uint32_t bytes) {
        const uint32_t size = align(bytes);
        TT_FATAL(next_ <= limit_ && size <= limit_ - next_, "Runtime reload test exhausted L1 scratch space");
        const uint32_t address = next_;
        next_ += size;
        return address;
    }

private:
    uint32_t next_;
    const uint32_t limit_;
};

class RuntimeReloadFirmwareTest : public MeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        const tt::llrt::RunTimeOptions options;
        if (options.get_brisc_firmware_variant() != tt::llrt::BriscFirmwareVariant::Blaze) {
            GTEST_SKIP() << "requires TT_METAL_FW_SRC_BRISC=blaze";
        }
        MeshDeviceSingleCardFixture::SetUp();
    }

    void run_reload(
        uint32_t stages,
        uint32_t rounds,
        uint32_t mode,
        uint32_t abi_version = RELOAD_ABI_VERSION,
        uint32_t num_cores = 1,
        bool fetch_weights = false,
        NOC noc = NOC::RISCV_0_default) {
        IDevice* device = devices_.front()->get_devices().at(0);
        const CoreRangeSet cores({CoreRange({0, 0}, {num_cores - 1, 0})});
        const auto coordinator = device->worker_core_from_logical_core({0, 0});
        const auto multicast_end = device->worker_core_from_logical_core({num_cores - 1, 0});
        // No CBs or user buffers are allocated; use the free allocator region as test scratch.
        L1AddressPool addresses(device);
        constexpr uint32_t tap_bytes = 64;
        constexpr uint32_t config_scratch_bytes = 1024;
        const uint32_t expected_rounds = mode == RELOAD_ABI_MODE_FIRMWARE_PERSISTENT ? 1 : rounds;
        const uint32_t output_addr = addresses.allocate(tap_bytes);
        constexpr uint32_t weights_per_stage = 2;
        const uint32_t weight_count = fetch_weights ? 1 + stages * weights_per_stage : 0;
        const uint32_t table_words =
            RELOAD_ABI_HEADER_WORDS + stages * RELOAD_ABI_ENTRY_WORDS + weight_count * RELOAD_ABI_WEIGHT_WORDS;
        const uint32_t table_bytes = table_words * sizeof(uint32_t);
        const uint32_t table_addr = addresses.allocate(table_bytes);
        const uint32_t arrive_addr = addresses.allocate(sizeof(uint32_t));
        const uint32_t release_addr = addresses.allocate(sizeof(uint32_t));
        const uint32_t stop_addr = addresses.allocate(sizeof(uint32_t));
        const uint32_t config_scratch = addresses.allocate(config_scratch_bytes);
        const uint32_t taps_bytes = stages * expected_rounds * tap_bytes;
        const uint32_t tap_addr = addresses.allocate(taps_bytes);
        const uint32_t weight_bytes = stages * weights_per_stage * tap_bytes;
        const uint32_t weight_dst = fetch_weights ? addresses.allocate(weight_bytes) : 0;
        const uint32_t scratch_end = fetch_weights ? weight_dst + weight_bytes : tap_addr + taps_bytes;
        std::vector<uint32_t> markers(stages);
        for (uint32_t stage = 0; stage < stages; ++stage) {
            markers[stage] = stage % 2 == 0 ? 0x1234 + stage : 42 + stage * 101;
        }
        std::vector<uint32_t> scratch((scratch_end - output_addr) / sizeof(uint32_t), 0);
        std::vector<uint32_t> stop{1};

        std::vector<uint32_t> table(table_words, 0);
        table[RELOAD_ABI_HDR_NUM_STAGES] = stages;
        table[RELOAD_ABI_HDR_MODE] = mode;
        table[RELOAD_ABI_HDR_NUM_ROUNDS] = rounds;
        table[RELOAD_ABI_HDR_TERM_SEM_ADDR] = stop_addr;
        table[RELOAD_ABI_HDR_ABI_VERSION] = abi_version;
        table[RELOAD_ABI_HDR_SYNC_ARRIVE_ADDR] = arrive_addr;
        table[RELOAD_ABI_HDR_SYNC_RELEASE_ADDR] = release_addr;
        table[RELOAD_ABI_HDR_SYNC_COORD_X] = coordinator.x;
        table[RELOAD_ABI_HDR_SYNC_COORD_Y] = coordinator.y;
        table[RELOAD_ABI_HDR_SYNC_NUM_CORES] = num_cores;
        table[RELOAD_ABI_HDR_SYNC_MCAST_START_X] = coordinator.x;
        table[RELOAD_ABI_HDR_SYNC_MCAST_START_Y] = coordinator.y;
        table[RELOAD_ABI_HDR_SYNC_MCAST_END_X] = multicast_end.x;
        table[RELOAD_ABI_HDR_SYNC_MCAST_END_Y] = multicast_end.y;
        table[RELOAD_ABI_HDR_SYNC_MCAST_DESTS] = num_cores - 1;
        table[RELOAD_ABI_HDR_CONFIG_SCRATCH_ADDR] = config_scratch;

        std::vector<std::vector<uint32_t>> tables(num_cores, table);
        std::vector<L1AddressPool> image_addresses(num_cores, addresses);
        for (uint32_t x = 0; x < num_cores; ++x) {
            const CoreCoord core{x, 0};
            detail::WriteToDeviceL1(device, core, output_addr, scratch);
            detail::WriteToDeviceL1(device, core, stop_addr, stop);
            tables[x][RELOAD_ABI_HDR_SYNC_IS_COORD] = x == 0;
        }

        std::vector<Program> programs(stages);
        for (uint32_t stage = 0; stage < stages; ++stage) {
            auto& program = programs[stage];
            program = CreateProgram();
            const auto kernel = CreateKernel(
                program,
                stage % 2 == 0 ? "tests/tt_metal/tt_metal/test_kernels/misc/write_l1_marker.cpp"
                               : "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
                cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = noc,
                    .noc_mode = NOC_MODE::DM_DYNAMIC_NOC,
                    .compile_args = {output_addr}});
            for (uint32_t x = 0; x < num_cores; ++x) {
                SetRuntimeArgs(
                    program,
                    kernel,
                    CoreCoord{x, 0},
                    stage % 2 == 0 ? std::vector<uint32_t>{output_addr, markers[stage] + x * 65536}
                                   : std::vector<uint32_t>{17 + stage * 101 + x * 65536, 25});
            }
            program.impl().set_reload_table(table_addr, cores);
            experimental::ConfigureProgramWithoutLaunch(device, program);
            for (uint32_t x = 0; x < num_cores; ++x) {
                const CoreCoord core{x, 0};
                const auto physical = device->worker_core_from_logical_core(core);
                const auto config = experimental::CaptureKernelConfig(device, core);
                ASSERT_GT(config.kernel_config_size(), 0u);
                ASSERT_LE(config.launch_kernel_config().size(), config_scratch_bytes);
                ASSERT_EQ(config.kernel_config_size() % sizeof(uint32_t), 0u);
                ASSERT_EQ(config.launch_kernel_config().size() % sizeof(uint32_t), 0u);
                const uint32_t block_bytes = L1AddressPool::align(config.kernel_config_size());
                std::vector<uint32_t> image;
                detail::ReadFromDeviceL1(device, core, config.kernel_config_base(), block_bytes, image);
                const auto block_words = image.size();
                image.resize(block_words + config.launch_kernel_config().size() / sizeof(uint32_t));
                std::memcpy(
                    image.data() + block_words,
                    config.launch_kernel_config().data(),
                    config.launch_kernel_config().size());
                const uint32_t image_addr = image_addresses[x].allocate(image.size() * sizeof(uint32_t));
                detail::WriteToDeviceL1(device, core, image_addr, image);

                const auto offset = RELOAD_ABI_HEADER_WORDS + stage * RELOAD_ABI_ENTRY_WORDS;
                tables[x][offset + RELOAD_ABI_ENTRY_SRC_NOC_X] = physical.x;
                tables[x][offset + RELOAD_ABI_ENTRY_SRC_NOC_Y] = physical.y;
                tables[x][offset + RELOAD_ABI_ENTRY_SRC_ADDR] = image_addr;
                tables[x][offset + RELOAD_ABI_ENTRY_BLOCK_BYTES] = block_bytes;
                tables[x][offset + RELOAD_ABI_ENTRY_LAUNCH_KERNEL_CONFIG_BYTES] = config.launch_kernel_config().size();
                tables[x][offset + RELOAD_ABI_ENTRY_ENABLES] = RELOAD_ABI_ENABLES_CAPTURED;
                tables[x][offset + RELOAD_ABI_ENTRY_TAP_BYTES] = tap_bytes;
                tables[x][offset + RELOAD_ABI_ENTRY_TAP_SRC_L1] = output_addr;
                tables[x][offset + RELOAD_ABI_ENTRY_TAP_DST_NOC_X] = physical.x;
                tables[x][offset + RELOAD_ABI_ENTRY_TAP_DST_NOC_Y] = physical.y;
                const uint32_t stage_tap = tap_addr + stage * expected_rounds * tap_bytes;
                tables[x][offset + RELOAD_ABI_ENTRY_TAP_DST_ADDR] = stage_tap;
                tables[x][offset + RELOAD_ABI_ENTRY_TAP_LIMIT] = stage_tap + expected_rounds * tap_bytes;
                if (fetch_weights) {
                    // Leave slot zero unused to exercise a nonzero weight_first on every stage.
                    const uint32_t first = 1 + stage * weights_per_stage;
                    tables[x][offset + RELOAD_ABI_ENTRY_WEIGHT_FIRST] = first;
                    tables[x][offset + RELOAD_ABI_ENTRY_WEIGHT_COUNT] = weights_per_stage;
                    for (uint32_t region = 0; region < weights_per_stage; ++region) {
                        const uint32_t source = image_addresses[x].allocate(tap_bytes);
                        std::vector<uint32_t> payload(tap_bytes / sizeof(uint32_t));
                        for (uint32_t word = 0; word < payload.size(); ++word) {
                            payload[word] = 0x100000 + x * 65536 + stage * 256 + region * 64 + word;
                        }
                        detail::WriteToDeviceL1(device, core, source, payload);
                        const auto weight_offset = RELOAD_ABI_HEADER_WORDS + stages * RELOAD_ABI_ENTRY_WORDS +
                                                   (first + region) * RELOAD_ABI_WEIGHT_WORDS;
                        tables[x][weight_offset + RELOAD_ABI_WEIGHT_SRC_NOC_X] = physical.x;
                        tables[x][weight_offset + RELOAD_ABI_WEIGHT_SRC_NOC_Y] = physical.y;
                        tables[x][weight_offset + RELOAD_ABI_WEIGHT_SRC_ADDR] = source;
                        tables[x][weight_offset + RELOAD_ABI_WEIGHT_BYTES] = tap_bytes;
                        tables[x][weight_offset + RELOAD_ABI_WEIGHT_DST_L1] =
                            weight_dst + (stage * weights_per_stage + region) * tap_bytes;
                        tables[x][weight_offset + RELOAD_ABI_WEIGHT_VC] = 1;
                        tables[x][weight_offset + RELOAD_ABI_WEIGHT_NOC] = region;
                    }
                }
            }
        }
        for (uint32_t x = 0; x < num_cores; ++x) {
            std::vector<uint32_t> output;
            detail::ReadFromDeviceL1(device, {x, 0}, output_addr, sizeof(uint32_t), output);
            ASSERT_EQ(output.at(0), 0u) << "capture must not execute any stage";
            detail::WriteToDeviceL1(device, {x, 0}, table_addr, tables[x]);
        }
        detail::LaunchProgram(device, programs[0], /*wait_until_cores_done=*/true);

        for (uint32_t x = 0; x < num_cores; ++x) {
            SCOPED_TRACE("core=" + std::to_string(x));
            const CoreCoord core{x, 0};
            std::vector<uint32_t> output;
            std::vector<uint32_t> result;
            detail::ReadFromDeviceL1(device, core, table_addr, table.size() * sizeof(uint32_t), result);
            detail::ReadFromDeviceL1(device, core, output_addr, sizeof(uint32_t), output);
            if (abi_version != RELOAD_ABI_VERSION) {
                EXPECT_EQ(output.at(0), markers[0] + x * 65536);
                EXPECT_EQ(result[RELOAD_ABI_HDR_DBG_PHASE], RELOAD_ABI_PHASE_ABI_MISMATCH);
                EXPECT_EQ(result[RELOAD_ABI_HDR_DBG_RELOAD_COUNT], 0u);
                continue;
            }
            EXPECT_EQ(output.at(0), markers.back() + x * 65536);
            EXPECT_EQ(result[RELOAD_ABI_HDR_DBG_PHASE], RELOAD_ABI_PHASE_DONE);
            EXPECT_EQ(result[RELOAD_ABI_HDR_DBG_RELOAD_COUNT], stages * expected_rounds - 1);
            EXPECT_EQ(result[RELOAD_ABI_HDR_DBG_LAST_STAGE], stages - 1);
            if (fetch_weights) {
                std::vector<uint32_t> weights;
                detail::ReadFromDeviceL1(device, core, weight_dst, weight_bytes, weights);
                for (uint32_t stage = 0; stage < stages; ++stage) {
                    for (uint32_t region = 0; region < weights_per_stage; ++region) {
                        for (uint32_t word = 0; word < tap_bytes / sizeof(uint32_t); ++word) {
                            EXPECT_EQ(
                                weights[(stage * weights_per_stage + region) * tap_bytes / sizeof(uint32_t) + word],
                                0x100000 + x * 65536 + stage * 256 + region * 64 + word)
                                << "stage=" << stage << ", region=" << region << ", word=" << word;
                        }
                    }
                }
            }
            std::vector<uint32_t> taps;
            detail::ReadFromDeviceL1(device, core, tap_addr, stages * expected_rounds * tap_bytes, taps);
            for (uint32_t stage = 0; stage < stages; ++stage) {
                for (uint32_t round = 0; round < expected_rounds; ++round) {
                    EXPECT_EQ(
                        taps[(stage * expected_rounds + round) * tap_bytes / sizeof(uint32_t)],
                        markers[stage] + x * 65536)
                        << "stage=" << stage << ", round=" << round;
                }
            }
        }
    }
};

class RuntimeReloadFirmwareCases : public RuntimeReloadFirmwareTest,
                                   public ::testing::WithParamInterface<std::pair<uint32_t, uint32_t>> {};

TEST_P(RuntimeReloadFirmwareCases, EveryStageEveryRound) {
    const auto [stages, rounds] = GetParam();
    run_reload(stages, rounds, RELOAD_ABI_MODE_BOUNDED);
}

INSTANTIATE_TEST_SUITE_P(
    StageAndRoundCounts,
    RuntimeReloadFirmwareCases,
    ::testing::Values(
        std::pair{2u, 1u},
        std::pair{1u, 100u},
        std::pair{3u, 17u},
        std::pair{8u, 32u},
        std::pair{17u, 100u},
        std::pair{9u, 1000u}),
    [](const ::testing::TestParamInfo<std::pair<uint32_t, uint32_t>>& info) {
        return "Stages" + std::to_string(info.param.first) + "Rounds" + std::to_string(info.param.second);
    });

TEST_F(RuntimeReloadFirmwareTest, PersistentStopsAtRoundBoundary) {
    run_reload(7, 0, RELOAD_ABI_MODE_FIRMWARE_PERSISTENT);
}
TEST_F(RuntimeReloadFirmwareTest, MultiCoreEveryStageEveryRound) {
    run_reload(5, 1000, RELOAD_ABI_MODE_BOUNDED, RELOAD_ABI_VERSION, 2);
}
TEST_F(RuntimeReloadFirmwareTest, MultiCorePersistentStopsAtRoundBoundary) {
    run_reload(7, 0, RELOAD_ABI_MODE_FIRMWARE_PERSISTENT, RELOAD_ABI_VERSION, 2);
}
TEST_F(RuntimeReloadFirmwareTest, MultiCoreNoc1EveryStageEveryRound) {
    run_reload(5, 1000, RELOAD_ABI_MODE_BOUNDED, RELOAD_ABI_VERSION, 2, false, NOC::NOC_1);
}
TEST_F(RuntimeReloadFirmwareTest, FetchesWeightRegionsOnBothNocs) {
    run_reload(5, 3, RELOAD_ABI_MODE_BOUNDED, RELOAD_ABI_VERSION, 2, true);
}
TEST_F(RuntimeReloadFirmwareTest, RejectsMismatchedAbi) {
    run_reload(3, 1, RELOAD_ABI_MODE_BOUNDED, RELOAD_ABI_VERSION + 1);
}

}  // namespace
}  // namespace tt::tt_metal
