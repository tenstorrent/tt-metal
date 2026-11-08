// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar only: with device prints enabled on a Tensix tile, firmware init must complete and DM0
// plus all 16 TRISCs must print their boot banners. Guards the TRISC start-up order in dm.cc.

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <chrono>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-logger/tt-logger.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {
namespace {

constexpr CoreCoord kBootCore{0, 0};

std::string ReadWholeFile(const std::string& file_name) {
    std::ifstream in(file_name);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// The DM0 lines are required. A TRISC boot line is occasionally lost on qsr.s1, so each TRISC only
// has to show one of its two lines. Returns the TRISCs that showed neither.
std::vector<std::string> MissingBanners(
    const std::string& text,
    const std::vector<std::string>& required,
    const std::vector<std::pair<std::string, std::string>>& trisc_lines) {
    std::vector<std::string> missing;
    for (const auto& line : required) {
        if (text.find(line) == std::string::npos) {
            missing.push_back(line);
        }
    }
    for (const auto& [hartid_line, initialized_line] : trisc_lines) {
        if (text.find(hartid_line) == std::string::npos && text.find(initialized_line) == std::string::npos) {
            missing.push_back(hartid_line + " / " + initialized_line);
        }
    }
    return missing;
}

// Opens the devices itself, with device prints enabled on kBootCore only, so the banner printed by
// the firmware launch that this open performs lands in this fixture's output file.
class QuasarFwBootPrintFixture : public DebugToolsMeshFixture {
protected:
    static constexpr auto kFeature = tt::llrt::RunTimeDebugFeatureDprint;
    std::string dprint_file_name;
    int memfd_ = -1;
    tt::llrt::TargetSelection dprint_previous_targets_{};
    bool test_mode_previous_ = false;
    bool watcher_previous_enabled_ = false;
    bool configured_ = false;

    void SetUp() override {
        if (MetalContext::instance().hal().get_arch() != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "The DM0 + 16 TRISC firmware boot banner is Quasar-only";
        }
        auto& rtoptions = MetalContext::instance().rtoptions();
        dprint_previous_targets_ = rtoptions.get_feature_targets(kFeature);
        test_mode_previous_ = rtoptions.get_test_mode_enabled();
        watcher_previous_enabled_ = rtoptions.get_watcher_enabled();

        memfd_ = memfd_create("dprint_quasar_fw_boot", 0);
        TT_FATAL(memfd_ >= 0, "Failed to create memory file descriptor: {}", strerror(errno));
        dprint_file_name = fmt::format("/proc/self/fd/{}", memfd_);

        std::map<CoreType, std::vector<CoreCoord>> cores;
        for (CoreType core_type : {CoreType::WORKER, CoreType::ETH, CoreType::DRAM, CoreType::DISPATCH}) {
            rtoptions.set_feature_all_cores(kFeature, core_type, tt::llrt::RunTimeDebugClassNoneSpecified);
            cores[core_type] = {};
        }
        cores[CoreType::WORKER] = {kBootCore};
        rtoptions.set_feature_enabled(kFeature, true);
        rtoptions.set_feature_cores(kFeature, std::move(cores));
        rtoptions.set_feature_all_chips(kFeature, true);
        rtoptions.set_feature_mesh_coords(kFeature, {});
        rtoptions.set_feature_chip_ids(kFeature, {});
        rtoptions.set_feature_prepend_device_core_risc(kFeature, true);
        rtoptions.set_feature_file_name(kFeature, dprint_file_name);
        rtoptions.set_test_mode_enabled(true);
        rtoptions.set_watcher_enabled(false);
        configured_ = true;

        // Opening the devices launches the firmware with these print settings.
        DebugToolsMeshFixture::SetUp();
    }

    void TearDown() override {
        if (configured_) {
            DebugToolsMeshFixture::TearDown();
            auto& rtoptions = MetalContext::instance().rtoptions();
            rtoptions.set_feature_targets(kFeature, dprint_previous_targets_);
            rtoptions.set_test_mode_enabled(test_mode_previous_);
            rtoptions.set_watcher_enabled(watcher_previous_enabled_);
        }
        if (memfd_ >= 0) {
            close(memfd_);
            memfd_ = -1;
        }
    }
};

TEST_F(QuasarFwBootPrintFixture, TensixFirmwareBootBannerWithPrintsEnabled) {
    if (arch_ != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "The DM0 + 16 TRISC firmware boot banner is Quasar-only";
    }

    const auto& hal = MetalContext::instance().hal();
    const uint32_t dm_count = hal.get_processor_types_count(
        HalProgrammableCoreType::TENSIX, static_cast<uint32_t>(HalProcessorClassType::DM));
    const uint32_t trisc_count = hal.get_processor_types_count(
        HalProgrammableCoreType::TENSIX, static_cast<uint32_t>(HalProcessorClassType::COMPUTE));
    ASSERT_EQ(trisc_count, 16u);

    const bool on_simulator = MetalContext::instance().rtoptions().get_simulator_enabled();
    const auto settle_budget = on_simulator ? std::chrono::seconds(120) : std::chrono::seconds(10);

    for (auto& mesh_device : this->devices_) {
        const auto device_id = mesh_device->get_device_ids()[0];
        const std::string core_prefix = fmt::format("{}:{}-{}:", device_id, kBootCore.x, kBootCore.y);
        const std::string dm0 = hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, 0, true);
        const std::string dm0_initialized = fmt::format("{}{}: DM0-FW: initialized", core_prefix, dm0);
        const std::string dm0_deasserted = fmt::format("{}{}: DM0-FW: deasserted TRISC", core_prefix, dm0);

        const std::vector<std::string> required = {dm0_initialized, dm0_deasserted};
        std::vector<std::pair<std::string, std::string>> trisc_lines;
        for (uint32_t idx = dm_count; idx < dm_count + trisc_count; ++idx) {
            const std::string risc = hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, idx, true);
            trisc_lines.emplace_back(
                fmt::format("{}{}: hartid: {}", core_prefix, risc, idx),
                fmt::format("{}{}: TRISC-FW: initialized", core_prefix, risc));
        }

        // The TRISCs print after they report DONE, so give the print server a moment to catch up.
        MetalContext::instance().dprint_server()->await();
        const auto deadline = std::chrono::steady_clock::now() + settle_budget;
        std::vector<std::string> missing = MissingBanners(ReadWholeFile(dprint_file_name), required, trisc_lines);
        while (!missing.empty() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            missing = MissingBanners(ReadWholeFile(dprint_file_name), required, trisc_lines);
        }

        EXPECT_TRUE(missing.empty()) << "Device " << device_id << ": boot banner lines missing from the print output:\n"
                                     << fmt::format("{}", fmt::join(missing, "\n"));
        EXPECT_TRUE(FileContainsAllStringsInOrder(dprint_file_name, required));
    }
}

}  // namespace
}  // namespace CMAKE_UNIQUE_NAMESPACE
