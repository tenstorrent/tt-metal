// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TT_METAL_SIMULATOR_DRAM_UNCAPPED=1 lets a Quasar emulator run address the DRAM its soc descriptor
// declares instead of the DRAM the model backs. On a simulator the option and the
// quasar_aether_2x3_uncapped ATT map imply each other. The option moves TT_METAL_NOC_ATT=quasar_aether_2x3
// to the uncapped map, and naming the uncapped map sets the option.

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

#include "llrt/rtoptions.hpp"

namespace {

// Set (or clear) one environment variable for the test's life and put the old value back.
class ScopedDramUncappedEnv {
public:
    ScopedDramUncappedEnv(const char* name, const std::optional<std::string>& value) : name_(name) {
        if (const char* old = std::getenv(name)) {
            old_ = old;
        }
        apply(value);
    }
    ~ScopedDramUncappedEnv() { apply(old_); }

private:
    void apply(const std::optional<std::string>& value) {
        if (value.has_value()) {
            setenv(name_, value->c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }
    const char* name_;
    std::optional<std::string> old_;
};

struct Parsed {
    bool dram_uncapped;
    std::string att_map;
};

Parsed parse(
    const std::optional<std::string>& simulator,
    const std::optional<std::string>& dram_uncapped,
    const std::optional<std::string>& att_map) {
    ScopedDramUncappedEnv simulator_env("TT_METAL_SIMULATOR", simulator);
    ScopedDramUncappedEnv uncapped_env("TT_METAL_SIMULATOR_DRAM_UNCAPPED", dram_uncapped);
    ScopedDramUncappedEnv att_env("TT_METAL_NOC_ATT", att_map);
    tt::llrt::RunTimeOptions opts;
    return {opts.get_simulator_dram_uncapped(), std::string(opts.get_noc_att_map().value_or(std::string_view{}))};
}

const std::string EMU_2X3 = "/models/emu-quasar-2x3";

}  // namespace

TEST(RunTimeOptionsSimulatorDramUncapped, CPU_OneEnablesTheOption) {
    EXPECT_TRUE(parse(EMU_2X3, "1", std::nullopt).dram_uncapped);
    EXPECT_FALSE(parse(EMU_2X3, "0", std::nullopt).dram_uncapped);
    EXPECT_FALSE(parse(EMU_2X3, std::nullopt, std::nullopt).dram_uncapped);
}

TEST(RunTimeOptionsSimulatorDramUncapped, CPU_TheOptionMovesAether2x3ToTheUncappedMap) {
    EXPECT_EQ(parse(EMU_2X3, "1", "quasar_aether_2x3").att_map, "quasar_aether_2x3_uncapped");
    // Other maps and plain XY are left alone.
    EXPECT_EQ(parse(EMU_2X3, "1", "grendel_qsr1").att_map, "grendel_qsr1");
    EXPECT_EQ(parse(EMU_2X3, "1", std::nullopt).att_map, "");
    EXPECT_EQ(parse(EMU_2X3, "1", "off").att_map, "");
    // Without the option the map is unchanged.
    EXPECT_EQ(parse(EMU_2X3, std::nullopt, "quasar_aether_2x3").att_map, "quasar_aether_2x3");
}

TEST(RunTimeOptionsSimulatorDramUncapped, CPU_TheUncappedMapSetsTheOption) {
    const Parsed unset = parse(EMU_2X3, std::nullopt, "quasar_aether_2x3_uncapped");
    EXPECT_TRUE(unset.dram_uncapped);
    EXPECT_EQ(unset.att_map, "quasar_aether_2x3_uncapped");
    // The map name is the more specific request and wins over an explicit 0.
    EXPECT_TRUE(parse(EMU_2X3, "0", "quasar_aether_2x3_uncapped").dram_uncapped);
}

TEST(RunTimeOptionsSimulatorDramUncapped, CPU_WithoutASimulatorNothingIsNormalised) {
    const Parsed option_only = parse(std::nullopt, "1", "quasar_aether_2x3");
    EXPECT_TRUE(option_only.dram_uncapped);
    EXPECT_EQ(option_only.att_map, "quasar_aether_2x3");
    EXPECT_FALSE(parse(std::nullopt, std::nullopt, "quasar_aether_2x3_uncapped").dram_uncapped);
}
