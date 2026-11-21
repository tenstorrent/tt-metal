// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>

namespace {

bool emulator_server_configured() {
    return std::getenv("TT_METAL_EMU_SERVER") != nullptr || std::getenv("TT_METAL_GRENDEL_JTAG_SERVER") != nullptr;
}

// A skip from a global Environment's SetUp stops the run but reports every test as passed (gtest
// 1.13). A skip recorded when the test starts is reported per test, and gtest then skips the body.
class GrendelEmuGate : public ::testing::EmptyTestEventListener {
public:
    void OnTestStart(const ::testing::TestInfo& /*test_info*/) override {
        if (!emulator_server_configured()) {
            GTEST_SKIP() << "These tests need a Grendel emulator. Set TT_METAL_EMU_SERVER=host:port and "
                            "TT_METAL_EMU_SOC_DESC=<package YAML>, or TT_METAL_GRENDEL_JTAG_SERVER and "
                            "TT_METAL_GRENDEL_JTAG_SOC_DESC.";
        }
    }
};

[[maybe_unused]] const bool grendel_emu_gate_installed = [] {
    ::testing::UnitTest::GetInstance()->listeners().Append(new GrendelEmuGate);
    return true;
}();

}  // namespace
