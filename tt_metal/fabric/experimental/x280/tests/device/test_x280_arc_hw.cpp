// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Raw-UMD ARC message canary. This remains outside X280Runtime because Metal's
// wrapper exposes neither get_clocks() nor arc_msg().
//
//   TT_METAL_RUNTIME_ROOT=<tt-metal> [TT_VISIBLE_DEVICES=N]
//   build/test/tt_metal/x280/x280_arc_test

#include "x280_device_test_support.hpp"

#include <gtest/gtest.h>

#include <tt-umd/types/blackhole_arc.hpp>
#include <tt-umd/types/noc_id.hpp>

#include <cstdint>
#include <cstdio>
#include <string>

using namespace x280::test;

TEST(X280Arc, ArcMessageRoundTrip) {
    tt::umd::NocIdSwitcher noc_guard(tt::umd::NocId::NOC0);

    auto clocks = driver().get_clocks();
    ASSERT_FALSE(clocks.empty()) << "no clocks returned; ARC message path failed";
    for (const auto& [id, mhz] : clocks) {
        printf("ARC get_clocks: chip %d = %d MHz\n", id, mhz);
    }

    Coverage cov{"ArcMessage", "chips"};
    for (const tt::ChipId chip : cluster().all_chip_ids()) {
        SCOPED_TRACE(chip_label(chip));
        cov.total++;

        try {
            uint32_t ret3 = 0;
            uint32_t ret4 = 0;
            const int rc = driver().arc_msg(
                chip,
                static_cast<uint32_t>(tt::umd::blackhole::ArcMessageType::NOP),
                /*wait_for_done=*/true,
                /*args=*/{},
                tt::umd::timeout::ARC_MESSAGE_TIMEOUT,
                &ret3,
                &ret4);
            printf("%s: arc_msg(NOP) accepted: rc=%d ret3=0x%x ret4=0x%x\n", chip_label(chip).c_str(), rc, ret3, ret4);
        } catch (const std::exception& error) {
            const std::string what = error.what();
            if (what.find("not recognized by ARC firmware") != std::string::npos) {
                printf(
                    "%s: arc_msg(NOP) rejected by CMFW (expected on CMFW bundles without NOP): %s\n",
                    chip_label(chip).c_str(),
                    what.c_str());
            } else {
                ADD_FAILURE() << "arc_msg(NOP) failed without a well-formed CMFW rejection: " << what;
            }
        }
        cov.covered++;
    }
    cov.finish();
}
