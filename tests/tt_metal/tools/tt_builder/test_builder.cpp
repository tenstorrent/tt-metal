// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include "tools/tt_builder/builder.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/common/logger.hpp"

TEST(BuilderTest, test_firmware_build) {
    tt::tt_metal::tt_builder builder;

    std::filesystem::path output_path("/tmp/test_builder_firmware_build/");
    if (std::filesystem::exists(output_path)) {
        std::filesystem::remove_all(output_path);
    }
    EXPECT_EQ(false, std::filesystem::exists(output_path));

    builder.set_built_path(output_path.c_str());
    EXPECT_EQ(output_path.c_str(), builder.get_built_path());

    EXPECT_NO_THROW(builder.build_firmware());

    EXPECT_EQ(true, std::filesystem::exists(output_path));

    uint32_t elf_file_count = 0;
    std::ranges::for_each(
        std::filesystem::directory_iterator{output_path.string() + "firmware/"},
        [&elf_file_count](const auto& fw_files) {
            elf_file_count += (fw_files.path().extension().compare(".elf") == 0);
        });

    EXPECT_EQ(elf_file_count, tt::tt_metal::hal.get_num_risc_processors());
}
