// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include "tools/tt_builder/builder.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

TEST(BuilderTest, test_firmware_build) {
    tt::tt_metal::BuilderTool builder;

    std::filesystem::path output_path(
        tt::llrt::RunTimeOptions::get_instance().get_root_dir() + "test_builder_firmware_build/");
    if (std::filesystem::exists(output_path)) {
        std::filesystem::remove_all(output_path);
    }
    EXPECT_EQ(false, std::filesystem::exists(output_path));

    builder.set_built_path(output_path.c_str());
    EXPECT_EQ(output_path.c_str(), builder.get_built_path());

    EXPECT_NO_THROW(builder.build_firmware());

    EXPECT_EQ(true, std::filesystem::exists(output_path));

    std::filesystem::path firmware_output_path(builder.get_firmware_root_path());
    // Remove the last '/' from path if it exists.
    if (!firmware_output_path.empty() && firmware_output_path.filename() == "") {
        firmware_output_path = firmware_output_path.parent_path();
    }
    // Remove the last '/' from path for comparison
    output_path = output_path.parent_path();

    std::filesystem::path firmware_parent_root(firmware_output_path.parent_path().parent_path());
    EXPECT_EQ(0, firmware_parent_root.compare(output_path));

    uint32_t elf_file_count = 0;
    std::ranges::for_each(
        std::filesystem::directory_iterator{firmware_output_path}, [&elf_file_count](const auto& fw_files) {
            elf_file_count += (fw_files.path().extension().compare(".elf") == 0);
        });

    EXPECT_EQ(elf_file_count, tt::tt_metal::hal.get_num_risc_processors());
}
