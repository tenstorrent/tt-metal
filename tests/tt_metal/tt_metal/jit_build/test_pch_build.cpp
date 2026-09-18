// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>

#include "jit_build/build.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {
namespace {

// Exercise compile_one's actual choice of PCH root, without linking firmware or opening a device.
class PchCompileProbe : public JitBuildState {
public:
    PchCompileProbe(const JitBuildEnv& env, const Hal& hal, const std::string& source) :
        JitBuildState(
            env,
            {.core_type = HalProgrammableCoreType::TENSIX,
             .processor_class = HalProcessorClassType::DM,
             .processor_id = 0,
             .is_fw = true},
            hal) {
        srcs_ = {source};
        objs_ = {"probe.o"};
        temp_objs_ = {"probe.tmp.o"};
        cflags_ = "-std=c++17 -mcpu=tt-wh -MMD";
        defines_.clear();
        includes_.clear();
    }
    void run(const std::filesystem::path& output) const {
        std::filesystem::create_directories(output);
        compile_one(output.string() + "/", nullptr, 0);
    }
};

TEST(PchBuildDeathTest, DifferentBuildKeysShareOneArtifact) {
    namespace fs = std::filesystem;
    auto pattern = (fs::temp_directory_path() / "pch_build_XXXXXX").string();
    const auto* directory = mkdtemp(pattern.data());
    ASSERT_NE(directory, nullptr);
    const fs::path root = directory;
    // Environment changes and JIT memoization are confined to the child.
    EXPECT_EXIT(
        ([&] {
            setenv("TT_METAL_CACHE", (root.string() + "/").c_str(), 1);
            llrt::RunTimeOptions options;
            const Hal hal(tt::ARCH::WORMHOLE_B0, false, false, 0, false);
            const JitDeviceConfig config{.hal = &hal, .arch = tt::ARCH::WORMHOLE_B0, .max_cbs = 32};
            JitBuildEnv first;
            JitBuildEnv second;
            first.init(1, config, options, {});
            second.init(2, config, options, {});
            if (first.get_build_key() == second.get_build_key()) {
                std::_Exit(1);
            }
            const auto source = root / "probe.cpp";
            std::ofstream(source) << "int probe() { return 42; }\n";
            PchCompileProbe(first, hal, source.string()).run(root / "first");
            PchCompileProbe(second, hal, source.string()).run(root / "second");
            std::_Exit(0);
        }()),
        ::testing::ExitedWithCode(0),
        "");
    std::size_t artifacts = 0;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.path().extension() == ".gch") {
            ++artifacts;
            EXPECT_EQ(entry.path().parent_path().parent_path(), root / "tt-metal-cache" / "pch");
            EXPECT_GT(entry.file_size(), 0u);
        }
    }
    EXPECT_EQ(artifacts, 1u);
    EXPECT_TRUE(fs::exists(root / "first/probe.tmp.o"));
    EXPECT_TRUE(fs::exists(root / "second/probe.tmp.o"));
    fs::remove_all(root);
}

}  // namespace
}  // namespace tt::tt_metal
