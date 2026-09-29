// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "jit_build/build.hpp"
#include "jit_build/jit_device_config.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// GCC's -H output marks a loaded PCH with "! "; textual fallback does not satisfy this check.
std::filesystem::path loaded_pch(const std::filesystem::path& log) {
    std::ifstream file(log);
    for (std::string line; std::getline(file, line);) {
        if (line.starts_with("! ") && line.ends_with(".gch")) {
            return line.substr(2);
        }
    }
    throw std::runtime_error("No PCH loaded in " + log.string() + ":\n" + read_text(log));
}

// Keep the environment and RISC-specific flags from the real constructor. Only substitute
// a tiny source so this exercises compile_one without linking firmware or opening a device.
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
        cflags_ += " -H";
    }
    std::filesystem::path run(const std::filesystem::path& output) const {
        namespace fs = std::filesystem;
        fs::create_directories(output);
        compile_one(output.string() + "/", nullptr, 0);
        const auto pch = loaded_pch(output / "probe.o.log");

        // compile_one permits invalid-PCH fallback. Recheck its selected artifact with
        // that warning made fatal, using the same exported recipe and source.
        const auto recipe = export_target_recipe(nullptr);
        auto defines = recipe.defines;
        defines.insert(defines.begin(), {"-include", fs::path(pch).replace_extension().string()});
        const auto args = jit_build::utils::build_gpp_argv(
            env_.get_gpp(),
            recipe.compiler_opt_level,
            recipe.cflags + " -Werror=invalid-pch",
            recipe.includes,
            defines,
            srcs_[0],
            jit_build::utils::GppAction::Compile,
            (output / "strict.o").string(),
            (output / "strict.d").string());
        const auto log = output / "strict.log";
        if (!jit_build::utils::exec_command(args, output.string(), log.string()) || loaded_pch(log) != pch) {
            throw std::runtime_error("PCH validation failed: " + read_text(log));
        }
        return pch;
    }
};

TEST(PchBuildDeathTest, SharesWithinArchitectureAndSeparatesHalTargetFlags) {
    namespace fs = std::filesystem;
    auto pattern = (fs::temp_directory_path() / "pch_build_XXXXXX").string();
    const auto* directory = mkdtemp(pattern.data());
    ASSERT_NE(directory, nullptr);
    const fs::path root = directory;
    // Environment changes and JIT memoization are confined to the child.
    const auto compile_profiles = [&] {
        setenv("TT_METAL_CACHE", (root.string() + "/").c_str(), 1);
        llrt::RunTimeOptions options;
        const Hal wh_hal(tt::ARCH::WORMHOLE_B0, false, false, 0, false);
        const Hal bh_hal(tt::ARCH::BLACKHOLE, false, false, 0, false);
        const JitDeviceConfig wh_config{.hal = &wh_hal, .arch = tt::ARCH::WORMHOLE_B0, .max_dfbs = 32};
        const JitDeviceConfig bh_config{.hal = &bh_hal, .arch = tt::ARCH::BLACKHOLE, .max_dfbs = 32};
        JitBuildEnv first;
        JitBuildEnv second;
        JitBuildEnv blackhole;
        first.init(1, wh_config, options, {});
        second.init(2, wh_config, options, {});
        blackhole.init(3, bh_config, options, {});
        ASSERT_NE(first.get_build_key(), second.get_build_key());
        ASSERT_NE(first.get_build_key(), blackhole.get_build_key());
        ASSERT_NE(second.get_build_key(), blackhole.get_build_key());

        const auto source = root / "probe.cpp";
        std::ofstream(source) << "int probe() { return 42; }\n";
        const PchCompileProbe wh_first(first, wh_hal, source.string());
        const PchCompileProbe wh_second(second, wh_hal, source.string());
        const PchCompileProbe bh(blackhole, bh_hal, source.string());
        const auto wh_flags = jit_build::utils::tokenize_flags(wh_first.export_target_recipe(nullptr).cflags);
        const auto bh_flags = jit_build::utils::tokenize_flags(bh.export_target_recipe(nullptr).cflags);
        ASSERT_NE(std::find(wh_flags.begin(), wh_flags.end(), "-mcpu=tt-wh"), wh_flags.end());
        ASSERT_NE(std::find(bh_flags.begin(), bh_flags.end(), "-mcpu=tt-bh"), bh_flags.end());
        ASSERT_EQ(wh_first.export_target_recipe(nullptr).cflags, wh_second.export_target_recipe(nullptr).cflags);

        const auto wh_pch = wh_first.run(root / "first");
        ASSERT_EQ(wh_pch, wh_second.run(root / "second"));
        ASSERT_NE(wh_pch, bh.run(root / "blackhole"));
        std::_Exit(0);
    };
    EXPECT_EXIT(compile_profiles(), ::testing::ExitedWithCode(0), "");
    std::size_t artifacts = 0;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.path().extension() == ".gch") {
            ++artifacts;
            EXPECT_EQ(entry.path().parent_path().parent_path(), root / "tt-metal-cache" / "pch");
            EXPECT_GT(entry.file_size(), 0u);
        }
    }
    EXPECT_EQ(artifacts, 2u);
    EXPECT_TRUE(fs::exists(root / "first/probe.tmp.o"));
    EXPECT_TRUE(fs::exists(root / "second/probe.tmp.o"));
    EXPECT_TRUE(fs::exists(root / "blackhole/probe.tmp.o"));
    fs::remove_all(root);
}

}  // namespace
}  // namespace tt::tt_metal
