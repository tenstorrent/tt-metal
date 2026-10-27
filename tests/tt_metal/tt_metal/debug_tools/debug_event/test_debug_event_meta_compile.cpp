// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Debug event metadata, layer 2: what the device compiler says about a site. Each case of
// test_kernels/misc/debug_event_meta_cases.cpp is built offline (experimental::CompileKernelOffline, mock mode, no
// device), so this runs the real device compiler with the real kernel flags and opens nothing that a failed build
// could leave wedged. A bad case must fail to compile with exactly one error, its own rule's; the valid case must
// compile in every profiler mode.
//
// The offline path has no firmware for a profiler build_key, so a kernel built under a profiler compiles but cannot
// link. There the valid case passes when the only failure is that link error. The streaming and DRAM profilers need a
// Tracy-enabled build; without one only the profiler-off tests run.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/experimental/offline_kernel_compile.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "llrt/rtoptions.hpp"

namespace debug_event_compile_test {

using namespace tt::tt_metal;
namespace fs = std::filesystem;

constexpr const char* kCasesKernel = "tests/tt_metal/tt_metal/test_kernels/misc/debug_event_meta_cases.cpp";

// What the linker says when a kernel has no firmware ELF to link against.
constexpr std::string_view kNoFirmwareLinkError = "non constant or forward reference address expression";

// Every rule message in hostdev/debug_event_meta.h, so a case can be checked for breaking only its own rule.
constexpr std::string_view kRuleMessages[] = {
    "must be a simple aggregate",
    "cannot have a base class",
    "cannot have C array members",
    "need 1..kMaxFields fields",
    "cannot have const members",
    "a metadata field must be a string literal",
    "must be named at namespace scope",
    "the first field of a debug event metadata type must be the site's name",
    "name_must_be_a_non_empty_string_literal",
    "debug_event_site_metadata_too_long_shorten_its_strings",
};

// Sets or clears an environment variable for one scope; CompileKernelOffline reads the environment on every call.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        if (const char* old = std::getenv(name)) {
            old_ = old;
        }
        value != nullptr ? setenv(name, value, 1) : unsetenv(name);
    }
    ~ScopedEnv() { old_ ? setenv(name_, old_->c_str(), 1) : unsetenv(name_); }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    const char* name_;
    std::optional<std::string> old_;
};

enum class Profiler { Off, Dram, Streaming };

struct ProfilerMode {
    explicit ProfilerMode(Profiler p) :
        streaming("TT_METAL_STREAMING_PROFILER", p == Profiler::Streaming ? "1" : nullptr),
        dram("TT_METAL_DEVICE_PROFILER", p == Profiler::Dram ? "1" : nullptr) {}
    ScopedEnv streaming;
    ScopedEnv dram;
};

struct ScopedTempDir {
    ScopedTempDir() {
        const auto ns = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("debug_event_meta_compile_" + std::to_string(ns));
        fs::create_directories(path);
    }
    ~ScopedTempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    fs::path path;
};

// The build's failure text, or nothing if every product built.
std::optional<std::string> compile_case(int which) {
    ScopedTempDir out;
    const DataMovementConfig config{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .defines = {{"DEBUG_EVENT_META_CASE", std::to_string(which)}},
    };
    try {
        experimental::CompileKernelOffline(kCasesKernel, config, {.output_dir = out.path});
    } catch (const std::exception& e) {
        return std::string(e.what());
    }
    return std::nullopt;
}

// A compiler diagnostic ("file:line:col: error:"), as opposed to a linker one.
bool has_compile_error(const std::string& text) {
    static const std::regex diagnostic(R"(\.(cpp|cc|h|hpp):[0-9]+:[0-9]+: error:)");
    return std::regex_search(text, diagnostic);
}

// A build that failed only because the offline path has no firmware to link against: no compiler error, and no
// assembler message either -- the assembler only warns about text it mangles (an unterminated string, say) and
// carries on.
void expect_only_the_link_error(const std::string& text) {
    EXPECT_FALSE(has_compile_error(text)) << text.substr(0, 2000);
    EXPECT_EQ(text.find("Assembler messages:"), std::string::npos) << text.substr(0, 2000);
    EXPECT_NE(text.find(kNoFirmwareLinkError), std::string::npos) << text.substr(0, 2000);
}

// How many compiler errors the build reported.
size_t error_count(const std::string& text) {
    static const std::regex error(R"(: error:)");
    return static_cast<size_t>(
        std::distance(std::sregex_iterator(text.begin(), text.end(), error), std::sregex_iterator()));
}

// Whether this build can arm profiler P: the streaming and DRAM profilers reject their variable without Tracy.
bool profiler_supported(Profiler p) {
    if (p == Profiler::Off) {
        return true;
    }
    ProfilerMode mode(p);
    try {
        tt::llrt::RunTimeOptions options;
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

// The first compiler error line, for failure messages; the whole log is long.
std::string first_error(const std::string& text) {
    std::smatch m;
    static const std::regex line(R"([^\n]*: error:[^\n]*)");
    return std::regex_search(text, m, line) ? m.str() : text.substr(0, 2000);
}

class DebugEventMetaCompile : public ::testing::Test {
protected:
    void SetUp() override {
        {
            ProfilerMode off(Profiler::Off);
            if (tt::llrt::RunTimeOptions{}.is_simulator_or_emulated()) {
                GTEST_SKIP() << "the offline compile has no firmware for a simulator build_key";
            }
        }
        experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1);
        mock_mode_ = true;
    }
    void TearDown() override {
        if (mock_mode_) {
            experimental::disable_mock_mode();
        }
    }

    // Case WHICH must fail to compile under profiler P with exactly one error, EXPECTED, and no other rule's message.
    // Skips the test when this build cannot arm P.
    void expect_rejected(int which, std::string_view expected, Profiler p = Profiler::Streaming) {
        if (!profiler_supported(p)) {
            GTEST_SKIP() << "this profiler needs a Tracy-enabled build";
        }
        ProfilerMode mode(p);
        const std::optional<std::string> error = compile_case(which);
        ASSERT_TRUE(error.has_value()) << "case " << which << " compiled";
        ASSERT_TRUE(has_compile_error(*error)) << "case " << which << " did not fail in the compiler:\n"
                                               << error->substr(0, 2000);
        EXPECT_NE(error->find(expected), std::string::npos)
            << "case " << which << " lacks \"" << expected << "\"; first error: " << first_error(*error);
        EXPECT_EQ(error_count(*error), 1u) << "case " << which << ":\n" << error->substr(0, 4000);
        for (std::string_view other : kRuleMessages) {
            if (other != expected) {
                EXPECT_EQ(error->find(other), std::string::npos)
                    << "case " << which << " also reports \"" << other << "\"";
            }
        }
    }

private:
    bool mock_mode_ = false;
};

// ---- The valid case, in every profiler mode -------------------------------------------------------------------------

TEST_F(DebugEventMetaCompile, CPU_ValidSitesCompileUnderTheStreamingProfiler) {
    if (!profiler_supported(Profiler::Streaming)) {
        GTEST_SKIP() << "the streaming profiler needs a Tracy-enabled build";
    }
    ProfilerMode streaming(Profiler::Streaming);
    const std::optional<std::string> error = compile_case(0);
    if (error) {
        expect_only_the_link_error(*error);
    }
}

TEST_F(DebugEventMetaCompile, CPU_ValidSitesCompileUnderTheDramProfiler) {
    if (!profiler_supported(Profiler::Dram)) {
        GTEST_SKIP() << "the DRAM profiler needs a Tracy-enabled build";
    }
    ProfilerMode dram(Profiler::Dram);
    const std::optional<std::string> error = compile_case(0);
    if (error) {
        expect_only_the_link_error(*error);
    }
}

TEST_F(DebugEventMetaCompile, CPU_ValidSitesBuildWithTheProfilerOff) {
    ProfilerMode off(Profiler::Off);
    const std::optional<std::string> error = compile_case(0);
    EXPECT_FALSE(error.has_value()) << error.value_or("").substr(0, 2000);
}

// ---- One rule broken per case, under the streaming profiler ---------------------------------------------------------

TEST_F(DebugEventMetaCompile, CPU_RejectsAnArrayMember) { expect_rejected(1, "cannot have C array members"); }

TEST_F(DebugEventMetaCompile, CPU_RejectsAnArrayInANestedStruct) { expect_rejected(2, "cannot have C array members"); }

TEST_F(DebugEventMetaCompile, CPU_RejectsAConstMember) { expect_rejected(3, "cannot have const members"); }

TEST_F(DebugEventMetaCompile, CPU_RejectsAFloatField) {
    expect_rejected(4, "a metadata field must be a string literal");
}

TEST_F(DebugEventMetaCompile, CPU_RejectsABaseClass) { expect_rejected(5, "cannot have a base class"); }

TEST_F(DebugEventMetaCompile, CPU_RejectsAFirstFieldThatIsNotTheName) {
    expect_rejected(6, "the first field of a debug event metadata type must be the site's name");
}

TEST_F(DebugEventMetaCompile, CPU_RejectsAnEmptyName) { expect_rejected(7, "name_must_be_a_non_empty_string_literal"); }

TEST_F(DebugEventMetaCompile, CPU_RejectsATypeInAnAnonymousNamespace) {
    expect_rejected(8, "must be named at namespace scope");
}

TEST_F(DebugEventMetaCompile, CPU_RejectsAFunctionLocalType) { expect_rejected(9, "must be named at namespace scope"); }

// GCC's own error, pinned so a clearer message is a deliberate change. The site's value is evaluated inside the
// site's own local struct, hence "from containing function" rather than "not a constant expression".
TEST_F(DebugEventMetaCompile, CPU_RejectsARunTimeValue) {
    expect_rejected(10, "use of local variable with automatic storage from containing function");
}

TEST_F(DebugEventMetaCompile, CPU_RejectsMoreThanMaxFields) { expect_rejected(11, "need 1..kMaxFields fields"); }

TEST_F(DebugEventMetaCompile, CPU_RejectsStringsTooLongForOneSite) {
    expect_rejected(12, "debug_event_site_metadata_too_long_shorten_its_strings");
}

}  // namespace debug_event_compile_test
