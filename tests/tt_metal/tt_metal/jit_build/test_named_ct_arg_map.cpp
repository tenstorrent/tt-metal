// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "jit_build/jit_build_utils.hpp"

// The named compile-time-arg map (KERNEL_COMPILE_TIME_ARG_MAP) is delivered to kernels as a
// force-included generated header rather than a -D define. These tests cover the formatter that
// produces that header, and pin the OS constraint that forces the design: a -D define lands in a
// single argv element, and one element is capped at MAX_ARG_STRLEN (128 KB) regardless of how the
// compiler is spawned. Whole-model fused ops emit maps larger than that on their own.
//
// See tenstorrent/tt-blaze#2017 and tenstorrent/tt-blaze#3209.

namespace tt::jit_build::utils {
namespace {

// Linux caps a single argv element at 32 pages, independent of the total ARG_MAX budget.
std::size_t max_arg_strlen() { return 32 * static_cast<std::size_t>(sysconf(_SC_PAGESIZE)); }

// Parse a formatted map body ({"a",1},{"b",2}) back into a map, so a round-trip can assert that
// formatting is lossless. Deliberately strict: anything unexpected fails the test rather than
// being skipped.
std::unordered_map<std::string, std::uint32_t> parse_map_body(std::string_view body) {
    std::unordered_map<std::string, std::uint32_t> parsed;
    std::size_t pos = 0;
    while (pos < body.size()) {
        if (body[pos] == ',') {
            ++pos;
            continue;
        }
        EXPECT_EQ(body[pos], '{') << "at offset " << pos;
        const std::size_t key_open = body.find('"', pos);
        const std::size_t key_close = body.find('"', key_open + 1);
        const std::size_t entry_end = body.find('}', key_close);
        if (key_open == std::string_view::npos || key_close == std::string_view::npos ||
            entry_end == std::string_view::npos) {
            ADD_FAILURE() << "malformed entry at offset " << pos;
            break;
        }
        const std::string key(body.substr(key_open + 1, key_close - key_open - 1));
        // Skip the comma between the closing key quote and the value.
        const std::string value(body.substr(key_close + 2, entry_end - key_close - 2));
        parsed[key] = static_cast<std::uint32_t>(std::stoul(value));
        pos = entry_end + 1;
    }
    return parsed;
}

// A map shaped like what a whole-model decoder layer emits: ~1750 entries whose keys average ~75
// characters, because every key repeats the op's naming scope. This is the size class that broke
// the -D delivery.
std::unordered_map<std::string, std::uint32_t> make_whole_model_sized_map(std::size_t num_entries = 1750) {
    std::unordered_map<std::string, std::uint32_t> named_args;
    named_args.reserve(num_entries);
    for (std::size_t i = 0; i < num_entries; ++i) {
        named_args.emplace(
            "gemma4_sliding_window_decoder_layer__attn__qkv_proj__mcast_" + std::to_string(i) +
                ".execute_receiver_on_ncrisc_arg",
            static_cast<std::uint32_t>(i));
    }
    return named_args;
}

// Try to exec /bin/true with one argv element of |len| bytes. Returns the posix_spawn errno
// (0 on success), which is E2BIG once the element crosses MAX_ARG_STRLEN.
int spawn_with_argv_element_of_length(std::size_t len) {
    std::string big(len, 'x');
    std::vector<char*> argv = {const_cast<char*>("/bin/true"), big.data(), nullptr};

    pid_t pid = 0;
    const int rc = posix_spawn(&pid, argv[0], nullptr, nullptr, argv.data(), environ);
    if (rc == 0) {
        int status = 0;
        waitpid(pid, &status, 0);
    }
    return rc;
}

TEST(NamedCtArgMap, EmptyMapFormatsToEmptyBody) { EXPECT_EQ(format_named_ct_arg_map({}), ""); }

TEST(NamedCtArgMap, FormatsEntriesAsCommaSeparatedBracedPairs) {
    EXPECT_EQ(format_named_ct_arg_map({{"cb_in0", 1}}), R"({"cb_in0",1})");
    EXPECT_EQ(format_named_ct_arg_map({{"b", 2}, {"a", 1}}), R"({"a",1},{"b",2})");
}

// The formatter reads an unordered_map, whose iteration order is not reproducible across runs or
// processes. Sorting matters because this text is written to disk and hashed by the per-object
// dephash cache: unsorted output would churn the cache and force spurious recompiles.
TEST(NamedCtArgMap, OrdersByKeySoOutputIsReproducible) {
    std::unordered_map<std::string, std::uint32_t> forward;
    std::unordered_map<std::string, std::uint32_t> reverse;
    for (int i = 0; i < 64; ++i) {
        forward.emplace("arg_" + std::to_string(i), i);
    }
    for (int i = 63; i >= 0; --i) {
        reverse.emplace("arg_" + std::to_string(i), i);
    }

    const std::string forward_body = format_named_ct_arg_map(forward);
    EXPECT_EQ(forward_body, format_named_ct_arg_map(reverse));

    // Spot-check that it really is ascending key order, not just consistent.
    EXPECT_LT(forward_body.find(R"({"arg_0",0})"), forward_body.find(R"({"arg_1",1})"));
}

// Covers the key shapes the existing named-CT-arg kernel test relies on, including the empty name
// and punctuation-only names.
TEST(NamedCtArgMap, RoundTripsEveryEntryLosslessly) {
    const std::unordered_map<std::string, std::uint32_t> named_args = {
        {"buffer_size", 1024},
        {"", 3},
        {"!@#$%^&*()", 12},
        {"very_long_parameter_name_that_someone_could_potentially_use_to_try_to_break_the_kernel", 456},
        {"zero", 0},
        {"max", 0xFFFFFFFFu},
    };

    EXPECT_EQ(parse_map_body(format_named_ct_arg_map(named_args)), named_args);
}

TEST(NamedCtArgMap, RoundTripsAWholeModelSizedMap) {
    const auto named_args = make_whole_model_sized_map();
    const auto parsed = parse_map_body(format_named_ct_arg_map(named_args));

    EXPECT_EQ(parsed.size(), named_args.size());
    EXPECT_EQ(parsed, named_args);
}

TEST(NamedCtArgMap, HeaderDefinesTheMacroAndIsIncludeGuarded) {
    const std::string header = format_named_ct_arg_map_header({{"cb_in0", 1}});

    EXPECT_NE(header.find("#pragma once"), std::string::npos);
    EXPECT_NE(header.find(R"(#define KERNEL_COMPILE_TIME_ARG_MAP {"cb_in0",1})"), std::string::npos);
    // The macro must be the whole body on one logical line: a stray newline inside the map would
    // terminate the #define and silently truncate the arg list.
    const std::size_t define_pos = header.find("#define KERNEL_COMPILE_TIME_ARG_MAP");
    EXPECT_EQ(header.find('\n', define_pos), header.size() - 1);
}

TEST(NamedCtArgMap, HeaderIsByteIdenticalForEqualMaps) {
    const auto named_args = make_whole_model_sized_map(256);
    std::unordered_map<std::string, std::uint32_t> rebuilt(named_args.begin(), named_args.end());

    EXPECT_EQ(format_named_ct_arg_map_header(named_args), format_named_ct_arg_map_header(rebuilt));
}

// The regression this delivery mechanism exists to prevent: a whole-model map does not fit in one
// argv element, but the recipe footprint that replaces it is a fixed couple of dozen bytes.
TEST(NamedCtArgMap, WholeModelMapExceedsArgvLimitWhileRecipeFootprintStaysTiny) {
    const std::string body = format_named_ct_arg_map(make_whole_model_sized_map());
    const std::string as_define = "-DKERNEL_COMPILE_TIME_ARG_MAP=" + body;

    EXPECT_GT(as_define.size(), max_arg_strlen())
        << "test map is no longer representative of a whole-model kernel; it must exceed " << max_arg_strlen()
        << " bytes to exercise the limit";

    // What actually goes on the command line now, regardless of how big the map is.
    const std::size_t recipe_footprint = std::strlen("-include") + NAMED_CT_ARG_MAP_HEADER.size();
    EXPECT_LT(recipe_footprint, 128u);

    // And the map itself still travels intact, in the header.
    EXPECT_NE(format_named_ct_arg_map_header(make_whole_model_sized_map()).find(body), std::string::npos);
}

// Pins the constraint the design is built around. posix_spawn removed tt-metal's whole-command
// ceiling, but not this per-element one -- which is why a large map still could not ride on a -D.
TEST(NamedCtArgMap, KernelRejectsAnArgvElementAtOrAboveMaxArgStrlen) {
    const std::size_t limit = max_arg_strlen();

    EXPECT_EQ(spawn_with_argv_element_of_length(limit - 1), 0) << "just under the limit should exec";
    EXPECT_EQ(spawn_with_argv_element_of_length(limit), E2BIG);
    EXPECT_EQ(spawn_with_argv_element_of_length(limit + 4096), E2BIG);
}

// A bare filename (rather than a path) is what lets one recipe compile on both the local host and
// the remote JIT compile server, whose per-kernel directory layout differs.
TEST(NamedCtArgMap, HeaderNameIsRelativeSoItResolvesOnBothCompileHosts) {
    EXPECT_FALSE(NAMED_CT_ARG_MAP_HEADER.empty());
    EXPECT_EQ(NAMED_CT_ARG_MAP_HEADER.find('/'), std::string_view::npos);
    EXPECT_TRUE(NAMED_CT_ARG_MAP_HEADER.ends_with(".h"));
}

// End-to-end check of the delivery mechanism, in the directory layout the real compiles use: the
// generated header sits in the per-kernel dir and the compiler runs with cwd = a per-target subdir,
// reaching it through the "-I.." on every compile. Both the local build and the JIT compile server
// arrange things this way, which is what lets a bare -include work identically on both -- an
// absolute client-side path would compile locally and then fail on the server.
//
// Uses the host compiler and only the preprocessor, so it needs neither a device nor the RISC-V
// toolchain: what is under test is name resolution and macro visibility, not code generation.
TEST(NamedCtArgMap, ForceIncludedHeaderResolvesFromTargetSubdirAndDefinesTheMap) {
    namespace fs = std::filesystem;

    const fs::path kernel_dir = fs::temp_directory_path() / "tt_named_ct_arg_map_test" / "kernel";
    const fs::path target_dir = kernel_dir / "ncrisc";
    fs::remove_all(kernel_dir.parent_path());
    fs::create_directories(target_dir);

    const std::unordered_map<std::string, std::uint32_t> named_args = {{"cb_in0", 7}, {"num_tiles", 64}};
    {
        std::ofstream header(kernel_dir / NAMED_CT_ARG_MAP_HEADER);
        header << format_named_ct_arg_map_header(named_args);
        ASSERT_TRUE(header.good());
    }

    // Mirrors hw/inc/api/compile_time_args.h: build the lookup table from the macro and resolve a
    // name at compile time, so a missing or malformed map fails the compile.
    const fs::path src = kernel_dir / "consumer.cpp";
    {
        std::ofstream f(src);
        f << "#include <string_view>\n#include <utility>\n#include <cstdint>\n"
             "constexpr std::pair<std::string_view, uint32_t> named_args_map[] = {KERNEL_COMPILE_TIME_ARG_MAP};\n"
             "constexpr uint32_t get_named_ct_arg(std::string_view name) {\n"
             "    for (const auto& [n, v] : named_args_map) { if (n == name) { return v; } }\n"
             "    return 0xFFFFFFFFu;\n"
             "}\n"
             "static_assert(get_named_ct_arg(\"cb_in0\") == 7);\n"
             "static_assert(get_named_ct_arg(\"num_tiles\") == 64);\n";
        ASSERT_TRUE(f.good());
    }

    // -I. / -I.. and cwd = target dir replicate JitBuildEnv's include list and exec_command's
    // working directory. The bare -include must resolve through -I.. alone.
    const std::vector<std::string> args = {
        "c++",
        "-std=c++17",
        "-fsyntax-only",
        "-I.",
        "-I..",
        "-include",
        std::string(NAMED_CT_ARG_MAP_HEADER),
        "../consumer.cpp"};
    if (!exec_command(args, target_dir.string(), (target_dir / "compile.log").string())) {
        std::ifstream log(target_dir / "compile.log");
        const std::string output((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
        // A host compiler is not guaranteed at test runtime; a missing one is not a product failure.
        if (output.find("c++") != std::string::npos && output.find("not found") != std::string::npos) {
            GTEST_SKIP() << "no host c++ compiler available";
        }
        FAIL() << "force-included map header failed to compile:\n" << output;
    }

    fs::remove_all(kernel_dir.parent_path());
}

}  // namespace
}  // namespace tt::jit_build::utils
