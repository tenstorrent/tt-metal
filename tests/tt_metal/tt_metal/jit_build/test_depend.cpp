// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include <unistd.h>

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>

#include "jit_build/depend.hpp"

TEST(JitBuildTests, ParseDependencyFile) {
    constexpr auto dep_file_content = R"(
main.o: main.cpp utils.h
utils.o: \
 utils.cpp \
 utils.h utils_internal.h
)";
    const tt::jit_build::ParsedDependencies expected{
        {"main.o", {"main.cpp", "utils.h"}},
        {"utils.o", {"utils.cpp", "utils.h", "utils_internal.h"}},
    };
    std::istringstream dep_file(dep_file_content);
    auto dependencies = tt::jit_build::parse_dependency_file(dep_file);
    ASSERT_EQ(dependencies, expected);
}

class JitBuildDependencyTests : public ::testing::Test {
protected:
    void SetUp() override {
        tt::jit_build::clear_file_hash_cache();
        // Create temporary directory
        auto temp_template = (std::filesystem::temp_directory_path() / "jit_build_test_XXXXXX").string();
        auto* temp_dir = mkdtemp(temp_template.data());
        ASSERT_NE(temp_dir, nullptr);
        out_dir_ = std::filesystem::path(temp_dir);
    }

    void TearDown() override {
        // Remove temporary directory
        std::filesystem::remove_all(out_dir_);
        tt::jit_build::clear_file_hash_cache();
    }

    void create_dependency_files(
        const tt::jit_build::ParsedDependencies& dependencies, const std::string& obj_file_name) const {
        const auto& deps = dependencies.at(obj_file_name);
        for (const auto& dep : deps) {
            std::ofstream{out_dir_ / dep} << "Content of " << dep;
        }
    }

    void create_hash(const tt::jit_build::ParsedDependencies& dependencies, const std::string& obj_file_name) const {
        std::ofstream hash_file{(out_dir_ / (obj_file_name + ".dephash"))};
        tt::jit_build::write_dependency_hashes(dependencies, out_dir_.string(), obj_file_name, hash_file);
        hash_file.close();
        ASSERT_FALSE(hash_file.fail());
    }

    void create_dependency_files_and_hash(
        const tt::jit_build::ParsedDependencies& dependencies, const std::string& obj_file_name) const {
        create_dependency_files(dependencies, obj_file_name);
        create_hash(dependencies, obj_file_name);
    }

    std::filesystem::path out_dir_;
};

TEST_F(JitBuildDependencyTests, UpToDate) {
    constexpr auto obj_file_name = "test.o";
    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, {"a.txt", "b.txt", "c.txt"}},
    };
    create_dependency_files_and_hash(dependencies, obj_file_name);

    // Verify that dependencies are up to date
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj_file_name));
}

TEST_F(JitBuildDependencyTests, OutOfDateAfterModification) {
    constexpr auto obj_file_name = "test.o";
    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, {"a.txt", "b.txt", "c.txt"}},
    };
    create_dependency_files_and_hash(dependencies, obj_file_name);

    // Modify one dependency
    std::ofstream{out_dir_ / "b.txt"} << "Modified content of b.txt";

    // Verify that dependencies are not up to date
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj_file_name));
}

TEST_F(JitBuildDependencyTests, OutOfDateAfterDeletion) {
    constexpr auto obj_file_name = "test.o";
    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, {"a.txt", "b.txt", "c.txt"}},
    };
    create_dependency_files_and_hash(dependencies, obj_file_name);

    // Delete one dependency
    std::filesystem::remove(out_dir_ / "c.txt");

    // Verify that dependencies are not up to date
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj_file_name));
}

TEST_F(JitBuildDependencyTests, DependencyHashesNotFound) {
    constexpr auto obj_file_name = "test.o";
    std::filesystem::remove_all(out_dir_);

    // Verify that dependencies are not up to date when hash file is missing
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj_file_name));
}

TEST(JitBuildTests, InvalidHashFile) {
    tt::jit_build::clear_file_hash_cache();
    // Corrupt the hash file
    std::istringstream corrupted_hash_file("corrupted content");

    // Verify that dependencies are not up to date
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(corrupted_hash_file));
    // Make sure the below message is not lost if logger level is set to warning.
    log_warning(tt::LogBuildKernels, "The above warning about malformed file is expected in this test.");
}

TEST(JitBuildTests, EmptyHashFile) {
    tt::jit_build::clear_file_hash_cache();
    // Create an empty hash file
    std::istringstream empty_hash_file("");

    // Verify that dependencies are not up to date when no dependencies are found
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(empty_hash_file));
}

TEST_F(JitBuildDependencyTests, ConcurrentUpToDateCheck) {
    constexpr auto obj_file_name = "test.o";
    constexpr int kNumFiles = 20;
    constexpr int kNumThreads = 16;

    std::vector<std::filesystem::path> dep_names;
    dep_names.reserve(kNumFiles);
    for (int i = 0; i < kNumFiles; ++i) {
        dep_names.push_back("dep_" + std::to_string(i) + ".txt");
    }
    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, dep_names},
    };
    create_dependency_files_and_hash(dependencies, obj_file_name);

    // All threads should see "up to date" for the same dephash.
    // Use int instead of bool to avoid std::vector<bool> bit-packing data race.
    std::vector<int> results(kNumThreads, 0);
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back(
            [&, t] { results[t] = tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj_file_name); });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (int t = 0; t < kNumThreads; ++t) {
        EXPECT_TRUE(results[t]) << "Thread " << t << " did not see up-to-date";
    }
}

// Validates that an absolute-path dependency round-trips correctly through
// write_dependency_hashes → dependencies_up_to_date (stream overloads).
TEST_F(JitBuildDependencyTests, AbsolutePathRoundTrip) {
    constexpr auto obj_file_name = "abs_test.o";

    // Create a file with an absolute path inside out_dir_
    auto abs_dep = out_dir_ / "absolute_header.h";
    std::ofstream{abs_dep} << "// absolute header content";

    // Build ParsedDependencies using the absolute path directly
    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, {abs_dep}},
    };

    // Write hashes to a stringstream
    std::stringstream hash_stream;
    tt::jit_build::write_dependency_hashes(dependencies, out_dir_, obj_file_name, hash_stream);
    ASSERT_FALSE(hash_stream.fail());

    // Read back and verify dependencies are up to date
    tt::jit_build::clear_file_hash_cache();
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(hash_stream));
}

// Validates that a relative-path dependency is stored as an absolute path and
// round-trips correctly through write → read.
TEST_F(JitBuildDependencyTests, RelativePathBecomesAbsoluteRoundTrip) {
    constexpr auto obj_file_name = "rel_test.o";

    // Create a file via a relative name (relative to out_dir_)
    std::filesystem::path rel_dep = "include/relative_dep.h";
    std::filesystem::create_directories(out_dir_ / "include");
    std::ofstream{out_dir_ / rel_dep} << "// relative header content";

    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, {rel_dep}},
    };

    // Write hashes — the writer should convert rel_dep to an absolute path
    std::stringstream hash_stream;
    tt::jit_build::write_dependency_hashes(dependencies, out_dir_, obj_file_name, hash_stream);
    ASSERT_FALSE(hash_stream.fail());

    // Verify the stored path is absolute by parsing it the same way the reader
    // does (operator>> uses std::quoted), rather than doing a raw substring search.
    std::stringstream verify_stream(hash_stream.str());
    std::filesystem::path stored_path;
    verify_stream >> stored_path;
    ASSERT_FALSE(verify_stream.fail()) << "Failed to parse stored path from hash file";
    EXPECT_TRUE(stored_path.is_absolute()) << "Expected absolute path, got: " << stored_path;
    EXPECT_EQ(stored_path, out_dir_ / rel_dep)
        << "Expected " << (out_dir_ / rel_dep) << ", got: " << stored_path;

    // Read back and verify round-trip
    tt::jit_build::clear_file_hash_cache();
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(hash_stream));
}

// Validates that a path containing spaces round-trips correctly.
// This is the scenario where the write/read semantic mismatch (quoted vs
// unquoted) would cause a failure: write uses dep_path.string() (unquoted),
// but read uses operator>>(istream&, path&) which internally uses std::quoted.
// When the first character is NOT a quote, operator>> falls back to
// whitespace-delimited extraction, which splits on the space mid-path.
TEST_F(JitBuildDependencyTests, PathWithSpacesRoundTrip) {
    constexpr auto obj_file_name = "spaces_test.o";

    // Create a directory and file with a space in the name
    auto dir_with_space = out_dir_ / "my headers";
    std::filesystem::create_directories(dir_with_space);
    auto dep_with_space = dir_with_space / "spaced header.h";
    std::ofstream{dep_with_space} << "// header with spaces in path";

    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, {dep_with_space}},
    };

    // Write hashes
    std::stringstream hash_stream;
    tt::jit_build::write_dependency_hashes(dependencies, out_dir_, obj_file_name, hash_stream);
    ASSERT_FALSE(hash_stream.fail());

    // Read back — this will FAIL if write is unquoted but read expects quoted
    tt::jit_build::clear_file_hash_cache();
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(hash_stream))
        << "Path with spaces failed round-trip. Hash file content: " << hash_stream.str();
}

TEST_F(JitBuildDependencyTests, ConcurrentInvalidation) {
    constexpr auto obj_file_name = "test.o";
    constexpr int kNumFiles = 20;
    constexpr int kNumThreads = 16;

    std::vector<std::filesystem::path> dep_names;
    dep_names.reserve(kNumFiles);
    for (int i = 0; i < kNumFiles; ++i) {
        dep_names.push_back("dep_" + std::to_string(i) + ".txt");
    }
    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, dep_names},
    };
    create_dependency_files_and_hash(dependencies, obj_file_name);

    // Warm the cache so entries are in ready state.
    ASSERT_TRUE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj_file_name));

    // Modify a dependency to trigger metadata-based rehash.
    std::ofstream{out_dir_ / dep_names[0]} << "Changed content for invalidation test";

    // All threads should see "out of date" after the modification.
    std::vector<int> results(kNumThreads, 1);
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back(
            [&, t] { results[t] = tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj_file_name); });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (int t = 0; t < kNumThreads; ++t) {
        EXPECT_FALSE(results[t]) << "Thread " << t << " did not detect invalidation";
    }
}
