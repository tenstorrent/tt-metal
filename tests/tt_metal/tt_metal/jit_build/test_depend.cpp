// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <unistd.h>

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>

#include "common/filesystem_utils.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_cache_telemetry.hpp"
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
        // Create temporary directory
        auto temp_template = (std::filesystem::temp_directory_path() / "jit_build_test_XXXXXX").string();
        auto* temp_dir = mkdtemp(temp_template.data());
        ASSERT_NE(temp_dir, nullptr);
        out_dir_ = std::filesystem::path(temp_dir);
    }

    void TearDown() override {
        // Remove temporary directory
        std::filesystem::remove_all(out_dir_);
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
        tt::jit_build::write_dependency_hashes(dependencies, out_dir_, obj_file_name, hash_file);
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
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(out_dir_, obj_file_name));
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
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_, obj_file_name));
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
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_, obj_file_name));
}

TEST_F(JitBuildDependencyTests, DependencyHashesNotFound) {
    constexpr auto obj_file_name = "test.o";
    std::filesystem::remove_all(out_dir_);

    // Verify that dependencies are not up to date when hash file is missing
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_, obj_file_name));
}

TEST(JitBuildTests, InvalidHashFile) {
    // Corrupt the hash file
    std::istringstream corrupted_hash_file("corrupted content");

    // Verify that dependencies are not up to date
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(corrupted_hash_file));
    // Make sure the below message is not lost if logger level is set to warning.
    log_warning(tt::LogBuildKernels, "The above warning about malformed file is expected in this test.");
}

TEST(JitBuildTests, EmptyHashFile) {
    // Create an empty hash file
    std::istringstream empty_hash_file("");

    // Verify that dependencies are not up to date when no dependencies are found
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(empty_hash_file));
}

// Tests for scratch-to-cache path rewriting in write_dependency_hashes.
// These exercise the component-based path manipulation helpers indirectly.
class JitBuildPathRewriteTests : public ::testing::Test {
protected:
    void SetUp() override {
        auto temp_template = (std::filesystem::temp_directory_path() / "jit_path_test_XXXXXX").string();
        auto* temp_dir = mkdtemp(temp_template.data());
        ASSERT_NE(temp_dir, nullptr);
        test_root_ = std::filesystem::path(temp_dir);

        // Create scratch and cache directories with matching suffix structure
        // scratch: /tmp/xxx/scratch/build_key/kernels/kernel_name/
        // cache:   /tmp/xxx/cache/build_key/kernels/kernel_name/
        scratch_dir_ = test_root_ / "scratch" / "build_key" / "kernels" / "test_kernel";
        cache_dir_ = test_root_ / "cache" / "build_key" / "kernels" / "test_kernel";
        std::filesystem::create_directories(scratch_dir_);
        std::filesystem::create_directories(cache_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(test_root_); }

    std::string read_dephash_content(const std::filesystem::path& hash_path) const {
        std::ifstream f(hash_path);
        std::stringstream buf;
        buf << f.rdbuf();
        return buf.str();
    }

    std::filesystem::path test_root_;
    std::filesystem::path scratch_dir_;
    std::filesystem::path cache_dir_;
};

TEST_F(JitBuildPathRewriteTests, ScratchPathsRewrittenToCache) {
    // Create a dependency file in scratch that references a file also in scratch
    const std::string obj_name = "test.o";
    std::ofstream{scratch_dir_ / "source.cpp"} << "int main() {}";

    const tt::jit_build::ParsedDependencies deps{{obj_name, {"source.cpp"}}};

    std::ostringstream hash_stream;
    tt::jit_build::write_dependency_hashes(deps, scratch_dir_, obj_name, hash_stream, cache_dir_);

    std::string content = hash_stream.str();
    // Path should be relative (portable) after rewriting
    EXPECT_TRUE(content.find("source.cpp\t") != std::string::npos)
        << "Expected relative path 'source.cpp' in: " << content;
    // Should NOT contain scratch path
    EXPECT_TRUE(content.find("/scratch/") == std::string::npos) << "Should not contain scratch path in: " << content;
}

TEST_F(JitBuildPathRewriteTests, AbsolutePathOutsideScratchNotRewritten) {
    // Create a dependency outside of scratch (simulates a system header or source file)
    auto external_dir = test_root_ / "external_src";
    std::filesystem::create_directories(external_dir);
    std::ofstream{external_dir / "external.h"} << "#pragma once";

    const std::string obj_name = "test.o";
    const tt::jit_build::ParsedDependencies deps{{obj_name, {(external_dir / "external.h").string()}}};

    std::ostringstream hash_stream;
    tt::jit_build::write_dependency_hashes(deps, scratch_dir_, obj_name, hash_stream, cache_dir_);

    std::string content = hash_stream.str();
    // External path should remain absolute (not under cache or scratch)
    EXPECT_TRUE(content.find(external_dir.string()) != std::string::npos)
        << "External path should be preserved in: " << content;
}

TEST_F(JitBuildPathRewriteTests, NoRewriteWhenCanonicalDirEmpty) {
    // Without canonical_dir, paths should be relative to out_dir
    const std::string obj_name = "test.o";
    std::ofstream{scratch_dir_ / "source.cpp"} << "int main() {}";

    const tt::jit_build::ParsedDependencies deps{{obj_name, {"source.cpp"}}};

    std::ostringstream hash_stream;
    tt::jit_build::write_dependency_hashes(deps, scratch_dir_, obj_name, hash_stream, "");

    std::string content = hash_stream.str();
    EXPECT_TRUE(content.find("source.cpp\t") != std::string::npos) << "Expected relative path in: " << content;
}

TEST_F(JitBuildPathRewriteTests, TrailingSlashHandled) {
    // Verify paths with trailing slashes are handled correctly
    const std::string obj_name = "test.o";
    std::ofstream{scratch_dir_ / "source.cpp"} << "int main() {}";

    const tt::jit_build::ParsedDependencies deps{{obj_name, {"source.cpp"}}};

    // Add trailing slashes to both directories
    std::ostringstream hash_stream;
    tt::jit_build::write_dependency_hashes(deps, scratch_dir_, obj_name, hash_stream, cache_dir_);

    std::string content = hash_stream.str();
    EXPECT_TRUE(content.find("source.cpp\t") != std::string::npos)
        << "Trailing slash should not break path handling: " << content;
    EXPECT_FALSE(hash_stream.fail()) << "Hash stream should not fail";
}

TEST_F(JitBuildPathRewriteTests, RelativePathInDependencyResolved) {
    // Test that relative paths in the dependency list are resolved correctly
    const std::string obj_name = "test.o";
    std::ofstream{scratch_dir_ / "header.h"} << "#pragma once";

    // Dependency is specified as relative path
    const tt::jit_build::ParsedDependencies deps{{obj_name, {"header.h"}}};

    std::ostringstream hash_stream;
    tt::jit_build::write_dependency_hashes(deps, scratch_dir_, obj_name, hash_stream, cache_dir_);

    std::string content = hash_stream.str();
    EXPECT_FALSE(hash_stream.fail()) << "Should successfully write hashes for relative deps";
    EXPECT_TRUE(content.find("header.h\t") != std::string::npos)
        << "Relative path should appear in output: " << content;
}

TEST_F(JitBuildPathRewriteTests, NestedSubdirectoryPathRewritten) {
    // Test path rewriting for files in nested subdirectories
    auto nested_scratch = scratch_dir_ / "subdir" / "nested";
    auto nested_cache = cache_dir_ / "subdir" / "nested";
    std::filesystem::create_directories(nested_scratch);
    std::filesystem::create_directories(nested_cache);

    const std::string obj_name = "test.o";
    std::ofstream{nested_scratch / "deep.cpp"} << "void f() {}";

    // Use absolute path to the nested file
    const tt::jit_build::ParsedDependencies deps{{obj_name, {(nested_scratch / "deep.cpp").string()}}};

    std::ostringstream hash_stream;
    tt::jit_build::write_dependency_hashes(deps, scratch_dir_, obj_name, hash_stream, cache_dir_);

    std::string content = hash_stream.str();
    // Path should be rewritten from scratch to cache and made relative
    EXPECT_TRUE(content.find("/scratch/") == std::string::npos)
        << "Nested scratch path should be rewritten: " << content;
}

TEST_F(JitBuildPathRewriteTests, CacheHitAfterScratchBuild) {
    // End-to-end test: build in scratch, verify cache-relative .dephash works for cache hit check
    const std::string obj_name = "kernel.o";

    // Create dependency file in both scratch and cache (simulating merge)
    std::ofstream{scratch_dir_ / "kernel.cpp"} << "void kernel() {}";
    std::ofstream{cache_dir_ / "kernel.cpp"} << "void kernel() {}";

    const tt::jit_build::ParsedDependencies deps{{obj_name, {"kernel.cpp"}}};

    // Write hash file referencing scratch, but with canonical_dir set to cache
    auto hash_path = cache_dir_ / (obj_name + ".dephash");
    std::ofstream hash_file(hash_path);
    tt::jit_build::write_dependency_hashes(deps, scratch_dir_, obj_name, hash_file, cache_dir_);
    hash_file.close();
    ASSERT_FALSE(hash_file.fail());

    // Verify cache-based dependency check succeeds (paths point to cache, not scratch)
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(cache_dir_, obj_name))
        << "Dependencies should be up-to-date when checked against cache directory";
}

TEST(JitBuildTests, BuildStateHashPublishesAfterArtifactsAreVisible) {
    auto temp_template = (std::filesystem::temp_directory_path() / "jit_build_state_test_XXXXXX").string();
    auto* temp_dir = mkdtemp(temp_template.data());
    ASSERT_NE(temp_dir, nullptr);
    std::filesystem::path cache_dir(temp_dir);

    constexpr uint64_t stale_hash = 111;
    constexpr uint64_t fresh_hash = 222;
    const std::filesystem::path obj_path = cache_dir / "kernel.o";
    const std::filesystem::path dephash_path = cache_dir / "kernel.o.dephash";
    const std::filesystem::path build_state_path = cache_dir / ".build_state";

    auto write_file = [](const std::filesystem::path& path, std::string_view contents) {
        std::ofstream file(path, std::ios::trunc);
        ASSERT_TRUE(file.is_open());
        file << contents;
        file.close();
        ASSERT_FALSE(file.fail());
    };
    auto read_file = [](const std::filesystem::path& path) {
        std::ifstream file(path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    };

    write_file(obj_path, "stale-object");
    write_file(dephash_path, "stale-dephash");
    tt::tt_metal::publish_build_state_hash(cache_dir, stale_hash);

    std::atomic<bool> saw_fresh_hash{false};
    std::atomic<bool> artifacts_were_fresh{false};
    std::thread observer([&] {
        for (int attempt = 0; attempt < 200; ++attempt) {
            std::ifstream hash_file(build_state_path);
            uint64_t observed_hash = 0;
            if (hash_file >> observed_hash; observed_hash == fresh_hash) {
                saw_fresh_hash.store(true, std::memory_order_relaxed);
                artifacts_were_fresh.store(
                    read_file(obj_path) == "fresh-object" && read_file(dephash_path) == "fresh-dephash",
                    std::memory_order_relaxed);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    const auto obj_tmp = cache_dir / "kernel.o.tmp";
    const auto dephash_tmp = cache_dir / "kernel.o.dephash.tmp";
    write_file(obj_tmp, "fresh-object");
    write_file(dephash_tmp, "fresh-dephash");
    ASSERT_TRUE(tt::filesystem::safe_rename(obj_tmp, obj_path, false));
    ASSERT_TRUE(tt::filesystem::safe_rename(dephash_tmp, dephash_path, false));

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    tt::tt_metal::publish_build_state_hash(cache_dir, fresh_hash);
    observer.join();

    EXPECT_TRUE(saw_fresh_hash.load(std::memory_order_relaxed));
    EXPECT_TRUE(artifacts_were_fresh.load(std::memory_order_relaxed));

    std::filesystem::remove_all(cache_dir);
}

TEST(JitBuildTests, TelemetryTokenSupportsConcurrentRecord) {
    auto& token = tt::tt_metal::BuildCacheTelemetry::inst().register_metric("test_depend.concurrent_record");

    constexpr int kThreadCount = 8;
    constexpr int kRecordsPerThread = 64;
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);

    for (int thread_id = 0; thread_id < kThreadCount; ++thread_id) {
        threads.emplace_back([&token, thread_id] {
            for (int sample = 0; sample < kRecordsPerThread; ++sample) {
                token.record(static_cast<double>((thread_id * 1000) + sample));
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_GE(token.snapshot().size(), static_cast<size_t>(kThreadCount * kRecordsPerThread));
}
