// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>

#include "jit_build/depend.hpp"
#include "jit_build/pch.hpp"
#include "jit_build/jit_build_utils.hpp"

TEST(JitBuildUtils, TemporaryPathsAreUniqueAcrossThreads) {
    constexpr size_t num_threads = 8;
    constexpr size_t paths_per_thread = 32;
    const std::filesystem::path target = "kernel.o";
    std::vector<std::string> paths(num_threads * paths_per_thread);
    std::vector<std::thread> threads;
    for (size_t thread = 0; thread < num_threads; ++thread) {
        threads.emplace_back([&, thread] {
            for (size_t path = 0; path < paths_per_thread; ++path) {
                paths[thread * paths_per_thread + path] = tt::jit_build::utils::FileRenamer::generate_temp_path(target);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    std::ranges::sort(paths);
    EXPECT_EQ(std::ranges::adjacent_find(paths), paths.end());
}

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

TEST_F(JitBuildDependencyTests, SingleNormalizedTargetIsAccepted) {
    constexpr auto normalized_target = "normalized-output.o";
    const auto obj_path = out_dir_ / normalized_target;
    const tt::jit_build::ParsedDependencies dependencies{{normalized_target, {"a.txt", "b.txt"}}};
    create_dependency_files(dependencies, normalized_target);

    std::ofstream hash_file{obj_path.string() + ".dephash"};
    tt::jit_build::write_dependency_hashes(dependencies, out_dir_.string(), obj_path.string(), hash_file);
    hash_file.close();

    ASSERT_FALSE(hash_file.fail());
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), normalized_target));
}

TEST_F(JitBuildDependencyTests, SingleUnrelatedTargetIsRejected) {
    const tt::jit_build::ParsedDependencies dependencies{{"unrelated.o", {}}};
    std::ofstream hash_file{out_dir_ / "expected.o.dephash"};
    tt::jit_build::write_dependency_hashes(
        dependencies, out_dir_.string(), (out_dir_ / "expected.o").string(), hash_file);
    EXPECT_TRUE(hash_file.fail());
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

TEST_F(JitBuildDependencyTests, MissingPchUmbrellaFailsBuild) {
    const auto umbrella = out_dir_ / "missing_pch.h";
    EXPECT_THROW(tt::jit_build::ensure_pch("", "Os", "", umbrella, out_dir_ / "pch"), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(out_dir_ / "pch"));
}

TEST_F(JitBuildDependencyTests, UnreadablePchUmbrellaFailsBuild) {
    if (::geteuid() == 0) {
        GTEST_SKIP() << "Root can read files regardless of permission bits";
    }
    const auto umbrella = out_dir_ / "pch.h";
    std::ofstream{umbrella} << "#include <array>\n";
    std::filesystem::permissions(umbrella, std::filesystem::perms::none);
    EXPECT_THROW(tt::jit_build::ensure_pch("", "Os", "", umbrella, out_dir_ / "pch"), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(out_dir_ / "pch"));
}

TEST_F(JitBuildDependencyTests, ExplicitPchDependency) {
    const std::string obj = "test.o";
    const std::string hash_path = (out_dir_ / (obj + ".dephash")).string();
    const std::string umbrella = (out_dir_ / "pch.h").string();
    std::ofstream{out_dir_ / "test.cpp"} << "int value;\n";
    std::ofstream{umbrella} << "#include <array>\n";
    // GCC can omit the umbrella from the consuming object's dependency file.
    std::ofstream{out_dir_ / "test.d"} << "test.o: test.cpp\n";

    tt::jit_build::write_dependency_hashes(out_dir_.string(), obj, hash_path);
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj));
    // Existing records without the umbrella remain reusable when it changes.
    std::ofstream{umbrella} << "#include <array>\n#include <tuple>\n";
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj));

    tt::jit_build::write_dependency_hashes(out_dir_.string(), obj, hash_path, umbrella);
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj));
    std::ofstream{umbrella} << "#include <array>\n#include <tuple>\n#include <utility>\n";
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj));

    tt::jit_build::write_dependency_hashes(out_dir_.string(), obj, hash_path, umbrella);
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj));
    std::filesystem::remove(umbrella);
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj));
}

TEST_F(JitBuildDependencyTests, NormalizedTargetRetainsExplicitPchDependency) {
    const auto obj = (out_dir_ / "test.o").string();
    const auto hash_path = obj + ".dephash";
    const auto umbrella = (out_dir_ / "pch.h").string();
    std::ofstream{out_dir_ / "test.cpp"} << "int value;\n";
    std::ofstream{umbrella} << "#include <array>\n";
    std::ofstream{out_dir_ / "test.d"} << "test.o: test.cpp\n";

    tt::jit_build::write_dependency_hashes(out_dir_.string(), obj, hash_path, umbrella);
    ASSERT_TRUE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj));
    std::ofstream{umbrella} << "#include <array>\n#include <tuple>\n";
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj));

    tt::jit_build::write_dependency_hashes(out_dir_.string(), obj, hash_path, umbrella);
    ASSERT_TRUE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj));
    std::filesystem::remove(umbrella);
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj));
}

TEST_F(JitBuildDependencyTests, MultipleNormalizedTargetsRejectExplicitPchDependency) {
    const auto obj = (out_dir_ / "test.o").string();
    const auto hash_path = obj + ".dephash";
    const auto umbrella = (out_dir_ / "pch.h").string();
    std::ofstream{out_dir_ / "test.cpp"} << "int value;\n";
    std::ofstream{umbrella} << "#include <array>\n";
    std::ofstream{out_dir_ / "test.d"} << "test.o: test.cpp\nother.o: test.cpp\n";

    tt::jit_build::write_dependency_hashes(out_dir_.string(), obj, hash_path, umbrella);
    EXPECT_FALSE(std::filesystem::exists(hash_path));
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

    std::vector<std::string> dep_names;
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

TEST_F(JitBuildDependencyTests, ConcurrentInvalidation) {
    constexpr auto obj_file_name = "test.o";
    constexpr int kNumFiles = 20;
    constexpr int kNumThreads = 16;

    std::vector<std::string> dep_names;
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
