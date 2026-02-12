// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <sstream>
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

// ---- DependencyCache tests ----

class DependencyCacheTests : public ::testing::Test {
protected:
    void SetUp() override {
        auto temp_template = (std::filesystem::temp_directory_path() / "dep_cache_test_XXXXXX").string();
        auto* temp_dir = mkdtemp(temp_template.data());
        ASSERT_NE(temp_dir, nullptr);
        out_dir_ = std::filesystem::path(temp_dir) / "";
        std::filesystem::create_directories(out_dir_);
        dephash_path_ = (out_dir_ / "dephash").string();
    }

    void TearDown() override { std::filesystem::remove_all(out_dir_); }

    // Create a file with given content
    void create_file(const std::string& name, const std::string& content) const {
        std::ofstream{out_dir_ / name} << content;
    }

    // Create an absolute-path file with given content
    void create_abs_file(const std::string& abs_path, const std::string& content) const {
        std::ofstream{abs_path} << content;
    }

    std::string abs(const std::string& name) const { return (out_dir_ / name).string(); }

    std::filesystem::path out_dir_;
    std::string dephash_path_;
};

TEST_F(DependencyCacheTests, ConstructorEmptyFile) {
    // No dephash file => cache has no entries, everything stale
    std::vector<std::string> targets = {"brisc.o", "ncrisc.o", "brisc.elf"};
    tt::jit_build::DependencyCache cache(dephash_path_, targets);
    EXPECT_EQ(cache.targets(), targets);
    EXPECT_TRUE(cache.entries().empty());
}

TEST_F(DependencyCacheTests, ConstructorLoadsFormat) {
    // Write a dephash file in the expected format
    create_file("header.h", "header content");
    {
        std::ofstream out(dephash_path_);
        out << abs("header.h") << "\t12345\tbrisc.o\tncrisc.o\n";
    }

    std::vector<std::string> targets = {"brisc.o", "ncrisc.o", "target.elf"};
    tt::jit_build::DependencyCache cache(dephash_path_, targets);

    ASSERT_EQ(cache.entries().size(), 1u);
    EXPECT_EQ(cache.entries()[0].dep_path, abs("header.h"));
    EXPECT_EQ(cache.entries()[0].hash, 12345u);
    // Both brisc.o (idx 0) and ncrisc.o (idx 1) should be set
    EXPECT_TRUE(cache.entries()[0].target_mask.test(0));
    EXPECT_TRUE(cache.entries()[0].target_mask.test(1));
    EXPECT_FALSE(cache.entries()[0].target_mask.test(2));
}

TEST_F(DependencyCacheTests, ConstructorDropsUnknownTargets) {
    create_file("header.h", "content");
    {
        std::ofstream out(dephash_path_);
        // "old_target.o" is not in current_targets, should be dropped
        out << abs("header.h") << "\t12345\told_target.o\n";
    }

    std::vector<std::string> targets = {"brisc.o"};
    tt::jit_build::DependencyCache cache(dephash_path_, targets);

    // Entry should be dropped entirely since no bits are set
    EXPECT_TRUE(cache.entries().empty());
}

TEST_F(DependencyCacheTests, FindStaleTargetsAllUpToDate) {
    create_file("header.h", "header content");
    create_file("brisc.o", "object");
    create_file("ncrisc.o", "object");

    // First, hash the file to get its actual hash
    std::vector<std::string> targets = {"brisc.o", "ncrisc.o"};

    // Build a cache with correct hashes via the update+write cycle
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        // All stale (no dephash), so update with deps
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_TRUE(stale.test(0));
        EXPECT_TRUE(stale.test(1));

        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("header.h")};
        new_deps["ncrisc.o"] = {abs("header.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        recompiled.set(1);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // Now reload and check: nothing should be stale
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_FALSE(stale.test(0));
        EXPECT_FALSE(stale.test(1));
    }
}

TEST_F(DependencyCacheTests, FindStaleTargetsAfterModification) {
    create_file("header.h", "original content");
    create_file("brisc.o", "object");
    create_file("ncrisc.o", "object");

    std::vector<std::string> targets = {"brisc.o", "ncrisc.o"};

    // Build cache
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("header.h")};
        new_deps["ncrisc.o"] = {abs("header.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        recompiled.set(1);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // Modify the header
    create_file("header.h", "modified content");

    // Both targets should now be stale
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_TRUE(stale.test(0));
        EXPECT_TRUE(stale.test(1));
    }
}

TEST_F(DependencyCacheTests, FindStaleTargetsAfterDeletion) {
    create_file("header.h", "content");
    create_file("brisc.o", "object");

    std::vector<std::string> targets = {"brisc.o"};

    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("header.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // Delete the dependency file
    std::filesystem::remove(out_dir_ / "header.h");

    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_TRUE(stale.test(0));
    }
}

TEST_F(DependencyCacheTests, FindStaleWhenOneOfTwoDepsChanges) {
    create_file("a.h", "aaa");
    create_file("b.h", "bbb");
    create_file("brisc.o", "object");

    std::vector<std::string> targets = {"brisc.o"};

    // Build cache: brisc.o depends on both a.h and b.h
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("a.h"), abs("b.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // Modify only b.h -- brisc.o should still be stale because one dep changed
    create_file("b.h", "bbb modified");

    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_TRUE(stale.test(0));
    }
}

TEST_F(DependencyCacheTests, FindStaleTargetsNewTarget) {
    create_file("header.h", "content");
    create_file("brisc.o", "object");

    // Build cache with just brisc.o
    {
        std::vector<std::string> targets = {"brisc.o"};
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("header.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // Now add a new target
    {
        std::vector<std::string> targets = {"brisc.o", "ncrisc.o"};
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        auto stale = cache.find_stale_targets(out_dir_.string());
        // brisc.o should be up to date
        EXPECT_FALSE(stale.test(0));
        // ncrisc.o has no deps, should be stale
        EXPECT_TRUE(stale.test(1));
    }
}

TEST_F(DependencyCacheTests, FindStaleTargetsMissingOutput) {
    create_file("header.h", "content");
    create_file("brisc.o", "object");

    std::vector<std::string> targets = {"brisc.o"};

    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("header.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // Delete the output file
    std::filesystem::remove(out_dir_ / "brisc.o");

    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_TRUE(stale.test(0));
    }
}

TEST_F(DependencyCacheTests, UpdateMergesCachedAndRecompiled) {
    create_file("common.h", "common");
    create_file("brisc_only.h", "brisc");
    create_file("ncrisc_only.h", "ncrisc");
    create_file("brisc.o", "object");
    create_file("ncrisc.o", "object");

    std::vector<std::string> targets = {"brisc.o", "ncrisc.o"};

    // Initial build: both targets compiled
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("common.h"), abs("brisc_only.h")};
        new_deps["ncrisc.o"] = {abs("common.h"), abs("ncrisc_only.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        recompiled.set(1);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // Modify brisc_only.h => only brisc.o should be stale
    create_file("brisc_only.h", "brisc modified");

    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_TRUE(stale.test(0));   // brisc.o stale
        EXPECT_FALSE(stale.test(1));  // ncrisc.o cached

        // Recompile only brisc.o
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("common.h"), abs("brisc_only.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // Verify: both should now be up to date
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_FALSE(stale.test(0));
        EXPECT_FALSE(stale.test(1));
    }
}

TEST_F(DependencyCacheTests, WriteFormat) {
    create_file("a.h", "aaa");
    create_file("b.h", "bbb");

    std::vector<std::string> targets = {"brisc.o", "ncrisc.o"};
    tt::jit_build::DependencyCache cache(dephash_path_, targets);

    tt::jit_build::ParsedDependencies new_deps;
    new_deps["brisc.o"] = {abs("a.h"), abs("b.h")};
    new_deps["ncrisc.o"] = {abs("a.h")};

    tt::jit_build::DependencyCache::TargetMask recompiled;
    recompiled.set(0);
    recompiled.set(1);
    cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);

    // Read back and verify format
    std::ifstream in(dephash_path_);
    ASSERT_TRUE(in.is_open());

    std::string line;
    size_t line_count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        ++line_count;
        // Each line should have at least 3 tab-separated fields
        size_t tab1 = line.find('\t');
        ASSERT_NE(tab1, std::string::npos) << "Line missing first tab: " << line;
        size_t tab2 = line.find('\t', tab1 + 1);
        ASSERT_NE(tab2, std::string::npos) << "Line missing second tab: " << line;
        // After tab2 should be target names
        std::string targets_str = line.substr(tab2 + 1);
        EXPECT_FALSE(targets_str.empty()) << "Line has no targets: " << line;
    }
    EXPECT_EQ(line_count, 2u);  // a.h and b.h
}

TEST_F(DependencyCacheTests, RoundTrip) {
    create_file("x.h", "xxx");
    create_file("y.h", "yyy");
    create_file("obj1.o", "o1");
    create_file("obj2.o", "o2");
    create_file("target.elf", "elf");

    std::vector<std::string> targets = {"obj1.o", "obj2.o", "target.elf"};

    // Build and write
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["obj1.o"] = {abs("x.h"), abs("y.h")};
        new_deps["obj2.o"] = {abs("y.h")};
        new_deps["target.elf"] = {abs("x.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        recompiled.set(1);
        recompiled.set(2);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // Load and check nothing stale
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_FALSE(stale.test(0));
        EXPECT_FALSE(stale.test(1));
        EXPECT_FALSE(stale.test(2));
    }
}

TEST_F(DependencyCacheTests, NoDephashFileAllStale) {
    create_file("brisc.o", "object");

    std::vector<std::string> targets = {"brisc.o"};
    tt::jit_build::DependencyCache cache(dephash_path_, targets);
    auto stale = cache.find_stale_targets(out_dir_.string());
    EXPECT_TRUE(stale.test(0));
}

TEST_F(DependencyCacheTests, MalformedDephashAllStale) {
    create_file("header.h", "content");
    create_file("brisc.o", "object");
    create_file("ncrisc.o", "object");

    std::vector<std::string> targets = {"brisc.o", "ncrisc.o"};

    // Build a valid cache first
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("header.h")};
        new_deps["ncrisc.o"] = {abs("header.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        recompiled.set(1);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // Corrupt the dephash file (line without tabs)
    {
        std::ofstream out(dephash_path_);
        out << "corrupted content without tabs\n";
    }

    // Everything should be stale because parsing failed
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        EXPECT_TRUE(cache.entries().empty());
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_TRUE(stale.test(0));
        EXPECT_TRUE(stale.test(1));
    }
}

TEST_F(DependencyCacheTests, MalformedHashFieldAllStale) {
    // Write a dephash file with a valid first line and a bad hash on the second line
    {
        std::ofstream out(dephash_path_);
        out << abs("header.h") << "\t12345\tbrisc.o\n";
        out << abs("other.h") << "\tNOT_A_NUMBER\tncrisc.o\n";
    }

    create_file("header.h", "content");
    create_file("other.h", "content");

    std::vector<std::string> targets = {"brisc.o", "ncrisc.o"};
    tt::jit_build::DependencyCache cache(dephash_path_, targets);

    // The malformed second line should discard the entire cache
    EXPECT_TRUE(cache.entries().empty());
    auto stale = cache.find_stale_targets(out_dir_.string());
    EXPECT_TRUE(stale.test(0));
    EXPECT_TRUE(stale.test(1));
}

TEST_F(DependencyCacheTests, WriteUpdatedDeletesDephashOnUnhashableDep) {
    create_file("header.h", "content");
    create_file("brisc.o", "object");

    std::vector<std::string> targets = {"brisc.o"};

    // Build a valid cache
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("header.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }
    ASSERT_TRUE(std::filesystem::exists(dephash_path_));

    // Now try to write_updated with a dep that doesn't exist
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        tt::jit_build::ParsedDependencies new_deps;
        new_deps["brisc.o"] = {abs("nonexistent.h")};

        tt::jit_build::DependencyCache::TargetMask recompiled;
        recompiled.set(0);
        cache.write_updated(dephash_path_, recompiled, out_dir_.string(), new_deps);
    }

    // The dephash file should have been deleted
    EXPECT_FALSE(std::filesystem::exists(dephash_path_));

    // Next load should see everything stale
    {
        tt::jit_build::DependencyCache cache(dephash_path_, targets);
        auto stale = cache.find_stale_targets(out_dir_.string());
        EXPECT_TRUE(stale.test(0));
    }
}
