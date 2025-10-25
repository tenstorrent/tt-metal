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

class JitBuildDependencyTests : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directory
        auto temp_template = (std::filesystem::temp_directory_path() / "jit_build_test_XXXXXX").string();
        auto temp_dir = mkdtemp(&temp_template[0]);
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
        std::ofstream hash_file{(out_dir_ / obj_file_name).replace_extension(".hash")};
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
    std::filesystem::path hash_path = (out_dir_ / obj_file_name).replace_extension(".hash");

    std::filesystem::remove(hash_path);

    // Verify that dependencies are not up to date when hash file is missing
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj_file_name));
}

TEST_F(JitBuildDependencyTests, InvalidHashFile) {
    constexpr auto obj_file_name = "test.o";
    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, {"a.txt", "b.txt"}},
    };
    create_dependency_files_and_hash(dependencies, obj_file_name);

    // Corrupt the hash file
    std::ofstream{(out_dir_ / obj_file_name).replace_extension(".hash")} << "corrupted content";

    // Verify that dependencies are not up to date
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj_file_name));
    // Make sure the below message is not lost if logger level is set to warning or above
    log_warning(tt::LogBuildKernels, "The above warning about cannot read file is expected in this test.");
}

TEST_F(JitBuildDependencyTests, NoDependenciesFound) {
    constexpr auto obj_file_name = "test.o";
    // Create an empty hash file
    std::ofstream{(out_dir_ / obj_file_name).replace_extension(".hash")}.close();

    // Verify that dependencies are not up to date when no dependencies are found
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(out_dir_.string(), obj_file_name));
}
