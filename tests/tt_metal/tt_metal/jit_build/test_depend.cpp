#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>

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

namespace {

void create_dependency_files_and_hash(
    const std::filesystem::path& out_dir,
    const tt::jit_build::ParsedDependencies& dependencies,
    const std::string& obj_file_name) {
    std::filesystem::create_directories(out_dir);
    const auto& deps = dependencies.at(obj_file_name);
    for (const auto& dep : deps) {
        std::ofstream{out_dir / dep} << "Content of " << dep;
    }
    // Write dependency hashes
    std::ofstream hash_file{(out_dir / obj_file_name).replace_extension(".hash")};
    tt::jit_build::write_dependency_hashes(dependencies, out_dir.string(), obj_file_name, hash_file);
    hash_file.close();
    ASSERT_FALSE(hash_file.fail());
}

}  // namespace

TEST(JitBuildTests, DependencyHashesUpToDate) {
    // Create temporary directory
    const auto temp_dir = std::filesystem::temp_directory_path() / "jit_build_test";
    constexpr auto obj_file_name = "test.o";
    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, {"a.txt", "b.txt", "c.txt"}},
    };
    create_dependency_files_and_hash(temp_dir, dependencies, obj_file_name);

    // Verify that dependencies are up to date
    EXPECT_TRUE(tt::jit_build::dependencies_up_to_date(temp_dir.string(), obj_file_name));
}

TEST(JitBuildTests, DependencyHashesOutOfDateAfterModification) {
    // Create temporary directory
    const auto temp_dir = std::filesystem::temp_directory_path() / "jit_build_test";
    constexpr auto obj_file_name = "test.o";
    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, {"a.txt", "b.txt", "c.txt"}},
    };
    create_dependency_files_and_hash(temp_dir, dependencies, obj_file_name);

    // Modify one dependency
    std::ofstream{temp_dir / "b.txt"} << "Modified content of b.txt";

    // Verify that dependencies are not up to date
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(temp_dir.string(), obj_file_name));
}

TEST(JitBuildTests, DependencyHashesOutOfDateAfterDeletion) {
    // Create temporary directory
    const auto temp_dir = std::filesystem::temp_directory_path() / "jit_build_test";
    constexpr auto obj_file_name = "test.o";
    const tt::jit_build::ParsedDependencies dependencies{
        {obj_file_name, {"a.txt", "b.txt", "c.txt"}},
    };
    create_dependency_files_and_hash(temp_dir, dependencies, obj_file_name);

    // Delete one dependency
    std::filesystem::remove(temp_dir / "c.txt");

    // Verify that dependencies are not up to date
    EXPECT_FALSE(tt::jit_build::dependencies_up_to_date(temp_dir.string(), obj_file_name));
}
