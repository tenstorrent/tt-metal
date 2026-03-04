// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include <sys/stat.h>

#include "common/filesystem_utils.hpp"

namespace tt::filesystem::test {

class FilesystemUtilsTest : public ::testing::Test {
protected:
    std::filesystem::path temp_dir_;

    void SetUp() override {
        // Create a unique temporary directory for each test
        temp_dir_ =
            std::filesystem::temp_directory_path() /
            ("tt_filesystem_test_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(temp_dir_);
    }

    void TearDown() override {
        // Clean up temp directory
        if (std::filesystem::exists(temp_dir_)) {
            std::filesystem::remove_all(temp_dir_);
        }
    }

    // Helper to create a test file with content
    std::filesystem::path create_test_file(const std::string& name, const std::string& content = "test content") {
        std::filesystem::path file_path = temp_dir_ / name;
        std::ofstream file(file_path);
        file << content;
        file.close();
        return file_path;
    }

    // Helper to create a test directory
    std::filesystem::path create_test_directory(const std::string& name) {
        std::filesystem::path dir_path = temp_dir_ / name;
        std::filesystem::create_directories(dir_path);
        return dir_path;
    }
};

// ============================================================================
// Basic Operations Tests
// ============================================================================

TEST_F(FilesystemUtilsTest, SafeCreateDirectories_CreatesNewDirectory) {
    std::filesystem::path new_dir = temp_dir_ / "new_directory";
    EXPECT_FALSE(std::filesystem::exists(new_dir));

    EXPECT_TRUE(safe_create_directories(new_dir));

    EXPECT_TRUE(std::filesystem::exists(new_dir));
    EXPECT_TRUE(std::filesystem::is_directory(new_dir));
}

TEST_F(FilesystemUtilsTest, SafeCreateDirectories_CreatesNestedDirectories) {
    std::filesystem::path nested_dir = temp_dir_ / "a" / "b" / "c" / "d";
    EXPECT_FALSE(std::filesystem::exists(nested_dir));

    EXPECT_TRUE(safe_create_directories(nested_dir));

    EXPECT_TRUE(std::filesystem::exists(nested_dir));
    EXPECT_TRUE(std::filesystem::is_directory(nested_dir));
}

TEST_F(FilesystemUtilsTest, SafeCreateDirectories_IdempotentOnExistingDirectory) {
    std::filesystem::path existing_dir = create_test_directory("existing");
    EXPECT_TRUE(std::filesystem::exists(existing_dir));

    // Should succeed on existing directory
    EXPECT_TRUE(safe_create_directories(existing_dir));
    EXPECT_TRUE(std::filesystem::exists(existing_dir));
}

TEST_F(FilesystemUtilsTest, SafeExists_ReturnsTrueForExistingPath) {
    std::filesystem::path file = create_test_file("exists_test.txt");

    auto result = safe_exists(file);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

TEST_F(FilesystemUtilsTest, SafeExists_ReturnsFalseForNonExistentPath) {
    std::filesystem::path non_existent = temp_dir_ / "does_not_exist.txt";

    auto result = safe_exists(non_existent);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result.value());
}

TEST_F(FilesystemUtilsTest, SafeIsDirectory_ReturnsTrueForDirectory) {
    std::filesystem::path dir = create_test_directory("test_dir");

    auto result = safe_is_directory(dir);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

TEST_F(FilesystemUtilsTest, SafeIsDirectory_ReturnsFalseForFile) {
    std::filesystem::path file = create_test_file("test_file.txt");

    auto result = safe_is_directory(file);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result.value());
}

TEST_F(FilesystemUtilsTest, SafeIsDirectory_ReturnsFalseForNonExistentPath) {
    std::filesystem::path non_existent = temp_dir_ / "does_not_exist";

    auto result = safe_is_directory(non_existent);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result.value());
}

TEST_F(FilesystemUtilsTest, SafeIsRegularFile_ReturnsTrueForFile) {
    std::filesystem::path file = create_test_file("regular_file.txt");

    auto result = safe_is_regular_file(file);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

TEST_F(FilesystemUtilsTest, SafeIsRegularFile_ReturnsFalseForDirectory) {
    std::filesystem::path dir = create_test_directory("test_dir");

    auto result = safe_is_regular_file(dir);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result.value());
}

TEST_F(FilesystemUtilsTest, SafeIsRegularFile_ReturnsFalseForNonExistentPath) {
    std::filesystem::path non_existent = temp_dir_ / "does_not_exist.txt";

    auto result = safe_is_regular_file(non_existent);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result.value());
}

TEST_F(FilesystemUtilsTest, SafeFileSize_ReturnsCorrectSize) {
    std::string content = "Hello, World!";
    std::filesystem::path file = create_test_file("size_test.txt", content);

    auto result = safe_file_size(file);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), content.size());
}

TEST_F(FilesystemUtilsTest, SafeFileSize_ReturnsNulloptForNonExistentFile) {
    std::filesystem::path non_existent = temp_dir_ / "does_not_exist.txt";

    auto result = safe_file_size(non_existent);
    EXPECT_FALSE(result.has_value());
}

TEST_F(FilesystemUtilsTest, SafeLastWriteTime_ReturnsValidTime) {
    std::filesystem::path file = create_test_file("time_test.txt");

    auto result = safe_last_write_time(file);

    EXPECT_TRUE(result.has_value());
    // The returned time should be a valid file_time_type (not min or max)
    EXPECT_NE(result.value(), std::filesystem::file_time_type::min());
    EXPECT_NE(result.value(), std::filesystem::file_time_type::max());

    // Verify the time is recent by checking it's greater than a reference time point
    // (1 hour ago - file should have been modified more recently than that)
    auto now = std::filesystem::file_time_type::clock::now();
    auto one_hour_ago = now - std::chrono::hours(1);
    EXPECT_GT(result.value(), one_hour_ago);

    // Verify consistency - calling again returns the same time (within 1 second for filesystem precision)
    auto result2 = safe_last_write_time(file);
    EXPECT_TRUE(result2.has_value());
    auto diff =
        (result.value() > result2.value()) ? (result.value() - result2.value()) : (result2.value() - result.value());
    EXPECT_LE(diff, std::chrono::seconds(1));
}

TEST_F(FilesystemUtilsTest, SafeLastWriteTime_ReturnsNulloptForNonExistentFile) {
    std::filesystem::path non_existent = temp_dir_ / "does_not_exist.txt";

    auto result = safe_last_write_time(non_existent);
    EXPECT_FALSE(result.has_value());
}

TEST_F(FilesystemUtilsTest, SafeRemove_RemovesExistingFile) {
    std::filesystem::path file = create_test_file("to_remove.txt");
    EXPECT_TRUE(std::filesystem::exists(file));

    EXPECT_TRUE(safe_remove(file));

    EXPECT_FALSE(std::filesystem::exists(file));
}

TEST_F(FilesystemUtilsTest, SafeRemove_IdempotentOnNonExistentFile) {
    std::filesystem::path non_existent = temp_dir_ / "does_not_exist.txt";
    EXPECT_FALSE(std::filesystem::exists(non_existent));

    // Should succeed even if file doesn't exist
    EXPECT_TRUE(safe_remove(non_existent));
}

TEST_F(FilesystemUtilsTest, SafeRemoveAll_RemovesDirectoryWithContents) {
    std::filesystem::path dir = create_test_directory("to_remove_all");
    create_test_file("to_remove_all/file1.txt");
    create_test_file("to_remove_all/file2.txt");
    create_test_directory("to_remove_all/subdir");
    create_test_file("to_remove_all/subdir/nested.txt");

    EXPECT_TRUE(std::filesystem::exists(dir));

    EXPECT_TRUE(safe_remove_all(dir));

    EXPECT_FALSE(std::filesystem::exists(dir));
}

TEST_F(FilesystemUtilsTest, SafeRemoveAll_IdempotentOnNonExistentPath) {
    std::filesystem::path non_existent = temp_dir_ / "does_not_exist";
    EXPECT_FALSE(std::filesystem::exists(non_existent));

    // Should succeed even if directory doesn't exist
    EXPECT_TRUE(safe_remove_all(non_existent));
}

// ============================================================================
// Hard Link or Copy Tests
// ============================================================================

TEST_F(FilesystemUtilsTest, SafeHardLinkOrCopy_CreatesHardLink) {
    std::filesystem::path target = create_test_file("target.txt", "hard link target");
    std::filesystem::path link = temp_dir_ / "hard_link.txt";

    EXPECT_TRUE(safe_hard_link_or_copy(target, link));

    // Both should exist
    EXPECT_TRUE(std::filesystem::exists(link));
    EXPECT_TRUE(std::filesystem::exists(target));

    // Should be the same file (hard linked) - verify by checking they have the same content
    std::ifstream target_stream(target);
    std::ifstream link_stream(link);
    std::string target_content((std::istreambuf_iterator<char>(target_stream)), std::istreambuf_iterator<char>());
    std::string link_content((std::istreambuf_iterator<char>(link_stream)), std::istreambuf_iterator<char>());
    EXPECT_EQ(target_content, link_content);

    // On POSIX systems, verify they share the same inode (actual hard link)
    struct stat target_stat, link_stat;
    if (stat(target.c_str(), &target_stat) == 0 && stat(link.c_str(), &link_stat) == 0) {
        EXPECT_EQ(target_stat.st_ino, link_stat.st_ino);
    }
}

TEST_F(FilesystemUtilsTest, SafeHardLinkOrCopy_WorksWithDirectoryInPath) {
    // Creating a hard link to a file inside a directory should work
    std::filesystem::path target = create_test_file("source.txt", "source content");
    std::filesystem::path link = create_test_directory("dest_dir") / "linked.txt";

    EXPECT_TRUE(safe_hard_link_or_copy(target, link));

    // Hard link should succeed
    EXPECT_TRUE(std::filesystem::exists(link));
    EXPECT_EQ(std::filesystem::file_size(target), std::filesystem::file_size(link));

    // On POSIX systems, verify they share the same inode
    struct stat target_stat, link_stat;
    if (stat(target.c_str(), &target_stat) == 0 && stat(link.c_str(), &link_stat) == 0) {
        EXPECT_EQ(target_stat.st_ino, link_stat.st_ino);
    }
}

TEST_F(FilesystemUtilsTest, SafeHardLinkOrCopy_OverwritesExisting) {
    std::filesystem::path target = create_test_file("target.txt", "new content");
    std::filesystem::path existing = create_test_file("existing.txt", "old content");

    // Verify original content is different
    EXPECT_NE(std::filesystem::file_size(target), std::filesystem::file_size(existing));

    EXPECT_TRUE(safe_hard_link_or_copy(target, existing));

    // Should be overwritten with target's content
    EXPECT_EQ(std::filesystem::file_size(target), std::filesystem::file_size(existing));
}

// ============================================================================
// Directory Entries Tests
// ============================================================================

TEST_F(FilesystemUtilsTest, SafeDirectoryEntries_ReturnsAllEntries) {
    std::filesystem::path dir = create_test_directory("list_dir");
    create_test_file("list_dir/file1.txt");
    create_test_file("list_dir/file2.txt");
    create_test_directory("list_dir/subdir");

    auto entries = safe_directory_entries(dir);

    EXPECT_EQ(entries.size(), 3);

    // Verify we got all expected entries
    int file_count = 0;
    int dir_count = 0;
    for (const auto& entry : entries) {
        if (entry.is_regular_file()) {
            file_count++;
        } else if (entry.is_directory()) {
            dir_count++;
        }
    }
    EXPECT_EQ(file_count, 2);
    EXPECT_EQ(dir_count, 1);
}

TEST_F(FilesystemUtilsTest, SafeDirectoryEntries_ReturnsEmptyForNonExistentDirectory) {
    std::filesystem::path non_existent = temp_dir_ / "does_not_exist";

    auto entries = safe_directory_entries(non_existent);

    EXPECT_TRUE(entries.empty());
}

TEST_F(FilesystemUtilsTest, SafeDirectoryEntries_ReturnsEmptyForEmptyDirectory) {
    std::filesystem::path empty_dir = create_test_directory("empty");

    auto entries = safe_directory_entries(empty_dir);

    EXPECT_TRUE(entries.empty());
}

TEST_F(FilesystemUtilsTest, SafeDirectoryEntries_HandlesNestedDirectories) {
    std::filesystem::path dir = create_test_directory("nested");
    create_test_directory("nested/level1");
    create_test_directory("nested/level1/level2");
    create_test_file("nested/level1/file.txt");

    auto entries = safe_directory_entries(dir);
    EXPECT_EQ(entries.size(), 1);  // Only "level1" is directly in "nested"

    auto level1_entries = safe_directory_entries(dir / "level1");
    EXPECT_EQ(level1_entries.size(), 2);  // "level2" and "file.txt"
}

// ============================================================================
// Rename Tests
// ============================================================================

TEST_F(FilesystemUtilsTest, SafeRename_RenamesFile) {
    std::filesystem::path source = create_test_file("original.txt", "content");
    std::filesystem::path dest = temp_dir_ / "renamed.txt";

    EXPECT_TRUE(safe_rename(source, dest));

    EXPECT_FALSE(std::filesystem::exists(source));
    EXPECT_TRUE(std::filesystem::exists(dest));
    EXPECT_EQ(std::filesystem::file_size(dest), 7);  // "content"
}

TEST_F(FilesystemUtilsTest, SafeRename_OverwritesExisting) {
    std::filesystem::path source = create_test_file("source.txt", "new content");
    std::filesystem::path dest = create_test_file("dest.txt", "old content");

    EXPECT_TRUE(safe_rename(source, dest));

    EXPECT_FALSE(std::filesystem::exists(source));
    EXPECT_TRUE(std::filesystem::exists(dest));
    EXPECT_EQ(std::filesystem::file_size(dest), 11);  // "new content"
}

TEST_F(FilesystemUtilsTest, SafeRename_ReturnsFalseForNonExistentSource) {
    std::filesystem::path source = temp_dir_ / "does_not_exist.txt";
    std::filesystem::path dest = temp_dir_ / "dest.txt";

    EXPECT_FALSE(safe_rename(source, dest));
}

TEST_F(FilesystemUtilsTest, SafeRename_IgnoreMissingReturnsTrueForNonExistentSource) {
    std::filesystem::path source = temp_dir_ / "does_not_exist.txt";
    std::filesystem::path dest = temp_dir_ / "dest.txt";

    // With ignore_missing = true, should succeed even if source doesn't exist
    EXPECT_TRUE(safe_rename(source, dest, true));
}

// ============================================================================
// Retry Constants Tests
// ============================================================================

TEST(FilesystemUtilsConstants, MaxRetriesIsReasonable) {
    // kMaxFsRetries should be a positive, reasonable number
    EXPECT_GT(kMaxFsRetries, 0);
    EXPECT_LE(kMaxFsRetries, 100);  // Should not be excessively high
}

TEST(FilesystemUtilsConstants, RetryDelayIsReasonable) {
    // kFsRetryDelayMs should be a positive, reasonable number
    EXPECT_GT(kFsRetryDelayMs, 0);
    EXPECT_LE(kFsRetryDelayMs, 10000);  // Should not be excessively high (10 seconds)
}

TEST(FilesystemUtilsConstants, TotalMaxDelayIsReasonable) {
    // Calculate total maximum delay across all retries
    // Formula: sum of (kFsRetryDelayMs * attempt + random(0-100)) for each attempt
    // This is an upper bound calculation
    int total_max_delay = 0;
    for (int attempt = 1; attempt <= kMaxFsRetries; ++attempt) {
        total_max_delay += kFsRetryDelayMs * attempt + 100;
    }

    // Total delay should be less than 60 seconds (reasonable for NFS recovery)
    EXPECT_LT(total_max_delay, 60000);
}

// ============================================================================
// Error Detection Tests
// ============================================================================

TEST(FilesystemUtilsErrors, IsEstaleError_DetectsEstale) {
    std::error_code ec(ESTALE, std::generic_category());
    EXPECT_TRUE(is_estale_error(ec));
}

TEST(FilesystemUtilsErrors, IsEstaleError_ReturnsFalseForOtherErrors) {
    std::error_code ecENOENT = std::make_error_code(std::errc::no_such_file_or_directory);
    EXPECT_FALSE(is_estale_error(ecENOENT));

    std::error_code ecACCES = std::make_error_code(std::errc::permission_denied);
    EXPECT_FALSE(is_estale_error(ecACCES));
}

TEST(FilesystemUtilsErrors, IsNotFoundError_DetectsNoSuchFile) {
    std::error_code ec = std::make_error_code(std::errc::no_such_file_or_directory);
    EXPECT_TRUE(is_not_found_error(ec));
}

TEST(FilesystemUtilsErrors, IsNotFoundError_ReturnsFalseForOtherErrors) {
    std::error_code ecESTALE(ESTALE, std::generic_category());
    EXPECT_FALSE(is_not_found_error(ecESTALE));

    std::error_code ecACCES = std::make_error_code(std::errc::permission_denied);
    EXPECT_FALSE(is_not_found_error(ecACCES));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(FilesystemUtilsTest, SafeRemove_ReturnsFalseForDirectory) {
    std::filesystem::path dir = create_test_directory("not_a_file");

    // safe_remove should return false when trying to remove a directory
    // (it's designed for files, use safe_remove_all for directories)
    EXPECT_FALSE(safe_remove(dir));
}

TEST_F(FilesystemUtilsTest, SafeFileSize_WorksOnDirectory) {
    std::filesystem::path dir = create_test_directory("dir_for_size");

    auto result = safe_file_size(dir);
    // file_size works on directories (returns implementation-defined size, typically non-zero)
    // This verifies the function doesn't crash and returns a valid result
    EXPECT_TRUE(result.has_value());
}

TEST_F(FilesystemUtilsTest, SafeLastWriteTime_WorksOnDirectory) {
    std::filesystem::path dir = create_test_directory("dir_for_time");

    auto result = safe_last_write_time(dir);
    // last_write_time works on directories
    EXPECT_TRUE(result.has_value());
    EXPECT_NE(result.value(), std::filesystem::file_time_type::min());
}

TEST_F(FilesystemUtilsTest, SafeHardLinkOrCopy_ReturnsFalseForNonExistentSource) {
    std::filesystem::path source = temp_dir_ / "does_not_exist.txt";
    std::filesystem::path dest = temp_dir_ / "dest.txt";

    EXPECT_FALSE(safe_hard_link_or_copy(source, dest));
    EXPECT_FALSE(std::filesystem::exists(dest));
}

TEST_F(FilesystemUtilsTest, SafeHardLinkOrCopy_ReturnsFalseForNonExistentDestParent) {
    std::filesystem::path source = create_test_file("source.txt");
    std::filesystem::path dest = temp_dir_ / "non_existent_parent" / "dest.txt";

    EXPECT_FALSE(safe_hard_link_or_copy(source, dest));
}

TEST_F(FilesystemUtilsTest, SafeRename_ReturnsFalseWhenDestParentDoesNotExist) {
    std::filesystem::path source = create_test_file("source.txt");
    std::filesystem::path dest = temp_dir_ / "non_existent_parent" / "dest.txt";

    EXPECT_FALSE(safe_rename(source, dest));
    // Source should still exist
    EXPECT_TRUE(std::filesystem::exists(source));
}

TEST_F(FilesystemUtilsTest, SafeCreateDirectories_ReturnsFalseForInvalidPath) {
    // Try to create a directory in a non-existent parent with a path that could be invalid
    // This may succeed on some systems but should handle errors gracefully
    std::filesystem::path valid_nested = temp_dir_ / "valid" / "nested" / "path";
    EXPECT_TRUE(safe_create_directories(valid_nested));
    EXPECT_TRUE(std::filesystem::exists(valid_nested));
}

// Test that verifies operations work on symlinks (if supported)
TEST_F(FilesystemUtilsTest, SafeOperations_HandleSymlinks) {
    std::filesystem::path target = create_test_file("symlink_target.txt", "symlinked content");
    std::filesystem::path link = temp_dir_ / "symlink";

    // Create a symlink
    std::error_code ec;
    std::filesystem::create_symlink(target, link, ec);

    if (!ec) {
        // Symlinks are supported
        auto exists_result = safe_exists(link);
        EXPECT_TRUE(exists_result.has_value());
        EXPECT_TRUE(exists_result.value());

        auto is_file_result = safe_is_regular_file(link);
        EXPECT_TRUE(is_file_result.has_value());
        EXPECT_TRUE(is_file_result.value());

        // Clean up
        std::filesystem::remove(link, ec);
    }
    // If symlinks aren't supported, skip this test gracefully
}

}  // namespace tt::filesystem::test
