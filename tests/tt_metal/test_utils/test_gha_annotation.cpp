// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tt_metal/test_utils/gha_annotation.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using namespace ttsl::gha;
using namespace ttsl::gha::detail;

std::string read_stream_contents(std::FILE* stream) {
    if (std::fseek(stream, 0, SEEK_SET) != 0) {
        ADD_FAILURE() << "fseek to start of stream failed";
        return {};
    }
    std::string out;
    std::array<char, 512> buffer{};
    while (true) {
        const std::size_t bytes = std::fread(buffer.data(), 1, buffer.size(), stream);
        if (bytes == 0) {
            break;
        }
        out.append(buffer.data(), bytes);
    }
    return out;
}

TEST(GhaAnnotationTest, ToStringMapsAllLevels) {
    EXPECT_EQ(to_string(annotation_level::notice), "notice");
    EXPECT_EQ(to_string(annotation_level::warning), "warning");
    EXPECT_EQ(to_string(annotation_level::error), "error");
}

TEST(GhaAnnotationTest, EnvValueIsTruthyRecognizesTruthyValues) {
    EXPECT_TRUE(env_value_is_truthy("1"));
    EXPECT_TRUE(env_value_is_truthy("true"));
    EXPECT_TRUE(env_value_is_truthy("YES"));
    EXPECT_TRUE(env_value_is_truthy("On"));
    EXPECT_FALSE(env_value_is_truthy("0"));
}

TEST(GhaAnnotationTest, EnvVarIsTruthyHandlesNullAndTruthyValues) {
    EXPECT_FALSE(env_var_is_truthy(nullptr));
    EXPECT_TRUE(env_var_is_truthy("true"));
    EXPECT_FALSE(env_var_is_truthy("false"));
}

TEST(GhaAnnotationTest, ShouldEmitAnnotationsUsesExplicitEnvVariable) {
    constexpr const char* kEnvVar = "TTSL_GHA_EXPLICIT_TEST";
    ASSERT_EQ(::setenv(kEnvVar, "1", 1), 0);
    EXPECT_TRUE(should_emit_annotations(kEnvVar));
    ASSERT_EQ(::setenv(kEnvVar, "false", 1), 0);
    EXPECT_FALSE(should_emit_annotations(kEnvVar));
    ASSERT_EQ(::unsetenv(kEnvVar), 0);
}

TEST(GhaAnnotationTest, MakeAnnotationFromLevelAndMessagePopulatesSourceLocation) {
    const annotation a = make_annotation(annotation_level::notice, "hello");
    EXPECT_FALSE(a.file.empty());
    ASSERT_TRUE(a.line.has_value());
    EXPECT_GT(*a.line, 0u);
    EXPECT_FALSE(a.title.empty());
}

TEST(GhaAnnotationTest, MakeAnnotationFromAnnotationPreservesExplicitFields) {
    const annotation base{
        .level = annotation_level::error,
        .message = "msg",
        .line = std::uint_least32_t{42},
        .title = "explicit_title",
    };
    const annotation a = make_annotation(base);
    EXPECT_EQ(a.title, "explicit_title");
    ASSERT_TRUE(a.line.has_value());
    EXPECT_EQ(*a.line, std::uint_least32_t{42});
    EXPECT_FALSE(a.file.empty());
}

TEST(GhaAnnotationTest, BuildWorkflowCommandFormatsCommand) {
    const annotation a = make_annotation(annotation_level::warning, "oops");
    const std::string cmd = build_workflow_command(a);
    EXPECT_THAT(cmd, ::testing::HasSubstr("::warning"));
    EXPECT_THAT(cmd, ::testing::HasSubstr("oops"));
    EXPECT_THAT(cmd, ::testing::HasSubstr("file="));
    EXPECT_THAT(cmd, ::testing::HasSubstr("line="));
}

TEST(GhaAnnotationTest, EmitAnnotationWritesToStream) {
    const annotation a = make_annotation(annotation_level::error, "stream_msg");
    std::FILE* stream = std::tmpfile();
    ASSERT_NE(stream, nullptr);
    emit_annotation(a, stream);
    const std::string output = read_stream_contents(stream);
    EXPECT_THAT(output, ::testing::HasSubstr("::error"));
    EXPECT_THAT(output, ::testing::HasSubstr("stream_msg"));
    std::fclose(stream);
}

TEST(GhaAnnotationTest, EmitAnnotationAtWritesToStream) {
    testing::internal::CaptureStdout();
    emit_annotation_at(annotation_level::warning, "Don't worry about it");
    const std::string stdout_output = testing::internal::GetCapturedStdout();
    // CaptureStdout redirects fd 1; echo to stderr so smoke / CI logs still show the workflow command.
    fmt::print(stderr, "{}", stdout_output);
    std::fflush(stderr);
    EXPECT_THAT(stdout_output, ::testing::HasSubstr("::warning"));
    EXPECT_THAT(stdout_output, ::testing::HasSubstr("Don't worry about it"));
}

TEST(GhaAnnotationTest, EmitAnnotationAtFromAnnotationWritesToStream) {
    std::FILE* stream = std::tmpfile();
    ASSERT_NE(stream, nullptr);
    const annotation ann{.level = annotation_level::warning, .message = "emit_at_annotation_msg"};
    emit_annotation_at(ann, stream);
    const std::string output = read_stream_contents(stream);
    EXPECT_THAT(output, ::testing::HasSubstr("::warning"));
    EXPECT_THAT(output, ::testing::HasSubstr("emit_at_annotation_msg"));
    std::fclose(stream);
}

}  // namespace
