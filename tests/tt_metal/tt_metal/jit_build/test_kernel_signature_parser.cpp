// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <string>

#include "jit_build/kernel_signature_parser.hpp"

namespace tt::tt_metal {

// No marker -> not a TT_KERNEL kernel (legacy hand-written kernel_main()).
TEST(KernelSignatureParser, NoMarkerReturnsNullopt) {
    const std::string src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t x = get_arg_val<uint32_t>(0);
        }
    )";
    EXPECT_FALSE(parse_kernel_main_signature(src).has_value());
}

// The §2 worked example: 3 template params (CTAs) + 5 function params (RTAs/CRTAs).
TEST(KernelSignatureParser, AllThreeKinds) {
    const std::string src = R"(
        #define TT_KERNEL [[tt::kernel_main]] FORCE_INLINE
        template <uint32_t block_h, uint32_t block_w, uint32_t untilize>
        TT_KERNEL void my_kernel(uint32_t src_addr, uint32_t dst_addr,
                                 uint32_t num_tiles, uint32_t scaler, uint32_t sem_addr) {
            // body referencing TT_KERNEL in a comment should be ignored
            if constexpr (untilize) {}
        }
    )";
    auto sig = parse_kernel_main_signature(src);
    ASSERT_TRUE(sig.has_value());
    EXPECT_EQ(sig->name, "my_kernel");
    EXPECT_EQ(sig->template_param_names, (std::vector<std::string>{"block_h", "block_w", "untilize"}));
    EXPECT_EQ(
        sig->fn_param_names, (std::vector<std::string>{"src_addr", "dst_addr", "num_tiles", "scaler", "sem_addr"}));
}

// No template head -> only function params.
TEST(KernelSignatureParser, NoTemplateParams) {
    const std::string src = "TT_KERNEL void k(uint32_t a, uint32_t b) {}";
    auto sig = parse_kernel_main_signature(src);
    ASSERT_TRUE(sig.has_value());
    EXPECT_EQ(sig->name, "k");
    EXPECT_TRUE(sig->template_param_names.empty());
    EXPECT_EQ(sig->fn_param_names, (std::vector<std::string>{"a", "b"}));
}

// Template params only -> empty function-parameter list.
TEST(KernelSignatureParser, NoFunctionParams) {
    const std::string src = "template <uint32_t z>\nTT_KERNEL void k() {}";
    auto sig = parse_kernel_main_signature(src);
    ASSERT_TRUE(sig.has_value());
    EXPECT_EQ(sig->name, "k");
    EXPECT_EQ(sig->template_param_names, (std::vector<std::string>{"z"}));
    EXPECT_TRUE(sig->fn_param_names.empty());
}

// std::uint32_t is accepted as a synonym for uint32_t.
TEST(KernelSignatureParser, StdUint32Accepted) {
    const std::string src = "TT_KERNEL void k(std::uint32_t a) {}";
    auto sig = parse_kernel_main_signature(src);
    ASSERT_TRUE(sig.has_value());
    EXPECT_EQ(sig->fn_param_names, (std::vector<std::string>{"a"}));
}

// A const qualifier (leading or trailing) on a uint32_t parameter is accepted in Phase 1.
TEST(KernelSignatureParser, ConstQualifiedUint32Accepted) {
    const std::string src = "TT_KERNEL void k(const uint32_t a, uint32_t const b, std::uint32_t c) {}";
    auto sig = parse_kernel_main_signature(src);
    ASSERT_TRUE(sig.has_value());
    EXPECT_EQ(sig->fn_param_names, (std::vector<std::string>{"a", "b", "c"}));
}

// A TT_KERNEL inside a comment or the #define line must not be mistaken for the marker.
TEST(KernelSignatureParser, MarkerInCommentAndDefineIgnored) {
    const std::string src = R"(
        // This kernel uses TT_KERNEL to tag its entry.
        #define TT_KERNEL [[tt::kernel_main]]
        const char* note = "TT_KERNEL goes here";
        TT_KERNEL void only_real(uint32_t a) {}
    )";
    auto sig = parse_kernel_main_signature(src);
    ASSERT_TRUE(sig.has_value());
    EXPECT_EQ(sig->name, "only_real");
    EXPECT_EQ(sig->fn_param_names, (std::vector<std::string>{"a"}));
}

// A // comment continued across a trailing backslash-newline stays a comment, so a TT_KERNEL on
// the continued line is not a real marker. Without continuation handling this would see two
// markers (the decoy + the real one) and throw.
TEST(KernelSignatureParser, LineCommentContinuationIgnored) {
    const std::string src =
        "// this comment continues onto the next line \\\n"
        "TT_KERNEL void decoy(uint32_t x) {}\n"
        "TT_KERNEL void only_real(uint32_t a) {}\n";
    auto sig = parse_kernel_main_signature(src);
    ASSERT_TRUE(sig.has_value());
    EXPECT_EQ(sig->name, "only_real");
    EXPECT_EQ(sig->fn_param_names, (std::vector<std::string>{"a"}));
}

// Two real markers -> error.
TEST(KernelSignatureParser, MultipleMarkersThrow) {
    const std::string src = "TT_KERNEL void a(uint32_t x) {}\nTT_KERNEL void b(uint32_t y) {}";
    EXPECT_THROW(parse_kernel_main_signature(src), std::runtime_error);
}

// Off-surface in Phase 1: non-uint32_t parameter type.
TEST(KernelSignatureParser, NonUint32TypeThrows) {
    const std::string src = "TT_KERNEL void k(float x) {}";
    EXPECT_THROW(parse_kernel_main_signature(src), std::runtime_error);
}

// Off-surface: a type template parameter.
TEST(KernelSignatureParser, TypeTemplateParamThrows) {
    const std::string src = "template <typename T>\nTT_KERNEL void k(uint32_t a) {}";
    EXPECT_THROW(parse_kernel_main_signature(src), std::runtime_error);
}

// Off-surface: non-void return type.
TEST(KernelSignatureParser, NonVoidReturnThrows) {
    const std::string src = "TT_KERNEL uint32_t k(uint32_t a) {}";
    EXPECT_THROW(parse_kernel_main_signature(src), std::runtime_error);
}

// Off-surface: defaulted parameter.
TEST(KernelSignatureParser, DefaultedParamThrows) {
    const std::string src = "TT_KERNEL void k(uint32_t a = 5) {}";
    EXPECT_THROW(parse_kernel_main_signature(src), std::runtime_error);
}

// Build a large "fake" kernel source: many functions, big comment blocks, string literals,
// preprocessor noise, and decoy mentions of TT_KERNEL inside comments/strings/#defines — with
// exactly one real TT_KERNEL entry buried in the middle. `blocks` controls the size.
static std::string make_noisy_source(int blocks) {
    std::string s;
    s.reserve(static_cast<size_t>(blocks) * 512 + 1024);
    s += "#define TT_KERNEL FORCE_INLINE\n";
    s += "#define SOME_MACRO(x) ((x) + 1)\n";
    const int marker_at = blocks / 2;  // bury the real entry in the middle
    for (int i = 0; i < blocks; ++i) {
        // A block comment that itself mentions TT_KERNEL (must be ignored).
        s += "/* block " + std::to_string(i) +
             ": TT_KERNEL appears here in a comment and must not match.\n"
             "   More filler text to grow the source and pressure strip_noise(). */\n";
        // A line comment decoy.
        s += "// line comment mentioning TT_KERNEL void decoy(uint32_t x) {} as text\n";
        // A string literal decoy.
        s += "static const char* note_" + std::to_string(i) + " = \"TT_KERNEL void str_decoy(uint32_t y) {}\";\n";
        // A preprocessor decoy.
        s += "#define DECOY_" + std::to_string(i) + " TT_KERNEL\n";
        // Real (but non-marked) functions and a non-marked template, to add compile-shaped noise.
        s += "inline uint32_t plain_fn_" + std::to_string(i) + "(uint32_t a, uint32_t b) { return a + b; }\n";
        s += "template <typename T> T tmpl_fn_" + std::to_string(i) + "(T v) { return v; }\n";

        // The single real TT_KERNEL entry, buried mid-file.
        if (i == marker_at) {
            s += "template <uint32_t block_h, uint32_t block_w, uint32_t untilize>\n"
                 "TT_KERNEL void real_entry(uint32_t src_addr, uint32_t dst_addr,\n"
                 "                          uint32_t num_tiles, uint32_t scaler, uint32_t sem_addr) {\n"
                 "    /* the actual entry */\n"
                 "}\n";
        }
    }
    return s;
}

// Stress + correctness-under-noise + timing. Confirms the parser finds the one real entry in a
// large noisy source and reports parse throughput (the parser is wrapped in a Tracy zone, so
// this also feeds the JIT-overhead measurement when captured under Tracy).
TEST(KernelSignatureParser, StressLargeNoisySource) {
    constexpr int kBlocks = 20000;
    const std::string src = make_noisy_source(kBlocks);

    const auto t0 = std::chrono::steady_clock::now();
    auto sig = parse_kernel_main_signature(src);
    const auto t1 = std::chrono::steady_clock::now();

    ASSERT_TRUE(sig.has_value());
    EXPECT_EQ(sig->name, "real_entry");
    EXPECT_EQ(sig->template_param_names, (std::vector<std::string>{"block_h", "block_w", "untilize"}));
    EXPECT_EQ(
        sig->fn_param_names, (std::vector<std::string>{"src_addr", "dst_addr", "num_tiles", "scaler", "sem_addr"}));

    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mb = static_cast<double>(src.size()) / (1024.0 * 1024.0);
    std::printf("[stress] parsed %.2f MB (%d blocks) in %.1f us  =>  %.1f MB/s\n", mb, kBlocks, us, mb / (us / 1e6));
}

}  // namespace tt::tt_metal
