// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

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

}  // namespace tt::tt_metal
