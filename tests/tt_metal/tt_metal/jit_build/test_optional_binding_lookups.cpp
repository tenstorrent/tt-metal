// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "jit_build/jit_build_utils.hpp"
#include "llrt/rtoptions.hpp"

// A Metal 2.0 kernel's resources arrive as codegen-emitted namespace-scope tokens (tensor::x,
// dfb::x, scratch::x). A token is emitted only if the host bound that resource to that kernel, so
// merely naming an unbound resource is a compile error -- which leaves no way to write a kernel
// parameter that the host may or may not supply. Today's only workaround is a host-set -D and an
// #ifdef in the kernel.
//
// format_optional_binding_lookups() closes that gap device-side: it emits three name-keyed lookups
// that answer "did the host bind this?" as a compile-time constant, so the absent branch is
// provably dead rather than ill-formed. These tests pin the emitted text (the exact spelling is
// part of the contract -- it is what kernels are written against and what the emule path must
// reproduce byte-for-byte) and then compile it, which is what actually proves the shape is valid
// C++ and that presence folds away at compile time.
namespace tt::jit_build::utils {
namespace {

// ---------------------------------------------------------------------------------------------
// Emitted text
// ---------------------------------------------------------------------------------------------

// The lookups exist precisely so a kernel can ask about a resource it may not have, so a kernel
// that binds nothing at all is not a degenerate case to skip -- it is the primary one.
TEST(OptionalBindingLookups, KernelWithNoBindingsStillGetsAllThreeLookups) {
    const std::string out = format_optional_binding_lookups({}, {}, {});

    EXPECT_EQ(
        out,
        R"(template <::binding::Name NAME>
[[nodiscard]] constexpr auto try_get_tensor_binding() {
    // This kernel has no tensor bindings, so every name is absent.
    return std::optional<::binding::NullTensorBindingToken>();
}

template <::binding::Name NAME>
[[nodiscard]] constexpr std::optional<::DFBBindingToken> try_get_dfb_binding() {
    // This kernel has no bindings of this kind, so every name is absent.
    return std::optional<::DFBBindingToken>();
}

template <::binding::Name NAME>
[[nodiscard]] constexpr std::optional<::ScratchpadBindingToken> try_get_scratchpad_binding() {
    // This kernel has no bindings of this kind, so every name is absent.
    return std::optional<::ScratchpadBindingToken>();
}
)");
}

// Each tensor binding has its own TensorBindingToken specialization, so the tensor lookup's return
// type differs per arm and the function is deduced (`auto`) rather than a fixed std::optional<T>.
TEST(OptionalBindingLookups, TensorLookupReturnsThePerBindingTokenType) {
    const std::string out = format_optional_binding_lookups({"in", "out"}, {}, {});

    EXPECT_NE(
        out.find(
            R"(template <::binding::Name NAME>
[[nodiscard]] constexpr auto try_get_tensor_binding() {
    if constexpr (NAME == ::binding::Name("in")) {
        return std::optional<::tensor::in_t>(::tensor::in);
    } else if constexpr (NAME == ::binding::Name("out")) {
        return std::optional<::tensor::out_t>(::tensor::out);
    } else {
        return std::optional<::binding::NullTensorBindingToken>();
    }
}
)"),
        std::string::npos)
        << out;
}

// DFBBindingToken is a single concrete type for every DFB binding, so this lookup can name its
// return type outright -- no null stand-in needed, and no `auto`.
TEST(OptionalBindingLookups, DfbLookupReturnsOptionalOfTheConcreteToken) {
    const std::string out = format_optional_binding_lookups({}, {"gamma", "beta"}, {});

    EXPECT_NE(
        out.find(
            R"(template <::binding::Name NAME>
[[nodiscard]] constexpr std::optional<::DFBBindingToken> try_get_dfb_binding() {
    if constexpr (NAME == ::binding::Name("gamma")) {
        return std::optional<::DFBBindingToken>(::dfb::gamma);
    } else if constexpr (NAME == ::binding::Name("beta")) {
        return std::optional<::DFBBindingToken>(::dfb::beta);
    } else {
        return std::optional<::DFBBindingToken>();
    }
}
)"),
        std::string::npos)
        << out;
}

TEST(OptionalBindingLookups, ScratchpadLookupReturnsOptionalOfTheConcreteToken) {
    const std::string out = format_optional_binding_lookups({}, {}, {"tmp"});

    EXPECT_NE(
        out.find(
            R"(template <::binding::Name NAME>
[[nodiscard]] constexpr std::optional<::ScratchpadBindingToken> try_get_scratchpad_binding() {
    if constexpr (NAME == ::binding::Name("tmp")) {
        return std::optional<::ScratchpadBindingToken>(::scratch::tmp);
    } else {
        return std::optional<::ScratchpadBindingToken>();
    }
}
)"),
        std::string::npos)
        << out;
}

// The three kinds are independent: binding one must not change what the other two emit. A kernel
// with tensors but no scratchpads still needs a working try_get_scratchpad_binding.
TEST(OptionalBindingLookups, EachKindIsEmittedIndependentlyOfTheOthers) {
    const std::string mixed = format_optional_binding_lookups({"in"}, {"gamma"}, {});
    const std::string tensor_only = format_optional_binding_lookups({"in"}, {}, {});
    const std::string all_empty = format_optional_binding_lookups({}, {}, {});

    // The scratchpad arm of the mixed kernel is the same text as for a kernel that binds nothing.
    const std::size_t marker = all_empty.find("try_get_scratchpad_binding");
    ASSERT_NE(marker, std::string::npos);
    EXPECT_NE(mixed.find(all_empty.substr(marker)), std::string::npos);
    EXPECT_NE(tensor_only.find(all_empty.substr(marker)), std::string::npos);
}

// Ordering is caller-supplied and must be preserved verbatim: the emitted arms reference the
// per-binding `<name>_t` aliases emitted alongside the tokens in the same header, and this text
// feeds a per-object dephash cache, so it has to be byte-stable for a given binding set.
TEST(OptionalBindingLookups, ArmOrderFollowsTheCallerSuppliedNameOrder) {
    const std::string forward = format_optional_binding_lookups({}, {"a", "b"}, {});
    const std::string reverse = format_optional_binding_lookups({}, {"b", "a"}, {});

    EXPECT_NE(forward, reverse);
    EXPECT_LT(forward.find(R"(Name("a"))"), forward.find(R"(Name("b"))"));
    EXPECT_LT(reverse.find(R"(Name("b"))"), reverse.find(R"(Name("a"))"));
    EXPECT_EQ(format_optional_binding_lookups({}, {"a", "b"}, {}), forward);
}

// Every lookup must be constexpr (emptiness has to be a compile-time constant, or the absent
// branch is not dead code) and [[nodiscard]] (calling one and dropping the result is a mistake).
TEST(OptionalBindingLookups, EveryLookupIsConstexprAndNodiscard) {
    const std::string out = format_optional_binding_lookups({"in"}, {"gamma"}, {"tmp"});

    for (std::string_view fn : {"try_get_tensor_binding", "try_get_dfb_binding", "try_get_scratchpad_binding"}) {
        const std::size_t at = out.find(fn);
        ASSERT_NE(at, std::string::npos) << fn;
        const std::size_t line_start = out.rfind('\n', at) + 1;
        const std::string_view signature(out.data() + line_start, at - line_start);
        EXPECT_NE(signature.find("[[nodiscard]]"), std::string_view::npos) << fn;
        EXPECT_NE(signature.find("constexpr"), std::string_view::npos) << fn;
    }
}

// The include block is emitted unconditionally, so it must cover every type the lookups can name --
// including DFBBindingToken and ScratchpadBindingToken, which a kernel with no bindings of that kind
// would otherwise never have declared. That is why the two headers below are here and not, as they
// once were, emitted only when the kernel actually binds a resource of that kind.
TEST(OptionalBindingLookups, IncludeBlockCoversEveryTypeTheLookupsName) {
    for (std::string_view header : {"api/optional_binding.h", "api/dataflow/dataflow_buffer.h", "api/scratchpad.h"}) {
        EXPECT_NE(OPTIONAL_BINDING_LOOKUP_INCLUDES.find(header), std::string_view::npos) << header;
    }
}

// ---------------------------------------------------------------------------------------------
// Compile check
// ---------------------------------------------------------------------------------------------

// Namespace bodies as genfiles.cpp emits them, so the lookups' arms have real tokens to return.
constexpr std::string_view BINDING_NAMESPACES = R"(
namespace tensor {
using in_t = ::tensor_accessor::TensorBindingToken<0u, 16u>;
constexpr in_t in{};
}  // namespace tensor

namespace dfb {
constexpr DFBBindingToken gamma{5};
}  // namespace dfb

namespace scratch {
constexpr ScratchpadBindingToken tmp{12u, 256u};
}  // namespace scratch
)";

// None of the three real token types is compilable off-device: TensorBindingToken pulls in the CTA
// plumbing and arch headers, and DFBBindingToken and ScratchpadBindingToken live inside
// dataflow_buffer.h and scratchpad.h, which pull in the NOC and L1 layers around them. Stand-ins
// reproduce the only properties this test depends on, copied from the real declarations:
//   - TensorBindingToken is a class TEMPLATE, so the tensor lookup's per-arm return types differ,
//     which is the whole reason that lookup returns `auto` and needs a null stand-in type;
//   - the other two are single concrete types, which is why their lookups can name
//     std::optional<Token> outright.
// binding::Name and binding::NullTensorBindingToken come from the real api/optional_binding.h --
// they are what this change actually adds, so standing them in would test nothing.
constexpr std::string_view TOKEN_STANDINS = R"(
namespace tensor_accessor {
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
struct TensorBindingToken {
    static constexpr uint32_t addr_crta_offset = ADDR_CRTA_OFFSET;
};
}  // namespace tensor_accessor

struct DFBBindingToken {
    explicit constexpr DFBBindingToken(uint16_t id) noexcept : id_(id) {}
    constexpr operator uint32_t() const noexcept { return id_; }
private:
    uint16_t id_;
};

class ScratchpadBindingToken {
public:
    explicit constexpr ScratchpadBindingToken(uint32_t crta_offset, uint32_t size_in_bytes) noexcept :
        crta_offset_(crta_offset), size_in_bytes_(size_in_bytes) {}
private:
    template <typename T>
    friend class Scratchpad;

    uint32_t crta_offset_;
    uint32_t size_in_bytes_;
};
)";

// Mirrors the constructor surface of the three real consumers, including the
// NullTensorBindingToken overload added to hw/inc/api/tensor/local_tensor_accessor.h. That overload
// is what makes an absent tensor branch well-formed: a discarded `if constexpr` branch is only left
// uninstantiated inside a template, and kernel_main is not a template.
constexpr std::string_view CONSUMER_STANDINS = R"(
template <typename T>
class LocalTensorAccessor {
public:
    template <uint32_t C, uint32_t A>
    explicit LocalTensorAccessor(tensor_accessor::TensorBindingToken<C, A>) noexcept : addr_(A) {}
    explicit LocalTensorAccessor(::binding::NullTensorBindingToken) noexcept : LocalTensorAccessor(uint32_t{0}) {}
    explicit LocalTensorAccessor(uint32_t base) noexcept : addr_(base) {}
    constexpr uint32_t addr() const { return addr_; }
private:
    uint32_t addr_;
};
class DataflowBuffer {
public:
    DataflowBuffer(DFBBindingToken t) : id_(static_cast<uint32_t>(t)) {}
    constexpr uint32_t id() const { return id_; }
private:
    uint32_t id_;
};
template <typename T>
class Scratchpad {
public:
    explicit Scratchpad(const ScratchpadBindingToken& t) noexcept :
        crta_offset_(t.crta_offset_), size_(t.size_in_bytes_) {}
    constexpr uint32_t size() const { return crta_offset_ + size_; }
private:
    uint32_t crta_offset_;
    uint32_t size_;
};
)";

// A kernel that treats all three resource kinds as optional parameters, written against the
// generated lookups. The static_asserts are the substance of the test: they assert that presence is
// known at compile time, which is exactly the property that lets the compiler delete the branch a
// kernel could not otherwise even name. The unbound names are the case that is a hard compile error
// today.
constexpr std::string_view OPTIONAL_PARAM_KERNEL = R"(
uint32_t kernel_main_test() {
    uint32_t acc = 0;

    constexpr auto t = try_get_tensor_binding<"in">();
    static_assert(t.has_value(), "a bound tensor must be reported present");
    LocalTensorAccessor<uint32_t> a(*t);
    acc += a.addr();

    constexpr auto d = try_get_dfb_binding<"gamma">();
    static_assert(d.has_value(), "a bound dfb must be reported present");
    DataflowBuffer dd(*d);
    acc += dd.id();

    constexpr auto s = try_get_scratchpad_binding<"tmp">();
    static_assert(s.has_value(), "a bound scratchpad must be reported present");
    Scratchpad<int> sp(*s);
    acc += sp.size();

    // Unbound names of all three kinds. Naming the token directly would not compile at all; here
    // absence is a compile-time constant, so both idioms below are well-formed and both branches
    // are dead code. `if constexpr` additionally requires the absent branch to still type-check.
    constexpr auto t_absent = try_get_tensor_binding<"not_bound">();
    static_assert(!t_absent.has_value(), "an unbound tensor name must be reported absent");
    if (t_absent.has_value()) {
        LocalTensorAccessor<uint32_t> b(*t_absent);
        acc += b.addr();
    }
    if constexpr (t_absent.has_value()) {
        LocalTensorAccessor<uint32_t> b(*t_absent);
        acc += b.addr();
    }

    constexpr auto d_absent = try_get_dfb_binding<"not_bound">();
    static_assert(!d_absent.has_value(), "an unbound dfb name must be reported absent");
    if constexpr (d_absent.has_value()) {
        DataflowBuffer dy(*d_absent);
        acc += dy.id();
    }

    constexpr auto s_absent = try_get_scratchpad_binding<"not_bound">();
    static_assert(!s_absent.has_value(), "an unbound scratchpad name must be reported absent");
    if constexpr (s_absent.has_value()) {
        Scratchpad<int> sp2(*s_absent);
        (void)sp2;
    }

    return acc;
}
)";

// End-to-end check that the emitted text is valid C++ and that presence/absence resolves at compile
// time, using the real api/optional_binding.h from the source tree. The other two headers the
// emitted block names (dataflow_buffer.h, scratchpad.h) are device-only, so the token types they
// carry are stood in for -- see TOKEN_STANDINS.
//
// Compiled with the host compiler at -std=c++20 and -fsyntax-only: what is under test is the shape
// of the emitted code and its constant-folding, not code generation, so this needs neither a device
// nor the RISC-V toolchain. C++20 is required only because the lookups take a class-type
// non-type template parameter (binding::Name); the real device build is C++17 and gets the same
// capability from the -ftt-nttp extension in JitBuildEnv's flags.
TEST(OptionalBindingLookups, EmittedLookupsCompileAndResolvePresenceAtCompileTime) {
    namespace fs = std::filesystem;

    const fs::path hw_inc = fs::path(llrt::RunTimeOptions().get_root_dir()) / "tt_metal" / "hw" / "inc";
    if (!fs::exists(hw_inc / "api" / "optional_binding.h")) {
        GTEST_SKIP() << "device headers not found under " << hw_inc << " (is TT_METAL_HOME set?)";
    }

    const fs::path dir = fs::temp_directory_path() / "tt_optional_binding_lookups_test";
    fs::remove_all(dir);
    fs::create_directories(dir);

    const fs::path src = dir / "kernel.cpp";
    {
        std::ofstream f(src);
        f << "#include <cstdint>\n"
          << "#include \"api/optional_binding.h\"\n"  // the real header; the rest is stood in for
          << TOKEN_STANDINS << BINDING_NAMESPACES << "\n"
          << format_optional_binding_lookups({"in"}, {"gamma"}, {"tmp"}) << CONSUMER_STANDINS << OPTIONAL_PARAM_KERNEL;
        ASSERT_TRUE(f.good());
    }

    const std::vector<std::string> args = {
        "c++", "-std=c++20", "-fsyntax-only", "-Wall", "-Werror", "-I", hw_inc.string(), src.string()};
    if (!exec_command(args, dir.string(), (dir / "compile.log").string())) {
        std::ifstream log(dir / "compile.log");
        const std::string output((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
        // A host compiler is not guaranteed at test runtime; a missing one is not a product failure.
        if (output.find("c++") != std::string::npos && output.find("not found") != std::string::npos) {
            GTEST_SKIP() << "no host c++ compiler available";
        }
        FAIL() << "generated optional-binding lookups failed to compile:\n" << output;
    }

    fs::remove_all(dir);
}

}  // namespace
}  // namespace tt::jit_build::utils
