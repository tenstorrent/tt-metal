// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Metal 2.0 kernel argument accessors.
//
// This header defines the device-side machinery for named kernel arguments:
//   - RtaArg<T> / CrtaArg<T> / CtaVal<T> accessor structs
//   - RtaArrayArg<T, N> / CrtaArrayArg<T, N> accessor structs (CTA arrays are CtaVal<std::array<T, N>>)
//   - ArgArrayView<T, N> / CtaArrayView<T, N>: what get_arg() returns for an array argument
//   - get_arg() overloads, one per accessor kind
//
// The accessor constants themselves (e.g., `args::input_dim`) are emitted per-kernel into
// kernel_args_generated.h. That generated header is included before the user kernel source,
// so kernel code can write:
//
//   auto dim = get_arg(args::input_dim);   // RTA
//   auto cnt = get_arg(args::tile_count);  // CRTA
//   auto bsz = get_arg(args::block_size);  // CTA (compile-time constant)
//
// The kernel source is identical regardless of whether an arg is dispatched via RTA, CRTA,
// or CTA — moving an arg between kinds only requires a host-side schema change.
// (Then get_arg() call is resolved via ADL on the accessor type's `experimental` namespace.)
//
// NOTE: Currently, only uint32_t args are supported. However, named kernel arguments via
// get_arg() will later be extended to support arbitrary POD types.
//
// DEPENDENCY NOTE:
// This header requires the following to be visible at the point of inclusion:
//  - `get_arg_addr`
//  - `get_common_arg_addr`
//  - `get_arg_val<T>`
//  - `get_common_arg_val<T>`
//  - `FORCE_INLINE` macros
//  - `tt_l1_ptr` macros
// On the DM path, those come from `api/dataflow/dataflow_api.h`, which the firmware wrapper
// (brisck.cc, ncrisck.cc, etc.) includes before <kernel_includes.hpp>.
// On the TRISC path, those come from `api/compute/common.h`, but nothing pulls that in before
// <kernel_includes.hpp>. We need to manually include here.
#ifdef COMPILE_FOR_TRISC
#include "api/compute/common.h"
#endif

#include <array>
#include <cstddef>
#include <cstdint>

#include "api/debug/assert.h"

// TT_KERNEL: marks the named-arg entry point; the JIT generates kernel_main() from its signature.
// `static` gives the entry internal linkage (it is private to the kernel TU — only the generated
// kernel_main() shim, in the same TU, ever calls it); FORCE_INLINE folds it into that shim.
#define TT_KERNEL static FORCE_INLINE

namespace experimental {

// byte_offset is measured from the start of the *named* section of the dispatch buffer.
// Varargs live after the named section; see get_vararg() in kernel_args_generated.h.
template <typename T>
struct RtaArg {
    uint32_t byte_offset;
};

template <typename T>
struct CrtaArg {
    uint32_t byte_offset;
};

template <typename T>
struct CtaVal {
    T value;
};

template <typename T>
FORCE_INLINE T get_arg(RtaArg<T> arg) {
    static_assert(sizeof(T) == 4, "Only uint32_t args are currently supported.");
    return *((tt_l1_ptr T*)(get_arg_addr(arg.byte_offset / sizeof(uint32_t))));
}

template <typename T>
FORCE_INLINE T get_arg(CrtaArg<T> arg) {
    static_assert(sizeof(T) == 4, "Only uint32_t args are currently supported.");
    return *((tt_l1_ptr T*)(get_common_arg_addr(arg.byte_offset / sizeof(uint32_t))));
}

// CTAs are compile-time constants, so this syntax is legal for CTAs (but not RTAs/CRTAs):
//   constexpr auto x = get_arg(args::x);
template <typename T>
FORCE_INLINE constexpr T get_arg(CtaVal<T> arg) {
    return arg.value;
}

//------------------------------------------------
// Array arguments
//------------------------------------------------
//
// An array argument holds N elements, read by index. As with scalar arguments, kernel code
// is the same whether the array is dispatched as an RTA, a CRTA, or a CTA:
//
//   auto addrs = get_arg(args::sem_addrs);
//   for (uint32_t i = 0; i < addrs.size(); ++i) { use(addrs[i]); }
//   for (uint32_t addr : get_arg(args::sem_addrs)) { use(addr); }
//
// get_arg() returns a *view* of the array for every kind: a small object that refers to the
// elements where they already are, and reads one only when it is indexed. A view is cheap to
// copy, so `auto` is always cheap. (Returning a std::array by value would copy every element
// onto the stack before the first one is read.)

// byte_offset is the offset of element 0, measured as for RtaArg / CrtaArg.
template <typename T, uint32_t N>
struct RtaArrayArg {
    uint32_t byte_offset;
};

template <typename T, uint32_t N>
struct CrtaArrayArg {
    uint32_t byte_offset;
};

// A read-only view of an RTA or CRTA array, in place in the dispatch buffer.
template <typename T, uint32_t N>
class ArgArrayView {
public:
    explicit FORCE_INLINE ArgArrayView(uintptr_t base_addr) : base_(reinterpret_cast<tt_l1_ptr T*>(base_addr)) {}

    FORCE_INLINE T operator[](uint32_t index) const {
        // Watcher's own check (in get_arg_addr) covers only the whole args buffer, so an index
        // past this array that lands on a neighbouring argument would go unnoticed without this.
        ASSERT(index < N);
        return base_[index];
    }
    static constexpr uint32_t size() { return N; }

    // Range-for support
    FORCE_INLINE tt_l1_ptr T* begin() const { return base_; }
    FORCE_INLINE tt_l1_ptr T* end() const { return base_ + N; }

private:
    tt_l1_ptr T* base_;
};

// Called when a CTA array index is out of range. ASSERT may expand to inline asm, which a C++17
// constexpr function cannot contain, so CtaArrayView calls it through this function instead.
// A side effect: an out-of-range index in a constant expression is a compile error.
inline void cta_array_index_out_of_range() {
    ASSERT(false);  // Attempt to access a CTA array element out of range.
}

// A read-only view of a CTA array. The elements are compile-time constants, and so is the view:
//   constexpr uint32_t p = get_arg(args::page_sizes)[2];
template <typename T, uint32_t N>
class CtaArrayView {
public:
    constexpr explicit CtaArrayView(const std::array<T, N>& values) : values_(&values) {}

    FORCE_INLINE constexpr T operator[](uint32_t index) const {
        if (index >= N) {
            cta_array_index_out_of_range();
        }
        return (*values_)[index];
    }
    static constexpr uint32_t size() { return N; }

    // Range-for support
    constexpr const T* begin() const { return values_->data(); }
    constexpr const T* end() const { return values_->data() + N; }

private:
    const std::array<T, N>* values_;
};

template <typename T, uint32_t N>
FORCE_INLINE ArgArrayView<T, N> get_arg(RtaArrayArg<T, N> arg) {
    static_assert(sizeof(T) == 4, "Only uint32_t args are currently supported.");
    return ArgArrayView<T, N>(get_arg_addr(arg.byte_offset / sizeof(uint32_t)));
}

template <typename T, uint32_t N>
FORCE_INLINE ArgArrayView<T, N> get_arg(CrtaArrayArg<T, N> arg) {
    static_assert(sizeof(T) == 4, "Only uint32_t args are currently supported.");
    return ArgArrayView<T, N>(get_common_arg_addr(arg.byte_offset / sizeof(uint32_t)));
}

// CTA arrays are emitted as CtaVal<std::array<T, N>>. This overload is chosen over the scalar
// get_arg(CtaVal<T>) because its parameter is more specialized.
// (N is std::size_t to match std::array: deduction requires the exact type.)
template <typename T, std::size_t N>
FORCE_INLINE constexpr CtaArrayView<T, N> get_arg(const CtaVal<std::array<T, N>>& arg) {
    return CtaArrayView<T, N>(arg.value);
}

}  // namespace experimental
