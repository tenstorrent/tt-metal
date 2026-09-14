// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <type_traits>
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

// Exercise the shared decoder with typed resources and a value-based coordinate view,
// independently of the ProgramSpec host adapter. Test data still comes from positional args.
namespace dataflow_kernel_lib {
template <uint32_t RT_BASE>
struct TestMcastRuntime {
    uint32_t offset;
    static uint32_t read(uint32_t index) { return get_arg_val<uint32_t>(RT_BASE + index); }
    static TestMcastRuntime coordinates(uint32_t index) { return {index}; }
    uint32_t operator[](uint32_t index) const { return read(offset + index); }
};

template <uint32_t ID>
constexpr auto test_mcast_binding() {
    if constexpr (ID == UNUSED_SEM_ID) {
        return nullptr;
    } else {
        return SemaphoreBindingToken<ID, SemScope::LOCAL_NONATOMIC>{};
    }
}

// Check propagation of every native mechanism without pretending BH executes Quasar paths.
static_assert(std::is_same_v<
              decltype(detail::make_mcast_semaphore<SemaphoreBindingToken<0, SemScope::EXTERNAL>{}>()),
              Semaphore<ProgrammableCoreType::TENSIX, SemScope::EXTERNAL>>);
static_assert(std::is_same_v<
              decltype(detail::make_mcast_semaphore<SemaphoreBindingToken<0, SemScope::DM_LOCAL_CACHED>{}>()),
              Semaphore<ProgrammableCoreType::TENSIX, SemScope::DM_LOCAL_CACHED>>);
static_assert(std::is_same_v<decltype(detail::make_mcast_semaphore<nullptr>()), std::nullptr_t>);

template <uint32_t CT_BASE, uint32_t RT_BASE>
struct McastTestArgs
    : detail::McastArgsImpl<
          (get_compile_time_arg_val(CT_BASE + mcast_wire::TAG) == mcast_wire::FAMILY),
          detail::positional_mcast_metadata<CT_BASE>(),
          TestMcastRuntime<RT_BASE>,
          test_mcast_binding<detail::positional_mcast_semaphore<CT_BASE, mcast_wire::DATA_READY>()>(),
          test_mcast_binding<detail::positional_mcast_semaphore<CT_BASE, mcast_wire::CONSUMER_READY>()>(),
          test_mcast_binding<detail::positional_mcast_semaphore<CT_BASE, mcast_wire::SIGNAL_SOURCE>()>()> {
    static constexpr uint32_t next_compile_time_args_offset() {
        return McastArgs<CT_BASE, RT_BASE>::next_compile_time_args_offset();
    }
    static constexpr uint32_t next_runtime_args_offset() {
        return McastArgs<CT_BASE, RT_BASE>::next_runtime_args_offset();
    }
};
}  // namespace dataflow_kernel_lib
