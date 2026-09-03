// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

namespace {

constexpr std::uint32_t kernel_owned_arg = get_compile_time_arg_val(0);
constexpr std::uint32_t reduce_args_offset = 1;
constexpr std::uint32_t call_count = get_compile_time_arg_val(reduce_args_offset);
constexpr std::uint32_t first_call_args_offset =
    reduce_args_offset + ttnn::kernel_lib::reduce_plan_args::call_count_word_count;

static_assert(kernel_owned_arg == 17, "The reduce args must preserve the kernel-owned prefix");
static_assert(call_count > 0, "A planned reduction sequence must contain at least one call");

template <std::uint32_t CallIndex>
using CallAt = ttnn::kernel_lib::ReduceCallAtT<first_call_args_offset, CallIndex>;

template <typename Call>
constexpr std::uint32_t input_tile_count() {
    constexpr std::uint32_t row_pitch = Call::row_stride == 0 ? Call::columns : Call::row_stride;
    return Call::rows * row_pitch * Call::batches;
}

template <typename Call>
ALWI void make_streamed_input_available() {
    if constexpr (Call::input_policy != compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop) {
        DataflowBuffer input(Call::input_cb_id);
        input.reserve_back(input_tile_count<Call>());
        input.push_back(input_tile_count<Call>());
    }
}

template <typename Call>
ALWI void release_caller_owned_input() {
    if constexpr (Call::input_policy == compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop) {
        DataflowBuffer input(Call::input_cb_id);
        input.pop_front(input_tile_count<Call>());
    }
}

template <std::uint32_t CallIndex = 0>
ALWI void issue_calls() {
    if constexpr (CallIndex < call_count) {
        using Call = CallAt<CallIndex>;
        make_streamed_input_available<Call>();
        compute_kernel_lib::reduce<Call>();
        release_caller_owned_input<Call>();

        // A fused kernel may perform arbitrary work here before issuing the
        // next independently decoded call.
        issue_calls<CallIndex + 1>();
    }
}

}  // namespace

void kernel_main() {
    using First = CallAt<0>;
    constexpr std::uint32_t startup_src_b = First::algorithm == compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd
                                                ? First::input_cb_id
                                                : First::auxiliary_cb_id;
    compute_kernel_hw_startup(First::input_cb_id, startup_src_b, First::output_cb_id);
    issue_calls();
}
