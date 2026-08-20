// SPDX-License-Identifier: Apache-2.0
// PROBE A (compute half): compute_kernel_lib::reduce<> driven entirely from dfb:: tokens.
// Template params 3/4/5 are `std::uint32_t` non-type template parameters; the tokens' implicit
// constexpr operator uint32_t() has to survive a *converted constant expression*.
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);

    compute_kernel_hw_startup(dfb::in_tiles, dfb::scaler, dfb::out_tiles);

    compute_kernel_lib::reduce<
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW,
        dfb::in_tiles,   // <-- DFBBindingToken in NTTP position
        dfb::scaler,     // <-- ditto
        dfb::out_tiles,  // <-- ditto
        compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, 1));
}
