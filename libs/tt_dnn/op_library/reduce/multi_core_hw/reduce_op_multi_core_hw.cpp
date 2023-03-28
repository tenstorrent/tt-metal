#include "libs/tt_dnn/op_library/reduce/reduce_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;
using u32 = std::uint32_t;

namespace tt {

namespace tt_metal {

Tensor reduce_multi_core_hw(const Tensor &a, ReduceOpMath::Enum reduce_op, ReduceOpDim::Enum reduce_dim, float scaler) {
    TT_ASSERT(reduce_dim == ReduceOpDim::HW);
    TT_ASSERT(scaler == 1.0f && "ReduceHW currently only works correctly with scaler == 1.0f!");
    tt_metal::Tensor output = reduce_multi_core_w(a, reduce_op, ReduceOpDim::W, scaler);
    output = reduce_multi_core_h(output, reduce_op, ReduceOpDim::H, scaler);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
