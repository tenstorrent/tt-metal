#include "libs/tt_dnn/op_library/groupnorm/groupnorm_op.hpp"

#include <optional>

#include "libs/tt_dnn/op_library/work_split.hpp"
#include "libs/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "libs/tt_dnn/op_library/composite/composite_ops.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

Tensor groupnorm(
    const Tensor& a,
    uint32_t group_size,
    float eps,
    std::optional<const Tensor> gamma,
    std::optional<const Tensor> beta,
    const MemoryConfig& output_mem_config) {
    TT_ASSERT(a.shape()[3] % TILE_WIDTH == 0, "Normalizing on last dim cannot be padded");
    if (gamma.has_value()) {
        TT_ASSERT(gamma.value().shape()[3] == a.shape()[3], "Gamma width must be equal to input width");
    }
    if (beta.has_value()) {
        TT_ASSERT(beta.value().shape()[3] == a.shape()[3], "Beta width must be equal to input width");
    }

    TT_ASSERT(group_size == 1 && "group norm size is only supported for size = 1");
    /**
     * shortcut when group size = 1 we use layernorm with transpose and non-transpose
     */

    Shape shape = a.shape();
    Tensor ar = reshape(const_cast<Tensor&>(a),shape[0],1,shape[1]*shape[2],shape[3],output_mem_config);
    Tensor group_norm_1 = normalize_hw(ar,output_mem_config);
    Tensor output = reshape (group_norm_1,shape[0],shape[1],shape[2],shape[3],output_mem_config);
    if (gamma.has_value() && beta.has_value()) {
        output = mac(output,gamma.value(),beta.value(),output_mem_config);
    } else {
        if (gamma.has_value()) {
            output = mul(output,gamma.value()); //gamma_t);
        } else if (beta.has_value()) {
            output = add(output,beta.value());
        }
    }
    return output;
}

}  // namespace tt_metal

}  // namespace tt
