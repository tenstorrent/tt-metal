#include "libs/tt_dnn/op_library/groupnorm/groupnorm_op.hpp"

#include <optional>

#include "libs/tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "libs/tt_dnn/op_library/transpose/transpose_op.hpp"
#include "libs/tt_dnn/op_library/work_split.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using u32 = std::uint32_t;
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
    const MemoryConfig& mem_config) {
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
    Tensor a_cw_tx = transpose_cw(a);
    Tensor layer_norm_tx = layernorm(a_cw_tx, eps, gamma, beta, mem_config);
    Tensor output = transpose_cw(layer_norm_tx);
    return output;
}

}  // namespace tt_metal

}  // namespace tt
