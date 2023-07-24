#include <functional>
#include <type_traits>

#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"

namespace tt {
namespace tt_metal {

Tensor fully_connected_(const Tensor& act, const Tensor& weights, std::optional<std::reference_wrapper<const Tensor>> bias = std::nullopt) {
    Tensor mm_output = matmul(act, weights);
    if (bias) {
        return bcast(mm_output, bias.value(), BcastOpMath::ADD, BcastOpDim::H);
    }
    return mm_output;
}

Tensor fully_connected(const Tensor &act, const Tensor& weights, std::optional<std::reference_wrapper<const Tensor>> bias = std::nullopt) {
    TT_ASSERT(act.storage_type() == StorageType::DEVICE && weights.storage_type() == StorageType::DEVICE, "Activation and weight tensors need to be on device");
    // Assuming padding is already included. Not adding padding here.
    // NOTE: Bias is never padded.
    Device * device = act.device();
    return fully_connected_(act, weights, bias);
}

}  // namespace tt_metal
}  // namespace tt
