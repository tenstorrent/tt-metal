#include "relu_sq_bw.hpp"

#include "core/compute_kernel_config.hpp"
#include "device/relu_sq_bw_device_operation.hpp"  // Direct include to ensure registration is visible

namespace ttml::metal::ops::relu_sq_bw {

ttnn::Tensor ReLUSquaredBackwardOperation::invoke(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& dL_dout_tensor) {
    return ttnn::prim::ttml_relu_sq_bw(input_tensor, dL_dout_tensor);
}

}  // namespace ttml::metal::ops::relu_sq_bw
