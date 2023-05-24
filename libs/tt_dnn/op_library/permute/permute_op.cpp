#include "tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"

using u32 = std::uint32_t;
using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor permute_(const Tensor &a, uint32_t N, uint32_t C, uint32_t H, uint32_t W) {
    if (N == 0 && C == 1 && H == 2 && W == 3) {
        return a;
    } else if (N == 0 && C == 1 && H == 3 && W == 2) {
        return transpose_wh(a);
    } else if (N == 0 && C == 2 && H == 1 && W == 3) {
        return transpose_hc(a);
    } else if (N == 0 && C == 2 && H == 3 && W == 1) {
        return transpose_wh(transpose_hc(a));
    } else if (N == 0 && C == 3 && H == 1 && W == 2) {
        return transpose_hc(transpose_wh(a));
    } else if (N == 0 && C == 3 && H == 2 && W == 1) {
        return transpose_wh(transpose_hc(transpose_wh(a)));
    } else if (N == 1 && C == 0 && H == 2 && W == 3) {
        return transpose_cn(a);
    } else if (N == 1 && C == 0 && H == 3 && W == 2) {
        return transpose_wh(transpose_cn(a));
    } else if (N == 1 && C == 2 && H == 0 && W == 3) {
        return transpose_hc(transpose_cn(a));
    } else if (N == 1 && C == 2 && H == 3 && W == 0) {
        return transpose_wh(transpose_hc(transpose_cn(a)));
    } else if (N == 1 && C == 3 && H == 0 && W == 2) {
        return transpose_hc(transpose_wh(transpose_cn(a)));
    } else if (N == 1 && C == 3 && H == 2 && W == 0) {
        return transpose_wh(transpose_hc(transpose_wh(transpose_cn(a))));
    } else if (N == 2 && C == 0 && H == 1 && W == 3) {
        return transpose_cn(transpose_hc(a));
    } else if (N == 2 && C == 0 && H == 3 && W == 1) {
        return transpose_wh(transpose_cn(transpose_hc(a)));
    } else if (N == 2 && C == 1 && H == 0 && W == 3) {
        return transpose_cn(transpose_hc(transpose_cn(a)));
    } else if (N == 2 && C == 1 && H == 3 && W == 0) {
        return transpose_wh(transpose_cn(transpose_hc(transpose_cn(a))));
    } else if (N == 2 && C == 3 && H == 0 && W == 1) {
        return transpose_hc(transpose_wh(transpose_cn(transpose_hc(a))));
    } else if (N == 2 && C == 3 && H == 1 && W == 0) {
        return transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(a)))));
    } else if (N == 3 && C == 0 && H == 1 && W == 2) {
        return transpose_cn(transpose_hc(transpose_wh(a)));
    } else if (N == 3 && C == 0 && H == 2 && W == 1) {
        return transpose_wh(transpose_cn(transpose_hc(transpose_wh(a))));
    } else if (N == 3 && C == 1 && H == 0 && W == 2) {
        return transpose_cn(transpose_hc(transpose_cn(transpose_wh(a))));
    } else if (N == 3 && C == 1 && H == 2 && W == 0) {
        return transpose_wh(transpose_cn(transpose_hc(transpose_cn(transpose_wh(a)))));
    } else if (N == 3 && C == 2 && H == 0 && W == 1) {
        return transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(a)))));
    } else if (N == 3 && C == 2 && H == 1 && W == 0) {
        return transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(a))))));
    } else {
        TT_ASSERT(false, "Illegal permute args");
    }
    return a;
}

Tensor permute(const Tensor &a, uint32_t N, uint32_t C, uint32_t H, uint32_t W) {
    Device * device;

    // Get the device
    if (a.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = a.device();
    }

    bool pad_n = H == 0 || W == 0;
    bool pad_c = H == 1 || W == 1;
    // Convert tensor back to original
    auto a_pad_shape = AutoPad::pad_to_tile_shape(a.shape(), pad_c, pad_n);
    auto out_shape = a.shape();
    out_shape = {out_shape[N], out_shape[C], out_shape[H], out_shape[W]};

    if (AutoPad::check_input_tensor_format(a, a_pad_shape)) {
        return permute_(a, N, C, H, W);
    } else {
        auto output = permute_(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), N, C, H, W);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;

    }
}

}  // namespace tt_metal

}  // namespace tt
