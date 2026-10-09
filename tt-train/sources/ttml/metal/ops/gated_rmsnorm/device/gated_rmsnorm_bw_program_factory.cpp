// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm_bw_program_factory.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

namespace {

GatedRmsNormBuffers bw_buffers(const bw::Inputs& tensor_args, const bw::tensor_return_value_t& outputs) {
    return GatedRmsNormBuffers{
        .input = tensor_args.input.buffer(),
        .gate = tensor_args.gate.buffer(),
        .gamma = tensor_args.gamma.buffer(),
        .dL_dout = tensor_args.dL_dout.buffer(),
        .out = outputs[0].value().buffer(),
        .dgate = outputs[1].value().buffer(),
        .dgamma = outputs[2].has_value() ? outputs[2]->buffer() : nullptr,
    };
}

}  // namespace

GatedRmsNormBwProgramFactory::cached_program_t GatedRmsNormBwProgramFactory::create(
    const bw::Params& args, const bw::Inputs& tensor_args, bw::tensor_return_value_t& outputs) {
    const auto geometry = validate_and_get_geometry(
        tensor_args.input, tensor_args.gate, tensor_args.gamma, tensor_args.dL_dout, "GatedRmsNormBw");

    auto* const device = tensor_args.input.device();
    const uint32_t available_l1_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    tt::tt_metal::Program program{};
    auto shared = build_gated_rmsnorm_program(
        program,
        device->compute_with_storage_grid_size(),
        available_l1_bytes,
        geometry,
        bw_buffers(tensor_args, outputs),
        GatedRmsNormProgramConfig{.epsilon = args.epsilon, .backward = true, .compute_dgamma = args.compute_dgamma});
    return cached_program_t{std::move(program), std::move(shared)};
}

void GatedRmsNormBwProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const bw::Params&,
    const bw::Inputs& tensor_args,
    bw::tensor_return_value_t& outputs) {
    override_gated_rmsnorm_addresses(
        cached_program.program, cached_program.shared_variables, bw_buffers(tensor_args, outputs));
}

}  // namespace ttml::metal::ops::gated_rmsnorm::device
