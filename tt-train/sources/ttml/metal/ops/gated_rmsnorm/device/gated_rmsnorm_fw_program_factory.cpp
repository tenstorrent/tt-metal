// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm_fw_program_factory.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

namespace {

GatedRmsNormBuffers fw_buffers(const fw::Inputs& tensor_args, const fw::tensor_return_value_t& output) {
    return GatedRmsNormBuffers{
        .input = tensor_args.input.buffer(),
        .gate = tensor_args.gate.buffer(),
        .gamma = tensor_args.gamma.buffer(),
        .out = output.buffer(),
    };
}

}  // namespace

GatedRmsNormFwProgramFactory::cached_program_t GatedRmsNormFwProgramFactory::create(
    const fw::Params& args, const fw::Inputs& tensor_args, fw::tensor_return_value_t& output) {
    const auto geometry = validate_and_get_geometry(
        tensor_args.input, tensor_args.gate, tensor_args.gamma, std::nullopt, "GatedRmsNormFw");

    auto* const device = tensor_args.input.device();
    const uint32_t available_l1_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    tt::tt_metal::Program program{};
    auto shared = build_gated_rmsnorm_program(
        program,
        device->compute_with_storage_grid_size(),
        available_l1_bytes,
        geometry,
        fw_buffers(tensor_args, output),
        GatedRmsNormProgramConfig{.epsilon = args.epsilon, .backward = false, .compute_dgamma = false});
    return cached_program_t{std::move(program), std::move(shared)};
}

void GatedRmsNormFwProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const fw::Params&,
    const fw::Inputs& tensor_args,
    fw::tensor_return_value_t& output) {
    override_gated_rmsnorm_addresses(
        cached_program.program, cached_program.shared_variables, fw_buffers(tensor_args, output));
}

}  // namespace ttml::metal::ops::gated_rmsnorm::device
