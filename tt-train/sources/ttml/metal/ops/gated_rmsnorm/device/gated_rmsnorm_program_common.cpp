// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm_program_common.hpp"

#include <bit>
#include <cstdint>
#include <map>
#include <string>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <vector>

#include "kernels/gated_rmsnorm_cbs.hpp"
#include "metal/common/program_utils.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/dataflow/reader_gated_rmsnorm.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/dataflow/writer_gated_rmsnorm.cpp";
constexpr auto kFwComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/compute/gated_rmsnorm_fw.cpp";
constexpr auto kBwComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/compute/gated_rmsnorm_bw.cpp";

// Runtime-arg slots holding buffer addresses, rewritten on program-cache hits.
constexpr uint32_t kReaderInputIdx = 0U;
constexpr uint32_t kReaderGateIdx = 1U;
constexpr uint32_t kReaderGammaIdx = 2U;
constexpr uint32_t kReaderDyIdx = 3U;
constexpr uint32_t kWriterOutIdx = 0U;
constexpr uint32_t kWriterDgateIdx = 1U;
constexpr uint32_t kWriterDgammaIdx = 2U;

namespace cb = ttml_gated_rmsnorm_cb;

uint32_t address_or_zero(const tt::tt_metal::Buffer* buffer) {
    return buffer != nullptr ? buffer->address() : 0U;
}

struct CbPlan {
    uint32_t index;
    tt::DataFormat format;
    uint32_t pages;
};

std::vector<CbPlan> plan_circular_buffers(const uint32_t Gt, const GatedRmsNormProgramConfig& config) {
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    std::vector<CbPlan> plan{
        {cb::x, bf16, 2U * Gt},
        {cb::gate, bf16, 2U * Gt},
        {cb::gamma_b, bf16, Gt},
        {cb::ones, bf16, 1U},
        {cb::sq, f32, 1U},
        {cb::inv, f32, 1U},
        {cb::out, bf16, 2U * Gt},
    };
    if (config.backward) {
        plan.push_back({cb::dy, bf16, 2U * Gt});
        plan.push_back({cb::u, f32, Gt});
        plan.push_back({cb::du, f32, Gt});
        plan.push_back({cb::prod, f32, Gt});
        plan.push_back({cb::acc, f32, 1U});
        plan.push_back({cb::t, f32, 1U});
        plan.push_back({cb::dgate, bf16, 2U * Gt});
        if (config.compute_dgamma) {
            plan.push_back({cb::dgamma, bf16, 2U * Gt});
        }
    }
    return plan;
}

tt::tt_metal::KernelHandle create_gated_rmsnorm_compute_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    const std::vector<uint32_t>& compile_args,
    const std::map<std::string, std::string>& defines,
    const bool backward) {
    // fp32 CBs that are only ever copied into DEST skip SrcA, which would round them to tf32.
    // sq and acc feed the matmul through SrcB and must stay on the default path.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    for (const uint32_t index : {cb::inv, cb::u, cb::du, cb::prod, cb::t}) {
        unpack_to_dest_mode[index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    return tt::tt_metal::CreateKernel(
        program,
        backward ? kBwComputeKernelPath : kFwComputeKernelPath,
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = compile_args,
            .defines = defines});
}

}  // namespace

GatedRmsNormSharedVariables build_gated_rmsnorm_program(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& grid_size,
    const uint32_t available_l1_bytes,
    const GatedRmsNormGeometry& geometry,
    const GatedRmsNormBuffers& buffers,
    const GatedRmsNormProgramConfig& config) {
    const uint32_t Gt = geometry.group_tiles;
    const uint32_t num_cores_y = grid_size.y;

    const auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid_size, geometry.total_work());

    const auto cb_plan = plan_circular_buffers(Gt, config);
    uint64_t required_l1_bytes = 0U;
    for (const auto& entry : cb_plan) {
        required_l1_bytes += static_cast<uint64_t>(entry.pages) * tt::tile_size(entry.format);
    }
    TT_FATAL(
        required_l1_bytes <= available_l1_bytes,
        "GatedRmsNorm: group width V = {} needs {} B of circular buffers per core, but only {} B of L1 is available",
        geometry.group,
        required_l1_bytes,
        available_l1_bytes);
    for (const auto& entry : cb_plan) {
        create_circular_buffer(program, all_cores, entry.index, entry.format, tt::tile_size(entry.format), entry.pages);
    }

    std::map<std::string, std::string> defines;
    if (config.backward) {
        defines["BACKWARD"] = "1";
        if (config.compute_dgamma) {
            defines["COMPUTE_DGAMMA"] = "1";
        }
    }

    const std::vector<uint32_t> layout_args{Gt, geometry.width_tiles, geometry.num_groups};

    std::vector<uint32_t> reader_ct_args = layout_args;
    tt::tt_metal::TensorAccessorArgs(buffers.input).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(buffers.gate).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(buffers.gamma).append_to(reader_ct_args);
    if (config.backward) {
        tt::tt_metal::TensorAccessorArgs(buffers.dL_dout).append_to(reader_ct_args);
    }

    std::vector<uint32_t> writer_ct_args = layout_args;
    tt::tt_metal::TensorAccessorArgs(buffers.out).append_to(writer_ct_args);
    if (config.backward) {
        tt::tt_metal::TensorAccessorArgs(buffers.dgate).append_to(writer_ct_args);
        if (config.compute_dgamma) {
            tt::tt_metal::TensorAccessorArgs(buffers.dgamma).append_to(writer_ct_args);
        }
    }

    GatedRmsNormSharedVariables shared{};
    shared.reader_kernel_id = create_reader_kernel(program, all_cores, reader_ct_args, defines, kReaderKernelPath);
    shared.writer_kernel_id = create_writer_kernel(program, all_cores, writer_ct_args, defines, kWriterKernelPath);
    shared.num_cores = num_cores;
    shared.num_cores_y = num_cores_y;

    const uint32_t inv_group_bits = std::bit_cast<uint32_t>(1.0F / static_cast<float>(geometry.group));
    const uint32_t eps_bits = std::bit_cast<uint32_t>(config.epsilon);
    create_gated_rmsnorm_compute_kernel(
        program, core_group_1, {work_per_core_1, Gt, inv_group_bits, eps_bits}, defines, config.backward);
    if (!core_group_2.ranges().empty()) {
        create_gated_rmsnorm_compute_kernel(
            program, core_group_2, {work_per_core_2, Gt, inv_group_bits, eps_bits}, defines, config.backward);
    }

    for (uint32_t i = 0, work_start = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t work_count = 0U;
        if (core_group_1.contains(core)) {
            work_count = work_per_core_1;
        } else if (core_group_2.contains(core)) {
            work_count = work_per_core_2;
        } else {
            TT_FATAL(false, "GatedRmsNorm: core {} is not in either work group", core.str());
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.reader_kernel_id,
            core,
            {address_or_zero(buffers.input),
             address_or_zero(buffers.gate),
             address_or_zero(buffers.gamma),
             address_or_zero(buffers.dL_dout),
             work_start,
             work_count});
        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.writer_kernel_id,
            core,
            {address_or_zero(buffers.out),
             address_or_zero(buffers.dgate),
             address_or_zero(buffers.dgamma),
             work_start,
             work_count});

        work_start += work_count;
    }

    return shared;
}

void override_gated_rmsnorm_addresses(
    tt::tt_metal::Program& program, const GatedRmsNormSharedVariables& shared, const GatedRmsNormBuffers& buffers) {
    auto& reader_rt = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_rt = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel_id);
    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / shared.num_cores_y, i % shared.num_cores_y};
        auto& reader_args = reader_rt[core.x][core.y];
        reader_args[kReaderInputIdx] = address_or_zero(buffers.input);
        reader_args[kReaderGateIdx] = address_or_zero(buffers.gate);
        reader_args[kReaderGammaIdx] = address_or_zero(buffers.gamma);
        reader_args[kReaderDyIdx] = address_or_zero(buffers.dL_dout);
        auto& writer_args = writer_rt[core.x][core.y];
        writer_args[kWriterOutIdx] = address_or_zero(buffers.out);
        writer_args[kWriterDgateIdx] = address_or_zero(buffers.dgate);
        writer_args[kWriterDgammaIdx] = address_or_zero(buffers.dgamma);
    }
}

}  // namespace ttml::metal::ops::gated_rmsnorm::device
