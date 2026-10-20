// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"
#include <tt-metalium/circular_buffer_constants.h>
#include <tt_stl/reflection.hpp>
#include <array>
#include <stdexcept>

namespace nb = nanobind;
using namespace tt::tt_metal;

// Exported by _ttnncpp.so, although not declared in the public generic-op header.
// Reuse the framework's complete descriptor hash before adding scalar offsets.
namespace ttnn::operations::generic {
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor&);
}

ttnn::Tensor attention(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& page_table,
    const std::optional<ttnn::Tensor>& offset,
    const std::optional<int64_t>& scalar_offset,
    uint32_t q_chunk,
    uint32_t k_chunk,
    const std::string& kernel_path,
    const std::string& include_path,
    bool fp32_output_accumulator,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t math_fidelity) {
    ttnn::prim::SDPAParams attrs{};
    attrs.output_mem_config = q.memory_config();
    attrs.program_config = ttnn::operations::transformer::SDPAProgramConfig{
        .compute_with_storage_grid_size = {grid_x, grid_y},
        .q_chunk_size = q_chunk,
        .k_chunk_size = k_chunk,
        .exp_approx_mode = false};
    attrs.is_causal = true;
    attrs.chunk_start_idx = scalar_offset;
    attrs.chunk_start_idx_tensor = offset;
    attrs.compute_kernel_config = ttnn::ComputeKernelConfig{
        .math_fidelity = static_cast<tt::tt_metal::MathFidelity>(math_fidelity),
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true};
    ttnn::prim::SDPAInputs inputs{.q = q, .k = k, .v = v, .page_table = page_table, .chunk_start_idx_tensor = offset};
    using Op = ttnn::prim::SDPAOperation;
    Op::validate_on_program_cache_miss(attrs, inputs);
    auto out = Op::create_output_tensors(attrs, inputs);
    auto desc = Op::SDPAProgramFactory::create_descriptor(attrs, inputs, out);

    KernelDescriptor* compute = nullptr;
    for (auto& kernel : desc.kernels) {
        if (std::holds_alternative<ComputeConfigDescriptor>(kernel.config)) {
            compute = &kernel;
        }
    }
    if (!compute ||
        compute->kernel_source != "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp" ||
        compute->compile_time_args.size() < 48 || compute->compile_time_args[24] != 0) {
        throw std::runtime_error("Unexpected SDPA descriptor layout or streaming kernel");
    }
    // These slots are the explicitly documented CB argument layout in sdpa.cpp.
    const std::array<uint32_t, 4> ids = {
        compute->compile_time_args[29 + 15],
        compute->compile_time_args[29 + 16],
        compute->compile_time_args[29 + 11],
        compute->compile_time_args[29 + 12]};
    const std::array<const char*, 4> names = {"K2_SUM_A", "K2_SUM_B", "K2_OUT_A", "K2_OUT_B"};
    uint32_t next_id = 0;
    for (const auto& cb : desc.cbs) {
        for (const auto& format : cb.format_descriptors) {
            next_id = std::max(next_id, uint32_t(format.buffer_index) + 1);
        }
    }
    auto& config = std::get<ComputeConfigDescriptor>(compute->config);
    config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    for (uint32_t index = 0; index < ids.size(); ++index) {
        auto id = ids[index];
        uint32_t alias = id;
        bool found = false;
        for (auto& cb : desc.cbs) {
            if (cb.format_descriptors.size() != 1 || cb.format_descriptors[0].buffer_index != id) {
                continue;
            }
            found = true;
            if (index < 2 || fp32_output_accumulator) {
                auto& format = cb.format_descriptors[0];
                const uint32_t tiles = cb.total_size / format.page_size;
                format.data_format = tt::DataFormat::Float32;
                format.page_size = 4096;
                cb.total_size = tiles * format.page_size;
                alias = next_id++;
                if (alias >= 32) {
                    throw std::runtime_error("SDPA exhausted Blackhole circular buffers");
                }
                auto alias_format = format;
                alias_format.buffer_index = alias;
                cb.format_descriptors.push_back(alias_format);
                config.unpack_to_dest_mode[alias] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
            }
            break;
        }
        if (!found) {
            throw std::runtime_error("SDPA accumulator CB schema changed");
        }
        compute->defines.emplace_back(names[index], std::to_string(id));
        compute->defines.emplace_back(std::string(names[index]) + "_ALIAS", std::to_string(alias));
    }
    compute->kernel_source = kernel_path;
    compute->compiler_include_paths.emplace_back(include_path);
    // GenericOp hashes runtime-argument schemas, not scalar values. Its binding
    // fast path updates tensor addresses only. Give distinct scalar positions
    // distinct programs without changing the device-kernel compilation key.
    std::size_t program_hash = ttnn::operations::generic::compute_program_descriptor_hash(desc);
    ttsl::hash::hash_combine(program_hash, scalar_offset);
    desc.custom_program_hash = program_hash;
    std::vector<ttnn::Tensor> io{q, k, v, page_table};
    if (offset) {
        io.push_back(*offset);
    }
    io.push_back(out);
    return ttnn::generic_op(io, desc);
}

NB_MODULE(_k2_accurate_attention, m) {
    m.def(
        "attention",
        &attention,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::arg("page_table"),
        nb::arg("offset"),
        nb::arg("scalar_offset"),
        nb::arg("q_chunk"),
        nb::arg("k_chunk"),
        nb::arg("kernel_path"),
        nb::arg("include_path"),
        nb::arg("fp32_output_accumulator") = true,
        nb::arg("grid_x") = 8,
        nb::arg("grid_y") = 8,
        nb::arg("math_fidelity") = static_cast<uint32_t>(tt::tt_metal::MathFidelity::HiFi4));
}
