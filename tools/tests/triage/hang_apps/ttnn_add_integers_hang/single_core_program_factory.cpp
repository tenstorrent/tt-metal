// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_integers_hang_op.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

namespace triage_hang_apps {

using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor AddIntegersHangOperation::SingleCore::create_descriptor(
    const operation_attributes_t&, const tensor_args_t& tensor_args, tensor_return_value_t& tensor_return_value) {
    const auto& a_tensor = tensor_args.input_tensor_a;
    const auto& b_tensor = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    auto* src0_buffer = a_tensor.buffer();
    auto* src1_buffer = b_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(a_tensor.dtype());
    uint32_t single_tile_size = tile_size(cb_data_format);
    tt::DataFormat cb_data_format_output = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    constexpr CoreCoord core = {0, 0};
    CoreRangeSet core_set(CoreRange(core, core));

    ProgramDescriptor desc;

    constexpr uint32_t num_tiles = 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_0,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_1,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * single_tile_size_output,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_16,
            .data_format = cb_data_format_output,
            .page_size = single_tile_size_output,
        }}},
    });

    // Reuse the sibling add_2_integers_hang kernels. CB indices c_0/c_1/c_16 are chosen
    // above to match the constants those kernels expect.
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "tools/tests/triage/hang_apps/add_2_integers_hang/kernels/dataflow/reader_binary_1_tile.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_set;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.emplace_back(
        core, KernelDescriptor::CoreRuntimeArgs{src0_buffer->address(), src1_buffer->address()});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "tools/tests/triage/hang_apps/add_2_integers_hang/kernels/dataflow/writer_1_tile.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_set;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{dst_buffer->address()});

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "tools/tests/triage/hang_apps/add_2_integers_hang/kernels/compute/add_2_tiles_hang.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_set;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace triage_hang_apps
