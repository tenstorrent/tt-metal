// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"
#include <tt-metalium/circular_buffer_constants.h>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nb = nanobind;
using namespace tt::tt_metal;

// Stock split-K paged flash decode (reader, writer, work split, tree reduction)
// with FP32 scores, outputs, sums, correction factors and tree-exchange buffers.
// The generated compute kernel reads every recurrence CB through lossless
// UnpackToDestFp32 copies and merges on SFPU (accurate_decode.hpp).
ttnn::Tensor flash_decode(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& page_table,
    const ttnn::Tensor& cur_pos,
    uint32_t max_cores_per_head,
    uint32_t k_chunk,
    const std::string& kernel_path,
    const std::string& include_path) {
    ttnn::prim::SdpaDecodeParams attrs{};
    attrs.is_causal = true;
    attrs.paged_attention = true;
    attrs.output_mem_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    attrs.program_config = ttnn::operations::transformer::SDPAProgramConfig{
        .compute_with_storage_grid_size = {11, 10},
        .q_chunk_size = 0,
        .k_chunk_size = k_chunk,
        .exp_approx_mode = false,
        .max_cores_per_head_batch = max_cores_per_head};
    attrs.compute_kernel_config = ttnn::ComputeKernelConfig{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = false};
    attrs.k_chunk_size = k_chunk;
    ttnn::prim::SdpaDecodeInputs inputs{
        .q = q, .k = k, .v = v, .cur_pos_tensor = cur_pos, .page_table_tensor = page_table};
    using Op = ttnn::prim::SdpaDecodeDeviceOperation;
    Op::validate_on_program_cache_miss(attrs, inputs);
    auto out = Op::create_output_tensors(attrs, inputs);
    auto desc = Op::create_descriptor(attrs, inputs, out);

    KernelDescriptor* compute = nullptr;
    KernelDescriptor* writer = nullptr;
    KernelDescriptor* reader = nullptr;
    for (auto& kernel : desc.kernels) {
        if (kernel.kernel_source ==
            "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp") {
            compute = &kernel;
        } else if (
            kernel.kernel_source ==
            "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp") {
            reader = &kernel;
        } else if (
            kernel.kernel_source ==
            "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp") {
            writer = &kernel;
        }
    }
    // Compute CT 17..21: causal, no mask, no sink, static K chunk, tiled Q; CT 27: speculative
    // multi-position mode off (its mask path is written for the stock BF16 recurrence).
    if (!compute || !writer || !reader || compute->compile_time_args.size() != 28 ||
        compute->compile_time_args[17] != 1 || compute->compile_time_args[18] != 0 ||
        compute->compile_time_args[19] != 0 || compute->compile_time_args[21] != 0 ||
        compute->compile_time_args[27] != 0 || writer->compile_time_args.size() < 22 ||
        reader->compile_time_args.size() < 35 || reader->compile_time_args[21] != 0 ||
        reader->compile_time_args[23] != compute->compile_time_args[23]) {
        throw std::runtime_error("Unexpected SDPA decode descriptor layout");
    }
    // Eight local query heads select 16x32 half tiles, which the FP32 unpack/pack
    // path does not support. Promote every half-tile CB to full 32x32 tiles, as the
    // stock factory does for more than 16 heads; reader/compute use_half_tile=0.
    if (compute->compile_time_args[23] == 1) {
        const uint32_t q_tiles = reader->compile_time_args[1] * reader->compile_time_args[3];  // PNHt * DHt
        if (reader->compile_time_args[24] != q_tiles * 1024) {
            throw std::runtime_error("Unexpected half-tile Q chunk size");
        }
        for (auto& cb : desc.cbs) {
            for (auto& format : cb.format_descriptors) {
                if (format.tile && format.tile->height == 16 && format.tile->width == 32) {
                    if (cb.buffer != nullptr || cb.format_descriptors.size() != 1 || format.face_geometry) {
                        throw std::runtime_error("Unexpected half-tile CB layout");
                    }
                    const uint32_t tiles = cb.total_size / format.page_size;
                    format.tile = TileDescriptor(32, 32, false);
                    format.page_size *= 2;
                    cb.total_size = tiles * format.page_size;
                }
            }
        }
        reader->compile_time_args[23] = 0;
        reader->compile_time_args[24] = q_tiles * 2048;
        compute->compile_time_args[23] = 0;
    }
    const uint32_t cores_per_head = writer->compile_time_args[12];
    const uint32_t rounds = cores_per_head > 1 ? 32 - __builtin_clz(cores_per_head - 1) : 0;

    // FP32 storage for every accumulator; maxima (c_27/c_28), masks, scalars, Q/K/V and the
    // BF16 output keep stock formats. Exchange CBs c_6/c_7/c_16..c_19 share one tile size,
    // as the writer sizes every l/m/o transfer from c_16 and c_19.
    const std::unordered_set<uint32_t> fp32 = {6, 7, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 29, 30, 31};
    // Recurrence state read only by A2D copies in the generated kernel (never by FPU).
    const std::unordered_set<uint32_t> lossless = {7, 16, 21, 22, 23, 25, 26, 29, 30, 31};
    std::unordered_map<uint32_t, uint32_t> tiles;
    std::unordered_set<uint32_t> seen;
    for (auto& cb : desc.cbs) {
        if (cb.format_descriptors.size() != 1) {
            continue;
        }
        auto& format = cb.format_descriptors[0];
        const uint32_t id = format.buffer_index;
        if (!fp32.count(id)) {
            continue;
        }
        if (format.data_format != tt::DataFormat::Float16_b || cb.buffer != nullptr ||
            cb.total_size % format.page_size != 0) {
            throw std::runtime_error("Unexpected SDPA decode intermediate CB " + std::to_string(id));
        }
        tiles[id] = cb.total_size / format.page_size;
        format.data_format = tt::DataFormat::Float32;
        format.page_size *= 2;  // BF16 -> FP32, same tile shape
        cb.total_size = tiles[id] * format.page_size;
        seen.insert(id);
    }
    if (seen.size() + (cores_per_head > 1 ? 0 : 1) != fp32.size()) {
        throw std::runtime_error("SDPA decode intermediate CB schema changed");
    }
    // Stock sizes c_19 for cores_per_head - 1 child blocks, but children write at
    // their send round, and a core receives at most one child per round.
    if (cores_per_head > 1) {
        const uint32_t block = tiles.at(16) + 2 * tiles.at(17);
        if (tiles.at(17) != tiles.at(18) || tiles.at(19) != block * (cores_per_head - 1)) {
            throw std::runtime_error("SDPA decode tree-exchange CB schema changed");
        }
        for (auto& cb : desc.cbs) {
            if (cb.format_descriptors.size() == 1 && cb.format_descriptors[0].buffer_index == 19) {
                cb.total_size = rounds * block * cb.format_descriptors[0].page_size;
            }
        }
    }
    auto& config = std::get<ComputeConfigDescriptor>(compute->config);
    if (!config.fp32_dest_acc_en) {
        throw std::runtime_error("Accurate flash decode requires FP32 destination accumulation");
    }
    config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    for (const auto id : lossless) {
        config.unpack_to_dest_mode[id] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    compute->kernel_source = kernel_path;
    compute->compiler_include_paths.emplace_back(include_path);
    return ttnn::generic_op({q, k, v, cur_pos, page_table, out}, desc);
}

NB_MODULE(_k2_accurate_attention, m) {
    m.def(
        "flash_decode",
        &flash_decode,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::arg("page_table"),
        nb::arg("cur_pos"),
        nb::arg("max_cores_per_head"),
        nb::arg("k_chunk"),
        nb::arg("kernel_path"),
        nb::arg("include_path"));
}
