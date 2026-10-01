// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/reflection.hpp>
#include <array>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nb = nanobind;
using namespace tt::tt_metal;

// Exported by _ttnncpp.so, although not declared in the public generic-op header.
// Reuse the framework's complete descriptor hash before adding scalar offsets.
namespace ttnn::operations::generic {
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor&);
}

namespace {
// Each request contributes eight Q-head work units on the fixed 64-core grid.
// B16/32 give each core two/four heads, all from one request, so its existing
// scalar offset remains valid throughout the unchanged kernel loop. Updating
// both the address and BufferBinding keeps generic-op cache hits replay-safe.
std::vector<uint32_t> bind_request_offsets(
    ProgramDescriptor& desc, const std::vector<ttnn::Tensor>& offsets, const ttnn::Tensor& q, bool packed_gqa) {
    const auto batch = offsets.size();
    const uint32_t query_heads = packed_gqa ? 2 : 8;
    const auto shape = q.logical_shape();
    if (!(batch == 2 || batch == 4 || batch == 8 || batch == 16 || batch == 32) || shape.size() != 4 ||
        shape[0] != batch || shape[1] != query_heads || shape[2] != 32 || shape[3] != 128) {
        throw std::runtime_error("Batched accurate attention has unsupported Q geometry");
    }
    const auto& first = offsets.front();
    const auto accessor = TensorAccessorArgs(first.buffer());
    std::unordered_set<Buffer*> owners;
    for (const auto& offset : offsets) {
        const auto offset_shape = offset.logical_shape();
        if (offset_shape.size() != 1 || offset_shape[0] != 1 || offset.dtype() != DataType::INT32 ||
            offset.storage_type() != ttnn::StorageType::DEVICE || offset.device() != q.device() ||
            offset.tensor_spec() != first.tensor_spec()) {
            throw std::runtime_error("Batched accurate attention requires compatible device scalar offsets");
        }
        const auto current_accessor = TensorAccessorArgs(offset.buffer());
        if (current_accessor.get_compile_time_args() != accessor.get_compile_time_args() ||
            current_accessor.get_common_runtime_args() != accessor.get_common_runtime_args() ||
            !owners.insert(offset.buffer()).second) {
            throw std::runtime_error("Batched accurate attention requires distinct compatible scalar offset owners");
        }
    }
    KernelDescriptor* reader = nullptr;
    for (auto& kernel : desc.kernels) {
        if (kernel.kernel_source ==
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp") {
            if (reader != nullptr) {
                throw std::runtime_error("Unexpected duplicate SDPA reader");
            }
            reader = &kernel;
        }
    }
    if (!reader) {
        throw std::runtime_error("Expected pinned SDPA reader for per-request offsets");
    }
    const auto& ct = reader->compile_time_args;
    // Causal zigzag is enabled, but with one Q chunk its permutation is identity.
    if (ct.size() < 34 || ct[0] != batch || ct[1] != query_heads || ct[2] != 2 || ct[3] != 2 || ct[9] != 1 ||
        ct[10] != 1 || ct[11] != 4 || ct[13] != 64 || ct[14] != 1 || ct[19] != 1 || ct[27] != 0 || ct[32] != 1 ||
        ct[33] != 0) {
        throw std::runtime_error("Pinned accurate attention reader geometry changed");
    }
    std::vector<uint32_t> mapping;
    uint32_t total_work = 0;
    for (auto& [core, args] : reader->runtime_args) {
        if (args.size() != 16 || args[7] != 1) {
            throw std::runtime_error("Pinned causal one-phase reader runtime layout changed");
        }
        const auto start = args[10];
        const auto count = args[11];
        // Even cores without Q work execute scalar-offset setup; retain a valid
        // owner there. B8/16/32 give every core exactly one/two/four Q heads.
        const auto row = count ? start / query_heads : 0;
        if (count > std::max(size_t{1}, batch * query_heads / 64) || row >= batch ||
            (count && (start + count - 1) / query_heads != row)) {
            throw std::runtime_error("Accurate attention core work crosses request boundaries");
        }
        total_work += count;
        auto* buffer = offsets[row].buffer();
        uint32_t matches = 0;
        for (auto& binding : reader->buffer_bindings) {
            if (binding.core == core && binding.arg_idx == 6) {
                binding.buffer = buffer;
                ++matches;
            }
        }
        if (matches != 1) {
            throw std::runtime_error("Expected one scalar-offset BufferBinding per SDPA reader core");
        }
        args[6] = buffer->address();
        mapping.insert(
            mapping.end(), {static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y), start, count, row});
    }
    if (total_work != batch * query_heads || reader->runtime_args.size() != 64) {
        throw std::runtime_error("Incomplete accurate attention per-core request mapping");
    }
    return mapping;
}
}  // namespace

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
    const std::vector<ttnn::Tensor>& offsets,
    bool packed_gqa,
    const std::string& packed_writer_path,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t math_fidelity) {
    if (packed_gqa && (offsets.size() < 2 || q.dtype() != DataType::BFLOAT16 || k.logical_shape()[1] != 2 ||
                       v.logical_shape()[1] != 2 || q_chunk != 32 || k_chunk != 128 || packed_writer_path.empty())) {
        throw std::runtime_error("Packed GQA requires BF16 Q, two KV heads, B>=2 raw tensor positions and Q32/K128");
    }
    if (!packed_gqa && !packed_writer_path.empty()) {
        throw std::runtime_error("Packed writer is only valid for explicit packed GQA decode");
    }
    if (!offsets.empty() &&
        (scalar_offset || !offset || offset->buffer() != offsets.front().buffer() || !fp32_output_accumulator)) {
        throw std::runtime_error("Batched accurate attention requires its first tensor offset and FP32 recurrence");
    }
    // Per-request offset binding is pinned to the 64-core grid; single-offset calls may use any grid.
    if (!offsets.empty() && (grid_x != 8 || grid_y != 8)) {
        throw std::runtime_error("Batched accurate attention requires the 8x8 grid");
    }
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
    const auto request_mapping =
        offsets.empty() ? std::vector<uint32_t>{} : bind_request_offsets(desc, offsets, q, packed_gqa);
    if (packed_gqa) {
        uint32_t writers = 0;
        for (auto& kernel : desc.kernels) {
            if (kernel.kernel_source !=
                "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp") {
                continue;
            }
            const auto& ct = kernel.compile_time_args;
            if (ct.size() < 22 || ct[1] != 2 || ct[5] != 1 || ct[6] != 1 || ct[7] != 4 || ct[11] != 1 || ct[12] != 0 ||
                ct[14] != 1 || ct[15] != 0 || ct[16] != 1 || ct[17] != 0 || ct[21] != 0) {
                throw std::runtime_error("Packed GQA requires the pinned causal Q32/K128 nonstream writer");
            }
            kernel.kernel_source = packed_writer_path;
            kernel.compiler_include_paths.emplace_back(include_path);
            ++writers;
        }
        if (writers != 1) {
            throw std::runtime_error("Packed GQA requires exactly one private writer");
        }
    }
    // GenericOp hashes runtime-argument schemas, not scalar values. Its binding
    // fast path updates tensor addresses only. Give distinct scalar positions
    // distinct programs without changing the device-kernel compilation key.
    std::size_t program_hash = ttnn::operations::generic::compute_program_descriptor_hash(desc);
    ttsl::hash::hash_combine(program_hash, scalar_offset);
    if (!offsets.empty()) {
        // GenericOp's hash omits BufferBinding mappings and runtime values.
        // Separate this transport schema from the original scalar-offset path.
        ttsl::hash::hash_combine(
            program_hash,
            std::string(packed_gqa ? "k2_packed_gqa4_raw_position_v1" : "k2_batched_accurate_offsets_b32_v2"));
        ttsl::hash::hash_combine(program_hash, offsets.size());
        for (const auto value : request_mapping) {
            ttsl::hash::hash_combine(program_hash, value);
        }
    }
    desc.custom_program_hash = program_hash;
    std::vector<ttnn::Tensor> io{q, k, v, page_table};
    if (!offsets.empty()) {
        io.insert(io.end(), offsets.begin(), offsets.end());
    } else if (offset) {
        io.push_back(*offset);
    }
    io.push_back(out);
    return ttnn::generic_op(io, desc);
}

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
    // Compute CT 17..21: causal, no mask, no sink, static K chunk, tiled Q.
    if (!compute || !writer || !reader || compute->compile_time_args.size() != 27 ||
        compute->compile_time_args[17] != 1 || compute->compile_time_args[18] != 0 ||
        compute->compile_time_args[19] != 0 || compute->compile_time_args[21] != 0 ||
        writer->compile_time_args.size() < 22 || reader->compile_time_args.size() < 35 ||
        reader->compile_time_args[21] != 0 || reader->compile_time_args[23] != compute->compile_time_args[23]) {
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
        nb::arg("offsets") = std::vector<ttnn::Tensor>{},
        nb::arg("packed_gqa") = false,
        nb::arg("packed_writer_path") = std::string{},
        nb::arg("grid_x") = 8,
        nb::arg("grid_y") = 8,
        nb::arg("math_fidelity") = static_cast<uint32_t>(tt::tt_metal::MathFidelity::HiFi4));
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
