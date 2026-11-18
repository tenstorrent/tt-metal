// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "row_softmax_program.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <memory>
#include <string>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <variant>
#include <vector>

#include "metal/common/program_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::row_softmax {
namespace {
constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/log_softmax/device/kernels/dataflow/reader_log_softmax.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/log_softmax/device/kernels/dataflow/writer_log_softmax.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/log_softmax/device/kernels/compute/log_softmax_kernel.cpp";
// Semaphores on every core of the program: a row's chunk-0 core counts arrivals, then releases the others.
constexpr uint32_t kArriveSemaphore = 0U;
constexpr uint32_t kGoSemaphore = 1U;

constexpr uint32_t kMinChunkTiles = 16;  // Below this the exchange costs more than the extra cores save.
// Statistics in one core's packed partial record (log_softmax_tile::kPartialStatistics), a column vector each.
constexpr uint32_t kPartialStatistics = 4;
// Bytes the reader fetches around each row's target logit: one DRAM-aligned block.
constexpr uint32_t kTargetSlotBytes = 64;

// Tile geometry for the kernels; the program hash keys the tile through the input spec.
std::map<std::string, std::string> tile_defines(const tt::tt_metal::Tile& tile) {
    const auto face = tile.get_face_shape();
    return {
        {"LOG_SOFTMAX_TILE_HEIGHT", std::to_string(tile.get_height())},
        {"LOG_SOFTMAX_TILE_WIDTH", std::to_string(tile.get_width())},
        {"LOG_SOFTMAX_FACE_HEIGHT", std::to_string(face[0])},
        {"LOG_SOFTMAX_FACE_WIDTH", std::to_string(face[1])}};
}

bool is_backward(Kind kind) {
    return kind == Kind::LogSoftmaxBackward || kind == Kind::CrossEntropyBackward;
}
bool is_cross_entropy(Kind kind) {
    return kind == Kind::CrossEntropyForward || kind == Kind::CrossEntropyBackward;
}
}  // namespace

bool supports_tile(const tt::tt_metal::Tile& tile) {
    return tile == tt::tt_metal::Tile();
}

bool can_fuse_upstream(const ttnn::Tensor& input, const ttnn::Tensor& grad) {
    if (grad.storage_type() != ttnn::StorageType::DEVICE || grad.layout() != ttnn::Layout::TILE ||
        (grad.dtype() != ttnn::DataType::BFLOAT16 && grad.dtype() != ttnn::DataType::FLOAT32) ||
        grad.memory_config() != ttnn::DRAM_MEMORY_CONFIG || grad.device() != input.device() ||
        grad.tensor_spec().tile() != input.tensor_spec().tile()) {
        return false;
    }
    if (grad.logical_shape() == ttnn::Shape({1, 1, 1, 1})) {
        return true;
    }
    auto per_row_shape = input.logical_shape();
    per_row_shape[-1] = 1U;
    return grad.logical_shape() == per_row_shape && grad.padded_shape()[-2] == input.padded_shape()[-2];
}

// Few rows: split each row's columns over several cores. Every core then holds exactly one (row, chunk)
// unit, so the per-row exchange never waits on a core that is busy with another row.
Layout layout_for(const ttnn::Tensor& input) {
    const auto& tile = input.tensor_spec().tile();
    const uint32_t width_tiles = input.padded_shape()[-1] / tile.get_width();
    const uint32_t rows = input.physical_volume() / (input.padded_shape()[-1] * tile.get_height());
    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t cores = grid.x * grid.y;
    uint32_t chunks = 1;
    if (rows * 2 <= cores) {
        // Partial statistics are exchanged as the lanes (columns) of one tile.
        chunks = std::min({tile.get_width(), cores / rows, std::max(1U, width_tiles / kMinChunkTiles)});
    }
    const uint32_t chunk_tiles = (width_tiles + chunks - 1) / chunks;
    chunks = (width_tiles + chunk_tiles - 1) / chunk_tiles;
    return {width_tiles, rows, chunks, chunk_tiles};
}

std::optional<tt::tt_metal::WorkloadBuffer> make_scratch(const ttnn::Tensor& input) {
    const auto layout = layout_for(input);
    if (layout.chunks == 1) {
        return std::nullopt;
    }
    // One tile-sized page per (row, chunk) unit; a packed record fills the start of it.
    const auto& tile = input.tensor_spec().tile();
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({1, 1, tile.get_height(), layout.rows * layout.chunks * tile.get_width()}),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::FLOAT32, tt::tt_metal::PageConfig(ttnn::Layout::TILE, tile), ttnn::DRAM_MEMORY_CONFIG));
    // The workload keeps the tensor alive: its destructor frees the device storage.
    auto owner = std::make_shared<ttnn::Tensor>(ttnn::create_device_tensor(spec, input.device()));
    auto* buffer = owner->buffer();
    return tt::tt_metal::WorkloadBuffer{.owner = std::move(owner), .buffer = buffer};
}

tt::tt_metal::WorkloadDescriptor replicated_workload(
    tt::tt_metal::ProgramDescriptor descriptor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    std::optional<tt::tt_metal::WorkloadBuffer> scratch) {
    tt::tt_metal::WorkloadDescriptor workload;
    if (scratch.has_value()) {
        workload.buffers.push_back(std::move(*scratch));
    }
    for (const auto& range : tensor_coords.ranges()) {
        workload.programs.push_back({.range = range, .descriptor = descriptor});
    }
    return workload;
}

tt::tt_metal::ProgramDescriptor make_descriptor(
    const Options& options, const ttnn::Tensor& input, const Buffers& buffers) {
    using namespace tt::tt_metal;
    using RuntimeArg = std::variant<uint32_t, Buffer*>;
    auto* device = input.device();
    const bool backward = is_backward(options.kind);
    const bool cross_entropy = is_cross_entropy(options.kind);
    const auto layout = layout_for(input);
    const uint32_t width_tiles = layout.width_tiles, rows = layout.rows, chunks = layout.chunks;
    const auto grid = device->compute_with_storage_grid_size();
    // With chunks > 1 a unit is one (row, chunk) pair and every core takes at most one.
    auto [cores, all, group1, group2, units1, units2] = split_work_to_cores(grid, rows * chunks);
    TT_FATAL(chunks == 1 || (units1 == 1 && units2 <= 1), "Split rows need one unit per core");
    const auto& tile = input.tensor_spec().tile();
    const auto fmt = datatype_to_dataformat_converter(input.dtype());
    const auto native_bytes = tile.get_tile_size(fmt), fp32_bytes = tile.get_tile_size(tt::DataFormat::Float32);
    // Stream rings hold two blocks of the largest block size; each core pushes blocks of 4, 2 or 1 tiles,
    // whichever divides its tile count, so every push stays aligned to the ring.
    constexpr uint32_t block = 4;
    const bool ce_upstream = options.kind == Kind::CrossEntropyBackward && options.upstream_kind != 0;
    const auto grad_dtype =
        options.kind == Kind::LogSoftmaxBackward || ce_upstream ? options.upstream_dtype : input.dtype();
    const auto grad_fmt = datatype_to_dataformat_converter(grad_dtype);
    // Bounded, double-buffered streams; scratch size is independent of vocabulary width.
    ProgramDescriptor program;
    const auto add_cb = [&](uint32_t cb, tt::DataFormat format, uint32_t page_bytes, uint32_t pages) {
        auto descriptor = make_cb_descriptor(all, cb, format, page_bytes, pages);
        descriptor.format_descriptors[0].tile = TileDescriptor(tile);
        program.cbs.push_back(std::move(descriptor));
    };
    add_cb(0, fmt, native_bytes, 2 * block);
    add_cb(1, grad_fmt, tile.get_tile_size(grad_fmt), 2 * block);
    add_cb(16, fmt, native_bytes, 2 * block);
    for (uint32_t cb : {3U, 4U, 5U, 6U, 7U, 10U, 11U, 12U, 13U, 14U, 15U, 17U, 18U, 19U, 20U, 21U, 22U})
        add_cb(cb, tt::DataFormat::Float32, fp32_bytes, 1);
    // 2: the column-index tile for the compute kernel, then four pages the reader keeps for its constant tiles
    // (-inf, +inf, -inf in the input's format, zero in the gradient's), NoC-copied wherever a tile needs a constant.
    add_cb(2, tt::DataFormat::Float32, fp32_bytes, 5);
    // Staging for the packed partial records of up to one core per lane (tile column).
    const uint32_t record_bytes = kPartialStatistics * tile.get_height() * sizeof(float);
    add_cb(24, tt::DataFormat::Float32, fp32_bytes, (tile.get_width() * record_bytes + fp32_bytes - 1) / fp32_bytes);
    for (uint32_t cb : {8U, 9U, 23U}) add_cb(cb, tt::DataFormat::Float32, fp32_bytes, 3);
    if (cross_entropy) {
        // 25: target logits (column 0) and 26: row targets (one per physical row of a tile row) for the writer,
        // double-buffered so the reader can start the next row; 27: column-0 mask; 28: aligned scratch for the
        // target-logit reads (one slot per physical row plus alignment slack).
        add_cb(25, tt::DataFormat::Float32, fp32_bytes, 2);
        add_cb(26, tt::DataFormat::Float32, tile.get_height() * sizeof(uint32_t), 2);
        add_cb(27, tt::DataFormat::Float32, fp32_bytes, 1);
        add_cb(28, tt::DataFormat::Float32, (tile.get_height() + 1) * kTargetSlotBytes, 1);
    }
    if (ce_upstream) {
        // 29: per-row upstream gradient (column 0), 30: the raw upstream tile it is taken from.
        add_cb(29, tt::DataFormat::Float32, fp32_bytes, 2);
        add_cb(30, grad_fmt, tile.get_tile_size(grad_fmt), 1);
    }
    program.semaphores.push_back(make_semaphore_descriptor(kArriveSemaphore, all));
    program.semaphores.push_back(make_semaphore_descriptor(kGoSemaphore, all));
    const uint32_t arrive_sem = kArriveSemaphore, go_sem = kGoSemaphore;

    auto defines = tile_defines(tile);
    if (backward) {
        defines["BACKWARD"] = "1";
    }
    if (cross_entropy) {
        defines["CROSS_ENTROPY"] = "1";
    }
    if (ce_upstream) {
        defines["UPSTREAM_GRAD"] = std::to_string(options.upstream_kind);
    }
    // The scratch accessor is always compiled in; without a split it aliases the input and is never used.
    auto* scratch = chunks > 1 ? buffers.scratch : buffers.input;
    TT_FATAL(scratch != nullptr, "Row-split programs need an exchange scratch buffer");
    auto* state = buffers.state != nullptr ? buffers.state : buffers.input;
    auto* target = buffers.target != nullptr ? buffers.target : buffers.input;
    const uint32_t width = input.logical_shape()[-1];
    std::vector<uint32_t> ra{
        width_tiles,
        block,
        width,
        input.dtype() == ttnn::DataType::FLOAT32,
        grad_dtype == ttnn::DataType::FLOAT32,
        chunks,
        arrive_sem,
        go_sem,
        options.target_page_bytes,
        options.target_tiled_height};
    TensorAccessorArgs(buffers.input).append_to(ra);
    if (options.kind == Kind::LogSoftmaxBackward) {
        TensorAccessorArgs(buffers.upstream).append_to(ra);
        TensorAccessorArgs(state).append_to(ra);
    }
    TensorAccessorArgs(scratch).append_to(ra);
    if (cross_entropy) {
        TensorAccessorArgs(target).append_to(ra);
    }
    if (ce_upstream) {
        TensorAccessorArgs(buffers.upstream).append_to(ra);
    }
    std::vector<uint32_t> wa{width_tiles, block, input.dtype() == ttnn::DataType::FLOAT32, chunks, arrive_sem};
    TensorAccessorArgs(buffers.output).append_to(wa);
    TensorAccessorArgs(state).append_to(wa);
    TensorAccessorArgs(scratch).append_to(wa);
    auto reader = make_reader_kernel_descriptor(all, ra, defines, kReaderKernelPath);
    auto writer = make_writer_kernel_descriptor(all, wa, defines, kWriterKernelPath);
    // Route FP32 loads directly into DST: the FPU copy path truncates to TF32.
    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb = 2; cb < NUM_CIRCULAR_BUFFERS; ++cb) unpack_modes[cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[16] = UnpackToDestMode::Default;
    if (input.dtype() == ttnn::DataType::FLOAT32) {
        unpack_modes[0] = UnpackToDestMode::UnpackToDestFp32;
    }
    if (grad_dtype == ttnn::DataType::FLOAT32) {
        unpack_modes[1] = UnpackToDestMode::UnpackToDestFp32;
    }
    auto compute = make_compute_kernel_descriptor(
        all,
        {chunks, std::bit_cast<uint32_t>(options.scaler)},
        defines,
        kComputeKernelPath,
        /* fp32_dest_acc_en */ true,
        std::move(unpack_modes));

    std::vector<CoreCoord> unit_cores;
    for_each_core_with_work(
        cores, grid.y, group1, group2, units1, units2, [&](const CoreWork& w) { unit_cores.push_back(w.core); });
    for_each_core_with_work(cores, grid.y, group1, group2, units1, units2, [&](const CoreWork& w) {
        // Without a split, a core takes num_units whole rows; with one, its single unit is (row, chunk).
        const uint32_t row = chunks > 1 ? w.start / chunks : w.start;
        const uint32_t chunk = chunks > 1 ? w.start % chunks : 0;
        const uint32_t count = chunks > 1 ? 1 : w.num_units;
        const uint32_t begin = chunk * layout.chunk_tiles;
        const uint32_t end = std::min(width_tiles, begin + layout.chunk_tiles);
        const uint32_t core_block = (end - begin) % 4 == 0 ? 4 : ((end - begin) % 2 == 0 ? 2 : 1);
        // The row's chunk-0 core gathers arrivals; without a split nobody signals it.
        const auto leader = device->worker_core_from_logical_core(unit_cores[chunks > 1 ? row * chunks : w.index]);
        std::vector<RuntimeArg> reader_args{
            buffers.input,
            buffers.upstream != nullptr ? RuntimeArg{buffers.upstream} : RuntimeArg{0U},
            state,
            scratch,
            count,
            row,
            begin,
            end,
            chunk,
            core_block,
            target};
        if (chunks > 1 && chunk == 0) {
            for (uint32_t g = 0; g < chunks; ++g) {
                const auto member = device->worker_core_from_logical_core(unit_cores[w.index + g]);
                reader_args.emplace_back(static_cast<uint32_t>(member.x));
                reader_args.emplace_back(static_cast<uint32_t>(member.y));
            }
        }
        reader.emplace_runtime_args(w.core, reader_args);
        writer.emplace_runtime_args(
            w.core,
            {buffers.output,
             state,
             scratch,
             count,
             row,
             begin,
             end,
             chunk,
             static_cast<uint32_t>(leader.x),
             static_cast<uint32_t>(leader.y),
             core_block});
        compute.emplace_runtime_args(w.core, {count, begin, end});
    });
    // Kernel handles in the built program follow this order.
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

}  // namespace ttml::metal::ops::row_softmax
