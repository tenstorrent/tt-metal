// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_fw_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

// Dual-NOC architecture:
//   RISCV_1 / NOC0: X reader + Y writer (runs on ALL cores)
//   RISCV_0 / NOC1: Weight sender (sender core) or Weight receiver (receiver cores)
constexpr auto kXReaderYWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/dataflow/reader_x_writer_y_swiglu_fw.cpp";

constexpr auto kWeightSenderKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/dataflow/reader_swiglu_fw_sender.cpp";

constexpr auto kWeightReceiverKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/dataflow/reader_swiglu_fw_receiver.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/compute/swiglu_fw_kernel.cpp";

// X reader + Y writer runtime arg indices
constexpr uint32_t kXReaderXAddrIdx = 0;
constexpr uint32_t kXReaderYAddrIdx = 1;

// Weight sender runtime arg indices
constexpr uint32_t kWSenderW1AddrIdx = 0;
constexpr uint32_t kWSenderW2AddrIdx = 1;
constexpr uint32_t kWSenderW3AddrIdx = 2;

// CBs with input data
constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kW1CbIndex = tt::CBIndex::c_1;
constexpr auto kW2CbIndex = tt::CBIndex::c_2;
constexpr auto kW3CbIndex = tt::CBIndex::c_3;
// CBs with intermediate computations (used when row of M fits in L1 with flash-attention optimization)
constexpr auto kXW1PartialCbIndex = tt::CBIndex::c_4;  // keeps track of partial (X @ W1) [r, k]
constexpr auto kXW3PartialCbIndex = tt::CBIndex::c_5;  // keeps track of partial (X @ W3) [r, k]
constexpr auto kXW1CbIndex = tt::CBIndex::c_6;         // keeps track of (X @ W1) [r, k]
constexpr auto kXW3CbIndex = tt::CBIndex::c_7;         // keeps track of (X @ W3) [r, k]
constexpr auto kMCbIndex = tt::CBIndex::c_8;           // keeps track of M[r, k]
constexpr auto kYPartialCbIndex = tt::CBIndex::c_9;    // keeps track of partial Y[r, c]
// CB with output data
constexpr auto kYCbIndex = tt::CBIndex::c_10;  // keeps track of final Y[r, c]

const std::string kRowOfMFitsInL1DefineKey = "ROW_OF_M_FITS_IN_L1";

}  // namespace

namespace ttml::metal::ops::swiglu_fw::device {

struct SwiGLUForwardKernels {
    tt::tt_metal::KernelHandle x_reader_y_writer;  // RISCV_1: X reader + Y writer (all cores)
    tt::tt_metal::KernelHandle weight_sender;      // RISCV_0: Weight sender (sender core)
    tt::tt_metal::KernelHandle weight_receiver;    // RISCV_0: Weight receiver (receiver cores)
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const SwiGLUForwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* w1,
    const tt::tt_metal::Buffer* w2,
    const tt::tt_metal::Buffer* w3,
    const tt::tt_metal::Buffer* swiglu_buffer,
    const uint32_t num_cores,
    const uint32_t num_cores_x,
    const uint32_t num_cores_y,
    const uint32_t num_rows_per_core_group_1,
    const uint32_t num_rows_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    ttnn::IDevice* device,
    const uint32_t mcast_sender_semaphore_id,
    const uint32_t mcast_receiver_semaphore_id,
    const bool use_multicast,
    const uint32_t max_rows_across_all_cores) {
    // DUAL-NOC ARCHITECTURE:
    //   RISCV_1 / NOC0 (all cores): X reader + Y writer
    //   RISCV_0 / NOC1 (sender core): Weight reader + multicast
    //   RISCV_0 / NOC1 (receiver cores): Weight receiver

    // Compute multicast bounding box (physical coordinates)
    uint32_t mcast_start_physical_x = 0;
    uint32_t mcast_start_physical_y = 0;
    uint32_t mcast_end_physical_x = 0;
    uint32_t mcast_end_physical_y = 0;
    uint32_t num_receivers_excluding_sender = 0;

    if (use_multicast && num_cores > 1) {
        uint32_t max_x = 0;
        uint32_t max_y = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            tt::tt_metal::CoreCoord core = {i % num_cores_x, i / num_cores_x};
            max_x = std::max(max_x, static_cast<uint32_t>(core.x));
            max_y = std::max(max_y, static_cast<uint32_t>(core.y));
        }
        auto mcast_start_physical = device->worker_core_from_logical_core({0, 0});
        auto mcast_end_physical = device->worker_core_from_logical_core({max_x, max_y});
        mcast_start_physical_x = mcast_start_physical.x;
        mcast_start_physical_y = mcast_start_physical.y;
        mcast_end_physical_x = mcast_end_physical.x;
        mcast_end_physical_y = mcast_end_physical.y;
        num_receivers_excluding_sender = num_cores - 1;
    }

    auto sender_physical = device->worker_core_from_logical_core({0, 0});

    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i % num_cores_x, i / num_cores_x};

        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        uint32_t max_rows_for_sync = use_multicast ? max_rows_across_all_cores : num_rows_per_core;
        bool is_sender = (core.x == 0 && core.y == 0);

        // --- RISCV_1: X reader + Y writer (ALL cores, same args) ---
        SetRuntimeArgs(
            program,
            kernels.x_reader_y_writer,
            core,
            {input_buffer->address(),
             swiglu_buffer->address(),
             num_rows_per_core,
             max_rows_for_sync,
             num_rows_written});

        // --- RISCV_0: Weight sender or receiver ---
        if (is_sender) {
            SetRuntimeArgs(
                program,
                kernels.weight_sender,
                core,
                {w1->address(),
                 w2->address(),
                 w3->address(),
                 max_rows_for_sync,
                 mcast_start_physical_x,
                 mcast_start_physical_y,
                 mcast_end_physical_x,
                 mcast_end_physical_y,
                 num_receivers_excluding_sender,
                 mcast_sender_semaphore_id,
                 mcast_receiver_semaphore_id});
        } else if (use_multicast) {
            SetRuntimeArgs(
                program,
                kernels.weight_receiver,
                core,
                {max_rows_for_sync,
                 static_cast<uint32_t>(sender_physical.x),
                 static_cast<uint32_t>(sender_physical.y),
                 mcast_sender_semaphore_id,
                 mcast_receiver_semaphore_id});
        } else {
            // No multicast (single core): use sender kernel with no receivers
            SetRuntimeArgs(
                program,
                kernels.weight_sender,
                core,
                {w1->address(),
                 w2->address(),
                 w3->address(),
                 max_rows_for_sync,
                 0,
                 0,
                 0,
                 0,  // unused mcast coords
                 0,  // num_receivers = 0
                 mcast_sender_semaphore_id,
                 mcast_receiver_semaphore_id});
        }

        num_rows_written += num_rows_per_core;
    }
}

bool row_of_m_fits_in_l1_check(
    const uint32_t hidden_Wt,
    const uint32_t block_size,
    const uint32_t bfloat16_single_tile_size_bytes,
    ttnn::IDevice* device) {
    const uint32_t twice_block_size = 2U * block_size;

    // Get available L1 memory
    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // Calculate memory requirements for "M fits in L1" algorithm
    // This algorithm caches full rows of XW1, XW3, and M in L1 for better performance
    // and uses flash-attention optimization to reduce X memory reads
    const uint32_t hidden_Wt_rounded_up = ((hidden_Wt + block_size - 1U) / block_size) * block_size;

    // Memory for input CBs (always needed regardless of algorithm)
    const uint64_t input_memory = twice_block_size * bfloat16_single_tile_size_bytes;  // cb_input
    // W1/W2/W3 use batched mcast which needs larger CB (2 × block_size^2 tiles)
    const uint64_t w1_memory = (2U * block_size * block_size) * bfloat16_single_tile_size_bytes;  // cb_w1
    const uint64_t w2_memory = (2U * block_size * block_size) * bfloat16_single_tile_size_bytes;  // cb_w2
    const uint64_t w3_memory = (2U * block_size * block_size) * bfloat16_single_tile_size_bytes;  // cb_w3

    // Memory for output CBs (always needed regardless of algorithm)
    const uint64_t y_partial_memory = twice_block_size * bfloat16_single_tile_size_bytes;  // cb_y_partial
    const uint64_t y_memory = twice_block_size * bfloat16_single_tile_size_bytes;          // cb_y

    // Additional memory ONLY needed for "M fits in L1" algorithm
    const uint64_t xw1_partial_memory = twice_block_size * bfloat16_single_tile_size_bytes;  // cb_xw1_partial
    const uint64_t xw3_partial_memory = twice_block_size * bfloat16_single_tile_size_bytes;  // cb_xw3_partial
    const uint64_t xw1_memory = hidden_Wt_rounded_up * bfloat16_single_tile_size_bytes;      // cb_xw1 (full row)
    const uint64_t xw3_memory = hidden_Wt_rounded_up * bfloat16_single_tile_size_bytes;      // cb_xw3 (full row)
    const uint64_t m_memory = hidden_Wt_rounded_up * bfloat16_single_tile_size_bytes;        // cb_m (full row)

    // Total L1 memory required for "M fits in L1" algorithm
    const uint64_t required_L1_in_bytes = input_memory + w1_memory + w2_memory + w3_memory + y_partial_memory +
                                          y_memory + xw1_partial_memory + xw3_partial_memory + xw1_memory + xw3_memory +
                                          m_memory;

    return required_L1_in_bytes <= available_L1_in_bytes;
}

SwiGLUForwardProgramFactory::cached_program_t SwiGLUForwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& w1 = tensor_args.w1;
    const auto& w2 = tensor_args.w2;
    const auto& w3 = tensor_args.w3;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    auto padded_tensor_shape = input.padded_shape();
    auto padded_tensor_volume = input.physical_volume();
    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    // Get the num_inner and hidden_num_inner
    uint32_t num_inner = input.logical_shape()[-1];
    uint32_t hidden_num_inner = w1.logical_shape()[-1];

    // Validate weight shapes for fused SwiGLU:
    // Expected layout: W1[embed, hidden], W3[embed, hidden], W2[hidden, embed]
    // This is TRANSPOSED compared to standard LinearLayer which stores [out, in]
    const auto& w1_shape = w1.logical_shape();
    const auto& w2_shape = w2.logical_shape();
    const auto& w3_shape = w3.logical_shape();

    TT_FATAL(
        w1_shape[-2] == num_inner,
        "W1 shape mismatch: W1[-2]={} must equal input[-1]={}. "
        "Fused SwiGLU expects transposed weights [embed, hidden], not [hidden, embed].",
        w1_shape[-2],
        num_inner);
    TT_FATAL(
        w3_shape[-2] == num_inner && w3_shape[-1] == hidden_num_inner,
        "W3 shape mismatch: W3={} must match W1={}. Both should be [embed, hidden].",
        w3_shape,
        w1_shape);
    TT_FATAL(
        w2_shape[-2] == hidden_num_inner && w2_shape[-1] == num_inner,
        "W2 shape mismatch: W2[-2]={}, W2[-1]={} must be [hidden={}, embed={}].",
        w2_shape[-2],
        w2_shape[-1],
        hidden_num_inner,
        num_inner);

    // Get the hidden dimension // 32
    uint32_t hidden_Wt = hidden_num_inner / tt::constants::TILE_WIDTH;

    // These parameters are used to determine if we need to mask tiles along input/hidden dimension, i.e. if the
    // operation applied over input/hidden dimension might produce incorrect results due to some random data in the
    // end of the last tile.
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;
    uint32_t mask_hw = hidden_num_inner % tt::constants::TILE_WIDTH;

    // TODO(maciek): Consider adding masking. Now we assume that the N and C % 32 == 0.
    TT_FATAL(mask_w == 0, "Input inner dimension must be multiple of TILE_WIDTH");
    TT_FATAL(mask_hw == 0, "Hidden inner dimension must be multiple of TILE_WIDTH");

    // Get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Compile arguments
    // We enforce to use block_size of 4. If num_inner % 4 != 0 or hidden_num_inner % 4 != 0, we will take care of
    // it in the kernels.
    const uint32_t block_size = 4U;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process, /*row_wise=*/true);

    // -------------------------------------------------------------------------
    // 1.5) Calculate max rows for multicast synchronization
    // -------------------------------------------------------------------------
    // SINGLE-SENDER MULTICAST SYNCHRONIZATION:
    // Core (0,0) reads weights from DRAM and multicasts to ALL other cores.
    // This requires all cores to participate in the same number of multicast operations.
    //
    // All cores loop for max_rows_across_all_cores iterations:
    // 1. Sender (0,0) reads X for its current row, then reads+multicasts W1/W2/W3
    // 2. All receivers read X for their current row, wait for W1/W2/W3 multicast
    // 3. Cores with fewer actual rows use padding iterations (read last valid row)
    // 4. Compute/Writer only process actual rows (num_rows_per_core)
    //
    // This enables single-point DRAM reads for weights → massive bandwidth savings.
    bool use_multicast = (num_cores > 1);
    uint32_t max_rows_across_all_cores = num_rows_per_core_group_1;  // group_1 always has >= group_2 rows

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;

    // Check if row of M fits in L1 - required for the optimized algorithm.
    // The M-fits-L1 algorithm caches XW1[r,:], XW3[r,:], and M[r,:] in L1 for flash-attention
    // optimization. This requires approximately 3 × hidden_Wt × tile_bytes of L1.
    //
    // L1 threshold analysis (Wormhole B0, ~1.46 MB available L1):
    //   Max hidden_dim ≈ 6,848 elements (214 tiles)
    //
    // With tensor parallelism (TP), hidden_dim is sharded across devices:
    //   - Llama-7B  (hidden=11008): fits with TP≥2
    //   - Llama-13B (hidden=13824): fits with TP≥4
    //   - Llama-70B (hidden=28672): fits with TP≥8
    //   - Llama-405B (hidden=53248): fits with TP≥8
    //
    // In all practical distributed training scenarios, M will fit in L1.
    // If this limit is exceeded, consider using composite ops (matmul + silu + mul + matmul)
    // as a fallback, though this would be ~2x slower.
    const bool row_of_m_fits_in_l1 =
        row_of_m_fits_in_l1_check(hidden_Wt, block_size, bfloat16_single_tile_size_bytes, device);

    TT_FATAL(
        row_of_m_fits_in_l1,
        "SwiGLU fused kernel requires M row to fit in L1. hidden_dim={} tiles ({} elements) exceeds L1 capacity. "
        "For large models, use tensor parallelism (TP) to shard hidden_dim across devices. "
        "Alternatively, use composite ops: matmul(silu(matmul(x,w1)) * matmul(x,w3), w2).",
        hidden_Wt,
        hidden_Wt * 32);

    // W1/W3 CB size: use batched mcast (block_size rows × block_size tiles per batch)
    // Plus double-buffering: 2 × block_size^2 = 32 tiles for block_size=4
    const uint32_t w1_w3_cb_tiles = 2U * block_size * block_size;

    // W2 CB size: use batched mcast (block_size cols × block_size tiles per batch)
    // Same size as W1/W3 for consistency
    const uint32_t w2_cb_tiles = 2U * block_size * block_size;

    // CB sizing for M-fits-L1 algorithm (full row caching)
    const uint32_t num_tiles_xw1 = ((hidden_Wt + block_size - 1U) / block_size) * block_size;
    const uint32_t num_tiles_xw3 = num_tiles_xw1;
    const uint32_t num_tiles_m = num_tiles_xw1;

    auto data_format = input_data_format;  // tt::DataFormat::Float16_b

    // NOTE(maciek):
    // - fp32 input/output CBs are possible, but here both are always bf16 to match pipeline formats.
    // - matmul_tiles seems to require matching input/output CB dtypes, otherwise NaNs/infs may occur.
    //   TODO(maciek): make minimal repro and report if this is a bug. See tracking issue #32529.
    // - Matmul runs on FPU; with fp32_dest_acc_en, accumulation is fp32.
    // - Using all CBs as fp32 showed no observable precision improvement in tests.
    [[maybe_unused]] auto cb_input = create_circular_buffer(
        program, all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    // W1/W3 CBs use larger size when batching is enabled for reduced mcast overhead
    [[maybe_unused]] auto cb_w1 = create_circular_buffer(
        program, all_cores, kW1CbIndex, data_format, bfloat16_single_tile_size_bytes, w1_w3_cb_tiles);
    // W2 CB uses same batched size as W1/W3 for reduced mcast overhead
    [[maybe_unused]] auto cb_w2 = create_circular_buffer(
        program, all_cores, kW2CbIndex, data_format, bfloat16_single_tile_size_bytes, w2_cb_tiles);
    [[maybe_unused]] auto cb_w3 = create_circular_buffer(
        program, all_cores, kW3CbIndex, data_format, bfloat16_single_tile_size_bytes, w1_w3_cb_tiles);
    // Partial CBs for flash-attention optimization (accumulate XW1/XW3 across p_blocks)
    [[maybe_unused]] auto cb_x_w1_partial = create_circular_buffer(
        program, all_cores, kXW1PartialCbIndex, data_format, bfloat16_single_tile_size_bytes, num_tiles_xw1);
    [[maybe_unused]] auto cb_x_w3_partial = create_circular_buffer(
        program, all_cores, kXW3PartialCbIndex, data_format, bfloat16_single_tile_size_bytes, num_tiles_xw3);
    // XW1, XW3, and M CBs for full row caching
    [[maybe_unused]] auto cb_x_w1 = create_circular_buffer(
        program, all_cores, kXW1CbIndex, data_format, bfloat16_single_tile_size_bytes, num_tiles_xw1);
    [[maybe_unused]] auto cb_x_w3 = create_circular_buffer(
        program, all_cores, kXW3CbIndex, data_format, bfloat16_single_tile_size_bytes, num_tiles_xw3);
    [[maybe_unused]] auto cb_m = create_circular_buffer(
        program, all_cores, kMCbIndex, data_format, bfloat16_single_tile_size_bytes, num_tiles_m);
    [[maybe_unused]] auto cb_y_partial = create_circular_buffer(
        program, all_cores, kYPartialCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_y = create_circular_buffer(
        program, all_cores, kYCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* w1_buffer = w1.buffer();
    TT_FATAL(
        w1_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "W1 buffer must be in DRAM. w1 buffer of type {}",
        enchantum::to_string(w1_buffer->buffer_type()));

    auto* w2_buffer = w2.buffer();
    TT_FATAL(
        w2_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "W2 buffer must be in DRAM. w2 buffer of type {}",
        enchantum::to_string(w2_buffer->buffer_type()));

    auto* w3_buffer = w3.buffer();
    TT_FATAL(
        w3_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "W3 buffer must be in DRAM. w3 buffer of type {}",
        enchantum::to_string(w3_buffer->buffer_type()));

    auto* swiglu_buffer = output.buffer();
    TT_FATAL(
        swiglu_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "SwiGLU buffer must be in DRAM. SwiGLU buffer of type {}",
        enchantum::to_string(swiglu_buffer->buffer_type()));

    // M-fits-L1 algorithm uses flash-attention optimization with full row caching
    std::map<std::string, std::string> defines;
    defines[kRowOfMFitsInL1DefineKey] = "1";

    // -------------------------------------------------------------------------
    // 3.1) Split cores into single sender (0,0) and all receivers
    // -------------------------------------------------------------------------
    // SINGLE-SENDER MULTICAST: Core (0,0) reads from DRAM and multicasts to ALL others.
    // This is much more efficient than per-row multicast for weight sharing.

    std::vector<tt::tt_metal::CoreRange> sender_ranges;
    std::vector<tt::tt_metal::CoreRange> receiver_ranges;

    for (const auto& core_range : all_cores.ranges()) {
        for (uint32_t x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
            for (uint32_t y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                tt::tt_metal::CoreCoord core = {x, y};
                // Single-sender: only core (0,0) is the sender
                if (use_multicast) {
                    if (core.x == 0 && core.y == 0) {
                        sender_ranges.push_back(tt::tt_metal::CoreRange(core, core));
                    } else {
                        receiver_ranges.push_back(tt::tt_metal::CoreRange(core, core));
                    }
                } else {
                    // No multicast (single core): use sender kernel
                    sender_ranges.push_back(tt::tt_metal::CoreRange(core, core));
                }
            }
        }
    }

    tt::tt_metal::CoreRangeSet sender_core_set(sender_ranges);
    tt::tt_metal::CoreRangeSet receiver_core_set;
    if (!receiver_ranges.empty()) {
        receiver_core_set = tt::tt_metal::CoreRangeSet(receiver_ranges);
    }

    // -------------------------------------------------------------------------
    // 3.2) Create shared semaphores for W1/W2/W3 multicast
    // NOTE: W1, W3, W2 execute sequentially (W1→W3 in Phase A, W2 in Phase C)
    // and share the same multicast topology, so they can reuse the same semaphores.
    // -------------------------------------------------------------------------
    uint32_t mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    uint32_t mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // -------------------------------------------------------------------------
    // 3.3) Create dual-NOC dataflow kernels
    // -------------------------------------------------------------------------
    // DUAL-NOC ARCHITECTURE (matches tt-metal matmul 1D mcast):
    //   RISCV_1 / in0_noc: X reader + Y writer (all cores)
    //   RISCV_0 / in1_noc: Weight sender (sender core) or Weight receiver (receiver cores)
    tt::tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt::tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    SwiGLUForwardKernels kernels;

    // --- RISCV_1: X reader + Y writer (runs on ALL cores) ---
    std::vector<uint32_t> x_reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(x_reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(swiglu_buffer).append_to(x_reader_compile_time_args);
    kernels.x_reader_y_writer = tt::tt_metal::CreateKernel(
        program,
        kXReaderYWriterKernelPath,
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = x_reader_compile_time_args,
            .defines = defines});

    // --- RISCV_0: Weight sender (sender core only) ---
    std::vector<uint32_t> weight_sender_compile_time_args{block_size, Wt, hidden_Wt};
    tt::tt_metal::TensorAccessorArgs(w1_buffer).append_to(weight_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(w2_buffer).append_to(weight_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(w3_buffer).append_to(weight_sender_compile_time_args);
    kernels.weight_sender = tt::tt_metal::CreateKernel(
        program,
        kWeightSenderKernelPath,
        sender_core_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = weight_sender_compile_time_args,
            .defines = defines});

    // --- RISCV_0: Weight receiver (receiver cores, only if multicast) ---
    if (use_multicast && !receiver_ranges.empty()) {
        std::vector<uint32_t> weight_receiver_compile_time_args{block_size, Wt, hidden_Wt};
        kernels.weight_receiver = tt::tt_metal::CreateKernel(
            program,
            kWeightReceiverKernelPath,
            receiver_core_set,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_noc,
                .compile_args = weight_receiver_compile_time_args,
                .defines = defines});
    }

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for swiglu_fw
    // -------------------------------------------------------------------------

    // UNIFORM PADDING FOR MULTICAST SYNC:
    // Both compute groups need to loop for max_rows_across_all_cores iterations.
    // For padding rows (where r >= num_rows_per_core for that group), compute
    // consumes inputs but doesn't produce output.

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // num_rows_per_core: actual work
        max_rows_across_all_cores,  // max_rows_for_sync: loop iterations for multicast sync
        block_size,                 // per_core_block_size
        Wt,                         // num_inner / TILE_W
        hidden_Wt                   // hidden_num_inner / TILE_W
    };

    kernels.compute_group_1 = create_compute_kernel(
        program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // num_rows_per_core: actual work (less than group 1)
            max_rows_across_all_cores,  // max_rows_for_sync: same as group 1 for sync!
            block_size,                 // per_core_block_size
            Wt,                         // num_inner / TILE_W
            hidden_Wt                   // hidden_num_inner / TILE_W
        };

        kernels.compute_group_2 = create_compute_kernel(
            program, core_group_2, compute_group_2_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        w1_buffer,
        w2_buffer,
        w3_buffer,
        swiglu_buffer,
        num_cores,
        num_cores_x,
        num_cores_y,
        num_rows_per_core_group_1,
        num_rows_per_core_group_2,
        core_group_1,
        core_group_2,
        device,
        mcast_sender_semaphore_id,
        mcast_receiver_semaphore_id,
        use_multicast,
        max_rows_across_all_cores);

    // -------------------------------------------------------------------------
    // 6) Return the fully configured program & relevant shared variables
    // -------------------------------------------------------------------------
    return cached_program_t{
        std::move(program),
        {/* x_reader_y_writer_kernel_id            = */ kernels.x_reader_y_writer,
         /* weight_sender_kernel_id                = */ kernels.weight_sender,
         /* weight_receiver_kernel_id              = */ kernels.weight_receiver,
         /* swiglu_fw_kernel_group_1_id            = */ kernels.compute_group_1,
         /* swiglu_fw_kernel_group_2_id            = */ kernels.compute_group_2,
         /* core_group_1                           = */ core_group_1,
         /* core_group_2                           = */ core_group_2,
         /* num_cores                              = */ num_cores,
         /* num_cores_x                            = */ num_cores_x,
         /* num_cores_y                            = */ num_cores_y,
         /* use_multicast                          = */ use_multicast}};
}

void SwiGLUForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_x = shared_variables.num_cores_x;
    bool use_multicast = shared_variables.use_multicast;

    auto* input_buffer = tensor_args.input.buffer();
    auto* w1_buffer = tensor_args.w1.buffer();
    auto* w2_buffer = tensor_args.w2.buffer();
    auto* w3_buffer = tensor_args.w3.buffer();
    auto* swiglu_buffer = output.buffer();

    // Update addresses for all kernels
    auto& x_reader_runtime_args = GetRuntimeArgs(program, shared_variables.x_reader_y_writer_kernel_id);
    auto& weight_sender_runtime_args = GetRuntimeArgs(program, shared_variables.weight_sender_kernel_id);

    for (uint32_t i = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i % num_cores_x, i / num_cores_x};
        bool is_sender = (core.x == 0 && core.y == 0);

        // Update X reader + Y writer (all cores): X address and Y address
        {
            auto& runtime_args = x_reader_runtime_args[core.x][core.y];
            runtime_args[kXReaderXAddrIdx] = input_buffer->address();
            runtime_args[kXReaderYAddrIdx] = swiglu_buffer->address();
        }

        // Update weight sender (sender core or single-core): W1/W2/W3 addresses
        if (is_sender || !use_multicast) {
            auto& runtime_args = weight_sender_runtime_args[core.x][core.y];
            runtime_args[kWSenderW1AddrIdx] = w1_buffer->address();
            runtime_args[kWSenderW2AddrIdx] = w2_buffer->address();
            runtime_args[kWSenderW3AddrIdx] = w3_buffer->address();
        }
        // Weight receiver has no buffer addresses to update (all via multicast)
    }
}

}  // namespace ttml::metal::ops::swiglu_fw::device
