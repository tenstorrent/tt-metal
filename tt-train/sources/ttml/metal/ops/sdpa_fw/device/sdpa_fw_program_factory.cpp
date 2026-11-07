// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_fw_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <utility>

#include "metal/common/program_utils.hpp"
#include "sdpa_fw_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/dataflow/sdpa_fw_reader_kernel.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/dataflow/sdpa_fw_writer_kernel.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp";

// writer runtime args
constexpr uint32_t kOutputBufferIdx = 0;
constexpr uint32_t kIntermediateBufferIdx = 1U;
constexpr uint32_t kWriterArgScratch = 2U;  // partials scratch (split units); set at create, never patched

const std::string kSplitUnitsDefKey = "SPLIT_UNITS";
const std::string kReturnIntermediates = "RETURN_INTERMEDIATES";
const std::string kUseAttnMaskDefKey = "USE_ATTN_MASK";
const std::string kCausalMaskDefKey = "CAUSAL_MASK";

const std::string kHasGateDefKey = "HAS_GATE";

}  // namespace

namespace ttml::metal::ops::sdpa_fw::device {

struct SDPAForwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute;
};

// Runtime args: the buffer addresses (reader, writer), then the core's item runs for all three kernels:
// num_units, three words per unit (sdpa_fw_config.hpp).
void assign_per_core_runtime_args(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    const SDPAForwardKernels& kernels,
    const tt::tt_metal::Buffer* query_buffer,
    const tt::tt_metal::Buffer* key_buffer,
    const tt::tt_metal::Buffer* value_buffer,
    const tt::tt_metal::Buffer* mask_buffer,
    const tt::tt_metal::Buffer* gate_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    const tt::tt_metal::Buffer* intermediates_buffer,
    const tt::tt_metal::Buffer* scratch_buffer,
    uint32_t num_cores_y,
    const std::vector<CoreUnits>& per_core_units) {
    for (uint32_t i = 0; i < per_core_units.size(); i++) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto& units = per_core_units[i];

        std::vector<uint32_t> work_args{static_cast<uint32_t>(units.size())};
        for (const auto& unit : units) {
            const tt::tt_metal::CoreCoord merger_noc = device->worker_core_from_logical_core(
                tt::tt_metal::CoreCoord{unit.merger_core / num_cores_y, unit.merger_core % num_cores_y});
            append_unit_words(work_args, unit, merger_noc.x, merger_noc.y);
        }
        std::vector<uint32_t> reader_args = {
            query_buffer->address(),
            key_buffer->address(),
            value_buffer->address(),
            get_address_or_zero(mask_buffer),
            get_address_or_zero(gate_buffer),
            get_address_or_zero(scratch_buffer)};
        reader_args.insert(reader_args.end(), work_args.begin(), work_args.end());
        SetRuntimeArgs(program, kernels.reader, core, reader_args);

        std::vector<uint32_t> writer_args(kWriterArgScratch + 1U);
        writer_args[kOutputBufferIdx] = output_buffer->address();
        writer_args[kIntermediateBufferIdx] = get_address_or_zero(intermediates_buffer);
        writer_args[kWriterArgScratch] = get_address_or_zero(scratch_buffer);
        writer_args.insert(writer_args.end(), work_args.begin(), work_args.end());
        SetRuntimeArgs(program, kernels.writer, core, writer_args);

        SetRuntimeArgs(program, kernels.compute, core, work_args);
    }
}

SDPAForwardProgramFactory::cached_program_t SDPAForwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& attn_mask = tensor_args.mask;
    const AttentionMaskType mask_type = args.mask_type;
    const auto& gate = tensor_args.gate;
    /*
    Shape note:
    Q: B x qNH x S x qE
    K: B x kNH x S x kE
    V: B x vNH x S x vE
    attn_mask: 1 x 1 x S x S
    */

    auto* device = query.device();

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram(*device);

    TT_FATAL(
        query.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            key.physical_volume() % tt::constants::TILE_WIDTH == 0 &&
            value.physical_volume() % tt::constants::TILE_WIDTH == 0,
        "Physical volume of input tensors must be multiple of TILE_WIDTH. Got query {}, key {}, value {}",
        query.physical_volume(),
        key.physical_volume(),
        value.physical_volume());

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_available_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    // The program config in tiles: Sq_chunk_t query tile-rows of heads_per_item heads are processed
    // together against one pass over K/V, Sk_chunk_t K/V tiles per step. Unset knobs follow the shape rule in
    // sdpa_fw_config.cpp; the ring SDPA path builds the program directly, so the knobs are checked here as
    // well as in validate.
    const SDPAForwardShape shape = shape_of(query, key, value, mask_type, args.has_gate, args.return_intermediates);
    const SDPAForwardChunks chunks =
        resolve_chunks(args.program_config, shape, available_l1_bytes(*device), num_available_cores);
    const uint32_t Sq_chunk_t = chunks.Sq_chunk_t;
    const uint32_t Sk_chunk_t = chunks.Sk_chunk_t;
    const uint32_t heads_per_item = chunks.heads_per_item;
    const uint32_t rows = chunks.rows();                              // tile-rows of one item
    const uint32_t heads_per_group = shape.q_heads / shape.kv_heads;  // Q heads that share one K/V head

    const float scale = 1.0F / std::sqrt(static_cast<float>(shape.qWt * tt::constants::TILE_WIDTH));
    const uint32_t scaler = std::bit_cast<uint32_t>(scale);
    // BF16 copy of the scale for the WH fused scale+exp path (SFPU multiplies P by a BF16
    // immediate there, while the LSE and the backward pass use the full FP32 scale). RNE
    // instead of `>> 16` truncation halves the worst-case rounding gap so it stays noise
    // rather than a systematic fw/bw mismatch. The conversion runs on host because the
    // kernel TU cannot see the host bfloat16 implementation.
    const uint32_t scaler_bf16 = fp32_to_bf16_bits_round_to_nearest_even(scale);
    const uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);  // used to transform mask from 1/0 to 0/-1
    const uint32_t custom_inf = std::bit_cast<uint32_t>(1e9F);  // used to transform mask from 0/-1 to 0/-1e9F

    const SDPAForwardSchedule schedule = schedule_for(args.program_config, shape, chunks, num_available_cores);
    const auto& per_core_units = schedule.per_core;
    const uint32_t num_cores = static_cast<uint32_t>(per_core_units.size());
    const auto all_cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size);
    const bool split_units = schedule.num_slots > 0U;

    // DST is the register file the math unit computes in: 4 tiles per half in fp32 mode (8 in bf16);
    // the kernels run half-sync, so one pass gets one half.
    // Everything below is "how many tiles go through DST per pass":
    //   block_size     output tiles per pass in the element-wise loops, the accumulator rescale and the
    //                  finalize (one slot stays free for the broadcast factor they multiply by)
    //   qk_block_*     Q rows x K tiles per QK^T matmul_block call
    //   pv_block_*     Q rows x V feature tiles per PV matmul_block call
    //   sm_block_size  score tiles per pass of the exp step
    constexpr bool fp32_dest_acc_en = true;
    const uint32_t dst_size = fp32_dest_acc_en ? 4U : 8U;
    const uint32_t block_size = get_block_size(shape.vWt, dst_size - 1U);
    // The matmul rectangles come from ttnn's SDPA subblock search: the biggest h x w that fits DST and
    // tiles the rows x cols output.
    using ttnn::prim::detail::determine_largest_subblock_size;
    const auto [qk_block_h, qk_block_w] = determine_largest_subblock_size(rows, Sk_chunk_t, dst_size);
    const auto [pv_block_h, pv_block_w] = determine_largest_subblock_size(rows, shape.vWt, dst_size);
    const uint32_t sm_block_size = dst_size;

    // -------------------------------------------------------------------------
    // 2) Create the circular buffers
    // -------------------------------------------------------------------------

    // Depths and formats come from the CB plan in sdpa_fw_config.cpp, the same table l1_bytes summed when
    // resolve_chunks checked the L1 budget. The mask, gate, intermediates and split-unit CBs are only
    // created when the config uses them (depth 0 otherwise).
    for (const SDPAForwardCb& cb : cb_plan(cb_tiles(shape, chunks, split_units))) {
        if (cb.tiles > 0U) {
            create_circular_buffer(program, all_cores, cb.index, cb.format, tt::tile_size(cb.format), cb.tiles);
        }
    }

    // Split units: the partials travel through DRAM scratch, one slot per sender, fp32, and the merger
    // waits on one of its core's semaphores until every sender has bumped it. The scratch is owned by the
    // cached program, so it lives as long as the cache entry: num_slots * rows * (vWt + 2) fp32 tiles.
    // An item is cut only when it is heavier than a core's fair share, so the cut items' pieces, and the
    // slots, number at most num_cores: 120 * 4 * 6 tiles = 11 MiB at D=128 with 4-row items, scaling with
    // the rows of an item, and less in practice (the default picks need about 60 slots). Shapes with
    // enough items per core cut nothing and own none. The scratch is DRAM whatever the inputs' placement.
    // Allocating it per call instead would need the output specs to run the chunk pick and the scheduler on
    // every launch (the slot count is the schedule's) and a third return tensor through the op; the knob
    // doc points workloads with many distinct shapes to max_pieces = 1 meanwhile.
    std::optional<ttnn::Tensor> scratch;
    if (split_units) {
        const uint32_t partial_rows = schedule.num_slots * rows;
        scratch = ttnn::create_device_tensor(
            tt::tt_metal::TensorSpec(
                ttnn::Shape(
                    {1U,
                     1U,
                     partial_rows * tt::constants::TILE_HEIGHT,
                     partial_tiles_per_row(shape) * tt::constants::TILE_WIDTH}),
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::FLOAT32,
                    tt::tt_metal::Layout::TILE,
                    tt::tt_metal::MemoryConfig(
                        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
            device);
    }
    // The mergers' semaphores, one per merger a core may hold; the kernels take their ids at fixed
    // compile-arg positions, so a program without split units passes zeros there.
    std::vector<uint32_t> semaphore_ids(kMaxMergersPerCore, 0U);
    if (split_units) {
        for (uint32_t& id : semaphore_ids) {
            id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
        }
    }
    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* query_buffer = query.buffer();
    TT_FATAL(
        query_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Query buffer must be in DRAM. Query buffer of type {}",
        enchantum::to_string(query_buffer->buffer_type()));

    auto* key_buffer = key.buffer();
    TT_FATAL(
        key_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Key buffer must be in DRAM. Key buffer of type {}",
        enchantum::to_string(key_buffer->buffer_type()));

    auto* value_buffer = value.buffer();
    TT_FATAL(
        value_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Value buffer must be in DRAM. Value buffer of type {}",
        enchantum::to_string(value_buffer->buffer_type()));

    auto* mask_buffer = attn_mask.has_value() ? attn_mask.value().buffer() : nullptr;
    if (mask_buffer != nullptr) {
        TT_FATAL(
            mask_buffer->buffer_type() == ttnn::BufferType::DRAM,
            "Mask buffer must be in DRAM. Mask buffer of type {}",
            enchantum::to_string(mask_buffer->buffer_type()));
    }

    auto* gate_buffer = args.has_gate ? gate.value().buffer() : nullptr;
    if (gate_buffer != nullptr) {
        TT_FATAL(
            gate_buffer->buffer_type() == ttnn::BufferType::DRAM,
            "Gate buffer must be in DRAM. Gate buffer of type {}",
            enchantum::to_string(gate_buffer->buffer_type()));
    }

    auto* output_buffer = output.front().buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(output_buffer->buffer_type()));

    auto* intermediates_buffer = args.return_intermediates ? output.back().buffer() : nullptr;
    auto* scratch_buffer = scratch.has_value() ? scratch->buffer() : nullptr;
    if (intermediates_buffer != nullptr) {
        TT_FATAL(
            intermediates_buffer->buffer_type() == ttnn::BufferType::DRAM,
            "Intermediates buffer must be in DRAM. Intermediates buffer of type {}",
            enchantum::to_string(intermediates_buffer->buffer_type()));
    }

    // configure defines
    std::map<std::string, std::string> defines;
    // setup defines for reduce
    // Compute kernel does not compile without these defines
    // LLK reduction uses define values as default template parameters
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    if (args.return_intermediates) {
        defines[kReturnIntermediates] = "1";
    }

    // Set defines based on mask type (mutually exclusive)
    switch (mask_type) {
        case AttentionMaskType::Causal: defines[kCausalMaskDefKey] = "1"; break;
        case AttentionMaskType::Arbitrary: defines[kUseAttnMaskDefKey] = "1"; break;
        case AttentionMaskType::None:
            // No mask define - kernel will just scale attention scores
            break;
    }

    if (args.has_gate) {
        defines[kHasGateDefKey] = "1";
    }
    if (split_units) {
        defines[kSplitUnitsDefKey] = "1";
    }

    // Per-arch math fidelity. Wormhole has HW bug TT #38306: HiFi4 + fp32_dest_acc + matmul_block
    // corrupts FP32 dest accumulation, so we use HiFi3 there. Blackhole is unaffected and uses HiFi4.
    const auto math_fidelity = (device->arch() == tt::ARCH::WORMHOLE_B0) ? tt::tt_metal::MathFidelity::HiFi3
                                                                         : tt::tt_metal::MathFidelity::HiFi4;

    SDPAForwardKernels kernels;

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_args = {
        shape.qWt,        // num tile in inner dim in query/key (d_qk/TILE_W)
        shape.vWt,        // num tile in inner dim in value (d_v/TILE_W)
        shape.St,         // num tile in seq len dim (S/TILE_H)
        shape.q_heads,    // number of heads in query
        heads_per_group,  // number of heads per group
        Sk_chunk_t,       // K/V tiles per inner-loop step
        Sq_chunk_t,       // query tile-rows per Q chunk
        heads_per_item,   // Q heads stacked in one item
    };
    reader_compile_args.insert(reader_compile_args.end(), semaphore_ids.begin(), semaphore_ids.end());
    tt::tt_metal::TensorAccessorArgs(query_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(key_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(value_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(mask_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(gate_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(scratch_buffer).append_to(reader_compile_args);

    kernels.reader = create_reader_kernel(
        program,
        all_cores,
        /* reader_compile_args */ reader_compile_args,
        defines,
        kReaderKernelPath);

    std::vector<uint32_t> writer_compile_args = {
        shape.vWt,       // num tile in inner dim in output/value (d_v/TILE_W)
        shape.St,        // num tile in seq len dim (S/TILE_H)
        shape.q_heads,   // number of heads in query
        Sq_chunk_t,      // query tile-rows per Q chunk
        heads_per_item,  // Q heads stacked in one item
    };
    writer_compile_args.insert(writer_compile_args.end(), semaphore_ids.begin(), semaphore_ids.end());
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(intermediates_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(scratch_buffer).append_to(writer_compile_args);
    kernels.writer = create_writer_kernel(
        program, all_cores, /* writer_compile_args */ writer_compile_args, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels
    // -------------------------------------------------------------------------

    // UnpackToDestFp32 for mm_out accum buffers and broadcast scratch:
    // copy_tile from these CBs bypasses SrcA and DMAs directly to DST at full FP32.
    // Only SFPU operations (mul_binary_tile, etc.) may be used on data loaded this way.
    auto create_unpack_to_dest_mode = []() {
        std::vector<tt::tt_metal::UnpackToDestMode> mode(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        mode[kPrevMmOutCbIndex] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        mode[kCurMmOutCbIndex] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        return mode;
    };

    const std::vector<uint32_t> compute_compile_args = {
        block_size,      // output tiles per DST pass in the element-wise loops
        shape.qWt,       // num tile in inner dim in query/key (d_qk/TILE_W)
        shape.vWt,       // num tile in inner dim in value (d_v/TILE_W)
        shape.St,        // num_seq_len / TILE_H
        scaler,          // 1/sqrt(d) softmax scale, fp32 bits
        scaler_bf16,     // BF16 (RNE) scale for the WH fused scale+exp path
        minus_one,       // used to transform mask from 1/0 to 0/-1
        custom_inf,      // used to transform mask from 0/-1 to 0/-1e9F
        Sq_chunk_t,      // query tile-rows per Q chunk
        Sk_chunk_t,      // K/V tiles per inner-loop step
        qk_block_h,      // item rows per QK^T matmul_block call
        qk_block_w,      // K tiles per QK^T matmul_block call
        pv_block_h,      // item rows per PV matmul_block call
        pv_block_w,      // V feature tiles per PV matmul_block call
        sm_block_size,   // score tiles per DST pass in the exp step
        heads_per_item,  // Q heads stacked in one item
    };
    kernels.compute = tt::tt_metal::CreateKernel(
        program,
        kComputeKernelPath,
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = create_unpack_to_dest_mode(),
            .math_approx_mode = false,
            .compile_args = compute_compile_args,
            .defines = defines});

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(
        device,
        program,
        kernels,
        query_buffer,
        key_buffer,
        value_buffer,
        mask_buffer,
        gate_buffer,
        output_buffer,
        intermediates_buffer,
        scratch_buffer,
        num_cores_y,
        per_core_units);

    // -------------------------------------------------------------------------
    // 6) Return the fully configured program & relevant shared variables
    // -------------------------------------------------------------------------
    return cached_program_t{
        std::move(program),
        {/* sdpa_fw_reader_kernel      = */ kernels.reader,
         /* sdpa_fw_writer_kernel      = */ kernels.writer,
         /* sdpa_fw_compute_kernel     = */ kernels.compute,
         /* num_cores                  = */ num_cores,
         /* num_cores_y                = */ num_cores_y,
         /* scratch                    = */ std::move(scratch)}};
}

void SDPAForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& shared_vars = cached_program.shared_variables;
    auto& sdpa_fw_reader_kernel = shared_vars.sdpa_fw_reader_kernel;
    auto& sdpa_fw_writer_kernel = shared_vars.sdpa_fw_writer_kernel;
    auto& program = cached_program.program;

    const uint32_t num_cores = shared_vars.num_cores;
    const uint32_t num_cores_y = shared_vars.num_cores_y;

    const auto* query_buffer = &tensor_args.query.mesh_buffer();
    const auto* key_buffer = &tensor_args.key.mesh_buffer();
    const auto* value_buffer = &tensor_args.value.mesh_buffer();
    const auto* mask_buffer = tensor_args.mask.has_value() ? tensor_args.mask.value().buffer() : nullptr;
    const auto* gate_buffer = tensor_args.gate.has_value() ? tensor_args.gate.value().buffer() : nullptr;
    const auto* output_buffer = &tensor_return_value.front().mesh_buffer();
    auto* intermediates_buffer =
        operation_attributes.return_intermediates ? tensor_return_value.back().buffer() : nullptr;

    // Only the input/output addresses change between runs; the scratch tensor is owned by the cached
    // program, so its address and the work lists stay as create() set them.
    auto& reader_runtime_args = GetRuntimeArgs(program, sdpa_fw_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, sdpa_fw_writer_kernel);

    for_each_core(num_cores, num_cores_y, [&](const tt::tt_metal::CoreCoord& core) {
        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kReaderArgQuery] = query_buffer->address();
            runtime_args[kReaderArgKey] = key_buffer->address();
            runtime_args[kReaderArgValue] = value_buffer->address();
            // For causal mask, mask is generated on-chip (mask_buffer is nullptr)
            // For DRAM mask, use actual mask buffer address
            runtime_args[kReaderArgMask] = get_address_or_zero(mask_buffer);
            runtime_args[kReaderArgGate] = get_address_or_zero(gate_buffer);
        }

        // Update output buffer for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputBufferIdx] = output_buffer->address();
            runtime_args[kIntermediateBufferIdx] = get_address_or_zero(intermediates_buffer);
        }
    });
}

}  // namespace ttml::metal::ops::sdpa_fw::device
