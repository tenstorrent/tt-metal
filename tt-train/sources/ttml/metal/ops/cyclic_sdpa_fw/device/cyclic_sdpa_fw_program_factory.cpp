// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cyclic_sdpa_fw_program_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "metal/common/program_utils.hpp"
#include "metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "metal/ops/cyclic_sdpa_bw/device/parity_snake.hpp"

namespace ttml::metal::ops::cyclic_sdpa_fw::device {

namespace {

constexpr auto kReaderPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_fw/device/kernels/dataflow/"
    "cyclic_sdpa_fw_relay_reader.cpp";
constexpr auto kWriterPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_fw/device/kernels/dataflow/"
    "cyclic_sdpa_fw_relay_writer.cpp";
constexpr auto kComputePath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_fw/device/kernels/compute/"
    "cyclic_sdpa_fw_compute.cpp";

constexpr uint32_t kTile = 32U;

// Where the buffer addresses sit in each kernel's runtime arguments.
constexpr uint32_t kReaderFirstAddressArg = 2U;
constexpr uint32_t kReaderAddressCount = 5U;  // query, key, value, state accumulator, state statistics
constexpr uint32_t kWriterFirstAddressArg = 2U;
constexpr uint32_t kWriterAddressCount = 2U;  // output, intermediates

using ttml::metal::ops::cyclic_sdpa_bw::placement_of;
using ttml::metal::ops::cyclic_sdpa_bw::snake_neighbors;
using ttml::metal::ops::cyclic_sdpa_bw::kNoCore;

}  // namespace

CyclicSDPAForwardProgramFactory::cached_program_t CyclicSDPAForwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& attn_output = output[0];
    const auto& intermediates = output[1];
    const auto& state_acc = output[2];
    const auto& state_stats = output[3];

    auto* device = query.device();
    const auto shape = query.logical_shape();
    const uint32_t chunks = std::max(1U, args.sequence_chunks);
    const uint32_t pairs = static_cast<uint32_t>(std::max<size_t>(1, args.row_chunks.size()));
    const uint32_t batch = static_cast<uint32_t>(shape[0]);
    const uint32_t q_heads = static_cast<uint32_t>(shape[1]);
    const uint32_t heads = batch * q_heads;
    const uint32_t slices = heads * pairs;
    // Grouped-query attention: the same slice decode as the backward, but the
    // query heads of a key head are independent here (columns have no
    // outputs), so no group cap and no seeding.
    const uint32_t kv_heads = static_cast<uint32_t>(key.logical_shape()[1]);
    const uint32_t heads_per_group = q_heads / kv_heads;
    const uint32_t kv_slices = batch * kv_heads;
    const uint32_t N = static_cast<uint32_t>(shape[2]) / chunks;
    const uint32_t d = static_cast<uint32_t>(shape[3]);

    auto layout = plan_layout(
        device->compute_with_storage_grid_size(), N, args.rows_per_block_tiles, slices, args.max_groups);
    std::vector<uint32_t> pair_table = {chunks, pairs, heads, kv_slices, q_heads, kv_heads, heads_per_group};
    for (uint32_t p = 0; p < pairs; ++p) {
        pair_table.push_back(args.row_chunks.empty() ? 0U : args.row_chunks[p]);
        pair_table.push_back(args.col_chunks.empty() ? 0U : args.col_chunks[p]);
    }
    const uint32_t C = layout.cores_per_group;
    const uint32_t Bt = args.rows_per_block_tiles;
    const uint32_t qWt = d / kTile;
    const uint32_t vWt = d / kTile;
    const uint32_t block_size = get_block_size(qWt, 4U);
    const uint32_t rowT = Bt * qWt;
    const uint32_t valT = Bt * vWt;
    const uint32_t scoreT = Bt * Bt;
    const auto& region = layout.region;

    auto program = CreateProgram();

    const uint32_t bf16_tile = 2U * kTile * kTile;
    const uint32_t fp32_tile = 4U * kTile * kTile;
    const auto make_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format) {
        const uint32_t page = (format == tt::DataFormat::Float32) ? fp32_tile : bf16_tile;
        CreateCircularBuffer(
            program, region, CircularBufferConfig(tiles * page, {{index, format}}).set_page_size(index, page));
    };
    // A buffer with two indices onto one memory: the compute kernel reads the
    // packet's state through the first and packs the updated state through
    // the second, in place. Two slots.
    const auto make_view_pair = [&](uint32_t seed_index, uint32_t out_index, uint32_t tiles) {
        CreateCircularBuffer(
            program, region,
            CircularBufferConfig(
                2U * tiles * fp32_tile,
                {{seed_index, tt::DataFormat::Float32}, {out_index, tt::DataFormat::Float32}})
                .set_page_size(seed_index, fp32_tile)
                .set_page_size(out_index, fp32_tile));
    };

    // ---- The packet, two slots: Q (bf16), the O^T accumulator (Float32,
    // every tile transposed within itself), and the statistics m and l as
    // row-layout Float32 tiles (one value per query in row 0), one tile per
    // row tile of the block. All three mutable fields are read through a
    // seed view and packed through an out view onto the same memory.
    make_cb(tt::CBIndex::c_0, 2U * rowT, tt::DataFormat::Float16_b);  // Q_i
    make_view_pair(tt::CBIndex::c_15, tt::CBIndex::c_17, rowT);       // O^T
    // m has a third view too: exact copies through c_13 (unpack to dest), the
    // FPU's row-broadcast subtraction reads its row 0 through c_25.
    CreateCircularBuffer(
        program, region,
        CircularBufferConfig(
            2U * Bt * fp32_tile,
            {{tt::CBIndex::c_13, tt::DataFormat::Float32},
             {tt::CBIndex::c_18, tt::DataFormat::Float32},
             {tt::CBIndex::c_25, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_13, fp32_tile)
            .set_page_size(tt::CBIndex::c_18, fp32_tile)
            .set_page_size(tt::CBIndex::c_25, fp32_tile));
    // l has a third view: the compute kernel reads it exactly through c_14
    // (unpack to dest) for the running sum, and through c_26 as a plain
    // matmul operand for the FPU broadcast of 1/l.
    CreateCircularBuffer(
        program, region,
        CircularBufferConfig(
            2U * Bt * fp32_tile,
            {{tt::CBIndex::c_14, tt::DataFormat::Float32},
             {tt::CBIndex::c_19, tt::DataFormat::Float32},
             {tt::CBIndex::c_26, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_14, fp32_tile)
            .set_page_size(tt::CBIndex::c_19, fp32_tile)
            .set_page_size(tt::CBIndex::c_26, fp32_tile));
    // ---- The resident column: K_j, V_j, and V_j^T made by the compute kernel.
    make_cb(tt::CBIndex::c_1, rowT, tt::DataFormat::Float16_b);       // K_j
    make_cb(tt::CBIndex::c_2, valT, tt::DataFormat::Float16_b);       // V_j
    make_cb(tt::CBIndex::c_16, valT, tt::DataFormat::Float16_b);      // V_j^T
    // ---- Constants the writer makes once.
    make_cb(tt::CBIndex::c_6, 2U, tt::DataFormat::Float16_b);         // causal mask: triangle, all -inf
    make_cb(tt::CBIndex::c_8, 1, tt::DataFormat::Float16_b);          // zero tile / transpose fence
    make_cb(tt::CBIndex::c_28, 1, tt::DataFormat::Float16_b);         // ones column (the lse mask)
    make_cb(tt::CBIndex::c_27, 1, tt::DataFormat::Float16_b);         // reduce scaler: all ones
    make_cb(tt::CBIndex::c_29, 1, tt::DataFormat::Float16_b);         // ones row (the row-0 mask)
    // ---- Intermediates of one timestep.
    make_cb(tt::CBIndex::c_10, scoreT, tt::DataFormat::Float32);      // S^T, 19-bit rounded
    make_cb(tt::CBIndex::c_12, scoreT, tt::DataFormat::Float32);      // P^T
    make_cb(tt::CBIndex::c_20, Bt, tt::DataFormat::Float32);          // r = exp(a (m_old - m_new)), full tile
    make_cb(tt::CBIndex::c_23, Bt, tt::DataFormat::Float32);          // colmax S^T, row layout (scratch)
    // ---- The finished row, at its last visit: O in bf16 and lse in column layout.
    make_cb(tt::CBIndex::c_21, rowT, tt::DataFormat::Float16_b);      // O_i
    make_cb(tt::CBIndex::c_22, Bt, tt::DataFormat::Float32);          // lse_i
    // ---- Words and tokens.
    make_cb(tt::CBIndex::c_24, 1U, tt::DataFormat::Float32);          // readiness scratch
    make_cb(tt::CBIndex::c_7, 2U, tt::DataFormat::Float32);           // slot-release tokens

    // Readiness per slot: the immutable field (Q) and the state arrive at
    // different times. Credits per neighbour and self; the two endpoints.
    const uint32_t ready_imm0_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready_imm1_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready_st0_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready_st1_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_prev_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_next_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_self_sem = CreateSemaphore(program, region, 0);
    const uint32_t endpoint1_sem = CreateSemaphore(program, region, 0);
    const uint32_t endpoint2_sem = CreateSemaphore(program, region, 0);

    std::map<std::string, std::string> defines;
    // Timing experiments (results wrong): TTML_CYCLIC_FW_EXPERIMENT=NO_EXP,GENERIC_EXP,NO_STATS,NO_PROBS,NO_RESCALE_O,NO_EXACT_PACK
    if (const char* env = std::getenv("TTML_CYCLIC_FW_EXPERIMENT"); env != nullptr && *env != '\0') {
        std::string list(env);
        size_t pos = 0;
        while (pos <= list.size()) {
            const size_t next = list.find(',', pos);
            const std::string item = list.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
            if (item == "FID_S2") {
                defines["FID_S"] = "2";  // matmul fidelity experiments
            } else if (item == "FID_O2") {
                defines["FID_O"] = "2";
            } else if (item == "FID_O3") {
                defines["FID_O"] = "3";
            } else if (!item.empty()) {
                defines["FW_EXPERIMENT_" + item] = "1";
            }
            if (next == std::string::npos) {
                break;
            }
            pos = next + 1;
        }
    }
    const bool dense = args.mask_type == ttml::metal::AttentionMaskType::None;
    if (dense) {
        defines["DENSE_MODE"] = "1";
    }

    std::vector<uint32_t> reader_args = {
        C, qWt, vWt, ready_imm0_sem, ready_imm1_sem, ready_st0_sem, ready_st1_sem,
        credit_prev_sem, credit_next_sem, credit_self_sem, endpoint1_sem, endpoint2_sem, Bt};
    for (const auto* t : {&query, &key, &value, &state_acc, &state_stats}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(reader_args);
    }
    const auto reader = CreateKernel(
        program, kReaderPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_args,
            .defines = defines});

    std::vector<uint32_t> writer_args = {C, qWt, vWt, Bt};
    for (const auto* t : {&attn_output, &intermediates}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(writer_args);
    }
    const auto writer = CreateKernel(
        program, kWriterPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_args,
            .defines = defines});

    const uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(static_cast<float>(d)));
    // Read straight into DST, all 32 bits: the state (m, l, O^T) and r, by
    // copies only. The FPU's operands -- S^T, P^T, the block maximum, the
    // plain views of m and l -- stay in the default mode (see the kernel).
    std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_mode[tt::CBIndex::c_13] = UnpackToDestMode::UnpackToDestFp32;
    unpack_mode[tt::CBIndex::c_14] = UnpackToDestMode::UnpackToDestFp32;
    unpack_mode[tt::CBIndex::c_15] = UnpackToDestMode::UnpackToDestFp32;
    unpack_mode[tt::CBIndex::c_20] = UnpackToDestMode::UnpackToDestFp32;
    const auto compute = CreateKernel(
        program, kComputePath, region,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            // Half-sync: every stage works in groups of at most three Float32
            // tiles, so the math and pack threads alternate halves of the file.
            .dst_full_sync_en = false,
            .unpack_to_dest_mode = unpack_mode,
            .compile_args = {C, qWt, vWt, scaler, block_size, Bt},
            .defines = defines});

    for (uint32_t g = 0; g < layout.groups; ++g) {
        const auto origin = layout.group_origin[g];
        const auto logical_of = [&](uint32_t core) {
            const auto xy = placement_of(C, layout.group_width, core);
            return CoreCoord{origin.x + xy.x, origin.y + xy.y};
        };
        const auto noc_of_core = [&](uint32_t core) {
            return device->worker_core_from_logical_core(logical_of(core));
        };
        for (uint32_t c = 1U; c <= C; ++c) {
            const auto core = logical_of(c);
            const auto neighbors = snake_neighbors(C, c);
            const auto prev = noc_of_core(neighbors.prev != kNoCore ? neighbors.prev : c);
            const auto next = noc_of_core(neighbors.next != kNoCore ? neighbors.next : c);
            std::vector<uint32_t> rt = {
                c,
                g,
                query.buffer()->address(),
                key.buffer()->address(),
                value.buffer()->address(),
                state_acc.buffer()->address(),
                state_stats.buffer()->address(),
                static_cast<uint32_t>(prev.x),
                static_cast<uint32_t>(prev.y),
                static_cast<uint32_t>(next.x),
                static_cast<uint32_t>(next.y)};
            for (uint32_t r = 1U; r <= C; ++r) {
                const auto rc = noc_of_core(r);
                rt.push_back(static_cast<uint32_t>(rc.x));
                rt.push_back(static_cast<uint32_t>(rc.y));
            }
            const uint32_t slice_count = slices_of_group(layout, g);
            rt.push_back(slice_count);
            rt.push_back(layout.groups);
            rt.insert(rt.end(), pair_table.begin(), pair_table.end());
            SetRuntimeArgs(program, reader, core, rt);
            std::vector<uint32_t> wt = {
                c, g, attn_output.buffer()->address(), intermediates.buffer()->address(), slice_count,
                layout.groups};
            wt.insert(wt.end(), pair_table.begin(), pair_table.end());
            SetRuntimeArgs(program, writer, core, wt);
            SetRuntimeArgs(program, compute, core, {c, slice_count});
        }
    }

    return cached_program_t{std::move(program), {reader, writer, compute, layout}};
}

void CyclicSDPAForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*args*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;
    const auto& layout = shared.layout;

    const std::array<uint32_t, kReaderAddressCount> reader_addresses = {
        tensor_args.query.buffer()->address(),
        tensor_args.key.buffer()->address(),
        tensor_args.value.buffer()->address(),
        output[2].buffer()->address(),
        output[3].buffer()->address()};
    const std::array<uint32_t, kWriterAddressCount> writer_addresses = {
        output[0].buffer()->address(), output[1].buffer()->address()};

    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (uint32_t g = 0; g < layout.groups; ++g) {
        const auto origin = layout.group_origin[g];
        for (uint32_t c = 1U; c <= layout.cores_per_group; ++c) {
            const auto xy = placement_of(layout.cores_per_group, layout.group_width, c);
            const tt::tt_metal::CoreCoord core{origin.x + xy.x, origin.y + xy.y};
            auto& rt = reader_args[core.x][core.y];
            for (uint32_t i = 0; i < kReaderAddressCount; ++i) {
                rt[kReaderFirstAddressArg + i] = reader_addresses[i];
            }
            auto& wt = writer_args[core.x][core.y];
            for (uint32_t i = 0; i < kWriterAddressCount; ++i) {
                wt[kWriterFirstAddressArg + i] = writer_addresses[i];
            }
        }
    }
}

}  // namespace ttml::metal::ops::cyclic_sdpa_fw::device
