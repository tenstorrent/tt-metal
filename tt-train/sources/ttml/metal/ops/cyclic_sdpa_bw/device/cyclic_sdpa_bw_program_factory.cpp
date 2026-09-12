// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cyclic_sdpa_bw_program_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "cyclic_schedule.hpp"
#include "metal/common/program_utils.hpp"
#include "parity_snake.hpp"

namespace ttml::metal::ops::cyclic_sdpa_bw::device {

namespace {

constexpr auto kReaderPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/"
    "cyclic_sdpa_bw_relay_reader.cpp";
constexpr auto kWriterPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/"
    "cyclic_sdpa_bw_relay_writer.cpp";
constexpr auto kComputePath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/compute/"
    "cyclic_sdpa_bw_compute.cpp";

constexpr uint32_t kTile = 32U;

// Where the buffer addresses sit in each kernel's runtime arguments, so that
// a cached program can be re-pointed at new tensors without rebuilding.
constexpr uint32_t kReaderFirstAddressArg = 2U;
constexpr uint32_t kReaderAddressCount = 9U;
constexpr uint32_t kWriterFirstAddressArg = 2U;
constexpr uint32_t kWriterAddressCount = 2U;

}  // namespace

CyclicLayout plan_layout(
    const tt::tt_metal::CoreCoord& compute_grid,
    uint32_t sequence_length,
    uint32_t rows_per_block_tiles,
    uint32_t slices) {
    const uint32_t block_rows = rows_per_block_tiles * kTile;
    TT_FATAL(
        rows_per_block_tiles >= 1U && rows_per_block_tiles <= 4U,
        "cyclic_sdpa_bw: rows_per_block_tiles must be 1, 2, 3 or 4; got {}. Score tiles occupy "
        "contiguous DST registers with two shared scratch registers above them, which is "
        "rows_per_block_tiles + 2 of the eight Float32 registers.",
        rows_per_block_tiles);
    TT_FATAL(
        sequence_length % (2U * block_rows) == 0U,
        "cyclic_sdpa_bw: sequence length {} must be a multiple of 2 * {} = {}, because the "
        "schedule has T = 2C blocks of {} rows and C must come out a whole number of cores.",
        sequence_length,
        block_rows,
        2U * block_rows,
        block_rows);

    CyclicLayout layout;
    layout.cores_per_group = sequence_length / (2U * block_rows);
    layout.groups = slices;
    const uint32_t C = layout.cores_per_group;
    TT_FATAL(C >= 1U, "cyclic_sdpa_bw: the schedule needs at least one core, got {}", C);

    // The group rectangle must have area exactly C: the parity snake is laid
    // along a serpentine path through it, and that is what makes every edge of
    // the snake a single hop. A rectangle with spare cells would break the
    // embedding, not merely waste cores.
    const auto grid_x = static_cast<uint32_t>(compute_grid.x);
    const auto grid_y = static_cast<uint32_t>(compute_grid.y);
    uint32_t best_capacity = 0U;
    for (uint32_t h = 1U; h <= grid_y; ++h) {
        if (C % h != 0U) {
            continue;
        }
        const uint32_t w = C / h;
        if (w > grid_x) {
            continue;
        }
        if (!placement_is_nearest_neighbor(C, w, h)) {
            continue;
        }
        const uint32_t across = grid_x / w;
        const uint32_t down = grid_y / h;
        const uint32_t capacity = across * down;
        // Prefer the shape that holds the most groups; among equals prefer the
        // wider one, which keeps the groups' rows shorter and the whole
        // footprint closer to square.
        if (capacity > best_capacity || (capacity == best_capacity && w > layout.group_width)) {
            best_capacity = capacity;
            layout.group_width = w;
            layout.group_height = h;
            layout.groups_across = across;
        }
    }
    TT_FATAL(
        best_capacity > 0U,
        "cyclic_sdpa_bw: no rectangle of area C = {} fits a {}x{} compute grid with every parity "
        "snake edge a single hop. C follows from the sequence length and the block height, so "
        "either change the sequence length or the block height.",
        C,
        grid_x,
        grid_y);
    TT_FATAL(
        slices <= best_capacity,
        "cyclic_sdpa_bw: {} (batch x head) slices of {} cores each do not fit a {}x{} grid, which "
        "holds {} such groups. Splitting the batch across invocations is the way round this; the "
        "kernels run one schedule per group and do not loop over slices.",
        slices,
        C,
        grid_x,
        grid_y,
        best_capacity);

    std::vector<tt::tt_metal::CoreRange> ranges;
    for (uint32_t g = 0; g < slices; ++g) {
        const tt::tt_metal::CoreCoord origin{
            (g % layout.groups_across) * layout.group_width,
            (g / layout.groups_across) * layout.group_height};
        layout.group_origin.push_back(origin);
        ranges.emplace_back(
            origin,
            tt::tt_metal::CoreCoord{
                origin.x + layout.group_width - 1U, origin.y + layout.group_height - 1U});
    }
    // The union of the rectangles, never their bounding box: when the groups
    // do not tile that box, the cores in the gap receive the kernels but no
    // runtime arguments, and then wait forever on semaphores nobody posts to.
    layout.region = tt::tt_metal::CoreRangeSet(ranges);
    return layout;
}

CyclicSDPABackwardProgramFactory::cached_program_t CyclicSDPABackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& grad_output = tensor_args.grad_output;
    const auto& lse = tensor_args.log_sum_exp;
    const auto& row_scalar = tensor_args.row_scalar;
    const auto& grad_query = output[0];
    const auto& grad_key = output[1];
    const auto& grad_value = output[2];

    auto* device = query.device();
    const auto shape = query.logical_shape();
    const uint32_t slices = static_cast<uint32_t>(shape[0]) * static_cast<uint32_t>(shape[1]);
    const uint32_t N = static_cast<uint32_t>(shape[2]);
    const uint32_t d = static_cast<uint32_t>(shape[3]);

    const auto layout = plan_layout(device->compute_with_storage_grid_size(), N, args.rows_per_block_tiles, slices);
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
    // Two packet slots for the row-side fields, one resident column.
    make_cb(tt::CBIndex::c_0, 2U * rowT, tt::DataFormat::Float16_b);   // Q_i
    make_cb(tt::CBIndex::c_3, 2U * valT, tt::DataFormat::Float16_b);   // dO_i
    make_cb(tt::CBIndex::c_4, 2U * Bt, tt::DataFormat::Float32);       // L_i
    make_cb(tt::CBIndex::c_5, 2U * Bt, tt::DataFormat::Float32);       // D_i
    make_cb(tt::CBIndex::c_15, 2U * rowT, tt::DataFormat::Float32);    // dQ_i, travels along
    make_cb(tt::CBIndex::c_1, rowT, tt::DataFormat::Float16_b);        // K_j
    make_cb(tt::CBIndex::c_2, valT, tt::DataFormat::Float16_b);        // V_j
    make_cb(tt::CBIndex::c_6, 1U, tt::DataFormat::Float16_b);          // causal mask
    make_cb(tt::CBIndex::c_10, scoreT, tt::DataFormat::Float32);       // P
    make_cb(tt::CBIndex::c_11, scoreT, tt::DataFormat::Float32);       // dP
    make_cb(tt::CBIndex::c_12, scoreT, tt::DataFormat::Float32);       // dS
    make_cb(tt::CBIndex::c_13, scoreT, tt::DataFormat::Float32);       // dS^T
    make_cb(tt::CBIndex::c_14, scoreT, tt::DataFormat::Float32);       // P^T
    make_cb(tt::CBIndex::c_16, rowT, tt::DataFormat::Float32);         // dQ accumulator
    make_cb(tt::CBIndex::c_17, rowT, tt::DataFormat::Float32);         // dQ to the relay
    make_cb(tt::CBIndex::c_18, rowT, tt::DataFormat::Float32);         // dK seed
    make_cb(tt::CBIndex::c_19, rowT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_20, rowT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_21, valT, tt::DataFormat::Float32);         // dV seed
    make_cb(tt::CBIndex::c_22, valT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_23, valT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_24, 1U, tt::DataFormat::Float32);           // readiness scratch
    make_cb(tt::CBIndex::c_25, 1U, tt::DataFormat::Float32);           // release word
    make_cb(tt::CBIndex::c_26, 1U, tt::DataFormat::Float32);           // column-gradient progress
    make_cb(tt::CBIndex::c_7, 2U, tt::DataFormat::Float32);            // slot-release tokens

    const uint32_t arrive_sem = CreateSemaphore(program, region, 0);
    const uint32_t release_sem = CreateSemaphore(program, region, 0);
    // Two readiness words per slot: the immutable fields and dQ arrive, and
    // are needed, at different times.
    const uint32_t ready_imm0_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready_imm1_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready_dq0_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready_dq1_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_prev_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_next_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_self_sem = CreateSemaphore(program, region, 0);
    const uint32_t endpoint1_sem = CreateSemaphore(program, region, 0);
    const uint32_t endpoint2_sem = CreateSemaphore(program, region, 0);

    std::map<std::string, std::string> sync_defines;
    if (!args.use_barrier) {
        sync_defines["ENDPOINT_SYNC"] = "1";
    }
    std::map<std::string, std::string> compute_defines = sync_defines;
    compute_defines["COLUMN_RESIDENT"] = "1";
    compute_defines["RELEASE_TOKEN"] = "1";

    std::vector<uint32_t> reader_args = {
        C, qWt, vWt, release_sem, ready_imm0_sem, ready_imm1_sem, ready_dq0_sem,
        ready_dq1_sem, credit_prev_sem, credit_next_sem, credit_self_sem, endpoint1_sem,
        endpoint2_sem, Bt};
    for (const auto* t : {&query, &key, &value, &grad_output, &lse, &row_scalar, &grad_query,
                          &grad_key, &grad_value}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(reader_args);
    }
    const auto reader = CreateKernel(
        program, kReaderPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_args,
            .defines = sync_defines});

    std::vector<uint32_t> writer_args = {C, qWt, vWt, arrive_sem, release_sem, Bt};
    for (const auto* t : {&grad_key, &grad_value}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(writer_args);
    }
    const auto writer = CreateKernel(
        program, kWriterPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_args,
            .defines = sync_defines});

    const uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(static_cast<float>(d)));
    const uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);
    // sqrt(d): the exponential applies the softmax scale to its whole
    // argument, so the statistic it subtracts is divided by this first.
    const uint32_t inv_scaler = std::bit_cast<uint32_t>(std::sqrt(static_cast<float>(d)));
    const uint32_t custom_inf = std::bit_cast<uint32_t>(tt::tt_metal::hal::get_inf());
    std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    // P is copied into DST for the elementwise dS chain, so it keeps Float32
    // through the copy. Nothing else may: the transposed buffers feed matmul
    // Src registers, which do not take Float32.
    unpack_mode[tt::CBIndex::c_10] = UnpackToDestMode::UnpackToDestFp32;
    const auto compute = CreateKernel(
        program, kComputePath, region,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            // Half-sync DST holds four Float32 registers, and Bt = 2 needs
            // exactly four: Bt score tiles plus two shared scratch. Taller
            // blocks want the whole file, at the cost of the math-against-pack
            // pipelining that half sync buys.
            .dst_full_sync_en = Bt > 2U,
            .unpack_to_dest_mode = unpack_mode,
            .compile_args = {C, qWt, vWt, scaler, minus_one, custom_inf, block_size, Bt, inv_scaler},
            .defines = compute_defines});

    // Everything below is per group: the snake, and the barrier's coordinator
    // and multicast rectangle, all live inside one group's rectangle, which is
    // why several schedules share a grid without knowing about each other.
    for (uint32_t g = 0; g < layout.groups; ++g) {
        const auto origin = layout.group_origin[g];
        const auto logical_of = [&](uint32_t core) {
            const auto xy = placement_of(C, layout.group_width, core);
            return CoreCoord{origin.x + xy.x, origin.y + xy.y};
        };
        const auto noc_of_core = [&](uint32_t core) {
            return device->worker_core_from_logical_core(logical_of(core));
        };
        const auto coordinator = noc_of_core(1U);
        const auto mcast_start = device->worker_core_from_logical_core(origin);
        const auto mcast_end = device->worker_core_from_logical_core(
            CoreCoord{origin.x + layout.group_width - 1U, origin.y + layout.group_height - 1U});

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
                grad_output.buffer()->address(),
                lse.buffer()->address(),
                row_scalar.buffer()->address(),
                grad_query.buffer()->address(),
                grad_key.buffer()->address(),
                grad_value.buffer()->address(),
                static_cast<uint32_t>(prev.x),
                static_cast<uint32_t>(prev.y),
                static_cast<uint32_t>(next.x),
                static_cast<uint32_t>(next.y)};
            // Every core of this group, for the endpoint reads.
            for (uint32_t r = 1U; r <= C; ++r) {
                const auto rc = noc_of_core(r);
                rt.push_back(static_cast<uint32_t>(rc.x));
                rt.push_back(static_cast<uint32_t>(rc.y));
            }
            SetRuntimeArgs(program, reader, core, rt);
            SetRuntimeArgs(
                program, writer, core,
                {c, g, grad_key.buffer()->address(), grad_value.buffer()->address(),
                 static_cast<uint32_t>(coordinator.x), static_cast<uint32_t>(coordinator.y),
                 static_cast<uint32_t>(mcast_start.x), static_cast<uint32_t>(mcast_start.y),
                 static_cast<uint32_t>(mcast_end.x), static_cast<uint32_t>(mcast_end.y),
                 c == 1U ? 1U : 0U});
            SetRuntimeArgs(program, compute, core, {c});
        }
    }

    return cached_program_t{std::move(program), {reader, writer, compute, layout}};
}

void CyclicSDPABackwardProgramFactory::override_runtime_arguments(
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
        tensor_args.grad_output.buffer()->address(),
        tensor_args.log_sum_exp.buffer()->address(),
        tensor_args.row_scalar.buffer()->address(),
        output[0].buffer()->address(),
        output[1].buffer()->address(),
        output[2].buffer()->address()};
    const std::array<uint32_t, kWriterAddressCount> writer_addresses = {
        output[1].buffer()->address(), output[2].buffer()->address()};

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

}  // namespace ttml::metal::ops::cyclic_sdpa_bw::device
