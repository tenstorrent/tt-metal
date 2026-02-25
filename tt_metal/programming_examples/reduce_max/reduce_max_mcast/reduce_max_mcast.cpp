// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core row-wise maximum with NoC multicast for the scaler tile.
//
// This is identical to reduce_max_multi_core in computation, but differs in how
// the constant scaler tile (all 1.0) reaches each core:
//
//   multi_core:   every core independently reads the scaler tile from DRAM.
//                 Cost: num_cores DRAM reads for a constant tile.
//
//   reduce_max_mcast (this file):
//                 One designated "sender" core reads the scaler from DRAM once,
//                 then multicasts it to all "receiver" cores via the NoC.
//                 Cost: 1 DRAM read + 1 NoC multicast.
//
// Core roles:
//   sender   = first core in all_cores (logical {0,0}).
//   receivers = all remaining cores in all_cores.
//
// Multicast protocol (two semaphores):
//   sem_sender:   Receivers increment it when their scaler CB slot is reserved.
//                 Sender waits until the count equals num_receivers.
//   sem_receiver: Sender pre-sets it to VALID locally, then multicasts that value
//                 to every receiver once the tile write-multicast is issued.
//                 Receivers spin-wait on VALID before calling cb_push_back.
//
// Physical coordinates for the multicast rectangle are computed from the
// bounding box of all receiver cores in physical NoC space.

#include <fmt/base.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// CPU reference: for each row i, output[i] = max(input[i*N .. i*N+N-1]).
void golden_row_max(const vector<bfloat16>& input, vector<bfloat16>& output, uint32_t M, uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j < N; j++) {
            float val = static_cast<float>(input[i * N + j]);
            row_max = std::max(row_max, val);
        }
        output[i] = bfloat16(row_max);
    }
}

void reduce_max_mcast(
    const vector<bfloat16>& input,
    vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    const shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};

    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // -------------------------------------------------------------------------
    // Work distribution (identical to multi_core)
    // -------------------------------------------------------------------------
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        split_work_to_cores(core_grid, Mt);

    // -------------------------------------------------------------------------
    // Sender / receiver split
    // -------------------------------------------------------------------------
    // Sender = first core in all_cores (logical {0,0} in practice).
    CoreCoord sender_logical = all_cores.ranges().cbegin()->start_coord;
    CoreCoord sender_phys    = mesh_device->worker_core_from_logical_core(sender_logical);

    // Receivers = all other active cores.
    vector<CoreRange> receiver_range_list;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            if (core != sender_logical) {
                receiver_range_list.push_back(CoreRange(core, core));
            }
        }
    }
    uint32_t num_receivers = num_cores - 1;

    // Physical bounding box of all receiver cores.
    // noc_async_write_multicast targets a physical rectangle; all cores within it
    // will receive the write.  We use the tightest enclosing rectangle of the
    // receiver set.  Any physical cores inside the box but outside the receiver set
    // are idle or running unrelated work at a different L1 address, so the write
    // is harmless.
    uint32_t recv_phys_start_x = UINT32_MAX, recv_phys_start_y = UINT32_MAX;
    uint32_t recv_phys_end_x   = 0,          recv_phys_end_y   = 0;
    CoreRangeSet receiver_cores;

    if (num_receivers > 0) {
        set<CoreRange> receiver_range_set(receiver_range_list.begin(), receiver_range_list.end());
        receiver_cores = CoreRangeSet(receiver_range_set);

        for (const auto& range : receiver_cores.ranges()) {
            CoreCoord phys_s = mesh_device->worker_core_from_logical_core(range.start_coord);
            CoreCoord phys_e = mesh_device->worker_core_from_logical_core(range.end_coord);
            recv_phys_start_x = std::min({recv_phys_start_x, (uint32_t)phys_s.x, (uint32_t)phys_e.x});
            recv_phys_start_y = std::min({recv_phys_start_y, (uint32_t)phys_s.y, (uint32_t)phys_e.y});
            recv_phys_end_x   = std::max({recv_phys_end_x,   (uint32_t)phys_s.x, (uint32_t)phys_e.x});
            recv_phys_end_y   = std::max({recv_phys_end_y,   (uint32_t)phys_s.y, (uint32_t)phys_e.y});
        }
    }

    // -------------------------------------------------------------------------
    // DRAM buffers
    // -------------------------------------------------------------------------
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig input_buf_config{.size = single_tile_size * Mt * Nt};
    auto src_dram_buffer = distributed::MeshBuffer::create(input_buf_config, dram_config, mesh_device.get());

    // Scaler tile: read by the sender only; receivers get it via multicast.
    distributed::ReplicatedBufferConfig scaler_buf_config{.size = single_tile_size};
    auto scaler_dram_buffer = distributed::MeshBuffer::create(scaler_buf_config, dram_config, mesh_device.get());

    distributed::ReplicatedBufferConfig output_buf_config{.size = single_tile_size * Mt};
    auto dst_dram_buffer = distributed::MeshBuffer::create(output_buf_config, dram_config, mesh_device.get());

    // -------------------------------------------------------------------------
    // Circular buffers (same layout on all cores so multicast lands correctly)
    // -------------------------------------------------------------------------
    // IMPORTANT: CBs must be created in the same order on every core.  The
    // multicast writes the scaler tile to get_write_ptr(cb_id_scaler) on the
    // sender, which is the same L1 address on all cores because the allocator
    // assigns CBs identically.
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    uint32_t src_cb_index = CBIndex::c_0;
    tt_metal::CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(2 * single_tile_size, {{src_cb_index, cb_data_format}})
            .set_page_size(src_cb_index, single_tile_size));

    uint32_t scaler_cb_index = CBIndex::c_1;
    tt_metal::CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(single_tile_size, {{scaler_cb_index, cb_data_format}})
            .set_page_size(scaler_cb_index, single_tile_size));

    uint32_t output_cb_index = CBIndex::c_16;
    tt_metal::CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(2 * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size));

    // -------------------------------------------------------------------------
    // Semaphores
    // -------------------------------------------------------------------------
    // Both semaphores are created on ALL cores so every core owns a local copy at
    // the same L1 address (needed for the multicast to hit the correct location).
    uint32_t sem_sender   = CreateSemaphore(program, all_cores, 0);  // receivers increment
    uint32_t sem_receiver = CreateSemaphore(program, all_cores, 0);  // sender multicasts VALID

    // -------------------------------------------------------------------------
    // Kernels
    // -------------------------------------------------------------------------
    CoreRangeSet sender_core_set(CoreRange(sender_logical, sender_logical));

    // Sender reader: reads scaler from DRAM, multicasts it, streams input tiles.
    vector<uint32_t> sender_ct_args;
    TensorAccessorArgs(*src_dram_buffer).append_to(sender_ct_args);
    TensorAccessorArgs(*scaler_dram_buffer).append_to(sender_ct_args);
    auto sender_reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX
        "reduce_max/reduce_max_mcast/kernels/dataflow/reader_reduce_max_mcast_sender.cpp",
        sender_core_set,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = sender_ct_args});

    // Receiver reader: waits for mcast scaler, then streams input tiles.
    KernelHandle receiver_reader_id = 0;
    if (num_receivers > 0) {
        vector<uint32_t> receiver_ct_args;
        TensorAccessorArgs(*src_dram_buffer).append_to(receiver_ct_args);
        receiver_reader_id = tt_metal::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX
            "reduce_max/reduce_max_mcast/kernels/dataflow/reader_reduce_max_mcast_receiver.cpp",
            receiver_cores,
            tt_metal::DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = receiver_ct_args});
    }

    // Writer: unchanged from multi_core.
    vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_ct_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "reduce_max/reduce_max_mcast/kernels/dataflow/writer_reduce_max_mcast.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    // Compute: unchanged from multi_core.
    vector<uint32_t> compute_ct_args = {Nt};
    auto compute_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "reduce_max/reduce_max_mcast/kernels/compute/reduce_max.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_ct_args});

    // -------------------------------------------------------------------------
    // Per-core runtime arguments
    // -------------------------------------------------------------------------
    // Collect (mt_start, mt_count) for every core first, then apply in one pass.
    map<CoreCoord, pair<uint32_t, uint32_t>> core_work;
    {
        uint32_t mt_offset = 0;
        auto work_groups = {
            std::make_pair(core_group_1, work_per_core_1),
            std::make_pair(core_group_2, work_per_core_2)};
        for (const auto& [ranges, mt_count] : work_groups) {
            for (const auto& range : ranges.ranges()) {
                for (const auto& core : range) {
                    core_work[core] = {mt_offset, mt_count};
                    mt_offset += mt_count;
                }
            }
        }
    }

    uint32_t src_addr    = src_dram_buffer->address();
    uint32_t scaler_addr = scaler_dram_buffer->address();
    uint32_t dst_addr    = dst_dram_buffer->address();

    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            auto [mt_start, mt_count] = core_work[core];

            tt_metal::SetRuntimeArgs(program, writer_id,  core, {dst_addr, mt_start, mt_count});
            tt_metal::SetRuntimeArgs(program, compute_id, core, {mt_count});

            if (core == sender_logical) {
                // Sender kernel uses NOC1 (RISCV_1_default).  On NOC1 the multicast
                // rectangle must be encoded with the LARGER coordinate as "start" and
                // the SMALLER coordinate as "end" (opposite of NOC0 convention).
                tt_metal::SetRuntimeArgs(program, sender_reader_id, core, {
                    src_addr, scaler_addr,
                    mt_start, mt_count, Nt,
                    recv_phys_end_x,   recv_phys_end_y,    // NOC1 start = max coords
                    recv_phys_start_x, recv_phys_start_y,  // NOC1 end   = min coords
                    num_receivers, sem_sender, sem_receiver
                });
            } else {
                tt_metal::SetRuntimeArgs(program, receiver_reader_id, core, {
                    src_addr, mt_start, mt_count, Nt,
                    (uint32_t)sender_phys.x, (uint32_t)sender_phys.y,
                    sem_sender, sem_receiver
                });
            }
        }
    }

    // -------------------------------------------------------------------------
    // Upload inputs, execute, read back
    // -------------------------------------------------------------------------
    vector<bfloat16> scaler_tile(TILE_HEIGHT * TILE_WIDTH, bfloat16(1.0f));

    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, input, false);
    distributed::EnqueueWriteMeshBuffer(cq, scaler_dram_buffer, scaler_tile, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);
}

int main() {
    bool pass = true;

    try {
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        constexpr uint32_t M = 5120;
        constexpr uint32_t N = 5120;

        static_assert(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT (32)");
        static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH (32)");

        mt19937 rng(42);
        uniform_real_distribution<float> dist(-10.0f, -1.0f);
        vector<bfloat16> src_vec(M * N);
        for (auto& v : src_vec) {
            v = bfloat16(dist(rng));
        }

        vector<bfloat16> golden_vec(M);
        golden_row_max(src_vec, golden_vec, M, N);

        vector<bfloat16> src_tilized = tilize_nfaces(src_vec, M, N);

        uint32_t Mt = M / TILE_HEIGHT;
        vector<bfloat16> result_tilized(Mt * TILE_HEIGHT * TILE_WIDTH, bfloat16(0.0f));
        reduce_max_mcast(src_tilized, result_tilized, M, N, mesh_device);

        vector<bfloat16> result_untilized = untilize_nfaces(result_tilized, M, TILE_WIDTH);
        vector<bfloat16> result_max(M);
        for (uint32_t i = 0; i < M; i++) {
            result_max[i] = result_untilized[i * TILE_WIDTH];
        }

        constexpr float kMaxAllowedError = 0.01f;
        float max_abs_err = 0.0f;
        uint32_t num_mismatches = 0;
        for (uint32_t i = 0; i < M; i++) {
            float err = std::abs(static_cast<float>(golden_vec[i]) - static_cast<float>(result_max[i]));
            if (err > kMaxAllowedError) {
                num_mismatches++;
                if (num_mismatches <= 5) {
                    fmt::print(
                        "Mismatch at row {:3d}: golden={:.6f}  result={:.6f}  diff={:.6f}\n",
                        i,
                        static_cast<float>(golden_vec[i]),
                        static_cast<float>(result_max[i]),
                        err);
                }
            }
            max_abs_err = std::max(max_abs_err, err);
        }

        fmt::print("Row-wise max (mcast scaler) on {}×{} matrix\n", M, N);
        fmt::print("Max absolute error : {:.6f}\n", max_abs_err);
        fmt::print("Number of mismatches: {} / {}\n", num_mismatches, M);

        TT_FATAL(
            num_mismatches == 0,
            "Row-wise max result does not match the golden reference ({} mismatches)",
            num_mismatches);

        pass &= mesh_device->close();

    } catch (const exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
