// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/core_coord.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace vit {

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;
constexpr uint32_t TILE_HW = TILE_H * TILE_W;
constexpr uint32_t SINGLE_TILE_SIZE = sizeof(bfloat16) * TILE_HW;

inline uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline uint32_t round_up_to_tile(uint32_t n) { return div_ceil(n, TILE_W) * TILE_W; }
inline uint32_t num_tiles(uint32_t n) { return div_ceil(n, TILE_W); }

// Find largest divisor of total that is <= max_cores
inline uint32_t choose_num_cores(uint32_t total, uint32_t max_cores = 7) {
    uint32_t n = std::min(total, max_cores);
    while (n > 1 && total % n != 0) n--;
    return n;
}

struct MeshContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq;
    distributed::MeshCoordinateRange device_range;

    static MeshContext create(int device_id = 0) {
        auto mesh = distributed::MeshDevice::create_unit_mesh(device_id);
        auto& cq = mesh->mesh_command_queue();
        auto range = distributed::MeshCoordinateRange(mesh->shape());
        return {mesh, &cq, range};
    }
};

inline std::shared_ptr<distributed::MeshBuffer> create_dram_buffer(
    MeshContext& ctx, uint32_t total_bytes) {
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = SINGLE_TILE_SIZE, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_bytes};
    return distributed::MeshBuffer::create(buffer_config, dram_config, ctx.mesh_device.get());
}

inline void write_to_device(
    MeshContext& ctx,
    std::shared_ptr<distributed::MeshBuffer> buf,
    const std::vector<bfloat16>& data) {
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf, data, false);
}

inline void read_from_device(
    MeshContext& ctx,
    std::vector<bfloat16>& data,
    std::shared_ptr<distributed::MeshBuffer> buf) {
    distributed::EnqueueReadMeshBuffer(*ctx.cq, data, buf, true);
}

inline void run_program(MeshContext& ctx, Program& program) {
    distributed::MeshWorkload workload;
    workload.add_program(ctx.device_range, std::move(program));
    distributed::EnqueueMeshWorkload(*ctx.cq, workload, false);
    distributed::Finish(*ctx.cq);
}

inline float compute_pcc(const std::vector<bfloat16>& a, const std::vector<bfloat16>& b) {
    float x_mean = 0.0f, y_mean = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        x_mean += static_cast<float>(a[i]);
        y_mean += static_cast<float>(b[i]);
    }
    x_mean /= a.size();
    y_mean /= b.size();

    float cov = 0.0f, x_var = 0.0f, y_var = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float xd = static_cast<float>(a[i]) - x_mean;
        float yd = static_cast<float>(b[i]) - y_mean;
        cov += xd * yd;
        x_var += xd * xd;
        y_var += yd * yd;
    }
    return cov / (std::sqrt(x_var) * std::sqrt(y_var));
}

inline void golden_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& c,
    uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                acc += static_cast<float>(a[i * K + k]) * static_cast<float>(b[k * N + j]);
            }
            c[i * N + j] = bfloat16(acc);
        }
    }
}

}  // namespace vit
