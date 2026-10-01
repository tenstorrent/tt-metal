// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <memory>
#include <random>
#include <string>

#include "core/distributed/ccl_resources.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/mesh_device.hpp"
#include "core/tt_profiler.hpp"
#include "graph.hpp"

namespace ttml::autograd {

enum class GradMode { ENABLED, DISABLED };

struct DistributedConfig {
    bool enable_ddp = false;
    bool enable_tp = false;
    bool enable_cp = false;
};

class ParallelismContext {
public:
    // Configure from device config flags
    // For TP+DP: dp_axis=0, tp_axis=1
    // For TP only: tp_axis=0
    // For DP only: dp_axis=0
    ParallelismContext(const ttnn::distributed::MeshDevice& mesh_device, const DistributedConfig& config);

    // Axis queries
    [[nodiscard]] std::optional<uint32_t> get_ddp_axis() const {
        return m_ddp_axis;
    }
    [[nodiscard]] std::optional<uint32_t> get_tp_axis() const {
        return m_tp_axis;
    }
    [[nodiscard]] std::optional<uint32_t> get_cp_axis() const {
        return m_cp_axis;
    }

    // Size queries (computed from mesh_device->shape())
    [[nodiscard]] const uint32_t get_ddp_size() const;
    [[nodiscard]] const uint32_t get_tp_size() const;
    [[nodiscard]] const uint32_t get_cp_size() const;

    [[nodiscard]] const bool is_tp_enabled() const {
        return m_tp_axis.has_value();
    }
    [[nodiscard]] const bool is_ddp_enabled() const {
        return m_ddp_axis.has_value();
    }
    [[nodiscard]] const bool is_cp_enabled() const {
        return m_cp_axis.has_value();
    }

private:
    std::optional<uint32_t> m_ddp_axis = std::nullopt;
    std::optional<uint32_t> m_tp_axis = std::nullopt;
    std::optional<uint32_t> m_cp_axis = std::nullopt;
    uint32_t m_num_cp_devices = 1U;
    uint32_t m_num_ddp_devices = 1U;
    uint32_t m_num_tp_devices = 1U;
};

class AutoContext {
public:
    // Delete copy constructor and assignment operator to prevent copying
    AutoContext(const AutoContext&) = delete;
    AutoContext& operator=(const AutoContext&) = delete;
    AutoContext(AutoContext&&) = delete;
    AutoContext& operator=(AutoContext&&) = delete;
    // Static method to access the singleton instance
    static AutoContext& get_instance();

    std::mt19937& get_generator();
    void set_generator(const std::mt19937& generator);

    [[nodiscard]] std::string get_generator_state() const;
    void set_generator_state(const std::string& state);

    void set_seed(uint32_t seed);

    [[nodiscard]] uint32_t get_seed() const;

    std::optional<NodeId> add_backward_node(GradFunction&& grad_function, std::span<NodeId> links);

    void reset_graph();

    void set_gradient_mode(GradMode mode);

    [[nodiscard]] GradMode get_gradient_mode() const;

    // Backward-pass bookkeeping. Tensor::backward() brackets its node loop with
    // enter/exit; the depth counter (not a flag) is needed because a grad function
    // may itself run a nested backward (gradient checkpointing recomputes a block
    // forward and calls backward on it). Lets module-level hooks (e.g. FSDP) tell a
    // recompute-forward, which is immediately followed by that block's backward,
    // apart from a regular forward.
    void enter_backward();
    void exit_backward();
    [[nodiscard]] bool is_backward_in_progress() const;

    // CCL sub-device. Splits every chip's Tensix grid into a compute sub-device (id 0) and a CCL
    // sub-device (id 1: the rightmost `num_columns` columns or the bottom `num_rows` rows; exactly
    // one of the two is non-zero). Programs on different sub-devices run concurrently, so
    // collectives issued on the CCL sub-device from the second command queue overlap with compute
    // on the first. The device reports the compute rectangle as its compute grid from then on, so
    // every op that sizes itself from compute_with_storage_grid_size() stays off the CCL cores.
    // Requires the device to have been opened with two command queues. Ordering between the two
    // queues is the caller's job (see ttml.fsdp and docs/FSDP.md).
    void enable_ccl_sub_device(uint32_t num_columns, uint32_t num_rows);
    [[nodiscard]] bool has_ccl_sub_device() const;
    [[nodiscard]] std::optional<tt::tt_metal::SubDeviceId> ccl_sub_device_id() const;
    [[nodiscard]] tt::tt_metal::SubDeviceId compute_sub_device_id() const;
    // The chip's whole compute grid, ignoring any CCL sub-device (peak-FLOPS accounting, and
    // resources such as global semaphores that CCL kernels on the reserved cores must find too).
    [[nodiscard]] tt::tt_metal::CoreCoord full_compute_grid_size();

    ~AutoContext() = default;  // to make it work with unique_ptr.

    [[nodiscard]] ttnn::distributed::MeshDevice& get_device();
    [[nodiscard]] std::shared_ptr<ttnn::distributed::MeshDevice> get_device_ptr();

    [[nodiscard]] tt::tt_metal::distributed::MeshShape get_mesh_shape() const;

    // num_command_queues: 1 (default) or 2. The second hardware queue is for collectives (see
    // enable_ccl_sub_device): launched on their own queue, they never stall compute launches.
    void open_device(
        const tt::tt_metal::distributed::MeshShape& mesh_shape = tt::tt_metal::distributed::MeshShape(1, 1),
        const std::vector<int>& device_ids = std::vector<int>{},
        size_t num_command_queues = 1);
    [[nodiscard]] size_t num_command_queues() const;

    void close_device();

    void initialize_distributed_context(int argc, char** argv);

    [[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context()
        const;

    core::TTProfiler& get_profiler();
    void close_profiler();

    [[nodiscard]] core::distributed::CCLResources& get_ccl_resources();

    void initialize_socket_manager(ttnn::distributed::SocketType socket_type);
    [[nodiscard]] core::distributed::SocketManager& get_socket_manager();

    [[nodiscard]] const ParallelismContext& get_parallelism_context() const;

    [[nodiscard]] bool is_parallelism_context_initialized() const;

    void initialize_parallelism_context(const DistributedConfig& config);

private:
    AutoContext();
    uint32_t m_seed = 5489U;
    std::mt19937 m_generator;

    GradMode m_grads_mode = GradMode::ENABLED;
    uint32_t m_backward_depth = 0U;
    size_t m_num_command_queues = 1;
    std::optional<tt::tt_metal::CoreCoord> m_full_compute_grid;  // set when the CCL sub-device narrows the grid
    std::optional<tt::tt_metal::SubDeviceId> m_ccl_sub_device_id;

    Graph m_graph;
    tt::tt_metal::distributed::MeshShape m_mesh_shape = tt::tt_metal::distributed::MeshShape(1, 1);
    std::unique_ptr<core::MeshDevice> m_device;

    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> m_distributed_context;
    std::unique_ptr<core::TTProfiler> m_profiler;

    std::unique_ptr<core::distributed::CCLResources> m_ccl_resources;

    std::unique_ptr<core::distributed::SocketManager> m_socket_manager;

    std::unique_ptr<ParallelismContext> m_parallelism_context;

    friend class ttsl::Indestructible<AutoContext>;
};

inline auto& ctx() {
    return AutoContext::get_instance();
}
}  // namespace ttml::autograd
