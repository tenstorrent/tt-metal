// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/experimental/context/metalium_env.hpp>
#include <memory>

namespace tt::tt_metal {

class IDevice;

namespace experimental {

int CreateContext(const std::shared_ptr<MetaliumEnv>& env);
void DestroyContext(int context_id);
void DestroyAllContexts();

namespace detail {

// Create devices associated to the given context ID
// TODO: As MetaliumEnv is not yet in the public API, we need to pass the context ID to the function.
std::map<ChipId, IDevice*> CreateDevices(
    int context_id,
    const std::vector<ChipId>& device_ids,
    uint8_t num_hw_cqs = 1,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const tt_metal::DispatchCoreConfig& dispatch_core_config = tt_metal::DispatchCoreConfig{},
    const std::vector<uint32_t>& l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
    bool init_profiler = true,
    [[deprecated]] bool ignored = false,  // This argument was not used
    bool initialize_fabric_and_dispatch_fw = true);

}  // namespace detail
}  // namespace experimental
}  // namespace tt::tt_metal
