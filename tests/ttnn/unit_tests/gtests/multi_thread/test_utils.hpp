// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <tt-metalium/mesh_device.hpp>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/system_memory_manager.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/erisc_datamover_builder_helper.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace ccl {
class EdmLineFabricOpInterface;
}  // namespace ccl
}  // namespace ttnn
namespace tt {
namespace tt_metal {
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::distributed::test {

using namespace tt;
using namespace tt_metal;

Tensor dispatch_ops_to_device(IDevice* dev, Tensor input_tensor, QueueId cq_id);

struct SubdeviceInfo {
    std::unordered_map<chip_id_t, SubDeviceManagerId> sub_device_managers;
    std::unordered_map<chip_id_t, SubDeviceId> worker_subdevice_id;
    std::unordered_map<chip_id_t, SubDeviceId> fabric_subdevice_id;
};

SubdeviceInfo create_subdevices(const std::vector<IDevice*>& devices);

void build_and_enqueue(const std::vector<IDevice*>& devices, std::vector<Program>& programs, bool enqueue_only = false);

void setup_test_with_persistent_fabric(
    const std::vector<IDevice*>& devices,
    std::vector<Program>& programs,
    std::optional<SubdeviceInfo>& subdevice_managers,
    std::optional<std::vector<Program>>& fabric_programs,
    std::vector<Program*>& fabric_program_ptrs,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& line_fabric,
    bool enable_persistent_fabric,
    std::optional<size_t> num_links = std::nullopt);

void persistent_fabric_teardown_sequence(
    const std::vector<IDevice*>& devices,
    std::optional<SubdeviceInfo>& subdevice_managers,
    ttnn::ccl::EdmLineFabricOpInterface& line_fabric,
    tt::tt_fabric::TerminationSignal termination_mode = tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE);

std::tuple<
    ttnn::global_semaphore::MultiDeviceGlobalSemaphore,
    ttnn::global_semaphore::MultiDeviceGlobalSemaphore,
    ttnn::global_semaphore::MultiDeviceGlobalSemaphore>
create_global_semaphores(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, IDevice* device);

}  // namespace ttnn::distributed::test
