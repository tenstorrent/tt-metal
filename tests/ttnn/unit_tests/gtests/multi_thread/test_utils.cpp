// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "test_utils.hpp"

#include <fmt/base.h>
#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <utility>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn {
namespace operations {
namespace unary {
enum class UnaryOpType;
struct UnaryWithParam;
}  // namespace unary
}  // namespace operations
}  // namespace ttnn
namespace tt {
namespace tt_metal {
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::distributed::test {

static constexpr size_t TEST_WORKERS_SUBDEVICE_INDEX = 0;
static constexpr size_t TEST_EDM_FABRIC_SUBDEVICE_INDEX = 1;

using namespace tt;
using namespace tt_metal;

Tensor dispatch_ops_to_device(IDevice* dev, Tensor input_tensor, QueueId cq_id) {
    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::UnaryWithParam;

    Tensor output_tensor = ttnn::mul_sfpu(cq_id, input_tensor, 2);
    for (int i = 0; i < 3; i++) {
        output_tensor = ttnn::neg(cq_id, output_tensor);
        output_tensor = ttnn::neg(cq_id, output_tensor);
        output_tensor = ttnn::mul_sfpu(cq_id, output_tensor, 2);
    }
    output_tensor = ttnn::neg(cq_id, output_tensor);
    output_tensor = ttnn::mul_sfpu(cq_id, output_tensor, 2);
    output_tensor = ttnn::add_sfpu(cq_id, output_tensor, 128);
    return output_tensor;
}

SubdeviceInfo create_subdevices(const std::vector<IDevice*>& devices) {
    SubdeviceInfo subdevice_info;
    std::unordered_map<chip_id_t, SubDeviceManagerId> sub_device_manager_ids;
    for (auto device : devices) {
        const auto& tensix_sub_device =
            tt_metal::SubDevice(std::array{device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0})});
        const auto& eth_sub_device = tt_metal::SubDevice(
            std::array{CoreRangeSet(), device->worker_cores(HalProgrammableCoreType::ACTIVE_ETH, SubDeviceId{0})});
        subdevice_info.sub_device_managers.insert(
            {device->id(), device->create_sub_device_manager({tensix_sub_device, eth_sub_device}, 0)});
        device->load_sub_device_manager(subdevice_info.sub_device_managers.at(device->id()));
        subdevice_info.worker_subdevice_id.insert(
            {device->id(), device->get_sub_device_ids().at(TEST_WORKERS_SUBDEVICE_INDEX)});
        subdevice_info.fabric_subdevice_id.insert(
            {device->id(), device->get_sub_device_ids().at(TEST_EDM_FABRIC_SUBDEVICE_INDEX)});
        device->set_sub_device_stall_group({subdevice_info.worker_subdevice_id.at(device->id())});
    }

    return subdevice_info;
}

void build_and_enqueue(const std::vector<IDevice*>& devices, std::vector<Program>& programs, bool enqueue_only) {
    TT_FATAL(
        devices.size() == programs.size(),
        "Number of devices must match number of programs when calling build_and_enqueue in test");
    if (!enqueue_only) {
        for (size_t i = 0; i < devices.size(); i++) {
            tt::tt_metal::detail::CompileProgram(devices[i], programs[i]);
        }
    }
    for (size_t i = 0; i < devices.size(); i++) {
        tt_metal::EnqueueProgram(devices[i]->command_queue(), programs[i], false);
    }
}

void setup_test_with_persistent_fabric(
    const std::vector<IDevice*>& devices,
    std::optional<SubdeviceInfo>& subdevice_managers,
    std::optional<std::vector<Program>>& fabric_programs,
    std::vector<Program*>& fabric_program_ptrs,
    std::optional<tt::tt_fabric::EdmLineFabricOpInterface>& line_fabric,
    std::optional<size_t> num_links) {
    log_info(tt::LogTest, "Enabling persistent fabric");
    fabric_programs = std::vector<Program>(devices.size());
    subdevice_managers = create_subdevices(devices);
    std::transform(
        fabric_programs->begin(), fabric_programs->end(), std::back_inserter(fabric_program_ptrs), [](auto& p) {
            return &p;
        });

    line_fabric = tt::tt_fabric::EdmLineFabricOpInterface(devices, fabric_program_ptrs, num_links.value_or(1));
    line_fabric->set_firmware_context_switch_interval(0);

    TT_FATAL(fabric_programs.has_value(), "Fabric programs must be set if fabric is enabled");
    TT_FATAL(devices.size() == fabric_programs->size(), "Number of devices must match number of programs");

    log_info(tt::LogTest, "Building EDM kernels");
    line_fabric->build_kernels();
    build_and_enqueue(devices, *fabric_programs);
}

void persistent_fabric_teardown_sequence(
    const std::vector<IDevice*>& devices,
    std::optional<SubdeviceInfo>& subdevice_managers,
    tt::tt_fabric::EdmLineFabricOpInterface& line_fabric,
    tt::tt_fabric::TerminationSignal termination_mode) {
    log_info(tt::LogTest, "Tearing down fabric");

    // Wait for workers to finish
    auto d0_worker_subdevice = devices[0]->get_sub_device_ids()[TEST_WORKERS_SUBDEVICE_INDEX];
    tt_metal::Finish(devices[0]->command_queue(), {subdevice_managers->worker_subdevice_id.at(devices[0]->id())});

    // Teardown the fabric
    line_fabric.teardown_from_host(termination_mode);

    // wait for fabric teardown to finish
    std::ranges::for_each(devices, [&](IDevice* d) {
        tt_metal::Finish(d->command_queue(), {subdevice_managers->fabric_subdevice_id.at(d->id())});
    });
}

std::tuple<
    ttnn::global_semaphore::MultiDeviceGlobalSemaphore,
    ttnn::global_semaphore::MultiDeviceGlobalSemaphore,
    ttnn::global_semaphore::MultiDeviceGlobalSemaphore>
create_global_semaphores(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, IDevice* device) {
    auto from_remote_multi_device_global_semaphore = ttnn::global_semaphore::create_global_semaphore_with_same_address(
        mesh_device->get_devices(),
        device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                             // initial value
        tt::tt_metal::BufferType::L1,  // buffer type
        10                             // attempts
    );

    auto to_remote_multi_device_global_semaphore = ttnn::global_semaphore::create_global_semaphore_with_same_address(
        mesh_device->get_devices(),
        device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                             // initial value
        tt::tt_metal::BufferType::L1,  // buffer type
        10                             // attempts
    );

    auto multi_device_global_semaphore = ttnn::global_semaphore::create_global_semaphore_with_same_address(
        mesh_device->get_devices(),
        device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                             // initial value
        tt::tt_metal::BufferType::L1,  // buffer type
        10                             // attempts
    );

    return {
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        multi_device_global_semaphore};
}
}  // namespace ttnn::distributed::test
