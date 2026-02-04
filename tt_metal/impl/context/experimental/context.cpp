// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <mutex>
#include <tt-metalium/experimental/context_descriptor.hpp>
#include <tt-metalium/experimental/runtime.hpp>
#include <tt-metalium/dispatch_core_common.hpp>

namespace tt::tt_metal::experimental {

class ContextDescriptor::ContextDescriptorImpl {
public:
    bool is_bound_ = false;
};

ContextDescriptor::ContextDescriptor(
    int num_cqs,
    int l1_small_size,
    int trace_region_size,
    int worker_l1_size,
    DispatchCoreConfig dispatch_core_config,
    tt_fabric::FabricConfig fabric_config,
    tt_fabric::FabricReliabilityMode reliability_mode,
    std::optional<uint8_t> num_routing_planes,
    tt_fabric::FabricTensixConfig fabric_tensix_config,
    tt_fabric::FabricUDMMode fabric_udm_mode,
    tt_fabric::FabricManagerMode fabric_manager,
    tt_fabric::FabricRouterConfig router_config) :
    num_cqs_(num_cqs),
    l1_small_size_(l1_small_size),
    trace_region_size_(trace_region_size),
    worker_l1_size_(worker_l1_size),
    dispatch_core_config_(dispatch_core_config),
    fabric_config_(fabric_config),
    reliability_mode_(reliability_mode),
    num_routing_planes_(num_routing_planes),
    fabric_tensix_config_(fabric_tensix_config),
    fabric_udm_mode_(fabric_udm_mode),
    fabric_manager_(fabric_manager),
    router_config_(router_config),
    impl_(std::make_unique<ContextDescriptorImpl>()) {}

ContextDescriptor::~ContextDescriptor() = default;

ContextDescriptor::ContextDescriptor(ContextDescriptor&&) noexcept = default;

ContextDescriptor& ContextDescriptor::operator=(ContextDescriptor&&) noexcept = default;

int ContextDescriptor::get_num_cqs() const { return num_cqs_; }
int ContextDescriptor::get_l1_small_size() const { return l1_small_size_; }
int ContextDescriptor::get_trace_region_size() const { return trace_region_size_; }
int ContextDescriptor::get_worker_l1_size() const { return worker_l1_size_; }
DispatchCoreConfig ContextDescriptor::get_dispatch_core_config() const { return dispatch_core_config_; }
tt_fabric::FabricConfig ContextDescriptor::get_fabric_config() const { return fabric_config_; }
tt_fabric::FabricReliabilityMode ContextDescriptor::get_reliability_mode() const { return reliability_mode_; }
std::optional<uint8_t> ContextDescriptor::get_num_routing_planes() const { return num_routing_planes_; }
tt_fabric::FabricTensixConfig ContextDescriptor::get_fabric_tensix_config() const { return fabric_tensix_config_; }
tt_fabric::FabricUDMMode ContextDescriptor::get_fabric_udm_mode() const { return fabric_udm_mode_; }
tt_fabric::FabricManagerMode ContextDescriptor::get_fabric_manager() const { return fabric_manager_; }
tt_fabric::FabricRouterConfig ContextDescriptor::get_router_config() const { return router_config_; }
bool ContextDescriptor::is_bound() const { return impl_->is_bound_; }
void ContextDescriptor::set_bound(bool bound) { impl_->is_bound_ = bound; }

}  // namespace tt::tt_metal::experimental
