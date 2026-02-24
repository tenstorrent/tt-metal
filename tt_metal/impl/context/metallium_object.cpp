// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "metallium_object.hpp"
#include "firmware_capability.hpp"
#include "get_platform_architecture.hpp"
#include "profiler_state_manager.hpp"
#include "tt_cluster.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt::tt_metal {

MetalliumObject::MetalliumObject(const std::shared_ptr<ContextDescriptor>& descriptor) {
    TT_ASSERT(descriptor->hal_ == nullptr);
    TT_ASSERT(descriptor->rtoptions_ == nullptr);
    TT_ASSERT(descriptor->cluster_ == nullptr);
    this->verify_fw_initialize_base_objects(descriptor);
}

MetalliumObject::~MetalliumObject() {
    control_plane_.reset();
    cluster_.reset();
    hal_.reset();
    rtoptions_.reset();
}

void MetalliumObject::verify_fw_initialize_base_objects(const std::shared_ptr<ContextDescriptor>& descriptor) {
    this->rtoptions_ = std::make_unique<llrt::RunTimeOptions>();

    if (descriptor->is_mock_device()) {
        this->rtoptions_->set_mock_cluster_desc(std::string(descriptor->mock_cluster_desc_path()));
    }

    const bool is_base_routing_fw_enabled =
        Cluster::is_base_routing_fw_enabled(Cluster::get_cluster_type_from_cluster_desc(*this->rtoptions_));
    const auto platform_arch = get_platform_architecture(*this->rtoptions_);

    cluster_ = std::make_unique<Cluster>(*this->rtoptions_);

    FirmwareCapabilityRequest req;
    req.enable_2_erisc_mode = this->rtoptions_->get_enable_2_erisc_mode();

    FirmwareCapabilityResult res;

    if (!check_firmware_capabilities(platform_arch, {.eth_fw = cluster_->get_ethernet_firmware_version()}, req, res)) {
        this->rtoptions_->set_enable_2_erisc_mode(res.enable_2_erisc_mode);
    }

    this->hal_ = std::make_unique<Hal>(
        platform_arch,
        is_base_routing_fw_enabled,
        this->rtoptions_->get_enable_2_erisc_mode(),
        get_profiler_dram_bank_size_for_hal_allocation(*this->rtoptions_));

    this->rtoptions_->ParseAllFeatureEnv(*hal_);
    this->cluster_->set_hal(hal_.get());
}

}  // namespace tt::tt_metal
