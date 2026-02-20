// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "firmware_initializer.hpp"

#include "impl/context/context_descriptor.hpp"

namespace tt::tt_metal {

FirmwareInitializer::FirmwareInitializer(std::shared_ptr<const ContextDescriptor> descriptor) :
    hal_(descriptor->hal()),
    cluster_(descriptor->cluster()),
    rtoptions_(descriptor->rtoptions()),
    descriptor_(std::move(descriptor)) {}

void FirmwareInitializer::post_teardown() {}

void FirmwareInitializer::init_by_device_ids(const std::set<tt::ChipId>& /*device_ids*/) {}

}  // namespace tt::tt_metal
