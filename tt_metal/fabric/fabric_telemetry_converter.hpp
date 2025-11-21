// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/fabric_telemetry.hpp>

#include "tt_metal/llrt/hal/generated/fabric_telemetry.hpp"

namespace tt::tt_metal::fabric_telemetry_converter {

void pack_static_info_to_hal(
    const tt::tt_fabric::FabricTelemetryStaticInfo& src, ::tt::tt_fabric::fabric_telemetry::StaticInfo::View dst);

void pack_dynamic_info_to_hal(
    const tt::tt_fabric::FabricTelemetryDynamicInfo& src, ::tt::tt_fabric::fabric_telemetry::DynamicInfo::View dst);

void pack_snapshot_to_hal(
    const tt::tt_fabric::FabricTelemetrySnapshot& src, ::tt::tt_fabric::fabric_telemetry::FabricTelemetry::View dst);

void pack_snapshot_to_hal(
    const tt::tt_fabric::FabricTelemetrySnapshot& src,
    ::tt::tt_fabric::fabric_telemetry::FabricTelemetryStaticOnly::View dst);

tt::tt_fabric::FabricTelemetryStaticInfo unpack_static_info_from_hal(
    ::tt::tt_fabric::fabric_telemetry::StaticInfo::ConstView src);

tt::tt_fabric::FabricTelemetryDynamicInfo unpack_dynamic_info_from_hal(
    ::tt::tt_fabric::fabric_telemetry::DynamicInfo::ConstView src);

tt::tt_fabric::FabricTelemetrySnapshot unpack_snapshot_from_hal(
    ::tt::tt_fabric::fabric_telemetry::FabricTelemetry::ConstView src);

tt::tt_fabric::FabricTelemetrySnapshot unpack_snapshot_from_hal(
    ::tt::tt_fabric::fabric_telemetry::FabricTelemetryStaticOnly::ConstView src);

}  // namespace tt::tt_metal::fabric_telemetry_converter
