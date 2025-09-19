// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <metal_soc_descriptor.h>
#include <tt_backend_api_types.hpp>
#include <unordered_map>

#include <umd/device/cluster.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

struct metal_SocDescriptor;

namespace ll_api {

void configure_static_tlbs(
    tt::ARCH arch, chip_id_t mmio_device_id, const metal_SocDescriptor& sdesc, tt::umd::Cluster& device_driver);

}  // namespace ll_api
