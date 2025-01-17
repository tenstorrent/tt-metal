// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "umd/device/device_api_metal.h"
#include <tt_backend_api_types.hpp>
#include <metal_soc_descriptor.h>

#include <unordered_map>

namespace ll_api {

void configure_static_tlbs(
    tt::ARCH arch, chip_id_t mmio_device_id, const metal_SocDescriptor& sdesc, tt_device& device_driver);

}  // namespace ll_api
