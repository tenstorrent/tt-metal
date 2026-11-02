// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

using hw_format_t = std::underlying_type_t<DataFormat>;

// Quasar HW DataFormat code for a host DataFormat (see hw_data_format.cpp).
// The one-argument form is the Quasar table. Wormhole and Blackhole host values are already
// the hardware codes, so call that form only for ARCH::QUASAR.
hw_format_t host_data_format_to_hw(DataFormat f);

// Arch seam for the table above. Quasar gets the remapped code; every other arch gets the host value.
hw_format_t host_data_format_to_hw(DataFormat f, tt::ARCH arch);

}  // namespace tt::tt_metal
