// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal::experimental::hal {

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific L1 Size
 *
 * @return Size in bytes of the L1 SRAM buffer associated with the currently present architecture.
 */
uint32_t get_l1_size();

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific DRAM alignment.
 *
 * @return Alignment requirement in bytes
 */
uint32_t get_dram_alignment();

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific L1 alignment.
 *
 * @return Alignment requirement in bytes
 */
uint32_t get_l1_alignment();

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific PCIE alignment.
 *
 * @return Alignment requirement in bytes
 */
uint32_t get_pcie_alignment();

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific address.
 * this address corresponds to the beginning of free space in the ERISC's L1 SRAM
 *
 * @return address
 */
uint32_t get_erisc_l1_unreserved_base();

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific size.
 * this size corresponds to the total free space in the ERISC's L1 SRAM for host usage
 *
 * @return size in bytes
 */
uint32_t get_erisc_l1_unreserved_size();

}  // namespace tt::tt_metal::experimental::hal
