// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <umd/device/types/arch.h>

namespace tt::tt_metal::hal {

/**
 * @brief Uses the hardware abstraction layer to inform client of the architecture
 *
 * @return Architecture enum defined by UMD
 */
tt::ARCH get_arch();

/**
 * @brief Uses the hardware abstraction layer to inform client of the architecture name
 *
 * @return Name
 */
std::string get_arch_name();

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

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific size.
 * This size corresponds to the maximum size of the L1 SRAM buffer that can be
 * used by the application, if the ringbuffer size is set to 0.
 *
 * @return size in bytes
 */
uint32_t get_max_worker_l1_unreserved_size();

/**
 * @brief Uses the hardware abstraction layer to fetch the representable epsilon value.
 *
 * @return SFPU epsilon value
 */
float get_eps();

/**
 * @brief Uses the hardware abstraction layer to fetch the representable NaN value.
 *
 * @return SFPU NaN value
 */
float get_nan();

/**
 * @brief Uses the hardware abstraction layer to fetch the representable Infinity value.
 *
 * @return SFPU Infinity value
 */
float get_inf();

}  // namespace tt::tt_metal::hal
