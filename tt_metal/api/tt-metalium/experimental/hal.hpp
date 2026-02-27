// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/context/metalium_env.hpp>

namespace tt::tt_metal::experimental::hal {

/**
 * @brief Uses the hardware abstraction layer to inform client of the architecture
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return Architecture enum defined by UMD
 */
tt::ARCH get_arch(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to inform client of the architecture name
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return Name
 */
std::string get_arch_name(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific L1 Size
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return Size in bytes of the L1 SRAM buffer associated with the currently present architecture.
 */
uint32_t get_l1_size(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific DRAM alignment.
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return Alignment requirement in bytes
 */
uint32_t get_dram_alignment(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific L1 alignment.
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return Alignment requirement in bytes
 */
uint32_t get_l1_alignment(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific PCIE alignment.
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return Alignment requirement in bytes
 */
uint32_t get_pcie_alignment(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific address.
 * this address corresponds to the beginning of free space in the ERISC's L1 SRAM
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return address
 */
uint32_t get_erisc_l1_unreserved_base(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific size.
 * this size corresponds to the total free space in the ERISC's L1 SRAM for host usage
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return size in bytes
 */
uint32_t get_erisc_l1_unreserved_size(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific size.
 * This size corresponds to the maximum size of the L1 SRAM buffer that can be
 * used by the application, if the ringbuffer size is set to 0.
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return size in bytes
 */
uint32_t get_max_worker_l1_unreserved_size(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to fetch the representable epsilon value.
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return SFPU epsilon value
 */
float get_eps(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to fetch the representable NaN value.
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return SFPU NaN value
 */
float get_nan(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to fetch the representable Infinity value.
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return SFPU Infinity value
 */
float get_inf(const MetaliumEnv& env);

/**
 * @brief Uses the hardware abstraction layer to get the maximum number of circular buffers per core.
 *
 * @param context_id The context ID to use. Defaults to the first context which is the silicon context.
 *
 * @return Maximum number of circular buffers
 */
uint32_t get_arch_num_circular_buffers(const MetaliumEnv& env);

}  // namespace tt::tt_metal::experimental::hal
