// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
// Version: FFN1.3.0

/**
 * @file cmdbuff_api.hpp
 * @brief Quasar command buffer hardware reference API.
 *
 * This header is the instruction-level description of the command buffer
 * hardware. Quasar has two equivalent complex command buffers and one simple
 * command buffer with a smaller, distinct instruction set.
 *
 * Complex command buffer IDs and register selectors are template arguments
 * because both are encoded directly in a RoCC instruction. The API therefore
 * cannot perform runtime command buffer or register selection.
 *
 * This layer intentionally does not assign NoC or iDMA ownership. Ownership is
 * kernel policy and belongs in the data movement layer above this hardware API.
 * It also does not reset or reconfigure a command buffer as a side effect of
 * issuing a transaction.
 */

#pragma once

#include <cstdint>

#include "meta/registers/overlay_reg.h"
#include "rocc_template_instructions.hpp"

namespace overlay {

/**
 * Instruction-level access to one of the two complex command buffers.
 *
 * @tparam CommandBufferId Physical complex command buffer ID. Valid values are
 *         0 and 1.
 */
template <std::uint32_t CommandBufferId>
class ComplexCommandBuffer {
    static_assert(CommandBufferId < 2, "Quasar has exactly two complex command buffers: 0 and 1");

    template <std::uint32_t InstructionIndex>
    static constexpr std::uint32_t function() {
        return (CommandBufferId * 64 + InstructionIndex) & 0x7f;
    }

public:
    static constexpr std::uint32_t id = CommandBufferId;

    /**
     * Write a generated command buffer register.
     *
     * RegisterSelector is the byte offset (or generated aligned register
     * selector) accepted by CMDBUF_WR_REG. It is encoded in the instruction and
     * must therefore be a compile-time constant.
     */
    template <std::uint32_t RegisterSelector>
    static inline __attribute__((always_inline)) void write_register(std::uint64_t value) {
        static_assert((RegisterSelector & 0x7) == 0, "Command buffer register selector must be 8-byte aligned");
        static_assert(
            RegisterSelector < TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REG_FILE_SIZE,
            "Command buffer register selector is outside the generated register file");
        rocc::instruction_s<rocc::kCustom0Opcode, function<RegisterSelector / 8>()>(value);
    }

    /** Read a generated command buffer register selected by byte offset. */
    template <std::uint32_t RegisterSelector>
    static inline __attribute__((always_inline)) std::uint64_t read_register() {
        static_assert((RegisterSelector & 0x7) == 0, "Command buffer register selector must be 8-byte aligned");
        static_assert(
            RegisterSelector < TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REG_FILE_SIZE,
            "Command buffer register selector is outside the generated register file");
        return rocc::instruction_d<rocc::kCustom0Opcode, function<RegisterSelector / 8>()>();
    }

    /** Reset every register in this command buffer to its hardware default. */
    static inline __attribute__((always_inline)) void reset() {
        rocc::instruction<rocc::kCustom0Opcode, function<59>()>();
    }

    /** Return free space for the currently configured NoC request VC. */
    static inline __attribute__((always_inline)) std::uint64_t noc_vc_space() {
        return rocc::instruction_d<rocc::kCustom0Opcode, function<62>()>();
    }

    /** Return free space for the specified NoC virtual channel. */
    static inline __attribute__((always_inline)) std::uint64_t noc_vc_space(std::uint32_t virtual_channel) {
        return rocc::instruction_ds<rocc::kCustom0Opcode, function<62>()>(virtual_channel);
    }

    /** Return pending NoC write sends for the configured transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_writes_pending() {
        return rocc::instruction_d<rocc::kCustom0Opcode, function<61>()>();
    }

    /** Return pending NoC write sends for the specified transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_writes_pending(std::uint32_t transaction_id) {
        return rocc::instruction_ds<rocc::kCustom0Opcode, function<61>()>(transaction_id);
    }

    /** Return pending NoC acknowledgements for the configured transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_acks_pending() {
        return rocc::instruction_d<rocc::kCustom0Opcode, function<60>()>();
    }

    /** Return pending NoC acknowledgements for the specified transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_acks_pending(std::uint32_t transaction_id) {
        return rocc::instruction_ds<rocc::kCustom0Opcode, function<60>()>(transaction_id);
    }

    /** Return free space for the currently configured iDMA request VC. */
    static inline __attribute__((always_inline)) std::uint64_t idma_vc_space() {
        return rocc::instruction_d<rocc::kCustom0Opcode, function<58>()>();
    }

    /** Return free space for the specified iDMA virtual channel. */
    static inline __attribute__((always_inline)) std::uint64_t idma_vc_space(std::uint32_t virtual_channel) {
        return rocc::instruction_ds<rocc::kCustom0Opcode, function<58>()>(virtual_channel);
    }

    /** Return pending iDMA acknowledgements for the configured transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t idma_acks_pending() {
        return rocc::instruction_d<rocc::kCustom0Opcode, function<57>()>();
    }

    /** Return pending iDMA acknowledgements for the specified transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t idma_acks_pending(std::uint32_t transaction_id) {
        return rocc::instruction_ds<rocc::kCustom0Opcode, function<57>()>(transaction_id);
    }

    /** Issue the transaction currently configured in the command buffer. */
    static inline __attribute__((always_inline)) void issue() {
        rocc::instruction<rocc::kCustom0Opcode, function<63>()>();
    }

    /** Issue an inline transaction using one packed operand. */
    static inline __attribute__((always_inline)) void issue_inline(std::uint64_t value) {
        rocc::instruction_s<rocc::kCustom0Opcode, function<63>()>(value);
    }

    /** Issue an inline transaction using packed data and address operands. */
    static inline __attribute__((always_inline)) void issue_inline(std::uint64_t value, std::uint64_t packed_address) {
        rocc::instruction_ss<rocc::kCustom0Opcode, function<63>()>(value, packed_address);
    }

    /** Issue a read using one packed operand and hardware defaults. */
    static inline __attribute__((always_inline)) void issue_read(std::uint64_t packed_parameters) {
        rocc::instruction_s<rocc::kCustom0Opcode, function<56>()>(packed_parameters);
    }

    /** Issue a read using two packed operands and hardware defaults. */
    static inline __attribute__((always_inline)) void issue_read(
        std::uint64_t packed_parameters_1, std::uint64_t packed_parameters_2) {
        rocc::instruction_ss<rocc::kCustom0Opcode, function<56>()>(packed_parameters_1, packed_parameters_2);
    }

    /** Issue a write using one packed operand and hardware defaults. */
    static inline __attribute__((always_inline)) void issue_write(std::uint64_t packed_parameters) {
        rocc::instruction_s<rocc::kCustom0Opcode, function<55>()>(packed_parameters);
    }

    /** Issue a write using two packed operands and hardware defaults. */
    static inline __attribute__((always_inline)) void issue_write(
        std::uint64_t packed_parameters_1, std::uint64_t packed_parameters_2) {
        rocc::instruction_ss<rocc::kCustom0Opcode, function<55>()>(packed_parameters_1, packed_parameters_2);
    }

    /** Return the NoC acknowledgement tiles-to-process count for a transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_ack_tiles_to_process(std::uint32_t transaction_id) {
        return rocc::instruction_ds<rocc::kCustom0Opcode, function<54>()>(transaction_id);
    }

    /** Return the NoC write tiles-to-process count for a transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_write_tiles_to_process(
        std::uint32_t transaction_id) {
        return rocc::instruction_ds<rocc::kCustom0Opcode, function<53>()>(transaction_id);
    }

    /** Return the iDMA acknowledgement tiles-to-process count for a transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t idma_ack_tiles_to_process(std::uint32_t transaction_id) {
        return rocc::instruction_ds<rocc::kCustom0Opcode, function<52>()>(transaction_id);
    }

    /** Clear the NoC acknowledgement tiles-to-process count. A following nop is required. */
    static inline __attribute__((always_inline)) void clear_noc_ack_tiles_to_process(std::uint32_t transaction_id) {
        rocc::instruction_s<rocc::kCustom0Opcode, function<51>()>(transaction_id);
    }

    /** Clear the NoC write tiles-to-process count. A following nop is required. */
    static inline __attribute__((always_inline)) void clear_noc_write_tiles_to_process(std::uint32_t transaction_id) {
        rocc::instruction_s<rocc::kCustom0Opcode, function<50>()>(transaction_id);
    }

    /** Clear the iDMA acknowledgement tiles-to-process count. A following nop is required. */
    static inline __attribute__((always_inline)) void clear_idma_ack_tiles_to_process(std::uint32_t transaction_id) {
        rocc::instruction_s<rocc::kCustom0Opcode, function<49>()>(transaction_id);
    }
};

using ComplexCommandBuffer0 = ComplexCommandBuffer<0>;
using ComplexCommandBuffer1 = ComplexCommandBuffer<1>;

/**
 * Instruction-level access to the simple command buffer.
 *
 * The simple command buffer is not complex command buffer 2. It uses CUSTOM_1
 * for its normal instruction interface and does not expose iDMA operations.
 */
class SimpleCommandBuffer {
    template <std::uint32_t InstructionIndex>
    static constexpr std::uint32_t function() {
        return (64 + InstructionIndex) & 0x7f;
    }

public:
    /** Write a generated command buffer register selected by byte offset. */
    template <std::uint32_t RegisterSelector>
    static inline __attribute__((always_inline)) void write_register(std::uint64_t value) {
        static_assert((RegisterSelector & 0x7) == 0, "Command buffer register selector must be 8-byte aligned");
        static_assert(
            RegisterSelector < TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REG_FILE_SIZE,
            "Command buffer register selector is outside the generated register file");
        rocc::instruction_s<rocc::kCustom1Opcode, function<RegisterSelector / 8>()>(value);
    }

    /** Read a generated command buffer register selected by byte offset. */
    template <std::uint32_t RegisterSelector>
    static inline __attribute__((always_inline)) std::uint64_t read_register() {
        static_assert((RegisterSelector & 0x7) == 0, "Command buffer register selector must be 8-byte aligned");
        static_assert(
            RegisterSelector < TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REG_FILE_SIZE,
            "Command buffer register selector is outside the generated register file");
        return rocc::instruction_d<rocc::kCustom1Opcode, function<RegisterSelector / 8>()>();
    }

    /** Reset every register in the simple command buffer to its hardware default. */
    static inline __attribute__((always_inline)) void reset() {
        rocc::instruction<rocc::kCustom1Opcode, function<59>()>();
    }

    /** Return free space for the currently configured NoC request VC. */
    static inline __attribute__((always_inline)) std::uint64_t noc_vc_space() {
        return rocc::instruction_d<rocc::kCustom1Opcode, function<62>()>();
    }

    /** Return free space for the specified NoC virtual channel. */
    static inline __attribute__((always_inline)) std::uint64_t noc_vc_space(std::uint32_t virtual_channel) {
        return rocc::instruction_ds<rocc::kCustom1Opcode, function<62>()>(virtual_channel);
    }

    /** Return pending NoC write sends for the configured transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_writes_pending() {
        return rocc::instruction_d<rocc::kCustom1Opcode, function<61>()>();
    }

    /** Return pending NoC write sends for the specified transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_writes_pending(std::uint32_t transaction_id) {
        return rocc::instruction_ds<rocc::kCustom1Opcode, function<61>()>(transaction_id);
    }

    /** Return pending NoC acknowledgements for the configured transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_acks_pending() {
        return rocc::instruction_d<rocc::kCustom1Opcode, function<60>()>();
    }

    /** Return pending NoC acknowledgements for the specified transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_acks_pending(std::uint32_t transaction_id) {
        return rocc::instruction_ds<rocc::kCustom1Opcode, function<60>()>(transaction_id);
    }

    /** Issue the transaction currently configured in the command buffer. */
    static inline __attribute__((always_inline)) void issue() {
        rocc::instruction<rocc::kCustom1Opcode, function<63>()>();
    }

    /** Issue an inline transaction using one packed operand. */
    static inline __attribute__((always_inline)) void issue_inline(std::uint64_t value) {
        rocc::instruction_s<rocc::kCustom1Opcode, function<63>()>(value);
    }

    /** Issue an inline transaction using packed data and address operands. */
    static inline __attribute__((always_inline)) void issue_inline(std::uint64_t value, std::uint64_t packed_address) {
        rocc::instruction_ss<rocc::kCustom1Opcode, function<63>()>(value, packed_address);
    }

    /** Issue a read using one packed operand and hardware defaults. */
    static inline __attribute__((always_inline)) void issue_read(std::uint64_t packed_parameters) {
        rocc::instruction_s<rocc::kCustom1Opcode, function<56>()>(packed_parameters);
    }

    /** Issue a read using two packed operands and hardware defaults. */
    static inline __attribute__((always_inline)) void issue_read(
        std::uint64_t packed_parameters_1, std::uint64_t packed_parameters_2) {
        rocc::instruction_ss<rocc::kCustom1Opcode, function<56>()>(packed_parameters_1, packed_parameters_2);
    }

    /** Issue a write using one packed operand and hardware defaults. */
    static inline __attribute__((always_inline)) void issue_write(std::uint64_t packed_parameters) {
        rocc::instruction_s<rocc::kCustom1Opcode, function<55>()>(packed_parameters);
    }

    /** Issue a write using two packed operands and hardware defaults. */
    static inline __attribute__((always_inline)) void issue_write(
        std::uint64_t packed_parameters_1, std::uint64_t packed_parameters_2) {
        rocc::instruction_ss<rocc::kCustom1Opcode, function<55>()>(packed_parameters_1, packed_parameters_2);
    }

    /**
     * Return the NoC acknowledgement tiles-to-process count.
     *
     * The instruction map places simple-buffer tile-counter operations on
     * CUSTOM_0 even though its normal interface uses CUSTOM_1.
     */
    static inline __attribute__((always_inline)) std::uint64_t noc_ack_tiles_to_process(std::uint32_t transaction_id) {
        return rocc::instruction_ds<rocc::kCustom0Opcode, function<54>()>(transaction_id);
    }

    /** Return the NoC write tiles-to-process count for a transaction ID. */
    static inline __attribute__((always_inline)) std::uint64_t noc_write_tiles_to_process(
        std::uint32_t transaction_id) {
        return rocc::instruction_ds<rocc::kCustom0Opcode, function<53>()>(transaction_id);
    }

    /** Clear the NoC acknowledgement tiles-to-process count. A following nop is required. */
    static inline __attribute__((always_inline)) void clear_noc_ack_tiles_to_process(std::uint32_t transaction_id) {
        rocc::instruction_s<rocc::kCustom0Opcode, function<51>()>(transaction_id);
    }

    /** Clear the NoC write tiles-to-process count. A following nop is required. */
    static inline __attribute__((always_inline)) void clear_noc_write_tiles_to_process(std::uint32_t transaction_id) {
        rocc::instruction_s<rocc::kCustom0Opcode, function<50>()>(transaction_id);
    }
};

}  // namespace overlay

// Existing bring-up kernels still use the generated free-function API. Keep it
// source compatible while new architecture backends move to the typed hardware
// reference above. Reference-layer tests can define this macro to exclude the
// compatibility surface and its higher-level NoC dependencies.
#ifndef TT_METAL_QUASAR_DISABLE_LEGACY_COMMAND_BUFFER_API
#include "cmdbuff_legacy_api.hpp"
#endif
