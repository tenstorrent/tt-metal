// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
/*************************************************************************
 * --------------------------------------------------------------------------------------
 * @brief Generic API for Counter Remapper Configuration
 * This class provides a high-level interface for configuring and monitoring
 * the counter remapper hardware registers. It supports arbitrary mapping
 * configurations between ClientL and ClientR.
 * --------------------------------------------------------------------------------------
 * Key Features:
 * -- Scenario-agnostic configuration interface
 * -- Granular control methods for individual fields
 * -- Slot-based ClientR configuration (set/get by index 0-3)
 * -- Individual ClientL field setters/getters
 * -- Combined configuration methods (write_all_configs(), clear_all_configs())
 * -- Status and error checking utilities
 * -- Direct raw register access for advanced use cases
 *
 * -- Usage example:
 *
 *          RemapperAPI api;
 *          api.set_pair_index(5);  // Switch to pair 5
 *          // Configure any mapping scenario
 *          api.set_clientL_id_counter(2, 13);
 *          api.set_clientL_valid(1);
 *          api.set_clientR_slot(0, 7, 22);
 *          api.write_all_configs();
 *
 *          // Check for errors
 *          if (api.check_clientL_errors()) {  }
 *************************************************************************/
#ifndef __DM__REMAPPER_API_HPP__
#define __DM__REMAPPER_API_HPP__

#include <cstdint>
#include "remapper_common.hpp"

/**
 * @brief Generic API for Counter Remapper Configuration
 *
 * This class provides a flexible, scenario-agnostic interface for configuring
 * and monitoring counter remapper hardware registers. It supports arbitrary
 * mapping configurations between ClientL and ClientR.
 */
class RemapperAPI {
private:
    tClientR_Config_Reg_u clientR_configs[REMAP_NUM_PAIRS];  // Uninitialized to avoid memset
    tClientL_Config_Reg_u clientL_configs[REMAP_NUM_PAIRS];  // Uninitialized to avoid memset
    uint32_t current_pair_idx;

public:
    /**
     * @brief Constructor - initializes pair index only
     * @param pair_idx Initial pair index to use (0-63, default 0)
     *
     * WARNING: Config arrays are left uninitialized to avoid linker issues with memset.
     * User MUST configure registers using set_* methods before calling write_* methods.
     */
    RemapperAPI(uint32_t pair_idx = 0) : current_pair_idx(pair_idx) {
        // Arrays intentionally left uninitialized to avoid memset() calls
        // which cause linker errors with soft-float ABI
    }

    // ========================================================================
    // Pair Management Methods
    // ========================================================================

    /**
     * @brief Set the current pair index for subsequent operations
     * @param pair_idx Pair index (0-63)
     * @param auto_clear If true, clears the pair's config before switching (default: false)
     */
    void set_pair_index(uint32_t pair_idx, bool auto_clear = false) {
        if (pair_idx < REMAP_NUM_PAIRS) {
            if (auto_clear) {
                clientR_configs[pair_idx].val = 0;
                clientL_configs[pair_idx].val = 0;
            }
            current_pair_idx = pair_idx;
        }
    }

    /**
     * @brief Get the current pair index
     * @return Current pair index
     */
    uint32_t get_pair_index() const { return current_pair_idx; }

    // ========================================================================
    // ClientR Configuration Methods
    // ========================================================================

    /**
     * @brief Set ClientR slot configuration by index (uses current pair)
     *
     * @param slot Slot index (0-3)
     * @param id Client ID (3 bits, 0-7)
     * @param cnt_sel Counter select (5 bits, 0-31)
     */
    void set_clientR_slot(uint32_t slot, uint32_t id, uint32_t cnt_sel) {
        set_clientR_slot(slot, id, cnt_sel, current_pair_idx);
    }

    /**
     * @brief Set ClientR slot configuration by index for a specific pair
     *
     * @param slot Slot index (0-3)
     * @param id Client ID (3 bits, 0-7)
     * @param cnt_sel Counter select (5 bits, 0-31)
     * @param pair_idx Pair index (0-63)
     */
    void set_clientR_slot(uint32_t slot, uint32_t id, uint32_t cnt_sel, uint32_t pair_idx) {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }
        switch (slot) {
            case 0:
                clientR_configs[pair_idx].f.id_0 = id & 0x7;
                clientR_configs[pair_idx].f.cnt_sel_0 = cnt_sel & 0x1F;
                break;
            case 1:
                clientR_configs[pair_idx].f.id_1 = id & 0x7;
                clientR_configs[pair_idx].f.cnt_sel_1 = cnt_sel & 0x1F;
                break;
            case 2:
                clientR_configs[pair_idx].f.id_2 = id & 0x7;
                clientR_configs[pair_idx].f.cnt_sel_2 = cnt_sel & 0x1F;
                break;
            case 3:
                clientR_configs[pair_idx].f.id_3 = id & 0x7;
                clientR_configs[pair_idx].f.cnt_sel_3 = cnt_sel & 0x1F;
                break;
        }
    }

    /**
     * @brief Get ClientR slot configuration by index (uses current pair)
     *
     * @param slot Slot index (0-3)
     * @param id Output: Client ID
     * @param cnt_sel Output: Counter select
     */
    void get_clientR_slot(uint32_t slot, uint32_t& id, uint32_t& cnt_sel) const {
        get_clientR_slot(slot, id, cnt_sel, current_pair_idx);
    }

    /**
     * @brief Find configuration pair index for a given ClientR (id, counter_select)
     *
     * Searches through all 64 configuration pairs and their 4 slots to find
     * a matching ClientR configuration. This reads directly from hardware registers.
     *
     * @param target_id ClientR ID to search for (3 bits, 0-7)
     * @param target_cnt_sel ClientR counter select to search for (5 bits, 0-31)
     * @param found_pair_idx Output: Pair index where match was found (0-63)
     * @param found_slot Output: Slot index where match was found (0-3)
     * @return true if a matching configuration was found, false otherwise
     */
    bool findClientRConfigIndex(
        uint32_t target_id, uint32_t target_cnt_sel, uint32_t& found_pair_idx, uint32_t& found_slot) {
        target_id &= 0x7;
        target_cnt_sel &= 0x1F;

        // Search through all 64 pairs by reading from hardware
        for (uint32_t pair_idx = 0; pair_idx < REMAP_NUM_PAIRS; pair_idx++) {
            // Read ClientR configuration register from hardware
            uint32_t reg_val = READ_REG32(REMAP_CLIENT_R_CONFIG_REG_ADDR32(pair_idx));
            tClientR_Config_Reg_u config;
            config.val = reg_val;

            // Check all 4 slots in this pair
            if (config.f.id_0 == target_id && config.f.cnt_sel_0 == target_cnt_sel) {
                found_pair_idx = pair_idx;
                found_slot = 0;
                return true;
            }
            if (config.f.id_1 == target_id && config.f.cnt_sel_1 == target_cnt_sel) {
                found_pair_idx = pair_idx;
                found_slot = 1;
                return true;
            }
            if (config.f.id_2 == target_id && config.f.cnt_sel_2 == target_cnt_sel) {
                found_pair_idx = pair_idx;
                found_slot = 2;
                return true;
            }
            if (config.f.id_3 == target_id && config.f.cnt_sel_3 == target_cnt_sel) {
                found_pair_idx = pair_idx;
                found_slot = 3;
                return true;
            }
        }
        // No match found
        return false;
    }

    /**
     * @brief Get ClientR slot configuration by index for a specific pair
     *
     * @param slot Slot index (0-3)
     * @param id Output: Client ID
     * @param cnt_sel Output: Counter select
     * @param pair_idx Pair index (0-63)
     */
    void get_clientR_slot(uint32_t slot, uint32_t& id, uint32_t& cnt_sel, uint32_t pair_idx) const {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }
        switch (slot) {
            case 0:
                id = clientR_configs[pair_idx].f.id_0;
                cnt_sel = clientR_configs[pair_idx].f.cnt_sel_0;
                break;
            case 1:
                id = clientR_configs[pair_idx].f.id_1;
                cnt_sel = clientR_configs[pair_idx].f.cnt_sel_1;
                break;
            case 2:
                id = clientR_configs[pair_idx].f.id_2;
                cnt_sel = clientR_configs[pair_idx].f.cnt_sel_2;
                break;
            case 3:
                id = clientR_configs[pair_idx].f.id_3;
                cnt_sel = clientR_configs[pair_idx].f.cnt_sel_3;
                break;
        }
    }

    /**
     * @brief Configure all ClientR slots at once (uses current pair)
     *
     * @param id_0 ID for slot 0
     * @param cnt_sel_0 Counter select for slot 0
     * @param id_1 ID for slot 1
     * @param cnt_sel_1 Counter select for slot 1
     * @param id_2 ID for slot 2
     * @param cnt_sel_2 Counter select for slot 2
     * @param id_3 ID for slot 3
     * @param cnt_sel_3 Counter select for slot 3
     */
    void configure_clientR_all_slots(
        uint32_t id_0,
        uint32_t cnt_sel_0,
        uint32_t id_1,
        uint32_t cnt_sel_1,
        uint32_t id_2,
        uint32_t cnt_sel_2,
        uint32_t id_3,
        uint32_t cnt_sel_3) {
        configure_clientR_all_slots(
            id_0, cnt_sel_0, id_1, cnt_sel_1, id_2, cnt_sel_2, id_3, cnt_sel_3, current_pair_idx);
    }

    /**
     * @brief Configure all ClientR slots at once for a specific pair
     *
     * @param id_0 ID for slot 0
     * @param cnt_sel_0 Counter select for slot 0
     * @param id_1 ID for slot 1
     * @param cnt_sel_1 Counter select for slot 1
     * @param id_2 ID for slot 2
     * @param cnt_sel_2 Counter select for slot 2
     * @param id_3 ID for slot 3
     * @param cnt_sel_3 Counter select for slot 3
     * @param pair_idx Pair index (0-63)
     */
    void configure_clientR_all_slots(
        uint32_t id_0,
        uint32_t cnt_sel_0,
        uint32_t id_1,
        uint32_t cnt_sel_1,
        uint32_t id_2,
        uint32_t cnt_sel_2,
        uint32_t id_3,
        uint32_t cnt_sel_3,
        uint32_t pair_idx) {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }
        clientR_configs[pair_idx].f.id_0 = id_0 & 0x7;
        clientR_configs[pair_idx].f.cnt_sel_0 = cnt_sel_0 & 0x1F;
        clientR_configs[pair_idx].f.id_1 = id_1 & 0x7;
        clientR_configs[pair_idx].f.cnt_sel_1 = cnt_sel_1 & 0x1F;
        clientR_configs[pair_idx].f.id_2 = id_2 & 0x7;
        clientR_configs[pair_idx].f.cnt_sel_2 = cnt_sel_2 & 0x1F;
        clientR_configs[pair_idx].f.id_3 = id_3 & 0x7;
        clientR_configs[pair_idx].f.cnt_sel_3 = cnt_sel_3 & 0x1F;
    }

    /**
     * @brief Clear ClientR configuration (uses current pair)
     */
    void clear_clientR() { clear_clientR(current_pair_idx); }

    /**
     * @brief Clear ClientR configuration for a specific pair
     * @param pair_idx Pair index (0-63)
     */
    void clear_clientR(uint32_t pair_idx) {
        if (pair_idx < REMAP_NUM_PAIRS) {
            clientR_configs[pair_idx].val = 0;
        }
    }

    /**
     * @brief Write ClientR configuration to hardware register (uses current pair)
     */
    void write_clientR_config() { write_clientR_config(current_pair_idx); }

    /**
     * @brief Write ClientR configuration to hardware register for a specific pair
     * @param pair_idx Pair index (0-63)
     */
    void write_clientR_config(uint32_t pair_idx) {
        if (pair_idx < REMAP_NUM_PAIRS) {
            WRITE_REG32(REMAP_CLIENT_R_CONFIG_REG_ADDR32(pair_idx), clientR_configs[pair_idx].val);
        }
    }

    /**
     * @brief Read ClientR status register from hardware (uses current pair)
     *
     * @return Status register union
     */
    tClientR_status_Reg_u read_clientR_status() { return read_clientR_status(current_pair_idx); }

    /**
     * @brief Read ClientR status register from hardware for a specific pair
     *
     * @param pair_idx Pair index (0-63)
     * @return Status register union
     */
    tClientR_status_Reg_u read_clientR_status(uint32_t pair_idx) {
        tClientR_status_Reg_u status;
        status.val = 0;
        if (pair_idx < REMAP_NUM_PAIRS) {
            status.val = READ_REG32(REMAP_CLIENT_R_STATUS_REG_ADDR32(pair_idx));
        }
        return status;
    }

    /**
     * @brief Get current ClientR configuration value (uses current pair)
     *
     * @return 32-bit register value
     */
    uint32_t get_clientR_config_value() const { return get_clientR_config_value(current_pair_idx); }

    /**
     * @brief Get ClientR configuration value for a specific pair
     *
     * @param pair_idx Pair index (0-63)
     * @return 32-bit register value
     */
    uint32_t get_clientR_config_value(uint32_t pair_idx) const {
        if (pair_idx < REMAP_NUM_PAIRS) {
            return clientR_configs[pair_idx].val;
        }
        return 0;
    }

    // ========================================================================
    // ClientL Configuration Methods
    // ========================================================================

    /**
     * @brief Set ClientL ID and counter select (uses current pair)
     *
     * @param id Client ID (3 bits, 0-7)
     * @param cnt_sel Counter select (5 bits, 0-31)
     */
    void set_clientL_id_counter(uint32_t id, uint32_t cnt_sel) {
        set_clientL_id_counter(id, cnt_sel, current_pair_idx);
    }

    /**
     * @brief Set ClientL ID and counter select for a specific pair
     *
     * @param id Client ID (3 bits, 0-7)
     * @param cnt_sel Counter select (5 bits, 0-31)
     * @param pair_idx Pair index (0-63)
     */
    void set_clientL_id_counter(uint32_t id, uint32_t cnt_sel, uint32_t pair_idx) {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }
        clientL_configs[pair_idx].f.id_L = id & 0x7;
        clientL_configs[pair_idx].f.cnt_sel_L = cnt_sel & 0x1F;
    }

    /**
     * @brief Get ClientL ID and counter select (uses current pair)
     *
     * @param id Output: Client ID
     * @param cnt_sel Output: Counter select
     */
    void get_clientL_id_counter(uint32_t& id, uint32_t& cnt_sel) const {
        get_clientL_id_counter(id, cnt_sel, current_pair_idx);
    }

    /**
     * @brief Get ClientL ID and counter select for a specific pair
     *
     * @param id Output: Client ID
     * @param cnt_sel Output: Counter select
     * @param pair_idx Pair index (0-63)
     */
    void get_clientL_id_counter(uint32_t& id, uint32_t& cnt_sel, uint32_t pair_idx) const {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }
        id = clientL_configs[pair_idx].f.id_L;
        cnt_sel = clientL_configs[pair_idx].f.cnt_sel_L;
    }

    /**
     * @brief Set ClientL valid bits (uses current pair)
     *
     * @param valid Valid bits (4 bits, 0-15)
     */
    void set_clientL_valid(uint32_t valid) { set_clientL_valid(valid, current_pair_idx); }

    /**
     * @brief Set ClientL valid bits for a specific pair
     *
     * @param valid Valid bits (4 bits, 0-15)
     * @param pair_idx Pair index (0-63)
     */
    void set_clientL_valid(uint32_t valid, uint32_t pair_idx) {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }
        clientL_configs[pair_idx].f.valid = valid & 0xF;
    }

    /**
     * @brief Get ClientL valid bits (uses current pair)
     *
     * @return Valid bits value
     */
    uint32_t get_clientL_valid() const { return get_clientL_valid(current_pair_idx); }

    /**
     * @brief Get ClientL valid bits for a specific pair
     *
     * @param pair_idx Pair index (0-63)
     * @return Valid bits value
     */
    uint32_t get_clientL_valid(uint32_t pair_idx) const {
        if (pair_idx < REMAP_NUM_PAIRS) {
            return clientL_configs[pair_idx].f.valid;
        }
        return 0;
    }

    /**
     * @brief Set ClientL producer/consumer mode (uses current pair)
     *
     * @param is_producer 1 if ClientL is producer, 0 if consumer
     */
    void set_clientL_is_producer(uint32_t is_producer) { set_clientL_is_producer(is_producer, current_pair_idx); }

    /**
     * @brief Set ClientL producer/consumer mode for a specific pair
     *
     * @param is_producer 1 if ClientL is producer, 0 if consumer
     * @param pair_idx Pair index (0-63)
     */
    void set_clientL_is_producer(uint32_t is_producer, uint32_t pair_idx) {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }
        clientL_configs[pair_idx].f.clientl_is_producer = is_producer & 0x1;
    }

    /**
     * @brief Set ClientR group flag (uses current pair)
     *
     * @param group Group flag (1 bit)
     */
    void set_clientR_group(uint32_t group) { set_clientR_group(group, current_pair_idx); }

    /**
     * @brief Set ClientR group flag for a specific pair
     *
     * @param group Group flag (1 bit)
     * @param pair_idx Pair index (0-63)
     */
    void set_clientR_group(uint32_t group, uint32_t pair_idx) {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }
        clientL_configs[pair_idx].f.clientr_group = group & 0x1;
    }

    /**
     * @brief Set distribute mode (uses current pair)
     *
     * @param distribute 1 to enable distribute mode, 0 to disable
     */
    void set_distribute(uint32_t distribute) { set_distribute(distribute, current_pair_idx); }

    /**
     * @brief Set distribute mode for a specific pair
     *
     * @param distribute 1 to enable distribute mode, 0 to disable
     * @param pair_idx Pair index (0-63)
     */
    void set_distribute(uint32_t distribute, uint32_t pair_idx) {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }
        clientL_configs[pair_idx].f.distribute = distribute & 0x1;
    }

    /**
     * @brief Set sync barrier by clearing the clientr_group bit (uses current pair)
     *
     * This method performs a Read-Modify-Write operation to set the clientr_group
     * bit to 0 while preserving all other configuration bits.
     */
    void setSyncBarrier() { setSyncBarrier(current_pair_idx); }

    /**
     * @brief Set sync barrier by clearing the clientr_group bit for a specific pair
     *
     * This method performs a Read-Modify-Write operation to set the clientr_group
     * bit to 0 while preserving all other configuration bits.
     *
     * @param pair_idx Pair index (0-63)
     */
    void setSyncBarrier(uint32_t pair_idx) {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }

        // Read current hardware register value
        uint32_t current_val = READ_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pair_idx));

        // Clear bit 13 (clientr_group) using bit mask
        uint32_t modified_val = current_val & ~(1 << 13);

        // Write back modified value
        WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pair_idx), modified_val);
    }

    /**
     * @brief Configure all ClientL fields at once (uses current pair)
     *
     * @param id_L ClientL ID (3 bits)
     * @param cnt_sel_L ClientL counter select (5 bits)
     * @param valid Valid bits (4 bits)
     * @param clientl_is_producer Producer flag (1 bit)
     * @param clientr_group ClientR group flag (1 bit)
     * @param distribute Distribute flag (1 bit)
     */
    void configure_clientL_all_fields(
        uint32_t id_L,
        uint32_t cnt_sel_L,
        uint32_t valid,
        uint32_t clientl_is_producer,
        uint32_t clientr_group,
        uint32_t distribute) {
        configure_clientL_all_fields(
            id_L, cnt_sel_L, valid, clientl_is_producer, clientr_group, distribute, current_pair_idx);
    }

    /**
     * @brief Configure all ClientL fields at once for a specific pair
     *
     * @param id_L ClientL ID (3 bits)
     * @param cnt_sel_L ClientL counter select (5 bits)
     * @param valid Valid bits (4 bits)
     * @param clientl_is_producer Producer flag (1 bit)
     * @param clientr_group ClientR group flag (1 bit)
     * @param distribute Distribute flag (1 bit)
     * @param pair_idx Pair index (0-63)
     */
    void configure_clientL_all_fields(
        uint32_t id_L,
        uint32_t cnt_sel_L,
        uint32_t valid,
        uint32_t clientl_is_producer,
        uint32_t clientr_group,
        uint32_t distribute,
        uint32_t pair_idx) {
        if (pair_idx >= REMAP_NUM_PAIRS) {
            return;
        }
        clientL_configs[pair_idx].f.id_L = id_L & 0x7;
        clientL_configs[pair_idx].f.cnt_sel_L = cnt_sel_L & 0x1F;
        clientL_configs[pair_idx].f.valid = valid & 0xF;
        clientL_configs[pair_idx].f.clientl_is_producer = clientl_is_producer & 0x1;
        clientL_configs[pair_idx].f.clientr_group = clientr_group & 0x1;
        clientL_configs[pair_idx].f.distribute = distribute & 0x1;
    }

    /**
     * @brief Clear ClientL configuration (uses current pair)
     */
    void clear_clientL() { clear_clientL(current_pair_idx); }

    /**
     * @brief Clear ClientL configuration for a specific pair
     * @param pair_idx Pair index (0-63)
     */
    void clear_clientL(uint32_t pair_idx) {
        if (pair_idx < REMAP_NUM_PAIRS) {
            clientL_configs[pair_idx].val = 0;
        }
    }

    /**
     * @brief Write ClientL configuration to hardware register (uses current pair)
     */
    void write_clientL_config() { write_clientL_config(current_pair_idx); }

    /**
     * @brief Write ClientL configuration to hardware register for a specific pair
     * @param pair_idx Pair index (0-63)
     */
    void write_clientL_config(uint32_t pair_idx) {
        if (pair_idx < REMAP_NUM_PAIRS) {
            WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pair_idx), clientL_configs[pair_idx].val);
        }
    }

    /**
     * @brief Read ClientL status register from hardware (uses current pair)
     *
     * @return Status register union
     */
    tClientL_status_Reg_u read_clientL_status() { return read_clientL_status(current_pair_idx); }

    /**
     * @brief Read ClientL status register from hardware for a specific pair
     *
     * @param pair_idx Pair index (0-63)
     * @return Status register union
     */
    tClientL_status_Reg_u read_clientL_status(uint32_t pair_idx) {
        tClientL_status_Reg_u status;
        status.val = 0;
        if (pair_idx < REMAP_NUM_PAIRS) {
            status.val = READ_REG32(REMAP_CLIENT_L_STATUS_REG_ADDR32(pair_idx));
        }
        return status;
    }

    /**
     * @brief Get current ClientL configuration value (uses current pair)
     *
     * @return 32-bit register value
     */
    uint32_t get_clientL_config_value() const { return get_clientL_config_value(current_pair_idx); }

    /**
     * @brief Get ClientL configuration value for a specific pair
     *
     * @param pair_idx Pair index (0-63)
     * @return 32-bit register value
     */
    uint32_t get_clientL_config_value(uint32_t pair_idx) const {
        if (pair_idx < REMAP_NUM_PAIRS) {
            return clientL_configs[pair_idx].val;
        }
        return 0;
    }

    // ========================================================================
    // Global Enable/Disable Methods
    // ========================================================================

    /**
     * @brief Enable the remapper globally
     */
    void enable_remapper() { WRITE_REG32(REMAP_GLOBAL_CONTROL_REG_ADDR32, 0x1); }

    /**
     * @brief Disable the remapper globally
     */
    void disable_remapper() { WRITE_REG32(REMAP_GLOBAL_CONTROL_REG_ADDR32, 0x0); }

    /**
     * @brief Read the remapper enable status
     * @return true if remapper is enabled, false otherwise
     */
    bool is_remapper_enabled() { return (READ_REG32(REMAP_GLOBAL_CONTROL_REG_ADDR32) & 0x1) != 0; }

    // ========================================================================
    // Combined Configuration Methods
    // ========================================================================

    /**
     * @brief Write both ClientL and ClientR configurations to hardware (uses current pair)
     */
    void write_all_configs() { write_all_configs(current_pair_idx); }

    /**
     * @brief Write both ClientL and ClientR configurations to hardware for a specific pair
     * @param pair_idx Pair index (0-63)
     */
    void write_all_configs(uint32_t pair_idx) {
        write_clientR_config(pair_idx);
        write_clientL_config(pair_idx);
    }

    /**
     * @brief Write configurations for all 64 pairs to hardware
     */
    void write_all_pairs() {
        for (uint32_t i = 0; i < REMAP_NUM_PAIRS; i++) {
            write_all_configs(i);
        }
    }

    /**
     * @brief Clear all configuration registers for current pair (local copies)
     */
    void clear_all_configs() { clear_all_configs(current_pair_idx); }

    /**
     * @brief Clear all configuration registers for a specific pair (local copies)
     * @param pair_idx Pair index (0-63)
     */
    void clear_all_configs(uint32_t pair_idx) {
        clear_clientR(pair_idx);
        clear_clientL(pair_idx);
    }

    /**
     * @brief Clear all configuration registers for all 64 pairs (local copies)
     */
    void clear_all_pairs() {
        for (uint32_t i = 0; i < REMAP_NUM_PAIRS; i++) {
            clear_all_configs(i);
        }
    }

    // ========================================================================
    // Status and Error Checking Methods
    // ========================================================================

    /**
     * @brief Check if ClientL status has any error flags set
     *
     * @param status ClientL status register
     * @return true if any error flags are set
     */
    bool has_error(const tClientL_status_Reg_u& status) const {
        return (status.f.err_inv_client_no != 0 || status.f.err_inv_client_id != 0 || status.f.err_inv_access != 0);
    }

    /**
     * @brief Read ClientL status and check for errors (uses current pair)
     *
     * @return true if any error flags are set
     */
    bool check_clientL_errors() { return check_clientL_errors(current_pair_idx); }

    /**
     * @brief Read ClientL status and check for errors for a specific pair
     *
     * @param pair_idx Pair index (0-63)
     * @return true if any error flags are set
     */
    bool check_clientL_errors(uint32_t pair_idx) {
        tClientL_status_Reg_u status = read_clientL_status(pair_idx);
        return has_error(status);
    }

    /**
     * @brief Get detailed error information from ClientL status
     *
     * @param status ClientL status register
     * @param err_inv_client_no Output: Invalid client number error
     * @param err_inv_client_id Output: Invalid client ID error
     * @param err_inv_access Output: Invalid access error
     */
    void get_error_details(
        const tClientL_status_Reg_u& status,
        bool& err_inv_client_no,
        bool& err_inv_client_id,
        bool& err_inv_access) const {
        err_inv_client_no = (status.f.err_inv_client_no != 0);
        err_inv_client_id = (status.f.err_inv_client_id != 0);
        err_inv_access = (status.f.err_inv_access != 0);
    }

    // ========================================================================
    // Direct Register Access Methods
    // ========================================================================

    /**
     * @brief Set ClientR configuration from raw 32-bit value (uses current pair)
     *
     * @param val 32-bit register value
     */
    void set_clientR_raw(uint32_t val) { set_clientR_raw(val, current_pair_idx); }

    /**
     * @brief Set ClientR configuration from raw 32-bit value for a specific pair
     *
     * @param val 32-bit register value
     * @param pair_idx Pair index (0-63)
     */
    void set_clientR_raw(uint32_t val, uint32_t pair_idx) {
        if (pair_idx < REMAP_NUM_PAIRS) {
            clientR_configs[pair_idx].val = val;
        }
    }

    /**
     * @brief Set ClientL configuration from raw 32-bit value (uses current pair)
     *
     * @param val 32-bit register value
     */
    void set_clientL_raw(uint32_t val) { set_clientL_raw(val, current_pair_idx); }

    /**
     * @brief Set ClientL configuration from raw 32-bit value for a specific pair
     *
     * @param val 32-bit register value
     * @param pair_idx Pair index (0-63)
     */
    void set_clientL_raw(uint32_t val, uint32_t pair_idx) {
        if (pair_idx < REMAP_NUM_PAIRS) {
            clientL_configs[pair_idx].val = val;
        }
    }

    /**
     * @brief Get direct reference to ClientR config register (uses current pair, for advanced use)
     *
     * @return Reference to ClientR config union
     */
    tClientR_Config_Reg_u& get_clientR_config_ref() { return get_clientR_config_ref(current_pair_idx); }

    /**
     * @brief Get direct reference to ClientR config register for a specific pair (for advanced use)
     *
     * @param pair_idx Pair index (0-63)
     * @return Reference to ClientR config union
     */
    tClientR_Config_Reg_u& get_clientR_config_ref(uint32_t pair_idx) {
        if (pair_idx < REMAP_NUM_PAIRS) {
            return clientR_configs[pair_idx];
        }
        // Return first element as fallback (should not happen with proper bounds checking)
        return clientR_configs[0];
    }

    /**
     * @brief Get direct reference to ClientL config register (uses current pair, for advanced use)
     *
     * @return Reference to ClientL config union
     */
    tClientL_Config_Reg_u& get_clientL_config_ref() { return get_clientL_config_ref(current_pair_idx); }

    /**
     * @brief Get direct reference to ClientL config register for a specific pair (for advanced use)
     *
     * @param pair_idx Pair index (0-63)
     * @return Reference to ClientL config union
     */
    tClientL_Config_Reg_u& get_clientL_config_ref(uint32_t pair_idx) {
        if (pair_idx < REMAP_NUM_PAIRS) {
            return clientL_configs[pair_idx];
        }
        // Return first element as fallback (should not happen with proper bounds checking)
        return clientL_configs[0];
    }
};

#endif  // __DM__REMAPPER_API_HPP__
