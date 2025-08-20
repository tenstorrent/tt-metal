// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_constants.hpp"

#include "risc_attribs.h"  // for FORCE_INLINE
#include "utils/utils.h"   // for is_power_of_2
#include "debug/assert.h"

#include <cstdint>
#include <array>

template <uint8_t MAX_TRANSACTION_IDS>
struct TransactionIdCounter {
    FORCE_INLINE void increment() {
        this->next_trid = tt::tt_fabric::wrap_increment<MAX_TRANSACTION_IDS>(this->next_trid);
    }

    FORCE_INLINE uint8_t get() const { return this->next_trid; }

    FORCE_INLINE void reset() { this->next_trid = 0; }

private:
    uint8_t next_trid = 0;
};

template <
    uint8_t NUM_CHANNELS,
    size_t MAX_TRANSACTION_IDS,
    size_t OFFSET,
    uint8_t EDM_TO_LOCAL_NOC,
    uint8_t EDM_TO_DOWNSTREAM_NOC>
struct WriteTransactionIdTracker {
    static constexpr size_t NUM_CHANNELS_PARAM = NUM_CHANNELS;
    static constexpr size_t MAX_TRANSACTION_IDS_PARAM = MAX_TRANSACTION_IDS;
    static constexpr size_t OFFSET_PARAM = OFFSET;

    static constexpr uint8_t INVALID_TRID = OFFSET_PARAM + MAX_TRANSACTION_IDS_PARAM;
    static constexpr bool N_TRIDS_IS_POW2 = is_power_of_2(MAX_TRANSACTION_IDS_PARAM);
    static constexpr bool N_CHANS_IS_POW2 = is_power_of_2(NUM_CHANNELS_PARAM);
    static constexpr uint8_t TRID_POW2_MASK = MAX_TRANSACTION_IDS_PARAM - 1;
    static constexpr bool BOTH_PARAMS_ARE_POW2 = N_TRIDS_IS_POW2 && N_CHANS_IS_POW2;
    static constexpr bool BOTH_PARAMS_ARE_EQUAL = NUM_CHANNELS_PARAM == MAX_TRANSACTION_IDS_PARAM;
    static_assert(OFFSET_PARAM + MAX_TRANSACTION_IDS - 1 <= NOC_MAX_TRANSACTION_ID, "Invalid transaction ID");

    WriteTransactionIdTracker() {
        if constexpr (!(BOTH_PARAMS_ARE_EQUAL || BOTH_PARAMS_ARE_POW2)) {
            for (size_t i = 0; i < NUM_CHANNELS_PARAM; i++) {
                this->buffer_slot_trids[i] = INVALID_TRID;
            }
        }
        this->completion_trid_id.reset();
        this->write_trid_id.reset();
        this->completion_trid_available = true;
        this->write_trid_available = true;
        this->write_buffer_index = tt::tt_fabric::BufferIndex{0};
        this->completion_buffer_index = tt::tt_fabric::BufferIndex{0};
    }

    FORCE_INLINE uint8_t update_buffer_slot_to_next_trid_and_advance_trid_counter() {
        if constexpr (BOTH_PARAMS_ARE_EQUAL) {
            return OFFSET_PARAM + write_trid_id.get();
        } else {
            uint8_t next_trid = OFFSET + this->write_trid_id.get();
            this->buffer_slot_trids[this->write_buffer_index] = next_trid;
            this->write_buffer_index = tt::tt_fabric::BufferIndex{
                tt::tt_fabric::wrap_increment<NUM_CHANNELS_PARAM>(this->write_buffer_index.get())};
            this->write_trid_id.increment();
            return next_trid;
        }
    }

    FORCE_INLINE void clear_trid_at_buffer_slot() {
        if constexpr (!(BOTH_PARAMS_ARE_EQUAL || BOTH_PARAMS_ARE_POW2)) {
            this->buffer_slot_trids[this->completion_buffer_index] = INVALID_TRID;
            this->completion_buffer_index = tt::tt_fabric::BufferIndex{
                tt::tt_fabric::wrap_increment<NUM_CHANNELS_PARAM>(this->completion_buffer_index.get())};
        }
    }

    FORCE_INLINE bool next_write_transaction_available() const { return this->write_trid_available; }

    FORCE_INLINE bool next_completion_transaction_available() const { return this->completion_trid_available; }

    FORCE_INLINE void increment_write_trid() { this->write_trid_id.increment(); }

    FORCE_INLINE void increment_completion_trid() { this->completion_trid_id.increment(); }

    FORCE_INLINE bool transaction_flushed_impl(uint8_t trid) const {
        // return true;
        if constexpr (EDM_TO_LOCAL_NOC == EDM_TO_DOWNSTREAM_NOC) {
            return ncrisc_noc_nonposted_write_with_transaction_id_sent(EDM_TO_LOCAL_NOC, trid);
        } else {
            return ncrisc_noc_nonposted_write_with_transaction_id_sent(EDM_TO_DOWNSTREAM_NOC, trid) &&
                   ncrisc_noc_nonposted_write_with_transaction_id_sent(EDM_TO_LOCAL_NOC, trid);
        }
    }
    FORCE_INLINE void update_is_next_write_transaction_available() {
        this->write_trid_available = transaction_flushed_impl(this->write_trid_id.get());
    }

    FORCE_INLINE void update_is_next_completion_transaction_available() {
        this->completion_trid_available = transaction_flushed_impl(this->completion_trid_id.get());
    }

    FORCE_INLINE void all_buffer_slot_transactions_acked() const {
        for (uint8_t trid = OFFSET_PARAM; trid < INVALID_TRID; ++trid) {
            if constexpr (EDM_TO_LOCAL_NOC == EDM_TO_DOWNSTREAM_NOC) {
                noc_async_write_barrier_with_trid(trid, EDM_TO_LOCAL_NOC);
            } else {
                noc_async_write_barrier_with_trid(trid, EDM_TO_DOWNSTREAM_NOC);
                noc_async_write_barrier_with_trid(trid, EDM_TO_LOCAL_NOC);
            }
        }
    }

private:
    // Not used when both params are equal
    std::array<uint8_t, NUM_CHANNELS_PARAM> buffer_slot_trids;
    TransactionIdCounter<MAX_TRANSACTION_IDS_PARAM> completion_trid_id;
    TransactionIdCounter<MAX_TRANSACTION_IDS_PARAM> write_trid_id;
    bool completion_trid_available;
    bool write_trid_available;
    tt::tt_fabric::BufferIndex write_buffer_index;
    tt::tt_fabric::BufferIndex completion_buffer_index;
};
