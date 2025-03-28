// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

private:
    uint8_t next_trid = 0;
};

template <size_t NUM_CHANNELS, size_t MAX_TRANSACTION_IDS, size_t OFFSET>
struct WriteTransactionIdTracker {
    static constexpr size_t NUM_CHANNELS_PARAM = NUM_CHANNELS;
    static constexpr size_t MAX_TRANSACTION_IDS_PARAM = MAX_TRANSACTION_IDS;
    static constexpr size_t OFFSET_PARAM = OFFSET;
    static constexpr uint8_t INVALID_TRID = OFFSET + MAX_TRANSACTION_IDS;
    static constexpr bool N_TRIDS_IS_POW2 = is_power_of_2(MAX_TRANSACTION_IDS);
    static constexpr bool N_CHANS_IS_POW2 = is_power_of_2(NUM_CHANNELS);
    static constexpr uint8_t TRID_POW2_MASK = MAX_TRANSACTION_IDS - 1;
    static constexpr bool BOTH_PARAMS_ARE_POW2 = N_TRIDS_IS_POW2 && N_CHANS_IS_POW2;

    WriteTransactionIdTracker() {
        for (size_t i = 0; i < NUM_CHANNELS; i++) {
            this->buffer_slot_trids[i] = INVALID_TRID;
        }
    }
    FORCE_INLINE void set_buffer_slot_trid(uint8_t trid, tt::tt_fabric::BufferIndex buffer_index) {
        if constexpr (!BOTH_PARAMS_ARE_POW2) {
            ASSERT(OFFSET_PARAM <= trid && trid <= INVALID_TRID);
            this->buffer_slot_trids[buffer_index] = trid;
        }
    }

    FORCE_INLINE uint8_t
    update_buffer_slot_to_next_trid_and_advance_trid_counter(tt::tt_fabric::BufferIndex buffer_index) {
        if constexpr (BOTH_PARAMS_ARE_POW2) {
            uint8_t next_trid = OFFSET + (buffer_index & TRID_POW2_MASK);
            this->trid_counter.increment();
            return next_trid;
        } else {
            uint8_t next_trid = OFFSET + this->trid_counter.get();
            this->buffer_slot_trids[buffer_index] = next_trid;
            this->trid_counter.increment();
            return next_trid;
        }
    }

    FORCE_INLINE void clear_trid_at_buffer_slot(tt::tt_fabric::BufferIndex buffer_index) {
        if constexpr (!BOTH_PARAMS_ARE_POW2) {
            this->buffer_slot_trids[buffer_index] = INVALID_TRID;
        }
    }

    FORCE_INLINE uint8_t get_buffer_slot_trid(tt::tt_fabric::BufferIndex buffer_index) const {
        if constexpr (BOTH_PARAMS_ARE_POW2) {
            return OFFSET + (buffer_index & TRID_POW2_MASK);
        } else {
            return this->buffer_slot_trids[buffer_index];
        }
    }
    FORCE_INLINE bool transaction_flushed(tt::tt_fabric::BufferIndex buffer_index) const {
        if constexpr (BOTH_PARAMS_ARE_POW2) {
            auto trid = this->get_buffer_slot_trid(buffer_index);
            return ncrisc_noc_nonposted_write_with_transaction_id_sent(tt::tt_fabric::edm_to_local_chip_noc, trid);
        } else {
            // TODO: should be able to remove compare against INVALID_TRID
            auto trid = this->get_buffer_slot_trid(buffer_index);
            return trid == INVALID_TRID ||
                   ncrisc_noc_nonposted_write_with_transaction_id_sent(tt::tt_fabric::edm_to_local_chip_noc, trid);
        }
    }
    FORCE_INLINE void all_buffer_slot_transactions_acked() const {
        for (uint8_t i = 0; i < NUM_CHANNELS; ++i) {
            tt::tt_fabric::BufferIndex buffer_index(i);
            auto trid = this->get_buffer_slot_trid(buffer_index);
            noc_async_write_barrier_with_trid(trid, tt::tt_fabric::edm_to_local_chip_noc);
        }
    }

private:
    std::array<uint8_t, NUM_CHANNELS> buffer_slot_trids;
    TransactionIdCounter<MAX_TRANSACTION_IDS> trid_counter;

    // TODO: cleanup - only used for when both params are pow2, else above are used.
    uint8_t next_trid = 0;
};
