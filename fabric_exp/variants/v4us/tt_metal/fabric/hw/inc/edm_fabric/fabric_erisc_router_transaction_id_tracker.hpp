// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"

#include "internal/risc_attribs.h"  // for FORCE_INLINE
#include "api/alignment.h"  // for is_power_of_2
#include "api/debug/assert.h"

#include <cstdint>
#include <array>

template <uint8_t MAX_TRANSACTION_IDS>
struct TransactionIdCounter {
    explicit TransactionIdCounter() = default;

    FORCE_INLINE void init() { this->next_trid = 0; }

    FORCE_INLINE void increment() {
        this->next_trid = tt::tt_fabric::wrap_increment<MAX_TRANSACTION_IDS>(this->next_trid);
    }

    FORCE_INLINE uint8_t get() const { return this->next_trid; }

private:
    uint8_t next_trid;
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
    // A slot's ID is its index masked to the ID count when that count is a power of 2 and the ring is a power of 2 or
    // a whole number of ID rounds: the IDs then go round in slot order, across the ring's wrap too, and no table is kept.
    static constexpr bool SLOT_INDEX_TRIDS =
        N_TRIDS_IS_POW2 && (N_CHANS_IS_POW2 || NUM_CHANNELS_PARAM % MAX_TRANSACTION_IDS_PARAM == 0);
    static_assert(OFFSET_PARAM + MAX_TRANSACTION_IDS - 1 <= NOC_MAX_TRANSACTION_ID, "Invalid transaction ID");

    explicit WriteTransactionIdTracker() = default;

    FORCE_INLINE void init() {
        this->trid_counter.init();
        if constexpr (!(BOTH_PARAMS_ARE_EQUAL || SLOT_INDEX_TRIDS)) {
            for (size_t i = 0; i < NUM_CHANNELS_PARAM; i++) {
                this->buffer_slot_trids[i] = INVALID_TRID;
            }
        }
    }
    FORCE_INLINE void set_buffer_slot_trid(uint8_t trid, tt::tt_fabric::BufferIndex buffer_index) {
        if constexpr (!(BOTH_PARAMS_ARE_EQUAL || SLOT_INDEX_TRIDS)) {
            ASSERT(OFFSET_PARAM <= trid && trid <= INVALID_TRID);
            this->buffer_slot_trids[buffer_index] = trid;
        }
    }

    FORCE_INLINE uint8_t
    update_buffer_slot_to_next_trid_and_advance_trid_counter(tt::tt_fabric::BufferIndex buffer_index) {
        if constexpr (BOTH_PARAMS_ARE_EQUAL) {
            return OFFSET_PARAM + buffer_index;
        } else if constexpr (SLOT_INDEX_TRIDS) {
            return OFFSET_PARAM + (buffer_index & TRID_POW2_MASK);
        } else {
            uint8_t next_trid = OFFSET + this->trid_counter.get();
            this->buffer_slot_trids[buffer_index] = next_trid;
            this->trid_counter.increment();
            return next_trid;
        }
    }

    FORCE_INLINE void clear_trid_at_buffer_slot(tt::tt_fabric::BufferIndex buffer_index) {
        if constexpr (!(BOTH_PARAMS_ARE_EQUAL || SLOT_INDEX_TRIDS)) {
            this->buffer_slot_trids[buffer_index] = INVALID_TRID;
        }
    }

    FORCE_INLINE uint8_t get_buffer_slot_trid(tt::tt_fabric::BufferIndex buffer_index) const {
        if constexpr (BOTH_PARAMS_ARE_EQUAL) {
            return OFFSET_PARAM + buffer_index;
        } else if constexpr (SLOT_INDEX_TRIDS) {
            return OFFSET_PARAM + (buffer_index & TRID_POW2_MASK);
        } else {
            return this->buffer_slot_trids[buffer_index];
        }
    }
    // Experiment (EXP_RX_GROUPED_READS): the NoC status register transaction_flushed() reads, so a caller can issue
    // the read together with others. Only when one NoC carries both local and downstream writes (one register).
    static constexpr bool SINGLE_NOC = EDM_TO_LOCAL_NOC == EDM_TO_DOWNSTREAM_NOC;
    FORCE_INLINE uint32_t transaction_sent_reg_addr(tt::tt_fabric::BufferIndex buffer_index) const {
        static_assert(SINGLE_NOC, "transaction_sent_reg_addr needs one NoC for local and downstream writes");
        return (static_cast<uint32_t>(EDM_TO_LOCAL_NOC) << NOC_INSTANCE_OFFSET_BIT) +
               NOC_STATUS(NIU_MST_WRITE_REQS_OUTGOING_ID(this->get_buffer_slot_trid(buffer_index)));
    }
    FORCE_INLINE bool transaction_flushed(tt::tt_fabric::BufferIndex buffer_index) const {
        auto trid = this->get_buffer_slot_trid(buffer_index);
        if constexpr (EDM_TO_LOCAL_NOC == EDM_TO_DOWNSTREAM_NOC) {
            return ncrisc_noc_nonposted_write_with_transaction_id_sent(EDM_TO_LOCAL_NOC, trid);
        } else {
            return ncrisc_noc_nonposted_write_with_transaction_id_sent(EDM_TO_DOWNSTREAM_NOC, trid) &&
                   ncrisc_noc_nonposted_write_with_transaction_id_sent(EDM_TO_LOCAL_NOC, trid);
        }
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
    TransactionIdCounter<MAX_TRANSACTION_IDS_PARAM> trid_counter;
};
