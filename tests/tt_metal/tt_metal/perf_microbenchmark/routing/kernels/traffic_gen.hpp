// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "debug/dprint.h"

inline uint32_t prng_next(uint32_t n) {
    uint32_t x = n;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

inline uint32_t packet_rnd_seed_to_size(uint32_t rnd_seed, uint32_t max_packet_size_words) {
    uint32_t packet_size = (rnd_seed & (max_packet_size_words-1)) + 1;
    if (packet_size < 2) {
        packet_size = 2;
    }
    return packet_size;
}

// Structure used for randomizing packet sequences based on a starting random seed.
// Used on the TX generator side to generate traffic and on the RX side to verify
// the correcntess of received packets.
typedef struct {

    uint64_t data_words_input;
    uint64_t num_packets;
    uint32_t packet_rnd_seed;
    uint32_t curr_packet_size_words;
    uint32_t curr_packet_words_remaining;
    uint32_t curr_packet_dest;
    uint32_t curr_packet_flags;
    uint32_t num_dests_sent_last_packet;
    bool data_packets_done;
    bool data_and_last_packets_done;

    void init(uint32_t prng_seed, uint32_t endpoint_id) {
        this->packet_rnd_seed = prng_seed ^ endpoint_id;
        this->curr_packet_words_remaining = 0;
        this->data_packets_done = false;
        this->data_and_last_packets_done = false;
        this->num_packets = 0;
        this->data_words_input = 0;
        this->num_dests_sent_last_packet = 0;
    }

    inline void rnd_packet_update(uint32_t num_dest_endpoints,
                                  uint32_t dest_endpoint_start_id,
                                  uint32_t max_packet_size_words,
                                  uint64_t total_data_words) {
        this->curr_packet_dest = ((this->packet_rnd_seed >> 16) & (num_dest_endpoints-1)) + dest_endpoint_start_id;
        this->curr_packet_size_words = packet_rnd_seed_to_size(this->packet_rnd_seed, max_packet_size_words);
        this->curr_packet_flags = 0;
        this->curr_packet_words_remaining = this->curr_packet_size_words;
        this->data_words_input += this->curr_packet_size_words;
        this->data_packets_done = this->data_words_input >= total_data_words;
        this->num_packets++;
    }

    inline void next_packet_rnd(uint32_t num_dest_endpoints,
                                uint32_t dest_endpoint_start_id,
                                uint32_t max_packet_size_words,
                                uint64_t total_data_words) {
        if (!this->data_packets_done) {
            this->packet_rnd_seed = prng_next(this->packet_rnd_seed);
            this->rnd_packet_update(num_dest_endpoints, dest_endpoint_start_id,
                                    max_packet_size_words, total_data_words);
        } else {
            this->packet_rnd_seed = 0xffffffff;
            this->curr_packet_dest = this->num_dests_sent_last_packet + dest_endpoint_start_id;
            this->curr_packet_flags = DispatchPacketFlag::PACKET_TEST_LAST;
            this->curr_packet_size_words = 2;
            this->curr_packet_words_remaining = this->curr_packet_size_words;
            this->data_words_input += 2;
            this->num_packets++;
            this->num_dests_sent_last_packet++;
            if (this->num_dests_sent_last_packet == num_dest_endpoints) {
                this->data_and_last_packets_done = true;
            }
        }
    }

    inline void next_packet_rnd_to_dest(uint32_t num_dest_endpoints,
                                        uint32_t dest_endpoint_id,
                                        uint32_t dest_endpoint_start_id,
                                        uint32_t max_packet_size_words,
                                        uint64_t total_data_words) {
        uint32_t rnd = this->packet_rnd_seed;
        uint32_t dest;
        do {
            rnd = prng_next(rnd);
            dest = (rnd >> 16) & (num_dest_endpoints-1);
        } while (dest != (dest_endpoint_id - dest_endpoint_start_id));
        this->packet_rnd_seed = rnd;
        this->rnd_packet_update(num_dest_endpoints, dest_endpoint_start_id,
                                max_packet_size_words, total_data_words);
    }

    inline bool start_of_packet() {
        return this->curr_packet_words_remaining == this->curr_packet_size_words;
    }

    inline bool packet_active() {
        return this->curr_packet_words_remaining != 0;
    }

    inline bool all_packets_done() {
        return this->data_and_last_packets_done && !this->packet_active();
    }

    inline uint64_t get_data_words_input() {
        return this->data_words_input;
    }

    inline uint64_t get_num_packets() {
        return this->num_packets;
    }

} input_queue_rnd_state_t;


inline void fill_packet_data(tt_l1_ptr uint32_t* start_addr, uint32_t num_words, uint32_t start_val) {
    tt_l1_ptr uint32_t* addr = start_addr + (PACKET_WORD_SIZE_BYTES/4 - 1);
    for (uint32_t i = 0; i < num_words; i++) {
        *addr = start_val++;
        addr += (PACKET_WORD_SIZE_BYTES/4);
    }
}


inline bool check_packet_data(tt_l1_ptr uint32_t* start_addr, uint32_t num_words, uint32_t start_val,
                              uint32_t& mismatch_addr, uint32_t& mismatch_val, uint32_t& expected_val) {
    tt_l1_ptr uint32_t* addr = start_addr + (PACKET_WORD_SIZE_BYTES/4 - 1);
    for (uint32_t i = 0; i < num_words; i++) {
        if (*addr != start_val) {
            mismatch_addr = reinterpret_cast<uint32_t>(addr);
            mismatch_val = *addr;
            expected_val = start_val;
            return false;
        }
        start_val++;
        addr += (PACKET_WORD_SIZE_BYTES/4);
    }
    return true;
}
