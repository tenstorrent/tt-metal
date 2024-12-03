// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "debug/dprint.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen_test.hpp"

#define is_power_of_2(x) (((x) > 0) && (((x) & ((x) - 1)) == 0))

inline uint32_t prng_next(uint32_t n) {
    uint32_t x = n;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

inline uint32_t packet_rnd_seed_to_size(uint32_t rnd_seed, uint32_t max_packet_size_words, uint32_t max_packet_size_mask) {
    uint32_t packet_size = (rnd_seed & (max_packet_size_mask)) + 1;
    if (packet_size > max_packet_size_words) {
        packet_size -= max_packet_size_words;
    }
    if (packet_size <= PACKET_HEADER_SIZE_WORDS) {
        packet_size = 4;
    }
    return packet_size;
}

struct input_queue_raw_state_t {

    uint64_t data_words_input = 0;
    uint64_t num_packets = 0;
    uint32_t curr_packet_size_words = 0;
    uint32_t curr_packet_words_remaining = 0;
    uint32_t curr_packet_dest = 0;
    uint32_t curr_packet_flags = 0;
    uint32_t num_dests_sent_last_packet = 0;
    bool data_packets_done = false;
    bool data_and_last_packets_done = false;
    uint32_t packet_dest_diff = 0; // difference from curr_packet_dest to dest_endpoint_start_id
    uint32_t packet_rnd_seed = 0;

    void init() {
        this->curr_packet_words_remaining = 0;
        this->data_packets_done = false;
        this->data_and_last_packets_done = false;
        this->num_packets = 0;
        this->data_words_input = 0;
        this->num_dests_sent_last_packet = 0;
    }

    inline void packet_update(uint32_t num_dest_endpoints, uint64_t total_data_words) {
        this->curr_packet_flags = 0;
        this->curr_packet_words_remaining = this->curr_packet_size_words;
        this->data_words_input += this->curr_packet_size_words;
        this->data_packets_done = this->data_words_input >= total_data_words;
        this->num_packets++;
    }

    inline void gen_last_pkt(uint32_t num_dest_endpoints,
                                uint32_t dest_endpoint_start_id,
                                uint32_t max_packet_size_words,
                                uint64_t total_data_words) {
        /*
        this->curr_packet_dest = this->num_dests_sent_last_packet + dest_endpoint_start_id;
        this->curr_packet_flags = TERMINATE;
        this->curr_packet_size_words = PACKET_HEADER_SIZE_WORDS;
        this->curr_packet_words_remaining = this->curr_packet_size_words;
        this->data_words_input += this->curr_packet_size_words;
        this->num_packets++;
        this->num_dests_sent_last_packet++;
        */
        //if (this->num_dests_sent_last_packet == num_dest_endpoints) {
            this->data_and_last_packets_done = true;
        //}
    }

    inline bool start_of_packet() {
        return this->curr_packet_words_remaining == this->curr_packet_size_words;
    }

    inline bool packet_active() {
        return this->curr_packet_words_remaining != 0;
    }

    inline bool all_packets_done() {
        //return this->data_and_last_packets_done && !this->packet_active();
        return this->data_packets_done && !this->packet_active();
    }

    inline uint64_t get_data_words_input() {
        return this->data_words_input;
    }

    inline uint64_t get_num_packets() {
        return this->num_packets;
    }

};

struct input_queue_same_start_rndrobin_fix_size_state_t : public input_queue_raw_state_t{

    void init(uint32_t pkt_size_words, uint32_t) { // same args as the input_queue_rnd_state_t
        input_queue_raw_state_t::init();

        packet_dest_diff = 0;
        curr_packet_size_words = pkt_size_words;
    }

    inline void packet_update(uint32_t num_dest_endpoints,
                                  uint32_t dest_endpoint_start_id,
                                  uint32_t max_packet_size_words,
                                  uint64_t total_data_words) {
        curr_packet_dest = packet_dest_diff + dest_endpoint_start_id;
        packet_dest_diff = (packet_dest_diff + 1) & (num_dest_endpoints - 1);
        input_queue_raw_state_t::packet_update(num_dest_endpoints, total_data_words);
    }
    inline void next_packet(uint32_t num_dest_endpoints,
                                uint32_t dest_endpoint_start_id,
                                uint32_t max_packet_size_words,
                                uint64_t total_data_words) {
        if (!data_packets_done) {
            this->packet_update(num_dest_endpoints, dest_endpoint_start_id, max_packet_size_words, total_data_words);
        } else {
            packet_rnd_seed = 0xffffffff;
            this->gen_last_pkt(num_dest_endpoints, dest_endpoint_start_id, max_packet_size_words, total_data_words);
        }
    }
};



// Structure used for randomizing packet sequences based on a starting random seed.
// Used on the TX generator side to generate traffic and on the RX side to verify
// the correcntess of received packets.
struct input_queue_rnd_state_t : public input_queue_raw_state_t {

    void init(uint32_t endpoint_id, uint32_t prng_seed) {
        packet_rnd_seed = prng_seed ^ endpoint_id;
        input_queue_raw_state_t::init();
    }

    inline void packet_update(uint32_t num_dest_endpoints,
                                  uint32_t dest_endpoint_start_id,
                                  uint32_t max_packet_size_words,
                                  uint32_t max_packet_size_mask,
                                  uint64_t total_data_words) {
        curr_packet_dest = ((packet_rnd_seed >> 16) & (num_dest_endpoints-1)) + dest_endpoint_start_id;
        curr_packet_size_words = packet_rnd_seed_to_size(packet_rnd_seed, max_packet_size_words, max_packet_size_mask);
        input_queue_raw_state_t::packet_update(num_dest_endpoints, total_data_words);
    }

    inline void next_packet(uint32_t num_dest_endpoints,
                                uint32_t dest_endpoint_start_id,
                                uint32_t max_packet_size_words,
                                uint32_t max_packet_size_mask,
                                uint64_t total_data_words) {
        if (!data_packets_done) {
            packet_rnd_seed = prng_next(packet_rnd_seed);
            this->packet_update(num_dest_endpoints, dest_endpoint_start_id, max_packet_size_words, max_packet_size_mask, total_data_words);
        } else {
            packet_rnd_seed = 0xffffffff;
            this->gen_last_pkt(num_dest_endpoints, dest_endpoint_start_id, max_packet_size_words, total_data_words);
        }
    }

    inline void next_inline_packet(uint64_t total_data_words) {
        if (!data_packets_done) {
            packet_rnd_seed = prng_next(packet_rnd_seed);
            curr_packet_size_words = PACKET_HEADER_SIZE_WORDS;
            input_queue_raw_state_t::packet_update(0, total_data_words);
        }
    }

    inline void next_packet_rnd_to_dest(uint32_t num_dest_endpoints,
                                        uint32_t dest_endpoint_id,
                                        uint32_t dest_endpoint_start_id,
                                        uint32_t max_packet_size_words,
                                        uint32_t max_packet_size_mask,
                                        uint64_t total_data_words) {
        uint32_t rnd = packet_rnd_seed;
        uint32_t dest;
        do {
            rnd = prng_next(rnd);
            dest = (rnd >> 16) & (num_dest_endpoints-1);
        } while (dest != (dest_endpoint_id - dest_endpoint_start_id));
        packet_rnd_seed = rnd;
        this->packet_update(num_dest_endpoints, dest_endpoint_start_id, max_packet_size_words, max_packet_size_mask, total_data_words);
    }

};

template <pkt_dest_size_choices_t which_t>
constexpr auto select_input_queue() {
    if constexpr (which_t == pkt_dest_size_choices_t::RANDOM) {
        return input_queue_rnd_state_t{};
    } else if constexpr (which_t == pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE) {
        return input_queue_same_start_rndrobin_fix_size_state_t{};
    } else {
        return input_queue_rnd_state_t{}; // default
    }
}


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
