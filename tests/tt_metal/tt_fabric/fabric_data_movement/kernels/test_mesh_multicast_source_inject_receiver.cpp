// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "test_mesh_multicast_source_inject_common.hpp"

using namespace tt::tt_fabric::fabric_router_tests::source_inject;

constexpr uint32_t test_results_address = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
constexpr uint32_t stop_address = get_compile_time_arg_val(2);
constexpr uint32_t quiescence_cycles = get_compile_time_arg_val(3);

namespace {

volatile tt_l1_ptr uint32_t* const test_results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(test_results_address);
volatile tt_l1_ptr uint32_t* const stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_address);

void record_failure(ValidationPhase phase, uint32_t packet, uint32_t address, uint32_t expected, uint32_t actual) {
    if (test_results[TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_DATA_MISMATCH ||
        test_results[TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_TIMEOUT) {
        return;
    }
    test_results[TT_FABRIC_MISC_INDEX] = static_cast<uint32_t>(phase);
    test_results[TT_FABRIC_MISC_INDEX + 1] = packet;
    test_results[TT_FABRIC_MISC_INDEX + 2] = address;
    test_results[TT_FABRIC_MISC_INDEX + 3] = expected;
    test_results[TT_FABRIC_MISC_INDEX + 4] = actual;
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_DATA_MISMATCH;
}

void record_timeout(ValidationPhase phase, uint32_t packet, uint32_t address, uint32_t expected, uint32_t actual) {
    test_results[TT_FABRIC_MISC_INDEX] = static_cast<uint32_t>(phase);
    test_results[TT_FABRIC_MISC_INDEX + 1] = packet;
    test_results[TT_FABRIC_MISC_INDEX + 2] = address;
    test_results[TT_FABRIC_MISC_INDEX + 3] = expected;
    test_results[TT_FABRIC_MISC_INDEX + 4] = actual;
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_TIMEOUT;
}

bool stop_requested() {
    invalidate_l1_cache();
    return *stop != 0;
}

bool wait_for_at_least_one(
    uint32_t address, ValidationPhase phase, uint32_t packet, volatile tt_l1_ptr uint32_t* value) {
    while (true) {
        invalidate_l1_cache();
        const uint32_t actual = *value;
        if (actual >= 1) {
            return true;
        }
        if (*stop != 0) {
            record_timeout(phase, packet, address, 1, actual);
            return false;
        }
    }
}

bool payload_matches(
    uint32_t address, uint32_t size, ValidationPhase phase, uint32_t packet, uint32_t starting_payload_word = 0) {
    auto* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    const uint32_t num_words = size / sizeof(uint32_t);
    invalidate_l1_cache();
    for (uint32_t word = 0; word < num_words; ++word) {
        if (payload[word] != payload_word(phase, packet, starting_payload_word + word)) {
            return false;
        }
    }
    return true;
}

bool validate_payload_words(
    uint32_t address, uint32_t size, ValidationPhase phase, uint32_t packet, uint32_t starting_payload_word = 0) {
    auto* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    const uint32_t num_words = size / sizeof(uint32_t);
    invalidate_l1_cache();
    for (uint32_t word = 0; word < num_words; ++word) {
        const uint32_t expected = payload_word(phase, packet, starting_payload_word + word);
        const uint32_t actual = payload[word];
        if (actual != expected) {
            record_failure(phase, packet, address + word * sizeof(uint32_t), expected, actual);
            return false;
        }
    }
    return true;
}

bool validate_sentinel_region(uint32_t address, uint32_t size, uint32_t region_index) {
    auto* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    const uint32_t num_words = size / sizeof(uint32_t);
    invalidate_l1_cache();
    for (uint32_t word = 0; word < num_words; ++word) {
        const uint32_t actual = payload[word];
        if (actual != SENTINEL) {
            record_failure(
                ValidationPhase::NON_TARGET, region_index, address + word * sizeof(uint32_t), SENTINEL, actual);
            return false;
        }
    }
    return true;
}

bool validate_all_counters(uint32_t data_base, uint32_t payload_size, uint32_t expected) {
    for (uint32_t index = 0; index < ATOMIC_COUNTER_COUNT; ++index) {
        const uint32_t address = counter_address(data_base, payload_size, index);
        auto* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
        invalidate_l1_cache();
        const uint32_t actual = *counter;
        if (actual != expected) {
            record_failure(ValidationPhase::ATOMIC, index, address, expected, actual);
            return false;
        }
    }
    return true;
}

bool validate_all_target_payloads(uint32_t data_base, uint32_t payload_size) {
    const uint32_t first_chunk = first_scatter_chunk_size(payload_size);
    const uint32_t second_chunk = second_scatter_chunk_size(payload_size);
    const uint32_t first_chunk_words = first_chunk / sizeof(uint32_t);

    for (uint32_t packet = 0; packet < PAYLOAD_PACKET_COUNT; ++packet) {
        if (!validate_payload_words(
                fused_write_address(data_base, payload_size, packet),
                payload_size,
                ValidationPhase::FUSED_WRITE,
                packet)) {
            return false;
        }
        if (!validate_payload_words(
                scatter_first_address(data_base, payload_size, packet),
                first_chunk,
                ValidationPhase::FUSED_SCATTER,
                packet) ||
            !validate_payload_words(
                scatter_second_address(data_base, payload_size, packet),
                second_chunk,
                ValidationPhase::FUSED_SCATTER,
                packet,
                first_chunk_words)) {
            return false;
        }
        if (!validate_payload_words(
                plain_write_address(data_base, payload_size, packet),
                payload_size,
                ValidationPhase::PLAIN_WRITE,
                packet)) {
            return false;
        }
    }
    return true;
}

bool validate_non_target_payloads(uint32_t data_base, uint32_t payload_size) {
    return validate_sentinel_region(
               fused_write_base(data_base, payload_size), PAYLOAD_PACKET_COUNT * payload_size, 0) &&
           validate_sentinel_region(
               scatter_first_base(data_base, payload_size),
               PAYLOAD_PACKET_COUNT * first_scatter_chunk_size(payload_size),
               1) &&
           validate_sentinel_region(
               plain_write_base(data_base, payload_size), PAYLOAD_PACKET_COUNT * payload_size, 2) &&
           validate_sentinel_region(
               scatter_second_base(data_base, payload_size),
               PAYLOAD_PACKET_COUNT * second_scatter_chunk_size(payload_size),
               3);
}

bool wait_for_target_traffic(uint32_t data_base, uint32_t payload_size) {
    for (uint32_t packet = 0; packet < ATOMIC_PACKET_COUNT; ++packet) {
        const uint32_t address = counter_address(data_base, payload_size, packet);
        auto* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
        if (!wait_for_at_least_one(address, ValidationPhase::ATOMIC, packet, counter)) {
            return false;
        }
    }

    for (uint32_t packet = 0; packet < PAYLOAD_PACKET_COUNT; ++packet) {
        const uint32_t counter_index = ATOMIC_PACKET_COUNT + packet;
        const uint32_t address = counter_address(data_base, payload_size, counter_index);
        auto* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
        if (!wait_for_at_least_one(address, ValidationPhase::FUSED_WRITE, packet, counter) ||
            !validate_payload_words(
                fused_write_address(data_base, payload_size, packet),
                payload_size,
                ValidationPhase::FUSED_WRITE,
                packet)) {
            return false;
        }
    }

    const uint32_t first_chunk = first_scatter_chunk_size(payload_size);
    const uint32_t second_chunk = second_scatter_chunk_size(payload_size);
    const uint32_t first_chunk_words = first_chunk / sizeof(uint32_t);
    for (uint32_t packet = 0; packet < PAYLOAD_PACKET_COUNT; ++packet) {
        const uint32_t counter_index = ATOMIC_PACKET_COUNT + PAYLOAD_PACKET_COUNT + packet;
        const uint32_t address = counter_address(data_base, payload_size, counter_index);
        auto* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
        if (!wait_for_at_least_one(address, ValidationPhase::FUSED_SCATTER, packet, counter) ||
            !validate_payload_words(
                scatter_first_address(data_base, payload_size, packet),
                first_chunk,
                ValidationPhase::FUSED_SCATTER,
                packet) ||
            !validate_payload_words(
                scatter_second_address(data_base, payload_size, packet),
                second_chunk,
                ValidationPhase::FUSED_SCATTER,
                packet,
                first_chunk_words)) {
            return false;
        }
    }

    for (uint32_t packet = 0; packet < PAYLOAD_PACKET_COUNT; ++packet) {
        const uint32_t address = plain_write_address(data_base, payload_size, packet);
        // Plain writes have no completion side effect. Poll the complete deterministic payload rather
        // than assuming that observing its tail word is a documented remote completion fence.
        while (!payload_matches(address, payload_size, ValidationPhase::PLAIN_WRITE, packet)) {
            if (stop_requested()) {
                validate_payload_words(address, payload_size, ValidationPhase::PLAIN_WRITE, packet);
                test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_TIMEOUT;
                return false;
            }
        }
    }

    return validate_all_counters(data_base, payload_size, 1);
}

}  // namespace

void kernel_main() {
    size_t rt_arg_idx = 0;
    const uint32_t data_base = get_arg_val<uint32_t>(rt_arg_idx++);
    const uint32_t payload_size = get_arg_val<uint32_t>(rt_arg_idx++);
    const bool is_target = get_arg_val<uint32_t>(rt_arg_idx++) != 0;

    for (uint32_t i = 0; i < test_results_size_bytes / sizeof(uint32_t); ++i) {
        test_results[i] = 0;
    }
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    bool valid = true;
    if (is_target) {
        valid = wait_for_target_traffic(data_base, payload_size);
    } else {
        // Source and non-target chips must never receive source-injected local delivery. They do not
        // wait for packets; the host STOP handshake tells them when the target-side observation window ends.
        valid =
            validate_all_counters(data_base, payload_size, 0) && validate_non_target_payloads(data_base, payload_size);
    }

    if (valid) {
        test_results[TT_FABRIC_STATUS_INDEX] = STATUS_READY_TO_STOP;
    }

    // READY_TO_STOP is intentionally not PASS. Destination atomics are posted and source teardown is
    // not a remote drain, so remain observable until the host sees every target ready and releases us.
    while (!stop_requested()) {
        if (is_target) {
            valid = validate_all_counters(data_base, payload_size, 1) && valid;
        } else {
            valid = validate_all_counters(data_base, payload_size, 0) && valid;
        }
    }

    // STOP begins a bounded quiescence window rather than ending validation immediately. Source
    // teardown drains source-to-EDM NOC traffic, not the remote fabric, so keep scanning long enough
    // to expose a late duplicate counter update or non-target write before publishing final PASS.
    const uint64_t quiescence_start = get_timestamp();
    do {
        if (is_target) {
            valid = validate_all_counters(data_base, payload_size, 1) &&
                    validate_all_target_payloads(data_base, payload_size) && valid;
        } else {
            valid = validate_all_counters(data_base, payload_size, 0) &&
                    validate_non_target_payloads(data_base, payload_size) && valid;
        }
    } while (get_timestamp() - quiescence_start < quiescence_cycles);

    if (valid) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    } else if (test_results[TT_FABRIC_STATUS_INDEX] != TT_FABRIC_STATUS_TIMEOUT) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_DATA_MISMATCH;
    }
}
