// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/flow-control/credits.hpp"

using namespace tt::tt_fabric;

// Test result values (for individual tests)
constexpr uint32_t TEST_PASS = 1;
constexpr uint32_t TEST_FAIL = 2;
constexpr uint32_t TEST_NOT_RUN = 0;

// Overall test completion status
enum class TestCompletionStatus : uint32_t {
    NOT_STARTED = 0,
    COMPLETED = 1,
    CRASHED = 2
};

// Get compile-time args
constexpr uint32_t results_l1_address = get_compile_time_arg_val(0);
constexpr uint32_t num_test_cases = get_compile_time_arg_val(1);

// Buffer layout offsets
constexpr uint32_t TEST_STATUS_OFFSET = 0;
constexpr uint32_t NUM_TESTS_RUN_OFFSET = 1;
constexpr uint32_t FIRST_TEST_RESULT_OFFSET = 2;

// Helper to write test result
inline void write_test_result(uint32_t test_id, uint32_t result) {
    volatile uint32_t* results = reinterpret_cast<volatile uint32_t*>(results_l1_address);
    results[FIRST_TEST_RESULT_OFFSET + test_id] = result;
}

// Helper to write overall test status (called at end)
inline void write_test_completion_status(uint32_t num_tests_run) {
    volatile uint32_t* results = reinterpret_cast<volatile uint32_t*>(results_l1_address);
    results[NUM_TESTS_RUN_OFFSET] = num_tests_run;
    results[TEST_STATUS_OFFSET] = static_cast<uint32_t>(TestCompletionStatus::COMPLETED);
}

// Test: CreditPacking extract_channel for 1 byte-aligned channel
void test_credit_packing_1ch_byte_aligned_extract(uint32_t& test_id) {
    using Packing = CreditPacking<1, 8>;

    // Test 1: Extract single channel
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0xAB);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        write_test_result(test_id++, (ch0 == 0xAB) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: CreditPacking sum for 1 byte-aligned channel
void test_credit_packing_1ch_byte_aligned_sum(uint32_t& test_id) {
    using Packing = CreditPacking<1, 8>;

    // Test 1: Sum of single channel (should equal the value)
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 42);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 42) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Sum of zero
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 0) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: CreditPacking extract_channel for 2 byte-aligned channels
void test_credit_packing_2ch_byte_aligned_extract(uint32_t& test_id) {
    using Packing = CreditPacking<2, 8>;

    // Test 1: Extract channel 0
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x12);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        write_test_result(test_id++, (ch0 == 0x12) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Extract channel 1
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<1>(packed, 0x34);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        write_test_result(test_id++, (ch1 == 0x34) ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Extract both channels
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x78);
        Packing::pack_channel<1>(packed, 0x56);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        write_test_result(test_id++, (ch0 == 0x78 && ch1 == 0x56) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: CreditPacking sum for 2 byte-aligned channels
void test_credit_packing_2ch_byte_aligned_sum(uint32_t& test_id) {
    using Packing = CreditPacking<2, 8>;

    // Test 1: Sum of zeros
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0);
        Packing::pack_channel<1>(packed, 0);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 0) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Sum of non-zero values
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 20);
        Packing::pack_channel<1>(packed, 10);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 30) ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Sum with reasonable values
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 100);
        Packing::pack_channel<1>(packed, 50);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 150) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: CreditPacking extract_channel for 3 byte-aligned channels
void test_credit_packing_3ch_byte_aligned_extract(uint32_t& test_id) {
    using Packing = CreditPacking<3, 8>;

    // Test 1: Extract each channel separately
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x12);
        Packing::pack_channel<1>(packed, 0xCD);
        Packing::pack_channel<2>(packed, 0xAB);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        uint32_t ch2 = Packing::extract_channel<2>(packed);
        write_test_result(test_id++, (ch0 == 0x12 && ch1 == 0xCD && ch2 == 0xAB) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: CreditPacking sum for 3 byte-aligned channels
void test_credit_packing_3ch_byte_aligned_sum(uint32_t& test_id) {
    using Packing = CreditPacking<3, 8>;

    // Test 1: Sum of values
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 40);
        Packing::pack_channel<1>(packed, 20);
        Packing::pack_channel<2>(packed, 10);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 70) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Sum with different values
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 100);
        Packing::pack_channel<1>(packed, 50);
        Packing::pack_channel<2>(packed, 25);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 175) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: CreditPacking extract_channel for 4 byte-aligned channels
void test_credit_packing_4ch_byte_aligned_extract(uint32_t& test_id) {
    using Packing = CreditPacking<4, 8>;

    // Test 1: Extract each channel
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0xEF);
        Packing::pack_channel<1>(packed, 0xBE);
        Packing::pack_channel<2>(packed, 0xAD);
        Packing::pack_channel<3>(packed, 0xDE);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        uint32_t ch2 = Packing::extract_channel<2>(packed);
        uint32_t ch3 = Packing::extract_channel<3>(packed);
        write_test_result(test_id++, (ch0 == 0xEF && ch1 == 0xBE && ch2 == 0xAD && ch3 == 0xDE) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: CreditPacking sum for 4 byte-aligned channels
void test_credit_packing_4ch_byte_aligned_sum(uint32_t& test_id) {
    using Packing = CreditPacking<4, 8>;

    // Test 1: Sum of values
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 40);
        Packing::pack_channel<1>(packed, 30);
        Packing::pack_channel<2>(packed, 20);
        Packing::pack_channel<3>(packed, 10);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 100) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Sum with different values
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 25);
        Packing::pack_channel<1>(packed, 25);
        Packing::pack_channel<2>(packed, 25);
        Packing::pack_channel<3>(packed, 25);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 100) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: CreditPacking pack_channel for byte-aligned channels
void test_credit_packing_pack_channel(uint32_t& test_id) {
    using Packing = CreditPacking<4, 8>;

    // Test 1: Pack channel 0
    {
        auto packed = Packing::pack_channel<0>(0x42);
        write_test_result(test_id++, (packed.value == 0x0000'0042) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Pack channel 1
    {
        auto packed = Packing::pack_channel<1>(0x42);
        write_test_result(test_id++, (packed.value == 0x0000'4200) ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Pack channel 2
    {
        auto packed = Packing::pack_channel<2>(0x42);
        write_test_result(test_id++, (packed.value == 0x0042'0000) ? TEST_PASS : TEST_FAIL);
    }

    // Test 4: Pack channel 3
    {
        auto packed = Packing::pack_channel<3>(0x42);
        write_test_result(test_id++, (packed.value == 0x4200'0000) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Non-byte-aligned 4 channels with 6-bit credits
void test_credit_packing_4ch_6bit_non_byte_aligned(uint32_t& test_id) {
    using Packing = CreditPacking<4, 6>;

    // Test 1: Extract channels
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 5);
        Packing::pack_channel<1>(packed, 10);
        Packing::pack_channel<2>(packed, 15);
        Packing::pack_channel<3>(packed, 20);

        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        uint32_t ch2 = Packing::extract_channel<2>(packed);
        uint32_t ch3 = Packing::extract_channel<3>(packed);

        write_test_result(test_id++, (ch0 == 5 && ch1 == 10 && ch2 == 15 && ch3 == 20) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Sum channels
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 5);
        Packing::pack_channel<1>(packed, 10);
        Packing::pack_channel<2>(packed, 15);
        Packing::pack_channel<3>(packed, 20);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 50) ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Max values for 6-bit (63)
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 63);
        Packing::pack_channel<1>(packed, 63);
        Packing::pack_channel<2>(packed, 63);
        Packing::pack_channel<3>(packed, 63);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 252) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: PackedCreditContainer basic operations
void test_packed_credit_container(uint32_t& test_id) {
    // Test 1: Container creation and get
    {
        PackedCreditContainer<2, 8> container{0x1234};
        write_test_result(test_id++, (container.get() == 0x1234) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Container equality
    {
        PackedCreditContainer<2, 8> c1{0x5678};
        PackedCreditContainer<2, 8> c2{0x5678};
        write_test_result(test_id++, (c1 == c2) ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Container inequality
    {
        PackedCreditContainer<2, 8> c1{0x1234};
        PackedCreditContainer<2, 8> c2{0x5678};
        write_test_result(test_id++, (c1 != c2) ? TEST_PASS : TEST_FAIL);
    }

    // Test 4: Container addition
    {
        PackedCreditContainer<2, 8> c1{0x0101};  // ch0=1, ch1=1
        PackedCreditContainer<2, 8> c2{0x0202};  // ch0=2, ch1=2
        auto c3 = c1 + c2;
        write_test_result(test_id++, (c3.value == 0x0303) ? TEST_PASS : TEST_FAIL);  // ch0=3, ch1=3
    }
}

// Test: 5 byte-aligned channels (uses uint64_t storage)
void test_credit_packing_5ch_byte_aligned(uint32_t& test_id) {
    using Packing = CreditPacking<5, 8>;

    // Test 1: Extract channels
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x11);
        Packing::pack_channel<1>(packed, 0x22);
        Packing::pack_channel<2>(packed, 0x33);
        Packing::pack_channel<3>(packed, 0x44);
        Packing::pack_channel<4>(packed, 0x55);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        uint32_t ch2 = Packing::extract_channel<2>(packed);
        uint32_t ch3 = Packing::extract_channel<3>(packed);
        uint32_t ch4 = Packing::extract_channel<4>(packed);

        bool pass = (ch0 == 0x11) && (ch1 == 0x22) && (ch2 == 0x33) && (ch3 == 0x44) && (ch4 == 0x55);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Sum all channels
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 10);
        Packing::pack_channel<1>(packed, 20);
        Packing::pack_channel<2>(packed, 30);
        Packing::pack_channel<3>(packed, 40);
        Packing::pack_channel<4>(packed, 50);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 150) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Pack channel for 5 channels
void test_credit_packing_5ch_pack(uint32_t& test_id) {
    using Packing = CreditPacking<5, 8>;

    // Test 1: Pack channel 0 (new container)
    {
        auto packed = Packing::pack_channel<0>(0xAB);
        write_test_result(test_id++, (packed.value == 0x0000'0000'0000'00ABULL) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Pack channel 4 (new container)
    {
        auto packed = Packing::pack_channel<4>(0xCD);
        write_test_result(test_id++, (packed.value == 0x0000'00CD'0000'0000ULL) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: pack_channel with existing packed value for all channel counts (new API)
void test_credit_packing_pack_into_existing_all_channel_counts(uint32_t& test_id) {
    // Test 1-channel pack_into
    {
        using Packing = CreditPacking<1, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0xAB);
        write_test_result(test_id++, (packed.value == 0xAB) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2-channel pack_into - individual then combined
    {
        using Packing = CreditPacking<2, 8>;

        // First: Test packing each channel INDIVIDUALLY on fresh values
        {
            auto p0 = Packing::PackedValueType{0};
            Packing::pack_channel<0>(p0, 0x12);
            write_test_result(test_id++, (Packing::extract_channel<0>(p0) == 0x12) ? TEST_PASS : TEST_FAIL);
        }
        {
            auto p1 = Packing::PackedValueType{0};
            Packing::pack_channel<1>(p1, 0x34);
            write_test_result(test_id++, (Packing::extract_channel<1>(p1) == 0x34) ? TEST_PASS : TEST_FAIL);
        }

        // Second: Test packing ALL channels TOGETHER sequentially
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x12);
        Packing::pack_channel<1>(packed, 0x34);
        write_test_result(test_id++, (packed.value == 0x3412) ? TEST_PASS : TEST_FAIL);
    }

    // Test 3-channel pack_into - individual then combined
    {
        using Packing = CreditPacking<3, 8>;

        // First: Test packing each channel INDIVIDUALLY on fresh values
        {
            auto p0 = Packing::PackedValueType{0};
            Packing::pack_channel<0>(p0, 0x11);
            write_test_result(test_id++, (Packing::extract_channel<0>(p0) == 0x11) ? TEST_PASS : TEST_FAIL);
        }
        {
            auto p1 = Packing::PackedValueType{0};
            Packing::pack_channel<1>(p1, 0x22);
            write_test_result(test_id++, (Packing::extract_channel<1>(p1) == 0x22) ? TEST_PASS : TEST_FAIL);
        }
        {
            auto p2 = Packing::PackedValueType{0};
            Packing::pack_channel<2>(p2, 0x33);
            write_test_result(test_id++, (Packing::extract_channel<2>(p2) == 0x33) ? TEST_PASS : TEST_FAIL);
        }

        // Second: Test packing ALL channels TOGETHER sequentially
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x11);
        Packing::pack_channel<1>(packed, 0x22);
        Packing::pack_channel<2>(packed, 0x33);
        write_test_result(test_id++, (packed.value == 0x332211) ? TEST_PASS : TEST_FAIL);
    }

    // Test 4-channel pack_into - individual then combined
    {
        using Packing = CreditPacking<4, 8>;

        // First: Test packing each channel INDIVIDUALLY on fresh values
        {
            auto p0 = Packing::PackedValueType{0};
            Packing::pack_channel<0>(p0, 0xAA);
            write_test_result(test_id++, (Packing::extract_channel<0>(p0) == 0xAA) ? TEST_PASS : TEST_FAIL);
        }
        {
            auto p1 = Packing::PackedValueType{0};
            Packing::pack_channel<1>(p1, 0xBB);
            write_test_result(test_id++, (Packing::extract_channel<1>(p1) == 0xBB) ? TEST_PASS : TEST_FAIL);
        }
        {
            auto p2 = Packing::PackedValueType{0};
            Packing::pack_channel<2>(p2, 0xCC);
            write_test_result(test_id++, (Packing::extract_channel<2>(p2) == 0xCC) ? TEST_PASS : TEST_FAIL);
        }
        {
            auto p3 = Packing::PackedValueType{0};
            Packing::pack_channel<3>(p3, 0xDD);
            write_test_result(test_id++, (Packing::extract_channel<3>(p3) == 0xDD) ? TEST_PASS : TEST_FAIL);
        }

        // Second: Test packing ALL channels TOGETHER sequentially
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0xAA);
        Packing::pack_channel<1>(packed, 0xBB);
        Packing::pack_channel<2>(packed, 0xCC);
        Packing::pack_channel<3>(packed, 0xDD);

        // Verify all channels are correct
        bool pass = (Packing::extract_channel<0>(packed) == 0xAA) &&
                    (Packing::extract_channel<1>(packed) == 0xBB) &&
                    (Packing::extract_channel<2>(packed) == 0xCC) &&
                    (Packing::extract_channel<3>(packed) == 0xDD);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 5-channel pack_into - individual then combined
    {
        using Packing = CreditPacking<5, 8>;

        // First: Test packing each channel INDIVIDUALLY on fresh values
        {
            auto p0 = Packing::PackedValueType{0};
            Packing::pack_channel<0>(p0, 0x10);
            write_test_result(test_id++, (Packing::extract_channel<0>(p0) == 0x10) ? TEST_PASS : TEST_FAIL);
        }
        {
            auto p1 = Packing::PackedValueType{0};
            Packing::pack_channel<1>(p1, 0x20);
            write_test_result(test_id++, (Packing::extract_channel<1>(p1) == 0x20) ? TEST_PASS : TEST_FAIL);
        }
        {
            auto p2 = Packing::PackedValueType{0};
            Packing::pack_channel<2>(p2, 0x30);
            write_test_result(test_id++, (Packing::extract_channel<2>(p2) == 0x30) ? TEST_PASS : TEST_FAIL);
        }
        {
            auto p3 = Packing::PackedValueType{0};
            Packing::pack_channel<3>(p3, 0x40);
            write_test_result(test_id++, (Packing::extract_channel<3>(p3) == 0x40) ? TEST_PASS : TEST_FAIL);
        }
        {
            auto p4 = Packing::PackedValueType{0};
            Packing::pack_channel<4>(p4, 0x50);
            write_test_result(test_id++, (Packing::extract_channel<4>(p4) == 0x50) ? TEST_PASS : TEST_FAIL);
        }

        // Second: Test packing ALL channels TOGETHER sequentially
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x10);
        Packing::pack_channel<1>(packed, 0x20);
        Packing::pack_channel<2>(packed, 0x30);
        Packing::pack_channel<3>(packed, 0x40);
        Packing::pack_channel<4>(packed, 0x50);

        // Verify all channels are correct
        bool all_ok = (Packing::extract_channel<0>(packed) == 0x10) &&
                      (Packing::extract_channel<1>(packed) == 0x20) &&
                      (Packing::extract_channel<2>(packed) == 0x30) &&
                      (Packing::extract_channel<3>(packed) == 0x40) &&
                      (Packing::extract_channel<4>(packed) == 0x50);
        write_test_result(test_id++, all_ok ? TEST_PASS : TEST_FAIL);
    }
}

// Test: pack_channel non-sequential order and special cases
void test_credit_packing_pack_into_existing_special_cases(uint32_t& test_id) {
    using Packing = CreditPacking<4, 8>;

    // Test 1: Pack channels in non-sequential order
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<3>(packed, 0xAA);
        Packing::pack_channel<1>(packed, 0xBB);
        Packing::pack_channel<0>(packed, 0xCC);
        Packing::pack_channel<2>(packed, 0xDD);
        write_test_result(test_id++, (packed.value == 0xAADD'BBCC) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Pack into partially filled container
    {
        auto packed = Packing::PackedValueType{0x0000'1200};  // ch1 = 0x12
        Packing::pack_channel<0>(packed, 0x34);               // Add ch0
        Packing::pack_channel<2>(packed, 0x56);               // Add ch2
        write_test_result(test_id++, (packed.value == 0x0056'1234) ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Runtime version (channel_id as parameter) - all channel counts
    {
        // 1 channel
        using Packing1 = CreditPacking<1, 8>;
        auto packed1 = Packing1::PackedValueType{0};
        Packing1::pack_channel(packed1, 0, 0xAA);
        write_test_result(test_id++, (packed1.value == 0xAA) ? TEST_PASS : TEST_FAIL);
    }

    {
        // 2 channels
        using Packing2 = CreditPacking<2, 8>;
        auto packed2 = Packing2::PackedValueType{0};
        Packing2::pack_channel(packed2, 0, 0xBB);
        Packing2::pack_channel(packed2, 1, 0xCC);
        write_test_result(test_id++, (packed2.value == 0xCCBB) ? TEST_PASS : TEST_FAIL);
    }

    {
        // 3 channels
        using Packing3 = CreditPacking<3, 8>;
        auto packed3 = Packing3::PackedValueType{0};
        Packing3::pack_channel(packed3, 0, 0x11);
        Packing3::pack_channel(packed3, 1, 0x22);
        Packing3::pack_channel(packed3, 2, 0x33);
        write_test_result(test_id++, (packed3.value == 0x3322'11) ? TEST_PASS : TEST_FAIL);
    }

    // NOTE: Removed old "Test 4: Verify OR behavior" because OR was a BUG.
    // pack_channel now correctly REPLACES values instead of ORing them.
    // The replacement behavior is thoroughly tested in test_pack_channel_replacement_not_or()
}

// Test: pack_channel with 5-channel uint64_t storage
void test_credit_packing_5ch_pack_into_existing(uint32_t& test_id) {
    using Packing = CreditPacking<5, 8>;

    // Test 1: Pack all 5 channels sequentially
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x11);
        Packing::pack_channel<1>(packed, 0x22);
        Packing::pack_channel<2>(packed, 0x33);
        Packing::pack_channel<3>(packed, 0x44);
        Packing::pack_channel<4>(packed, 0x55);
        write_test_result(test_id++, (packed.value == 0x0000'0055'4433'2211ULL) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Pack channels in reverse order
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<4>(packed, 0xAA);
        Packing::pack_channel<3>(packed, 0xBB);
        Packing::pack_channel<2>(packed, 0xCC);
        Packing::pack_channel<1>(packed, 0xDD);
        Packing::pack_channel<0>(packed, 0xEE);
        write_test_result(test_id++, (packed.value == 0x0000'00AA'BBCC'DDEEULL) ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Pack into existing with some channels already set
    {
        auto packed = Packing::PackedValueType{0x0000'0000'0000'00FFULL};  // ch0 = 0xFF
        Packing::pack_channel<4>(packed, 0x99);  // Add ch4
        write_test_result(test_id++, (packed.value == 0x0000'0099'0000'00FFULL) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Comprehensive extract test for all channel counts
void test_credit_packing_extract_all_channel_counts(uint32_t& test_id) {
    // 1 channel
    {
        using Packing = CreditPacking<1, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 100);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        write_test_result(test_id++, (ch0 == 100) ? TEST_PASS : TEST_FAIL);
    }

    // 2 channels
    {
        using Packing = CreditPacking<2, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 10);
        Packing::pack_channel<1>(packed, 20);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        write_test_result(test_id++, (ch0 == 10 && ch1 == 20) ? TEST_PASS : TEST_FAIL);
    }

    // 3 channels
    {
        using Packing = CreditPacking<3, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 5);
        Packing::pack_channel<1>(packed, 10);
        Packing::pack_channel<2>(packed, 15);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        uint32_t ch2 = Packing::extract_channel<2>(packed);
        write_test_result(test_id++, (ch0 == 5 && ch1 == 10 && ch2 == 15) ? TEST_PASS : TEST_FAIL);
    }

    // 4 channels
    {
        using Packing = CreditPacking<4, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 1);
        Packing::pack_channel<1>(packed, 2);
        Packing::pack_channel<2>(packed, 3);
        Packing::pack_channel<3>(packed, 4);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        uint32_t ch2 = Packing::extract_channel<2>(packed);
        uint32_t ch3 = Packing::extract_channel<3>(packed);
        write_test_result(test_id++, (ch0 == 1 && ch1 == 2 && ch2 == 3 && ch3 == 4) ? TEST_PASS : TEST_FAIL);
    }

    // 5 channels
    {
        using Packing = CreditPacking<5, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 11);
        Packing::pack_channel<1>(packed, 22);
        Packing::pack_channel<2>(packed, 33);
        Packing::pack_channel<3>(packed, 44);
        Packing::pack_channel<4>(packed, 55);
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        uint32_t ch2 = Packing::extract_channel<2>(packed);
        uint32_t ch3 = Packing::extract_channel<3>(packed);
        uint32_t ch4 = Packing::extract_channel<4>(packed);
        write_test_result(test_id++, (ch0 == 11 && ch1 == 22 && ch2 == 33 && ch3 == 44 && ch4 == 55) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Comprehensive sum test for all channel counts
void test_credit_packing_sum_all_channel_counts(uint32_t& test_id) {
    // 1 channel
    {
        using Packing = CreditPacking<1, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 77);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 77) ? TEST_PASS : TEST_FAIL);
    }

    // 2 channels
    {
        using Packing = CreditPacking<2, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 30);
        Packing::pack_channel<1>(packed, 20);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 50) ? TEST_PASS : TEST_FAIL);
    }

    // 3 channels
    {
        using Packing = CreditPacking<3, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 10);
        Packing::pack_channel<1>(packed, 20);
        Packing::pack_channel<2>(packed, 30);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 60) ? TEST_PASS : TEST_FAIL);
    }

    // 4 channels
    {
        using Packing = CreditPacking<4, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 5);
        Packing::pack_channel<1>(packed, 10);
        Packing::pack_channel<2>(packed, 15);
        Packing::pack_channel<3>(packed, 20);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 50) ? TEST_PASS : TEST_FAIL);
    }

    // 5 channels
    {
        using Packing = CreditPacking<5, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 1);
        Packing::pack_channel<1>(packed, 2);
        Packing::pack_channel<2>(packed, 3);
        Packing::pack_channel<3>(packed, 4);
        Packing::pack_channel<4>(packed, 5);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 15) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: pack_channel with non-byte-aligned storage
void test_credit_packing_6bit_pack_into_existing(uint32_t& test_id) {
    using Packing = CreditPacking<4, 6>;

    // Test 1: Pack all channels with 6-bit values
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x3F);  // Max 6-bit (63)
        Packing::pack_channel<1>(packed, 0x2A);  // 42
        Packing::pack_channel<2>(packed, 0x15);  // 21
        Packing::pack_channel<3>(packed, 0x08);  // 8

        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        uint32_t ch2 = Packing::extract_channel<2>(packed);
        uint32_t ch3 = Packing::extract_channel<3>(packed);

        bool pass = (ch0 == 0x3F) && (ch1 == 0x2A) && (ch2 == 0x15) && (ch3 == 0x08);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Verify sum works correctly with pack_into
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 10);
        Packing::pack_channel<1>(packed, 20);
        Packing::pack_channel<2>(packed, 30);
        Packing::pack_channel<3>(packed, 3);
        uint32_t sum = Packing::sum_all_channels(packed);
        write_test_result(test_id++, (sum == 63) ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Wraparound behavior for 8-bit credits (all channel counts)
// Tests that unbounded credits correctly handle overflow and distance calculation
void test_credit_wraparound_8bit_all_channels(uint32_t& test_id) {
    // Test 1: 1-channel wraparound (8-bit, max=255)
    {
        using Packing = CreditPacking<1, 8>;
        auto packed = Packing::PackedValueType{0};

        // Start at 250, increment by 10 to wrap to 4
        uint8_t old_value = 250;
        uint8_t new_value = static_cast<uint8_t>(old_value + 10); // Wraps to 4

        Packing::pack_channel<0>(packed, new_value);
        uint32_t extracted = Packing::extract_channel<0>(packed);

        // Verify wraparound: 250 + 10 = 260 % 256 = 4
        bool wraparound_correct = (extracted == 4);

        // Verify distance calculation: (4 - 250) as uint8_t = 10
        uint8_t distance = static_cast<uint8_t>(new_value - old_value);
        bool distance_correct = (distance == 10);

        write_test_result(test_id++, (wraparound_correct && distance_correct) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: 2-channel wraparound
    {
        using Packing = CreditPacking<2, 8>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_ch0 = 252;
        uint8_t new_ch0 = static_cast<uint8_t>(old_ch0 + 15); // Wraps to 11
        uint8_t old_ch1 = 248;
        uint8_t new_ch1 = static_cast<uint8_t>(old_ch1 + 20); // Wraps to 12

        Packing::pack_channel<0>(packed, new_ch0);
        Packing::pack_channel<1>(packed, new_ch1);

        uint32_t ext_ch0 = Packing::extract_channel<0>(packed);
        uint32_t ext_ch1 = Packing::extract_channel<1>(packed);

        uint8_t dist_ch0 = static_cast<uint8_t>(new_ch0 - old_ch0);
        uint8_t dist_ch1 = static_cast<uint8_t>(new_ch1 - old_ch1);

        bool test_pass = (ext_ch0 == 11) && (ext_ch1 == 12) &&
                         (dist_ch0 == 15) && (dist_ch1 == 20);
        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: 3-channel wraparound
    {
        using Packing = CreditPacking<3, 8>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_ch0 = 250;
        uint8_t new_ch0 = static_cast<uint8_t>(old_ch0 + 30); // Wraps to 24
        uint8_t old_ch1 = 240;
        uint8_t new_ch1 = static_cast<uint8_t>(old_ch1 + 25); // Wraps to 9
        uint8_t old_ch2 = 255;
        uint8_t new_ch2 = static_cast<uint8_t>(old_ch2 + 1); // Wraps to 0

        Packing::pack_channel<0>(packed, new_ch0);
        Packing::pack_channel<1>(packed, new_ch1);
        Packing::pack_channel<2>(packed, new_ch2);

        uint32_t ext_ch0 = Packing::extract_channel<0>(packed);
        uint32_t ext_ch1 = Packing::extract_channel<1>(packed);
        uint32_t ext_ch2 = Packing::extract_channel<2>(packed);

        uint8_t dist_ch0 = static_cast<uint8_t>(new_ch0 - old_ch0);
        uint8_t dist_ch1 = static_cast<uint8_t>(new_ch1 - old_ch1);
        uint8_t dist_ch2 = static_cast<uint8_t>(new_ch2 - old_ch2);

        bool test_pass = (ext_ch0 == 24) && (ext_ch1 == 9) && (ext_ch2 == 0) &&
                         (dist_ch0 == 30) && (dist_ch1 == 25) && (dist_ch2 == 1);
        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 4: 4-channel wraparound
    {
        using Packing = CreditPacking<4, 8>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_vals[4] = {250, 245, 255, 200};
        uint8_t increments[4] = {20, 30, 5, 100};
        uint8_t new_vals[4];

        for (int i = 0; i < 4; i++) {
            new_vals[i] = static_cast<uint8_t>(old_vals[i] + increments[i]);
        }

        Packing::pack_channel<0>(packed, new_vals[0]);
        Packing::pack_channel<1>(packed, new_vals[1]);
        Packing::pack_channel<2>(packed, new_vals[2]);
        Packing::pack_channel<3>(packed, new_vals[3]);

        bool test_pass = true;
        for (int i = 0; i < 4; i++) {
            uint32_t extracted = Packing::extract_channel(packed, i);
            uint8_t expected = new_vals[i];
            uint8_t distance = static_cast<uint8_t>(new_vals[i] - old_vals[i]);

            if (extracted != expected || distance != increments[i]) {
                test_pass = false;
                break;
            }
        }

        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 5: 5-channel wraparound (uses uint64_t storage)
    {
        using Packing = CreditPacking<5, 8>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_vals[5] = {250, 245, 255, 200, 253};
        uint8_t increments[5] = {20, 30, 5, 100, 15};
        uint8_t new_vals[5];

        for (int i = 0; i < 5; i++) {
            new_vals[i] = static_cast<uint8_t>(old_vals[i] + increments[i]);
        }

        Packing::pack_channel<0>(packed, new_vals[0]);
        Packing::pack_channel<1>(packed, new_vals[1]);
        Packing::pack_channel<2>(packed, new_vals[2]);
        Packing::pack_channel<3>(packed, new_vals[3]);
        Packing::pack_channel<4>(packed, new_vals[4]);

        bool test_pass = true;
        for (int i = 0; i < 5; i++) {
            uint32_t extracted = Packing::extract_channel(packed, i);
            uint8_t expected = new_vals[i];
            uint8_t distance = static_cast<uint8_t>(new_vals[i] - old_vals[i]);

            if (extracted != expected || distance != increments[i]) {
                test_pass = false;
                break;
            }
        }

        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Wraparound behavior for 6-bit credits (all channel counts)
// 6-bit credits have max value of 63 (2^6 - 1)
void test_credit_wraparound_6bit_all_channels(uint32_t& test_id) {
    constexpr uint8_t MAX_6BIT = 63;
    constexpr uint8_t MASK_6BIT = 0x3F;

    // Test 1: 1-channel wraparound (6-bit, max=63)
    {
        using Packing = CreditPacking<1, 6>;
        auto packed = Packing::PackedValueType{0};

        // Start at 60, increment by 8 to wrap to 4 (68 % 64 = 4)
        uint8_t old_value = 60;
        uint8_t new_value = (old_value + 8) & MASK_6BIT;

        Packing::pack_channel<0>(packed, new_value);
        uint32_t extracted = Packing::extract_channel<0>(packed);

        // Verify wraparound: 60 + 8 = 68, masked to 6 bits = 4
        bool wraparound_correct = (extracted == 4);

        // Verify distance calculation with 6-bit mask
        uint8_t distance = static_cast<uint8_t>((new_value - old_value) & MASK_6BIT);
        bool distance_correct = (distance == 8);

        write_test_result(test_id++, (wraparound_correct && distance_correct) ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: 2-channel wraparound (6-bit)
    {
        using Packing = CreditPacking<2, 6>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_ch0 = 62;
        uint8_t new_ch0 = (old_ch0 + 5) & MASK_6BIT; // Wraps to 3
        uint8_t old_ch1 = 58;
        uint8_t new_ch1 = (old_ch1 + 10) & MASK_6BIT; // Wraps to 4

        Packing::pack_channel<0>(packed, new_ch0);
        Packing::pack_channel<1>(packed, new_ch1);

        uint32_t ext_ch0 = Packing::extract_channel<0>(packed);
        uint32_t ext_ch1 = Packing::extract_channel<1>(packed);

        uint8_t dist_ch0 = static_cast<uint8_t>((new_ch0 - old_ch0) & MASK_6BIT);
        uint8_t dist_ch1 = static_cast<uint8_t>((new_ch1 - old_ch1) & MASK_6BIT);

        bool test_pass = (ext_ch0 == 3) && (ext_ch1 == 4) &&
                         (dist_ch0 == 5) && (dist_ch1 == 10);
        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: 3-channel wraparound (6-bit)
    {
        using Packing = CreditPacking<3, 6>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_ch0 = 60;
        uint8_t new_ch0 = (old_ch0 + 20) & MASK_6BIT; // Wraps to 16
        uint8_t old_ch1 = 55;
        uint8_t new_ch1 = (old_ch1 + 15) & MASK_6BIT; // Wraps to 6
        uint8_t old_ch2 = MAX_6BIT;
        uint8_t new_ch2 = (old_ch2 + 1) & MASK_6BIT; // Wraps to 0

        Packing::pack_channel<0>(packed, new_ch0);
        Packing::pack_channel<1>(packed, new_ch1);
        Packing::pack_channel<2>(packed, new_ch2);

        uint32_t ext_ch0 = Packing::extract_channel<0>(packed);
        uint32_t ext_ch1 = Packing::extract_channel<1>(packed);
        uint32_t ext_ch2 = Packing::extract_channel<2>(packed);

        uint8_t dist_ch0 = static_cast<uint8_t>((new_ch0 - old_ch0) & MASK_6BIT);
        uint8_t dist_ch1 = static_cast<uint8_t>((new_ch1 - old_ch1) & MASK_6BIT);
        uint8_t dist_ch2 = static_cast<uint8_t>((new_ch2 - old_ch2) & MASK_6BIT);

        bool test_pass = (ext_ch0 == 16) && (ext_ch1 == 6) && (ext_ch2 == 0) &&
                         (dist_ch0 == 20) && (dist_ch1 == 15) && (dist_ch2 == 1);
        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 4: 4-channel wraparound (6-bit)
    {
        using Packing = CreditPacking<4, 6>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_vals[4] = {60, 58, MAX_6BIT, 50};
        uint8_t increments[4] = {10, 12, 5, 30};
        uint8_t new_vals[4];

        for (int i = 0; i < 4; i++) {
            new_vals[i] = (old_vals[i] + increments[i]) & MASK_6BIT;
        }

        Packing::pack_channel<0>(packed, new_vals[0]);
        Packing::pack_channel<1>(packed, new_vals[1]);
        Packing::pack_channel<2>(packed, new_vals[2]);
        Packing::pack_channel<3>(packed, new_vals[3]);

        bool test_pass = true;
        for (int i = 0; i < 4; i++) {
            uint32_t extracted = Packing::extract_channel(packed, i);
            uint8_t expected = new_vals[i];
            uint8_t distance = static_cast<uint8_t>((new_vals[i] - old_vals[i]) & MASK_6BIT);

            if (extracted != expected || distance != increments[i]) {
                test_pass = false;
                break;
            }
        }

        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 5: 5-channel wraparound (6-bit, uses uint64_t storage)
    {
        using Packing = CreditPacking<5, 6>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_vals[5] = {60, 58, MAX_6BIT, 50, 61};
        uint8_t increments[5] = {10, 12, 5, 30, 8};
        uint8_t new_vals[5];

        for (int i = 0; i < 5; i++) {
            new_vals[i] = (old_vals[i] + increments[i]) & MASK_6BIT;
        }

        Packing::pack_channel<0>(packed, new_vals[0]);
        Packing::pack_channel<1>(packed, new_vals[1]);
        Packing::pack_channel<2>(packed, new_vals[2]);
        Packing::pack_channel<3>(packed, new_vals[3]);
        Packing::pack_channel<4>(packed, new_vals[4]);

        bool test_pass = true;
        for (int i = 0; i < 5; i++) {
            uint32_t extracted = Packing::extract_channel(packed, i);
            uint8_t expected = new_vals[i];
            uint8_t distance = static_cast<uint8_t>((new_vals[i] - old_vals[i]) & MASK_6BIT);

            if (extracted != expected || distance != increments[i]) {
                test_pass = false;
                break;
            }
        }

        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Pack channel replacement (NOT OR) - this exposes the bug!
// When updating the same channel twice, the new value should REPLACE, not OR
void test_pack_channel_replacement_not_or(uint32_t& test_id) {
    DPRINT << "=== test_pack_channel_replacement_not_or START ===" << ENDL();

    // Test 1: Single channel replacement (1-channel, 8-bit)
    {
        using Packing = CreditPacking<1, 8>;
        auto packed = Packing::PackedValueType{0};

        DPRINT << "Test1: Setting ch0=100" << ENDL();
        // Set channel 0 to 100
        Packing::pack_channel<0>(packed, 100);
        uint32_t val1 = Packing::extract_channel<0>(packed);
        DPRINT << "  After set: val1=" << val1 << " packed.value=0x" << HEX() << packed.value << DEC() << ENDL();

        // Update channel 0 to 50 (should REPLACE, not OR)
        DPRINT << "Test1: Updating ch0=50 (REPLACE)" << ENDL();
        Packing::pack_channel<0>(packed, 50);
        uint32_t val2 = Packing::extract_channel<0>(packed);
        DPRINT << "  After update: val2=" << val2 << " packed.value=0x" << HEX() << packed.value << DEC() << ENDL();

        // If using OR: 100 | 50 = 118 (WRONG)
        // If using REPLACE: 50 (CORRECT)
        bool pass = (val1 == 100 && val2 == 50);
        DPRINT << "Test1: " << (pass ? "PASS" : "FAIL") << " (expected val1=100, val2=50)" << ENDL();
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Multi-channel, update same channel multiple times (2-channel, 8-bit)
    {
        using Packing = CreditPacking<2, 8>;
        auto packed = Packing::PackedValueType{0};

        DPRINT << "Test2: Initial setup ch0=100, ch1=200" << ENDL();
        // Initial setup: ch0=100, ch1=200
        Packing::pack_channel<0>(packed, 100);
        Packing::pack_channel<1>(packed, 200);
        DPRINT << "  After setup: packed.value=0x" << HEX() << packed.value << DEC() << ENDL();

        // Update ch0 to 50 (should REPLACE 100 with 50, leave ch1 unchanged)
        DPRINT << "Test2: Updating ch0=50 (should REPLACE, keep ch1)" << ENDL();
        Packing::pack_channel<0>(packed, 50);
        DPRINT << "  After update: packed.value=0x" << HEX() << packed.value << DEC() << ENDL();

        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        DPRINT << "  Extracted: ch0=" << ch0 << " ch1=" << ch1 << ENDL();

        // ch0 should be 50 (not 100 | 50 = 116)
        // ch1 should still be 200
        bool pass = (ch0 == 50 && ch1 == 200);
        DPRINT << "Test2: " << (pass ? "PASS" : "FAIL") << " (expected ch0=50, ch1=200)" << ENDL();
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Update with zero (should clear the channel)
    {
        using Packing = CreditPacking<2, 8>;
        auto packed = Packing::PackedValueType{0};

        Packing::pack_channel<0>(packed, 255);
        Packing::pack_channel<1>(packed, 128);

        // Update ch0 to 0 (should clear it completely)
        Packing::pack_channel<0>(packed, 0);

        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);

        // ch0 should be 0 (not 255 | 0 = 255)
        write_test_result(test_id++, (ch0 == 0 && ch1 == 128) ? TEST_PASS : TEST_FAIL);
    }

    // Test 4: Multiple updates in sequence (simulating unbounded counter updates)
    {
        using Packing = CreditPacking<1, 8>;
        auto packed = Packing::PackedValueType{0};

        // Simulate unbounded counter: 0 -> 10 -> 20 -> 30
        Packing::pack_channel<0>(packed, 0);
        Packing::pack_channel<0>(packed, 10);
        uint32_t val1 = Packing::extract_channel<0>(packed);

        Packing::pack_channel<0>(packed, 20);
        uint32_t val2 = Packing::extract_channel<0>(packed);

        Packing::pack_channel<0>(packed, 30);
        uint32_t val3 = Packing::extract_channel<0>(packed);

        write_test_result(test_id++, (val1 == 10 && val2 == 20 && val3 == 30) ? TEST_PASS : TEST_FAIL);
    }

    // Test 5: 4-channel, update middle channel (non-byte-aligned 6-bit)
    {
        using Packing = CreditPacking<4, 6>;
        auto packed = Packing::PackedValueType{0};

        // Set all channels
        Packing::pack_channel<0>(packed, 10);
        Packing::pack_channel<1>(packed, 20);
        Packing::pack_channel<2>(packed, 30);
        Packing::pack_channel<3>(packed, 40);

        // Update ch1 to 5 (should replace 20 with 5)
        Packing::pack_channel<1>(packed, 5);

        uint32_t ch0 = Packing::extract_channel<0>(packed);
        uint32_t ch1 = Packing::extract_channel<1>(packed);
        uint32_t ch2 = Packing::extract_channel<2>(packed);
        uint32_t ch3 = Packing::extract_channel<3>(packed);

        write_test_result(test_id++, (ch0 == 10 && ch1 == 5 && ch2 == 30 && ch3 == 40) ? TEST_PASS : TEST_FAIL);
    }

    // Test 6: Runtime channel_id version
    {
        using Packing = CreditPacking<3, 8>;
        auto packed = Packing::PackedValueType{0};

        // Set initial values using runtime channel_id
        Packing::pack_channel(packed, 0, 100);
        Packing::pack_channel(packed, 1, 150);
        Packing::pack_channel(packed, 2, 200);

        // Update channel 1 to 75
        Packing::pack_channel(packed, 1, 75);

        uint32_t ch0 = Packing::extract_channel(packed, 0);
        uint32_t ch1 = Packing::extract_channel(packed, 1);
        uint32_t ch2 = Packing::extract_channel(packed, 2);

        write_test_result(test_id++, (ch0 == 100 && ch1 == 75 && ch2 == 200) ? TEST_PASS : TEST_FAIL);
    }

    // Test 7: Edge case - update to max value
    {
        using Packing = CreditPacking<2, 8>;
        auto packed = Packing::PackedValueType{0};

        Packing::pack_channel<0>(packed, 127);
        Packing::pack_channel<0>(packed, 255);  // Update to max

        uint32_t ch0 = Packing::extract_channel<0>(packed);
        write_test_result(test_id++, (ch0 == 255) ? TEST_PASS : TEST_FAIL);
    }

    // Test 8: 5-channel byte-aligned (uses uint64_t)
    {
        using Packing = CreditPacking<5, 8>;
        auto packed = Packing::PackedValueType{0};

        // Set all channels
        for (uint8_t i = 0; i < 5; i++) {
            Packing::pack_channel(packed, i, 50 + i * 10);
        }

        // Update channel 2 from 70 to 15
        Packing::pack_channel<2>(packed, 15);

        bool all_correct = true;
        uint32_t expected[5] = {50, 60, 15, 80, 90};
        for (uint8_t i = 0; i < 5; i++) {
            uint32_t val = Packing::extract_channel(packed, i);
            if (val != expected[i]) {
                all_correct = false;
                break;
            }
        }

        write_test_result(test_id++, all_correct ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Edge case wraparound scenarios
void test_credit_wraparound_edge_cases(uint32_t& test_id) {
    // Test 1: 8-bit wraparound from 255 to 0
    {
        using Packing = CreditPacking<1, 8>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_value = 255;
        uint8_t new_value = static_cast<uint8_t>(old_value + 1); // Wraps to 0

        Packing::pack_channel<0>(packed, new_value);
        uint32_t extracted = Packing::extract_channel<0>(packed);
        uint8_t distance = static_cast<uint8_t>(new_value - old_value);

        bool test_pass = (extracted == 0) && (distance == 1);
        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: 8-bit multiple full wraparounds (increment by 256)
    {
        using Packing = CreditPacking<1, 8>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_value = 100;
        uint8_t new_value = static_cast<uint8_t>(old_value + 256); // Should equal old_value

        Packing::pack_channel<0>(packed, new_value);
        uint32_t extracted = Packing::extract_channel<0>(packed);
        uint8_t distance = static_cast<uint8_t>(new_value - old_value);

        bool test_pass = (extracted == 100) && (distance == 0);
        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: 6-bit wraparound from 63 to 0
    {
        using Packing = CreditPacking<1, 6>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_value = 63;
        uint8_t new_value = (old_value + 1) & 0x3F; // Wraps to 0

        Packing::pack_channel<0>(packed, new_value);
        uint32_t extracted = Packing::extract_channel<0>(packed);
        uint8_t distance = static_cast<uint8_t>((new_value - old_value) & 0x3F);

        bool test_pass = (extracted == 0) && (distance == 1);
        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 4: 6-bit multiple full wraparounds (increment by 64)
    {
        using Packing = CreditPacking<1, 6>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_value = 30;
        uint8_t new_value = (old_value + 64) & 0x3F; // Should equal old_value

        Packing::pack_channel<0>(packed, new_value);
        uint32_t extracted = Packing::extract_channel<0>(packed);
        uint8_t distance = static_cast<uint8_t>((new_value - old_value) & 0x3F);

        bool test_pass = (extracted == 30) && (distance == 0);
        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 5: Large increment with 8-bit (simulate many wraps)
    {
        using Packing = CreditPacking<2, 8>;
        auto packed = Packing::PackedValueType{0};

        uint8_t old_ch0 = 100;
        uint8_t old_ch1 = 200;
        uint8_t new_ch0 = static_cast<uint8_t>(old_ch0 + 500); // Multiple wraps
        uint8_t new_ch1 = static_cast<uint8_t>(old_ch1 + 1000); // Multiple wraps

        Packing::pack_channel<0>(packed, new_ch0);
        Packing::pack_channel<1>(packed, new_ch1);

        uint32_t ext_ch0 = Packing::extract_channel<0>(packed);
        uint32_t ext_ch1 = Packing::extract_channel<1>(packed);

        uint8_t dist_ch0 = static_cast<uint8_t>(new_ch0 - old_ch0);
        uint8_t dist_ch1 = static_cast<uint8_t>(new_ch1 - old_ch1);

        // 500 % 256 = 244, so new_ch0 = (100 + 244) % 256 = 88
        // 1000 % 256 = 232, so new_ch1 = (200 + 232) % 256 = 176
        bool test_pass = (ext_ch0 == 88) && (ext_ch1 == 176) &&
                         (dist_ch0 == 244) && (dist_ch1 == 232);
        write_test_result(test_id++, test_pass ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Safe arithmetic - add_to_channel (prevents carries)
void test_safe_arithmetic_add_to_channel(uint32_t& test_id) {
    // Test 1: Simple addition - no wraparound
    {
        using Packing = CreditPacking<4, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 10);
        Packing::pack_channel<1>(packed, 20);
        Packing::pack_channel<2>(packed, 30);
        Packing::pack_channel<3>(packed, 40);

        // Add 5 to channel 1
        auto result = Packing::template add_to_channel<1>(packed, 5);

        // Verify: ch1 should be 25, others unchanged
        bool pass = (Packing::extract_channel<0>(result) == 10) &&
                    (Packing::extract_channel<1>(result) == 25) &&
                    (Packing::extract_channel<2>(result) == 30) &&
                    (Packing::extract_channel<3>(result) == 40);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Addition with wraparound - NO CARRY to adjacent channel!
    {
        using Packing = CreditPacking<4, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 100);
        Packing::pack_channel<1>(packed, 255);  // Will wrap on add
        Packing::pack_channel<2>(packed, 50);
        Packing::pack_channel<3>(packed, 75);

        // Add 10 to channel 1 (255 + 10 = 265 → wraps to 9)
        auto result = Packing::template add_to_channel<1>(packed, 10);

        // CRITICAL: Channel 2 must remain 50 (no carry corruption!)
        bool pass = (Packing::extract_channel<0>(result) == 100) &&
                    (Packing::extract_channel<1>(result) == 9) &&    // 255 + 10 wraps to 9
                    (Packing::extract_channel<2>(result) == 50) &&   // UNCHANGED!
                    (Packing::extract_channel<3>(result) == 75);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Multiple channels, all with potential carries
    {
        using Packing = CreditPacking<3, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 250);
        Packing::pack_channel<1>(packed, 240);
        Packing::pack_channel<2>(packed, 230);

        // Add to ch0 - would carry if unsafe
        auto result = Packing::template add_to_channel<0>(packed, 20);

        // Verify: only ch0 changes, no corruption of ch1 or ch2
        bool pass = (Packing::extract_channel<0>(result) == 14) &&  // 250 + 20 = 270 → 14
                    (Packing::extract_channel<1>(result) == 240) && // UNCHANGED
                    (Packing::extract_channel<2>(result) == 230);   // UNCHANGED
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Safe arithmetic - subtract_from_channel (prevents borrows)
void test_safe_arithmetic_subtract_from_channel(uint32_t& test_id) {
    // Test 1: Simple subtraction - no wraparound
    {
        using Packing = CreditPacking<4, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 100);
        Packing::pack_channel<1>(packed, 50);
        Packing::pack_channel<2>(packed, 75);
        Packing::pack_channel<3>(packed, 25);

        // Subtract 10 from channel 2
        auto result = Packing::template subtract_from_channel<2>(packed, 10);

        // Verify: ch2 should be 65, others unchanged
        bool pass = (Packing::extract_channel<0>(result) == 100) &&
                    (Packing::extract_channel<1>(result) == 50) &&
                    (Packing::extract_channel<2>(result) == 65) &&
                    (Packing::extract_channel<3>(result) == 25);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Subtraction with wraparound - NO BORROW from adjacent channel!
    {
        using Packing = CreditPacking<4, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 200);
        Packing::pack_channel<1>(packed, 5);    // Will wrap on subtract
        Packing::pack_channel<2>(packed, 100);
        Packing::pack_channel<3>(packed, 150);

        // Subtract 10 from channel 1 (5 - 10 = -5 → wraps to 251)
        auto result = Packing::template subtract_from_channel<1>(packed, 10);

        // CRITICAL: Channels 0 and 2 must remain unchanged (no borrow corruption!)
        bool pass = (Packing::extract_channel<0>(result) == 200) &&  // UNCHANGED!
                    (Packing::extract_channel<1>(result) == 251) &&  // 5 - 10 wraps to 251
                    (Packing::extract_channel<2>(result) == 100) &&  // UNCHANGED!
                    (Packing::extract_channel<3>(result) == 150);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Edge case - subtract from zero
    {
        using Packing = CreditPacking<2, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0);
        Packing::pack_channel<1>(packed, 100);

        // Subtract 1 from ch0 (0 - 1 = 255)
        auto result = Packing::template subtract_from_channel<0>(packed, 1);

        bool pass = (Packing::extract_channel<0>(result) == 255) &&
                    (Packing::extract_channel<1>(result) == 100);  // UNCHANGED!
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Safe arithmetic - diff_channels
void test_safe_arithmetic_diff_channels(uint32_t& test_id) {
    // Test 1: Simple difference
    {
        using Packing = CreditPacking<3, 8>;
        auto a = Packing::PackedValueType{0};
        auto b = Packing::PackedValueType{0};

        Packing::pack_channel<0>(a, 100);
        Packing::pack_channel<1>(a, 200);
        Packing::pack_channel<2>(a, 150);

        Packing::pack_channel<0>(b, 90);
        Packing::pack_channel<1>(b, 180);
        Packing::pack_channel<2>(b, 140);

        // Compute differences
        uint8_t diff0 = Packing::template diff_channels<0>(a, b);
        uint8_t diff1 = Packing::template diff_channels<1>(a, b);
        uint8_t diff2 = Packing::template diff_channels<2>(a, b);

        bool pass = (diff0 == 10) && (diff1 == 20) && (diff2 == 10);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Difference with wraparound (new < old)
    {
        using Packing = CreditPacking<2, 8>;
        auto new_val = Packing::PackedValueType{0};
        auto old_val = Packing::PackedValueType{0};

        Packing::pack_channel<0>(new_val, 10);   // Wrapped counter
        Packing::pack_channel<0>(old_val, 250);  // Old value before wrap

        // Difference: 10 - 250 = -240, as uint8_t = 16
        uint8_t diff = Packing::template diff_channels<0>(new_val, old_val);

        // This represents 16 acks: 250 → 255 (6 acks) → 0 → 10 (10 acks) = 16 total
        bool pass = (diff == 16);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: Zero difference
    {
        using Packing = CreditPacking<4, 8>;
        auto a = Packing::PackedValueType{0};
        auto b = Packing::PackedValueType{0};

        Packing::pack_channel<2>(a, 123);
        Packing::pack_channel<2>(b, 123);

        uint8_t diff = Packing::template diff_channels<2>(a, b);
        bool pass = (diff == 0);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }
}

// Test: Safe arithmetic - add_packed (multi-channel addition, NO carries between channels)
void test_safe_arithmetic_add_packed(uint32_t& test_id) {
    // Test 1: Simple multi-channel addition (2 channels)
    {
        using Packing = CreditPacking<2, 8>;
        auto a = Packing::PackedValueType{0};
        auto b = Packing::PackedValueType{0};

        Packing::pack_channel<0>(a, 10);
        Packing::pack_channel<1>(a, 20);

        Packing::pack_channel<0>(b, 5);
        Packing::pack_channel<1>(b, 8);

        auto result = Packing::add_packed(a, b);

        bool pass = (Packing::extract_channel<0>(result) == 15) &&
                    (Packing::extract_channel<1>(result) == 28);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 2: Addition with wraparound in one channel - NO CARRY to adjacent!
    {
        using Packing = CreditPacking<4, 8>;
        auto a = Packing::PackedValueType{0};
        auto b = Packing::PackedValueType{0};

        Packing::pack_channel<0>(a, 100);
        Packing::pack_channel<1>(a, 250);  // Will wrap
        Packing::pack_channel<2>(a, 50);
        Packing::pack_channel<3>(a, 75);

        Packing::pack_channel<0>(b, 10);
        Packing::pack_channel<1>(b, 20);   // 250 + 20 = 270 → wraps to 14
        Packing::pack_channel<2>(b, 5);
        Packing::pack_channel<3>(b, 15);

        auto result = Packing::add_packed(a, b);

        // CRITICAL: ch2 must be 55, NOT 55 + carry!
        bool pass = (Packing::extract_channel<0>(result) == 110) &&
                    (Packing::extract_channel<1>(result) == 14) &&   // Wrapped
                    (Packing::extract_channel<2>(result) == 55) &&   // NO CARRY!
                    (Packing::extract_channel<3>(result) == 90);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 3: All channels wrap simultaneously
    {
        using Packing = CreditPacking<3, 8>;
        auto a = Packing::PackedValueType{0};
        auto b = Packing::PackedValueType{0};

        Packing::pack_channel<0>(a, 255);
        Packing::pack_channel<1>(a, 254);
        Packing::pack_channel<2>(a, 253);

        Packing::pack_channel<0>(b, 2);
        Packing::pack_channel<1>(b, 3);
        Packing::pack_channel<2>(b, 4);

        auto result = Packing::add_packed(a, b);

        // All channels wrap independently
        bool pass = (Packing::extract_channel<0>(result) == 1) &&   // 255 + 2 = 1
                    (Packing::extract_channel<1>(result) == 1) &&   // 254 + 3 = 1
                    (Packing::extract_channel<2>(result) == 1);     // 253 + 4 = 1
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 4: 1-channel (trivial case)
    {
        using Packing = CreditPacking<1, 8>;
        auto a = Packing::PackedValueType{0};
        auto b = Packing::PackedValueType{0};

        Packing::pack_channel<0>(a, 100);
        Packing::pack_channel<0>(b, 50);

        auto result = Packing::add_packed(a, b);

        bool pass = (Packing::extract_channel<0>(result) == 150);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 5: 5-channel (uint64_t storage)
    {
        using Packing = CreditPacking<5, 8>;
        auto a = Packing::PackedValueType{0};
        auto b = Packing::PackedValueType{0};

        Packing::pack_channel<0>(a, 10);
        Packing::pack_channel<1>(a, 20);
        Packing::pack_channel<2>(a, 30);
        Packing::pack_channel<3>(a, 40);
        Packing::pack_channel<4>(a, 50);

        Packing::pack_channel<0>(b, 5);
        Packing::pack_channel<1>(b, 10);
        Packing::pack_channel<2>(b, 15);
        Packing::pack_channel<3>(b, 20);
        Packing::pack_channel<4>(b, 25);

        auto result = Packing::add_packed(a, b);

        bool pass = (Packing::extract_channel<0>(result) == 15) &&
                    (Packing::extract_channel<1>(result) == 30) &&
                    (Packing::extract_channel<2>(result) == 45) &&
                    (Packing::extract_channel<3>(result) == 60) &&
                    (Packing::extract_channel<4>(result) == 75);
        write_test_result(test_id++, pass ? TEST_PASS : TEST_FAIL);
    }

    // Test 6: Exhaustive byte isolation - channel 1 wraps, verify NO spillover
    {
        using Packing = CreditPacking<4, 8>;

        // Test multiple starting values for channel 1
        for (uint8_t start = 240; start <= 255; start++) {
            auto a = Packing::PackedValueType{0};
            auto b = Packing::PackedValueType{0};

            // Set all channels to distinct values
            Packing::pack_channel<0>(a, 100);
            Packing::pack_channel<1>(a, start);  // Channel that will wrap
            Packing::pack_channel<2>(a, 150);
            Packing::pack_channel<3>(a, 200);

            // Add enough to cause wraparound in ch1
            Packing::pack_channel<0>(b, 0);
            Packing::pack_channel<1>(b, 20);     // start + 20 will wrap
            Packing::pack_channel<2>(b, 0);
            Packing::pack_channel<3>(b, 0);

            auto result = Packing::add_packed(a, b);

            // Verify OTHER channels unchanged
            bool channels_isolated =
                (Packing::extract_channel<0>(result) == 100) &&  // UNCHANGED
                (Packing::extract_channel<2>(result) == 150) &&  // UNCHANGED
                (Packing::extract_channel<3>(result) == 200);    // UNCHANGED

            if (!channels_isolated) {
                write_test_result(test_id++, TEST_FAIL);
                return;  // Early exit on first failure
            }
        }
        // All iterations passed - no spillover detected
        write_test_result(test_id++, TEST_PASS);
    }
}

// Test: diff_channels returns value that works with caller's masking pattern
// Reproduces bug where unpacked return value gets zeroed by caller's mask
void test_diff_channels_packed_format_compatibility(uint32_t& test_id) {
    // This test simulates the interaction pattern in fabric_erisc_router.cpp:1785-1790
    // where get_num_unprocessed_acks_from_receiver returns a diff and the caller
    // masks it with (0xFF << (channel_index * 8))

    using Packing = CreditPacking<5, 8>;  // 5 channels, 8-bit credits

    // Test channel 0
    {
        auto received = Packing::PackedValueType{0};
        auto processed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(received, 10);
        Packing::pack_channel<0>(processed, 3);
        uint8_t diff_unpacked = Packing::template diff_channels<0>(received, processed);
        uint32_t diff_packed = static_cast<uint32_t>(diff_unpacked) << (0 * 8);
        uint32_t masked = diff_packed & (0xFF << (0 * 8));
        bool ok = (masked != 0) && ((masked >> 0) == diff_unpacked);
        write_test_result(test_id++, ok ? TEST_PASS : TEST_FAIL);
    }

    // Test channel 1 (this is where the bug manifests!)
    {
        auto received = Packing::PackedValueType{0};
        auto processed = Packing::PackedValueType{0};
        Packing::pack_channel<1>(received, 10);
        Packing::pack_channel<1>(processed, 3);
        uint8_t diff_unpacked = Packing::template diff_channels<1>(received, processed);
        uint32_t diff_packed = static_cast<uint32_t>(diff_unpacked) << (1 * 8);

        // Simulate caller's masking
        uint32_t byte_mask = 0xFF << (1 * 8);
        uint32_t masked = diff_packed & byte_mask;

        // BUG: If we returned unpacked (diff_unpacked), then:
        // masked = 7 & 0xFF00 = 0 (FAIL!)
        // With fix: masked = 0x0700 & 0xFF00 = 0x0700 (PASS!)
        bool mask_preserves_value = (masked != 0);
        uint8_t extracted = masked >> 8;
        bool value_correct = (extracted == diff_unpacked);

        write_test_result(test_id++, (mask_preserves_value && value_correct) ? TEST_PASS : TEST_FAIL);
    }

    // Test channel 2
    {
        auto received = Packing::PackedValueType{0};
        auto processed = Packing::PackedValueType{0};
        Packing::pack_channel<2>(received, 10);
        Packing::pack_channel<2>(processed, 3);
        uint8_t diff_unpacked = Packing::template diff_channels<2>(received, processed);
        uint32_t diff_packed = static_cast<uint32_t>(diff_unpacked) << (2 * 8);
        uint32_t masked = diff_packed & (0xFF << (2 * 8));
        bool ok = (masked != 0) && ((masked >> 16) == diff_unpacked);
        write_test_result(test_id++, ok ? TEST_PASS : TEST_FAIL);
    }

    // Test channel 3
    {
        auto received = Packing::PackedValueType{0};
        auto processed = Packing::PackedValueType{0};
        Packing::pack_channel<3>(received, 10);
        Packing::pack_channel<3>(processed, 3);
        uint8_t diff_unpacked = Packing::template diff_channels<3>(received, processed);
        uint32_t diff_packed = static_cast<uint32_t>(diff_unpacked) << (3 * 8);
        uint32_t masked = diff_packed & (0xFF << (3 * 8));
        bool ok = (masked != 0) && ((masked >> 24) == diff_unpacked);
        write_test_result(test_id++, ok ? TEST_PASS : TEST_FAIL);
    }

    // Test channel 4
    {
        auto received = Packing::PackedValueType{0};
        auto processed = Packing::PackedValueType{0};
        Packing::pack_channel<4>(received, 10);
        Packing::pack_channel<4>(processed, 3);
        uint8_t diff_unpacked = Packing::template diff_channels<4>(received, processed);
        uint32_t diff_packed = static_cast<uint32_t>(diff_unpacked) << (4 * 8);
        uint32_t masked = diff_packed & (0xFFULL << (4 * 8));  // Use ULL for shift beyond 32 bits
        bool ok = (masked != 0) && ((masked >> 32) == diff_unpacked);
        write_test_result(test_id++, ok ? TEST_PASS : TEST_FAIL);
    }
}

// Test: add_to_channel rejects packed values (expects unpacked uint8_t)
// Reproduces bug where caller passes packed value which gets truncated
void test_add_to_channel_packed_vs_unpacked(uint32_t& test_id) {
    // This test simulates the bug in fabric_erisc_router.cpp:1793-1798
    // where increment_num_processed_acks is called with a PACKED value
    // but add_to_channel expects an UNPACKED uint8_t delta

    using Packing = CreditPacking<5, 8>;

    // Test each channel (0-4) to show truncation behavior
    // Channel 0: Works by accident (packed value fits in uint8_t)
    {
        auto counter = Packing::PackedValueType{0};
        Packing::pack_channel<0>(counter, 100);  // Start at 100

        // Simulate caller passing packed value (0x0007) for channel 0
        uint32_t packed_value = 7 << (0 * 8);  // = 0x0007
        uint8_t truncated = static_cast<uint8_t>(packed_value);  // = 7 (OK by accident!)

        // add_to_channel expects unpacked value
        auto result = Packing::template add_to_channel<0>(counter, truncated);
        uint8_t final_val = Packing::extract_channel<0>(result);

        // Should be 100 + 7 = 107
        write_test_result(test_id++, (final_val == 107) ? TEST_PASS : TEST_FAIL);
    }

    // Channel 1: BUG EXPOSED - packed value gets truncated!
    {
        auto counter = Packing::PackedValueType{0};
        Packing::pack_channel<1>(counter, 100);  // Start at 100

        // Simulate BUGGY caller passing packed value (0x0700) for channel 1
        uint32_t packed_value = 7 << (1 * 8);  // = 0x0700
        uint8_t truncated = static_cast<uint8_t>(packed_value);  // = 0 (BUG!)

        // add_to_channel gets 0 instead of 7
        auto result_buggy = Packing::template add_to_channel<1>(counter, truncated);
        uint8_t buggy_val = Packing::extract_channel<1>(result_buggy);

        // WRONG: 100 + 0 = 100 (should be 107!)
        bool detects_bug = (buggy_val != 107);

        // Now test with CORRECT unpacked value
        uint8_t unpacked_value = (packed_value >> (1 * 8)) & 0xFF;  // = 7 (correct!)
        auto result_fixed = Packing::template add_to_channel<1>(counter, unpacked_value);
        uint8_t fixed_val = Packing::extract_channel<1>(result_fixed);

        // CORRECT: 100 + 7 = 107
        bool fixed_correct = (fixed_val == 107);

        write_test_result(test_id++, (detects_bug && fixed_correct) ? TEST_PASS : TEST_FAIL);
    }

    // Channel 2: Same truncation bug
    {
        auto counter = Packing::PackedValueType{0};
        Packing::pack_channel<2>(counter, 100);

        uint32_t packed_value = 7 << (2 * 8);  // = 0x070000
        uint8_t truncated = static_cast<uint8_t>(packed_value);  // = 0 (BUG!)
        uint8_t unpacked = (packed_value >> (2 * 8)) & 0xFF;  // = 7 (correct!)

        auto result_buggy = Packing::template add_to_channel<2>(counter, truncated);
        auto result_fixed = Packing::template add_to_channel<2>(counter, unpacked);

        bool buggy_wrong = (Packing::extract_channel<2>(result_buggy) != 107);
        bool fixed_correct = (Packing::extract_channel<2>(result_fixed) == 107);

        write_test_result(test_id++, (buggy_wrong && fixed_correct) ? TEST_PASS : TEST_FAIL);
    }

    // Channel 3: Same truncation bug
    {
        auto counter = Packing::PackedValueType{0};
        Packing::pack_channel<3>(counter, 100);

        uint32_t packed_value = 7 << (3 * 8);  // = 0x07000000
        uint8_t truncated = static_cast<uint8_t>(packed_value);  // = 0 (BUG!)
        uint8_t unpacked = (packed_value >> (3 * 8)) & 0xFF;  // = 7 (correct!)

        auto result_buggy = Packing::template add_to_channel<3>(counter, truncated);
        auto result_fixed = Packing::template add_to_channel<3>(counter, unpacked);

        bool buggy_wrong = (Packing::extract_channel<3>(result_buggy) != 107);
        bool fixed_correct = (Packing::extract_channel<3>(result_fixed) == 107);

        write_test_result(test_id++, (buggy_wrong && fixed_correct) ? TEST_PASS : TEST_FAIL);
    }

    // Channel 4: Truncation at 64-bit boundary
    {
        auto counter = Packing::PackedValueType{0};
        Packing::pack_channel<4>(counter, 100);

        uint64_t packed_value = 7ULL << (4 * 8);  // = 0x0700000000
        uint8_t truncated = static_cast<uint8_t>(packed_value);  // = 0 (BUG!)
        uint8_t unpacked = (packed_value >> (4 * 8)) & 0xFF;  // = 7 (correct!)

        auto result_buggy = Packing::template add_to_channel<4>(counter, truncated);
        auto result_fixed = Packing::template add_to_channel<4>(counter, unpacked);

        bool buggy_wrong = (Packing::extract_channel<4>(result_buggy) != 107);
        bool fixed_correct = (Packing::extract_channel<4>(result_fixed) == 107);

        write_test_result(test_id++, (buggy_wrong && fixed_correct) ? TEST_PASS : TEST_FAIL);
    }
}

void kernel_main() {
    uint32_t test_id = 0;

    // Comprehensive tests for all channel counts (1-5)
    test_credit_packing_1ch_byte_aligned_extract(test_id);
    test_credit_packing_1ch_byte_aligned_sum(test_id);
    test_credit_packing_extract_all_channel_counts(test_id);
    test_credit_packing_sum_all_channel_counts(test_id);

    // Individual channel count tests (legacy - kept for detailed coverage)
    test_credit_packing_2ch_byte_aligned_extract(test_id);
    test_credit_packing_2ch_byte_aligned_sum(test_id);
    test_credit_packing_3ch_byte_aligned_extract(test_id);
    test_credit_packing_3ch_byte_aligned_sum(test_id);
    test_credit_packing_4ch_byte_aligned_extract(test_id);
    test_credit_packing_4ch_byte_aligned_sum(test_id);
    test_credit_packing_pack_channel(test_id);
    test_credit_packing_4ch_6bit_non_byte_aligned(test_id);
    test_packed_credit_container(test_id);
    test_credit_packing_5ch_byte_aligned(test_id);
    test_credit_packing_5ch_pack(test_id);

    // New API tests: pack_channel with existing packed value (all channel counts)
    test_credit_packing_pack_into_existing_all_channel_counts(test_id);
    test_credit_packing_pack_into_existing_special_cases(test_id);
    test_credit_packing_5ch_pack_into_existing(test_id);
    test_credit_packing_6bit_pack_into_existing(test_id);

    // Pack channel replacement tests (exposes OR bug)
    test_pack_channel_replacement_not_or(test_id);

    // Wraparound behavior tests (unbounded counter credits)
    test_credit_wraparound_8bit_all_channels(test_id);
    test_credit_wraparound_6bit_all_channels(test_id);
    test_credit_wraparound_edge_cases(test_id);

    // NEW: Safe arithmetic tests (Phase 2 - prevent carry/borrow bugs)
    test_safe_arithmetic_add_to_channel(test_id);
    test_safe_arithmetic_subtract_from_channel(test_id);
    test_safe_arithmetic_diff_channels(test_id);
    test_safe_arithmetic_add_packed(test_id);

    // NEW: Test packed format compatibility with caller's masking pattern
    // Reproduces bug where get_num_unprocessed_acks_from_receiver returned
    // unpacked value that got zeroed by caller's mask for channels > 0
    test_diff_channels_packed_format_compatibility(test_id);

    // NEW: Test add_to_channel truncation bug
    // Reproduces bug where increment_num_processed_acks was called with
    // packed value that got truncated to 0 for channels > 0
    test_add_to_channel_packed_vs_unpacked(test_id);

    // Write completion status as the VERY LAST operation
    // This signals to the host that the kernel ran to completion
    write_test_completion_status(test_id);
}
