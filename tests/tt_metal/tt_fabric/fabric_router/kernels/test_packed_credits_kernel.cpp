// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
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

    // Test 2-channel pack_into
    {
        using Packing = CreditPacking<2, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x12);
        Packing::pack_channel<1>(packed, 0x34);
        write_test_result(test_id++, (packed.value == 0x3412) ? TEST_PASS : TEST_FAIL);
    }

    // Test 3-channel pack_into
    {
        using Packing = CreditPacking<3, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x11);
        Packing::pack_channel<1>(packed, 0x22);
        Packing::pack_channel<2>(packed, 0x33);
        write_test_result(test_id++, (packed.value == 0x3322'11) ? TEST_PASS : TEST_FAIL);
    }

    // Test 4-channel pack_into
    {
        using Packing = CreditPacking<4, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0xAA);
        Packing::pack_channel<1>(packed, 0xBB);
        Packing::pack_channel<2>(packed, 0xCC);
        Packing::pack_channel<3>(packed, 0xDD);
        write_test_result(test_id++, (packed.value == 0xDDCC'BBAA) ? TEST_PASS : TEST_FAIL);
    }

    // Test 5-channel pack_into
    {
        using Packing = CreditPacking<5, 8>;
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x10);
        Packing::pack_channel<1>(packed, 0x20);
        Packing::pack_channel<2>(packed, 0x30);
        Packing::pack_channel<3>(packed, 0x40);
        Packing::pack_channel<4>(packed, 0x50);
        write_test_result(test_id++, (packed.value == 0x0000'0050'4030'2010ULL) ? TEST_PASS : TEST_FAIL);
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

    // Test 4: Verify OR behavior (packing same channel twice ORs values)
    {
        auto packed = Packing::PackedValueType{0};
        Packing::pack_channel<0>(packed, 0x0F);  // Lower nibble
        Packing::pack_channel<0>(packed, 0xF0);  // Upper nibble
        // Should OR to 0xFF
        uint32_t ch0 = Packing::extract_channel<0>(packed);
        write_test_result(test_id++, (ch0 == 0xFF) ? TEST_PASS : TEST_FAIL);
    }
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

    // Write completion status as the VERY LAST operation
    // This signals to the host that the kernel ran to completion
    write_test_completion_status(test_id);
}
