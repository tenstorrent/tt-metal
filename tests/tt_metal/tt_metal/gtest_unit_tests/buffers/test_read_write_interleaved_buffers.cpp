#include "../basic_harness.hpp"
#include "gtest/gtest.h"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/common/tile_math.hpp"

using namespace tt::tt_metal;

namespace basic_tests::buffers {

struct BufferConfig {
    u32 page_size;
    u32 total_size;
    BufferType type;
};

vector<u32> generate_arange_vector(u32 size_bytes) {
    TT_ASSERT(size_bytes % sizeof(u32) == 0);
    vector<u32> src(size_bytes / sizeof(u32), 0);

    for (u32 i = 0; i < src.size(); i++) {
        src.at(i) = i;
    }
    return src;
}

bool test_buffer_loopback(Device* device, const BufferConfig& config) {
    Buffer buf(device, config.total_size, config.page_size, config.type);

    vector<u32> src = generate_arange_vector(buf.size());

    WriteToBuffer(buf, src);

    vector<u32> result;
    ReadFromBuffer(buf, result);

    TT_ASSERT(src.size() == result.size());

    if (src != result) {
        for (int i = 0; i < src.size(); i++) {
            if (src.at(i) != result.at(i)) {
                std::cout << "at " << i << " src is " << src.at(i) << " result is " << result.at(i) << std::endl;
            }
        }
    }

    return src == result;
}

TEST_F(L1BankingDeviceHarness, Test32BAlignedDramBufferAndPageSize) {
    constexpr u32 page_size = ADDRESS_ALIGNMENT * 64;
    constexpr BufferType buff_type = BufferType::DRAM;
    auto num_banks = this->device->num_banks(buff_type);
    BufferConfig buffer_config { .page_size = page_size, .total_size = page_size * (num_banks * 128) };
    EXPECT_TRUE(test_buffer_loopback(this->device, buffer_config));
}

TEST_F(L1BankingDeviceHarness, Test32BAlignedL1BufferAndPageSize) {
    constexpr u32 page_size = ADDRESS_ALIGNMENT * 64;
    constexpr BufferType buff_type = BufferType::L1;
    auto num_banks = this->device->num_banks(buff_type);
    BufferConfig buffer_config { .page_size = page_size, .total_size = page_size * (num_banks * 128) };
    EXPECT_TRUE(test_buffer_loopback(this->device, buffer_config));
}

TEST_F(L1BankingDeviceHarness, Test32BAlignedDramBufferAndNon32BAlignedPageSize) {
    u32 page_size = roundup((ADDRESS_ALIGNMENT - 1) * (64 - 1), sizeof(u32));
    constexpr BufferType buff_type = BufferType::DRAM;
    auto num_banks = this->device->num_banks(buff_type);
    BufferConfig buffer_config { .page_size = page_size, .total_size = page_size * (num_banks * 128) };
    EXPECT_NE(buffer_config.page_size % ADDRESS_ALIGNMENT, 0);
    EXPECT_EQ(buffer_config.total_size % ADDRESS_ALIGNMENT, 0);
    EXPECT_TRUE(test_buffer_loopback(this->device, buffer_config));
}

TEST_F(L1BankingDeviceHarness, Test32BAlignedL1BufferAndNon32BAlignedPageSize) {
    u32 page_size = roundup((ADDRESS_ALIGNMENT - 1) * (64 - 1), sizeof(u32));
    constexpr BufferType buff_type = BufferType::L1;
    auto num_banks = this->device->num_banks(buff_type);
    BufferConfig buffer_config { .page_size = page_size, .total_size = page_size * (num_banks * 2) };
    EXPECT_NE(buffer_config.page_size % ADDRESS_ALIGNMENT, 0);
    EXPECT_EQ(buffer_config.total_size % ADDRESS_ALIGNMENT, 0);
    EXPECT_TRUE(test_buffer_loopback(this->device, buffer_config));
}

TEST_F(L1BankingDeviceHarness, TestNon32BAlignedDramBufferAndPageSize) {
    u32 page_size = roundup((ADDRESS_ALIGNMENT - 1) * (64 - 1), sizeof(u32));
    constexpr BufferType buff_type = BufferType::DRAM;
    auto num_banks = this->device->num_banks(buff_type);
    BufferConfig buffer_config { .page_size = page_size, .total_size = page_size * ((num_banks + 1) * 127) };
    EXPECT_NE(buffer_config.page_size % ADDRESS_ALIGNMENT, 0);
    EXPECT_NE(buffer_config.total_size % ADDRESS_ALIGNMENT, 0);
    EXPECT_TRUE(test_buffer_loopback(this->device, buffer_config));
}

TEST_F(L1BankingDeviceHarness, TestNon32BAlignedL1BufferAndPageSize) {
    u32 page_size = roundup((ADDRESS_ALIGNMENT - 1) * (64 - 1), sizeof(u32));
    constexpr BufferType buff_type = BufferType::L1;
    auto num_banks = this->device->num_banks(buff_type);
    BufferConfig buffer_config { .page_size = page_size, .total_size = page_size * ((num_banks + 1) * 63) };
    EXPECT_NE(buffer_config.page_size % ADDRESS_ALIGNMENT, 0);
    EXPECT_NE(buffer_config.total_size % ADDRESS_ALIGNMENT, 0);
    EXPECT_TRUE(test_buffer_loopback(this->device, buffer_config));
}

TEST_F(L1BankingDeviceHarness, TestBackToBackNon32BAlignedPageSize) {
    constexpr BufferType buff_type = BufferType::DRAM;
    auto num_banks = this->device->num_banks(buff_type);

    Buffer bufa(device, 125000, 100, buff_type);
    auto src_a = generate_arange_vector(bufa.size());
    WriteToBuffer(bufa, src_a);

    Buffer bufb(device, 152000, 152, buff_type);
    auto src_b = generate_arange_vector(bufb.size());
    WriteToBuffer(bufb, src_b);

    vector<u32> result_a;
    ReadFromBuffer(bufa, result_a);

    vector<u32> result_b;
    ReadFromBuffer(bufb, result_b);

    EXPECT_EQ(src_a, result_a);
    EXPECT_EQ(src_b, result_b);
}


}   // end namespace basic_tests::buffers
