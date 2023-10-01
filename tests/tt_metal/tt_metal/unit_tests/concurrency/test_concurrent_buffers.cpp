// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iterator>
#include <thread>
#include <unistd.h>

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>

#include "concurrent_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/executor.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

using namespace tt::tt_metal;
using namespace tt::test_utils;

namespace unit_tests::concurrent::buffers {

void multi_thread_create_buffers(Device *device, const BufferType &buffer_type) {
    uint32_t num_banks = buffer_type == BufferType::L1 ? device->num_banks(BufferType::L1) : device->num_banks(BufferType::DRAM);
    EXPECT_EQ(num_banks % 2, 0);
    uint32_t num_tiles1 = num_banks / 2;
    uint32_t num_tiles2 = num_banks;
    uint32_t num_tiles3 = num_banks * 2;

    const uint32_t page_size = 2048;

    std::set<std::pair<uint32_t, uint32_t>> buffer_addr_ranges;
    boost::interprocess::named_mutex buffer_test_mutex(boost::interprocess::open_or_create, "buffer_test_mutex");

    Buffer buffer_th1_1;
    Buffer buffer_th1_2;
    Buffer buffer_th2;
    Buffer buffer_th3;

    std::vector<std::future<void>> events;

    events.emplace_back(
        detail::async (
            [&] {
                buffer_th1_1 = CreateBuffer(device, num_tiles1 * page_size, page_size, buffer_type);
                buffer_th1_2 = CreateBuffer(device, num_tiles1 * page_size, page_size, buffer_type);
                uint32_t buffer1_bytes_per_bank = detail::SizeBytesPerBank(buffer_th1_1.size(), page_size, num_banks); // both buffers are same size
                boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(buffer_test_mutex);
                buffer_addr_ranges.insert({buffer_th1_1.address(), buffer_th1_1.address() + buffer1_bytes_per_bank});
                buffer_addr_ranges.insert({buffer_th1_2.address(), buffer_th1_2.address() + buffer1_bytes_per_bank});
            }
        )
    );

    events.emplace_back(
        detail::async (
            [&] {
                buffer_th2 = CreateBuffer(device, num_tiles2 * page_size, page_size, buffer_type);
                uint32_t buffer2_bytes_per_bank = detail::SizeBytesPerBank(buffer_th2.size(), page_size, num_banks);
                boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(buffer_test_mutex);
                buffer_addr_ranges.insert({buffer_th2.address(), buffer_th2.address() + buffer2_bytes_per_bank});
            }
        )
    );

    events.emplace_back(
        detail::async (
            [&] {
                buffer_th3 = CreateBuffer(device, num_tiles3 * page_size, page_size, buffer_type);
                uint32_t buffer3_bytes_per_bank = detail::SizeBytesPerBank(buffer_th3.size(), page_size, num_banks);
                boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(buffer_test_mutex);
                buffer_addr_ranges.insert({buffer_th3.address(), buffer_th3.address() + buffer3_bytes_per_bank});
            }
        )
    );

    for (auto &f : events) {
        f.wait();
    }

    EXPECT_TRUE(buffer_addr_ranges.size() == 4);
    auto prev_buff_range_it = buffer_addr_ranges.begin();
    auto buff_range_it = buffer_addr_ranges.begin();
    std::advance(buff_range_it, 1);
    for (; buff_range_it != buffer_addr_ranges.end(); buff_range_it++) {
        std::pair<uint32_t, uint32_t> prev_range = *prev_buff_range_it;
        std::pair<uint32_t, uint32_t> curr_range = *buff_range_it;
        EXPECT_TRUE(curr_range.first >= prev_range.second);
        prev_buff_range_it++;
    }

    boost::interprocess::named_mutex::remove("buffer_test_mutex");
}

std::vector<uint32_t> generate_arange_vector(uint64_t size_bytes, uint32_t start_val=0) {
    TT_ASSERT(size_bytes % sizeof(uint32_t) == 0);
    std::vector<uint32_t> src(size_bytes / sizeof(uint32_t), 0);

    uint32_t val = start_val;
    for (uint32_t i = 0; i < src.size(); i++) {
        src.at(i) = val++;
    }
    return src;
}

void multi_thread_create_wr_rd_buffers(Device *device, const BufferType &buffer_type) {
    uint32_t num_banks = buffer_type == BufferType::L1 ? device->num_banks(BufferType::L1) : device->num_banks(BufferType::DRAM);
    EXPECT_EQ(num_banks % 2, 0);
    uint32_t num_tiles1 = num_banks / 2;
    uint32_t num_tiles2 = num_banks;
    uint32_t num_tiles3 = num_banks * 2;

    const uint32_t page_size = 2048;

    uint32_t buffer_size1 = num_tiles1 * page_size;
    uint32_t buffer_size2 = num_tiles2 * page_size;
    uint32_t buffer_size3 = num_tiles3 * page_size;

    std::vector<uint32_t> th1_1_in = generate_arange_vector(buffer_size1, 0);
    std::vector<uint32_t> th1_2_in = generate_arange_vector(buffer_size1, th1_1_in.back() + 1);
    std::vector<uint32_t> th2_in = generate_arange_vector(buffer_size2, th1_2_in.back() + 1);
    std::vector<uint32_t> th3_in = generate_arange_vector(buffer_size3, th2_in.back() + 1);

    std::vector<std::future<void>> events;

    events.emplace_back(
        detail::async (
            [&] {
                Buffer buffer_th1_1 = CreateBuffer(device, buffer_size1, page_size, buffer_type);
                Buffer buffer_th1_2 = CreateBuffer(device, buffer_size1, page_size, buffer_type);
                WriteToBuffer(buffer_th1_1, th1_1_in);
                WriteToBuffer(buffer_th1_2, th1_2_in);
                std::vector<uint32_t> th1_1_out;
                std::vector<uint32_t> th1_2_out;
                ReadFromBuffer(buffer_th1_1, th1_1_out);
                ReadFromBuffer(buffer_th1_2, th1_2_out);
                EXPECT_EQ(th1_1_in, th1_1_out);
                EXPECT_EQ(th1_2_in, th1_2_out);
            }
        )
    );

    events.emplace_back(
        detail::async (
            [&] {
                Buffer buffer_th2 = CreateBuffer(device, buffer_size2, page_size, buffer_type);
                WriteToBuffer(buffer_th2, th2_in);
                std::vector<uint32_t> th2_out;
                ReadFromBuffer(buffer_th2, th2_out);
                EXPECT_EQ(th2_in, th2_out);
            }
        )
    );

    events.emplace_back(
        detail::async (
            [&] {
                Buffer buffer_th3 = CreateBuffer(device, buffer_size3, page_size, buffer_type);
                WriteToBuffer(buffer_th3, th3_in);
                std::vector<uint32_t> th3_out;
                ReadFromBuffer(buffer_th3, th3_out);
                EXPECT_EQ(th3_in, th3_out);
            }
        )
    );

    for (auto &f : events) {
        f.wait();
    }
}

void multi_process_create_buffers(chip_id_t device_id, const BufferType &buffer_type) {
    pid_t child_a, child_b;
    child_a = fork();

    const uint32_t page_size = 2048;

    typedef boost::interprocess::allocator<std::pair<uint32_t, uint32_t>, boost::interprocess::managed_shared_memory::segment_manager>  ShmemBufferTestAllocator;
    typedef boost::interprocess::vector<std::pair<uint32_t, uint32_t>, ShmemBufferTestAllocator> AddrRangeVector;


    if (child_a == 0) {
        Device *device = CreateDevice(device_id);
        uint32_t num_banks = buffer_type == BufferType::L1 ? device->num_banks(BufferType::L1) : device->num_banks(BufferType::DRAM);
        uint32_t num_tiles1 = num_banks * 2;
        Buffer buffer_th1 = CreateBuffer(device, num_tiles1 * page_size, page_size, buffer_type);
        uint32_t buffer1_bytes_per_bank = detail::SizeBytesPerBank(buffer_th1.size(), page_size, num_banks);
        boost::interprocess::named_mutex buffer_test_mutex(boost::interprocess::open_or_create, "mp_buffer_test_mutex");
        {
            boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(buffer_test_mutex);
            AddrRangeVector *addr_range_vec = tt::concurrent::get_shared_mem_segment().find_or_construct<AddrRangeVector>("AddrRangeVector")(tt::concurrent::get_shared_mem_segment().get_segment_manager());
            addr_range_vec->push_back({buffer_th1.address(), buffer_th1.address() + buffer1_bytes_per_bank});
        }
        CloseDevice(device);
        exit(0);
    } else {
        child_b = fork();

        if (child_b == 0) {
            Device *device = CreateDevice(device_id);
            uint32_t num_banks = buffer_type == BufferType::L1 ? device->num_banks(BufferType::L1) : device->num_banks(BufferType::DRAM);
            uint32_t num_tiles2 = num_banks * 2;
            Buffer buffer_th2 = CreateBuffer(device, num_tiles2 * page_size, page_size, buffer_type);
            uint32_t buffer2_bytes_per_bank = detail::SizeBytesPerBank(buffer_th2.size(), page_size, num_banks);
            boost::interprocess::named_mutex buffer_test_mutex(boost::interprocess::open_or_create, "mp_buffer_test_mutex");
            {
                boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(buffer_test_mutex);
                AddrRangeVector *addr_range_vec = tt::concurrent::get_shared_mem_segment().find_or_construct<AddrRangeVector>("AddrRangeVector")(tt::concurrent::get_shared_mem_segment().get_segment_manager());
                addr_range_vec->push_back({buffer_th2.address(), buffer_th2.address() + buffer2_bytes_per_bank});
            }
            CloseDevice(device);
            exit(0);
        } else {
            // parent process
            Device *device = CreateDevice(device_id);
            uint32_t num_banks = buffer_type == BufferType::L1 ? device->num_banks(BufferType::L1) : device->num_banks(BufferType::DRAM);
            uint32_t num_tiles3 = num_banks * 2;
            Buffer buffer_th3 = CreateBuffer(device, num_tiles3 * page_size, page_size, buffer_type);
            uint32_t buffer3_bytes_per_bank = detail::SizeBytesPerBank(buffer_th3.size(), page_size, num_banks);
            boost::interprocess::named_mutex buffer_test_mutex(boost::interprocess::open_or_create, "mp_buffer_test_mutex");
            AddrRangeVector *addr_range_vec = tt::concurrent::get_shared_mem_segment().find_or_construct<AddrRangeVector>("AddrRangeVector")(tt::concurrent::get_shared_mem_segment().get_segment_manager());
            {
                boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(buffer_test_mutex);
                AddrRangeVector *addr_range_vec = tt::concurrent::get_shared_mem_segment().find_or_construct<AddrRangeVector>("AddrRangeVector")(tt::concurrent::get_shared_mem_segment().get_segment_manager());
                addr_range_vec->push_back({buffer_th3.address(), buffer_th3.address() + buffer3_bytes_per_bank});
            }
            CloseDevice(device);

            int status_a;
            waitpid(child_a, &status_a, 0);

            int status_b;
            waitpid(child_b, &status_b, 0);

            if ( WIFEXITED(status_a) ) {
                const int exit_status = WEXITSTATUS(status_a);
                ASSERT_EQ(0, exit_status);
            }

            if ( WIFEXITED(status_b) ) {
                const int exit_status = WEXITSTATUS(status_b);
                ASSERT_EQ(0, exit_status);
            }

            // Only one process can use the device at a time so memory allocations are also local to a process.
            // This means buffer from each process should be in the same address space.
            EXPECT_TRUE(addr_range_vec->size() == 3);
            std::set<std::pair<uint32_t, uint32_t>> unique_addr_ranges;
            for (int idx = 0; idx < addr_range_vec->size(); idx++) {
                unique_addr_ranges.insert(addr_range_vec->at(idx));
            }
            EXPECT_TRUE(unique_addr_ranges.size() == 1);

            tt::concurrent::get_shared_mem_segment().destroy<AddrRangeVector>("AddrRangeVector");
            boost::interprocess::named_mutex::remove("mp_buffer_test_mutex");
        }
    }
}

bool create_write_and_read_buffer(Device *device, const BufferType &buffer_type, uint64_t page_size, float bank_multiplier) {
    uint32_t num_banks = buffer_type == BufferType::L1 ? device->num_banks(BufferType::L1) : device->num_banks(BufferType::DRAM);
    uint64_t buffer_size = num_banks * bank_multiplier * page_size;
    Buffer buffer = CreateBuffer(device, buffer_size, page_size, buffer_type);
    std::vector<uint32_t> input = generate_arange_vector(buffer_size);
    WriteToBuffer(buffer, input);
    std::vector<uint32_t> output;
    ReadFromBuffer(buffer, output);
    EXPECT_EQ(input, output);
    return input == output;
}

void multi_process_create_wr_rd_buffers(chip_id_t device_id, const BufferType &buffer_type) {
    pid_t child_a, child_b;
    child_a = fork();

    const uint32_t page_size = 2048;

    if (child_a == 0) {
        Device *device = CreateDevice(device_id);
        bool pass = create_write_and_read_buffer(device, buffer_type, page_size, 0.5);
        ASSERT_TRUE(pass);
        pass &= create_write_and_read_buffer(device, buffer_type, page_size, 0.5);
        ASSERT_TRUE(pass);
        CloseDevice(device);
        exit(int(not pass));
    } else {
        child_b = fork();

        if (child_b == 0) {
            Device *device = CreateDevice(device_id);
            bool pass = create_write_and_read_buffer(device, buffer_type, page_size, 1);
            ASSERT_TRUE(pass);
            CloseDevice(device);
            exit(int(not pass));
        } else {
            // parent process
            Device *device = CreateDevice(device_id);
            bool pass = create_write_and_read_buffer(device, buffer_type, page_size, 2);
            ASSERT_TRUE(pass);
            CloseDevice(device);

            int status_a;
            waitpid(child_a, &status_a, 0);

            int status_b;
            waitpid(child_b, &status_b, 0);

            if ( WIFEXITED(status_a) ) {
                const int exit_status = WEXITSTATUS(status_a);
                ASSERT_EQ(0, exit_status);
            }

            if ( WIFEXITED(status_b) ) {
                const int exit_status = WEXITSTATUS(status_b);
                ASSERT_EQ(0, exit_status);
            }
        }
    }
}

} // unit_tests::concurrent::buffers

TEST_F(ConcurrentFixture, TestMultiThreadCreateL1Buffer) {
    const chip_id_t device_id = 0;
    Device *device = CreateDevice(device_id);
    unit_tests::concurrent::buffers::multi_thread_create_buffers(device, BufferType::L1);
    CloseDevice(device);
}

TEST_F(ConcurrentFixture, TestMultiThreadCreateDramBuffer) {
    const chip_id_t device_id = 0;
    Device *device = CreateDevice(device_id);
    unit_tests::concurrent::buffers::multi_thread_create_buffers(device, BufferType::DRAM);
    CloseDevice(device);
}

TEST_F(ConcurrentFixture, TestMultiProcessCreateL1Buffer) {
    const chip_id_t device_id = 0;
    unit_tests::concurrent::buffers::multi_process_create_buffers(device_id, BufferType::L1);
}

TEST_F(ConcurrentFixture, TestMultiProcessCreateDramBuffer) {
    const chip_id_t device_id = 0;
    unit_tests::concurrent::buffers::multi_process_create_buffers(device_id, BufferType::DRAM);
}

TEST_F(ConcurrentFixture, TestMultiThreadWrRdL1Buffer) {
    const chip_id_t device_id = 0;
    Device *device = CreateDevice(device_id);
    unit_tests::concurrent::buffers::multi_thread_create_wr_rd_buffers(device, BufferType::L1);
    CloseDevice(device);
}

TEST_F(ConcurrentFixture, TestMultiThreadWrRdDramBuffer) {
    const chip_id_t device_id = 0;
    Device *device = CreateDevice(device_id);
    unit_tests::concurrent::buffers::multi_thread_create_wr_rd_buffers(device, BufferType::DRAM);
    CloseDevice(device);
}

TEST_F(ConcurrentFixture, TestMultiProcessWrRdL1Buffer) {
    const chip_id_t device_id = 0;
    unit_tests::concurrent::buffers::multi_process_create_wr_rd_buffers(device_id, BufferType::L1);
}

TEST_F(ConcurrentFixture, TestMultiProcessWrRdDramBuffer) {
    const chip_id_t device_id = 0;
    unit_tests::concurrent::buffers::multi_process_create_wr_rd_buffers(device_id, BufferType::DRAM);
}
