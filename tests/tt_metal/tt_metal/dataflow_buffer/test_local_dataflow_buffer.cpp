// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstring>
#include <thread>
#include <tuple>
#include <vector>

#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "host/dataflow_buffer.hpp"

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::dataflow_buffer {

constexpr uint8_t MAX_NUM_DM_THREADS = 8;
constexpr uint8_t MAX_NUM_COMPUTE_THREADS = 4;

void random_delay(uint32_t min_ms = 10, uint32_t max_ms = 150) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(min_ms, max_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(dist(rng)));
}

// Writer produces data in the dataflow buffer, readers consumes
struct TestParams {
    uint32_t num_buffers = 1;
    uint32_t num_writer_threads = 1;
    uint32_t num_reader_threads = 1;
    DataflowBufferAccessPattern writer_access_pattern = DataflowBufferAccessPattern::NONE;
    DataflowBufferAccessPattern reader_access_pattern = DataflowBufferAccessPattern::NONE;
    bool writer_is_dm = false;
    bool reader_is_dm = false;
};

class DataflowBufferTestSuite : public ::testing::TestWithParam<TestParams> {
protected:
    void SetUp() override {
        params = GetParam();
        data.resize(this->data_size_bytes / sizeof(uint32_t));
        std::iota(data.begin(), data.end(), 0);
        dfb_data.resize(this->dataflow_buffer_size_bytes / sizeof(uint32_t));
    }

    void TearDown() override {}

    void run_writer_thread(uint32_t thread_id, uint32_t dfb_index) {
        random_delay();

        switch (this->params.writer_access_pattern) {
            case DataflowBufferAccessPattern::STRIDED: {
                dev::DataflowBuffer<DataflowBufferAccessPattern::STRIDED, DataflowBufferAccessPattern::STRIDED> dfb(
                    dfb_index);
                uint32_t total_num_pages = this->data_size_bytes / this->page_size_bytes;
                std::cout << "total_num_pages: " << total_num_pages << std::endl;

                auto src_data_rd_ptr = this->data.data();
                std::cout << "src_data_rd_ptr: " << std::hex << src_data_rd_ptr << std::dec << std::endl;
                for (uint32_t i = 0; i < total_num_pages; i += 8) {
                    std::cout << "reserve_back" << std::endl;
                    dfb.reserve_back(8, thread_id);

                    uint32_t num_bytes = 8 * dev::overlay_cluster_dfb_access_pattern_tracker[0].page_size;

                    std::cout << "starting memcpy dfb wr ptr: " << std::hex
                              << dev::overlay_cluster_dfb_access_pattern_tracker[0].wr_ptr << std::dec << std::endl;
                    memcpy(
                        reinterpret_cast<uintptr_t*>(dev::overlay_cluster_dfb_access_pattern_tracker[0].wr_ptr),
                        src_data_rd_ptr,
                        num_bytes);
                    std::cout << "memcpy done" << std::endl;
                    src_data_rd_ptr += static_cast<size_t>(num_bytes) / sizeof(*src_data_rd_ptr);

                    // for (auto x : this->dfb_data) {
                    //     std::cout << x << "\t";
                    // }
                    // std::cout << "\n";

                    random_delay();

                    std::cout << "push_back" << std::endl;
                    dfb.push_back(8, thread_id);
                }

            } break;
            default: {
                ASSERT_TRUE(false) << "Unsupported writer access pattern";
            } break;
        }
    }

    void run_reader_thread(uint32_t thread_id, uint32_t dfb_index) {
        // random_delay();

        switch (this->params.reader_access_pattern) {
            case DataflowBufferAccessPattern::STRIDED: {
                dev::DataflowBuffer<DataflowBufferAccessPattern::STRIDED, DataflowBufferAccessPattern::STRIDED> dfb(
                    dfb_index);
                uint32_t total_num_pages = this->data_size_bytes / this->page_size_bytes;
                std::cout << "total_num_pages: " << total_num_pages << std::endl;

                for (uint32_t i = 0; i < total_num_pages; i += 8) {
                    dfb.wait_front(8, thread_id);

                    uint32_t num_bytes = 8 * dev::overlay_cluster_dfb_access_pattern_tracker[0].page_size;
                    uint32_t num_elements = num_bytes / sizeof(uint32_t);

                    uintptr_t rd_ptr = dev::overlay_cluster_dfb_access_pattern_tracker[0].rd_ptr;
                    std::cout << "rd_ptr: " << std::hex << rd_ptr << std::dec << std::endl;
                    uint32_t* read_data_ptr = reinterpret_cast<uint32_t*>(rd_ptr);

                    for (int i = 0; i < num_elements; i++) {
                        std::cout << read_data_ptr[i] << "\t";
                    }
                    std::cout << "\n";
                    random_delay();

                    std::cout << "pop_front" << std::endl;
                    dfb.pop_front(8, thread_id);
                }

            } break;
            default: {
                ASSERT_TRUE(false) << "Unsupported reader access pattern";
            } break;
        }
    }

    TestParams params;
    uint32_t data_size_bytes = 4096;
    uint32_t page_size_bytes = 32;
    uint32_t dataflow_buffer_size_bytes = 1024;
    std::vector<uint32_t> data;
    std::vector<uint32_t> dfb_data;
};

TEST_P(DataflowBufferTestSuite, TestLocalDataflowBuffer) {
    // Writer adds data into the dataflow buffer and reader consumes data from the dataflow buffer
    if (this->params.writer_is_dm) {
        ASSERT_LE(this->params.num_writer_threads, MAX_NUM_DM_THREADS);
    } else {
        ASSERT_LE(this->params.num_writer_threads, MAX_NUM_COMPUTE_THREADS);
    }

    if (this->params.reader_is_dm) {
        ASSERT_LE(this->params.num_reader_threads, MAX_NUM_DM_THREADS);
    } else {
        ASSERT_LE(this->params.num_reader_threads, MAX_NUM_COMPUTE_THREADS);
    }

    DataflowBufferConfig config(this->dataflow_buffer_size_bytes);
    // this can be based on logical user index
    config.builder()
        .set_data_format(tt::DataFormat::Float32)
        .set_page_size(this->page_size_bytes)
        .set_access_pattern(DataflowBufferConfig::AccessPattern{
            .write_pattern = this->params.writer_access_pattern,
            .read_pattern = this->params.reader_access_pattern,
            .num_writer_threads = (uint8_t)this->params.num_writer_threads,
            .num_reader_threads = (uint8_t)this->params.num_reader_threads});

    auto dfb_index = CreateDataflowBuffer(config, CoreCoord(0, 0));

    // Below local_dfb_interface_twould be done before kernel launch
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(dfb_data.data());

    uintptr_t src_addr = reinterpret_cast<uintptr_t>(this->data.data());
    std::cout << "src_addr: 0x" << std::hex << src_addr << std::dec << std::endl;
    std::cout << "base_addr: 0x" << std::hex << base_addr << std::dec << std::endl;

    dev::overlay_cluster_dfb_access_pattern_tracker[dfb_index] = {
        .size = config.total_size(),
        .limit = base_addr + config.total_size(),
        .page_size = config.page_size(),
        .num_pages = config.total_size() / config.page_size(),
        .rd_ptr = base_addr,
        .wr_ptr = base_addr,
        .wapt = config.access_pattern().write_pattern,
        .rapt = config.access_pattern().read_pattern,
    };
    // End local_dfb_interface_t setup

    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < this->params.num_writer_threads; ++i) {
        threads.emplace_back([this, i, dfb_index]() { this->run_writer_thread(i, dfb_index); });
    }

    for (uint32_t i = 0; i < this->params.num_reader_threads; ++i) {
        threads.emplace_back([this, i, dfb_index]() { this->run_reader_thread(i, dfb_index); });
    }

    for (auto& t : threads) {
        t.join();
    }

    // std::cout << "Printing original data" << std::endl;
    // for (int i = 0; i < this->data.size(); i++) {
    //     std::cout << this->data[i] << "\t";
    // }
    // std::cout << "\n";

    // std::cout << "Printing dfb data" << std::endl;
    // for (int i = 0; i < this->dfb_data.size(); i++) {
    //     std::cout << this->dfb_data[i] << "\t";
    // }
    // std::cout << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    DataflowBufferTestSuite,
    DataflowBufferTestSuite,
    ::testing::Values(
        TestParams{1, 1, 1, DataflowBufferAccessPattern::STRIDED, DataflowBufferAccessPattern::STRIDED, true, false}
        // TestParams{
        //     1, 1, 4, DataflowBufferAccessPattern::STRIDED, DataflowBufferAccessPattern::STRIDED, true, false},
        ));

}  // namespace unit_tests::dataflow_buffer
}  // namespace tt::tt_metal
