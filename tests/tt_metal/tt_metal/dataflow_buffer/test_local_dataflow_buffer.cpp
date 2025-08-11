// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstring>
#include <thread>
#include <tuple>
#include <vector>

#include "dfb_test_common.hpp"
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

    void setup_thread_local_dfb_interface(uint32_t dfb_index, uint32_t thread_local_size) {
        // Each thread needs to set up its own copy of the dfb interface
        uintptr_t base_addr = reinterpret_cast<uintptr_t>(dfb_data.data()) + (current_thread_id * thread_local_size);
        dev::overlay_cluster_dfb_access_pattern_tracker[dfb_index] = {
            .size = thread_local_size,
            .limit = base_addr + thread_local_size,
            .page_size = this->page_size_bytes,
            .num_pages = this->dataflow_buffer_size_bytes / this->page_size_bytes,
            .rd_ptr = base_addr,
            .wr_ptr = base_addr,
            .wapt = this->params.writer_access_pattern,
            .rapt = this->params.reader_access_pattern,
        };
    }

    void run_writer_thread(uint32_t dfb_index, uint32_t size_per_overlay_thread) {
        // Set up this thread's copy of overlay_cluster_dfb_access_pattern_tracker
        auto& out = get_thread_output_stream(); 
        setup_thread_local_dfb_interface(dfb_index, size_per_overlay_thread);

        random_delay();

        switch (this->params.writer_access_pattern) {
            case DataflowBufferAccessPattern::STRIDED: {
                dev::DataflowBuffer<DataflowBufferAccessPattern::STRIDED, DataflowBufferAccessPattern::STRIDED> dfb(
                    dfb_index);
                uint32_t total_num_pages = (this->data_size_bytes / this->page_size_bytes) / this->params.num_writer_threads;
                out << "total_num_pages: " << total_num_pages << std::endl;

                uint32_t num_bytes = 8 * dev::overlay_cluster_dfb_access_pattern_tracker[dfb_index].page_size;

                auto src_data_rd_ptr = this->data.data();
                src_data_rd_ptr += static_cast<size_t>(num_bytes * current_thread_id) / sizeof(*src_data_rd_ptr);
                out << "src_data_rd_ptr: " << std::hex << src_data_rd_ptr << std::dec << std::endl;
                for (uint32_t i = 0; i < total_num_pages; i += 8) {
                    out << "reserve_back" << std::endl;
                    dfb.reserve_back(8);

                    out << "starting memcpy dfb wr ptr: " << std::hex
                              << dev::overlay_cluster_dfb_access_pattern_tracker[dfb_index].wr_ptr << std::dec << std::endl;
                    memcpy(
                        reinterpret_cast<uintptr_t*>(dev::overlay_cluster_dfb_access_pattern_tracker[dfb_index].wr_ptr),
                        src_data_rd_ptr,
                        num_bytes);
                    out << "memcpy done" << std::endl;
                    src_data_rd_ptr += static_cast<size_t>(num_bytes * this->params.num_writer_threads) / sizeof(*src_data_rd_ptr);

                    // for (auto x : this->dfb_data) {
                    //     std::cout << x << "\t";
                    // }
                    // std::cout << "\n";

                    random_delay();

                    out << "push_back" << std::endl;
                    dfb.push_back(8);
                }

            } break;
            default: {
                ASSERT_TRUE(false) << "Unsupported writer access pattern";
            } break;
        }
    }

    void run_reader_thread(uint32_t dfb_index, uint32_t size_per_compute_thread) {
        // Set up this thread's copy of overlay_cluster_dfb_access_pattern_tracker
        auto& out = get_thread_output_stream();
        setup_thread_local_dfb_interface(dfb_index, size_per_compute_thread);

        // random_delay();

        switch (this->params.reader_access_pattern) {
            case DataflowBufferAccessPattern::STRIDED: {
                dev::DataflowBuffer<DataflowBufferAccessPattern::STRIDED, DataflowBufferAccessPattern::STRIDED> dfb(
                    dfb_index);
                uint32_t total_num_pages = (this->data_size_bytes / this->page_size_bytes) / this->params.num_reader_threads;
                out << "total_num_pages: " << total_num_pages << std::endl;

                for (uint32_t i = 0; i < total_num_pages; i += 8) {
                    dfb.wait_front(8);

                    uint32_t num_bytes = 8 * dev::overlay_cluster_dfb_access_pattern_tracker[dfb_index].page_size;
                    uint32_t num_elements = num_bytes / sizeof(uint32_t);

                    uintptr_t rd_ptr = dev::overlay_cluster_dfb_access_pattern_tracker[dfb_index].rd_ptr;
                    out << "rd_ptr: " << std::hex << rd_ptr << std::dec << std::endl;
                    uint32_t* read_data_ptr = reinterpret_cast<uint32_t*>(rd_ptr);

                    for (int i = 0; i < num_elements; i++) {
                        out << read_data_ptr[i] << "\t";
                    }
                    out << "\n";
                    random_delay();

                    out << "pop_front" << std::endl;
                    dfb.pop_front(8);
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

// These tests assume that overlay is the producer of data into a dataflow buffer and compute is the consumer
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
    uint8_t dfb_index = 0;
    config.index(dfb_index)
        .set_data_format(tt::DataFormat::Float32)
        .set_page_size(this->page_size_bytes)
        .set_access_pattern(DataflowBufferConfig::AccessPattern{
            .write_pattern = this->params.writer_access_pattern,
            .read_pattern = this->params.reader_access_pattern,
            .num_writer_threads = (uint8_t)this->params.num_writer_threads,
            .num_reader_threads = (uint8_t)this->params.num_reader_threads});

    // atm for sim purposes, below api also sets the hw sync registers capacities
    CreateDataflowBuffer(config, CoreCoord(0, 0));

    uintptr_t base_addr = reinterpret_cast<uintptr_t>(dfb_data.data());
    uint32_t size_per_overlay_thread = this->dataflow_buffer_size_bytes / this->params.num_writer_threads;
    uint32_t size_per_compute_thread = this->dataflow_buffer_size_bytes / this->params.num_reader_threads;

    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < this->params.num_writer_threads; ++i) {
        threads.emplace_back([this, i, dfb_index, size_per_overlay_thread]() { 
            current_thread_id = i;
            is_overlay_thread = true;
            this->run_writer_thread(dfb_index, size_per_overlay_thread); 
        });
    }

    for (uint32_t i = 0; i < this->params.num_reader_threads; ++i) {
        threads.emplace_back([this, i, dfb_index, size_per_compute_thread]() { 
            current_thread_id = i;
            is_overlay_thread = false;
            this->run_reader_thread(dfb_index, size_per_compute_thread); 
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

INSTANTIATE_TEST_SUITE_P(
    DataflowBufferTestSuite,
    DataflowBufferTestSuite,
    ::testing::Values(
        // TestParams{1, 1, 1, DataflowBufferAccessPattern::STRIDED, DataflowBufferAccessPattern::STRIDED, true, false}
        // TestParams{1, 1, 4, DataflowBufferAccessPattern::STRIDED, DataflowBufferAccessPattern::STRIDED, true, false}
        TestParams{1, 4, 4, DataflowBufferAccessPattern::STRIDED, DataflowBufferAccessPattern::STRIDED, true, false}));

}  // namespace unit_tests::dataflow_buffer
}  // namespace tt::tt_metal
