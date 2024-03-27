// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include "tt_metal/common/core_coord.h"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "tensor/tensor_impl.hpp"
#include "tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include <sstream>
#include <type_traits>

using namespace tt::constants;

namespace tt {

namespace tt_metal {


std::tuple<CoreRangeSet,CoreRangeSet> select_worker_cores(AllGatherConfig const& all_gather_config, uint32_t num_links, uint32_t link) {
    constexpr uint32_t worker_grid_width = 8;
    const bool fit_sender_and_receiver_workers_on_same_row = (worker_grid_width / 2) >= all_gather_config.get_num_buffers_per_link();
    std::set<CoreRange> receiver_worker_cores = {};
    std::set<CoreRange> sender_worker_cores = {};
    uint32_t max_cols = 8;
    uint32_t curr_row = link * (((all_gather_config.get_num_buffers_per_link() * 2 - 1) / max_cols) + 1);
    uint32_t curr_col = 0;
    for (uint32_t r = 0; r < all_gather_config.get_num_buffers_per_link(); r++) {
        receiver_worker_cores.insert(CoreRange(CoreCoord(curr_col, curr_row)));
        curr_col ++;
        if (curr_col == max_cols) {
            curr_col = 0;
            curr_row++;
        }
    }
    for (uint32_t s = 0; s < all_gather_config.get_num_buffers_per_link(); s++) {
        sender_worker_cores.insert(CoreRange(CoreCoord(curr_col, curr_row)));
        curr_col ++;
        if (curr_col == max_cols) {
            curr_col = 0;
            curr_row++;
        }
    }
    return {CoreRangeSet(receiver_worker_cores), CoreRangeSet(sender_worker_cores)};
}

class AllGatherOpTensorConfig {
   public:
    AllGatherOpTensorConfig(Tensor const& input_tensor) :
        df(tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype()))
    {}

   protected:
    uint32_t page_size;
    DataFormat df;
};

class AllGatherOpInterleavedTensorConfig final : public AllGatherOpTensorConfig {
   public:
    AllGatherOpInterleavedTensorConfig(Tensor const& input_tensor) :
        AllGatherOpTensorConfig(input_tensor),
        shard_spec(input_tensor.shard_spec().value()) {
        if (input_tensor.get_layout() == Layout::TILE) {
            this->page_size = tt_metal::detail::TileSize(this->df);
            this->unit_size = page_size;
        } else {
            this->unit_size = shard_spec.shape[1] * input_tensor.element_size();
            this->page_size = input_tensor.get_legacy_shape()[-1] * input_tensor.element_size();
        }
    }

   private:
    uint32_t unit_size;
    ShardSpec const shard_spec;
};

class AllGatherOpShardedTensorConfig final : public AllGatherOpTensorConfig {
   public:
    AllGatherOpShardedTensorConfig(Tensor const& input_tensor) :
        AllGatherOpTensorConfig(input_tensor),
        shard_spec(input_tensor.shard_spec().value()) {
        if (input_tensor.get_layout() == Layout::TILE) {
            this->page_size = tt_metal::detail::TileSize(this->df);
            this->unit_size = page_size;
        } else {
            this->unit_size = shard_spec.shape[1] * input_tensor.element_size();
            this->page_size = input_tensor.get_legacy_shape()[-1] * input_tensor.element_size();
        }
    }

   private:
    uint32_t unit_size;
    ShardSpec const shard_spec;
};

operation::ProgramWithCallbacks all_gather_multi_core_with_workers(const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t num_links, const uint32_t ring_size, const uint32_t ring_index, const chip_id_t receiver_device_id, const chip_id_t sender_device_id) {

    tt_metal::Program program{};
    auto const& all_gather_config = AllGatherConfig(input_tensor, output_tensor, dim, ring_size, num_links);

    auto const& sharding_info = ShardedAllGatherConfig(input_tensor, output_tensor, dim);
    bool enable_print = false; // ring_index == 0
    if (enable_print) {
        all_gather_config.print();
    }

    bool is_sharded = input_tensor.is_sharded();

    TT_FATAL(input_tensor.buffer()->page_size() <= all_gather_config.get_eth_buffer_size(), "Page size too large");

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    int32_t shard_size_in_bytes = is_sharded ?
        (input_buffer->shard_spec().page_shape[0] * input_buffer->shard_spec().page_shape[1] * input_buffer->shard_spec().tensor2d_shape[0] * input_buffer->shard_spec().tensor2d_shape[1] * input_tensor.element_size()) / input_tensor.shard_spec()->num_cores() :
        -1;
    uint32_t input_page_size = is_sharded ? shard_size_in_bytes : input_buffer->page_size();
    if (is_sharded) {
        log_trace(tt::LogOp, "input_buffer->shard_spec().page_shape[0]: {}", input_buffer->shard_spec().page_shape[0]);
        log_trace(tt::LogOp, "input_buffer->shard_spec().page_shape[1]: {}", input_buffer->shard_spec().page_shape[1]);
        log_trace(tt::LogOp, "input_buffer->shard_spec().tensor2d_shape[0]: {}", input_buffer->shard_spec().tensor2d_shape[0]);
        log_trace(tt::LogOp, "input_buffer->shard_spec().tensor2d_shape[1]: {}", input_buffer->shard_spec().tensor2d_shape[1]);
        log_trace(tt::LogOp, "input_tensor.element_size(): {}", input_tensor.element_size());
    }
    const uint32_t max_buffer_per_chunk = is_sharded ?
        round_down(all_gather_config.get_eth_buffer_size(), shard_size_in_bytes):
        round_down(all_gather_config.get_eth_buffer_size(), input_page_size);
    const uint32_t max_pages_per_chunk = is_sharded ?
        max_buffer_per_chunk / shard_size_in_bytes :
        max_buffer_per_chunk / input_page_size;
    log_trace(tt::LogOp, "shard_size_in_bytes: {}", shard_size_in_bytes);
    log_trace(tt::LogOp, "input_page_size: {}", input_page_size);
    log_trace(tt::LogOp, "max_buffer_per_chunk: {}", max_buffer_per_chunk);
    log_trace(tt::LogOp, "max_pages_per_chunk: {}", max_pages_per_chunk);
    const auto& device = input_tensor.device();
    uint32_t sender_socket_idx = 0;
    uint32_t receiver_socket_idx = 0;
    if (receiver_device_id == sender_device_id) {
        if (ring_index == 0) {
            receiver_socket_idx = 1;
        } else {
            sender_socket_idx = 1;
        }
    }

    const uint32_t num_transfers = ring_size - 1;

    bool rm = input_tensor.get_layout() == Layout::ROW_MAJOR;
    bool width = input_tensor.get_legacy_shape().rank() - 1 == dim;
    DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t global_num_workers = all_gather_config.get_num_buffers_per_link() * num_links;
    uint32_t shard_width = 0;
    uint32_t shard_height = 0;
    if (is_sharded) {
        shard_width =  input_tensor.buffer()->shard_spec().page_shape.back();
        shard_height = input_tensor.buffer()->shard_spec().page_shape.front();
    }

    std::map<string, string> worker_defines;
    if (rm) {
        worker_defines["RM_INTERLEAVED"] = "1";
    } else {
        worker_defines["TILE_INTERLEAVED"] = "1";
    }

    // number of worker cores is 2x this since there is 1 worker for the sender buffer and 1 worker for the receiver buffer
    uint32_t total_worker_core_pairs_used = num_links * all_gather_config.get_num_buffers_per_link();
    std::vector<CoreCoord> eth_sender_cores;
    eth_sender_cores.reserve(num_links);
    std::vector<CoreCoord> eth_receiver_cores;
    eth_receiver_cores.reserve(num_links);
    std::vector<KernelHandle> eth_sender_kernels;
    eth_sender_kernels.reserve(num_links);
    std::vector<KernelHandle> eth_receiver_kernels;
    eth_receiver_kernels.reserve(num_links);

    std::vector<CoreRange> worker_sender_cores;
    worker_sender_cores.reserve(num_links);
    std::vector<KernelHandle> worker_reader_sender_kernels;
    worker_reader_sender_kernels.reserve(total_worker_core_pairs_used);
    std::vector<KernelHandle> worker_writer_sender_kernels;
    worker_writer_sender_kernels.reserve(total_worker_core_pairs_used);

    std::vector<CoreRange> worker_receiver_cores;
    worker_receiver_cores.reserve(num_links);
    std::vector<KernelHandle> worker_reader_receiver_kernels;
    worker_reader_receiver_kernels.reserve(total_worker_core_pairs_used);
    std::vector<KernelHandle> worker_writer_receiver_kernels;
    worker_writer_receiver_kernels.reserve(total_worker_core_pairs_used);

    std::vector<CoreCoord> all_worker_sender_cores;
    all_worker_sender_cores.reserve(total_worker_core_pairs_used);
    std::vector<CoreCoord> all_worker_receiver_cores;
    all_worker_receiver_cores.reserve(total_worker_core_pairs_used);

    for (uint32_t l = 0; l < num_links; ++l) {
        // Get the cores for the sender and receiver worker cores
        auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id).at(sender_socket_idx + l);
        eth_sender_cores.push_back(eth_sender_core);
        auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id).at(receiver_socket_idx + l);
        eth_receiver_cores.push_back(eth_receiver_core);
    }

    uint32_t num_input_pages = input_tensor.buffer()->size() / input_page_size;
    uint32_t min_pages_per_link = num_input_pages / num_links;

    std::vector<uint32_t> pages_per_link(num_links, min_pages_per_link);
    for (uint32_t i = 0; i < num_input_pages % min_pages_per_link; ++i) {
        pages_per_link.at(i)++;
    }

    uint32_t num_rows = 0, num_cols = 0, row_offset = 0, col_offset = 0, num_tiles = 0;

    if (rm) {
        num_cols = input_tensor.get_legacy_shape()[-1];
        auto input_shape = input_tensor.get_legacy_shape();
        auto output_shape = output_tensor.get_legacy_shape();
        num_rows = std::accumulate(input_shape.begin()+dim, input_shape.end() - 1, 1, std::multiplies<uint32_t>());
        row_offset = std::accumulate(output_shape.begin()+dim, output_shape.end() - 1, 1, std::multiplies<uint32_t>()) - num_rows;
    } else {
        num_cols = input_tensor.get_legacy_shape()[-1] / TILE_WIDTH;
        auto input_shape = input_tensor.get_legacy_shape();
        auto output_shape = output_tensor.get_legacy_shape();
        uint32_t num_output_cols = output_tensor.get_legacy_shape()[-1] / TILE_WIDTH;
        num_rows = std::accumulate(input_shape.begin()+dim, input_shape.end() - 1, 1, std::multiplies<uint32_t>()) / TILE_HEIGHT;
        row_offset = (std::accumulate(output_shape.begin()+dim, output_shape.end() - 1, 1, std::multiplies<uint32_t>()) / TILE_HEIGHT - num_rows) * num_output_cols;
        col_offset = num_output_cols - num_cols;
        num_tiles = num_rows * num_cols;
    }


    uint32_t output_page_size = output_buffer->page_size();

    uint32_t total_output_pages = output_buffer->size() / output_page_size;

    uint32_t input_start_page_idx = 0;
    uint32_t output_addr_offset = 0;
    uint32_t col_idx = 0;
    uint32_t row_idx = 0;
    uint32_t output_page_offset = 0;

    if (rm) {
        if (width) {
            output_addr_offset = input_page_size;
        } else {
            output_page_offset = num_rows;
        }
    } else {
        if (width) {
            output_page_offset = num_cols;
        } else {
            output_page_offset = num_tiles;
        }
    }
    uint32_t output_start_page_idx = ring_index * output_page_offset;
    uint32_t output_start_addr_offset = ring_index * output_addr_offset;

    ///
    /// (counter clockwise sender) < ----- (this chip) < ----- (counter-clockwise receiver)
    ///
    /// (clockwise receiver)       ------> (this chip) ------> (clockwise sender)
    /// So clockwise sender and counter-clockwise receiver are on the same core
    //  and counter-clockwise sender and clockwise receiver are on the same corwe

    // Clockwise Direction
    std::vector<uint32_t> link_clockwise_sender_channels_offsets =
        std::vector<uint32_t>(num_links, 0);
    std::vector<uint32_t> link_clockwise_sender_num_channels =
        std::vector<uint32_t>(num_links, all_gather_config.get_num_buffers_in_clockwise_direction());
    std::vector<uint32_t> link_clockwise_receiver_num_channels = link_clockwise_sender_num_channels;
    // The clockwise direction's erisc's receiver offsets (i.e. for transfers coming INTO this chip)
    std::vector<uint32_t> link_clockwise_receiver_channels_offsets = link_clockwise_sender_channels_offsets;

    // Counter Clockwise Direction
    std::vector<uint32_t> link_counter_clockwise_sender_channels_offsets =
        std::vector<uint32_t>(num_links, all_gather_config.get_num_buffers_in_clockwise_direction());
    // Counter clock-wise buffers start after clockwise buffers in L1
    std::vector<uint32_t> link_counter_clockwise_sender_num_channels =
        std::vector<uint32_t>(num_links, all_gather_config.get_num_buffers_in_counter_clockwise_direction());
    std::vector<uint32_t> link_counter_clockwise_receiver_channels_offsets = link_counter_clockwise_sender_channels_offsets;
    std::vector<uint32_t> link_counter_clockwise_receiver_num_channels = link_counter_clockwise_sender_num_channels;

    std::vector<uint32_t> eth_sem_addrs;
    std::vector<uint32_t> eth_buffer_addrs;
    eth_sem_addrs.reserve(all_gather_config.get_num_buffers_per_link());
    eth_buffer_addrs.reserve(all_gather_config.get_num_buffers_per_link());

    for (uint32_t b = 0, eth_sem_addr = all_gather_config.get_eth_sems_l1_base_byte_address(), eth_buffer_addr = all_gather_config.get_eth_buffers_l1_base_byte_address(); b < all_gather_config.get_num_buffers_per_link(); ++b) {
        eth_sem_addrs.push_back(eth_sem_addr);
        eth_sem_addr += all_gather_config.get_semaphore_size();
        eth_buffer_addrs.push_back(eth_buffer_addr);
        eth_buffer_addr += (all_gather_config.get_eth_buffer_size() + 16);
        TT_ASSERT((eth_buffer_addr - 16) % 16 == 0);
    }

    for (uint32_t i = 0; i < num_links; ++i) {
        // We can't have overlap between the mcast grid for worker cores for different links since mcasting the semaphore in receiver would corrupt other link semaphores
        // We can have overlap between a link's sender and receiver worker grids if we have the semaphores at different addresses
        auto const& [receiver_workers, sender_workers] = select_worker_cores(all_gather_config, num_links, i);

        // Circular Buffer Setup
        uint32_t cb_page_size = is_sharded ? shard_size_in_bytes : input_page_size;
        log_trace(tt::LogOp, "input_page_size: {}", input_page_size);
        uint32_t cb_num_pages = 2 * max_pages_per_chunk;
        log_trace(tt::LogOp, "cb_num_pages: {}", cb_num_pages);
        uint32_t src0_cb_index = CB::c_in0;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb_num_pages * cb_page_size, {{src0_cb_index, df}})
		.set_page_size(src0_cb_index, cb_page_size);
        CBHandle cb_src0_sender_workers = CreateCircularBuffer(program, sender_workers, cb_src0_config);
        CBHandle cb_src0_receiver_workers = CreateCircularBuffer(program, receiver_workers, cb_src0_config);

        // This semaphore is used by the receiver core to tell workers that data is available to read
        auto receiver_worker_semaphore_addr = tt_metal::CreateSemaphore(program, receiver_workers, 0);
        // This semaphore is used by the receiver core to tell the worker sender writer that sender buffer is available to write to
        auto sender_worker_writer_semaphore_addr = tt_metal::CreateSemaphore(program, sender_workers, 0);
        // This semaphore is used by the worker receiver writer to tell the worker sender reader that data has been committed to memory
        // This is currently a running counter of how many chunks were committed since the sender worker never decrements this buffer
        // Potentially avoid overflow by having it actually decrement (using noc atomic inc with value of -1)
        auto sender_worker_reader_semaphore_addr = tt_metal::CreateSemaphore(program, sender_workers, 0);

        auto sender_noc = detail::GetPreferredNOCForDRAMRead(tt::Cluster::instance().arch());
        auto receiver_noc = detail::GetPreferredNOCForDRAMWrite(tt::Cluster::instance().arch());

        auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id).at(sender_socket_idx);
        eth_sender_cores.push_back(eth_sender_core);
        auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id).at(receiver_socket_idx);
        eth_receiver_cores.push_back(eth_receiver_core);

        // Rename this the _channel
        std::vector<uint32_t> pages_per_buffer;

        // number of pages that can fit in a single ethernet L1 buffer (not the number of pages sent to this channel)
        std::vector<uint32_t> pages_per_eth_l1_buffer;
        pages_per_buffer.reserve(all_gather_config.get_num_buffers_per_link());
        // std::cout << "all_gather_config.get_eth_buffer_size()=" << all_gather_config.get_eth_buffer_size() << std::endl;
        // std::cout << "input_tensor.buffer()->page_size()=" << input_tensor.buffer()->page_size() << std::endl;
        uint32_t max_pages_per_eth_l1_sender_buffer = all_gather_config.get_eth_buffer_size() / input_page_size;
        for(uint32_t b = 0; b < all_gather_config.get_num_buffers_per_link(); ++b) {
            pages_per_buffer.push_back((pages_per_link.at(i) / all_gather_config.get_num_buffers_per_link()));
            pages_per_eth_l1_buffer.push_back(
                is_sharded ? std::min(pages_per_buffer.back(), max_pages_per_eth_l1_sender_buffer)
                           : max_pages_per_eth_l1_sender_buffer);
            if (b < pages_per_link.at(i) % all_gather_config.get_num_buffers_per_link()) {
                pages_per_buffer.back()++;
            }

            log_trace(tt::LogOp, "pages_per_link[{}]: {}", i, pages_per_link.at(i));
            log_trace(tt::LogOp, "pages_per_buffer[{}]: {}", b, pages_per_buffer.at(b));
            log_trace(tt::LogOp, "max_pages_per_eth_l1_sender_buffer: {}",max_pages_per_eth_l1_sender_buffer);
        }
        TT_ASSERT(std::accumulate(pages_per_buffer.begin(), pages_per_buffer.end(), 0) == pages_per_link.at(i));


        uint32_t bytes_per_chunk = 0, pages_per_chunk = 0, num_full_chunks = 0, rem_bytes = 0, rem_pages = 0;
        uint32_t link_size_bytes = pages_per_link.at(i) * input_page_size;
        if (pages_per_link.at(i) >= max_pages_per_chunk) {
            bytes_per_chunk = max_buffer_per_chunk;
            pages_per_chunk = max_pages_per_chunk;
            TT_ASSERT(max_buffer_per_chunk == max_pages_per_chunk * input_page_size);
            num_full_chunks = link_size_bytes / bytes_per_chunk;
            rem_bytes = link_size_bytes % bytes_per_chunk;
            rem_pages = pages_per_link.at(i) % max_pages_per_chunk;
        } else {
            rem_bytes = link_size_bytes;
            rem_pages = pages_per_link.at(i);
        }

        auto sender_worker_cores = corerange_to_cores(sender_workers, std::nullopt, true);
        auto receiver_worker_cores = corerange_to_cores(receiver_workers, std::nullopt, true);
        all_worker_sender_cores.insert(all_worker_sender_cores.end(), sender_worker_cores.begin(), sender_worker_cores.end());
        all_worker_receiver_cores.insert(all_worker_receiver_cores.end(), receiver_worker_cores.begin(), receiver_worker_cores.end());

        TT_ASSERT(rem_pages < pages_per_chunk || num_full_chunks == 0);
        TT_ASSERT(rem_pages <= max_pages_per_chunk);
        std::vector<uint32_t> num_full_chunks_per_worker(all_gather_config.get_num_buffers_per_link(), num_full_chunks / all_gather_config.get_num_buffers_per_link());
        std::vector<uint32_t> rem_pages_per_worker(all_gather_config.get_num_buffers_per_link(), 0);
        {
            uint32_t worker_idx = 0;
            for (worker_idx = 0; worker_idx < num_full_chunks % all_gather_config.get_num_buffers_per_link(); ++worker_idx) {
                num_full_chunks_per_worker.at(worker_idx)++;
            }
            if (rem_pages != 0) {
                rem_pages_per_worker.at(worker_idx % all_gather_config.get_num_buffers_per_link()) = rem_pages;
                TT_ASSERT(rem_pages_per_worker.at(worker_idx % all_gather_config.get_num_buffers_per_link()) * 2 <= cb_num_pages);
            }
        }

        std::vector<uint32_t> link_buffer_num_messages_to_send;
        std::vector<uint32_t> edm_semaphores_base_address;
        std::vector<uint32_t> link_buffer_sender_addresses;
        link_buffer_num_messages_to_send.reserve(all_gather_config.get_num_buffers_per_link());
        edm_semaphores_base_address.reserve(all_gather_config.get_num_buffers_per_link());
        link_buffer_sender_addresses.reserve(all_gather_config.get_num_buffers_per_link());
        for(uint32_t b = 0; b < all_gather_config.get_num_buffers_per_link(); ++b) {
            // link num messages
            link_buffer_num_messages_to_send.push_back(
                (num_full_chunks_per_worker.at(b) + (rem_pages_per_worker.at(b) > 0 ? 1 : 0)) *
                num_transfers);
            edm_semaphores_base_address.push_back(all_gather_config.get_eth_sems_l1_base_byte_address() + b * all_gather_config.get_semaphore_size());
            link_buffer_sender_addresses.push_back(all_gather_config.get_eth_buffers_l1_base_byte_address() + b * (all_gather_config.get_eth_buffer_size() + 16));
            TT_ASSERT((link_buffer_sender_addresses.back() + all_gather_config.get_eth_buffer_size()) % 16 == 0);
        }
        for(uint32_t b = 0; b < all_gather_config.get_num_buffers_per_link(); ++b) {
            log_trace(tt::LogOp, "rem_pages_per_worker[{}]: {}", b, rem_pages_per_worker.at(b));
            log_trace(tt::LogOp, "num_full_chunks_per_worker[{}]: {}", b, num_full_chunks_per_worker.at(b));
            log_trace(tt::LogOp, "link_buffer_num_messages_to_send[{}]: {}", b, link_buffer_num_messages_to_send.at(b));
        }

        std::vector<uint32_t> link_buffer_receiver_num_messages_to_send;
        std::vector<uint32_t> receiver_semaphores_base_address;
        std::vector<uint32_t> link_buffer_receiver_addresses;
        link_buffer_receiver_num_messages_to_send.reserve(all_gather_config.get_num_buffers_per_link());
        receiver_semaphores_base_address.reserve(all_gather_config.get_num_buffers_per_link());
        link_buffer_receiver_addresses.reserve(all_gather_config.get_num_buffers_per_link());
        for(uint32_t b = 0; b < all_gather_config.get_num_buffers_per_link(); ++b) {
            link_buffer_receiver_num_messages_to_send.push_back(
                (num_full_chunks_per_worker.at(b) + (rem_pages_per_worker.at(b) > 0 ? 1 : 0)) *
                num_transfers);
            receiver_semaphores_base_address.push_back(all_gather_config.get_eth_sems_l1_base_byte_address() + b * all_gather_config.get_semaphore_size());
            link_buffer_receiver_addresses.push_back(all_gather_config.get_eth_buffers_l1_base_byte_address() + b * (all_gather_config.get_eth_buffer_size() + 16));
            TT_ASSERT((link_buffer_sender_addresses.back() + all_gather_config.get_eth_buffer_size()) % 16 == 0);
        }


        std::vector<uint32_t> edm_clockwise_kernel_rt_args = {
            static_cast<uint32_t>(all_gather_config.get_erisc_handshake_address()),
            static_cast<uint32_t>(link_clockwise_sender_channels_offsets.at(i))
        };

        std::vector<uint32_t> edm_counter_clockwise_kernel_rt_args = {
            static_cast<uint32_t>(all_gather_config.get_erisc_handshake_address()),
            static_cast<uint32_t>(link_counter_clockwise_sender_channels_offsets.at(i))
        };

        for (uint32_t b = 0; b < all_gather_config.get_num_buffers_per_link(); ++b) {
            const uint32_t num_workers_per_channel = 1;
            // Setup sender direction args
            auto& edm_kernel_rt_args = all_gather_config.is_buffer_in_clockwise_ring(b) ? edm_clockwise_kernel_rt_args : edm_counter_clockwise_kernel_rt_args;
            // eth sender args
            // sender_buffer_address
            edm_kernel_rt_args.push_back(link_buffer_sender_addresses.at(b));

            // sender_num_messages_to_send
            edm_kernel_rt_args.push_back(link_buffer_num_messages_to_send.at(b));

            // sender_channel_size
            edm_kernel_rt_args.push_back(all_gather_config.get_eth_buffer_size());

            // edm_semaphores_base_address -> erisc L1 address
            edm_kernel_rt_args.push_back(eth_sem_addrs.at(b));

            // worker_semaphore_address
            edm_kernel_rt_args.push_back(sender_worker_writer_semaphore_addr);

            // sender_num_workers - only 1 per channel right now
            edm_kernel_rt_args.push_back(num_workers_per_channel);

            // for (uint32_t b = 0; b < sender_worker_cores.size(); ++b) {
            edm_kernel_rt_args.push_back((uint32_t)(
                (device->worker_core_from_logical_core(sender_worker_cores.at(b)).y << 16) |
                (device->worker_core_from_logical_core(sender_worker_cores.at(b)).x)
            ));
            log_trace(tt::LogOp, "sender_worker_writer_semaphore_addr: {}", sender_worker_writer_semaphore_addr);
            log_trace(tt::LogOp, "edm_kernel_rt_args worker.x: {}", device->worker_core_from_logical_core(sender_worker_cores.at(b)).y);
            log_trace(tt::LogOp, "edm_kernel_rt_args worker.y: {}", device->worker_core_from_logical_core(sender_worker_cores.at(b)).x);
        }

        // Setup receiver direction args. Clockwise receiver is same offset as sender offset for clockwise direction
        edm_clockwise_kernel_rt_args.push_back(static_cast<uint32_t>(link_counter_clockwise_receiver_channels_offsets.at(i)));

        edm_counter_clockwise_kernel_rt_args.push_back(static_cast<uint32_t>(link_clockwise_receiver_channels_offsets.at(i)));

        for (uint32_t b = 0; b < all_gather_config.get_num_buffers_per_link(); ++b) {

            const uint32_t num_workers_per_channel = 1;
            auto& edm_kernel_rt_args = all_gather_config.is_buffer_in_clockwise_ring(b) ? edm_counter_clockwise_kernel_rt_args : edm_clockwise_kernel_rt_args ;
            // eth receiver args
            // sender_buffer_address
            edm_kernel_rt_args.push_back(link_buffer_receiver_addresses.at(b));

            // sender_num_messages_to_send
            edm_kernel_rt_args.push_back(link_buffer_receiver_num_messages_to_send.at(b));

            // sender_channel_size
            edm_kernel_rt_args.push_back(all_gather_config.get_eth_buffer_size());

            // edm_semaphores_base_address -> erisc L1 address
            edm_kernel_rt_args.push_back(eth_sem_addrs.at(b));

            // worker_semaphore_address
            edm_kernel_rt_args.push_back(receiver_worker_semaphore_addr);

            // sender_num_workers - only 1 per channel right now
            edm_kernel_rt_args.push_back(num_workers_per_channel);

            edm_kernel_rt_args.push_back((uint32_t)(
                (device->worker_core_from_logical_core(receiver_worker_cores.at(b)).y << 16) |
                (device->worker_core_from_logical_core(receiver_worker_cores.at(b)).x)
            ));
        }

        // 1 Worker per buffer
        for (uint32_t b = 0; b < all_gather_config.get_num_buffers_per_link(); ++b) {
            uint32_t global_worker_index = all_gather_config.get_num_buffers_per_link() * i + b;

            bool is_clockwise_direction = all_gather_config.is_buffer_in_clockwise_ring(b);

            // Not fully sure about these two
            uint32_t last_output_page_offset = (num_transfers) * output_page_offset;
            uint32_t last_output_addr_offset = (num_transfers) * output_addr_offset;
            uint32_t receiver_ring_index = is_clockwise_direction ?
                (ring_index == 0 ? ring_size - 1 : ring_index - 1):
                (ring_index == ring_size - 1 ? 0 : ring_index + 1);

            uint32_t receiver_output_start_addr_offset = receiver_ring_index * output_addr_offset;

            uint32_t receiver_output_start_page_idx = output_start_page_idx;
            if (is_clockwise_direction) {
                bool is_wraparound_ring_index = ring_index == 0;
                if (is_wraparound_ring_index) {
                    receiver_output_start_page_idx += last_output_page_offset;
                } else {
                    receiver_output_start_page_idx -= output_page_offset;
                }
            } else {
                // counter clockwise direction
                bool is_wraparound_ring_index = ring_index == ring_size - 1;
                if (is_wraparound_ring_index) {
                    receiver_output_start_page_idx -= last_output_page_offset;
                } else {
                    receiver_output_start_page_idx += output_page_offset;
                }
            }

            log_trace(tt::LogOp,"Counter Clock-wise");
            log_trace(tt::LogOp,"\tlast_output_page_offset={}", last_output_page_offset);
            log_trace(tt::LogOp,"\tlast_output_addr_offset={}", last_output_addr_offset);
            log_trace(tt::LogOp,"\treceiver_ring_index={}", receiver_ring_index);
            log_trace(tt::LogOp,"\treceiver_output_start_addr_offset={}", receiver_output_start_addr_offset);
            log_trace(tt::LogOp,"\treceiver_output_start_page_idx={}", receiver_output_start_page_idx);

            // Sender Worker Kernels
            log_trace(tt::LogOp, "HOST RWS ARGS: ");
            log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);
            log_trace(tt::LogOp, "\tnum_full_chunks_per_worker.at(b): {}", num_full_chunks_per_worker.at(b));
            log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
            log_trace(tt::LogOp, "\toutput_page_size: {}", output_page_size);
            log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer.at(b): {}", pages_per_eth_l1_buffer.at(b));
            log_trace(tt::LogOp, "\trem_pages_per_worker.at(b): {}", rem_pages_per_worker.at(b));

            //// Send Reader
            auto build_worker_send_reader_ct_args = [&]() {
                if (is_sharded) {
                    // # Send Reader (CT)
                    // 1) Shard Type
                    // 2) num_transfers
                    std::vector<uint32_t> worker_reader_sender_ct_args = {
                        static_cast<uint32_t>(sharding_info.get_shard_type()),
                        static_cast<uint32_t>(num_transfers)
                    };
                    log_trace(tt::LogOp, "----worker_reader_sender_ct_args size={}", worker_reader_sender_ct_args.size());
                    log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());
                    log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);

                    return worker_reader_sender_ct_args;
                } else {
                    std::vector<uint32_t> worker_reader_sender_ct_args = {
                        static_cast<uint32_t>(all_gather_config.is_input_dram()),
                        static_cast<uint32_t>(all_gather_config.is_output_dram()),
                        static_cast<uint32_t>(num_transfers),
                        static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                        static_cast<uint32_t>(input_page_size),
                        static_cast<uint32_t>(output_page_size),
                        static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),
                        static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                        static_cast<uint32_t>(input_start_page_idx),
                        static_cast<uint32_t>(output_start_page_idx),
                        static_cast<uint32_t>(output_start_addr_offset),
                        static_cast<uint32_t>(row_idx),
                        static_cast<uint32_t>(col_idx),
                        static_cast<uint32_t>(row_offset),
                        static_cast<uint32_t>(col_offset),
                        static_cast<uint32_t>(num_rows),
                        static_cast<uint32_t>(num_cols),
                        static_cast<uint32_t>(last_output_page_offset),
                        static_cast<uint32_t>(output_page_offset),
                        static_cast<uint32_t>(last_output_addr_offset),
                        static_cast<uint32_t>(output_addr_offset),
                        static_cast<uint32_t>(ring_index),
                        static_cast<uint32_t>(sender_worker_reader_semaphore_addr),
                        static_cast<uint32_t>(is_clockwise_direction ? 1 : 0),
                        static_cast<uint32_t>(cb_num_pages / 2)
                    };
                    return worker_reader_sender_ct_args;
                }
            };

            std::vector<uint32_t> const& worker_send_reader_ct_args = build_worker_send_reader_ct_args();

            auto build_worker_send_reader_rt_args = [&]() {
                bool is_clockwise = all_gather_config.is_buffer_in_clockwise_ring(b);
                if (is_sharded) {
                    // # Send Reader (RT)
                    // 1) local semaphore address (same as before)
                    // 2) input tensor shard reader
                    // 3) output tensor shard reader
                    auto curr_link = i;

                    TT_ASSERT(all_gather_config.get_num_buffers_per_link() == 1 || all_gather_config.get_num_buffers_per_link() == 2 || all_gather_config.get_num_buffers_per_link() == 4 || all_gather_config.get_num_buffers_per_link() == 8);
                    TT_ASSERT(input_tensor.buffer() != nullptr);
                    auto input_tensor_shard_arg_generator =
                        InputTensorShardAddrGenArgGenerator(
                            device,
                            input_tensor,
                            ring_index,
                            ring_size,
                            global_num_workers,
                            global_worker_index,
                            0,
                            0,
                            is_clockwise);
                    auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                        all_gather_config, input_tensor, output_tensor,
                        is_clockwise ?
                            (ring_index == 0 ? ring_size - 1 : ring_index - 1) :
                            (ring_index == ring_size - 1 ? 0 : ring_index + 1),
                        global_worker_index);

                    log_trace(tt::LogOp, "SendReader {} ring_index: {}, start dest worker index: {}, starting chunk into shard: {}", global_worker_index, ring_index, starting_dest_worker_index, starting_chunk_into_shard);
                    auto output_tensor_shard_arg_generator =
                        OutputTensorShardAddrGenArgGenerator(
                            all_gather_config,
                            device,
                            input_tensor,
                            output_tensor,
                            ring_index,
                            ring_size,
                            global_num_workers,
                            global_worker_index,
                            starting_dest_worker_index,
                            starting_chunk_into_shard,
                            is_clockwise);

                    auto const& input_shard_addr_generator_args = input_tensor_shard_arg_generator.generate();
                    auto const& output_shard_addr_generator_args = output_tensor_shard_arg_generator.generate();
                    std::vector<uint32_t> worker_send_reader_rt_args;
                    worker_send_reader_rt_args.reserve(1 + input_shard_addr_generator_args.size() + output_shard_addr_generator_args.size());
                    worker_send_reader_rt_args.push_back(sender_worker_reader_semaphore_addr);
                    std::copy(input_shard_addr_generator_args.begin(), input_shard_addr_generator_args.end(), std::back_inserter(worker_send_reader_rt_args));
                    std::copy(output_shard_addr_generator_args.begin(), output_shard_addr_generator_args.end(), std::back_inserter(worker_send_reader_rt_args));

                    log_trace(tt::LogOp, "---worker_send_reader_rt_args.size()={}-----", worker_send_reader_rt_args.size());
                    log_trace(tt::LogOp, "\tsender_worker_reader_semaphore_addr: {}", sender_worker_reader_semaphore_addr);
                    log_trace(tt::LogOp, "\tinput_shard_addr_generator_args:");
                    input_tensor_shard_arg_generator.dump_to_log();
                    log_trace(tt::LogOp, "\toutput_tensor_shard_arg_generator:");
                    output_tensor_shard_arg_generator.dump_to_log();

                    return worker_send_reader_rt_args;
                } else {
                    std::vector<uint32_t> args = {
                        static_cast<uint32_t>(input_buffer->address()),
                        static_cast<uint32_t>(output_buffer->address())
                    };
                    return args;
                }
            };
            std::vector<uint32_t> const& worker_send_reader_rt_args = build_worker_send_reader_rt_args();

            std::string const& send_reader_kernel_path = is_sharded ?
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_send_reader.cpp" :
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_send_reader.cpp";
            KernelHandle worker_reader_sender_kernel_id = tt_metal::CreateKernel(
                program,
                send_reader_kernel_path,
                sender_worker_cores.at(b),
                tt_metal::ReaderDataMovementConfig(worker_send_reader_ct_args, worker_defines));

            worker_reader_sender_kernels.push_back(worker_reader_sender_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_reader_sender_kernel_id,
                sender_worker_cores.at(b),
                worker_send_reader_rt_args);


            //// Send Writer
            auto build_worker_sender_writer_ct_args = [&]() {
                if (is_sharded) {
                    std::vector<uint32_t> worker_sender_writer_ct_args = {
                        static_cast<uint32_t>(sharding_info.get_shard_type())
                    };
                    log_trace(tt::LogOp, "----worker_sender_writer_ct_args size={}", worker_sender_writer_ct_args.size());
                    log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());

                    return worker_sender_writer_ct_args;
                } else {
                    CoreCoord const& worker_eth_sender_core = is_clockwise_direction ? eth_sender_cores.at(i) : eth_receiver_cores.at(i);
                    std::vector<uint32_t> worker_writer_sender_ct_args = {
                        static_cast<uint32_t>(all_gather_config.is_output_dram()),
                        static_cast<uint32_t>(num_transfers),
                        static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                        static_cast<uint32_t>(input_page_size),
                        static_cast<uint32_t>(output_page_size),
                        static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),
                        static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                        static_cast<uint32_t>(input_start_page_idx),
                        static_cast<uint32_t>(output_start_page_idx),
                        static_cast<uint32_t>(output_start_addr_offset),
                        static_cast<uint32_t>(row_idx),
                        static_cast<uint32_t>(col_idx),
                        static_cast<uint32_t>(row_offset),
                        static_cast<uint32_t>(col_offset),
                        static_cast<uint32_t>(num_rows),
                        static_cast<uint32_t>(num_cols),
                        static_cast<uint32_t>(ring_index),

                        // worker local L1 address of semaphore
                        static_cast<uint32_t>(sender_worker_writer_semaphore_addr),
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).x),
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).y),
                        static_cast<uint32_t>(cb_num_pages / 2),
                    };

                    log_trace(tt::LogOp, "HOST SWS ARGS:");
                    log_trace(tt::LogOp, "\toutput_is_dram: {}", all_gather_config.is_output_dram());
                    log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);
                    log_trace(tt::LogOp, "\tnum_full_chunks_per_worker.at(b): {}", num_full_chunks_per_worker.at(b));
                    log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
                    log_trace(tt::LogOp, "\toutput_page_size: {}", output_page_size);
                    log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer.at(b): {}", pages_per_eth_l1_buffer.at(b));
                    log_trace(tt::LogOp, "\trem_pages_per_worker.at(b): {}", rem_pages_per_worker.at(b));

                    return worker_writer_sender_ct_args;
                }
            };

            std::vector<uint32_t> const& worker_sender_writer_ct_args = build_worker_sender_writer_ct_args();

            auto build_worker_sender_writer_rt_args = [&]() {
                if (is_sharded) {
                    // Send Writer Args (RT)
                    // 1) eth_sender_l1_base_addr
                    // 2) eth_sender_l1_sem_addr
                    // 3) eth_sender_noc_x
                    // 4) eth_sender_noc_y
                    // 5) writer_send_sem_addr
                    // 6) num_transfers
                    // 7)

                    bool is_clockwise = all_gather_config.is_buffer_in_clockwise_ring(b);
                    auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                        all_gather_config,
                        input_tensor,
                        output_tensor,
                        // this writes the input tensor to the first output location
                        ring_index,
                        global_worker_index);
                    auto output_tensor_shard_arg_generator =
                        OutputTensorShardAddrGenArgGenerator(
                            all_gather_config,
                            device,
                            input_tensor,
                            output_tensor,
                            ring_index,
                            ring_size,
                            global_num_workers,
                            global_worker_index,
                            starting_dest_worker_index,
                            starting_chunk_into_shard,
                            all_gather_config.is_buffer_in_clockwise_ring(b)
                        );
                    auto const& output_tensor_shard_addr_gen_args = output_tensor_shard_arg_generator.generate();

                    CoreCoord const& worker_eth_sender_core = is_clockwise_direction ? eth_sender_cores.at(i) : eth_receiver_cores.at(i);
                    std::vector<uint32_t> worker_writer_sender_rt_args = {
                        static_cast<uint32_t>(eth_buffer_addrs.at(b)), // eth_sender_l1_base_addr
                        static_cast<uint32_t>(eth_sem_addrs.at(b)), // eth_sender_l1_sem_addr
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).x), // eth_sender_noc_x
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).y), // eth_sender_noc_y
                        static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)), //output_tensor_shard_arg_generator.args_struct.num_dest_cores),//pages_per_eth_l1_buffer.at(b)),
                        static_cast<uint32_t>(sender_worker_writer_semaphore_addr), // writer_send_sem_addr
                        static_cast<uint32_t>(num_transfers) // num_transfers
                    };
                    std::copy(output_tensor_shard_addr_gen_args.begin(), output_tensor_shard_addr_gen_args.end(), std::back_inserter(worker_writer_sender_rt_args));

                    // Currently the kernel assumes we don't need to break up the initial local tensor send to EDM into multiple
                    // chunks

                    log_trace(tt::LogOp, "----worker_writer_sender_rt_args size={}", worker_writer_sender_rt_args.size());
                    log_trace(tt::LogOp, "\teth_sender_l1_base_addr: {}", eth_buffer_addrs.at(b));
                    log_trace(tt::LogOp, "\teth_sender_l1_sem_addr: {}", eth_sem_addrs.at(b));
                    log_trace(tt::LogOp, "\teth_sender_noc_x: {}", device->ethernet_core_from_logical_core(worker_eth_sender_core).x);
                    log_trace(tt::LogOp, "\teth_sender_noc_y: {}", device->ethernet_core_from_logical_core(worker_eth_sender_core).y);
                    log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer: {}", pages_per_eth_l1_buffer.at(b));
                    log_trace(tt::LogOp, "\twriter_send_sem_addr: {}", sender_worker_writer_semaphore_addr);
                    log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);
                    output_tensor_shard_arg_generator.dump_to_log();

                    return worker_writer_sender_rt_args;
                } else {
                    std::vector<uint32_t> worker_writer_sender_rt_args = {
                        static_cast<uint32_t>(output_buffer->address()),
                        static_cast<uint32_t>(eth_buffer_addrs.at(b)),
                        static_cast<uint32_t>(eth_sem_addrs.at(b))
                    };
                    return worker_writer_sender_rt_args;
                }
            };
            std::vector<uint32_t> const& worker_sender_writer_rt_args = build_worker_sender_writer_rt_args();

            std::string const& sender_writer_kernel_path = is_sharded ?
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_send_writer.cpp" :
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_send_writer.cpp";
            KernelHandle worker_sender_writer_kernel_id = tt_metal::CreateKernel(
                program,
                sender_writer_kernel_path,
                sender_worker_cores.at(b),
                tt_metal::WriterDataMovementConfig(worker_sender_writer_ct_args, worker_defines));

            worker_writer_sender_kernels.push_back(worker_sender_writer_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_sender_writer_kernel_id,
                sender_worker_cores.at(b),
                worker_sender_writer_rt_args);

            log_trace(tt::LogOp, "HOST RWR ARGS:");
            log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);
            log_trace(tt::LogOp, "\tnum_full_chunks_per_worker.at(b): {}", num_full_chunks_per_worker.at(b));
            log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
            log_trace(tt::LogOp, "\tpages_per_chunk: {}", pages_per_chunk);
            log_trace(tt::LogOp, "\trem_pages_per_worker.at(b): {}", rem_pages_per_worker.at(b));

            //// Receive Reader
            auto build_worker_receiver_reader_ct_args = [&]() {
                if (is_sharded) {
                    // Receiver Reader Args (CT)
                    std::vector<uint32_t> worker_receiver_reader_ct_args = {
                        static_cast<uint32_t>(sharding_info.get_shard_type())
                    };
                    log_trace(tt::LogOp, "----worker_receiver_reader_ct_args size={}", worker_receiver_reader_ct_args.size());
                    log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());

                    return worker_receiver_reader_ct_args;
                } else {
                    CoreCoord const& worker_eth_receiver_core = is_clockwise_direction ? eth_receiver_cores.at(i) : eth_sender_cores.at(i);
                    std::vector<uint32_t> worker_receiver_reader_ct_args = {
                        static_cast<uint32_t>(num_transfers),
                        static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                        static_cast<uint32_t>(input_page_size),
                        static_cast<uint32_t>(pages_per_chunk),
                        static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).x),
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).y),
                        static_cast<uint32_t>(eth_sem_addrs.at(b)),
                        static_cast<uint32_t>(receiver_worker_semaphore_addr),
                        static_cast<uint32_t>(cb_num_pages / 2)
                    };
                    return worker_receiver_reader_ct_args;
                }
            };
            std::vector<uint32_t> const& worker_receiver_reader_ct_args = build_worker_receiver_reader_ct_args();

            auto build_worker_receiver_reader_rt_args = [&]() {
                if (is_sharded) {
                    // Receiver Reader Args (RT)
                    // 1) eth_receiver_noc_x
                    // 2) eth_receiver_noc_y
                    // 3) eth_receiver_l1_base_addr
                    // 4) eth_receiver_l1_semaphore_addr
                    // 5) (local) receiver_read_sem_addr
                    // 6) output tensor shard addr gen
                    auto curr_link = i;
                    bool is_clockwise = all_gather_config.is_buffer_in_clockwise_ring(b);
                    auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                        all_gather_config,
                        input_tensor,
                        output_tensor,
                        is_clockwise ?
                            (ring_index == 0 ? ring_size - 1 : ring_index - 1) :
                            (ring_index == ring_size - 1 ? 0 : ring_index + 1), // ring_index
                        global_worker_index);
                    CoreCoord const& worker_eth_receiver_core = is_clockwise_direction ? eth_receiver_cores.at(i) : eth_sender_cores.at(i);
                    auto output_tensor_shard_arg_generator =
                        OutputTensorShardAddrGenArgGenerator(
                            all_gather_config,
                            device,
                            input_tensor,
                            output_tensor,
                            ring_index,
                            ring_size,
                            global_num_workers,
                            global_worker_index,
                            starting_dest_worker_index,
                            starting_chunk_into_shard,
                            all_gather_config.is_buffer_in_clockwise_ring(b));
                    auto const& output_tensor_shard_addr_gen_args = output_tensor_shard_arg_generator.generate();
                    std::vector<uint32_t> worker_reader_receiver_rt_args;
                    worker_reader_receiver_rt_args.reserve(7 + output_tensor_shard_addr_gen_args.size());

                    worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).x)); // eth_receiver_noc_x
                    worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).y)); // eth_receiver_noc_y
                    worker_reader_receiver_rt_args.push_back(eth_buffer_addrs.at(b)); // eth_receiver_l1_base_addr
                    worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(eth_sem_addrs.at(b))); // eth_receiver_l1_semaphore_addr
                    worker_reader_receiver_rt_args.push_back(receiver_worker_semaphore_addr); // local_receiver_read_sem_addr
                    worker_reader_receiver_rt_args.push_back(pages_per_eth_l1_buffer.at(b)), //output_tensor_shard_arg_generator.args_struct.num_dest_cores), //pages_per_eth_l1_buffer.at(b)); // num_shards_per_eth_buf
                    worker_reader_receiver_rt_args.push_back(num_transfers); // local_receiver_read_sem_addr
                    std::copy(output_tensor_shard_addr_gen_args.begin(), output_tensor_shard_addr_gen_args.end(), std::back_inserter(worker_reader_receiver_rt_args));

                    log_trace(tt::LogOp, "----worker_receiver_reader_ct_args size={}", worker_receiver_reader_ct_args.size());
                    log_trace(tt::LogOp, "\teth_receiver_noc_x: {}", static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).x));
                    log_trace(tt::LogOp, "\teth_receiver_noc_y: {}", static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).y));
                    log_trace(tt::LogOp, "\teth_receiver_l1_base_addr: {}", eth_buffer_addrs.at(b));
                    log_trace(tt::LogOp, "\teth_receiver_l1_semaphore_addr: {}", static_cast<uint32_t>(eth_sem_addrs.at(b)));
                    log_trace(tt::LogOp, "\tlocal_receiver_read_sem_addr: {}", receiver_worker_semaphore_addr);
                    log_trace(tt::LogOp, "\tnum_shards_per_eth_buf: {}", pages_per_eth_l1_buffer.at(b));

                    output_tensor_shard_arg_generator.dump_to_log();

                    return worker_reader_receiver_rt_args;
                } else {
                    std::vector<uint32_t> worker_reader_receiver_rt_args = {
                        static_cast<uint32_t>(eth_buffer_addrs.at(b))
                    };
                    return worker_reader_receiver_rt_args;
                }
            };
            std::vector<uint32_t> worker_receiver_reader_rt_args = build_worker_receiver_reader_rt_args();

            std::string const& receiver_reader_kernel_path = is_sharded ?
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_receive_reader.cpp" :
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_receive_reader.cpp";
            KernelHandle worker_receiver_reader_kernel_id = tt_metal::CreateKernel(
                program,
                receiver_reader_kernel_path,
                receiver_worker_cores.at(b),
                tt_metal::ReaderDataMovementConfig(worker_receiver_reader_ct_args, worker_defines));

            worker_reader_receiver_kernels.push_back(worker_receiver_reader_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_receiver_reader_kernel_id,
                receiver_worker_cores.at(b),
                worker_receiver_reader_rt_args);

            log_trace(tt::LogOp, "HOST SWR ARGS: \n");
            log_trace(tt::LogOp, "\toutput_is_dram: {}", all_gather_config.is_output_dram());
            log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);
            log_trace(tt::LogOp, "\tnum_full_chunks_per_worker.at(b): {}", num_full_chunks_per_worker.at(b));
            log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
            log_trace(tt::LogOp, "\toutput_page_size: {}", output_page_size);
            log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer.at(b): {}", pages_per_eth_l1_buffer.at(b));
            log_trace(tt::LogOp, "\trem_pages_per_worker.at(b): {}", rem_pages_per_worker.at(b));
            log_trace(tt::LogOp, "\treceiver_output_start_page_idx: {}", receiver_output_start_page_idx);
            log_trace(tt::LogOp, "\treceiver_output_start_addr_offset: {}", receiver_output_start_addr_offset);

            //// Receive Writer
            auto build_worker_receive_writer_ct_args = [&]() {
                if (is_sharded) {
                    // # Receiver Writer (CT)
                    // 1) Shard Type
                    std::vector<uint32_t> worker_receive_writer_ct_args = {
                        static_cast<uint32_t>(sharding_info.get_shard_type())
                    };
                    log_trace(tt::LogOp, "----worker_receive_writer_ct_args size={}", worker_receive_writer_ct_args.size());
                    log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());

                    return worker_receive_writer_ct_args;
                } else {
                    std::vector<uint32_t> worker_writer_receiver_ct_args = {
                        static_cast<uint32_t>(all_gather_config.is_output_dram()),
                        static_cast<uint32_t>(num_transfers),
                        static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                        static_cast<uint32_t>(input_page_size),
                        static_cast<uint32_t>(output_page_size),
                        static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),
                        static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                        static_cast<uint32_t>(receiver_output_start_page_idx),
                        static_cast<uint32_t>(receiver_output_start_addr_offset),
                        static_cast<uint32_t>(row_idx),
                        static_cast<uint32_t>(col_idx),
                        static_cast<uint32_t>(row_offset),
                        static_cast<uint32_t>(col_offset),
                        static_cast<uint32_t>(num_rows),
                        static_cast<uint32_t>(num_cols),
                        static_cast<uint32_t>(last_output_page_offset),
                        static_cast<uint32_t>(output_page_offset),
                        static_cast<uint32_t>(last_output_addr_offset),
                        static_cast<uint32_t>(output_addr_offset),
                        static_cast<uint32_t>(receiver_ring_index),
                        static_cast<uint32_t>(sender_worker_reader_semaphore_addr),
                        static_cast<uint32_t>(is_clockwise_direction ? 1 : 0),
                        static_cast<uint32_t>(cb_num_pages / 2)
                    };
                    return worker_writer_receiver_ct_args;
                }
            };
            std::vector<uint32_t> const& worker_receive_writer_ct_args = build_worker_receive_writer_ct_args();

            auto build_worker_receive_writer_rt_args = [&]() {
                auto worker_sender_reader = device->worker_core_from_logical_core(sender_worker_cores.at(b));
                if (is_sharded) {
                    // # Receiver Writer (RT)
                    // 1) Remote sender reader semaphore address
                    // 2) Output tensor Writer shard addr gen
                    bool is_clockwise = all_gather_config.is_buffer_in_clockwise_ring(b);

                    auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                        all_gather_config,
                        input_tensor,
                        output_tensor,
                        is_clockwise ?
                            (ring_index == 0 ? ring_size - 1 : ring_index - 1) :
                            (ring_index == ring_size - 1 ? 0 : ring_index + 1),
                        global_worker_index);
                    log_trace(tt::LogOp, "ReceiverWriter {} ring_index: {}, start dest worker index: {}, starting chunk into shard: {}", global_worker_index, ring_index, starting_dest_worker_index, starting_chunk_into_shard);
                    OutputTensorShardAddrGenArgGenerator output_tensor_shard_arg_generator(
                        all_gather_config,
                        device,
                        input_tensor,
                        output_tensor,
                        ring_index,
                        ring_size,
                        global_num_workers,
                        all_gather_config.get_num_buffers_per_link() * i + b,
                        starting_dest_worker_index,
                        starting_chunk_into_shard,
                        all_gather_config.is_buffer_in_clockwise_ring(b));
                    auto const& output_shard_addr_generator_args = output_tensor_shard_arg_generator.generate();
                    std::vector<uint32_t> worker_receive_writer_rt_args;
                    worker_receive_writer_rt_args.reserve(5 + output_shard_addr_generator_args.size());
                    worker_receive_writer_rt_args.push_back(static_cast<uint32_t>(worker_sender_reader.x));
                    worker_receive_writer_rt_args.push_back(static_cast<uint32_t>(worker_sender_reader.y));
                    worker_receive_writer_rt_args.push_back(sender_worker_reader_semaphore_addr);

                    worker_receive_writer_rt_args.push_back(output_tensor_shard_arg_generator.args_struct.num_dest_cores), //pages_per_eth_l1_buffer.at(b));
                    worker_receive_writer_rt_args.push_back(num_transfers);

                    std::copy(output_shard_addr_generator_args.begin(), output_shard_addr_generator_args.end(), std::back_inserter(worker_receive_writer_rt_args));

                    log_trace(tt::LogOp, "----worker_receive_writer_rt_args size={}", worker_receive_writer_rt_args.size());
                    log_trace(tt::LogOp, "\tsender_worker_reader_semaphore_addr: {}", sender_worker_reader_semaphore_addr);
                    output_tensor_shard_arg_generator.dump_to_log();

                    return worker_receive_writer_rt_args;
                } else {
                    std::vector<uint32_t> worker_writer_receiver_rt_args = {
                        static_cast<uint32_t>(output_buffer->address()),
                        static_cast<uint32_t>(worker_sender_reader.x),
                        static_cast<uint32_t>(worker_sender_reader.y),
                    };
                    return worker_writer_receiver_rt_args;
                }
            };
            std::vector<uint32_t> worker_receive_writer_rt_args = build_worker_receive_writer_rt_args();

            std::string const& receiver_writer_kernel_path = is_sharded ?
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_receive_writer.cpp" :
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_receive_writer.cpp";
            KernelHandle worker_receive_writer_kernel_id = tt_metal::CreateKernel(
                program,
                receiver_writer_kernel_path,
                receiver_worker_cores.at(b),
                tt_metal::WriterDataMovementConfig(worker_receive_writer_ct_args, worker_defines));

            worker_writer_receiver_kernels.push_back(worker_receive_writer_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_receive_writer_kernel_id,
                receiver_worker_cores.at(b),
                worker_receive_writer_rt_args);

            uint32_t pages_per_worker = num_full_chunks_per_worker.at(b) * pages_per_chunk + rem_pages_per_worker.at(b);
            if (is_sharded) {
                // nothing to do here - is handled by
            } else {
                if (rm) {
                    uint32_t num_rows_shifted = row_idx + pages_per_worker;
                    uint32_t num_blocks_shifted = width ? 0 : num_rows_shifted / num_rows;
                    output_start_page_idx += pages_per_worker + num_blocks_shifted * row_offset;
                    row_idx = width ? 0 : num_rows_shifted % num_rows;
                } else {
                    uint32_t num_cols_shifted = col_idx + pages_per_worker;
                    uint32_t num_rows_shifted = num_cols_shifted / num_cols;
                    uint32_t num_blocks_shifted = width ? 0 : num_rows_shifted / num_rows;
                    output_start_page_idx += pages_per_worker + num_rows_shifted * col_offset + num_blocks_shifted * row_offset;
                    col_idx = num_cols_shifted % num_cols;
                    row_idx = width ? 0 : num_rows_shifted % num_rows;
                }
                input_start_page_idx += pages_per_worker;
            }
        }

        // Ethernet Kernels
        std::vector<uint32_t> eth_sender_ct_args = {
            static_cast<uint32_t>(all_gather_config.get_num_buffers_in_clockwise_direction() ? 1 : 0),
            static_cast<uint32_t>(all_gather_config.get_num_buffers_in_counter_clockwise_direction() ? 1 : 0),
            static_cast<uint32_t>(link_clockwise_sender_num_channels.at(i)),
            static_cast<uint32_t>(link_counter_clockwise_receiver_num_channels.at(i))
        };

        log_trace(tt::LogOp, "EDM sender side link_clockwise_sender_num_channels.at(i) {}", link_clockwise_sender_num_channels.at(i));
        log_trace(tt::LogOp, "EDM sender side link_counter_clockwise_receiver_num_channels.at(i) {}", link_counter_clockwise_receiver_num_channels.at(i));

        auto eth_sender_kernel = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/ccl/edm/erisc_datamover.cpp",
            eth_sender_cores.at(i),
            tt_metal::EthernetConfig{.noc=sender_noc, .compile_args=eth_sender_ct_args});


        tt_metal::SetRuntimeArgs(
            program,
            eth_sender_kernel,
            eth_sender_cores.at(i),
            edm_clockwise_kernel_rt_args);

        eth_sender_kernels.push_back(eth_sender_kernel);

        std::vector<uint32_t> eth_receiver_ct_args = {
            static_cast<uint32_t>(all_gather_config.get_num_buffers_in_counter_clockwise_direction() ? 1 : 0),
            static_cast<uint32_t>(all_gather_config.get_num_buffers_in_clockwise_direction() ? 1 : 0),
            static_cast<uint32_t>(link_counter_clockwise_sender_num_channels.at(i)),
            static_cast<uint32_t>(link_clockwise_receiver_num_channels.at(i))
        };

        auto eth_receiver_kernel = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/ccl/edm/erisc_datamover.cpp",
            eth_receiver_cores.at(i),
            tt_metal::EthernetConfig{.noc=receiver_noc, .compile_args=eth_receiver_ct_args});

        eth_receiver_kernels.push_back(eth_receiver_kernel);

        log_trace(tt::LogOp, "RingIndex: {}. Link {}. Clockwise EDM Core (x={},y={}), Counter-clockwise EDM Core (x={},y={})", ring_index, i, eth_sender_cores.at(i).x, eth_sender_cores.at(i).y, eth_receiver_cores.at(i).x, eth_receiver_cores.at(i).y);

        if (enable_print) {
            std::stringstream ss;
            ss << "HOST SENDER EDM ARGS:\n";
            for (auto const& s : edm_clockwise_kernel_rt_args) {
                ss << "\t" << s << "\n";
            }
            std::cout << ss.str() << std::endl;
        }
        if (enable_print) {
            std::stringstream ss;
            ss << "HOST RECEIVER EDM ARGS:\n";
            for (auto const& s : edm_counter_clockwise_kernel_rt_args) {
                ss << "\t" << s << "\n";
            }
            std::cout << ss.str() << std::endl;
        }


        tt_metal::SetRuntimeArgs(
            program,
            eth_receiver_kernel,
            eth_receiver_cores.at(i),
            edm_counter_clockwise_kernel_rt_args);

        if (receiver_device_id == sender_device_id) {
            receiver_socket_idx += 2;
            sender_socket_idx += 2;
        } else {
            receiver_socket_idx += 1;
            sender_socket_idx += 1;
        }
    }

    auto override_runtime_arguments_callback = [num_links, total_worker_core_pairs_used, worker_reader_sender_kernels, worker_writer_sender_kernels, worker_reader_receiver_kernels, worker_writer_receiver_kernels, all_worker_sender_cores, all_worker_receiver_cores] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        for (uint32_t i = 0; i < total_worker_core_pairs_used; ++i) {
            auto &worker_reader_sender_runtime_args = GetRuntimeArgs(program, worker_reader_sender_kernels.at(i), all_worker_sender_cores.at(i));
            worker_reader_sender_runtime_args.at(0) = input.buffer()->address();
            worker_reader_sender_runtime_args.at(1) = output.buffer()->address();
            auto &worker_writer_sender_runtime_args = GetRuntimeArgs(program, worker_writer_sender_kernels.at(i), all_worker_sender_cores.at(i));
            worker_writer_sender_runtime_args.at(0) = output.buffer()->address();

            auto &worker_writer_receiver_runtime_args = GetRuntimeArgs(program, worker_writer_receiver_kernels.at(i), all_worker_receiver_cores.at(i));
            worker_writer_receiver_runtime_args.at(0) = output.buffer()->address();
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
