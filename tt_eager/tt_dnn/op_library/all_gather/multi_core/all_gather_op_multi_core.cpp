// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "eth_l1_address_map.h"
#include "tensor/tensor_impl.hpp"
#include "tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include <sstream>

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks all_gather_multi_core_with_workers(const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t num_links, const uint32_t ring_size, const uint32_t ring_index, const chip_id_t receiver_device_id, const chip_id_t sender_device_id) {

    tt_metal::Program program{};

    const uint32_t max_buffer_per_chunk = round_down(all_gather_buffer_params::eth_buffer_size, input_tensor.buffer()->page_size());
    const uint32_t max_pages_per_chunk = max_buffer_per_chunk / input_tensor.buffer()->page_size();
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

    std::map<string, string> worker_defines;
    if (rm) {
        worker_defines["RM_INTERLEAVED"] = "1";
    } else {
        worker_defines["TILE_INTERLEAVED"] = "1";
    }

    // number of worker cores is 2x this since there is 1 worker for the sender buffer and 1 worker for the receiver buffer
    uint32_t total_worker_core_pairs_used = num_links * all_gather_buffer_params::num_buffers;
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


    uint32_t num_input_pages = input_tensor.buffer()->size() / input_tensor.buffer()->page_size();
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

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    bool output_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    uint32_t input_page_size = input_buffer->page_size();
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
    uint32_t last_output_page_offset = (num_transfers) * output_page_offset;
    uint32_t last_output_addr_offset = (num_transfers) * output_addr_offset;
    uint32_t receiver_ring_index = ring_index == 0 ? num_transfers : ring_index - 1;
    uint32_t receiver_output_start_addr_offset = receiver_ring_index * output_addr_offset;

    std::vector<uint32_t> link_sender_channels_offsets = std::vector<uint32_t>(num_links, 0);
    std::vector<uint32_t> link_sender_num_channels = std::vector<uint32_t>(num_links, all_gather_buffer_params::num_buffers);
    std::vector<uint32_t> link_receiver_channels_offsets = std::vector<uint32_t>(num_links, 0);
    std::vector<uint32_t> link_receiver_num_channels = std::vector<uint32_t>(num_links, all_gather_buffer_params::num_buffers);
    std::vector<uint32_t> eth_sem_addrs;
    std::vector<uint32_t> eth_buffer_addrs;
    eth_sem_addrs.reserve(all_gather_buffer_params::num_buffers);
    eth_buffer_addrs.reserve(all_gather_buffer_params::num_buffers);
    for (uint32_t b = 0, eth_sem_addr = all_gather_buffer_params::eth_sem_l1_byte_address, eth_buffer_addr = all_gather_buffer_params::eth_buffer_l1_byte_address; b < all_gather_buffer_params::num_buffers; ++b) {
        eth_sem_addrs.push_back(eth_sem_addr);
        eth_sem_addr += all_gather_buffer_params::semaphore_size;
        eth_buffer_addrs.push_back(eth_buffer_addr);
        eth_buffer_addr += all_gather_buffer_params::eth_buffer_size;
    }

    for (uint32_t i = 0; i < num_links; ++i) {
        // We can't have overlap between the mcast grid for worker cores for different links since mcasting the semaphore in receiver would corrupt other link semaphores
        // We can have overlap between a link's sender and receiver worker grids if we have the semaphores at different addresses
        auto receiver_workers = CoreRange({0, i}, {all_gather_buffer_params::num_buffers - 1, i});
        auto sender_workers = CoreRange({all_gather_buffer_params::num_buffers, i}, {2 * all_gather_buffer_params::num_buffers - 1, i});
        worker_receiver_cores.push_back(receiver_workers);
        worker_sender_cores.push_back(sender_workers);

        // Circular Buffer Setup
        // TODO: Optimize mem usage
        uint32_t cb_page_size = input_page_size;
        uint32_t num_input_pages = 2 * max_pages_per_chunk;
        uint32_t src0_cb_index = CB::c_in0;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_pages * cb_page_size, {{src0_cb_index, df}})
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
        pages_per_buffer.reserve(all_gather_buffer_params::num_buffers);
        // std::cout << "all_gather_buffer_params::eth_buffer_size=" << all_gather_buffer_params::eth_buffer_size << std::endl;
        // std::cout << "input_tensor.buffer()->page_size()=" << input_tensor.buffer()->page_size() << std::endl;
        uint32_t pages_per_eth_l1_sender_buffer = all_gather_buffer_params::eth_buffer_size / input_tensor.buffer()->page_size();
        for(uint32_t b = 0; b < all_gather_buffer_params::num_buffers; ++b) {
            pages_per_buffer.push_back((pages_per_link.at(i) / all_gather_buffer_params::num_buffers));
            pages_per_eth_l1_buffer.push_back(pages_per_eth_l1_sender_buffer);
            if (b < pages_per_link.at(i) % all_gather_buffer_params::num_buffers) {
                pages_per_buffer.back()++;
            }
        }


        uint32_t bytes_per_chunk = 0, pages_per_chunk = 0, num_full_chunks = 0, rem_bytes = 0, rem_pages = 0;
        uint32_t link_size_bytes = pages_per_link.at(i) * input_tensor.buffer()->page_size();
        if (pages_per_link.at(i) >= max_pages_per_chunk) {
            bytes_per_chunk = max_buffer_per_chunk;
            pages_per_chunk = max_pages_per_chunk;
            num_full_chunks = link_size_bytes / bytes_per_chunk;
            rem_bytes = link_size_bytes % bytes_per_chunk;
            rem_pages = pages_per_link.at(i) % max_pages_per_chunk;
        } else {
            rem_bytes = link_size_bytes;
            rem_pages = pages_per_link.at(i);
        }

        auto sender_worker_cores = grid_to_cores(sender_workers.start, sender_workers.end, true);
        auto receiver_worker_cores = grid_to_cores(receiver_workers.start, receiver_workers.end, true);
        all_worker_sender_cores.insert(all_worker_sender_cores.end(), sender_worker_cores.begin(), sender_worker_cores.end());
        all_worker_receiver_cores.insert(all_worker_receiver_cores.end(), receiver_worker_cores.begin(), receiver_worker_cores.end());

        std::vector<uint32_t> num_full_chunks_per_worker(all_gather_buffer_params::num_buffers, num_full_chunks / all_gather_buffer_params::num_buffers);
        std::vector<uint32_t> rem_pages_per_worker(all_gather_buffer_params::num_buffers, 0);
        {
            uint32_t worker_idx = 0;
            for (worker_idx = 0; worker_idx < num_full_chunks % all_gather_buffer_params::num_buffers; ++worker_idx) {
                num_full_chunks_per_worker.at(worker_idx)++;
            }
            if (rem_pages != 0) {
                rem_pages_per_worker.at(worker_idx % all_gather_buffer_params::num_buffers) = rem_pages;
            }
        }

        std::vector<uint32_t> link_buffer_sender_num_messages_to_send;
        std::vector<uint32_t> sender_semaphores_base_address;
        std::vector<uint32_t> link_buffer_sender_addresses;
        link_buffer_sender_num_messages_to_send.reserve(all_gather_buffer_params::num_buffers);
        sender_semaphores_base_address.reserve(all_gather_buffer_params::num_buffers);
        link_buffer_sender_addresses.reserve(all_gather_buffer_params::num_buffers);
        for(uint32_t b = 0; b < all_gather_buffer_params::num_buffers; ++b) {
            // link num messages
            link_buffer_sender_num_messages_to_send.push_back(
                (num_full_chunks_per_worker.at(b) + (rem_pages_per_worker.at(b) > 0 ? 1 : 0)) *
                num_transfers);
            sender_semaphores_base_address.push_back(all_gather_buffer_params::eth_sem_l1_byte_address + b * all_gather_buffer_params::semaphore_size);
            link_buffer_sender_addresses.push_back(all_gather_buffer_params::eth_buffer_l1_byte_address + b * all_gather_buffer_params::eth_buffer_size);
        }

        std::vector<uint32_t> link_buffer_receiver_num_messages_to_send;
        std::vector<uint32_t> receiver_semaphores_base_address;
        std::vector<uint32_t> link_buffer_receiver_addresses;
        link_buffer_receiver_num_messages_to_send.reserve(all_gather_buffer_params::num_buffers);
        receiver_semaphores_base_address.reserve(all_gather_buffer_params::num_buffers);
        link_buffer_receiver_addresses.reserve(all_gather_buffer_params::num_buffers);
        for(uint32_t b = 0; b < all_gather_buffer_params::num_buffers; ++b) {
            link_buffer_receiver_num_messages_to_send.push_back(
                (num_full_chunks_per_worker.at(b) + (rem_pages_per_worker.at(b) > 0 ? 1 : 0)) *
                num_transfers);
            receiver_semaphores_base_address.push_back(all_gather_buffer_params::eth_sem_l1_byte_address + b * all_gather_buffer_params::semaphore_size);
            link_buffer_receiver_addresses.push_back(all_gather_buffer_params::eth_buffer_l1_byte_address + b * all_gather_buffer_params::eth_buffer_size);
        }
        // TT_FATAL(
        //     std::accumulate(link_buffer_receiver_num_messages_to_send.begin(), link_buffer_receiver_num_messages_to_send.end(), 0) ==
        //     (
        //         std::accumulate(num_full_chunks_per_worker.begin(),num_full_chunks_per_worker.end(),0) * pages_per_chunk +
        //         std::accumulate(rem_pages_per_worker.begin(),rem_pages_per_worker.end(),0)
        //     )
        // );

        std::vector<uint32_t> eth_receiver_rt_args = {
            static_cast<uint32_t>(all_gather_buffer_params::erisc_handshake_address),
            static_cast<uint32_t>(0), // No senders for now
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(link_receiver_channels_offsets.at(i)),
            static_cast<uint32_t>(link_receiver_num_channels.at(i)),
        };

        std::vector<uint32_t> eth_sender_rt_args = {
            static_cast<uint32_t>(all_gather_buffer_params::erisc_handshake_address),
            static_cast<uint32_t>(link_sender_channels_offsets.at(i)),
            static_cast<uint32_t>(link_sender_num_channels.at(i)),
        };

        // 1 Worker per buffer
        for (uint32_t b = 0; b < all_gather_buffer_params::num_buffers; ++b) {
            uint32_t receiver_output_start_page_idx = output_start_page_idx;
            if (ring_index == 0) {
                receiver_output_start_page_idx += last_output_page_offset;
            } else {
                receiver_output_start_page_idx -= output_page_offset;
            }

            const uint32_t num_workers_per_channel = 1;
            // eth sender args
            // for (uint32_t c = 0; c < link_sender_num_channels.at(i); c++) {
                // sender_buffer_address
                eth_sender_rt_args.push_back(link_buffer_sender_addresses.at(b));

                // sender_num_messages_to_send
                eth_sender_rt_args.push_back(link_buffer_sender_num_messages_to_send.at(b));

                // sender_channel_size
                eth_sender_rt_args.push_back(all_gather_buffer_params::eth_buffer_size);

                // sender_semaphores_base_address -> erisc L1 address
                eth_sender_rt_args.push_back(eth_sem_addrs.at(b));

                // worker_semaphore_address
                eth_sender_rt_args.push_back(sender_worker_writer_semaphore_addr);

                // sender_num_workers - only 1 per channel right now
                eth_sender_rt_args.push_back(num_workers_per_channel);

                // for (uint32_t b = 0; b < sender_worker_cores.size(); ++b) {
                eth_sender_rt_args.push_back((uint32_t)(
                    (device->worker_core_from_logical_core(sender_worker_cores.at(b)).y << 16) |
                    (device->worker_core_from_logical_core(sender_worker_cores.at(b)).x)
                ));
                // }
            // }
            // eth receiver args
            // for (uint32_t c = 0; c < link_receiver_num_channels.at(i); c++) {
                // sender_buffer_address
                eth_receiver_rt_args.push_back(link_buffer_receiver_addresses.at(b));

                // sender_num_messages_to_send
                eth_receiver_rt_args.push_back(link_buffer_receiver_num_messages_to_send.at(b));

                // sender_channel_size
                eth_receiver_rt_args.push_back(all_gather_buffer_params::eth_buffer_size);

                // sender_semaphores_base_address -> erisc L1 address
                eth_receiver_rt_args.push_back(eth_sem_addrs.at(b));

                // worker_semaphore_address
                eth_receiver_rt_args.push_back(receiver_worker_semaphore_addr);

                // sender_num_workers - only 1 per channel right now
                eth_receiver_rt_args.push_back(num_workers_per_channel);

                eth_receiver_rt_args.push_back((uint32_t)(
                    (device->worker_core_from_logical_core(receiver_worker_cores.at(b)).y << 16) |
                    (device->worker_core_from_logical_core(receiver_worker_cores.at(b)).x)
                ));
                // for (CoreCoord const& core: receiver_worker_cores) {
                //     auto const& worker_core = device->worker_core_from_logical_core(core);
                //     eth_receiver_rt_args.push_back((uint32_t)(worker_core.x << 16 | worker_core.y));
                // }
            // }

            // std::cout << "sws " << b << " pages = " << num_full_chunks_per_worker.at(b) * pages_per_chunk + rem_pages_per_worker.at(b) << std::endl;

            // Sender Worker Kernels
            std::vector<uint32_t> worker_reader_sender_ct_args = {
                static_cast<uint32_t>(input_is_dram),
                static_cast<uint32_t>(output_is_dram),
                static_cast<uint32_t>(num_transfers),
                static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                static_cast<uint32_t>(input_page_size),
                static_cast<uint32_t>(output_page_size),
                static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),//pages_per_chunk),
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
                static_cast<uint32_t>(sender_worker_reader_semaphore_addr)
            };

            std::vector<uint32_t> worker_reader_sender_rt_args = {
                static_cast<uint32_t>(input_buffer->address()),
                static_cast<uint32_t>(output_buffer->address())
            };
            KernelHandle worker_reader_sender_kernel_id = tt_metal::CreateKernel(
                program,
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_send_reader.cpp",
                sender_worker_cores.at(b),
                tt_metal::ReaderDataMovementConfig(worker_reader_sender_ct_args, worker_defines));

            worker_reader_sender_kernels.push_back(worker_reader_sender_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_reader_sender_kernel_id,
                sender_worker_cores.at(b),
                worker_reader_sender_rt_args);

            std::vector<uint32_t> worker_writer_sender_ct_args = {
                static_cast<uint32_t>(output_is_dram),
                static_cast<uint32_t>(num_transfers),
                static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                static_cast<uint32_t>(input_page_size),
                static_cast<uint32_t>(output_page_size),
                static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),//pages_per_chunk),
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
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).x),
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).y),
            };

            std::vector<uint32_t> worker_writer_sender_rt_args = {
                static_cast<uint32_t>(output_buffer->address()),
                static_cast<uint32_t>(eth_buffer_addrs.at(b)),
                static_cast<uint32_t>(eth_sem_addrs.at(b))
            };

            KernelHandle worker_writer_sender_kernel_id = tt_metal::CreateKernel(
                program,
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_send_writer.cpp",
                sender_worker_cores.at(b),
                tt_metal::WriterDataMovementConfig(worker_writer_sender_ct_args, worker_defines));

            worker_writer_sender_kernels.push_back(worker_writer_sender_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_writer_sender_kernel_id,
                sender_worker_cores.at(b),
                worker_writer_sender_rt_args);

            // Receiver Worker Kernels
            std::vector<uint32_t> worker_reader_receiver_ct_args = {
                static_cast<uint32_t>(num_transfers),
                static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                static_cast<uint32_t>(input_page_size),
                static_cast<uint32_t>(pages_per_chunk),
                static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),//bytes_per_chunk),
                static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                static_cast<uint32_t>(rem_bytes),
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_receiver_core).x),
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_receiver_core).y),
                static_cast<uint32_t>(eth_sem_addrs.at(b)),
                static_cast<uint32_t>(receiver_worker_semaphore_addr)
            };
            std::vector<uint32_t> worker_reader_receiver_rt_args = {
                static_cast<uint32_t>(eth_buffer_addrs.at(b))
            };

            KernelHandle worker_reader_receiver_kernel_id = tt_metal::CreateKernel(
                program,
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_receive_reader.cpp",
                receiver_worker_cores.at(b),
                tt_metal::ReaderDataMovementConfig(worker_reader_receiver_ct_args, worker_defines));

            worker_reader_receiver_kernels.push_back(worker_reader_receiver_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_reader_receiver_kernel_id,
                receiver_worker_cores.at(b),
                worker_reader_receiver_rt_args);

            std::vector<uint32_t> worker_writer_receiver_ct_args = {
                static_cast<uint32_t>(output_is_dram),
                static_cast<uint32_t>(num_transfers),
                static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                static_cast<uint32_t>(input_page_size),
                static_cast<uint32_t>(output_page_size),
                static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),//pages_per_chunk),
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
                static_cast<uint32_t>(sender_worker_reader_semaphore_addr)

            };
            // Each receiver writer signals the sender reader that data for their portion is available
            auto worker_sender_reader = device->worker_core_from_logical_core(sender_worker_cores.at(b));
            std::vector<uint32_t> worker_writer_receiver_rt_args = {
                static_cast<uint32_t>(output_buffer->address()),
                static_cast<uint32_t>(worker_sender_reader.x),
                static_cast<uint32_t>(worker_sender_reader.y),
            };

            KernelHandle worker_writer_receiver_kernel_id = tt_metal::CreateKernel(
                program,
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_receive_writer.cpp",
                receiver_worker_cores.at(b),
                tt_metal::WriterDataMovementConfig(worker_writer_receiver_ct_args, worker_defines));

            worker_writer_receiver_kernels.push_back(worker_writer_receiver_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_writer_receiver_kernel_id,
                receiver_worker_cores.at(b),
                worker_writer_receiver_rt_args);

            uint32_t pages_per_worker = num_full_chunks_per_worker.at(b) * pages_per_chunk + rem_pages_per_worker.at(b);
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

        eth_sender_rt_args.push_back(0); // num receiver channel
        eth_sender_rt_args.push_back(0); // num receiver channels

        // Ethernet Kernels
        std::vector<uint32_t> eth_sender_ct_args = {
            static_cast<uint32_t>(1), // enable_sender_side
            static_cast<uint32_t>(0) // enable_receiver_side
        };

        {
            std::stringstream ss;
            ss << "device " << device->id() <<  "Sending sender datamover program to x=" <<
                device->ethernet_core_from_logical_core(eth_sender_core).x <<
                ",y=" << device->ethernet_core_from_logical_core(eth_sender_core).y << "(" <<
                eth_sender_core.x << "," << eth_sender_core.y << ") logical" << std::endl;
            // std::cout << ss.str();
        }
        auto eth_sender_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/erisc_datamover.cpp",
            // "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/eth_ring_gather_send.cpp",
            eth_sender_core,
            tt_metal::EthernetConfig{.noc=sender_noc, .compile_args=eth_sender_ct_args});

        {
            std::stringstream ss;
            ss << "eth_sender_rt_args:" << std::endl;
            for (auto arg : eth_sender_rt_args) {
                ss << "\t" << arg << std::endl;
            }
            // std::cout << ss.str();
        }

        {
            std::stringstream ss;
            ss << "eth_receiver_rt_args:" << std::endl;
            for (auto arg : eth_receiver_rt_args) {
                ss << "\t" << arg << std::endl;
            }
            // std::cout << ss.str();
        }

        tt_metal::SetRuntimeArgs(
            program,
            eth_sender_kernel,
            eth_sender_core,
            eth_sender_rt_args);

        eth_sender_kernels.push_back(eth_sender_kernel);

        std::vector<uint32_t> eth_receiver_ct_args = {
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(1)};

        {
            std::stringstream ss;
            ss << "device " << device->id() << "Sending receiver datamover program to x=" <<
                device->ethernet_core_from_logical_core(eth_receiver_core).x <<
                ",y=" << device->ethernet_core_from_logical_core(eth_receiver_core).y << "(" <<
                eth_receiver_core.x << "," << eth_receiver_core.y << ") logical" << std::endl;
            // std::cout << ss.str();
        }
        auto eth_receiver_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/erisc_datamover.cpp",
            // "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/eth_ring_gather_receive.cpp",
            eth_receiver_core,
            tt_metal::EthernetConfig{.noc=receiver_noc, .compile_args=eth_receiver_ct_args});

        eth_receiver_kernels.push_back(eth_receiver_kernel);

        tt_metal::SetRuntimeArgs(
            program,
            eth_receiver_kernel,
            eth_receiver_core,
            eth_receiver_rt_args);

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
