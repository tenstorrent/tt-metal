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

    bool rm = input_tensor.layout() == Layout::ROW_MAJOR;
    bool width = input_tensor.shape().rank() - 1 == dim;
    DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

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
        pages_per_link[i]++;
    }

    bool rm = input_tensor.get_layout() == Layout::ROW_MAJOR;
    bool width = input_tensor.get_legacy_shape().rank() - 1 == dim;
    DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
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
        auto sender_worker_writer_semaphore_addr = tt_metal::CreateSemaphore(program, sender_workers, 1);
        // This semaphore is used by the worker receiver writer to tell the worker sender reader that data has been committed to memory
        // This is currently a running counter of how many chunks were committed since the sender worker never decrements this buffer
        // Potentially avoid overflow by having it actually decrement (using noc atomic inc with value of -1)
        auto sender_worker_reader_semaphore_addr = tt_metal::CreateSemaphore(program, sender_workers, 0);

        auto sender_noc = detail::GetPreferredNOCForDRAMRead(tt::Cluster::instance().arch());
        auto receiver_noc = detail::GetPreferredNOCForDRAMWrite(tt::Cluster::instance().arch());

        auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id)[sender_socket_idx];
        eth_sender_cores.push_back(eth_sender_core);
        auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id)[receiver_socket_idx];
        eth_receiver_cores.push_back(eth_receiver_core);

        uint32_t bytes_per_chunk = 0, pages_per_chunk = 0, num_full_chunks = 0, rem_bytes = 0, rem_pages = 0;
        uint32_t buffer_size = pages_per_link[i] * input_tensor.buffer()->page_size();
        if (pages_per_link[i] > max_pages_per_chunk) {
            bytes_per_chunk = max_buffer_per_chunk;
            pages_per_chunk = max_pages_per_chunk;
            num_full_chunks = buffer_size / bytes_per_chunk;
            rem_bytes = buffer_size % bytes_per_chunk;
            rem_pages = pages_per_link[i] % max_pages_per_chunk;
        } else {
            rem_bytes = buffer_size;
            rem_pages = pages_per_link[i];
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
                num_full_chunks_per_worker[worker_idx]++;
            }
            if (rem_pages != 0) {
                rem_pages_per_worker[worker_idx % all_gather_buffer_params::num_buffers] = rem_pages;
            }
        }

        // 1 Worker per buffer
        for (uint32_t b = 0; b < all_gather_buffer_params::num_buffers; ++b) {
            uint32_t receiver_output_start_page_idx = output_start_page_idx;
            if (ring_index == 0) {
                receiver_output_start_page_idx += last_output_page_offset;
            } else {
                receiver_output_start_page_idx -= output_page_offset;
            }

            // Sender Worker Kernels
            std::vector<uint32_t> worker_reader_sender_ct_args = {
                static_cast<uint32_t>(input_is_dram),
                static_cast<uint32_t>(output_is_dram),
                static_cast<uint32_t>(num_transfers),
                static_cast<uint32_t>(num_full_chunks_per_worker[b]),
                static_cast<uint32_t>(input_page_size),
                static_cast<uint32_t>(output_page_size),
                static_cast<uint32_t>(pages_per_chunk),
                static_cast<uint32_t>(rem_pages_per_worker[b]),
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
                sender_worker_cores[b],
                tt_metal::ReaderDataMovementConfig(worker_reader_sender_ct_args, worker_defines));

            worker_reader_sender_kernels.push_back(worker_reader_sender_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_reader_sender_kernel_id,
                sender_worker_cores[b],
                worker_reader_sender_rt_args);

            std::vector<uint32_t> worker_writer_sender_ct_args = {
                static_cast<uint32_t>(output_is_dram),
                static_cast<uint32_t>(num_transfers),
                static_cast<uint32_t>(num_full_chunks_per_worker[b]),
                static_cast<uint32_t>(input_page_size),
                static_cast<uint32_t>(output_page_size),
                static_cast<uint32_t>(pages_per_chunk),
                static_cast<uint32_t>(rem_pages_per_worker[b]),
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
                static_cast<uint32_t>(sender_worker_writer_semaphore_addr),
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).x),
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).y),
            };

            std::vector<uint32_t> worker_writer_sender_rt_args = {
                static_cast<uint32_t>(output_buffer->address()),
                static_cast<uint32_t>(eth_buffer_addrs[b]),
                static_cast<uint32_t>(eth_sem_addrs[b])
            };

            KernelHandle worker_writer_sender_kernel_id = tt_metal::CreateKernel(
                program,
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_send_writer.cpp",
                sender_worker_cores[b],
                tt_metal::WriterDataMovementConfig(worker_writer_sender_ct_args, worker_defines));

            worker_writer_sender_kernels.push_back(worker_writer_sender_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_writer_sender_kernel_id,
                sender_worker_cores[b],
                worker_writer_sender_rt_args);

            // Receiver Worker Kernels
            std::vector<uint32_t> worker_reader_receiver_ct_args = {
                static_cast<uint32_t>(num_transfers),
                static_cast<uint32_t>(num_full_chunks_per_worker[b]),
                static_cast<uint32_t>(input_page_size),
                static_cast<uint32_t>(pages_per_chunk),
                static_cast<uint32_t>(bytes_per_chunk),
                static_cast<uint32_t>(rem_pages_per_worker[b]),
                static_cast<uint32_t>(rem_bytes),
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_receiver_core).x),
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_receiver_core).y),
                static_cast<uint32_t>(eth_sem_addrs[b]),
                static_cast<uint32_t>(receiver_worker_semaphore_addr)
            };
            std::vector<uint32_t> worker_reader_receiver_rt_args = {
                static_cast<uint32_t>(eth_buffer_addrs[b])
            };

            KernelHandle worker_reader_receiver_kernel_id = tt_metal::CreateKernel(
                program,
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_receive_reader.cpp",
                receiver_worker_cores[b],
                tt_metal::ReaderDataMovementConfig(worker_reader_receiver_ct_args, worker_defines));

            worker_reader_receiver_kernels.push_back(worker_reader_receiver_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_reader_receiver_kernel_id,
                receiver_worker_cores[b],
                worker_reader_receiver_rt_args);

            std::vector<uint32_t> worker_writer_receiver_ct_args = {
                static_cast<uint32_t>(output_is_dram),
                static_cast<uint32_t>(num_transfers),
                static_cast<uint32_t>(num_full_chunks_per_worker[b]),
                static_cast<uint32_t>(input_page_size),
                static_cast<uint32_t>(output_page_size),
                static_cast<uint32_t>(pages_per_chunk),
                static_cast<uint32_t>(rem_pages_per_worker[b]),
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
            auto worker_sender_reader = device->worker_core_from_logical_core(sender_worker_cores[b]);
            std::vector<uint32_t> worker_writer_receiver_rt_args = {
                static_cast<uint32_t>(output_buffer->address()),
                static_cast<uint32_t>(worker_sender_reader.x),
                static_cast<uint32_t>(worker_sender_reader.y),
            };

            KernelHandle worker_writer_receiver_kernel_id = tt_metal::CreateKernel(
                program,
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_receive_writer.cpp",
                receiver_worker_cores[b],
                tt_metal::WriterDataMovementConfig(worker_writer_receiver_ct_args, worker_defines));

            worker_writer_receiver_kernels.push_back(worker_writer_receiver_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_writer_receiver_kernel_id,
                receiver_worker_cores[b],
                worker_writer_receiver_rt_args);

            uint32_t pages_per_worker = num_full_chunks_per_worker[b] * pages_per_chunk + rem_pages_per_worker[b];
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

        // Ethernet Kernels
        std::vector<uint32_t> eth_sender_ct_args = {
            static_cast<uint32_t>(num_transfers),
            static_cast<uint32_t>(num_full_chunks),
            static_cast<uint32_t>(bytes_per_chunk),
            static_cast<uint32_t>(rem_bytes),
            static_cast<uint32_t>(all_gather_buffer_params::num_buffers),
            static_cast<uint32_t>(all_gather_buffer_params::eth_buffer_size),
            static_cast<uint32_t>(all_gather_buffer_params::eth_buffer_l1_byte_address),
            static_cast<uint32_t>(1),
            static_cast<uint32_t>(sender_worker_writer_semaphore_addr)};

        std::vector<uint32_t> eth_sender_rt_args;
        eth_sender_rt_args.reserve(2 * sender_workers.size() + eth_sem_addrs.size());
        for (uint32_t b = 0; b < sender_worker_cores.size(); ++b) {
            eth_sender_rt_args.push_back(device->worker_core_from_logical_core(sender_worker_cores[b]).x);
            eth_sender_rt_args.push_back(device->worker_core_from_logical_core(sender_worker_cores[b]).y);
        }
        eth_sender_rt_args.insert(eth_sender_rt_args.end(), eth_sem_addrs.begin(), eth_sem_addrs.end());

        auto eth_sender_kernel = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/eth_ring_gather_send.cpp",
            eth_sender_core,
            tt_metal::EthernetConfig{.noc=sender_noc, .compile_args=eth_sender_ct_args});

        tt_metal::SetRuntimeArgs(
            program,
            eth_sender_kernel,
            eth_sender_core,
            eth_sender_rt_args);

        eth_sender_kernels.push_back(eth_sender_kernel);

        std::vector<uint32_t> eth_receiver_ct_args = {
            static_cast<uint32_t>(num_transfers),
            static_cast<uint32_t>(num_full_chunks),
            static_cast<uint32_t>(bytes_per_chunk),
            static_cast<uint32_t>(rem_bytes),
            static_cast<uint32_t>(all_gather_buffer_params::num_buffers),
            static_cast<uint32_t>(all_gather_buffer_params::eth_buffer_size),
            static_cast<uint32_t>(all_gather_buffer_params::eth_buffer_l1_byte_address),
            static_cast<uint32_t>(1),
            static_cast<uint32_t>(receiver_worker_semaphore_addr)};

        std::vector<uint32_t> eth_receiver_rt_args;
        eth_receiver_rt_args.reserve(2 * receiver_workers.size() + eth_sem_addrs.size());
        for (uint32_t b = 0; b < receiver_workers.size(); ++b) {
            eth_receiver_rt_args.push_back(device->worker_core_from_logical_core(receiver_worker_cores[b]).x);
            eth_receiver_rt_args.push_back(device->worker_core_from_logical_core(receiver_worker_cores[b]).y);
        }
        eth_receiver_rt_args.insert(eth_receiver_rt_args.end(), eth_sem_addrs.begin(), eth_sem_addrs.end());
        auto eth_receiver_kernel = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/eth_ring_gather_receive.cpp",
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
        const auto& input = input_tensors[0];
        const auto& output = output_tensors[0];
        for (uint32_t i = 0; i < total_worker_core_pairs_used; ++i) {
            auto &worker_reader_sender_runtime_args = GetRuntimeArgs(program, worker_reader_sender_kernels[i], all_worker_sender_cores[i]);
            worker_reader_sender_runtime_args[0] = input.buffer()->address();
            worker_reader_sender_runtime_args[1] = output.buffer()->address();
            auto &worker_writer_sender_runtime_args = GetRuntimeArgs(program, worker_writer_sender_kernels[i], all_worker_sender_cores[i]);
            worker_writer_sender_runtime_args[0] = output.buffer()->address();

            auto &worker_writer_receiver_runtime_args = GetRuntimeArgs(program, worker_writer_receiver_kernels[i], all_worker_receiver_cores[i]);
            worker_writer_receiver_runtime_args[0] = output.buffer()->address();
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
