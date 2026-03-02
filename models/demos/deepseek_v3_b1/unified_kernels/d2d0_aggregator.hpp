// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_NCRISC)
#include <cstddef>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#endif

namespace deepseek_b1_ops {

struct D2D0Aggregator {
    template <
        uint32_t senderSocketConfigAddr,
        uint32_t terminationSemaphoreAddr,
        uint32_t senderPageSize,
        uint32_t upstreamPageSize,
        uint32_t fabPktsLink0,
        uint32_t fabPktsLink1,
        uint32_t wholePacketSize,
        uint32_t partialPacketSize,
        uint32_t fabricHeaderCbId,
        uint32_t useFabricOnSender,
        uint32_t numUpstreamSockets,
        uint32_t socket0Addr,
        uint32_t socket1Addr,
        uint32_t socket2Addr,
        uint32_t socket3Addr,
        uint32_t socket4Addr,
        uint32_t socket5Addr,
        uint32_t socket6Addr,
        uint32_t socket7Addr>
    struct CTArgs {
        static constexpr uint32_t sender_socket_config_addr = senderSocketConfigAddr;
        static constexpr uint32_t termination_semaphore_addr = terminationSemaphoreAddr;
        static constexpr uint32_t sender_page_size = senderPageSize;
        static constexpr uint32_t upstream_page_size = upstreamPageSize;
        static constexpr uint32_t fab_pkts_link_0 = fabPktsLink0;
        static constexpr uint32_t fab_pkts_link_1 = fabPktsLink1;
        static constexpr uint32_t whole_packet_size = wholePacketSize;
        static constexpr uint32_t partial_packet_size = partialPacketSize;
        static constexpr uint32_t fabric_header_cb_id = fabricHeaderCbId;
        static constexpr bool use_fabric_on_sender = useFabricOnSender != 0;
        static constexpr uint32_t num_upstream_sockets = numUpstreamSockets;
        static constexpr uint32_t socket_addrs[8] = {
            socket0Addr, socket1Addr, socket2Addr, socket3Addr, socket4Addr, socket5Addr, socket6Addr, socket7Addr};
    };

    struct RTArgs {};

    template <typename CT, bool IsD2D0Core>
    class Op {
    public:
        void setup() {
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (IsD2D0Core) {
                setup_impl();
            }
#endif
        }

        void run() {
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (IsD2D0Core) {
                run_impl();
            }
#endif
        }

    private:
#if defined(COMPILE_FOR_NCRISC)
        SocketSenderInterface sender_socket_{};
        SocketReceiverInterface receiver_sockets_[8]{};
        sender_downstream_encoding downstream_enc_{};
        uint64_t downstream_bytes_sent_noc_addr_{};
        uint32_t downstream_fifo_l1_addr_{};
        tt::tt_fabric::WorkerToFabricEdmSender fabric_conn_1_{};
        tt::tt_fabric::WorkerToFabricEdmSender fabric_conn_2_{};
        volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_hdr_1_{nullptr};
        volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_hdr_2_{nullptr};

        static FORCE_INLINE bool socket_wait_for_pages_with_termination(
            const SocketReceiverInterface& socket,
            uint32_t num_pages,
            volatile tt_l1_ptr uint32_t* termination_semaphore) {
            constexpr uint32_t termination_value = 1;
            while (!socket_wait_for_pages(socket, num_pages, 1000)) {
                invalidate_l1_cache();
                if (termination_semaphore[0] == termination_value) {
                    return false;
                }
            }
            return true;
        }

        static FORCE_INLINE void write_data_to_remote(
            tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
            volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
            uint32_t l1_addr,
            uint64_t dst_addr,
            uint64_t bytes_sent_noc_addr,
            uint32_t packet_size) {
            packet_header->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{dst_addr, bytes_sent_noc_addr, packet_size}, packet_size);
            fabric_connection.wait_for_empty_write_slot();
            fabric_connection.send_payload_without_header_non_blocking_from_address(l1_addr, packet_size);
            fabric_connection.send_payload_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
        }

        void send_pages_over_socket(uint32_t l1_read_addr, uint64_t dst_addr) {
            for (uint32_t i = 0; i < CT::fab_pkts_link_0; ++i) {
                write_data_to_remote(
                    fabric_conn_1_,
                    pkt_hdr_1_,
                    l1_read_addr,
                    dst_addr,
                    downstream_bytes_sent_noc_addr_,
                    CT::whole_packet_size);
                l1_read_addr += CT::whole_packet_size;
                dst_addr += CT::whole_packet_size;
            }
            for (uint32_t i = 0; i < CT::fab_pkts_link_1; ++i) {
                write_data_to_remote(
                    fabric_conn_2_,
                    pkt_hdr_2_,
                    l1_read_addr,
                    dst_addr,
                    downstream_bytes_sent_noc_addr_,
                    CT::whole_packet_size);
                l1_read_addr += CT::whole_packet_size;
                dst_addr += CT::whole_packet_size;
            }
            if constexpr (CT::partial_packet_size > 0) {
                write_data_to_remote(
                    fabric_conn_2_,
                    pkt_hdr_2_,
                    l1_read_addr,
                    dst_addr,
                    downstream_bytes_sent_noc_addr_,
                    CT::partial_packet_size);
            }
        }

        void setup_impl() {
            size_t rt_args_idx = 0;
            if constexpr (CT::use_fabric_on_sender) {
                fabric_conn_1_ =
                    tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
                fabric_conn_2_ =
                    tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
            }

            sender_socket_ = create_sender_socket_interface(CT::sender_socket_config_addr);
            set_sender_socket_page_size(sender_socket_, CT::sender_page_size);
            downstream_enc_ = get_downstream_encoding(sender_socket_, 0);

            receiver_sockets_[0] = create_receiver_socket_interface(CT::socket_addrs[0]);
            receiver_sockets_[1] = create_receiver_socket_interface(CT::socket_addrs[1]);
            receiver_sockets_[2] = create_receiver_socket_interface(CT::socket_addrs[2]);
            receiver_sockets_[3] = create_receiver_socket_interface(CT::socket_addrs[3]);
            receiver_sockets_[4] = create_receiver_socket_interface(CT::socket_addrs[4]);
            receiver_sockets_[5] = create_receiver_socket_interface(CT::socket_addrs[5]);
            receiver_sockets_[6] = create_receiver_socket_interface(CT::socket_addrs[6]);
            receiver_sockets_[7] = create_receiver_socket_interface(CT::socket_addrs[7]);
            for (uint32_t i = 0; i < CT::num_upstream_sockets; i++) {
                set_receiver_socket_page_size(receiver_sockets_[i], CT::upstream_page_size);
            }

            downstream_bytes_sent_noc_addr_ = get_noc_addr(
                downstream_enc_.d2d.downstream_noc_x,
                downstream_enc_.d2d.downstream_noc_y,
                sender_socket_.downstream_bytes_sent_addr);
            downstream_fifo_l1_addr_ = sender_socket_.downstream_fifo_addr;

            if constexpr (CT::use_fabric_on_sender) {
                pkt_hdr_1_ =
                    reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(CT::fabric_header_cb_id));
                pkt_hdr_2_ = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                    get_write_ptr(CT::fabric_header_cb_id) + sizeof(PACKET_HEADER_TYPE));
                fabric_conn_1_.open();
                fabric_conn_2_.open();
                fabric_set_unicast_route(pkt_hdr_1_, downstream_enc_);
                fabric_set_unicast_route(pkt_hdr_2_, downstream_enc_);
            }
        }

        void run_impl() {
            volatile tt_l1_ptr uint32_t* termination_semaphore =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CT::termination_semaphore_addr);

            while (true) {
                uint32_t bytes_accumulated = 0;
                bool terminated = false;

                socket_reserve_pages(sender_socket_, 1);

                for (uint32_t i = 0; i < CT::num_upstream_sockets; i++) {
                    if (!socket_wait_for_pages_with_termination(receiver_sockets_[i], 1, termination_semaphore)) {
                        terminated = true;
                        break;
                    }

                    auto l1_read_addr = receiver_sockets_[i].read_ptr;
                    uint32_t dst_l1_addr = downstream_fifo_l1_addr_ + sender_socket_.write_ptr + bytes_accumulated;
                    uint64_t dst_addr = get_noc_addr(
                        downstream_enc_.d2d.downstream_noc_x, downstream_enc_.d2d.downstream_noc_y, dst_l1_addr);

                    noc_async_write(l1_read_addr, dst_addr, CT::upstream_page_size);
                    noc_async_writes_flushed();
                    socket_pop_pages(receiver_sockets_[i], 1);
                    socket_notify_sender(receiver_sockets_[i]);
                    invalidate_l1_cache();
                    bytes_accumulated += CT::upstream_page_size;
                }

                if (terminated) {
                    break;
                }

                if (bytes_accumulated >= CT::sender_page_size) {
                    if constexpr (CT::use_fabric_on_sender) {
                        send_pages_over_socket(
                            downstream_fifo_l1_addr_ + sender_socket_.write_ptr,
                            get_noc_addr(
                                downstream_enc_.d2d.downstream_noc_x,
                                downstream_enc_.d2d.downstream_noc_y,
                                downstream_fifo_l1_addr_ + sender_socket_.write_ptr));
                    }
                    socket_push_pages(sender_socket_, 1);
                    socket_notify_receiver(sender_socket_);
                }
                invalidate_l1_cache();
            }

            // Teardown
            update_socket_config(sender_socket_);
            for (uint32_t i = 0; i < CT::num_upstream_sockets; i++) {
                update_socket_config(receiver_sockets_[i]);
            }
            if constexpr (CT::use_fabric_on_sender) {
                fabric_conn_1_.close();
                fabric_conn_2_.close();
            }
        }
#endif  // COMPILE_FOR_NCRISC
    };  // class Op
};  // struct D2D0Aggregator

}  // namespace deepseek_b1_ops
