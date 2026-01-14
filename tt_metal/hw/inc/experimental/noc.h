// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

namespace experimental {

struct MulticastEndpoint;

template <typename T>
struct noc_traits_t {
    static_assert(sizeof(T) == 0, "NoC transactions are not supported for this type");
};

/**
 * @brief Noc class that provides a high-level interface for asynchronous read and write operations.
 *
 * It abstracts the details of source and destination address calculations.
 */
class Noc {
public:
    enum class AddressType { NOC, LOCAL_L1 };

    enum class TxnIdMode { ENABLED, DISABLED };

    enum class ResponseMode { NON_POSTED, POSTED };

    enum class BarrierMode { TXN_ID, FULL };

    enum class McastMode { INCLUDE_SRC, EXCLUDE_SRC };

    enum class VcSelection { DEFAULT, CUSTOM };

    static constexpr uint32_t INVALID_TXN_ID = 0xFFFFFFFF;

private:
    template <typename T>
    using src_args_t = typename noc_traits_t<T>::src_args_type;
    template <typename T>
    using dst_args_t = typename noc_traits_t<T>::dst_args_type;
    template <typename T>
    using dst_args_mcast_t = typename noc_traits_t<T>::dst_args_mcast_type;

    template <AddressType address_type>
    using addr_underlying_t = std::conditional_t<address_type == AddressType::LOCAL_L1, uintptr_t, uint64_t>;

    template <AddressType address_type, typename Src>
    auto get_src_ptr(const Src& src, const src_args_t<Src>& src_args) const {
        return addr_underlying_t<address_type>{
            noc_traits_t<Src>::template src_addr<address_type>(src, *this, src_args)};
    }

    template <AddressType address_type, typename Dst>
    auto get_dst_ptr(const Dst& dst, const dst_args_t<Dst>& dst_args) const {
        return addr_underlying_t<address_type>{
            noc_traits_t<Dst>::template dst_addr<address_type>(dst, *this, dst_args)};
    }

    template <AddressType address_type, typename Dst>
    auto get_dst_ptr_mcast(const Dst& dst, const dst_args_mcast_t<Dst>& dst_args) const {
        return addr_underlying_t<address_type>{
            noc_traits_t<Dst>::template dst_addr_mcast<address_type>(dst, *this, dst_args)};
    }

public:
    Noc() : noc_id_(noc_index) {}
    explicit Noc(uint8_t noc_id) : noc_id_(noc_id) {}

    uint8_t get_noc_id() const { return noc_id_; }

    bool is_local_bank(uint32_t virtual_x, uint32_t virtual_y) const {
        return virtual_x == my_x[noc_id_] && virtual_y == my_y[noc_id_];
    }

    bool is_local_addr(const uint64_t noc_addr) const {
        uint32_t x = NOC_UNICAST_ADDR_X(noc_addr);
        uint32_t y = NOC_UNICAST_ADDR_Y(noc_addr);
        return is_local_bank(x, y);
    }

    /**
     * @brief Initiates an asynchronous read from a specified source.
     *
     * The destination is in L1 memory on the Tensix core executing this function call.
     *
     * @see async_read_barrier.
     *
     * @param src Source object (e.g., TensorAccessor)
     * @param dst Destination object (e.g., local L1 memory)
     * @param size_bytes Size of the data transfer in bytes
     * @param src_args Additional arguments for source address calculation
     * @param dst_args Additional arguments for destination address calculation
     * @param read_req_vc Virtual channel to use for the read request (default: NOC_UNICAST_WRITE_VC)
     * @param trid Transaction ID to use when transaction id mode is enabled (default: INVALID_TXN_ID)
     * @tparam txn_id_mode Whether transaction id will be used for the noc transaction (default: DISABLED)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     * @tparam enable_noc_tracing Enable NoC tracing for debugging (default: true)
     */
    template <
        TxnIdMode txn_id_mode = TxnIdMode::DISABLED,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        bool enable_noc_tracing = true,
        typename Src,
        typename Dst>
    void async_read(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        uint32_t read_req_vc = NOC_UNICAST_WRITE_VC,
        uint32_t trid = INVALID_TXN_ID) const {
        if constexpr (txn_id_mode == TxnIdMode::ENABLED) {
            noc_async_read_set_trid(trid, noc_id_);
            uint64_t src_noc_addr = get_src_ptr<AddressType::NOC>(src, src_args);
            static_assert(
                max_page_size <= NOC_MAX_BURST_SIZE,
                "Read with transaction id is not supported for page sizes greater than NOC_MAX_BURST_SIZE");
            noc_async_read_one_packet_set_state(src_noc_addr, size_bytes, read_req_vc, noc_id_);
            noc_async_read_one_packet_with_state_with_trid(
                static_cast<uint32_t>((src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK),
                static_cast<uint32_t>(src_noc_addr),
                get_dst_ptr<AddressType::LOCAL_L1>(dst, dst_args),
                trid,
                noc_id_);
        } else {
            noc_async_read<max_page_size, enable_noc_tracing>(
                get_src_ptr<AddressType::NOC>(src, src_args),
                get_dst_ptr<AddressType::LOCAL_L1>(dst, dst_args),
                size_bytes,
                noc_id_,
                read_req_vc);
        }
    }

    /**
     * @brief Sets the stateful registers for an asynchronous read from a specified source
     *
     * This is used to set up state for async_read_with_state, use async_read instead if state preservation is not
     * needed.
     *
     * @see async_read_with_state and async_read_barrier.
     *
     * @param src Source object (e.g., TensorAccessor)
     * @param size_bytes Size of the data transfer in bytes
     * @param src_args Additional arguments for source address calculation
     * @param vc Virtual channel to use for the read request when vc_selection is CUSTOM (default: 0)
     * @tparam vc_selection Whether to use a custom specified virtual channel (default: DEFAULT)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     */
    template <
        VcSelection vc_selection = VcSelection::DEFAULT,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src>
    void set_async_read_state(
        const Src& src, uint32_t size_bytes, const src_args_t<Src>& src_args, uint8_t vc = 0) const {
        auto src_noc_addr = get_src_ptr<AddressType::NOC>(src, src_args);
        DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc_id_, DEBUG_SANITIZE_NOC_UNICAST);
        RECORD_NOC_EVENT_WITH_ADDR(
            NocEventType::READ_SET_STATE,
            0,
            src_noc_addr,
            size_bytes,
            (vc_selection == VcSelection::CUSTOM) ? static_cast<int8_t>(vc) : -1,
            false);

        WAYPOINT("NASW");
        ncrisc_noc_read_set_state<noc_mode, max_page_size <= NOC_MAX_BURST_SIZE, vc_selection == VcSelection::CUSTOM>(
            noc_id_, read_cmd_buf, src_noc_addr, size_bytes, vc);
        WAYPOINT("NASD");
    }

    /**
     * @brief Initiates an asynchronous read from a specified source based on previously set state
     *
     * This must be preceded by a call to set_async_read_state where Src is at same noc location as the one used in
     * set_async_read_state
     *
     * @see set_async_read_state and async_read_barrier.
     *
     * @param src Source object (e.g., TensorAccessor)
     * @param dst Destination object (e.g., local L1 memory)
     * @param size_bytes Size of the data transfer in bytes, this must be equal to the value set in set_async_read_state
     * if max_page_size <= NOC_MAX_BURST_SIZE
     * @param src_args Additional arguments for source address calculation
     * @param dst_args Additional arguments for destination address calculation
     * @param vc Virtual channel to use for the read request when vc_selection is CUSTOM (default: 0)
     * @tparam vc_selection Whether to use a custom specified virtual channel (default: DEFAULT)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     */
    template <
        VcSelection vc_selection = VcSelection::DEFAULT,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src,
        typename Dst>
    void async_read_with_state(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        uint8_t vc = 0) const {
        // TODO (#33966): Need to make sure set state was called and with same template params
        if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
            noc_async_read_one_packet_with_state<true, vc_selection == VcSelection::CUSTOM>(
                (uint32_t)get_src_ptr<AddressType::NOC>(src, src_args),
                get_dst_ptr<AddressType::LOCAL_L1>(dst, dst_args),
                vc,
                noc_id_);
        } else {
            noc_async_read_with_state(
                (uint32_t)get_src_ptr<AddressType::NOC>(src, src_args),
                get_dst_ptr<AddressType::LOCAL_L1>(dst, dst_args),
                size_bytes,
                noc_id_);
        }
    }

    /** @brief Initiates an asynchronous write.
     *
     * @see async_write_barrier.
     *
     * @param src Source object (e.g., local L1 memory)
     * @param dst Destination object (e.g., TensorAccessor)
     * @param size_bytes Size of the data transfer in bytes
     * @param src_args Additional arguments for source address calculation
     * @param dst_args Additional arguments for destination address calculation
     * @param vc Virtual channel to use for the write transaction (default: NOC_UNICAST_WRITE_VC)
     * @param trid Transaction ID to use when transaction id mode is enabled (default: INVALID_TXN_ID)
     * @tparam txn_id_mode Whether transaction id will be used for the noc transaction (default: DISABLED)
     * @tparam response_mode Posted noc transactions do not get ack from receiver, non-posted ones do (default:
     * NON_POSTED)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     * @tparam enable_noc_tracing Enable NoC tracing for debugging (default: true)
     */
    template <
        TxnIdMode txn_id_mode = TxnIdMode::DISABLED,
        ResponseMode response_mode = ResponseMode::NON_POSTED,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        bool enable_noc_tracing = true,
        typename Src,
        typename Dst>
    void async_write(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        uint32_t vc = NOC_UNICAST_WRITE_VC,
        uint32_t trid = INVALID_TXN_ID) const {
        constexpr bool posted = response_mode == ResponseMode::POSTED;

        if constexpr (txn_id_mode == TxnIdMode::ENABLED) {
            // TODO (#31535): Need to add check in ncrisc_noc_fast_write_any_len to ensure outstanding transaction
            // register does not overflow
            WAYPOINT("NAWW");
            ASSERT(trid != INVALID_TXN_ID);
            auto src_addr = get_src_ptr<AddressType::LOCAL_L1>(src, src_args);
            auto dst_noc_addr = get_dst_ptr<AddressType::NOC>(dst, dst_args);
            if constexpr (enable_noc_tracing) {
                RECORD_NOC_EVENT_WITH_ADDR(
                    NocEventType::WRITE_WITH_TRID, src_addr, dst_noc_addr, size_bytes, -1, posted);
            }
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id_, dst_noc_addr, src_addr, size_bytes);
            constexpr bool one_packet = max_page_size <= NOC_MAX_BURST_SIZE;
            ncrisc_noc_fast_write_any_len<noc_mode, true, one_packet>(
                noc_id_,
                write_cmd_buf,
                src_addr,
                dst_noc_addr,
                size_bytes,
                vc,
                false,  // mcast
                false,  // linked
                1,      // num_dests
                true,   // multicast_path_reserve
                posted,
                trid);
            WAYPOINT("NWPD");
        } else {
            noc_async_write<max_page_size, enable_noc_tracing, posted>(
                get_src_ptr<AddressType::LOCAL_L1>(src, src_args),
                get_dst_ptr<AddressType::NOC>(dst, dst_args),
                size_bytes,
                noc_id_,
                vc);
        }
    }

    /** @brief Initiates an asynchronous write from a source address in memory on the core executing this function call
     * to a rectangular destination grid.
     *
     * The destination nodes must be a set of Tensix cores and must form a rectangular grid.
     *
     * @see async_write_barrier.
     *
     * @param src Source object (e.g., local L1 memory)
     * @param dst Destination object (e.g., TensorAccessor)
     * @param size_bytes Size of the data transfer in bytes
     * @param num_dsts Number of destinations that the multicast source is targeting
     * @param src_args Additional arguments for source address calculation
     * @param dst_args Additional arguments for destination address calculation
     * @param linked
     * @param trid Transaction ID to use when transaction id mode is enabled (default: INVALID_TXN_ID)
     * @tparam mcast_mode Indicates whether the sender is included in the multicast destinations
     * @tparam txn_id_mode Whether transaction id will be used for the noc transaction (default: DISABLED)
     * @tparam response_mode Posted noc transactions do not get ack from receiver, non-posted ones do (default:
     * NON_POSTED)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     * @tparam enable_noc_tracing Enable NoC tracing for debugging (default: true)
     */
    template <
        McastMode mcast_mode = McastMode::EXCLUDE_SRC,
        TxnIdMode txn_id_mode = TxnIdMode::DISABLED,
        ResponseMode response_mode = ResponseMode::NON_POSTED,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        bool enable_noc_tracing = true,
        typename Src,
        typename Dst>
    void async_write_multicast(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        uint32_t num_dsts,
        const src_args_t<Src>& src_args,
        const dst_args_mcast_t<Dst>& dst_args,
        bool linked = false,
        uint32_t trid = INVALID_TXN_ID) const {
        static_assert(txn_id_mode == TxnIdMode::DISABLED, "Mcasts with transaction id are not supported yet");
        static_assert(
            response_mode == ResponseMode::NON_POSTED,
            "Mcasts with posted transactions are not supported");  // TODO: Make this an arch specific assertion

        auto src_addr = get_src_ptr<AddressType::LOCAL_L1>(src, src_args);
        auto dst_noc_addr = get_dst_ptr_mcast<AddressType::NOC>(dst, dst_args);
        if constexpr (mcast_mode == McastMode::INCLUDE_SRC) {
            noc_async_write_multicast_loopback_src(src_addr, dst_noc_addr, size_bytes, num_dsts, linked, noc_id_);
        } else if constexpr (mcast_mode == McastMode::EXCLUDE_SRC) {
            noc_async_write_multicast<max_page_size>(src_addr, dst_noc_addr, size_bytes, num_dsts, linked, noc_id_);
        }
    }

    /**
     * @brief Sets the stateful registers for an asynchronous write
     *
     * This function is used to set up the state for async_write_with_state, async_write can be used if state
     * preservation is not needed
     *
     * @see async_write_with_state and async_write_barrier.
     *
     * @param dst Destination object (e.g., local L1 memory)
     * @param size_bytes Size of the data transfer in bytes
     * @param dst_args Additional arguments for destination address calculation
     * @param vc Virtual channel to use for the write request (default: NOC_UNICAST_WRITE_VC)
     * @tparam response_mode Whether the write is posted or non-posted (default: NON_POSTED)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     */
    template <
        ResponseMode response_mode = ResponseMode::NON_POSTED,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Dst>
    void set_async_write_state(
        const Dst& dst, uint32_t size_bytes, const dst_args_t<Dst>& dst_args, uint8_t vc = NOC_UNICAST_WRITE_VC) const {
        DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc_id_, DEBUG_SANITIZE_NOC_UNICAST);
        auto dst_noc_addr = get_dst_ptr<AddressType::NOC>(dst, dst_args);
        constexpr bool posted = response_mode == ResponseMode::POSTED;
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_SET_STATE, 0, dst_noc_addr, size_bytes, vc, posted);

        WAYPOINT("NWPW");
        ncrisc_noc_write_set_state<posted, max_page_size <= NOC_MAX_BURST_SIZE>(
            noc_id_, write_cmd_buf, dst_noc_addr, size_bytes, vc);
        WAYPOINT("NWPD");
    }

    /**
     * @brief Initiates an asynchronous write to a specified destination based on previously set state
     *
     * This must be preceded by a call to set_async_write_state where Dst is at same noc location as the one used in
     * set_async_write_state
     *
     * @see set_async_write_state and async_write_barrier.
     *
     * @param src Source object (e.g., local L1 memory)
     * @param dst Destination object (e.g., TensorAccessor)
     * @param size_bytes Size of the data transfer in bytes, this must be equal to the value set in
     * set_async_write_state if max_page_size <= NOC_MAX_BURST_SIZE
     * @param src_args Additional arguments for source address calculation
     * @param dst_args Additional arguments for destination address calculation
     * @param vc Virtual channel to use for the write request (default: NOC_UNICAST_WRITE_VC)
     * @tparam response_mode Whether the write is posted or non-posted (default: NON_POSTED)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     */
    template <
        ResponseMode response_mode = ResponseMode::NON_POSTED,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src,
        typename Dst>
    void async_write_with_state(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        uint8_t vc = NOC_UNICAST_WRITE_VC) const {
        constexpr bool posted = response_mode == ResponseMode::POSTED;

        if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
            noc_async_write_one_packet_with_state<posted>(
                get_src_ptr<AddressType::LOCAL_L1>(src, src_args),
                (uint32_t)get_dst_ptr<AddressType::NOC>(dst, dst_args),
                noc_id_);
        } else {
            // In order to sanitize, need to grab full noc addr + xfer size from state.
            auto src_addr = get_src_ptr<AddressType::LOCAL_L1>(src, src_args);
            auto dst_addr =
                get_dst_ptr<AddressType::NOC>(dst, dst_args);  // NoC target was programmed in set_async_write_state
            RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_STATE, src_addr, 0ull, 0, -1, posted);
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id_, dst_addr, src_addr);

            WAYPOINT("NWPW");
            ncrisc_noc_write_any_len_with_state<noc_mode, posted>(
                noc_id_,
                write_cmd_buf,
                get_src_ptr<AddressType::LOCAL_L1>(src, src_args),
                (uint32_t)get_dst_ptr<AddressType::NOC>(dst, dst_args),
                size_bytes);
            WAYPOINT("NWPD");
        }
    }

    /** @brief Initiates an asynchronous write of a 32-bit value to a NOC destination.
     *
     * Typically used for writing registers, but can be used for memory locations as well.
     * The advantage over using noc_async_write is that we don't use a Tensix L1 memory source location; the write value
     * is written directly into a register. Unlike using noc_async_write, there are also no address alignment concerns.
     * The destination can be either a Tensix core+L1 memory address or a PCIe controller; This API does not support
     * DRAM addresses. Note: Due to HW bug on Blackhole, inline writes to L1 will use a scratch location in L1 memory.
     *
     * @see async_write_barrier.
     *
     * @param dst Destination object (e.g., UnicastEndpoint)
     * @param val The value to be written
     * @param dst_args Additional arguments for destination address calculation
     * @param be Byte-enable mask controls which bytes are written to at an L1 destination
     * @param vc Virtual channel to use for the transaction
     * @param trid Transaction ID to use for the transaction (default: INVALID_TXN_ID)
     * @tparam txn_id_mode Whether transaction id will be used for the noc transaction (default: DISABLED)
     * @tparam dst_type Whether the write is targeting L1 or a Stream Register
     * @tparam response_mode Posted noc transactions do not get ack from receiver, non-posted ones do (default:
     * NON_POSTED)
     */
    template <
        TxnIdMode txn_id_mode = TxnIdMode::DISABLED,
        InlineWriteDst dst_type = InlineWriteDst::DEFAULT,
        ResponseMode response_mode = ResponseMode::NON_POSTED,
        typename Dst>
    void inline_dw_write(
        const Dst& dst,
        uint32_t val,
        const dst_args_t<Dst>& dst_args,
        uint8_t be = 0xF,
        uint32_t vc = NOC_UNICAST_WRITE_VC,
        uint32_t trid = INVALID_TXN_ID) const {
        static_assert(txn_id_mode == TxnIdMode::DISABLED);
        static_assert(!std::is_same_v<Dst, MulticastEndpoint>);  // Can be removed when #30023 is resolved
        WAYPOINT("NWIW");
        auto dst_addr = get_dst_ptr<AddressType::NOC>(dst, dst_args);
        DEBUG_SANITIZE_NOC_ADDR(noc_id_, dst_addr, 4);
        DEBUG_SANITIZE_NO_DRAM_ADDR(noc_id_, dst_addr, 4);
#if defined(ARCH_BLACKHOLE) && defined(WATCHER_ENABLED)
        if constexpr (dst_type == InlineWriteDst::L1) {
            uint32_t src_addr = noc_get_interim_inline_value_addr(noc_id_, dst_addr);
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id_, dst_addr, src_addr, 4);
        }
#endif

        noc_fast_write_dw_inline<noc_mode, dst_type>(
            noc_id_,
            write_at_cmd_buf,
            val,
            dst_addr,
            be,
            vc,
            std::is_same_v<Dst, MulticastEndpoint>,
            response_mode == ResponseMode::POSTED);
        WAYPOINT("NWID");
    }

    /** @brief Initiates a read barrier for synchronization.
     *
     * This blocking call waits for all the outstanding enqueued read transactions
     * issued on the current Tensix core to complete.
     * After returning from this call there will be no outstanding read transactions for this noc for the current core.
     *
     * @param trid Transaction ID to wait on for outstanding reads (default: INVALID_TXN_ID for full barrier)
     * @tparam barrier_type Indicates whether to issue a full barrier or on a transaction id
     */
    template <BarrierMode barrier_type = BarrierMode::FULL>
    void async_read_barrier(uint32_t trid = INVALID_TXN_ID) const {
        if constexpr (barrier_type == BarrierMode::FULL) {
            noc_async_read_barrier(noc_id_);
        } else if constexpr (barrier_type == BarrierMode::TXN_ID) {
            ASSERT(trid != INVALID_TXN_ID);
            noc_async_read_barrier_with_trid(trid, noc_id_);
        }
    }

    /** @brief Initiates a write barrier for synchronization.
     *
     * This blocking call waits for all the outstanding enqueued write transactions
     * issued on the current Tensix core to complete.
     * After returning from this call there will be no outstanding write transactions for this noc for the current core.
     *
     * @param trid Transaction ID to wait on for outstanding writes (default: INVALID_TXN_ID for full barrier)
     * @tparam barrier_type Indicates whether to issue a full barrier or on a transaction id
     */
    template <BarrierMode barrier_type = BarrierMode::FULL>
    void async_write_barrier(uint32_t trid = INVALID_TXN_ID) const {
        if constexpr (barrier_type == BarrierMode::FULL) {
            noc_async_write_barrier(noc_id_);
        } else if constexpr (barrier_type == BarrierMode::TXN_ID) {
            ASSERT(trid != INVALID_TXN_ID);
            noc_async_write_barrier_with_trid(trid, noc_id_);
        }
    }

    /** @brief Waits for all outstanding write transactions to be flushed.
     *
     * This blocking call waits for all the outstanding enqueued write transactions
     * issued on the current Tensix core to depart, but will not wait for them to complete.
     * Can wait on posted or non-posted transactions.
     *
     * @param trid Transaction ID to wait on for outstanding writes (default: INVALID_TXN_ID for full barrier)
     * @tparam response_mode Indicates whether to wait for outstanding posted or non-posted transactions (default:
     * NON_POSTED)
     * @tparam barrier_type Indicates whether to issue a full barrier or on a transaction id
     */
    // TODO (#31405): there is no variant of this for transaction ids. Use
    // ncrisc_noc_nonposted_write_with_transaction_id_sent but none for dynamic noc version exists atm.
    template <ResponseMode response_mode = ResponseMode::NON_POSTED, BarrierMode barrier_type = BarrierMode::FULL>
    void async_writes_flushed(uint32_t trid = INVALID_TXN_ID) const {
        if constexpr (response_mode == ResponseMode::POSTED) {
            static_assert(barrier_type == BarrierMode::FULL);
            noc_async_posted_writes_flushed(noc_id_);
        } else {  // ResponseMode::NON_POSTED
            if constexpr (barrier_type == BarrierMode::FULL) {
                noc_async_writes_flushed(noc_id_);
            } else if constexpr (barrier_type == BarrierMode::TXN_ID) {
                static_assert(noc_mode != DM_DYNAMIC_NOC);  // TODO make an issue for this
                ASSERT(trid != INVALID_TXN_ID);
                noc_async_write_flushed_with_trid(trid, noc_id_);
            }
        }
    }

    /** @brief Initiates an atomic barrier for synchronization.
     *
     * This blocking call waits for all the outstanding enqueued atomic transactions
     * issued on the current Tensix core to complete.
     * After returning from this call there will be no outstanding atomic transactions for this noc for the current
     * core.
     */
    void async_atomic_barrier() const { noc_async_atomic_barrier(noc_id_); }

    /** @brief Initiates a full barrier for synchronization.
     *
     * This blocking call waits for all the outstanding read, write and atomic noc transactions
     * issued on the current Tensix core to complete.
     * After returning from this call there will be no outstanding transactions for this noc for the current
     * core.
     */
    void async_full_barrier() const { noc_async_full_barrier(noc_id_); }

private:
    uint8_t noc_id_;
};

}  // namespace experimental
