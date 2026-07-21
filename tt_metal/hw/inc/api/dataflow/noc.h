// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "internal/debug/noc_zero_guard.h"

template <typename DSpecT>
class TensorAccessor;

struct MulticastEndpoint;
class CircularBuffer;
class DataflowBuffer;

// Concrete arg struct for the DFB-specific Noc overloads.
// Defined here so noc.h can use it in async_read/async_write specializations defined in
// dataflow_buffer.h which includes noc.h
struct DataflowBufferArgs {
    uint32_t offset_bytes{};
};

template <typename T>
struct noc_traits_t {
    static_assert(sizeof(T) == 0, "NoC transactions are not supported for this type");
};

/**
 * @brief Compile-time bit-flags that control optional NoC transaction behaviours.
 *
 * Flags can be OR-combined at a template call site:
 *   noc.async_write<NocOptions::TXN_ID | NocOptions::CUSTOM_VC>(...)
 *
 * | Flag            |Meaning                                                                                       |
 * |-----------------|----------------------------------------------------------------------------------------------|
 * | TXN_ID          | Tag transaction or barrier with NocOptVals::trid (default: 0)                                   |
 * | POSTED          | Fire-and-forget; no ack expected from receiver (default: false (non-posted))                    |
 * | CUSTOM_VC       | Use NocOptVals::vc instead of default VC                                                        |
 * | MCAST_INCL_SRC  | Multicast loopback: include sender in mcast group (default: false)                           |
 * | INLINE_L1       | inline_dw_write targets L1 memory                                                            |
 * | INLINE_REG      | inline_dw_write targets a stream register                                                    |
 */
enum class NocOptions : uint32_t {
    DEFAULT        = 0,
    TXN_ID         = 1u << 0,
    POSTED         = 1u << 1,
    CUSTOM_VC      = 1u << 2,
    MCAST_INCL_SRC = 1u << 3,
    INLINE_L1      = 1u << 4,
    INLINE_REG     = 1u << 5,
};

constexpr NocOptions operator|(NocOptions a, NocOptions b) noexcept {
    return static_cast<NocOptions>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

constexpr bool has_flag(NocOptions opts, NocOptions flag) noexcept {
    return (static_cast<uint32_t>(opts) & static_cast<uint32_t>(flag)) != 0;
}

/**
 * @brief Runtime values associated with optional NocOptions flags.
 *
 * Fields are only inspected when the corresponding NocOptions flag is set:
 *   - vc  : used when NocOptions::CUSTOM_VC is set
 *   - trid: used when NocOptions::TXN_ID is set
 */
struct NocOptVals {
    uint32_t vc   = NOC_UNICAST_WRITE_VC;
    uint32_t trid = 0;
};

/**
 * @brief Noc class that provides a high-level interface for asynchronous read and write operations.
 *
 * It abstracts the details of source and destination address calculations.
 */
class Noc {
public:
    enum class AddressType { NOC, LOCAL_L1 };

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
     * @param noc_opts Optional NoC parameters: noc_opts.vc used when NocOptions::CUSTOM_VC;
     *                 noc_opts.trid used when NocOptions::TXN_ID (default: {})
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     * @tparam enable_noc_tracing Enable NoC tracing for debugging (default: true)
     */
    template <
        NocOptions opts = NocOptions::DEFAULT,
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
        const NocOptVals& noc_opts = {}) const {
        if constexpr (has_flag(opts, NocOptions::TXN_ID)) {
            noc_async_read_set_trid(noc_opts.trid, noc_id_);
            while (noc_available_transactions(noc_id_, noc_opts.trid) < ((NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2)) {
                // Busy-wait until sufficient transactions are available for the configured transaction ID.
            }
        }
        const uint32_t req_vc = has_flag(opts, NocOptions::CUSTOM_VC)
                                    ? static_cast<uint32_t>(noc_opts.vc)
                                    : NOC_UNICAST_WRITE_VC;
        noc_async_read<max_page_size, enable_noc_tracing>(
            get_src_ptr<AddressType::NOC>(src, src_args),
            get_dst_ptr<AddressType::LOCAL_L1>(dst, dst_args),
            size_bytes,
            noc_id_,
            req_vc);
    }

    /**
     * @brief Sets the stateful registers for an asynchronous read from a specified source.
     *
     * This is used to set up state for async_read_with_state; use async_read instead if state
     * preservation is not needed.
     *
     * When NocOptions::TXN_ID is set, noc_opts.trid is written to the sticky NOC_PACKET_TAG register
     * once here so that every subsequent async_read_with_state<NocOptions::TXN_ID> in the block
     * inherits it.
     *
     * @see async_read_with_state and async_read_barrier.
     *
     * @param src Source object (e.g., TensorAccessor)
     * @param size_bytes Size of the data transfer in bytes
     * @param src_args Additional arguments for source address calculation
     * @param noc_opts Optional NoC parameters: noc_opts.vc used when NocOptions::CUSTOM_VC;
     *                 noc_opts.trid used when NocOptions::TXN_ID (default: {})
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     */
    template <
        NocOptions opts = NocOptions::DEFAULT,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src>
    void set_async_read_state(
        const Src& src, uint32_t size_bytes, const src_args_t<Src>& src_args,
        const NocOptVals& noc_opts = {}) const {
        auto src_noc_addr = get_src_ptr<AddressType::NOC>(src, src_args);
        DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc_id_, DEBUG_SANITIZE_NOC_UNICAST);
        RECORD_NOC_EVENT_WITH_ADDR(
            NocEventType::READ_SET_STATE,
            0,
            src_noc_addr,
            size_bytes,
            has_flag(opts, NocOptions::CUSTOM_VC) ? static_cast<int8_t>(noc_opts.vc) : -1,
            false,
            noc_id_);

        WAYPOINT("NASW");
        ncrisc_noc_read_set_state<noc_mode, max_page_size <= NOC_MAX_BURST_SIZE, has_flag(opts, NocOptions::CUSTOM_VC)>(
            noc_id_, read_cmd_buf, src_noc_addr, size_bytes, noc_opts.vc);
        if constexpr (has_flag(opts, NocOptions::TXN_ID)) {
            noc_async_read_set_trid(noc_opts.trid, noc_id_);
        }
        WAYPOINT("NASD");
    }

    /**
     * @brief Initiates an asynchronous read from a specified source based on previously set state.
     *
     * This must be preceded by a call to set_async_read_state where Src is at the same NoC location.
     *
     * When NocOptions::TXN_ID is set (requires max_page_size <= NOC_MAX_BURST_SIZE), noc_opts.trid
     * must match the trid passed to set_async_read_state. The NOC_PACKET_TAG register is sticky —
     * it was set once in set_async_read_state and is not re-written here.
     *
     * @see set_async_read_state and async_read_barrier.
     *
     * @param src Source object (e.g., TensorAccessor)
     * @param dst Destination object (e.g., local L1 memory)
     * @param size_bytes Size of the data transfer in bytes; must equal the value set in
     *                   set_async_read_state if max_page_size <= NOC_MAX_BURST_SIZE
     * @param src_args Additional arguments for source address calculation
     * @param dst_args Additional arguments for destination address calculation
     * @param noc_opts Optional NoC parameters: noc_opts.vc used when NocOptions::CUSTOM_VC;
     *                 noc_opts.trid used when NocOptions::TXN_ID (default: {})
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     */
    template <
        NocOptions opts = NocOptions::DEFAULT,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src,
        typename Dst>
    void async_read_with_state(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        const NocOptVals& noc_opts = {}) const {
        // TODO (#33966): Need to make sure set state was called and with same template params
        if constexpr (has_flag(opts, NocOptions::TXN_ID)) {
            static_assert(
                max_page_size <= NOC_MAX_BURST_SIZE,
                "NocOptions::TXN_ID for async_read_with_state requires one-packet mode "
                "(max_page_size <= NOC_MAX_BURST_SIZE)");
            noc_async_read_one_packet_with_state_with_trid(
                0,
                (uint32_t)get_src_ptr<AddressType::NOC>(src, src_args),
                get_dst_ptr<AddressType::LOCAL_L1>(dst, dst_args),
                noc_opts.trid,
                noc_id_);
        } else if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
            noc_async_read_one_packet_with_state<true, has_flag(opts, NocOptions::CUSTOM_VC)>(
                (uint32_t)get_src_ptr<AddressType::NOC>(src, src_args),
                get_dst_ptr<AddressType::LOCAL_L1>(dst, dst_args),
                noc_opts.vc,
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
     * @param noc_opts Optional NoC parameters: noc_opts.vc used when NocOptions::CUSTOM_VC;
     *                 noc_opts.trid used when NocOptions::TXN_ID (default: {})
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT)
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     * @tparam enable_noc_tracing Enable NoC tracing for debugging (default: true)
     */
    template <
        NocOptions opts = NocOptions::DEFAULT,
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
        const NocOptVals& noc_opts = {}) const {
        NOC_ASSERT_NOT_ZERO_MODE();  // no NoC write between async_write_zeros and write_zeros_l1_barrier
        constexpr bool posted = has_flag(opts, NocOptions::POSTED);

        if constexpr (has_flag(opts, NocOptions::TXN_ID)) {
            // TODO (#31535): Need to add check in ncrisc_noc_fast_write_any_len to ensure outstanding transaction
            // register does not overflow
            WAYPOINT("NAWW");
            auto src_addr = get_src_ptr<AddressType::LOCAL_L1>(src, src_args);
            auto dst_noc_addr = get_dst_ptr<AddressType::NOC>(dst, dst_args);
            if constexpr (enable_noc_tracing) {
                RECORD_NOC_EVENT_WITH_ADDR(
                    NocEventType::WRITE_WITH_TRID, src_addr, dst_noc_addr, size_bytes, -1, posted, noc_id_);
            }
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id_, dst_noc_addr, src_addr, size_bytes);
            constexpr bool one_packet = max_page_size <= NOC_MAX_BURST_SIZE;
            const uint32_t vc = has_flag(opts, NocOptions::CUSTOM_VC)
                                    ? static_cast<uint32_t>(noc_opts.vc)
                                    : NOC_UNICAST_WRITE_VC;
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
                noc_opts.trid);
            WAYPOINT("NWPD");
        } else {
            const uint32_t vc = has_flag(opts, NocOptions::CUSTOM_VC)
                                    ? static_cast<uint32_t>(noc_opts.vc)
                                    : NOC_UNICAST_WRITE_VC;
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
     * @param linked Whether to link this operation with the next (default: false)
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT).
     *             NocOptions::MCAST_INCL_SRC includes the sender in the multicast group.
     *             NocOptions::TXN_ID is not supported (static_assert).
     *             NocOptions::POSTED is not supported (static_assert).
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     * @tparam enable_noc_tracing Enable NoC tracing for debugging (default: true)
     */
    template <
        NocOptions opts = NocOptions::DEFAULT,
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
        bool linked = false) const {
        static_assert(!has_flag(opts, NocOptions::TXN_ID), "Mcasts with transaction id are not supported yet");
        static_assert(
            !has_flag(opts, NocOptions::POSTED),
            "Mcasts with posted transactions are not supported");  // TODO: Make this an arch specific assertion

        NOC_ASSERT_NOT_ZERO_MODE();  // no NoC write between async_write_zeros and write_zeros_l1_barrier
        auto src_addr = get_src_ptr<AddressType::LOCAL_L1>(src, src_args);
        auto dst_noc_addr = get_dst_ptr_mcast<AddressType::NOC>(dst, dst_args);
        if constexpr (has_flag(opts, NocOptions::MCAST_INCL_SRC)) {
            noc_async_write_multicast_loopback_src(src_addr, dst_noc_addr, size_bytes, num_dsts, linked, noc_id_);
        } else {
            noc_async_write_multicast<max_page_size>(src_addr, dst_noc_addr, size_bytes, num_dsts, linked, noc_id_);
        }
    }

    /**
     * @brief Sets the stateful registers for an asynchronous write.
     *
     * This function is used to set up the state for async_write_with_state; async_write can be used
     * if state preservation is not needed.
     *
     * NocOptions::TXN_ID is not supported for stateful writes (no underlying 1.0 primitive exists);
     * use async_write<NocOptions::TXN_ID> for non-stateful writes with a transaction ID.
     *
     * @see async_write_with_state and async_write_barrier.
     *
     * @param dst Destination object (e.g., local L1 memory)
     * @param size_bytes Size of the data transfer in bytes
     * @param dst_args Additional arguments for destination address calculation
     * @param noc_opts Optional NoC parameters: noc_opts.vc used when NocOptions::CUSTOM_VC (default: {})
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT).
     *             NocOptions::TXN_ID is not supported (static_assert).
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     */
    template <
        NocOptions opts = NocOptions::DEFAULT,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Dst>
    void set_async_write_state(
        const Dst& dst, uint32_t size_bytes, const dst_args_t<Dst>& dst_args,
        const NocOptVals& noc_opts = {}) const {
        static_assert(
            !has_flag(opts, NocOptions::TXN_ID),
            "NocOptions::TXN_ID is not supported for set_async_write_state; "
            "use async_write<NocOptions::TXN_ID> for non-stateful writes with a transaction ID");
        constexpr bool posted = has_flag(opts, NocOptions::POSTED);
        NOC_ASSERT_NOT_ZERO_MODE();  // no cmd-buffer-0 op between async_write_zeros and write_zeros_l1_barrier
        DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc_id_, DEBUG_SANITIZE_NOC_UNICAST);
        auto dst_noc_addr = get_dst_ptr<AddressType::NOC>(dst, dst_args);
        RECORD_NOC_EVENT_WITH_ADDR(
            NocEventType::WRITE_SET_STATE, 0, dst_noc_addr, size_bytes, noc_opts.vc, posted, noc_id_);

        WAYPOINT("NWPW");
        ncrisc_noc_write_set_state<posted, max_page_size <= NOC_MAX_BURST_SIZE>(
            noc_id_, write_cmd_buf, dst_noc_addr, size_bytes, noc_opts.vc);
        WAYPOINT("NWPD");
    }

    /**
     * @brief Initiates an asynchronous write to a specified destination based on previously set state.
     *
     * This must be preceded by a call to set_async_write_state where Dst is at the same NoC location.
     *
     * NocOptions::TXN_ID is not supported for stateful writes. The noc_opts parameter is accepted for
     * API symmetry with async_read_with_state but its vc field is not used (vc is sticky from
     * set_async_write_state).
     *
     * @see set_async_write_state and async_write_barrier.
     *
     * @param src Source object (e.g., local L1 memory)
     * @param dst Destination object (e.g., TensorAccessor)
     * @param size_bytes Size of the data transfer in bytes; must equal the value set in
     *                   set_async_write_state if max_page_size <= NOC_MAX_BURST_SIZE
     * @param src_args Additional arguments for source address calculation
     * @param dst_args Additional arguments for destination address calculation
     * @param noc_opts Optional NoC parameters (default: {}); accepted for symmetry, vc is sticky from
     *                 set_async_write_state
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT).
     *             NocOptions::TXN_ID is not supported (static_assert).
     * @tparam max_page_size Maximum page size for the transfer (default: NOC_MAX_BURST_SIZE + 1)
     */
    template <
        NocOptions opts = NocOptions::DEFAULT,
        uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src,
        typename Dst>
    void async_write_with_state(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        const NocOptVals& noc_opts = {}) const {
        static_assert(
            !has_flag(opts, NocOptions::TXN_ID),
            "NocOptions::TXN_ID is not supported for async_write_with_state; "
            "use async_write<NocOptions::TXN_ID> for non-stateful writes with a transaction ID");
        constexpr bool posted = has_flag(opts, NocOptions::POSTED);
        NOC_ASSERT_NOT_ZERO_MODE();  // no cmd-buffer-0 op between async_write_zeros and write_zeros_l1_barrier

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
            RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_STATE, src_addr, 0ull, 0, -1, posted, noc_id_);
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
     * @param noc_opts Optional NoC parameters: noc_opts.vc used when NocOptions::CUSTOM_VC (default: {})
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT).
     *             NocOptions::TXN_ID is not supported (static_assert).
     *             NocOptions::INLINE_L1 targets L1; otherwise targets stream register.
     *             NocOptions::POSTED uses fire-and-forget semantics.
     */
    template <NocOptions opts = NocOptions::DEFAULT, typename Dst>
    void inline_dw_write(
        const Dst& dst,
        uint32_t val,
        const dst_args_t<Dst>& dst_args,
        uint8_t be = 0xF,
        const NocOptVals& noc_opts = {}) const {
        static_assert(!has_flag(opts, NocOptions::TXN_ID), "TxnId is not supported for inline_dw_write");
        static_assert(
            !(has_flag(opts, NocOptions::INLINE_L1) && has_flag(opts, NocOptions::INLINE_REG)),
            "INLINE_L1 and INLINE_REG are mutually exclusive in inline_dw_write");
        static_assert(!std::is_same_v<Dst, MulticastEndpoint>);  // Can be removed when #30023 is resolved
        WAYPOINT("NWIW");
        auto dst_addr = get_dst_ptr<AddressType::NOC>(dst, dst_args);
        DEBUG_SANITIZE_NOC_ADDR(noc_id_, dst_addr, 4);
        DEBUG_SANITIZE_NO_DRAM_ADDR(noc_id_, dst_addr, 4);

        constexpr auto dst_type = has_flag(opts, NocOptions::INLINE_L1)  ? InlineWriteDst::L1
                                : has_flag(opts, NocOptions::INLINE_REG) ? InlineWriteDst::REG
                                                                         : InlineWriteDst::DEFAULT;
#if defined(ARCH_BLACKHOLE) && defined(WATCHER_ENABLED)
        if constexpr (has_flag(opts, NocOptions::INLINE_L1)) {
            uint32_t src_addr = noc_get_interim_inline_value_addr(noc_id_, dst_addr);
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id_, dst_addr, src_addr, 4);
        }
#endif

        const uint32_t vc = has_flag(opts, NocOptions::CUSTOM_VC)
                                ? static_cast<uint32_t>(noc_opts.vc)
                                : NOC_UNICAST_WRITE_VC;
        noc_fast_write_dw_inline<noc_mode, dst_type>(
            noc_id_,
            write_at_cmd_buf,
            val,
            dst_addr,
            be,
            vc,
            std::is_same_v<Dst, MulticastEndpoint>,
            has_flag(opts, NocOptions::POSTED));
        WAYPOINT("NWID");
    }

    /**
     * @brief Non-blocking poll: returns true if all reads tagged with the given transaction ID have completed.
     *
     * Equivalent to checking whether the hardware outstanding-reads counter for trid has reached zero.
     * Intended for implementing non-blocking, per-slot progress tracking — issue reads with different trids,
     * then poll each slot independently rather than stalling on a full barrier.
     *
     * @param trid Transaction ID to check (must match what was passed to async_read<NocOptions::TXN_ID>)
     */
    bool is_read_trid_flushed(uint32_t trid) const {
        return ncrisc_noc_read_with_transaction_id_flushed(noc_id_, trid);
    }

    /** @brief Waits for all outstanding read transactions to complete.
     *
     * This blocking call waits for all the outstanding enqueued read transactions
     * issued on the current Tensix core to complete.
     * After returning from this call there will be no outstanding read transactions for this noc for the current core.
     *
     * When NocOptions::TXN_ID is set, only waits for reads tagged with noc_opts.trid.
     *
     * @param noc_opts Optional NoC parameters: noc_opts.trid used when NocOptions::TXN_ID (default: {})
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT)
     */
    template <NocOptions opts = NocOptions::DEFAULT>
    void async_read_barrier(const NocOptVals& noc_opts = {}) const {
        if constexpr (has_flag(opts, NocOptions::TXN_ID)) {
            noc_async_read_barrier_with_trid(noc_opts.trid, noc_id_);
        } else {
            noc_async_read_barrier(noc_id_);
        }
    }

    /** @brief Waits for all outstanding write transactions to complete.
     *
     * This blocking call waits for all the outstanding enqueued write transactions
     * issued on the current Tensix core to complete.
     * After returning from this call there will be no outstanding write transactions for this noc for the current core.
     *
     * When NocOptions::TXN_ID is set, only waits for writes tagged with noc_opts.trid.
     *
     * @param noc_opts Optional NoC parameters: noc_opts.trid used when NocOptions::TXN_ID (default: {})
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT)
     */
    template <NocOptions opts = NocOptions::DEFAULT>
    void async_write_barrier(const NocOptVals& noc_opts = {}) const {
        if constexpr (has_flag(opts, NocOptions::TXN_ID)) {
            noc_async_write_barrier_with_trid(noc_opts.trid, noc_id_);
        } else {
            noc_async_write_barrier(noc_id_);
        }
    }

    /** @brief Waits for all outstanding write transactions to be flushed (departed, not necessarily completed).
     *
     * This blocking call waits for all the outstanding enqueued write transactions
     * issued on the current Tensix core to depart, but will not wait for them to complete.
     *
     * When NocOptions::POSTED is set, waits for posted (fire-and-forget) writes to flush.
     * When NocOptions::TXN_ID is set, only waits for writes tagged with noc_opts.trid.
     *
     * @param noc_opts Optional NoC parameters: noc_opts.trid used when NocOptions::TXN_ID (default: {})
     * @tparam opts Bit-flag combination of NocOptions (default: DEFAULT)
     */
    // TODO (#31405): there is no variant of this for transaction ids. Use
    // ncrisc_noc_nonposted_write_with_transaction_id_sent but none for dynamic noc version exists atm.
    template <NocOptions opts = NocOptions::DEFAULT>
    void async_writes_flushed(const NocOptVals& noc_opts = {}) const {
        if constexpr (has_flag(opts, NocOptions::POSTED)) {
            static_assert(!has_flag(opts, NocOptions::TXN_ID), "Posted writes flushed does not support TXN_ID");
            noc_async_posted_writes_flushed(noc_id_);
        } else {  // non-posted
            if constexpr (has_flag(opts, NocOptions::TXN_ID)) {
                static_assert(noc_mode != DM_DYNAMIC_NOC);  // TODO make an issue for this
                noc_async_write_flushed_with_trid(noc_opts.trid, noc_id_);
            } else {
                noc_async_writes_flushed(noc_id_);
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

    /**
     * @brief Zeroes a local-L1 destination buffer (overload 1).
     *
     * @note Quasar: this temporarily reprograms the overlay write command buffer (cmd buffer 0)
     *       into iDMA zero mode; it is restored to normal write mode only by
     *       write_zeros_l1_barrier(). Do NOT issue any other NOC write (noc.async_write /
     *       noc_async_write, also cmd buffer 0) on the same core between this call and
     *       write_zeros_l1_barrier() -- those writes would run in zero mode and corrupt their
     *       data. Barrier first, then reuse cmd buffer 0.
     *
     * @see write_zeros_l1_barrier.
     *
     * @param dst Destination object (CircularBuffer or DataflowBuffer)
     * @param size_bytes Number of bytes to zero
     * @param args Additional arguments for destination address calculation (offset within @p dst)
     * @tparam Dst Must be CircularBuffer or DataflowBuffer
     */
    template <typename Dst>
    void async_write_zeros(const Dst& dst, uint32_t size_bytes, const dst_args_t<Dst>& args = {}) const;

    /**
     * @brief Zeroes pages of a DRAM tensor using a caller-pre-zeroed scratch buffer (overload 2).
     *
     * The source bytes are read starting at @p scratch's current READ pointer.
     * Reads up to NOC_MAX_BURST_SIZE bytes per chunk, so the zeroed prefix at that read pointer
     * must cover at least min(@p size_bytes, NOC_MAX_BURST_SIZE) bytes; otherwise the impl reads
     * garbage past the zero region and streams it to DRAM.
     *
     * Contract: the zeroed bytes must sit at @p scratch's read pointer. Two valid patterns:
     *   - Same-kernel scratch: a fresh/empty CB or DFB has read_ptr == write_ptr, so zeroing it
     *     via overload (1) (which writes the write pointer) lands where overload (2) reads.
     *   - Producer/consumer handoff: the producer zeroes via overload (1) then push_back()s the
     *     entry; the consumer wait_front()s it before passing it here.
     *
     * Caller MUST zero the scratch via overload (1) + write_zeros_l1_barrier() before the first call.
     *
     * Each call zeroes within a single page: @p args.offset_bytes + @p size_bytes must not exceed
     * the accessor's aligned page size, otherwise the write spills into a neighbouring page.
     *
     * @see write_zeros_dram_barrier.
     *
     * @param accessor Destination DRAM tensor accessor
     * @param size_bytes Number of bytes to zero per page
     * @param args Destination page args (page_id, offset_bytes)
     * @param scratch Pre-zeroed L1 scratch buffer (CircularBuffer or DataflowBuffer); read at its read pointer
     * @tparam DSpecT TensorAccessor type spec; must satisfy DSpecT::is_dram
     * @tparam Scratch Must be CircularBuffer or DataflowBuffer
     *
     * @code
     *   // Same-kernel scratch: fresh CB, so read_ptr == write_ptr.
     *   noc.async_write_zeros(scratch, scratch_bytes);
     *   noc.write_zeros_l1_barrier();
     *   for (p) noc.async_write_zeros(addr_gen, page_size, {.page_id = p}, scratch);
     *   noc.write_zeros_dram_barrier();
     * @endcode
     */
    template <typename DSpecT, typename Scratch>
    void async_write_zeros(
        const ::TensorAccessor<DSpecT>& accessor,
        uint32_t size_bytes,
        const dst_args_t<::TensorAccessor<DSpecT>>& args,
        const Scratch& scratch) const;

    /**
     * @brief Barrier for L1 destinations zeroed via async_write_zeros overload (1).
     *
     * @see async_write_zeros (overload 1).
     */
    void write_zeros_l1_barrier() const;

    /**
     * @brief Barrier for DRAM destinations zeroed via async_write_zeros overload (2).
     *
     * @see async_write_zeros (overload 2).
     */
    void write_zeros_dram_barrier() const;

#ifdef ARCH_QUASAR
    /**
     * @brief Implicit-sync read into a DataflowBuffer.
     *
     * Selects this overload when NocOptions::TXN_ID is specified and the destination is a DataflowBuffer.
     * No trid is accepted here because the DataflowBuffer manages txn_ids internally
     * via its private prepare/commit helpers.
     * Size of the read is not accepted here because the DataflowBuffer provides parameters for the read internally.
     */
    template <NocOptions opts, typename Src>
    std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
    async_read(
        const Src& src,
        DataflowBuffer& dst,
        const src_args_t<Src>& src_args,
        const DataflowBufferArgs& dst_args = {}) const;

    /**
     * @brief Implicit-sync write from a DataflowBuffer.
     *
     * Selects this overload when NocOptions::TXN_ID is specified and the source is a DataflowBuffer.
     * No trid is accepted here because the DataflowBuffer manages txn_ids internally
     * via its private prepare/commit helpers.
     * Size of the write is not accepted here because the DataflowBuffer provides parameters for the write internally.
     */
    template <NocOptions opts, typename Dst>
    std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
    async_write(
        DataflowBuffer& src,
        const Dst& dst,
        const DataflowBufferArgs& src_args,
        const dst_args_t<Dst>& dst_args) const;
#endif

private:
    uint8_t noc_id_;
};
