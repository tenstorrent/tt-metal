// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <tt-metalium/fabric_edm_types.hpp>
#include "lite_fabric_memory_config.h"
#include "hal_types.hpp"
#include "lite_fabric_constants.hpp"
#include "lite_fabric_header.hpp"

#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

#include <fmt/ranges.h>
#include <umd/device/types/xy_pair.h>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "utils/utils.h"

#endif

namespace lite_fabric {

template <size_t LIMIT = 0, typename T>
auto wrap_increment(T val) -> T {
    constexpr bool is_pow2 = LIMIT != 0 && is_power_of_2(LIMIT);
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr (is_pow2) {
        return (val + 1) & (static_cast<T>(LIMIT - 1));
    } else {
        return (val == static_cast<T>(LIMIT - 1)) ? static_cast<T>(0) : static_cast<T>(val + 1);
    }
}

/*

Initialization process for Lite Fabric

    1. Host writes the lite fabric kernel to an arbitrary active ethernet core on MMIO capable chips. This
    is designated as the Primary core with an initial state of ETH_INIT_LOCAL. This core will launch
    lite fabric kernels on other active ethernet cores on the same chip with an initial state of
ETH_INIT_LOCAL_HANDSHAKE.

    2. The primary core will stall for the ETH_INIT_LOCAL_HANDSHAKE cores to be ready

    3. Primary core transitions state to ETH_INIT_NEIGHBOUR. It will launch a primary lite fabric kernel on the eth
device.

    4. Subordinate core transitions state to ETH_INIT_NEIGHBOUR_HANDSHAKE

    5. The primary lite fabric kernel on the eth device will launch lite fabric kernels on other active ethernet cores
on the eth device with an initial state of ETH_INIT_LOCAL_HANDSHAKE

*/

enum class InitState : uint16_t {
    // Unknown initial state
    UNKNOWN = 0,
    // Indicates that this is written directly from host
    ETH_INIT_FROM_HOST,
    // Write kernel to local ethernet cores and wait for ack
    ETH_INIT_LOCAL,
    // Wait for ack from connected ethernet core
    ETH_HANDSHAKE_NEIGHBOUR,
    // Write primary kernel to connected ethernet core and wait for ack
    ETH_INIT_NEIGHBOUR,
    // Wait for ack from local ethernet cores
    ETH_HANDSHAKE_LOCAL,
    // Ready for traffic
    READY,
    // Terminated
    TERMINATED,
};

struct LiteFabricConfig {
    // Starting address of the Lite Fabric binary to be copied locally and to the neighbour.
    volatile uint32_t binary_addr = 0;

    // Size of the Lite Fabric binary.
    volatile uint32_t binary_size = 0;

    // Bit N is 1 if channel N is an active ethernet core. Relies on eth_chan_to_noc_xy to
    // get the ethernet core coordinate.
    volatile uint32_t eth_chans_mask = 0;

    unsigned char padding0[4];

    // Subordinate cores on the same chip increment this value when they are ready. The primary core
    // will stall until this value shows all eth cores are ready.
    volatile uint32_t primary_local_handshake = 0;

    unsigned char padding1[12];

    // Becomes 1 when the neighbour is ready
    volatile uint32_t neighbour_handshake = 0;

    unsigned char padding2[14];

    // This is the local primary core
    volatile uint16_t is_primary = false;

    volatile uint8_t primary_eth_core_x = 0;

    volatile uint8_t primary_eth_core_y = 0;

    // This is on the MMIO
    volatile uint16_t is_mmio = false;

    volatile InitState initial_state = InitState::UNKNOWN;

    volatile InitState current_state = InitState::UNKNOWN;

    // Set to 1 to enable routing
    volatile uint32_t routing_enabled = 1;
} __attribute__((packed));

static_assert(sizeof(LiteFabricConfig) % 16 == 0);
static_assert(offsetof(LiteFabricConfig, primary_local_handshake) % 16 == 0);
static_assert(offsetof(LiteFabricConfig, neighbour_handshake) % 16 == 0);

class HostToLiteFabricReadEvent {
private:
    inline static std::atomic<uint64_t> event{0};

public:
    static uint64_t get() { return event.load(); }

    static void increment() { event.fetch_add(1); }
};

// Interface for Host to MMIO Lite Fabric
template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
struct HostToLiteFabricInterface {
    static constexpr uint32_t k_ConnectedDeviceId = 1;

    // This values are updated by the device and read to the host
    struct DeviceToHost {
        volatile uint8_t fabric_sender_channel_index = 0;
        volatile uint8_t fabric_receiver_channel_index = 0;
    } __attribute((packed)) d2h;

    // These values are updated by the host and written to the device
    struct HostToDevice {
        volatile uint8_t sender_host_write_index = 0;
        volatile uint8_t receiver_host_read_index = 0;
    } __attribute((packed)) h2d;

    // Host only fields
    uint32_t host_interface_on_device_addr = 0;
    uint32_t sender_channel_base = 0;
    uint32_t receiver_channel_base = 0;
    uint32_t eth_barrier_addr = 0;
    uint32_t tensix_barrier_addr = 0;
    uint32_t l1_alignment_bytes = 0;  // Assumed to be 16B

    inline void init() volatile {
        h2d.sender_host_write_index = 0;
        h2d.receiver_host_read_index = 0;
        d2h.fabric_sender_channel_index = 0;
        d2h.fabric_receiver_channel_index = 0;
    }

    constexpr uint32_t get_max_payload_data_size_bytes() const {
        // Additional 16B to be used only for unaligned reads/writes
        return CHANNEL_BUFFER_SIZE - sizeof(LiteFabricHeader) - 16;
    }

    // Host Only Methods below
#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

    uint32_t get_next_send_buffer_slot_address(uint32_t channel_address) const {
        auto buffer_index = h2d.sender_host_write_index;
        return channel_address + buffer_index * CHANNEL_BUFFER_SIZE;
    }

    uint32_t get_next_receiver_buffer_slot_address(uint32_t channel_address) const {
        auto buffer_index = h2d.receiver_host_read_index;
        return channel_address + buffer_index * CHANNEL_BUFFER_SIZE;
    }

    void wait_for_empty_write_slot(tt_cxy_pair virtual_core_sender) {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        uint32_t offset =
            offsetof(HostToLiteFabricInterface, d2h) + offsetof(DeviceToHost, fabric_sender_channel_index);
        do {
            tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                (void*)(reinterpret_cast<uintptr_t>(this) + offset),
                sizeof(uint32_t),
                virtual_core_sender,
                host_interface_on_device_addr + offset);
        } while ((h2d.sender_host_write_index + 1) % NUM_BUFFERS == d2h.fabric_sender_channel_index);
    }

    void wait_for_read_event(tt_cxy_pair virtual_core_sender, uint32_t read_event_addr) {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        tt_driver_atomics::mfence();

        volatile LiteFabricHeader header;
        header.command_fields.noc_read.event = 0;
        const auto expectedOrderId = HostToLiteFabricReadEvent::get();
        log_debug(
            tt::LogMetal,
            "Waiting for read event {} from {} {:#x}",
            expectedOrderId,
            virtual_core_sender.str(),
            read_event_addr);
        while (true) {
            tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                const_cast<void*>(static_cast<volatile void*>(&header)),
                sizeof(LiteFabricHeader),
                virtual_core_sender,
                read_event_addr);
            if (header.command_fields.noc_read.event == expectedOrderId) {
                break;
            } else if (
                header.command_fields.noc_read.event != 0xdeadbeef &&
                header.command_fields.noc_read.event > expectedOrderId) {
                TT_THROW("Read event out of order: {} > {}", header.command_fields.noc_read.event, expectedOrderId);
            }
        };

        HostToLiteFabricReadEvent::increment();
    }

    void barrier(tt_cxy_pair virtual_core_sender) {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        auto soc_d = cluster.get_soc_desc(k_ConnectedDeviceId);
        const auto& eth_cores = soc_d.get_cores(CoreType::ETH, CoordSystem::TRANSLATED);
        const auto& tensix_cores = soc_d.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);

        std::vector<uint32_t> barrier_value{rand(), rand(), rand(), rand()};

        const auto do_barrier = [&](const std::vector<tt::umd::CoreCoord>& virtual_cores,
                                    const std::string& core_type_name,
                                    uint32_t barrier_addr) -> void {
            for (const auto& virtual_core : virtual_cores) {
                const uint64_t dest_noc_upper =
                    (uint64_t(virtual_core.y) << (36 + 6)) | (uint64_t(virtual_core.x) << 36);
                uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)barrier_addr;
                write(
                    barrier_value.data(), barrier_value.size() * sizeof(uint32_t), virtual_core_sender, dest_noc_addr, lite_fabric::edm_to_local_chip_noc);

                std::vector<uint32_t> read_barrier(barrier_value.size(), 0);
                read(read_barrier.data(), barrier_value.size() * sizeof(uint32_t), virtual_core_sender, dest_noc_addr, lite_fabric::edm_to_local_chip_noc);
                TT_FATAL(
                    read_barrier == barrier_value,
                    "Chip memory corruption on {} virtual core {}: barrier value mismatch: Read {} but expected {}",
                    core_type_name,
                    virtual_core.str(),
                    fmt::format("{:#x}", fmt::join(read_barrier, ", ")),
                    fmt::format("{:#x}", fmt::join(barrier_value, ", ")));
            }
        };

        do_barrier(eth_cores, "ethernet", eth_barrier_addr);
        do_barrier(tensix_cores, "tensix", tensix_barrier_addr);
    }

    void send_payload_flush_non_blocking_from_address(
        LiteFabricHeader& header, tt_cxy_pair virtual_core_sender, uint32_t channel_address) {
        if (!header.get_payload_size_excluding_header()) {
            return;
        }
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        uint32_t addr = get_next_send_buffer_slot_address(channel_address);
        if (header.get_base_send_type() == lite_fabric::NocSendTypeEnum::NOC_READ) {
            log_debug(
                tt::LogMetal,
                "Send {}B read payload header address {:#x} source address {:#x} Host IF on Device {:#x}",
                header.get_payload_size_including_header(),
                addr,
                header.command_fields.noc_read.noc_address,
                host_interface_on_device_addr);
        } else {
            log_debug(
                tt::LogMetal,
                "Send {}B write payload header address {:#x} dest address {:#x} Host IF on Device {:#x}",
                header.get_payload_size_including_header(),
                addr,
                header.command_fields.noc_unicast.noc_address,
                host_interface_on_device_addr);
        }
        cluster.write_core(&header, sizeof(LiteFabricHeader), virtual_core_sender, addr);

        cluster.l1_barrier(virtual_core_sender.chip);

        h2d.sender_host_write_index =
            lite_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[0]>(h2d.sender_host_write_index);

        log_debug(tt::LogMetal, "Flushing h2d sender_host_write_index to {}", h2d.sender_host_write_index);
        flush_h2d(virtual_core_sender);
    }

    void send_payload_without_header_non_blocking_from_address(
        void* data, size_t size, tt_cxy_pair virtual_core_sender, uint32_t channel_address) {
        if (!size) {
            return;
        }
        if (size > CHANNEL_BUFFER_SIZE - sizeof(LiteFabricHeader)) {
            throw std::runtime_error("Payload size exceeds channel buffer size");
        }
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        uint32_t addr = get_next_send_buffer_slot_address(channel_address) + sizeof(LiteFabricHeader);
        log_debug(tt::LogMetal, "Send {}B payload only {:#x}", size, addr);
        cluster.write_core(data, size, virtual_core_sender, addr);
    }

    void flush_h2d(tt_cxy_pair virtual_core_sender) {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

        tt_driver_atomics::mfence();

        cluster.write_core(
            (void*)(reinterpret_cast<uintptr_t>(this) + offsetof(HostToLiteFabricInterface, h2d)),
            sizeof(HostToDevice),
            virtual_core_sender,
            host_interface_on_device_addr + offsetof(HostToLiteFabricInterface, h2d));

        cluster.l1_barrier(virtual_core_sender.chip);
    }

    // Only up to max buffer size is supported
    void write(void* mem_ptr, size_t size, tt_cxy_pair sender_core, uint64_t dst_noc_addr, uint8_t noc_index) {
        LiteFabricHeader header;
        header.to_chip_unicast(1);
        header.to_noc_unicast_write(lite_fabric::NocUnicastCommandHeader{dst_noc_addr}, size, noc_index);
        // Unaligned address
        header.unaligned_offset = dst_noc_addr & (l1_alignment_bytes - 1);

        wait_for_empty_write_slot(sender_core);
        send_payload_without_header_non_blocking_from_address(
            mem_ptr, size, sender_core, sender_channel_base + header.unaligned_offset);
        send_payload_flush_non_blocking_from_address(header, sender_core, sender_channel_base);
    }

    void write_any_len(void* mem_ptr, size_t size, tt_cxy_pair sender_core, uint64_t dst_noc_addr, uint8_t noc_index = lite_fabric::edm_to_local_chip_noc) {
        size_t num_pages = size / get_max_payload_data_size_bytes();
        for (size_t i = 0; i < num_pages; i++) {
            write(
                reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(mem_ptr) + i * get_max_payload_data_size_bytes()),
                get_max_payload_data_size_bytes(),
                sender_core,
                dst_noc_addr + i * get_max_payload_data_size_bytes(),
                noc_index);
        }
        // Remaining bytes
        size_t remaining_bytes = size % get_max_payload_data_size_bytes();
        if (remaining_bytes > 0) {
            write(
                reinterpret_cast<void*>(
                    reinterpret_cast<uintptr_t>(mem_ptr) + num_pages * get_max_payload_data_size_bytes()),
                remaining_bytes,
                sender_core,
                dst_noc_addr + num_pages * get_max_payload_data_size_bytes(),
                noc_index);
        }
    }

    void read(void* mem_ptr, size_t size, tt_cxy_pair receiver_core, uint64_t src_noc_addr, uint8_t noc_index = lite_fabric::edm_to_local_chip_noc) {
        LiteFabricHeader header;
        header.to_chip_unicast(1);
        header.to_noc_read(lite_fabric::NocReadCommandHeader{src_noc_addr, HostToLiteFabricReadEvent::get()}, size, noc_index);
        header.unaligned_offset = src_noc_addr & (l1_alignment_bytes - 1);

        uint32_t receiver_header_address = get_next_receiver_buffer_slot_address(receiver_channel_base);
        log_debug(
            tt::LogMetal,
            "Reading data from {} {:#x} unaligned {}",
            receiver_core.str(),
            receiver_header_address,
            header.unaligned_offset);
        uint32_t receiver_data_address = receiver_header_address + sizeof(LiteFabricHeader) + header.unaligned_offset;

        wait_for_empty_write_slot(receiver_core);
        send_payload_flush_non_blocking_from_address(header, receiver_core, sender_channel_base);

        wait_for_read_event(receiver_core, receiver_header_address);

        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            mem_ptr, size, receiver_core, receiver_data_address);

        // Ack to device we read
        h2d.receiver_host_read_index =
            lite_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[0]>(h2d.receiver_host_read_index);
        flush_h2d(receiver_core);
    }

#endif
} __attribute__((packed));

struct LiteFabricMemoryMap {
    lite_fabric::LiteFabricConfig config;
    tt::tt_fabric::EDMChannelWorkerLocationInfo sender_location_info;
    uint32_t sender_flow_control_semaphore;
    unsigned char padding0[12];
    uint32_t sender_connection_live_semaphore;
    unsigned char padding1[12];
    uint32_t worker_semaphore;
    unsigned char padding2[12];
    unsigned char sender_channel_buffer[SENDER_NUM_BUFFERS_ARRAY[0] * CHANNEL_BUFFER_SIZE];
    unsigned char receiver_channel_buffer[RECEIVER_NUM_BUFFERS_ARRAY[0] * CHANNEL_BUFFER_SIZE];
    unsigned char padding3[16];
    // Must be last because it has members that are only stored on the host
    HostToLiteFabricInterface<SENDER_NUM_BUFFERS_ARRAY[0], CHANNEL_BUFFER_SIZE> host_interface;

#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))
    static auto make_host_interface() {
        lite_fabric::HostToLiteFabricInterface<SENDER_NUM_BUFFERS_ARRAY[0], CHANNEL_BUFFER_SIZE> host_interface;
        host_interface.host_interface_on_device_addr = lite_fabric::LiteFabricMemoryMap::get_host_interface_addr();
        host_interface.sender_channel_base = lite_fabric::LiteFabricMemoryMap::get_send_channel_addr();
        host_interface.receiver_channel_base = lite_fabric::LiteFabricMemoryMap::get_receiver_channel_addr();
        host_interface.eth_barrier_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_LITE_BARRIER);
        host_interface.tensix_barrier_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::FABRIC_LITE_BARRIER);
        host_interface.l1_alignment_bytes =
            tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);
        host_interface.init();
        return host_interface;
    }

    static uint32_t get_address() {
        auto addr = LITE_FABRIC_CONFIG_START;
        return addr;
    }

    static uint32_t get_host_interface_addr() {
        return get_address() + offsetof(lite_fabric::LiteFabricMemoryMap, host_interface);
    }

    static uint32_t get_send_channel_addr() {
        return get_address() + offsetof(lite_fabric::LiteFabricMemoryMap, sender_channel_buffer);
    }

    static uint32_t get_receiver_channel_addr() {
        return get_address() + offsetof(lite_fabric::LiteFabricMemoryMap, receiver_channel_buffer);
    }
#endif
};

static_assert(offsetof(LiteFabricMemoryMap, sender_flow_control_semaphore) % 16 == 0);
static_assert(offsetof(LiteFabricMemoryMap, sender_connection_live_semaphore) % 16 == 0);
static_assert(offsetof(LiteFabricMemoryMap, worker_semaphore) % 16 == 0);
static_assert(offsetof(LiteFabricMemoryMap, sender_channel_buffer) % 16 == 0);
static_assert(offsetof(LiteFabricMemoryMap, receiver_channel_buffer) % 16 == 0);
static_assert(offsetof(LiteFabricMemoryMap, host_interface) % 16 == 0);

}  // namespace lite_fabric
