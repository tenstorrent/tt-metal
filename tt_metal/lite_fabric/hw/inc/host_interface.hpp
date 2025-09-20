// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include "tt_metal/hw/inc/utils/utils.h"
#include <tt-metalium/fabric_edm_types.hpp>
#include "tt_metal/lite_fabric/hw/inc/constants.hpp"
#include "tt_metal/lite_fabric/hw/inc/header.hpp"

#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

#include <fmt/ranges.h>
#include <umd/device/types/xy_pair.hpp>
#include <tt-logger/tt-logger.hpp>

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

struct State {
    // Unknown initial state
    static constexpr uint32_t Unknown = 0x0;
    // Indicates that this core was written directly from host
    static constexpr uint32_t EthInitFromHost = 0x00010000;
    // Write kernel to local ethernet cores and wait for ack
    static constexpr uint32_t EthInitLocal = 0x00020000;
    // Wait for ack from connected ethernet core
    static constexpr uint32_t EthHandshakeNeighbour = 0x00030000;
    // Write primary kernel to connected ethernet core and wait for ack
    static constexpr uint32_t EthInitNeighbour = 0x00040000;
    // Wait for ack from local ethernet cores
    static constexpr uint32_t EthHandshakeLocal = 0x00050000;
    // Ready for traffic
    static constexpr uint32_t Ready = 0x1F1F0000;
    // Terminated
    static constexpr uint32_t Terminated = 0xDEAD0000;
    // Initialization Error
    static constexpr uint32_t InitError = 0xE0E0E0E0;

    // Mask the state value to get the heartbeat portion
    static constexpr uint32_t HeartbeatMask = 0x0000FFFF;
    // Mask the state value to get the state portion
    static constexpr uint32_t StateMask = 0xFFFF0000;
};

struct BinaryWritten {
    static constexpr uint32_t Written = 0x8A8A8A8A;
};

struct FabricLiteConfig {
    // Starting address of the Lite Fabric binary to be copied locally and to the neighbour.
    volatile uint32_t binary_addr = 0;

    // Size of the Lite Fabric binary.
    volatile uint32_t binary_size = 0;

    // Bit N is 1 if channel N is an active ethernet core. Relies on eth_chan_to_noc_xy to
    // get the ethernet core coordinate.
    volatile uint32_t eth_chans_mask = 0;

    unsigned char padding0[4]{};

    // Subordinate cores on the same chip increment this value when they are ready. The primary core
    // will stall until this value shows all eth cores are ready.
    volatile uint32_t primary_local_handshake = 0;

    unsigned char padding1[12]{};

    // Becomes 1 when the neighbour is ready
    volatile uint32_t neighbour_handshake = 0;

    unsigned char padding2[10]{};

    // This is the local primary core
    volatile uint16_t is_primary = false;

    volatile uint8_t primary_eth_core_x = 0;

    volatile uint8_t primary_eth_core_y = 0;

    // This is on the MMIO
    volatile uint16_t is_mmio = false;

    volatile uint32_t initial_state = State::Unknown;

    volatile uint32_t current_state = State::Unknown;

    // Set to 1 to enable routing
    volatile uint32_t routing_enabled = 1;
} __attribute__((packed));

static_assert(sizeof(FabricLiteConfig) % 16 == 0);
static_assert(offsetof(FabricLiteConfig, primary_local_handshake) % 16 == 0);
static_assert(offsetof(FabricLiteConfig, neighbour_handshake) % 16 == 0);

class HostToFabricLiteReadEvent {
private:
    inline static std::atomic<uint64_t> event{0};

public:
    static uint64_t get() { return event.load(); }

    static void increment() { event.fetch_add(1); }
};

// Interface for Host to MMIO Lite Fabric
template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
struct HostToFabricLiteInterface {
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
    // The core to process requests
    uint32_t mmio_device_id = 0;
    uint32_t mmio_eth_core_x = 0;
    uint32_t mmio_eth_core_y = 0;

    explicit HostToFabricLiteInterface() = default;

    void init() volatile {
        h2d.sender_host_write_index = 0;
        h2d.receiver_host_read_index = 0;
        d2h.fabric_sender_channel_index = 0;
        d2h.fabric_receiver_channel_index = 0;
    }

    constexpr uint32_t get_max_payload_data_size_bytes() const {
        // Additional 16B to be used only for unaligned reads/writes
        return CHANNEL_BUFFER_SIZE - sizeof(FabricLiteHeader) - 16;
    }
} __attribute__((packed));

struct FabricLiteMemoryMap {
    lite_fabric::FabricLiteConfig config;
    // Values below preserved through initializations
    tt::tt_fabric::EDMChannelWorkerLocationInfo sender_location_info;
    uint32_t sender_flow_control_semaphore{};
    unsigned char padding0[12]{};
    uint32_t sender_connection_live_semaphore{};
    unsigned char padding1[12]{};
    uint32_t worker_semaphore{};
    unsigned char padding2[12]{};
    // This is set to 1 the first time lite fabric is initialized. Subsuquent inits will check if this value is set.
    // If it's set then re-init is not required and the kernel can go directly into the while loop to service channels.
    uint32_t binary_written{};
    uint32_t version_major{};
    uint32_t version_minor{};
    uint32_t version_patch{};
    bool on_mmio_chip{};
    unsigned char padding3[63]{};
    unsigned char sender_channel_buffer[lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0] * lite_fabric::CHANNEL_BUFFER_SIZE]{};
    unsigned char padding4[192]{};
    unsigned char
        receiver_channel_buffer[lite_fabric::RECEIVER_NUM_BUFFERS_ARRAY[0] * lite_fabric::CHANNEL_BUFFER_SIZE]{};
    // L1 address of the service_lite_fabric function
    uint32_t service_lite_fabric_addr{};
    unsigned char padding5[12]{};
    // Must be last because it has members that are only stored on the host
    HostToFabricLiteInterface<lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0], lite_fabric::CHANNEL_BUFFER_SIZE>
        host_interface;

#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))
    // Returns a Host Interface for the tunnel starting at the MMIO core
    static uint32_t get_address();
    static uint32_t get_host_interface_addr();
    static uint32_t get_send_channel_addr();
    static uint32_t get_receiver_channel_addr();
    static uint32_t get_service_channel_func_addr();
#endif
};

static_assert(offsetof(FabricLiteMemoryMap, sender_flow_control_semaphore) % 16 == 0);
static_assert(offsetof(FabricLiteMemoryMap, sender_connection_live_semaphore) % 16 == 0);
static_assert(offsetof(FabricLiteMemoryMap, worker_semaphore) % 16 == 0);
static_assert(offsetof(FabricLiteMemoryMap, sender_channel_buffer) % GLOBAL_ALIGNMENT == 0);
static_assert(offsetof(FabricLiteMemoryMap, receiver_channel_buffer) % GLOBAL_ALIGNMENT == 0);
static_assert(offsetof(FabricLiteMemoryMap, host_interface) % 16 == 0);

}  // namespace lite_fabric
