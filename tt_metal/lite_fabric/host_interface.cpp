// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/lite_fabric/hw/inc/lf_dev_mem_map.hpp"
#include "tt_metal/lite_fabric/hw/inc/host_interface.hpp"
#include <thread>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/api/tt-metalium/hal_types.hpp"

namespace lite_fabric {

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
tt_cxy_pair HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::get_mmio_eth_core() const {
    return {mmio_device_id, {mmio_eth_core_x, mmio_eth_core_y}};
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
uint32_t HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::get_next_send_buffer_slot_address(
    uint32_t channel_address) const {
    auto buffer_index = h2d.sender_host_write_index;
    return channel_address + buffer_index * CHANNEL_BUFFER_SIZE;
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
uint32_t HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::get_next_receiver_buffer_slot_address(
    uint32_t channel_address) const {
    auto buffer_index = h2d.receiver_host_read_index;
    return channel_address + buffer_index * CHANNEL_BUFFER_SIZE;
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::wait_for_empty_write_slot() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    uint32_t offset = offsetof(HostToFabricLiteInterface, d2h);
    do {
        log_debug(
            tt::LogMetal,
            "Waiting for empty write slot {} {}",
            h2d.sender_host_write_index,
            d2h.fabric_sender_channel_index);
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            (void*)(reinterpret_cast<uintptr_t>(this) + offset),
            sizeof(DeviceToHost),
            get_mmio_eth_core(),
            host_interface_on_device_addr + offset);
    } while ((h2d.sender_host_write_index + 1) % NUM_BUFFERS == d2h.fabric_sender_channel_index);
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::wait_for_read_event(uint32_t read_event_addr) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    tt_driver_atomics::mfence();

    volatile FabricLiteHeader header{};
    header.command_fields.noc_read.event = 0;
    const auto expectedOrderId = HostToFabricLiteReadEvent::get();
    log_debug(
        tt::LogMetal,
        "Waiting for read event {} from {} {:#x}",
        expectedOrderId,
        get_mmio_eth_core().str(),
        read_event_addr);
    while (true) {
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            const_cast<void*>(static_cast<volatile void*>(&header)),
            sizeof(FabricLiteHeader),
            get_mmio_eth_core(),
            read_event_addr);
        if (header.command_fields.noc_read.event == expectedOrderId) {
            break;
        } else if (
            header.command_fields.noc_read.event != 0xdeadbeef &&
            header.command_fields.noc_read.event > expectedOrderId) {
            TT_THROW("Read event out of order: {} > {}", header.command_fields.noc_read.event, expectedOrderId);
        }
    };

    HostToFabricLiteReadEvent::increment();
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::barrier() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto soc_d = cluster.get_soc_desc(k_ConnectedDeviceId);
    const auto& eth_cores = soc_d.get_cores(CoreType::ETH, CoordSystem::TRANSLATED);
    const auto& tensix_cores = soc_d.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);

    std::vector<uint32_t> barrier_value{rand(), rand()};

    const auto do_barrier = [&](const std::vector<tt::umd::CoreCoord>& virtual_cores,
                                const std::string& core_type_name,
                                uint32_t barrier_addr) -> void {
        for (const auto& virtual_core : virtual_cores) {
            const uint64_t dest_noc_upper = (uint64_t(virtual_core.y) << (36 + 6)) | (uint64_t(virtual_core.x) << 36);
            uint64_t dest_noc_addr = dest_noc_upper | (uint64_t)barrier_addr;
            write(
                barrier_value.data(),
                barrier_value.size() * sizeof(uint32_t),
                dest_noc_addr,
                lite_fabric::edm_to_local_chip_noc);

            std::vector<uint32_t> read_barrier(barrier_value.size(), 0);
            read(
                read_barrier.data(),
                barrier_value.size() * sizeof(uint32_t),
                dest_noc_addr,
                lite_fabric::edm_to_local_chip_noc);
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

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::send_payload_flush_non_blocking_from_address(
    FabricLiteHeader& header, uint32_t channel_address) {
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
    cluster.write_core(&header, sizeof(FabricLiteHeader), get_mmio_eth_core(), addr);

    cluster.l1_barrier(get_mmio_eth_core().chip);

    h2d.sender_host_write_index = lite_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[0]>(h2d.sender_host_write_index);

    log_debug(tt::LogMetal, "Flushing h2d sender_host_write_index to {}", h2d.sender_host_write_index);
    flush_h2d();
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::send_payload_without_header_non_blocking_from_address(
    void* data, size_t size, uint32_t channel_address) {
    if (!size) {
        return;
    }
    if (size > CHANNEL_BUFFER_SIZE - sizeof(FabricLiteHeader)) {
        throw std::runtime_error("Payload size exceeds channel buffer size");
    }
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    uint32_t addr = get_next_send_buffer_slot_address(channel_address) + sizeof(FabricLiteHeader);
    log_debug(tt::LogMetal, "Send {}B payload only {:#x}", size, addr);
    cluster.write_core(data, size, get_mmio_eth_core(), addr);
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::flush_h2d() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    tt_driver_atomics::mfence();

    cluster.write_core(
        (void*)(reinterpret_cast<uintptr_t>(this) + offsetof(HostToFabricLiteInterface, h2d)),
        sizeof(HostToDevice),
        get_mmio_eth_core(),
        host_interface_on_device_addr + offsetof(HostToFabricLiteInterface, h2d));

    cluster.l1_barrier(get_mmio_eth_core().chip);
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::write(
    void* mem_ptr, size_t size, uint64_t dst_noc_addr, uint8_t noc_index) {
    FabricLiteHeader header{};
    header.to_chip_unicast(1);
    header.to_noc_unicast_write(lite_fabric::NocUnicastCommandHeader{dst_noc_addr}, size, noc_index);
    // Unaligned address
    header.unaligned_offset = dst_noc_addr & (l1_alignment_bytes - 1);

    wait_for_empty_write_slot();
    send_payload_without_header_non_blocking_from_address(mem_ptr, size, sender_channel_base + header.unaligned_offset);
    send_payload_flush_non_blocking_from_address(header, sender_channel_base);
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::write_any_len(
    void* mem_ptr, size_t size, uint64_t dst_noc_addr, uint8_t noc_index) {
    size_t num_pages = size / get_max_payload_data_size_bytes();
    for (size_t i = 0; i < num_pages; i++) {
        write(
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(mem_ptr) + i * get_max_payload_data_size_bytes()),
            get_max_payload_data_size_bytes(),
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
            dst_noc_addr + num_pages * get_max_payload_data_size_bytes(),
            noc_index);
    }
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::read(
    void* mem_ptr, size_t size, uint64_t src_noc_addr, uint8_t noc_index) {
    FabricLiteHeader header{};
    header.to_chip_unicast(1);
    header.to_noc_read(
        lite_fabric::NocReadCommandHeader{src_noc_addr, HostToFabricLiteReadEvent::get()}, size, noc_index);
    // The device will calculate the proper alignment offset, so we just initialize this to 0
    header.unaligned_offset = 0;

    uint32_t receiver_header_address = get_next_receiver_buffer_slot_address(receiver_channel_base);
    log_debug(tt::LogMetal, "Reading data from {} {:#x}", get_mmio_eth_core().str(), receiver_header_address);
    // Base data address is immediately after header - device will handle alignment
    uint32_t receiver_data_address = receiver_header_address + sizeof(FabricLiteHeader);

    wait_for_empty_write_slot();
    send_payload_flush_non_blocking_from_address(header, sender_channel_base);

    wait_for_read_event(receiver_header_address);

    // Read back the alignment offset that the device calculated and stored in the header
    uint8_t read_back_unaligned_offset = 0;
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        &read_back_unaligned_offset,
        sizeof(uint8_t),
        get_mmio_eth_core(),
        receiver_header_address + offsetof(FabricLiteHeader, unaligned_offset));

    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        mem_ptr, size, get_mmio_eth_core(), receiver_data_address + read_back_unaligned_offset);

    // Ack to device we read
    h2d.receiver_host_read_index =
        lite_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[0]>(h2d.receiver_host_read_index);
    flush_h2d();
}

template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::read_any_len(
    void* mem_ptr, size_t size, uint64_t src_noc_addr, uint8_t noc_index) {
    size_t num_pages = size / get_max_payload_data_size_bytes();
    for (size_t i = 0; i < num_pages; i++) {
        read(
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(mem_ptr) + i * get_max_payload_data_size_bytes()),
            get_max_payload_data_size_bytes(),
            src_noc_addr + i * get_max_payload_data_size_bytes(),
            noc_index);
    }
    // Remaining bytes
    size_t remaining_bytes = size % get_max_payload_data_size_bytes();
    if (remaining_bytes > 0) {
        read(
            reinterpret_cast<void*>(
                reinterpret_cast<uintptr_t>(mem_ptr) + num_pages * get_max_payload_data_size_bytes()),
            remaining_bytes,
            src_noc_addr + num_pages * get_max_payload_data_size_bytes(),
            noc_index);
    }
}

// Write the register of the connected ethernet core directly from the sender using the Ethernet Dataflow API.
// If you need to write registers to cores on the receiver chip, use write() instead
template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::write_reg(uint32_t reg_address, uint32_t reg_value) {
    FabricLiteHeader header{};
    header.to_write_reg(lite_fabric::WriteRegCommandHeader{reg_address, reg_value});

    wait_for_empty_write_slot();
    send_payload_flush_non_blocking_from_address(header, sender_channel_base);
}

// Wait for device to send requests. Does not guarantee that the requests have been processed by the destination
// core.
template <uint32_t NUM_BUFFERS, uint32_t CHANNEL_BUFFER_SIZE>
void HostToFabricLiteInterface<NUM_BUFFERS, CHANNEL_BUFFER_SIZE>::finish() {
    uint32_t offset = offsetof(HostToFabricLiteInterface, d2h);
    do {
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            (void*)(reinterpret_cast<uintptr_t>(this) + offset),
            sizeof(DeviceToHost),
            get_mmio_eth_core(),
            host_interface_on_device_addr + offset);
    } while (h2d.sender_host_write_index != d2h.fabric_sender_channel_index);
}

// Returns a Host Interface for the tunnel starting at the MMIO core
HostToFabricLiteInterface<lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0], lite_fabric::CHANNEL_BUFFER_SIZE>
FabricLiteMemoryMap::make_host_interface(const tt_cxy_pair& mmio_core) {
    HostToFabricLiteInterface<lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0], lite_fabric::CHANNEL_BUFFER_SIZE>
        host_interface;
    host_interface.host_interface_on_device_addr = lite_fabric::FabricLiteMemoryMap::get_host_interface_addr();
    host_interface.sender_channel_base = lite_fabric::FabricLiteMemoryMap::get_send_channel_addr();
    host_interface.receiver_channel_base = lite_fabric::FabricLiteMemoryMap::get_receiver_channel_addr();
    host_interface.eth_barrier_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::BARRIER);
    host_interface.tensix_barrier_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BARRIER);
    host_interface.l1_alignment_bytes = GLOBAL_ALIGNMENT;
    host_interface.mmio_device_id = mmio_core.chip;
    host_interface.mmio_eth_core_x = mmio_core.x;
    host_interface.mmio_eth_core_y = mmio_core.y;
    host_interface.init();
    return host_interface;
}

uint32_t FabricLiteMemoryMap::get_address() {
    auto addr = LITE_FABRIC_CONFIG_START;
    return addr;
}

uint32_t FabricLiteMemoryMap::get_host_interface_addr() {
    return get_address() + offsetof(lite_fabric::FabricLiteMemoryMap, host_interface);
}

uint32_t FabricLiteMemoryMap::get_send_channel_addr() {
    return get_address() + offsetof(lite_fabric::FabricLiteMemoryMap, sender_channel_buffer);
}

uint32_t FabricLiteMemoryMap::get_receiver_channel_addr() {
    return get_address() + offsetof(lite_fabric::FabricLiteMemoryMap, receiver_channel_buffer);
}

uint32_t FabricLiteMemoryMap::get_service_channel_func_addr() {
    return get_address() + offsetof(lite_fabric::FabricLiteMemoryMap, service_lite_fabric_addr);
}

template class HostToFabricLiteInterface<lite_fabric::SENDER_NUM_BUFFERS_ARRAY[0], lite_fabric::CHANNEL_BUFFER_SIZE>;

}  // namespace lite_fabric
