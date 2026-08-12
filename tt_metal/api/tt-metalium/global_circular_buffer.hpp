// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <tuple>
#include <variant>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

class Buffer;
class IDevice;
class Program;

namespace experimental {

class GlobalCircularBufferImpl;

// Forward declarations for the experimental DRAM-sender extension defined in
// tt-metalium/experimental/global_circular_buffer.hpp. The DRAM-sender feature is an
// opt-in mode that is not part of the public GlobalCircularBuffer API surface; existing
// callers continue to see the original public interface unchanged.
class GlobalCircularBuffer;
enum class SenderCoreType : uint8_t;
namespace global_circular_buffer_dram_sender {
struct GlobalCircularBufferDramSenderInternals;
}  // namespace global_circular_buffer_dram_sender

/**
 * @brief Allocates a global circular buffer in L1 on the device.
 *
 * @param device The device to create the global circular buffer on.
 * @param sender_receiver_core_mapping The mapping of remote sender to remote receiver cores for the circular buffer.
 * @param size Size of the global circular buffer per core in bytes.
 * @param buffer_type Buffer type to store the global circular buffer. Can only be an L1 buffer type.
 * @return The allocated global circular buffer.
 */
GlobalCircularBuffer CreateGlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

class GlobalCircularBuffer {
public:
    GlobalCircularBuffer(const GlobalCircularBuffer&) = default;
    GlobalCircularBuffer& operator=(const GlobalCircularBuffer&) = default;

    GlobalCircularBuffer(GlobalCircularBuffer&&) noexcept = default;
    GlobalCircularBuffer& operator=(GlobalCircularBuffer&&) noexcept = default;

    // Internal constructor (internal use only)
    explicit GlobalCircularBuffer(std::shared_ptr<GlobalCircularBufferImpl> impl);

    const Buffer& cb_buffer() const;

    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    DeviceAddr config_address() const;
    uint32_t size() const;
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping() const;

    // Reflection / hashing hooks used by ttnn device-op attribute machinery.
    static constexpr auto attribute_names =
        std::forward_as_tuple("sender_receiver_core_mapping", "size", "buffer_type");
    std::tuple<std::vector<std::pair<CoreCoord, CoreRangeSet>>, uint32_t, BufferType> attribute_values() const;

    // Internal-only accessors (all_cores(), buffer_address(), get_device()) live on the Impl; see
    // tt_metal/impl/buffers/global_circular_buffer_impl.hpp.
    GlobalCircularBufferImpl& impl() { return *impl_; }
    const GlobalCircularBufferImpl& impl() const { return *impl_; }

private:
    GlobalCircularBuffer(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type = BufferType::L1);

    std::shared_ptr<GlobalCircularBufferImpl> impl_;

    friend GlobalCircularBuffer CreateGlobalCircularBuffer(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type);
    friend struct global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals;
};

/**
 * @brief Creates a Circular Buffer in L1 memory of specified cores using the address space of the
 * global circular bufferand adds it to the program.
 *
 * @param program The program to which the buffer will be added.
 * @param core_spec Specifies the cores where the circular buffer will be configured.
 * @param config Configuration for the circular buffer.
 * @param global_circular_buffer Global circular buffer to use the address space and configuration of.
 * @return CBHandle representing the Circular Buffer ID.
 */
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config,
    const GlobalCircularBuffer& global_circular_buffer);

/**
 * @brief Updates the address of a dynamic global circular buffer.
 *
 * @param program The program containing the circular buffer.
 * @param cb_handle Handle of the circular buffer.
 * @param buffer Dynamically allocated global L1 buffer that shares address space with the circular buffer.
 */
void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const GlobalCircularBuffer& global_circular_buffer);

}  // namespace experimental

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::experimental::GlobalCircularBuffer> {
    std::size_t operator()(const tt::tt_metal::experimental::GlobalCircularBuffer& global_circular_buffer) const;
};

}  // namespace std
