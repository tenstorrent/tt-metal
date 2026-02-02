// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_socket.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/vector_aligned.hpp>
#include <utility>

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal::distributed {

enum class H2DMode : uint8_t {
    HOST_PUSH,    // Host pushes data to device over UMD.
    DEVICE_PULL,  // Device pulls data from host over PCIe. This uses PinnedMemory -> requires systems to have vIOMMU
                  // enabled.
};

class H2DSocket {
public:
    H2DSocket(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& recv_core,
        BufferType buffer_type,
        uint32_t fifo_size,
        H2DMode h2d_mode);
    uint32_t get_page_size() const { return page_size_; }
    uint32_t get_config_buffer_address() const { return config_buffer_->address(); }
    void set_page_size(uint32_t page_size);
    void write(void* data, uint32_t num_pages);
    void barrier();
    std::shared_ptr<tt::tt_metal::vector_aligned<uint32_t>> get_bytes_acked_buffer() const {
        return bytes_acked_buffer_;
    }

private:
    void reserve_bytes(uint32_t num_bytes);
    void push_bytes(uint32_t num_bytes);
    void notify_receiver();

    std::shared_ptr<MeshBuffer> config_buffer_ = nullptr;
    std::shared_ptr<MeshBuffer> data_buffer_ = nullptr;
    MeshCoreCoord recv_core_;
    BufferType buffer_type_ = BufferType::L1;
    uint32_t fifo_size_ = 0;
    uint32_t page_size_ = 0;
    uint32_t bytes_sent_ = 0;
    uint32_t bytes_acked_ = 0;
    uint32_t write_ptr_ = 0;
    uint32_t fifo_curr_size_ = 0;
    uint32_t aligned_data_buf_start_ = 0;
    tt::umd::TlbWindow* receiver_core_tlb_ = nullptr;
    std::unique_ptr<tt::tt_metal::experimental::PinnedMemory> bytes_acked_pinned_memory_ = nullptr;
    std::unique_ptr<tt::tt_metal::experimental::PinnedMemory> data_pinned_memory_ = nullptr;
    std::shared_ptr<tt::tt_metal::vector_aligned<uint32_t>> bytes_acked_buffer_ = nullptr;
    std::shared_ptr<std::vector<uint32_t, tt::stl::aligned_allocator<uint32_t, 64>>> host_data_buffer_ = nullptr;
    H2DMode h2d_mode_ = H2DMode::HOST_PUSH;
};

}  // namespace tt::tt_metal::distributed
