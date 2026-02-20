// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/vector_aligned.hpp>
#include <utility>

namespace tt::tt_metal::distributed {

class D2HSocket {
public:
    D2HSocket(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& sender_core,
        BufferType buffer_type,
        uint32_t fifo_size,
        uint32_t l1_data_buffer_size = 0);

    void wait_for_pages(uint32_t num_pages);
    uint32_t pages_available();
    void pop_pages(uint32_t num_pages);
    void notify_sender();
    uint32_t get_page_size() const { return page_size_; }
    uint32_t get_config_buffer_address() const { return config_buffer_->address(); }
    uint32_t* get_read_ptr() const;
    void set_page_size(uint32_t page_size);
    void barrier();

    uint32_t get_l1_data_buffer_address() const { return l1_data_buffer_address_; }
    uint32_t get_l1_data_buffer_size() const { return l1_data_buffer_size_; }

private:
    std::shared_ptr<MeshBuffer> config_buffer_ = nullptr;
    std::shared_ptr<MeshBuffer> l1_data_buffer_ = nullptr;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> data_pinned_memory_ = nullptr;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> bytes_sent_pinned_memory_ = nullptr;
    std::shared_ptr<tt::tt_metal::vector_aligned<uint32_t>> data_buffer_ = nullptr;
    std::shared_ptr<tt::tt_metal::vector_aligned<uint32_t>> bytes_sent_buffer_ = nullptr;

    MeshCoreCoord sender_core_ = {};
    uint32_t fifo_size_ = 0;
    uint32_t page_size_ = 0;
    uint32_t bytes_acked_ = 0;
    uint32_t bytes_sent_ = 0;
    uint32_t read_ptr_ = 0;
    uint32_t fifo_curr_size_ = 0;
    uint32_t l1_data_buffer_address_ = 0;
    uint32_t l1_data_buffer_size_ = 0;

    // Hugepage fallback (used on Wormhole when IOMMU is unavailable)
    bool using_hugepage_ = false;
    uint32_t* hugepage_data_host_ptr_ = nullptr;
    volatile uint32_t* hugepage_bytes_sent_host_ptr_ = nullptr;
};

}  // namespace tt::tt_metal::distributed
