// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "models/demos/deepseek_v3_b1/pipeline_manager/socket_pipeline.hpp"

#include <atomic>
#include <chrono>
#include <thread>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>

#include "models/demos/deepseek_v3_b1/pipeline_manager/wire_format.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

struct SocketPipeline::Impl {
    std::unique_ptr<tt::tt_metal::distributed::H2DSocket> h2d_socket;
    std::unique_ptr<tt::tt_metal::distributed::D2HSocket> d2h_socket;
    PageBuffer write_buf = {};
    PageBuffer read_buf = {};
    std::atomic<bool> stop_requested{false};

    Impl(const std::string& h2d_socket_id, const std::string& d2h_socket_id, uint32_t connect_timeout_ms) {
        h2d_socket = tt::tt_metal::distributed::H2DSocket::connect(h2d_socket_id, connect_timeout_ms);
        d2h_socket = tt::tt_metal::distributed::D2HSocket::connect(d2h_socket_id, connect_timeout_ms);
        h2d_socket->set_page_size(PAGE_SIZE_BYTES);
        d2h_socket->set_page_size(PAGE_SIZE_BYTES);
    }
};

SocketPipeline::SocketPipeline(
    const std::string& h2d_socket_id, const std::string& d2h_socket_id, uint32_t connect_timeout_ms) :
    impl_(std::make_unique<Impl>(h2d_socket_id, d2h_socket_id, connect_timeout_ms)) {}

SocketPipeline::~SocketPipeline() = default;

void SocketPipeline::inject(const InjectDescriptor& desc) {
    impl_->write_buf = serialize_inject(desc);
    impl_->h2d_socket->write(impl_->write_buf.data(), 1);
}

ResultDescriptor SocketPipeline::read_result() {
    while (!impl_->stop_requested.load(std::memory_order_acquire)) {
        if (impl_->d2h_socket->has_data()) {
            impl_->read_buf.fill(0);
            impl_->d2h_socket->read(impl_->read_buf.data(), 1);
            return deserialize_result(impl_->read_buf);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    return ResultDescriptor{.user_id = -1, .sampled = false};
}

void SocketPipeline::reset_kv(int /*user_id*/) {}

void SocketPipeline::request_stop() { impl_->stop_requested.store(true, std::memory_order_release); }

void SocketPipeline::shutdown() {
    InjectDescriptor sentinel{};
    sentinel.user_id = -1;
    impl_->write_buf = serialize_inject(sentinel);
    impl_->h2d_socket->write(impl_->write_buf.data(), 1);

    // Drain all pending results until the kernel echoes the sentinel back.
    // In-flight tokens from cancelled users may still have results queued
    // in the D2H socket ahead of the sentinel echo.
    while (true) {
        impl_->read_buf.fill(0);
        impl_->d2h_socket->read(impl_->read_buf.data(), 1);
        auto result = deserialize_result(impl_->read_buf);
        if (result.user_id < 0) {
            break;
        }
    }

    impl_->h2d_socket->barrier();
    impl_->d2h_socket->barrier();
}

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
