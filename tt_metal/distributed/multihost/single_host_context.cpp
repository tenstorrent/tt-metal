// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "single_host_context.hpp"
#include <algorithm>
#include <cstring>

namespace tt::tt_metal::distributed::multihost {

// ---------------------------------------------------------------------
//                           Exception implementation
// ---------------------------------------------------------------------
SingleHostException::SingleHostException(Rank rank, int error_code, std::string msg) :
    rank_(rank), error_code_(error_code), message_(std::move(msg)), error_string_("Dummy MPI error") {}

Rank SingleHostException::rank() const noexcept { return rank_; }

int SingleHostException::error_code() const noexcept { return error_code_; }

const std::string& SingleHostException::message() const noexcept { return message_; }

const std::string& SingleHostException::error_string() const noexcept { return error_string_; }

// ---------------------------------------------------------------------
//                           Request implementation
// ---------------------------------------------------------------------
Status SingleHostRequest::wait() {
    done_ = true;
    return Status{Rank(0), Tag(0), 0};
}

std::optional<Status> SingleHostRequest::test() {
    done_ = true;
    return Status{Rank(0), Tag(0), 0};
}

void SingleHostRequest::cancel() { done_ = true; }

bool SingleHostRequest::active() const { return !done_; }

// ---------------------------------------------------------------------
//                           Context implementation
// ---------------------------------------------------------------------
void SingleHostContext::create(int argc, char** argv) {
    // No-op for single host implementation
    current_world_ = std::make_shared<SingleHostContext>();
}

const ContextPtr& SingleHostContext::get_current_world() {
    if (!current_world_) {
        current_world_ = std::make_shared<SingleHostContext>();
    }
    return current_world_;
}

Rank SingleHostContext::rank() const { return Rank(rank_); }

Size SingleHostContext::size() const { return Size(size_); }

bool SingleHostContext::supports_fault_tolerance() const { return false; }

void SingleHostContext::barrier() const {
    // No-op for single process
}

void SingleHostContext::send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    // No-op for single host implementation
}

void SingleHostContext::recv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const {
    // No-op for single host implementation
}

RequestPtr SingleHostContext::isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    return std::make_shared<SingleHostRequest>();
}

RequestPtr SingleHostContext::irecv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const {
    return std::make_shared<SingleHostRequest>();
}

void SingleHostContext::broadcast(tt::stl::Span<std::byte> buf, Rank root) const {
    // No-op for single process - data is already in place
}

void SingleHostContext::all_reduce(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    // For single process, just copy send_buf to recv_buf
    if (send_buf.data() != recv_buf.data()) {
        std::memcpy(recv_buf.data(), send_buf.data(), std::min(send_buf.size(), recv_buf.size()));
    }
}

void SingleHostContext::reduce(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype, Rank root) const {
    // For single process, just copy send_buf to recv_buf
    if (send_buf.data() != recv_buf.data()) {
        std::memcpy(recv_buf.data(), send_buf.data(), std::min(send_buf.size(), recv_buf.size()));
    }
}

void SingleHostContext::gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    // For single process, just copy send_buf to recv_buf
    if (send_buf.data() != recv_buf.data()) {
        std::memcpy(recv_buf.data(), send_buf.data(), std::min(send_buf.size(), recv_buf.size()));
    }
}

void SingleHostContext::scatter(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    // For single process, just copy send_buf to recv_buf
    if (send_buf.data() != recv_buf.data()) {
        std::memcpy(recv_buf.data(), send_buf.data(), std::min(send_buf.size(), recv_buf.size()));
    }
}

void SingleHostContext::all_gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
    // For single process, just copy send_buf to recv_buf
    if (send_buf.data() != recv_buf.data()) {
        std::memcpy(recv_buf.data(), send_buf.data(), std::min(send_buf.size(), recv_buf.size()));
    }
}

void SingleHostContext::all_to_all(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
    // For single process, just copy send_buf to recv_buf
    if (send_buf.data() != recv_buf.data()) {
        std::memcpy(recv_buf.data(), send_buf.data(), std::min(send_buf.size(), recv_buf.size()));
    }
}

void SingleHostContext::reduce_scatter(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    // For single process, just copy send_buf to recv_buf
    if (send_buf.data() != recv_buf.data()) {
        std::memcpy(recv_buf.data(), send_buf.data(), std::min(send_buf.size(), recv_buf.size()));
    }
}

void SingleHostContext::scan(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    // For single process, just copy send_buf to recv_buf
    if (send_buf.data() != recv_buf.data()) {
        std::memcpy(recv_buf.data(), send_buf.data(), std::min(send_buf.size(), recv_buf.size()));
    }
}

void SingleHostContext::translate_ranks_to_other_ctx(
    tt::stl::Span<int> ranks, const ContextPtr& other_ctx, tt::stl::Span<int> translated_ranks) const {
    // For single process, ranks are always 0
    for (size_t i = 0; i < std::min(ranks.size(), translated_ranks.size()); ++i) {
        translated_ranks[i] = 0;
    }
}

ContextPtr SingleHostContext::duplicate() const { return std::make_shared<SingleHostContext>(); }

ContextPtr SingleHostContext::split(Color color, Key key) const { return std::make_shared<SingleHostContext>(); }

ContextPtr SingleHostContext::create_sub_context(tt::stl::Span<int> ranks) const {
    return std::make_shared<SingleHostContext>();
}

void SingleHostContext::abort(int error_code) const { std::exit(error_code); }

void SingleHostContext::revoke_and_shrink() {
    // No-op for single host implementation
}

bool SingleHostContext::is_revoked() { return false; }

SingleHostContext::SingleHostContext() : rank_(0), size_(1) {}

void SingleHostContext::set_current_world(const ContextPtr& ctx) { current_world_ = ctx; }

}  // namespace tt::tt_metal::distributed::multihost
