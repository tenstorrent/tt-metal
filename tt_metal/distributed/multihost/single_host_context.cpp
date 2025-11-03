// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "single_host_context.hpp"
#include <tt_stl/assert.hpp>
#include <algorithm>
#include <cstring>

namespace tt::tt_metal::distributed::multihost {
// ---------------------------------------------------------------------
//                           Context implementation
// ---------------------------------------------------------------------
SingleHostContext::SingleHostContext() : rank_(0), size_(1) { id_ = DistributedContext::generate_unique_id(); }

void SingleHostContext::create(int argc [[maybe_unused]], char** argv [[maybe_unused]]) {
    current_world_ = std::make_shared<SingleHostContext>();
}

const ContextPtr& SingleHostContext::get_current_world() {
    if (!current_world_) {
        current_world_ = std::make_shared<SingleHostContext>();
    }
    return current_world_;
}

void SingleHostContext::set_current_world(const ContextPtr& ctx) {
    TT_FATAL(
        ctx != nullptr && std::dynamic_pointer_cast<SingleHostContext>(ctx) != nullptr,
        "SingleHostContext::set_current_world: context is not a SingleHostContext or a nullptr");
    SingleHostContext::current_world_ = ctx;
}

bool SingleHostContext::is_initialized() { return current_world_ != nullptr; }

// basic info
Rank SingleHostContext::rank() const { return Rank(rank_); }
Size SingleHostContext::size() const { return Size(size_); }
bool SingleHostContext::supports_fault_tolerance() const { return false; }
bool SingleHostContext::is_revoked() { return false; }

void SingleHostContext::abort(int error_code) const { std::exit(error_code); }

/* Remaining methods throw for single-host context */
void SingleHostContext::barrier() const {
    TT_THROW("method barrier is unsupported for single-host distributed contexts.");
}

void SingleHostContext::send(
    tt::stl::Span<std::byte> buf [[maybe_unused]], Rank dest [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method send is unsupported for single-host distributed contexts.");
}

void SingleHostContext::ssend(
    tt::stl::Span<std::byte> buf [[maybe_unused]], Rank dest [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method ssend is unsupported for single-host distributed contexts.");
}

void SingleHostContext::recv(
    tt::stl::Span<std::byte> buf [[maybe_unused]], Rank source [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method recv is unsupported for single-host distributed contexts.");
}

RequestPtr SingleHostContext::isend(
    tt::stl::Span<std::byte> buf [[maybe_unused]], Rank dest [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method isend is unsupported for single-host distributed contexts.");
}

RequestPtr SingleHostContext::irecv(
    tt::stl::Span<std::byte> buf [[maybe_unused]], Rank source [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method irecv is unsupported for single-host distributed contexts.");
}

void SingleHostContext::broadcast(tt::stl::Span<std::byte> buf [[maybe_unused]], Rank root [[maybe_unused]]) const {
    TT_THROW("method broadcast is unsupported for single-host distributed contexts.");
}

void SingleHostContext::all_reduce(
    tt::stl::Span<std::byte> send_buf [[maybe_unused]],
    tt::stl::Span<std::byte> recv_buf [[maybe_unused]],
    ReduceOp op [[maybe_unused]],
    DType dtype [[maybe_unused]]) const {
    TT_THROW("method all_reduce is unsupported for single-host distributed contexts.");
}

void SingleHostContext::reduce(
    tt::stl::Span<std::byte> send_buf [[maybe_unused]],
    tt::stl::Span<std::byte> recv_buf [[maybe_unused]],
    ReduceOp op [[maybe_unused]],
    DType dtype [[maybe_unused]],
    Rank root [[maybe_unused]]) const {
    TT_THROW("method reduce is unsupported for single-host distributed contexts.");
}

void SingleHostContext::gather(
    tt::stl::Span<std::byte> send_buf [[maybe_unused]],
    tt::stl::Span<std::byte> recv_buf [[maybe_unused]],
    Rank root [[maybe_unused]]) const {
    TT_THROW("method gather is unsupported for single-host distributed contexts.");
}

void SingleHostContext::scatter(
    tt::stl::Span<std::byte> send_buf [[maybe_unused]],
    tt::stl::Span<std::byte> recv_buf [[maybe_unused]],
    Rank root [[maybe_unused]]) const {
    TT_THROW("method scatter is unsupported for single-host distributed contexts.");
}

void SingleHostContext::all_gather(
    tt::stl::Span<std::byte> send_buf [[maybe_unused]], tt::stl::Span<std::byte> recv_buf [[maybe_unused]]) const {
    TT_THROW("method all_gather is unsupported for single-host distributed contexts.");
}

void SingleHostContext::all_to_all(
    tt::stl::Span<std::byte> send_buf [[maybe_unused]], tt::stl::Span<std::byte> recv_buf [[maybe_unused]]) const {
    TT_THROW("method all_to_all is unsupported for single-host distributed contexts.");
}

void SingleHostContext::reduce_scatter(
    tt::stl::Span<std::byte> send_buf [[maybe_unused]],
    tt::stl::Span<std::byte> recv_buf [[maybe_unused]],
    ReduceOp op [[maybe_unused]],
    DType dtype [[maybe_unused]]) const {
    TT_THROW("method reduce_scatter is unsupported for single-host distributed contexts.");
}

void SingleHostContext::scan(
    tt::stl::Span<std::byte> send_buf [[maybe_unused]],
    tt::stl::Span<std::byte> recv_buf [[maybe_unused]],
    ReduceOp op [[maybe_unused]],
    DType dtype [[maybe_unused]]) const {
    TT_THROW("method scan is unsupported for single-host distributed contexts.");
}

void SingleHostContext::translate_ranks_to_other_ctx(
    tt::stl::Span<int> ranks [[maybe_unused]],
    const ContextPtr& other_ctx [[maybe_unused]],
    tt::stl::Span<int> translated_ranks [[maybe_unused]]) const {
    TT_THROW("method translate_ranks_to_other_ctx is unsupported for single-host distributed contexts.");
}

ContextPtr SingleHostContext::duplicate() const {
    TT_THROW("method duplicate is unsupported for single-host distributed contexts.");
}

ContextPtr SingleHostContext::split(Color color [[maybe_unused]], Key key [[maybe_unused]]) const {
    TT_THROW("method split is unsupported for single-host distributed contexts.");
}

ContextPtr SingleHostContext::create_sub_context(tt::stl::Span<int> ranks [[maybe_unused]]) const {
    TT_THROW("method create_sub_context is unsupported for single-host distributed contexts.");
}

void SingleHostContext::revoke_and_shrink() {
    TT_THROW("method revoke_and_shrink is unsupported for single-host distributed contexts.");
}

std::size_t SingleHostContext::snoop_incoming_msg_size(Rank source [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method snoop_incoming_msg_size is unsupported for single-host distributed contexts.");
}

}  // namespace tt::tt_metal::distributed::multihost
