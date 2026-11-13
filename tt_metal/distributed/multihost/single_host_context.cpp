// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "single_host_context.hpp"
#include "dtype_size.hpp"
#include <tt_stl/assert.hpp>
#include <algorithm>
#include <cstring>
#include <string_view>

namespace tt::tt_metal::distributed::multihost {

// Collectives follow MPI on a one-rank communicator, so code written for MPI also runs in a process that was not
// launched as part of an MPI job.
namespace {

void check_root(Rank root, std::string_view op) {
    TT_FATAL(*root == 0, "{}: root {} out of range for a single-host context", op, *root);
}

void copy_to_recv(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, std::string_view op) {
    TT_FATAL(
        recv_buf.size() == send_buf.size(),
        "{}: recv buffer {} bytes, expected {}",
        op,
        recv_buf.size(),
        send_buf.size());
    if (send_buf.data() != recv_buf.data()) {
        std::copy(send_buf.begin(), send_buf.end(), recv_buf.begin());
    }
}

}  // namespace

// ---------------------------------------------------------------------
//                           Context implementation
// ---------------------------------------------------------------------
SingleHostContext::SingleHostContext() : rank_(0), size_(1) { id_ = DistributedContext::generate_unique_id(); }

void SingleHostContext::create(int argc [[maybe_unused]], char** argv [[maybe_unused]]) {
    if (!current_world_) {
        current_world_ = std::make_shared<SingleHostContext>();
    }
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

void SingleHostContext::barrier() const {}

void SingleHostContext::send(
    ttsl::Span<std::byte> buf [[maybe_unused]], Rank dest [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method send is unsupported for single-host distributed contexts.");
}

void SingleHostContext::ssend(
    ttsl::Span<std::byte> buf [[maybe_unused]], Rank dest [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method ssend is unsupported for single-host distributed contexts.");
}

void SingleHostContext::recv(
    ttsl::Span<std::byte> buf [[maybe_unused]], Rank source [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method recv is unsupported for single-host distributed contexts.");
}

RequestPtr SingleHostContext::isend(
    ttsl::Span<std::byte> buf [[maybe_unused]], Rank dest [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method isend is unsupported for single-host distributed contexts.");
}

RequestPtr SingleHostContext::irecv(
    ttsl::Span<std::byte> buf [[maybe_unused]], Rank source [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method irecv is unsupported for single-host distributed contexts.");
}

void SingleHostContext::broadcast(ttsl::Span<std::byte> buf [[maybe_unused]], Rank root) const {
    check_root(root, "broadcast");
}

void SingleHostContext::all_reduce(
    ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op [[maybe_unused]], DType dtype) const {
    // MPI defines a reduction over a one-rank communicator as that rank's input for every operator, so op is
    // not consulted.
    TT_FATAL(
        recv_buf.size() == send_buf.size(),
        "all_reduce: recv buffer {} bytes, expected {}",
        recv_buf.size(),
        send_buf.size());
    const std::size_t element_size = dtype_size(dtype);
    TT_FATAL(
        send_buf.size() % element_size == 0,
        "all_reduce: buffer size {} is not a multiple of element size {}",
        send_buf.size(),
        element_size);

    if (send_buf.data() != recv_buf.data()) {
        std::copy(send_buf.begin(), send_buf.end(), recv_buf.begin());
    }
}

void SingleHostContext::reduce(
    ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype, Rank root) const {
    check_root(root, "reduce");
    all_reduce(send_buf, recv_buf, op, dtype);
}

void SingleHostContext::gather(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, Rank root) const {
    check_root(root, "gather");
    copy_to_recv(send_buf, recv_buf, "gather");
}

void SingleHostContext::scatter(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, Rank root) const {
    check_root(root, "scatter");
    copy_to_recv(send_buf, recv_buf, "scatter");
}

void SingleHostContext::all_gather(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf) const {
    copy_to_recv(send_buf, recv_buf, "all_gather");
}

void SingleHostContext::all_to_all(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf) const {
    copy_to_recv(send_buf, recv_buf, "all_to_all");
}

void SingleHostContext::reduce_scatter(
    ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    all_reduce(send_buf, recv_buf, op, dtype);
}

void SingleHostContext::scan(
    ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    all_reduce(send_buf, recv_buf, op, dtype);
}

void SingleHostContext::translate_ranks_to_other_ctx(
    ttsl::Span<int> ranks, const ContextPtr& other_ctx, ttsl::Span<int> translated_ranks) const {
    TT_FATAL(
        ranks.size() == translated_ranks.size(),
        "translate_ranks_to_other_ctx: ranks size {} != translated_ranks size {}",
        ranks.size(),
        translated_ranks.size());
    TT_FATAL(
        std::dynamic_pointer_cast<SingleHostContext>(other_ctx) != nullptr,
        "translate_ranks_to_other_ctx: other_ctx is not a SingleHostContext");
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        TT_FATAL(
            ranks[i] == 0, "translate_ranks_to_other_ctx: rank {} out of range for a single-host context", ranks[i]);
        translated_ranks[i] = 0;
    }
}

ContextPtr SingleHostContext::duplicate() const { return std::make_shared<SingleHostContext>(); }

ContextPtr SingleHostContext::split(Color color [[maybe_unused]], Key key [[maybe_unused]]) const {
    // nop on single-host context
    return get_current_world();
}

ContextPtr SingleHostContext::create_sub_context(ttsl::Span<int> ranks) const {
    TT_FATAL(
        ranks.size() == 1 && ranks[0] == 0,
        "create_sub_context: a single-host context only has rank 0, got {} ranks",
        ranks.size());
    return std::make_shared<SingleHostContext>();
}

void SingleHostContext::revoke_and_shrink() {
    TT_THROW("method revoke_and_shrink is unsupported for single-host distributed contexts.");
}

std::size_t SingleHostContext::snoop_incoming_msg_size(Rank source [[maybe_unused]], Tag tag [[maybe_unused]]) const {
    TT_THROW("method snoop_incoming_msg_size is unsupported for single-host distributed contexts.");
}

}  // namespace tt::tt_metal::distributed::multihost
