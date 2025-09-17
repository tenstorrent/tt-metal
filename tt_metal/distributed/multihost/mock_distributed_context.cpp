// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mock_distributed_context.hpp"
#include "assert.hpp"

namespace tt::tt_metal::distributed::multihost {
// ---------------------------------------------------------------------
//                           Context implementation
// ---------------------------------------------------------------------
MockDistributedContext::MockDistributedContext(const tt_ClusterDescriptor* cluster_desc) {}

// basic info
Rank MockDistributedContext::rank() const { TT_THROW("Not supported"); }
Size MockDistributedContext::size() const { TT_THROW("Not supported"); }
bool MockDistributedContext::supports_fault_tolerance() const { TT_THROW("Not supported"); }
bool MockDistributedContext::is_revoked() { TT_THROW("Not supported"); }

void MockDistributedContext::abort(int error_code) const { TT_THROW("Not supported"); }

/* Remaining methods throw for mock context */
void MockDistributedContext::barrier() const { TT_THROW("Not supported"); }

void MockDistributedContext::send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const { TT_THROW("Not supported"); }

void MockDistributedContext::ssend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    TT_THROW("Not supported");
}

void MockDistributedContext::recv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const {
    TT_THROW("Not supported");
}

RequestPtr MockDistributedContext::isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const {
    TT_THROW("Not supported");
}

RequestPtr MockDistributedContext::irecv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const {
    TT_THROW("Not supported");
}

void MockDistributedContext::broadcast(tt::stl::Span<std::byte> buf, Rank root) const { TT_THROW("Not supported"); }

void MockDistributedContext::all_reduce(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    TT_THROW("Not supported");
}

void MockDistributedContext::reduce(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype, Rank root) const {
    TT_THROW("Not supported");
}

void MockDistributedContext::gather(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    TT_THROW("Not supported");
}

void MockDistributedContext::scatter(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const {
    TT_THROW("Not supported");
}

void MockDistributedContext::all_gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
    TT_THROW("Not supported");
}

void MockDistributedContext::all_to_all(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const {
    TT_THROW("Not supported");
}

void MockDistributedContext::reduce_scatter(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    TT_THROW("Not supported");
}

void MockDistributedContext::scan(
    tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const {
    TT_THROW("Not supported");
}

void MockDistributedContext::translate_ranks_to_other_ctx(
    tt::stl::Span<int> ranks, const ContextPtr& other_ctx, tt::stl::Span<int> translated_ranks) const {
    TT_THROW("Not supported");
}

ContextPtr MockDistributedContext::duplicate() const { TT_THROW("Not supported"); }

ContextPtr MockDistributedContext::split(Color color, Key key) const { TT_THROW("Not supported"); }

ContextPtr MockDistributedContext::create_sub_context(tt::stl::Span<int> ranks) const { TT_THROW("Not supported"); }

void MockDistributedContext::revoke_and_shrink() { TT_THROW("Not supported"); }

std::size_t MockDistributedContext::snoop_incoming_msg_size(Rank source, Tag tag) const { TT_THROW("Not supported"); }

}  // namespace tt::tt_metal::distributed::multihost
