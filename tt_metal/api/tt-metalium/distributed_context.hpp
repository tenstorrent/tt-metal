// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <tt_stl/strong_type.hpp>
#include <tt_stl/span.hpp>
#include <optional>
#include <cstddef>

namespace tt::tt_metal::distributed::multihost {

enum class ReduceOp { SUM, MAX, MIN, PROD };

using Rank = tt::stl::StrongType<int, struct RankTag>;
using Tag = tt::stl::StrongType<int, struct TagTag>;
using Color = tt::stl::StrongType<int, struct ColorTag>;
using Key = tt::stl::StrongType<int, struct KeyTag>;
using Size = tt::stl::StrongType<int, struct SizeTag>;

struct Status {
    Rank source = Rank(0);
    Tag tag = Tag(0);
    int count = 0;
};

class Request {
public:
    virtual ~Request() = default;

    /// Block until the operation completes, then return its Status.
    [[nodiscard]] virtual Status wait() = 0;

    /// Poll for completion. If done, return Status; otherwise return std::nullopt.
    [[nodiscard]] virtual std::optional<Status> test() = 0;

    /// Cancel the pending operation (if supported).
    virtual void cancel() = 0;

    /// Is the request still active (i.e. not yet completed or cancelled)?
    [[nodiscard]] virtual bool active() const = 0;
};

using RequestPtr = std::shared_ptr<Request>;

class DistributedContext {
public:
    static std::shared_ptr<DistributedContext> create(int argc, char** argv);
    //--- Topology ------------------------------------------------------------
    [[nodiscard]] virtual Rank rank() const = 0;
    [[nodiscard]] virtual Size size() const = 0;

    //--- Synchronization ----------------------------------------------------
    virtual void barrier() const = 0;

    //--- Point-to-point (blocking) -----------------------------------------
    virtual void send(tt::stl::Span<std::byte> buffer, Rank dest, Tag tag) const = 0;

    virtual void recv(tt::stl::Span<std::byte> buffer, Rank source, Tag tag) const = 0;

    //--- Point-to-point (non-blocking) -------------------------------------
    [[nodiscard]] virtual RequestPtr isend(tt::stl::Span<std::byte> buffer, Rank dest, Tag tag) const = 0;

    [[nodiscard]] virtual RequestPtr irecv(tt::stl::Span<std::byte> buffer, Rank source, Tag tag) const = 0;

    //--- Collective operations ---------------------------------------------
    virtual void broadcast(tt::stl::Span<std::byte> buffer, Rank root) const = 0;

    virtual void all_reduce(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const = 0;

    virtual void reduce(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, Rank root) const = 0;

    virtual void gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const = 0;

    virtual void scatter(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const = 0;

    virtual void all_gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const = 0;

    virtual void all_to_all(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const = 0;

    virtual void reduce_scatter(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const = 0;

    virtual void scan(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const = 0;

    //--- Communicator management -------------------------------------------
    [[nodiscard]] virtual std::shared_ptr<DistributedContext> duplicate() const = 0;
    [[nodiscard]] virtual std::shared_ptr<DistributedContext> split(Color color, Key key) const = 0;
    [[nodiscard]] virtual std::shared_ptr<DistributedContext> create_sub_context(tt::stl::Span<Rank> ranks) const = 0;

    //--- Error handling -----------------------------------------------------
    virtual ~DistributedContext() = default;
};
}  // namespace tt::tt_metal::distributed::multihost
