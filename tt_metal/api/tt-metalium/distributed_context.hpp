// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt_stl/strong_type.hpp>
#include <tt_stl/span.hpp>
#include <optional>
#include <cstddef>

namespace tt::tt_metal::distributed::multihost {

enum class ReduceOp { SUM, MAX, MIN, PROD };

struct Rank : public tt::stl::StrongType<int, Rank> {};
struct Tag : public tt::stl::StrongType<int, Tag> {};
struct Color : public tt::stl::StrongType<int, Color> {};
struct Key : public tt::stl::StrongType<int, Key> {};
struct Size : public tt::stl::StrongType<int, Size> {};

struct Status {
    Rank source{};
    Tag tag{};
    int count{};
};

class IRequest {
public:
    virtual ~IRequest() = default;

    /// Block until the operation completes, then return its Status.
    virtual Status wait() = 0;

    /// Poll for completion. If done, return Status; otherwise return std::nullopt.
    virtual std::optional<Status> test() = 0;

    /// Cancel the pending operation (if supported).
    virtual void cancel() = 0;

    /// Is the request still active (i.e. not yet completed or cancelled)?
    virtual bool active() const = 0;
};

using RequestPtr = std::shared_ptr<IRequest>;

class IDistributedContext {
public:
    static std::shared_ptr<IDistributedContext> create(int argc, char** argv);
    //--- Topology ------------------------------------------------------------
    virtual Rank rank() const = 0;
    virtual Size size() const = 0;

    //--- Synchronization ----------------------------------------------------
    virtual void barrier() const = 0;

    //--- Point-to-point (blocking) -----------------------------------------
    virtual void send(tt::stl::Span<std::byte> buffer, Rank dest, Tag tag) const = 0;

    virtual void recv(tt::stl::Span<std::byte> buffer, Rank source, Tag tag, Status* status = nullptr) const = 0;

    //--- Point-to-point (non-blocking) -------------------------------------
    virtual IRequest isend(tt::stl::Span<std::byte> buffer, Rank dest, Tag tag) const = 0;

    virtual IRequest irecv(tt::stl::Span<std::byte> buffer, Rank source, Tag tag) const = 0;

    virtual Status wait(IRequest& req) const = 0;
    virtual std::vector<Status> wait_all(tt::stl::Span<RequestPtr> reqs) const = 0;
    virtual Status wait_any(tt::stl::Span<RequestPtr> reqs, int& index) const = 0;

    //--- Collective operations ---------------------------------------------
    virtual void broadcast(tt::stl::Span<std::byte> buffer, Rank root) const = 0;

    virtual void all_reduce(
        tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const = 0;

    virtual void reduce(
        tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, Rank root) const = 0;

    virtual void gather(
        tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const = 0;

    virtual void scatter(
        tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const = 0;

    virtual void all_gather(tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const = 0;

    virtual void all_to_all(tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const = 0;

    virtual void reduce_scatter(
        tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const = 0;

    virtual void scan(
        tt::stl::Span<const std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op) const = 0;

    //--- Communicator management -------------------------------------------
    virtual std::shared_ptr<IDistributedContext> duplicate() const = 0;
    virtual std::shared_ptr<IDistributedContext> split(Color color, Key key) const = 0;
    virtual std::shared_ptr<IDistributedContext> create_sub_context(tt::stl::Span<Rank> ranks) const = 0;

    //--- Error handling -----------------------------------------------------
    virtual ~IDistributedContext() = 0;
};
}  // namespace tt::tt_metal::distributed::multihost
