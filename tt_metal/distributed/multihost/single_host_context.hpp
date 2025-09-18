// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "api/tt-metalium/distributed_context.hpp"

namespace tt::tt_metal::distributed::multihost {
// ---------------------------------------------------------------------
//                       Main distributed context
// ---------------------------------------------------------------------
class SingleHostContext : public DistributedContext {
public:
    /* ----------------- single host constructor ---------------- */
    explicit SingleHostContext();

    // factory
    static void create(int argc, char** argv);
    static const ContextPtr& get_current_world();
    static void set_current_world(const ContextPtr&);
    static bool is_initialized();

    // destructor – no-op
    ~SingleHostContext() override = default;

    /* ---------------- basic info ---------------- */
    [[nodiscard]] Rank rank() const override;
    [[nodiscard]] Size size() const override;
    [[nodiscard]] bool supports_fault_tolerance() const override;
    [[nodiscard]] bool is_revoked() override;

    void abort(int error_code) const override;

    /* --------------- the rest of the methods are unsupported and throw --------- */
    void barrier() const override;

    /* ---------------- point‑to‑point ------------------- */
    void send(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const override;
    void ssend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const override;
    void recv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const override;

    [[nodiscard]] RequestPtr isend(tt::stl::Span<std::byte> buf, Rank dest, Tag tag) const override;
    [[nodiscard]] RequestPtr irecv(tt::stl::Span<std::byte> buf, Rank source, Tag tag) const override;

    /* ---------------- collectives ---------------------- */
    void broadcast(tt::stl::Span<std::byte> buf, Rank root) const override;
    void all_reduce(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const override;
    void reduce(
        tt::stl::Span<std::byte> send_buf,
        tt::stl::Span<std::byte> recv_buf,
        ReduceOp op,
        DType dtype,
        Rank root) const override;
    void gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const override;
    void scatter(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, Rank root) const override;
    void all_gather(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const override;
    void all_to_all(tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf) const override;
    void reduce_scatter(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const override;
    void scan(
        tt::stl::Span<std::byte> send_buf, tt::stl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const override;

    void translate_ranks_to_other_ctx(
        tt::stl::Span<int> ranks, const ContextPtr& other_ctx, tt::stl::Span<int> translated_ranks) const override;

    /* ------------- communicator management ------------- */
    [[nodiscard]] ContextPtr duplicate() const override;
    [[nodiscard]] ContextPtr split(Color color, Key key) const override;
    [[nodiscard]] ContextPtr create_sub_context(tt::stl::Span<int> ranks) const override;
    void revoke_and_shrink() override;

    /* ------------- message snooping ------------- */
    std::size_t snoop_incoming_msg_size(Rank source, Tag tag) const override;

private:
    int rank_{0};
    int size_{1};

    // caching our own world communicator
    inline static ContextPtr current_world_;
};

}  // namespace tt::tt_metal::distributed::multihost
