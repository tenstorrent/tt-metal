// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <zmq.hpp>

#include "api/tt-metalium/distributed_context.hpp"

namespace tt::tt_metal::distributed::multihost {

// ---------------------------------------------------------------------
//                              Exception
// ---------------------------------------------------------------------
class ZmqDistributedException : public DistributedException {
public:
    ZmqDistributedException(Rank rank, int error_code, std::string msg);

    Rank rank() const noexcept override;
    int error_code() const noexcept override;
    const std::string& message() const noexcept override;
    const std::string& error_string() const noexcept override;

private:
    Rank rank_{0};
    int error_code_{0};
    std::string message_;
    std::string error_string_;
};

// ---------------------------------------------------------------------
//   ZmqTransport: process-global transport shared by every ZmqContext
// ---------------------------------------------------------------------
// One instance per process. Owns the ZMQ context, a single ROUTER socket for
// receiving from every peer, one DEALER socket per peer for sending, a
// background progress thread that drains the ROUTER into a tag-matched inbox,
// and the monotonic context-id allocator used to give sub-contexts isolated
// message namespaces.
//
// Thread-safety model (mirrors "one socket, one owner" ZMQ guidance):
//   * ROUTER is touched only by the progress thread.
//   * Each DEALER is guarded by its own mutex; caller threads may send in
//     parallel to different peers.
//   * The inbox is guarded by inbox_mtx_ / inbox_cv_.
class ZmqTransport {
public:
    // Matching key: messages are demultiplexed by (context, source, tag). The
    // context id isolates sub-contexts; negative tags are reserved for internal
    // collective/barrier traffic so they never collide with user tags (>= 0).
    struct Key {
        std::uint32_t context_id = 0;
        int src_world_rank = 0;
        int tag = 0;
        bool operator<(const Key& o) const {
            return std::tie(context_id, src_world_rank, tag) < std::tie(o.context_id, o.src_world_rank, o.tag);
        }
    };

    // `endpoints[r]` is the ZMQ endpoint (e.g. "tcp://host:5555") that rank r
    // binds its ROUTER to; every other rank connects a DEALER to it.
    ZmqTransport(int world_rank, std::vector<std::string> endpoints);
    ~ZmqTransport();

    ZmqTransport(const ZmqTransport&) = delete;
    ZmqTransport& operator=(const ZmqTransport&) = delete;

    [[nodiscard]] int world_rank() const { return world_rank_; }
    [[nodiscard]] int world_size() const { return static_cast<int>(endpoints_.size()); }

    // Buffered, ordered send (loopback short-circuits to the inbox for self).
    void post(int dst_world_rank, std::uint32_t context_id, int tag, ttsl::Span<std::byte> bytes);

    // Blocking receive: waits for a message matching (context_id, src, tag).
    Status recv_into(std::uint32_t context_id, int src_world_rank, int tag, ttsl::Span<std::byte> out);

    // Non-blocking receive: std::nullopt if no matching message is queued yet.
    std::optional<Status> try_recv_into(
        std::uint32_t context_id, int src_world_rank, int tag, ttsl::Span<std::byte> out);

    // Blocking probe: waits for a matching message, returns its size WITHOUT consuming it.
    std::size_t probe(std::uint32_t context_id, int src_world_rank, int tag);

    // Reserve `n` fresh, globally-consistent context ids. Because sub-context
    // creation is collective and executed in the same order on every rank, the
    // per-process counter advances identically everywhere and the returned base
    // agrees across ranks.
    std::uint32_t reserve_context_ids(int n);

private:
    void progress_loop();
    bool find_locked(const Key& key) const;

    int world_rank_ = 0;
    std::vector<std::string> endpoints_;

    zmq::context_t zmq_ctx_;
    zmq::socket_t router_;
    std::vector<zmq::socket_t> dealers_;                   // indexed by world rank; self slot unused
    std::vector<std::unique_ptr<std::mutex>> dealer_mtx_;  // one per peer

    std::thread progress_;
    std::atomic<bool> running_{false};

    mutable std::mutex inbox_mtx_;
    std::condition_variable inbox_cv_;
    std::map<Key, std::deque<std::vector<std::byte>>> inbox_;

    std::uint32_t next_context_id_ = 1;  // 0 is reserved for the world context
};

// ---------------------------------------------------------------------
//                         Non-blocking request
// ---------------------------------------------------------------------
// isend completes immediately (ZMQ buffers the payload on send), so an isend
// request is born completed. irecv defers the match to wait()/test().
class ZmqRequest : public Request {
public:
    // Completed request (used by isend).
    explicit ZmqRequest(Status done_status);

    // Pending receive request (used by irecv).
    ZmqRequest(
        std::shared_ptr<ZmqTransport> transport,
        std::uint32_t context_id,
        int src_world_rank,
        int tag,
        ttsl::Span<std::byte> out);

    Status wait() override;
    std::optional<Status> test() override;
    void cancel() override;
    bool active() const override;

private:
    std::shared_ptr<ZmqTransport> transport_;  // null for a completed request
    std::uint32_t context_id_ = 0;
    int src_world_rank_ = 0;
    int tag_ = 0;
    ttsl::Span<std::byte> out_{};
    std::optional<Status> status_;
    bool cancelled_ = false;
};

// ---------------------------------------------------------------------
//                       Main distributed context
// ---------------------------------------------------------------------
class ZmqContext : public DistributedContext {
public:
    // Rank returned by translate_ranks_to_other_ctx for ranks with no counterpart.
    static constexpr int RANK_UNDEFINED = -1;

    ZmqContext(
        std::shared_ptr<ZmqTransport> transport,
        std::uint32_t context_id,
        std::vector<int> world_ranks,
        int local_rank);

    // factory / singletons
    static void create(int argc, char** argv);
    static const ContextPtr& get_current_world();
    static ContextPtr get_world_context();
    static void set_current_world(const ContextPtr& ctx);
    static bool is_initialized();

    ~ZmqContext() override = default;

    /* ---------------- basic info / sync ---------------- */
    [[nodiscard]] Rank rank() const override;
    [[nodiscard]] Size size() const override;
    [[nodiscard]] bool supports_fault_tolerance() const override;
    [[nodiscard]] bool is_revoked() override;
    void barrier() const override;

    /* ---------------- point-to-point ------------------- */
    void send(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const override;
    void ssend(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const override;
    void recv(ttsl::Span<std::byte> buf, Rank source, Tag tag) const override;
    [[nodiscard]] RequestPtr isend(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const override;
    [[nodiscard]] RequestPtr irecv(ttsl::Span<std::byte> buf, Rank source, Tag tag) const override;

    /* ---------------- collectives ---------------------- */
    void broadcast(ttsl::Span<std::byte> buf, Rank root) const override;
    void all_gather(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf) const override;

    // COLD (test-only in production): currently unimplemented. See .cpp for rationale.
    void all_reduce(
        ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const override;
    void reduce(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype, Rank root)
        const override;
    void gather(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, Rank root) const override;
    void scatter(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, Rank root) const override;
    void all_to_all(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf) const override;
    void reduce_scatter(
        ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const override;
    void scan(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf, ReduceOp op, DType dtype) const override;

    void translate_ranks_to_other_ctx(
        ttsl::Span<int> ranks, const ContextPtr& other_ctx, ttsl::Span<int> translated_ranks) const override;

    /* ------------- communicator management ------------- */
    [[nodiscard]] ContextPtr duplicate() const override;
    [[nodiscard]] ContextPtr split(Color color, Key key) const override;
    [[nodiscard]] ContextPtr create_sub_context(ttsl::Span<int> ranks) const override;

    /* ------------- error handling / fault tolerance ---- */
    void abort(int error_code) const override;
    void revoke_and_shrink() override;

    /* ------------- message snooping -------------------- */
    std::size_t snoop_incoming_msg_size(Rank source, Tag tag) const override;

private:
    // Returns a fresh negative tag for one collective invocation. All ranks call
    // collectives in the same order (SPMD), so this advances in lockstep and the
    // returned tag agrees across ranks.
    [[nodiscard]] int next_coll_tag() const;
    [[nodiscard]] int to_world(Rank local) const;

    std::shared_ptr<ZmqTransport> transport_;
    std::uint32_t context_id_ = 0;
    std::vector<int> world_ranks_;  // local rank -> world rank
    int local_rank_ = 0;
    mutable int coll_seq_ = 0;

    inline static ContextPtr current_world_;
};

}  // namespace tt::tt_metal::distributed::multihost
