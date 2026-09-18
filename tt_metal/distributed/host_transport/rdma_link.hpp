// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct ibv_context;
struct ibv_pd;
struct ibv_cq;
struct ibv_qp;
struct ibv_mr;

namespace tt::tt_metal::distributed::host_transport {

// Descriptor for a peer-accessible memory region, exchanged out-of-band.
struct RemoteRegion {
    uint64_t addr = 0;
    uint64_t len = 0;
    uint32_t rkey = 0;
};

// Everything a peer needs to reach this side: QP identity plus the regions it
// may write into. Exchanged over DistributedContext during socket init.
struct RdmaEndpoint {
    uint8_t gid[16] = {};
    uint32_t qpn = 0;
    uint32_t psn = 0;
    RemoteRegion fifo;
    RemoteRegion doorbell;  // receiver advertises: where the sender publishes its total
    RemoteRegion credit;    // sender advertises: where the receiver publishes consumption
};

// RDMA device and protection domain. One per host process; shared by every
// channel. Completion queues are per-channel so that polling one channel never
// consumes another's completions.
class RdmaContext {
public:
    struct Config {
        std::string device_name;  // empty selects the first device
        int gid_index = -1;       // negative auto-selects a RoCEv2 IPv4-mapped GID
    };

    RdmaContext();
    explicit RdmaContext(const Config& config);
    ~RdmaContext();
    RdmaContext(const RdmaContext&) = delete;
    RdmaContext& operator=(const RdmaContext&) = delete;

    ibv_pd* pd() const { return pd_; }
    ibv_context* ctx() const { return ctx_; }
    uint8_t port() const { return port_; }
    int gid_index() const { return gid_index_; }
    const uint8_t* gid() const { return gid_; }

private:
    ibv_context* ctx_ = nullptr;
    ibv_pd* pd_ = nullptr;
    uint8_t port_ = 1;
    int gid_index_ = -1;
    uint8_t gid_[16] = {};
};

// A registered local buffer. Wraps ibv_reg_mr so socket FIFOs can be pinned
// once and written straight out of, with no staging copy.
class RdmaRegion {
public:
    RdmaRegion(RdmaContext& ctx, void* addr, size_t len);
    ~RdmaRegion();
    RdmaRegion(const RdmaRegion&) = delete;
    RdmaRegion& operator=(const RdmaRegion&) = delete;

    ibv_mr* mr() const { return mr_; }
    void* addr() const { return addr_; }
    size_t len() const { return len_; }
    uint32_t lkey() const;
    RemoteRegion descriptor() const;

private:
    ibv_mr* mr_ = nullptr;
    void* addr_ = nullptr;
    size_t len_ = 0;
};

// One RC queue pair: an ordered, pipelined, one-way byte stream.
//
// RC delivery is in-order, which is what preserves end-to-end write ordering:
// payload writes are posted back-to-back unsignaled, and the trailing doorbell
// write cannot be observed by the peer before the payload it follows has landed.
// That removes any need for a read fence on the fast path.
//
// Both signals are plain RDMA writes of an absolute counter into a small
// registered slot, which the peer reads with an ordinary load: the same
// "each side writes the other's memory and polls only its own" shape the rest of
// the socket protocol uses. Immediate data is deliberately not used -- whether a
// completion reports IBV_WC_RECV_RDMA_WITH_IMM and sets IBV_WC_WITH_IMM varies by
// provider, and a silently dropped or bogus immediate is very hard to diagnose.
class RdmaChannel {
public:
    struct Config {
        uint32_t sq_depth = 512;
        uint32_t sig_every = 64;  // one signaled WR per this many, to reap the SQ
    };

    explicit RdmaChannel(RdmaContext& ctx);
    RdmaChannel(RdmaContext& ctx, const Config& config);
    ~RdmaChannel();
    RdmaChannel(const RdmaChannel&) = delete;
    RdmaChannel& operator=(const RdmaChannel&) = delete;

    // Local identity; caller fills in the region descriptors before exchanging.
    RdmaEndpoint local_endpoint() const;
    void connect(const RdmaEndpoint& remote);
    bool connected() const { return connected_; }

    // Room left in the send queue. Callers must not exceed it.
    uint32_t send_slots_available() const;

    // Work-request sequence numbers. A batch is locally complete -- its source
    // bytes reusable -- once reaped() passes the id returned when it was posted.
    uint64_t last_posted_id() const { return posted_ - 1; }
    uint64_t reaped() const { return reaped_; }

    // Queue a payload write. Unsignaled unless the signal cadence falls due.
    // Returns false if the send queue is full, leaving nothing posted.
    bool post_write(const RdmaRegion& src, uint64_t src_offset, uint64_t dst_offset, uint32_t len);

    // Ordered doorbell: lands after every write already queued on this channel.
    // Carries an absolute count rather than a delta, so a duplicated or coalesced
    // doorbell is harmless -- the same reason the socket's own credits are absolute.
    bool post_doorbell(uint32_t total);

    // Credit update written inline to the peer's credit region; needs no
    // registered source buffer.
    bool post_credit(uint32_t value);

    // Reap send completions. Non-blocking; returns how many were reaped.
    uint32_t poll_send();

private:
    // inline_data sends the payload inside the work request, which is how the
    // 4-byte doorbell and credit writes avoid needing a registered source.
    bool post(
        uint64_t local_addr,
        uint32_t lkey,
        uint64_t remote_addr,
        uint32_t rkey,
        uint32_t len,
        bool inline_data,
        bool force_signal);

    RdmaContext& ctx_;
    Config config_;
    ibv_qp* qp_ = nullptr;
    ibv_cq* cq_ = nullptr;
    uint32_t psn_ = 0;
    uint64_t posted_ = 0;
    uint64_t reaped_ = 0;
    uint32_t since_signal_ = 0;
    uint32_t credit_scratch_ = 0;
    uint32_t doorbell_scratch_ = 0;
    RemoteRegion peer_fifo_;
    RemoteRegion peer_doorbell_;
    RemoteRegion peer_credit_;
    bool connected_ = false;
};

}  // namespace tt::tt_metal::distributed::host_transport
