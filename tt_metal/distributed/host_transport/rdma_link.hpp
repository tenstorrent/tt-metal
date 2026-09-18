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

struct RemoteRegion {
    uint64_t addr = 0;
    uint64_t len = 0;
    uint32_t rkey = 0;
};

struct RdmaEndpoint {
    uint8_t gid[16] = {};
    uint32_t qpn = 0;
    uint32_t psn = 0;
    RemoteRegion fifo;
    RemoteRegion doorbell;  // receiver advertises: where the sender publishes its total
    RemoteRegion credit;    // sender advertises: where the receiver publishes consumption
};

// Completion queues are per-channel: polling one channel must not consume
// another's completions.
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

// One RC queue pair. RC is in-order, so a trailing doorbell cannot be seen
// before the payload it follows: no read fence needed.
//
// Doorbell and credit are absolute counters written to a registered slot, not
// RDMA immediates. These NICs report the completion without IBV_WC_WITH_IMM, so
// imm_data is unusable.
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

    // Caller fills in the region descriptors before exchanging.
    RdmaEndpoint local_endpoint() const;
    void connect(const RdmaEndpoint& remote);
    bool connected() const { return connected_; }

    uint32_t send_slots_available() const;

    // Source bytes are reusable once reaped() passes the id post returned.
    uint64_t last_posted_id() const { return posted_ - 1; }
    uint64_t reaped() const { return reaped_; }

    // False if the send queue is full; nothing is posted.
    bool post_write(const RdmaRegion& src, uint64_t src_offset, uint64_t dst_offset, uint32_t len);

    // Absolute count, not a delta: a duplicated doorbell must be a no-op.
    bool post_doorbell(uint32_t total);

    bool post_credit(uint32_t value);

    uint32_t poll_send();

private:
    // inline_data avoids needing a registered source for the 4-byte signals.
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
