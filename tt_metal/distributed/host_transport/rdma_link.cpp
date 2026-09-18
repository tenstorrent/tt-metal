// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_transport/rdma_link.hpp"

#include <infiniband/verbs.h>

#include <cstdio>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::distributed::host_transport {

namespace {

constexpr uint8_t kIpv4MappedPrefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};

// ibv_query_gid does not report the RoCE version, and a port usually exposes
// both v1 and v2 GIDs for one IPv4 address. Picking v1 fails to connect.
bool gid_is_rocev2(const char* device_name, uint8_t port, int index) {
    char path[256];
    std::snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%u/gid_attrs/types/%d", device_name, port, index);
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        return false;
    }
    char buf[64];
    ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);
    if (n <= 0) {
        return false;
    }
    buf[n] = '\0';
    return std::strstr(buf, "RoCE v2") != nullptr;
}

int pick_rocev2_gid(ibv_context* ctx, const char* device_name, uint8_t port) {
    for (int i = 0; i < 16; i++) {
        ibv_gid gid;
        if (ibv_query_gid(ctx, port, i, &gid) != 0) {
            break;
        }
        if (std::memcmp(gid.raw, kIpv4MappedPrefix, sizeof(kIpv4MappedPrefix)) != 0) {
            continue;
        }
        if (gid_is_rocev2(device_name, port, i)) {
            return i;
        }
    }
    return -1;
}

ibv_context* open_device(const std::string& name) {
    int num_devices = 0;
    ibv_device** list = ibv_get_device_list(&num_devices);
    TT_FATAL(list != nullptr && num_devices > 0, "No RDMA devices found");
    ibv_context* ctx = nullptr;
    std::string opened;
    for (int i = 0; i < num_devices; i++) {
        if (name.empty() || name == ibv_get_device_name(list[i])) {
            opened = ibv_get_device_name(list[i]);
            ctx = ibv_open_device(list[i]);
            break;
        }
    }
    ibv_free_device_list(list);
    TT_FATAL(ctx != nullptr, "RDMA device '{}' not found or could not be opened", name.empty() ? "(first)" : name);
    log_debug(tt::LogDistributed, "host_transport: opened RDMA device {}", opened);
    return ctx;
}

}  // namespace

RdmaContext::RdmaContext() : RdmaContext(Config{}) {}

RdmaContext::RdmaContext(const Config& config) {
    ctx_ = open_device(config.device_name);
    pd_ = ibv_alloc_pd(ctx_);
    TT_FATAL(pd_ != nullptr, "ibv_alloc_pd failed");

    const char* name = ibv_get_device_name(ctx_->device);
    gid_index_ = config.gid_index >= 0 ? config.gid_index : pick_rocev2_gid(ctx_, name, port_);
    TT_FATAL(
        gid_index_ >= 0,
        "No RoCEv2 (IPv4-mapped) GID on {}. The port most likely has no IP address configured -- check "
        "'ip addr' and /sys/class/infiniband/{}/ports/1/gid_attrs/types/*, pick a port that has one, or set "
        "gid_index explicitly.",
        name,
        name);

    ibv_gid gid;
    TT_FATAL(ibv_query_gid(ctx_, port_, gid_index_, &gid) == 0, "ibv_query_gid failed");
    std::memcpy(gid_, gid.raw, sizeof(gid_));
}

RdmaContext::~RdmaContext() {
    if (pd_ != nullptr) {
        ibv_dealloc_pd(pd_);
    }
    if (ctx_ != nullptr) {
        ibv_close_device(ctx_);
    }
}

RdmaRegion::RdmaRegion(RdmaContext& ctx, void* addr, size_t len) : addr_(addr), len_(len) {
    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    mr_ = ibv_reg_mr(ctx.pd(), addr, len, access);
    TT_FATAL(mr_ != nullptr, "ibv_reg_mr failed for {} bytes at {}", len, addr);
}

RdmaRegion::~RdmaRegion() {
    if (mr_ != nullptr) {
        ibv_dereg_mr(mr_);
    }
}

uint32_t RdmaRegion::lkey() const { return mr_->lkey; }

RemoteRegion RdmaRegion::descriptor() const {
    return RemoteRegion{
        .addr = reinterpret_cast<uint64_t>(addr_),
        .len = len_,
        .rkey = mr_->rkey,
    };
}

RdmaChannel::RdmaChannel(RdmaContext& ctx) : RdmaChannel(ctx, Config{}) {}

RdmaChannel::RdmaChannel(RdmaContext& ctx, const Config& config) : ctx_(ctx), config_(config) {
    cq_ = ibv_create_cq(ctx_.ctx(), static_cast<int>(config_.sq_depth + 16), nullptr, nullptr, 0);
    TT_FATAL(cq_ != nullptr, "ibv_create_cq failed");

    ibv_qp_init_attr attr{};
    attr.qp_type = IBV_QPT_RC;
    attr.send_cq = cq_;
    attr.recv_cq = cq_;
    attr.cap.max_send_wr = config_.sq_depth;
    attr.cap.max_recv_wr = 1;  // unused: both signals are RDMA writes, not sends
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.cap.max_inline_data = 64;
    qp_ = ibv_create_qp(ctx_.pd(), &attr);
    TT_FATAL(qp_ != nullptr, "ibv_create_qp failed");

    psn_ = static_cast<uint32_t>(::lrand48()) & 0xffffff;
}

RdmaChannel::~RdmaChannel() {
    if (qp_ != nullptr) {
        ibv_destroy_qp(qp_);
    }
    if (cq_ != nullptr) {
        ibv_destroy_cq(cq_);
    }
}

RdmaEndpoint RdmaChannel::local_endpoint() const {
    RdmaEndpoint ep;
    std::memcpy(ep.gid, ctx_.gid(), sizeof(ep.gid));
    ep.qpn = qp_->qp_num;
    ep.psn = psn_;
    return ep;
}

void RdmaChannel::connect(const RdmaEndpoint& remote) {
    ibv_port_attr port_attr{};
    ibv_mtu mtu = IBV_MTU_4096;
    if (ibv_query_port(ctx_.ctx(), ctx_.port(), &port_attr) == 0 && port_attr.active_mtu < mtu) {
        mtu = port_attr.active_mtu;
    }

    ibv_qp_attr init{};
    init.qp_state = IBV_QPS_INIT;
    init.pkey_index = 0;
    init.port_num = ctx_.port();
    init.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    TT_FATAL(
        ibv_modify_qp(qp_, &init, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
        "QP transition to INIT failed");

    ibv_qp_attr rtr{};
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = mtu;
    rtr.dest_qp_num = remote.qpn;
    rtr.rq_psn = remote.psn;
    rtr.max_dest_rd_atomic = 16;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1;
    rtr.ah_attr.port_num = ctx_.port();
    rtr.ah_attr.grh.sgid_index = ctx_.gid_index();
    rtr.ah_attr.grh.hop_limit = 64;
    std::memcpy(rtr.ah_attr.grh.dgid.raw, remote.gid, sizeof(remote.gid));
    TT_FATAL(
        ibv_modify_qp(
            qp_,
            &rtr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER) == 0,
        "QP transition to RTR failed (check GID selection and routing)");

    ibv_qp_attr rts{};
    rts.qp_state = IBV_QPS_RTS;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;
    rts.sq_psn = psn_;
    rts.max_rd_atomic = 16;
    TT_FATAL(
        ibv_modify_qp(
            qp_,
            &rts,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                IBV_QP_MAX_QP_RD_ATOMIC) == 0,
        "QP transition to RTS failed");

    peer_fifo_ = remote.fifo;
    peer_doorbell_ = remote.doorbell;
    peer_credit_ = remote.credit;
    connected_ = true;
}

uint32_t RdmaChannel::send_slots_available() const {
    uint64_t in_flight = posted_ - reaped_;
    return in_flight >= config_.sq_depth ? 0 : config_.sq_depth - static_cast<uint32_t>(in_flight);
}

bool RdmaChannel::post(
    uint64_t local_addr,
    uint32_t lkey,
    uint64_t remote_addr,
    uint32_t rkey,
    uint32_t len,
    bool inline_data,
    bool force_signal) {
    if (send_slots_available() == 0) {
        return false;
    }

    ibv_sge sge{};
    sge.addr = local_addr;
    sge.length = len;
    sge.lkey = lkey;

    ibv_send_wr wr{};
    wr.wr_id = posted_;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    if (inline_data) {
        wr.send_flags |= IBV_SEND_INLINE;
    }
    if (len != 0) {
        wr.sg_list = &sge;
        wr.num_sge = 1;
    }

    // A doorbell must always signal: batches retire on its completion, so an
    // unsignaled one strands the last batch of a stream forever.
    if (++since_signal_ >= config_.sig_every || force_signal) {
        wr.send_flags |= IBV_SEND_SIGNALED;
        since_signal_ = 0;
    }

    ibv_send_wr* bad = nullptr;
    int rc = ibv_post_send(qp_, &wr, &bad);
    if (rc == ENOMEM) {
        // Unreachable given the slot check; report back-pressure, do not abort.
        since_signal_ = since_signal_ == 0 ? 0 : since_signal_ - 1;
        return false;
    }
    TT_FATAL(rc == 0, "ibv_post_send failed: {}", std::strerror(rc));
    posted_++;
    return true;
}

bool RdmaChannel::post_write(const RdmaRegion& src, uint64_t src_offset, uint64_t dst_offset, uint32_t len) {
    TT_ASSERT(connected_, "RdmaChannel::post_write before connect()");
    TT_ASSERT(dst_offset + len <= peer_fifo_.len, "write past the peer FIFO");
    return post(
        reinterpret_cast<uint64_t>(src.addr()) + src_offset,
        src.lkey(),
        peer_fifo_.addr + dst_offset,
        peer_fifo_.rkey,
        len,
        /*inline_data=*/false,
        /*force_signal=*/false);
}

bool RdmaChannel::post_doorbell(uint32_t total) {
    TT_ASSERT(connected_, "RdmaChannel::post_doorbell before connect()");
    TT_ASSERT(peer_doorbell_.len >= sizeof(uint32_t), "peer advertised no doorbell slot");
    doorbell_scratch_ = total;
    return post(
        reinterpret_cast<uint64_t>(&doorbell_scratch_),
        0,
        peer_doorbell_.addr,
        peer_doorbell_.rkey,
        sizeof(uint32_t),
        /*inline_data=*/true,
        /*force_signal=*/true);
}

bool RdmaChannel::post_credit(uint32_t value) {
    TT_ASSERT(connected_, "RdmaChannel::post_credit before connect()");
    TT_ASSERT(peer_credit_.len >= sizeof(uint32_t), "peer has no credit region");
    credit_scratch_ = value;
    return post(
        reinterpret_cast<uint64_t>(&credit_scratch_),
        0,
        peer_credit_.addr,
        peer_credit_.rkey,
        sizeof(uint32_t),
        /*inline_data=*/true,
        /*force_signal=*/true);
}

uint32_t RdmaChannel::poll_send() {
    ibv_wc wc[16];
    uint32_t completed = 0;
    int n = ibv_poll_cq(cq_, 16, wc);
    for (int i = 0; i < n; i++) {
        TT_FATAL(
            wc[i].status == IBV_WC_SUCCESS,
            "RDMA completion failed: {} (opcode {})",
            ibv_wc_status_str(wc[i].status),
            static_cast<int>(wc[i].opcode));
        // Unsignaled WRs retire behind the signaled one, so this accounts for
        // every WR up to wr_id.
        reaped_ = wc[i].wr_id + 1;
        completed++;
    }
    return completed;
}

}  // namespace tt::tt_metal::distributed::host_transport
