// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "host_transport.hpp"

#include "host_uva_layout.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(HAS_LIBFABRIC)
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_eq.h>
#endif

namespace tt::tt_metal::experimental {

const char* provider_name(Provider p) {
    switch (p) {
        case Provider::Tcp: return "tcp";
        case Provider::Verbs: return "verbs";
    }
    return "?";
}

const char* provider_label(Provider p) {
    switch (p) {
        case Provider::Tcp: return "tcp";
        case Provider::Verbs: return "verbs";
    }
    return "?";
}

bool provider_from_string(const std::string& s, Provider& out) {
    if (s == "tcp") { out = Provider::Tcp; return true; }
    if (s == "verbs") { out = Provider::Verbs; return true; }
    return false;
}

namespace {

// --- Out-of-band bootstrap ---------------------------------------------------
//
// A plain TCP socket, used before the fabric exists and kept afterwards. It carries
// three things: the fi address and MR key exchange, the clock synchronisation, and the
// end-of-run verdict. ONE channel rather than three, because three would be three things
// that can deadlock against each other in a two-process run.
//
// TCP_NODELAY is not an optimisation here -- the clock sync measures round-trip times on
// this socket, and Nagle would add up to 40 ms to a small message and quietly corrupt
// every offset estimate.
int oob_listen_accept(uint16_t port, uint32_t timeout_ms, std::string& err) {
    const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) {
        err = std::string("socket: ") + std::strerror(errno);
        return -1;
    }
    int one = 1;
    ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons(port);
    if (::bind(ls, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
        err = std::string("bind port ") + std::to_string(port) + ": " + std::strerror(errno);
        ::close(ls);
        return -1;
    }
    if (::listen(ls, 1) != 0) {
        err = std::string("listen: ") + std::strerror(errno);
        ::close(ls);
        return -1;
    }

    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    ::setsockopt(ls, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    const int fd = ::accept(ls, nullptr, nullptr);
    ::close(ls);
    if (fd < 0) {
        err = std::string("accept (peer never connected within ") + std::to_string(timeout_ms) +
              " ms): " + std::strerror(errno);
        return -1;
    }
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

int oob_connect(const std::string& host, uint16_t port, uint32_t timeout_ms, std::string& err) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const std::string svc = std::to_string(port);
    const int gai = ::getaddrinfo(host.c_str(), svc.c_str(), &hints, &res);
    if (gai != 0) {
        err = std::string("getaddrinfo(") + host + "): " + gai_strerror(gai);
        return -1;
    }

    // Retry rather than fail on the first refusal: in a two-process launch the client
    // frequently starts before the server has bound. A single attempt turns an ordinary
    // startup race into a failed run.
    const uint64_t deadline_ms = timeout_ms;
    uint64_t waited = 0;
    int fd = -1;
    while (waited < deadline_ms) {
        fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd >= 0 && ::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
            break;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        timespec ts{0, 100 * 1000 * 1000};  // 100 ms
        nanosleep(&ts, nullptr);
        waited += 100;
    }
    ::freeaddrinfo(res);
    if (fd < 0) {
        err = "could not reach " + host + ":" + svc + " within " + std::to_string(timeout_ms) + " ms";
        return -1;
    }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

bool xfer_all(int fd, void* buf, size_t len, bool sending) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        const ssize_t n = sending ? ::send(fd, p + done, len - done, 0) : ::recv(fd, p + done, len - done, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) {
                continue;
            }
            // EAGAIN/EWOULDBLOCK here means SO_RCVTIMEO fired -- the caller armed a
            // deadline and it expired. Report it rather than looping, which is what turns
            // the barrier's timeout into an actual timeout.
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

}  // namespace

#if !defined(HAS_LIBFABRIC)

bool transport_available() { return false; }

std::unique_ptr<Transport> make_transport(const TransportConfig&, std::string& error) {
    error =
        "built without libfabric. Set TT_LIBFABRIC_ROOT to a prefix containing "
        "include/rdma/fabric.h and re-run cmake. Every local mode still works.";
    return nullptr;
}

#else

bool transport_available() { return true; }

namespace {

// What crosses the bootstrap socket. Fixed-width and explicitly sized: the two ends may
// be different builds, and a struct whose layout depends on the compiler is a wire format
// that works until it does not.
struct WireHello {
    uint64_t magic;
    uint64_t region_base;
    uint64_t region_bytes;
    uint64_t mr_key;
    uint32_t host_id;
    uint32_t chips_per_host;
    uint32_t grid_width;
    uint32_t cores_in_use;
    uint32_t provisioned_cores;
    uint32_t addr_len;
    uint64_t arena_stride;
    uint64_t arena_bytes;
    uint8_t addr[256];  // fi_getname output
};

constexpr uint64_t kHelloMagic = 0x54364855'56414831ull;  // "T6HUVAH1"

// See OpHandle in the header for why the global completion counter had to go. This is the
// state machine that replaces it, and the states are not decoration -- each transition is
// arbitrated by a compare-exchange because a waiter timing out and the progress thread
// reaping the same operation is an ordinary race, not a rare one:
//
//   Free -> Posted        acquire_op(), before the op is handed to libfabric
//   Posted -> Completing  the progress thread claims it; the error text is written HERE
//   Completing -> Done    ... and only then is the final state published, so a waiter that
//   Completing -> Failed      sees Failed is guaranteed to see the error that explains it
//   Posted -> Abandoned   the waiter's deadline expired first
//   Abandoned -> Free     the completion finally arrived; the PROGRESS THREAD frees the slot
enum OpState : uint32_t {
    kOpFree = 0,
    kOpPosted,
    kOpCompleting,
    kOpDone,
    kOpFailed,
    kOpAbandoned,
};

// Far more than this design can have in flight: sends are serialised behind one endpoint
// and one sender thread, so the depth is 1 in practice. Sized for headroom rather than for
// the expected case, and exhaustion is reported by name rather than blocked on.
constexpr uint32_t kMaxOutstandingOps = 256;

// How long the startup probe waits for a small write's completion before concluding the
// provider suppresses it.
constexpr uint32_t kProbeTimeoutMs = 250;

struct FabricOp {
    // fi_context2, not fi_context: it is the larger of the two shapes a provider may
    // demand (FI_CONTEXT vs FI_CONTEXT2), so one struct satisfies either and no code has
    // to branch on which the provider asked for; undersizing this would let the provider
    // write past the object it was given.
    struct fi_context2 provider;

    std::atomic<uint32_t> state{kOpFree};
    uint32_t slot = 0;      // 1-based, matching OpHandle::slot
    uint64_t tag = 0;       // caller's core index, for attributing a stall
    uint64_t posted_ns = 0;
    std::string error;      // written while Completing; read only after Done/Failed

    bool no_waiter = false;
    uint64_t bytes = 0;     // payload size, so a retire sample can carry its own volume
};

class FabricTransport final : public Transport {
public:
    explicit FabricTransport(TransportConfig cfg) : cfg_(std::move(cfg)) {
        // Built here, not in connect(): the context->op table must be immutable for the
        // whole life of the transport, because the progress thread reads it without a lock.
        ops_.reserve(kMaxOutstandingOps);
        by_context_.reserve(kMaxOutstandingOps * 2);
        free_slots_.reserve(kMaxOutstandingOps);
        for (uint32_t i = 0; i < kMaxOutstandingOps; ++i) {
            auto op = std::make_unique<FabricOp>();
            op->slot = i + 1;
            by_context_.emplace(static_cast<const void*>(&op->provider), op.get());
            ops_.push_back(std::move(op));
        }
        // Handed out from the back, so the first run uses slot 1 and a dump reads in order.
        for (uint32_t i = kMaxOutstandingOps; i > 0; --i) {
            free_slots_.push_back(i);
        }
    }

    ~FabricTransport() override {
        stop_progress();
        if (mr_) { fi_close(&mr_->fid); }
        if (ep_ && msg_mode_) {
            // Tell the peer we are going away before the endpoint disappears underneath it.
            // Without this the far side's next write fails with a transport error instead of a
            // clean shutdown, which reads like a fabric fault.
            fi_shutdown(ep_, 0);
        }
        if (ep_) { fi_close(&ep_->fid); }
        if (pep_) { fi_close(&pep_->fid); }
        if (eq_) { fi_close(&eq_->fid); }
        if (av_) { fi_close(&av_->fid); }
        if (cq_) { fi_close(&cq_->fid); }
        if (domain_) { fi_close(&domain_->fid); }
        if (fabric_) { fi_close(&fabric_->fid); }
        if (info_) { fi_freeinfo(info_); }
        if (oob_fd_ >= 0) { ::close(oob_fd_); }
    }

    std::string connect(uint8_t* region_base, uint64_t region_bytes) override {
        region_ = region_base;
        region_bytes_ = region_bytes;

        std::string err;
        oob_fd_ = cfg_.is_server ? oob_listen_accept(cfg_.oob_port, cfg_.timeout_ms, err)
                                 : oob_connect(cfg_.peer_host, cfg_.oob_port, cfg_.timeout_ms, err);
        if (oob_fd_ < 0) {
            return "out-of-band bootstrap failed: " + err;
        }

        msg_mode_ = (cfg_.provider == Provider::Verbs);
        if (msg_mode_) {
            return connect_msg();
        }
        if (std::string e = open_fabric(); !e.empty()) {
            return e;
        }
        if (std::string e = register_region(); !e.empty()) {
            return e;
        }
        return exchange();
    }

    std::string post(uint64_t local_offset, uint64_t remote_offset, uint64_t bytes, uint64_t tag,
                     OpHandle& op) override {
        op = OpHandle{};
        if (local_offset + bytes > region_bytes_) {
            return "post: local offset+len runs off the end of the region";
        }
        void* src = region_ + local_offset;

        if (rma_ && !wait_allowed_) {
            FabricOp* mop = nullptr;
            if (measure_retire_) {
                std::string ignored;
                mop = acquire_op(tag, ignored);
                if (mop == nullptr) {
                    // Gated with the sample it stands in for: `unmeasured` says "this row's
                    // sample is a subset", and a warmup post that was never going to be part
                    // of the row cannot make it one.
                    if (recording_.load(std::memory_order_relaxed)) {
                        retire_unmeasured_.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    mop->no_waiter = true;
                    mop->bytes = bytes;
                }
            }
            const ssize_t rc =
                rma_write(src, bytes, remote_addr(remote_offset), mop != nullptr ? &mop->provider : nullptr);
            if (rc != 0) {
                if (mop != nullptr) {
                    // Never accepted, so no completion is coming for it and the slot is ours to
                    // take back -- the same reasoning as the waited path below.
                    mop->no_waiter = false;
                    release_op(mop);
                }
                return std::string("fi_writemsg(payload, no-wait): ") + fi_strerror(static_cast<int>(-rc)) +
                       diag_text();
            }
            op.inline_done = true;
            posted_.fetch_add(1, std::memory_order_relaxed);
            return {};
        }

        if (rma_ && bytes <= inject_size_ && waw_ordered_) {
            const std::string e = inject_write(src, bytes, remote_addr(remote_offset), "payload");
            if (!e.empty()) {
                return e;
            }
            op.inline_done = true;
            posted_.fetch_add(1, std::memory_order_relaxed);
            injected_.fetch_add(1, std::memory_order_relaxed);
            return {};
        }

        uint64_t wire_bytes = bytes;
        if (rma_ && bytes <= inject_size_ && !small_writes_complete_) {
            const uint64_t padded = inject_size_ + 1;
            if (padded <= kArenaBytes && local_offset + padded <= region_bytes_) {
                wire_bytes = padded;
                announce_padding(bytes, padded);
            }
        }

        std::string acq_err;
        FabricOp* fop = acquire_op(tag, acq_err);
        if (fop == nullptr) {
            return acq_err;
        }
        ssize_t rc;
        if (rma_) {
            // FI_MR_VIRT_ADDR decides what the remote address MEANS: with it set the
            // target is the peer's virtual address, without it the target is an offset
            // from the start of the peer's MR. Getting this wrong does not fail -- it
            // writes to a legal-looking wrong place, which is the worst available
            // outcome, so it is derived from the provider's own mr_mode rather than
            // assumed.
            rc = rma_write(src, wire_bytes, remote_addr(remote_offset), &fop->provider);
        } else {
            do {
                rc = fi_send(ep_, src, bytes, fi_mr_desc(mr_), peer_addr_, &fop->provider);
                if (rc == -FI_EAGAIN) {
                    std::this_thread::yield();
                }
            } while (rc == -FI_EAGAIN);
        }
        if (rc != 0) {
            // Never accepted, so no completion is coming and the slot is ours to take back.
            release_op(fop);
            return std::string(rma_ ? "fi_write: " : "fi_send: ") + fi_strerror(static_cast<int>(-rc));
        }
        op.slot = fop->slot;
        posted_.fetch_add(1, std::memory_order_relaxed);
        return {};
    }

    std::string post_notice(uint32_t dest_core, uint32_t rx_slot, uint64_t length,
                            uint32_t origin_selector, uint64_t elapsed_ns,
                            bool reply, uint32_t stage_slot, OpHandle& op, uint64_t dest_uva) override {
        op = OpHandle{};
        if (!rma_) {
            // Without RMA there is no way to arm a word in the peer's memory. Refuse by
            // name rather than silently skipping the notice and leaving the peer's bytes
            // sitting undelivered in its arena.
            return "post_notice needs FI_RMA; this provider negotiated message mode only";
        }

        // one 32 byte write: control word plus its operands, in the peer's RX control line.
        // See kNoticeCtrlOffset in host_uva_layout.hpp for why this is not four writes.
        // Nothing needs ordering because there is nothing to order -- the receiver cannot
        // observe the trigger without the operands, they are the same transfer.
        // A store notice is one word longer and its opcode says so. Everything downstream --
        // the bounds check, the inject decision, the RMA length -- keys off this one value so
        // the two forms cannot disagree about how many bytes are on the wire.
        const bool is_store = (dest_uva != 0);
        const uint32_t notice_bytes = is_store ? kNoticeStoreBytes : kNoticeBytes;

        const uint64_t stage_off = notice_stage_offset(stage_slot);
        if (stage_off + notice_bytes > region_bytes_) {
            // The source of an RMA must be inside the registration. Off the end is a wrong
            // key for an unregistered page, which providers report as a local protection
            // error on the completion -- a long way from the staging-slot arithmetic that
            // caused it.
            return "post_notice: staging slot " + std::to_string(stage_slot) +
                   " lies outside the registered region";
        }

        auto* slot = reinterpret_cast<uint64_t*>(region_ + stage_off);
        // kOpRdmaWrite rather than kOpRdmaWriteImm on the wire even when the ISSUER used the
        // immediate form: the immediate is a device-side encoding that saves an operand
        // register write over PCIe, and by the time the length has crossed a host it is just
        // a byte count. Re-encoding it into 10 bits here would cap a host-to-host store at
        // 1023 bytes for no reason.
        slot[0] = ctrl_encode(is_store ? kOpRdmaWrite : kOpSendUva, 1u /*base, unused on this path*/, 3u,
                              kFlagStamped | kFlagRemoteNotice | (reply ? kFlagReply : 0ull),
                              notice_seq_++ % kCtrlSeqModulus);
        slot[1] = length;
        slot[2] = elapsed_ns;
        slot[3] = static_cast<uint64_t>(origin_selector);
        if (is_store) {
            slot[4] = dest_uva;
        }

        const uint64_t target = remote_addr(reg_offset(dest_core, rx_slot_reg(rx_slot)));

        if (notice_injectable()) {
            const std::string e = inject_write(slot, notice_bytes, target, "notice");
            if (!e.empty()) {
                return e;
            }
            op.inline_done = true;
            posted_.fetch_add(1, std::memory_order_relaxed);
            injected_.fetch_add(1, std::memory_order_relaxed);
            return {};
        }

        if (!wait_allowed_) {
            const ssize_t rc = rma_write(slot, notice_bytes, target, nullptr);
            if (rc != 0) {
                return std::string("fi_writemsg(notice, no-wait): ") + fi_strerror(static_cast<int>(-rc)) +
                       diag_text();
            }
            op.inline_done = true;
            posted_.fetch_add(1, std::memory_order_relaxed);
            return {};
        }

        // inject_size below 32 B: this provider cannot inject a notice, and by the same token
        // it is not suppressing one either -- a suppressed completion only happens at or under
        // inject_size. So the waited path is safe here by construction.
        std::string acq_err;
        FabricOp* fop = acquire_op(static_cast<uint64_t>(origin_selector), acq_err);
        if (fop == nullptr) {
            return acq_err;
        }
        const ssize_t rc = rma_write(slot, notice_bytes, target, &fop->provider);
        if (rc != 0) {
            release_op(fop);
            return std::string("fi_write(notice): ") + fi_strerror(static_cast<int>(-rc));
        }
        op.slot = fop->slot;
        posted_.fetch_add(1, std::memory_order_relaxed);
        return {};
    }

    std::string post_recv(uint64_t local_offset, uint64_t bytes) override {
        if (rma_) {
            return {};  // one-sided: the bytes land without the target posting anything
        }
        if (local_offset + bytes > region_bytes_) {
            return "post_recv: offset+len runs off the end of the region";
        }
        ssize_t rc;
        do {
            rc = fi_recv(ep_, region_ + local_offset, bytes, fi_mr_desc(mr_), FI_ADDR_UNSPEC, nullptr);
            if (rc == -FI_EAGAIN) {
                std::this_thread::yield();
            }
        } while (rc == -FI_EAGAIN);
        if (rc != 0) {
            return std::string("fi_recv: ") + fi_strerror(static_cast<int>(-rc));
        }
        return {};
    }

    bool try_wait(OpHandle& handle, Completion& out) override {
        if (handle.inline_done) {
            handle = OpHandle{};
            out = Completion{true, {}};
            return true;
        }
        if (!handle.valid()) {
            out = Completion{false, "try_wait() on an operation that was never posted"};
            return true;  // terminal: never posted, so it will never retire
        }
        FabricOp& op = *ops_[handle.slot - 1];
        const uint32_t st = op.state.load(std::memory_order_acquire);
        if (st == kOpDone) {
            handle.slot = 0;
            release_op(&op);
            out = Completion{true, {}};
            return true;
        }
        if (st == kOpFailed) {
            const std::string e = op.error;
            handle.slot = 0;
            release_op(&op);
            out = Completion{false, e};
            return true;
        }
        // kOpPosted or kOpCompleting: still outstanding. The handle is deliberately left
        // INTACT -- unlike wait(), which consumes it on entry -- so the caller polls again.
        //
        // A failed CQ is reported even though the operation has not retired, because nothing
        // ever will: the progress thread is the only thing that could, and it is the thing
        // that failed.
        if (st == kOpPosted) {
            const std::string pe = progress_error();
            if (!pe.empty() && abandon(op)) {
                handle.slot = 0;
                out = Completion{false, pe};
                return true;
            }
        }
        return false;
    }

    Completion wait(OpHandle& handle, uint32_t timeout_ms) override {
        if (handle.inline_done) {
            // Injected: the data left with the call and there is no completion to reap. See
            // OpHandle in the header.
            handle = OpHandle{};
            return Completion{true, {}};
        }
        if (!handle.valid()) {
            return Completion{false, "wait() on an operation that was never posted"};
        }
        FabricOp& op = *ops_[handle.slot - 1];
        handle.slot = 0;  // consumed: waiting twice on one handle would wait on a reused slot

        const uint64_t deadline = now_ms() + (timeout_ms ? timeout_ms : 30000);
        for (;;) {
            const uint32_t st = op.state.load(std::memory_order_acquire);
            if (st == kOpDone) {
                release_op(&op);
                return Completion{true, {}};
            }
            if (st == kOpFailed) {
                const std::string e = op.error;
                release_op(&op);
                return Completion{false, e};
            }
            // kOpCompleting: the progress thread has claimed it and is a few instructions
            // from publishing the verdict. Neither give up nor report anything yet.
            if (st == kOpPosted) {
                const std::string pe = progress_error();
                if (!pe.empty() && abandon(op)) {
                    // The CQ itself has failed, so nothing will ever retire this.
                    return Completion{false, pe};
                }
                if (now_ms() > deadline && abandon(op)) {
                    std::ostringstream m;
                    m << "completion timeout after " << timeout_ms << " ms: the operation (tag "
                      << op.tag << ") was accepted by the provider and never retired. "
                      << "outstanding=" << outstanding() << " unmatched=" << unmatched_.load()
                      << " retired=" << retired_.load() << " of " << posted_.load() << " posted";
                    if (const std::string le = last_error(); !le.empty()) {
                        m << "; last CQ error: " << le;
                    }
                    return Completion{false, m.str()};
                }
            }
            std::this_thread::yield();
        }
    }

    std::string barrier(uint32_t timeout_ms) override {
        struct RecvTimeout {
            int fd;
            RecvTimeout(int f, uint32_t ms) : fd(f) {
                timeval tv{};
                tv.tv_sec = ms / 1000;
                tv.tv_usec = (ms % 1000) * 1000;
                ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            }
            ~RecvTimeout() {
                timeval tv{};
                ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            }
        } guard(oob_fd_, timeout_ms ? timeout_ms : 60000);

        uint8_t tok = 0xB0;
        uint8_t got = 0;
        const char* who = cfg_.is_server ? "server" : "client";
        // Asymmetric order: both sides writing before either reads deadlocks the moment a
        // message outgrows a socket buffer.
        if (cfg_.is_server) {
            if (!xfer_all(oob_fd_, &tok, 1, true)) {
                return std::string("barrier: send failed (") + who + ")";
            }
            if (!xfer_all(oob_fd_, &got, 1, false)) {
                return std::string("barrier: peer never arrived within ") + std::to_string(timeout_ms / 1000) +
                       " s (" + who +
                       " side). It is stuck earlier in its own shutdown, or it died.";
            }
        } else {
            if (!xfer_all(oob_fd_, &got, 1, false)) {
                return std::string("barrier: peer never arrived within ") + std::to_string(timeout_ms / 1000) +
                       " s (" + who +
                       " side). It is stuck earlier in its own shutdown, or it died.";
            }
            if (!xfer_all(oob_fd_, &tok, 1, true)) {
                return std::string("barrier: send failed (") + who + ")";
            }
        }
        if (got != 0xB0) {
            return "barrier: peer sent an unexpected token -- the bootstrap stream is out of sync";
        }
        return {};
    }

    bool atomics_available() const override { return atomics_ok_; }

    // Used to claim a receive slot.
    //
    // fi_fetch_atomic with FI_SUM on FI_UINT64: the peer's word is incremented and its PREVIOUS
    // value lands in `result`. That previous value is the ticket, and ticket % slots is the slot
    // -- so two senders racing get different slots with no lock and no handshake.
    //
    // waited, unlike every other post on this path. The ticket has to be in hand before the
    // payload can be placed, so there is nothing to overlap; and a fire-and-forget atomic whose
    // result never arrives would hand out a slot nobody knows about.
    //
    // The result buffer must live in registered memory for providers that require it, so it is
    // taken from the same per-worker staging area the notices use rather than off the stack.
    std::string fetch_add(uint64_t remote_offset, uint64_t add, uint64_t& out) override {
        if (!atomics_ok_) {
            return "fetch_add: this endpoint did not negotiate FI_ATOMIC for FI_SUM/FI_UINT64";
        }
        std::lock_guard<std::mutex> g(fab_m_);
        uint64_t operand = add;
        uint64_t result = 0;
        const uint64_t target = remote_addr(remote_offset);
        FabricOp* fop = acquire_op(0, out_err_scratch_);
        if (fop == nullptr) {
            return "fetch_add: " + out_err_scratch_;
        }
        // peer_addr_ and peer_.mr_key, exactly as rma_write() addresses the same region.
        const ssize_t rc = fi_fetch_atomic(ep_, &operand, 1, fi_mr_desc(mr_), &result, fi_mr_desc(mr_),
                                           peer_addr_, target, peer_.mr_key,
                                           FI_UINT64, FI_SUM, &fop->provider);
        if (rc != 0) {
            release_op(fop);
            return std::string("fi_fetch_atomic: ") + fi_strerror(static_cast<int>(-rc));
        }
        OpHandle h;
        h.slot = fop->slot;
        const Completion c = wait(h, 30000);
        if (!c.ok) {
            return "fetch_add completion: " + c.error;
        }
        out = result;
        return {};
    }

    std::string post_word(uint64_t remote_offset, uint64_t value) override {
        if (!rma_) {
            return "post_word needs FI_RMA; this provider negotiated message mode only";
        }
        const uint64_t target = remote_addr(remote_offset);
        return inject_write(&value, sizeof(value), target, "word");
    }

    std::string post_credit(uint32_t core, uint32_t my_host, uint64_t count, uint32_t stage_slot) override {
        if (!rma_) {
            return {};  // message mode cannot write the peer's memory; caller copes
        }

        const uint64_t target = remote_addr(credit_word_offset(core, my_host));
        const uint64_t value = count;
        {
            return inject_write(&value, sizeof(value), target, "credit");
        }

        auto* staged = reinterpret_cast<uint64_t*>(region_ + notice_stage_offset(stage_slot) + sizeof(uint64_t) * 4);
        *staged = value;
        std::string acq_err;
        FabricOp* fop = acquire_op(core, acq_err);
        if (fop == nullptr) {
            return acq_err;
        }
        const ssize_t rc = rma_write(staged, sizeof(uint64_t), target, &fop->provider);
        if (rc != 0) {
            release_op(fop);
            return std::string("fi_writemsg(credit): ") + fi_strerror(static_cast<int>(-rc));
        }
        posted_.fetch_add(1, std::memory_order_relaxed);
        OpHandle h;
        h.slot = fop->slot;
        const Completion c = wait(h, 5000);
        return c.ok ? std::string{} : ("credit completion: " + c.error);
    }

    bool rma_in_use() const override { return rma_; }
    const PeerInfo& peer() const override { return peer_; }
    int oob_fd() const override { return oob_fd_; }

    RetireStats retire_stats() const override {
        RetireStats s;
        {
            std::lock_guard<std::mutex> g(retire_m_);
            s.n = retire_n_;
            s.min_ns = retire_min_;
            s.max_ns = retire_max_;
            s.mean_ns = retire_mean_;
            s.m2 = retire_m2_;
            s.bytes = retire_bytes_;
        }
        s.unmeasured = retire_unmeasured_.load(std::memory_order_relaxed);
        return s;
    }

    void set_recording(bool on) override { recording_.store(on, std::memory_order_relaxed); }

    uint64_t tx_depth() const override { return tx_depth_; }

    std::string describe() const override {
        std::ostringstream o;
        o << provider_name(cfg_.provider) << " (" << (rma_ ? "one-sided RMA fi_write" : "two-sided fi_send/fi_recv")
          << ", " << (virt_addr_ ? "FI_MR_VIRT_ADDR" : "MR-relative offsets") << ", "
          << (cfg_.is_server ? "server" : "client") << ")";
        if (info_ && info_->fabric_attr && info_->fabric_attr->prov_name) {
            o << " prov=" << info_->fabric_attr->prov_name;
        }
        if (!local_addr_.empty()) {
            o << " local=" << local_addr_;
        }
        if (info_ != nullptr && info_->tx_attr != nullptr) {
            o << " inject_size=" << inject_size_ << " tx_depth=" << tx_depth_
              << (atomics_ok_ ? " FI_SUM/u64" : " NO-ATOMICS")
              << (waw_ordered_ ? " WAW-ordered" : " unordered-writes")
              << (small_writes_complete_ ? " small-writes-complete" : " SMALL-WRITES-SUPPRESSED");
        }
        if (info_ != nullptr && info_->domain_attr != nullptr) {
            o << " threading=";
            switch (info_->domain_attr->threading) {
                case FI_THREAD_SAFE: o << "SAFE"; break;
                case FI_THREAD_DOMAIN: o << "DOMAIN(not safe)"; break;
                case FI_THREAD_COMPLETION: o << "COMPLETION(not safe)"; break;
                case FI_THREAD_ENDPOINT: o << "ENDPOINT(not safe)"; break;
                case FI_THREAD_FID: o << "FID(not safe)"; break;
                case FI_THREAD_UNSPEC: o << "UNSPEC"; break;
                default: o << static_cast<int>(info_->domain_attr->threading); break;
            }
            o << " data_progress=" << (info_->domain_attr->data_progress == FI_PROGRESS_AUTO ? "AUTO" : "MANUAL");
        }
        if (wait_allowed_) {
            o << (rma_ ? " WAITED(writes not ordered)" : " WAITED(message mode)");
        } else {
            o << " no-wait(ordered writes; credit is the fence)";
        }
        if (info_ != nullptr) {
            o << " mode=";
            if ((info_->mode & FI_CONTEXT) != 0) {
                o << "FI_CONTEXT";
            } else if ((info_->mode & FI_CONTEXT2) != 0) {
                o << "FI_CONTEXT2";
            } else if (info_->mode == 0) {
                o << "none";
            } else {
                o << "0x" << std::hex << info_->mode << std::dec;
            }
        }
        return o.str();
    }

    TransportDiag diag() const override {
        TransportDiag d;
        d.posted = posted_.load(std::memory_order_relaxed);
        d.retired = retired_.load(std::memory_order_relaxed);
        d.unmatched = unmatched_.load(std::memory_order_relaxed);
        d.abandoned = abandoned_.load(std::memory_order_relaxed);
        d.injected = injected_.load(std::memory_order_relaxed);
        uint64_t oldest = UINT64_MAX;
        for (const auto& op : ops_) {
            if (op->state.load(std::memory_order_acquire) == kOpFree) {
                continue;
            }
            ++d.outstanding;
            // The longest-outstanding operation is the one a stall is about.
            if (op->posted_ns < oldest) {
                oldest = op->posted_ns;
                d.oldest_tag = op->tag;
            }
        }
        d.last_error = last_error();
        return d;
    }

private:
    // FI_MR_VIRT_ADDR decides what a remote address MEANS: with it set the target is the
    // peer's virtual address, without it an offset from the start of the peer's MR.
    // Getting this wrong does not fail -- it writes to a legal-looking wrong place.
    uint64_t remote_addr(uint64_t offset) const {
        return virt_addr_ ? (peer_.region_base + offset) : offset;
    }

    // The documented response to -FI_EAGAIN is to read the
    // completion queue and retry, not to spin -- and on a provider with FI_PROGRESS_MANUAL data
    // progress (a layered verbs provider reports exactly that) it is the only thing that can free the
    // resource the post is waiting for. The retry loops here only yielded, which left the
    // posting thread waiting on a provider that was waiting to be poked.
    void progress_once() {
        fi_cq_data_entry e{};
        ssize_t rc;
        {
            std::lock_guard<std::mutex> g(fab_m_);
            rc = fi_cq_read(cq_, &e, 1);
        }
        if (rc == 1) {
            dispatch(e.op_context, true, {});
        }
    }

    ssize_t rma_write(const void* src, size_t bytes, uint64_t target, void* ctx) {
        iovec iov{};
        iov.iov_base = const_cast<void*>(src);
        iov.iov_len = bytes;
        void* desc = fi_mr_desc(mr_);
        fi_rma_iov rma{};
        rma.addr = target;
        rma.len = bytes;
        rma.key = peer_.mr_key;
        fi_msg_rma msg{};
        msg.msg_iov = &iov;
        msg.desc = &desc;
        msg.iov_count = 1;
        msg.addr = peer_addr_;
        msg.rma_iov = &rma;
        msg.rma_iov_count = 1;
        msg.context = ctx;
        msg.data = 0;
        const uint64_t deadline = now_ms() + 10000;
        ssize_t rc;
        for (;;) {
            {
                std::lock_guard<std::mutex> g(fab_m_);
                rc = fi_writemsg(ep_, &msg, FI_COMPLETION);
            }
            if (rc != -FI_EAGAIN) {
                return rc;
            }
            if (now_ms() > deadline) {
                eagain_stalls_.fetch_add(1, std::memory_order_relaxed);
                return -FI_EAGAIN;
            }
            progress_once();            // the documented EAGAIN remedy: read the CQ, then retry
            std::this_thread::yield();
        }
    }

    // An RMA write that carries its data inline and generates no completion.
    std::string inject_write(const void* src, size_t bytes, uint64_t target, const char* what) {
        const uint64_t deadline = now_ms() + 5000;
        for (;;) {
            ssize_t rc;
            {
                std::lock_guard<std::mutex> g(fab_m_);
                rc = fi_inject_write(ep_, src, bytes, peer_addr_, target, peer_.mr_key);
            }
            if (rc == 0) {
                return {};
            }
            if (rc != -FI_EAGAIN) {
                return std::string("fi_inject_write(") + what + "): " + fi_strerror(static_cast<int>(-rc));
            }
            if (now_ms() > deadline) {
                return std::string("fi_inject_write(") + what + "): still EAGAIN after 5 s" + diag_text();
            }
            progress_once();
            std::this_thread::yield();
        }
    }

    std::string diag_text() const {
        std::ostringstream o;
        o << " [posted=" << posted_.load(std::memory_order_relaxed)
          << " retired=" << retired_.load(std::memory_order_relaxed)
          << " injected=" << injected_.load(std::memory_order_relaxed)
          << " unmatched=" << unmatched_.load(std::memory_order_relaxed)
          << " outstanding=" << outstanding() << " eagain_stalls=" << eagain_stalls_.load(std::memory_order_relaxed)
          << "]";
        return o.str();
    }

    bool notice_injectable() const { return rma_ && inject_size_ >= kNoticeStoreBytes; }

    void announce_padding(uint64_t payload, uint64_t padded) {
        if (padding_announced_.exchange(true, std::memory_order_relaxed)) {
            return;
        }
        std::cout << "  NOTE: measured at startup -- this provider did not report a completion for an "
                     "8 B RMA write within "
                  << kProbeTimeoutMs << " ms, and does not advertise FI_ORDER_WAW. Payloads at or under "
                  << inject_size_ << " B (inject_size) are therefore written as " << padded
                  << " B on the wire, so the payload keeps a completion to order it ahead of its "
                     "notice. The peer still reads only the payload length, but stage 2 at "
                  << payload << " B times a " << padded << " B transfer." << std::endl;
    }

    static uint64_t host_now_ns() {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
    }

    static uint64_t now_ms() {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000ull + static_cast<uint64_t>(ts.tv_nsec) / 1000000ull;
    }

    void start_progress() {
        progress_run_.store(true, std::memory_order_release);
        progress_ = std::thread([this] {
            while (progress_run_.load(std::memory_order_acquire)) {
                fi_cq_data_entry e{};
                ssize_t rc;
                {
                    std::lock_guard<std::mutex> g(fab_m_);
                    rc = fi_cq_read(cq_, &e, 1);
                }
                if (rc == 1) {
                    dispatch(e.op_context, true, {});
                    continue;
                }
                if (rc == -FI_EAGAIN) {
                    continue;  // the common case: keep driving progress
                }
                if (rc == -FI_EAVAIL) {
                    // The detail lives in a separate queue; fi_cq_read only says one
                    // exists. Reading it is the difference between a diagnosis and
                    // "something failed".
                    fi_cq_err_entry ee{};
                    ssize_t er;
                    {
                        std::lock_guard<std::mutex> g(fab_m_);
                        er = fi_cq_readerr(cq_, &ee, 0);
                    }
                    std::string msg = "completion error";
                    if (er > 0) {
                        msg += std::string(": ") + fi_cq_strerror(cq_, ee.prov_errno, ee.err_data, nullptr, 0);
                    }
                    set_last_error(msg);
                    // An error entry names the operation that failed, so it fails THAT
                    // waiter. An error with no owner -- a credit inject, which carries no
                    // context by design -- has nobody to report to, so it becomes the
                    // transport-wide error instead of being counted and forgotten.
                    if (er <= 0 || !dispatch(ee.op_context, false, msg)) {
                        set_progress_error(msg);
                    }
                    continue;
                }
                // The queue itself is unusable. Every waiter has to learn this, or they all
                // sit out their full deadlines waiting on a thread that has exited.
                set_progress_error(std::string("fi_cq_read: ") + fi_strerror(static_cast<int>(-rc)));
                return;
            }
        });
    }

    // Routes one completion to the operation that produced it. Returns false when the
    // context belongs to no outstanding operation of ours.
    bool dispatch(void* ctx, bool ok, std::string msg) {
        const auto it = by_context_.find(static_cast<const void*>(ctx));
        if (it == by_context_.end()) {
            unmatched_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        FabricOp* op = it->second;

        if (op->no_waiter) {
            uint32_t want = kOpPosted;
            if (!op->state.compare_exchange_strong(want, kOpCompleting, std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
                unmatched_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            if (ok) {
                // THE MEASUREMENT IS GATED, THE HEALTH COUNTER IS NOT. `retired_` answers
                // "did every posted operation come back", which has to cover the warmup or
                // it stops balancing against `posted_`; record_retire() answers "how long did
                // a steady-state transfer take", which must not.
                if (recording_.load(std::memory_order_relaxed)) {
                    record_retire(host_now_ns() - op->posted_ns, op->bytes);
                }
                retired_.fetch_add(1, std::memory_order_relaxed);
            }
            op->no_waiter = false;
            release_op(op);
            return ok;
        }

        uint32_t expected = kOpPosted;
        if (op->state.compare_exchange_strong(expected, kOpCompleting, std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
            op->error = ok ? std::string{} : std::move(msg);
            // Published last, with release: a waiter that sees Failed is guaranteed to see
            // the error text written above it.
            op->state.store(ok ? kOpDone : kOpFailed, std::memory_order_release);
            retired_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        if (expected == kOpAbandoned) {
            // Its waiter gave up. The provider has now released the context, so this is the
            // first moment the slot can safely be reused -- and we are the only thread left
            // that knows it.
            retired_.fetch_add(1, std::memory_order_relaxed);
            release_op(op);
            return true;
        }
        // A second completion for an operation already retired, or one for a slot between
        // uses. Counted, never acted on.
        unmatched_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // Gives up on an operation the provider still owns. Returns false if it completed while
    // we were deciding, in which case the caller must re-read the state.
    bool abandon(FabricOp& op) {
        uint32_t expected = kOpPosted;
        if (op.state.compare_exchange_strong(expected, kOpAbandoned, std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
            abandoned_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    void record_retire(uint64_t ns, uint64_t bytes) {
        std::lock_guard<std::mutex> g(retire_m_);
        ++retire_n_;
        retire_bytes_ += bytes;
        if (retire_n_ == 1) {
            retire_min_ = retire_max_ = ns;
        } else {
            retire_min_ = std::min(retire_min_, ns);
            retire_max_ = std::max(retire_max_, ns);
        }
        const double d = static_cast<double>(ns) - retire_mean_;
        retire_mean_ += d / static_cast<double>(retire_n_);
        retire_m2_ += d * (static_cast<double>(ns) - retire_mean_);
    }

    FabricOp* acquire_op(uint64_t tag, std::string& err) {
        std::lock_guard<std::mutex> g(pool_m_);
        if (free_slots_.empty()) {
            err = "no free completion slots: " + std::to_string(kMaxOutstandingOps) +
                  " operations are outstanding. Sends are serialised behind one endpoint, so "
                  "this means completions are not retiring, not that the depth is real.";
            return nullptr;
        }
        FabricOp* op = ops_[free_slots_.back() - 1].get();
        free_slots_.pop_back();
        // The provider owns these bytes from here until the completion; hand it a clean
        // object rather than one holding a previous transfer's state.
        std::memset(&op->provider, 0, sizeof(op->provider));
        op->tag = tag;
        op->error.clear();
        op->posted_ns = host_now_ns();
        op->state.store(kOpPosted, std::memory_order_release);
        return op;
    }

    void release_op(FabricOp* op) {
        op->state.store(kOpFree, std::memory_order_release);
        std::lock_guard<std::mutex> g(pool_m_);
        free_slots_.push_back(op->slot);
    }

    uint64_t outstanding() const {
        uint64_t n = 0;
        for (const auto& op : ops_) {
            const uint32_t st = op->state.load(std::memory_order_acquire);
            if (st != kOpFree) {
                ++n;
            }
        }
        return n;
    }

    // progress_error_ and last_error_ are written by the progress thread and read by
    // waiters. They were plain std::strings read without synchronisation, which is a data
    // race on a path whose entire job is to explain a stall.
    void set_progress_error(std::string e) {
        std::lock_guard<std::mutex> g(err_m_);
        if (progress_error_.empty()) {
            progress_error_ = std::move(e);
        }
    }
    std::string progress_error() const {
        std::lock_guard<std::mutex> g(err_m_);
        return progress_error_;
    }
    void set_last_error(std::string e) {
        std::lock_guard<std::mutex> g(err_m_);
        last_error_ = std::move(e);
    }
    std::string last_error() const {
        std::lock_guard<std::mutex> g(err_m_);
        return last_error_;
    }

    void stop_progress() {
        progress_run_.store(false, std::memory_order_release);
        if (progress_.joinable()) {
            progress_.join();
        }
    }

    // ---------------------------------------------------------------------------
    // NATIVE VERBS: FI_EP_MSG, RC, no utility provider.
    //
    // An RDM endpoint is addressed through an
    // address vector: insert the peer's address and every write names it. A MSG endpoint is
    // CONNECTED -- one listener, one accept, one connection -- and after that the RMA calls
    // carry no address at all, because the endpoint IS the peer.
    //
    // THE ORDER IS FORCED, and it is worth reading once because nothing about it is arbitrary:
    //
    //   server                                   client
    //   fi_getinfo(FI_SOURCE) + listen
    //   getname(pep) ---- hello (address) ---->   fi_getinfo(dest_addr = that address)
    //                 <--- hello (geometry) ----  connect
    //   accept                                    (both wait for FI_CONNECTED)
    //   register MR                               register MR
    //                 <--- keys (base, rkey) --->
    //
    // The client cannot call fi_getinfo until it knows the server's address, and neither side
    // can publish an MR key until it has a domain -- which on the server only exists once the
    // connection request has arrived. Hence TWO out-of-band exchanges: addresses first,
    // registration keys after connecting. The single-exchange RDM path cannot be reused, and
    // pretending otherwise is how the mr_key ends up as zero on one side.
    // ---------------------------------------------------------------------------
    std::string connect_msg() {
        std::string e;
        if (cfg_.is_server) {
            if (e = msg_listen(); !e.empty()) {
                return e;
            }
            if (e = exchange_addresses(); !e.empty()) {
                return e;
            }
            if (e = msg_accept(); !e.empty()) {
                return e;
            }
        } else {
            if (e = exchange_addresses(); !e.empty()) {
                return e;
            }
            if (e = msg_do_connect(); !e.empty()) {
                return e;
            }
        }
        if (e = register_region(); !e.empty()) {
            return e;
        }
        if (e = exchange_keys(); !e.empty()) {
            return e;
        }
        // A connected endpoint ignores the address on every transfer. Zero rather than
        // FI_ADDR_UNSPEC: the field is unused, and a sentinel there invites someone to believe
        // it means something.
        peer_addr_ = 0;
        start_progress();
        return probe_small_writes();
    }

    fi_info* msg_hints() const {
        fi_info* hints = fi_allocinfo();
        if (hints == nullptr) {
            return nullptr;
        }
        hints->ep_attr->type = FI_EP_MSG;
        hints->caps = FI_MSG | FI_RMA;
        hints->mode = FI_CONTEXT;
        hints->tx_attr->op_flags = FI_COMPLETION;
        hints->domain_attr->mr_mode =
            FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
        hints->domain_attr->threading = FI_THREAD_SAFE;
        hints->fabric_attr->prov_name = strdup(provider_name(cfg_.provider));
        return hints;
    }

    std::string open_eq() {
        fi_eq_attr eqa{};
        eqa.size = 32;
        // FI_WAIT_UNSPEC so fi_eq_sread can block with a timeout. Connection setup is the one
        // place in this transport where blocking is right: there is nothing to poll for and a
        // spin would just burn a core waiting on a peer that may be seconds away.
        eqa.wait_obj = FI_WAIT_UNSPEC;
        const int rc = fi_eq_open(fabric_, &eqa, &eq_, nullptr);
        return rc == 0 ? std::string{} : std::string("fi_eq_open: ") + fi_strerror(-rc);
    }

    // Reads one CM event, expecting `want`. Bounded, and it names what it was waiting for --
    // a silent hang here is indistinguishable from a wrong address, which is the failure this
    // whole cluster keeps producing.
    std::string await_cm(uint32_t want, const char* what, fi_eq_cm_entry* entry, size_t entry_len) {
        uint32_t event = 0;
        const ssize_t rd = fi_eq_sread(eq_, &event, entry, entry_len,
                                       static_cast<int>(cfg_.timeout_ms), 0);
        if (rd < 0) {
            if (rd == -FI_EAVAIL) {
                fi_eq_err_entry ee{};
                if (fi_eq_readerr(eq_, &ee, 0) > 0) {
                    return std::string("waiting for ") + what + ": " +
                           fi_eq_strerror(eq_, ee.prov_errno, ee.err_data, nullptr, 0);
                }
            }
            return std::string("waiting for ") + what + ": " + fi_strerror(static_cast<int>(-rd)) +
                   " (after " + std::to_string(cfg_.timeout_ms) + " ms)";
        }
        if (event != want) {
            return std::string("waiting for ") + what + ": got CM event " + std::to_string(event) +
                   " instead. A shutdown here means the peer gave up first.";
        }
        return {};
    }

    std::string msg_listen() {
        fi_info* hints = msg_hints();
        if (hints == nullptr) {
            return "fi_allocinfo failed";
        }
        // Service "0": let the fabric pick the port and publish whatever it picked. A fixed
        // fabric port would be a second thing to keep in sync with the peer, on top of the
        // bootstrap port that already has to match.
        const char* node = cfg_.bind_addr.empty() ? nullptr : cfg_.bind_addr.c_str();
        int rc = fi_getinfo(FI_VERSION(1, 18), node, "0", FI_SOURCE, hints, &info_);
        fi_freeinfo(hints);
        if (rc != 0) {
            return std::string("fi_getinfo(") + provider_name(cfg_.provider) + ", FI_EP_MSG): " +
                   fi_strerror(-rc) + ". `fi_info -p " + provider_name(cfg_.provider) +
                   "` lists what this libfabric offers.";
        }
        record_negotiated();
        if ((rc = fi_fabric(info_->fabric_attr, &fabric_, nullptr)) != 0) {
            return std::string("fi_fabric: ") + fi_strerror(-rc);
        }
        if (std::string e = open_eq(); !e.empty()) {
            return e;
        }
        if ((rc = fi_passive_ep(fabric_, info_, &pep_, nullptr)) != 0) {
            return std::string("fi_passive_ep: ") + fi_strerror(-rc);
        }
        if ((rc = fi_pep_bind(pep_, &eq_->fid, 0)) != 0) {
            return std::string("fi_pep_bind(eq): ") + fi_strerror(-rc);
        }
        if ((rc = fi_listen(pep_)) != 0) {
            return std::string("fi_listen: ") + fi_strerror(-rc);
        }
        return {};
    }

    std::string msg_accept() {
        // The connection request carries its own fi_info describing the accepted connection.
        // The endpoint MUST be created from that, not from the listener's info: it is the one
        // that names this particular peer.
        std::vector<uint8_t> buf(sizeof(fi_eq_cm_entry) + 256);
        auto* entry = reinterpret_cast<fi_eq_cm_entry*>(buf.data());
        if (std::string e = await_cm(FI_CONNREQ, "a connection request", entry, buf.size());
            !e.empty()) {
            return e;
        }
        fi_info* cr = entry->info;
        int rc = fi_domain(fabric_, cr, &domain_, nullptr);
        if (rc != 0) {
            fi_freeinfo(cr);
            return std::string("fi_domain: ") + fi_strerror(-rc);
        }
        if (std::string e = open_cq(cr); !e.empty()) {
            fi_freeinfo(cr);
            return e;
        }
        if ((rc = fi_endpoint(domain_, cr, &ep_, nullptr)) != 0) {
            fi_freeinfo(cr);
            return std::string("fi_endpoint: ") + fi_strerror(-rc);
        }
        fi_freeinfo(cr);
        if (std::string e = bind_and_enable(); !e.empty()) {
            return e;
        }
        if ((rc = fi_accept(ep_, nullptr, 0)) != 0) {
            return std::string("fi_accept: ") + fi_strerror(-rc);
        }
        std::vector<uint8_t> buf2(sizeof(fi_eq_cm_entry) + 256);
        return await_cm(FI_CONNECTED, "the connection to complete",
                        reinterpret_cast<fi_eq_cm_entry*>(buf2.data()), buf2.size());
    }

    std::string msg_do_connect() {
        // The server's listener address arrived in the hello. Passed as dest_addr rather than
        // re-resolved from --peer: the bytes the peer published are authoritative, and resolving
        // the name again is how a client ends up connecting to a different interface than the
        // one the server is listening on.
        fi_info* hints = msg_hints();
        if (hints == nullptr) {
            return "fi_allocinfo failed";
        }
        hints->dest_addrlen = hello_in_.addr_len;
        hints->dest_addr = std::malloc(hello_in_.addr_len);
        if (hints->dest_addr == nullptr) {
            fi_freeinfo(hints);
            return "out of memory for the peer address";
        }
        std::memcpy(hints->dest_addr, hello_in_.addr, hello_in_.addr_len);
        int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &info_);
        fi_freeinfo(hints);
        if (rc != 0) {
            return std::string("fi_getinfo(") + provider_name(cfg_.provider) +
                   ", FI_EP_MSG, dest_addr from the peer's hello): " + fi_strerror(-rc);
        }
        record_negotiated();
        if ((rc = fi_fabric(info_->fabric_attr, &fabric_, nullptr)) != 0) {
            return std::string("fi_fabric: ") + fi_strerror(-rc);
        }
        if ((rc = fi_domain(fabric_, info_, &domain_, nullptr)) != 0) {
            return std::string("fi_domain: ") + fi_strerror(-rc);
        }
        if (std::string e = open_eq(); !e.empty()) {
            return e;
        }
        if (std::string e = open_cq(info_); !e.empty()) {
            return e;
        }
        if ((rc = fi_endpoint(domain_, info_, &ep_, nullptr)) != 0) {
            return std::string("fi_endpoint: ") + fi_strerror(-rc);
        }
        if (std::string e = bind_and_enable(); !e.empty()) {
            return e;
        }
        if ((rc = fi_connect(ep_, info_->dest_addr, nullptr, 0)) != 0) {
            return std::string("fi_connect: ") + fi_strerror(-rc);
        }
        std::vector<uint8_t> buf(sizeof(fi_eq_cm_entry) + 256);
        return await_cm(FI_CONNECTED, "the server to accept",
                        reinterpret_cast<fi_eq_cm_entry*>(buf.data()), buf.size());
    }

    std::string open_cq(fi_info* from) {
        fi_cq_attr cqa{};
        cqa.format = FI_CQ_FORMAT_DATA;
        cqa.wait_obj = FI_WAIT_NONE;
        cqa.size = from->tx_attr->size;
        const int rc = fi_cq_open(domain_, &cqa, &cq_, nullptr);
        return rc == 0 ? std::string{} : std::string("fi_cq_open: ") + fi_strerror(-rc);
    }

    std::string bind_and_enable() {
        int rc = fi_ep_bind(ep_, &eq_->fid, 0);
        if (rc != 0) {
            return std::string("fi_ep_bind(eq): ") + fi_strerror(-rc);
        }
        if ((rc = fi_ep_bind(ep_, &cq_->fid, FI_TRANSMIT | FI_RECV)) != 0) {
            return std::string("fi_ep_bind(cq): ") + fi_strerror(-rc);
        }
        if ((rc = fi_enable(ep_)) != 0) {
            return std::string("fi_enable: ") + fi_strerror(-rc);
        }
        // before any traffic probe: fi_atomicvalid needs a live endpoint,
        // and a run that needs the receive-slot pool should refuse at startup rather than
        // fail on its first message. caps says atomics EXIST; this says FI_SUM on u64 works.
        probe_atomics();
        return {};
    }

    // `caps` carrying FI_ATOMIC says the provider does atomics; it does not say WHICH ops on
    // WHICH datatypes, and RoCE RC offers a short list (FetchAdd and CmpSwap on 8 bytes).
    // fi_atomicvalid is the authoritative answer and it costs one call at connect -- so a run
    // that needs the slot pool can refuse at startup rather than failing on its first message.
    void probe_atomics() {
        size_t count = 0;
        atomics_ok_ = (ep_ != nullptr) && (fi_atomicvalid(ep_, FI_UINT64, FI_SUM, &count) == 0) && count > 0;
    }

    void record_negotiated() {
        rma_ = cfg_.prefer_rma && (info_->caps & FI_RMA) != 0;
        virt_addr_ = (info_->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0;
        if (info_->tx_attr != nullptr) {
            inject_size_ = info_->tx_attr->inject_size;
            tx_depth_ = info_->tx_attr->size;
            waw_ordered_ = (info_->tx_attr->msg_order & (FI_ORDER_WAW | FI_ORDER_RMA_WAW)) ==
                           (FI_ORDER_WAW | FI_ORDER_RMA_WAW);
        }
        wait_allowed_ = true;
        (void)waw_ordered_;  // still recorded and still printed on the connected: line
    }

    // msg boostrap: addresses and geometry, before either side has a domain.
    // mr_key and region_base are deliberately left zero here -- there is no MR yet -- and are
    // exchanged by exchange_keys() once the connection exists.
    std::string exchange_addresses() {
        WireHello mine{};
        mine.magic = kHelloMagic;
        mine.arena_stride = kArenaStride;
        mine.arena_bytes = kArenaBytes;
        mine.provisioned_cores = kProvisionedCores;
        mine.host_id = cfg_.host_id;
        mine.chips_per_host = cfg_.chips_per_host;
        mine.grid_width = cfg_.grid_width;
        mine.cores_in_use = cfg_.cores_in_use;
        mine.region_bytes = region_bytes_;

        // The server publishes its listener; the client has nothing to be reached at, because
        // nobody connects to it.
        if (cfg_.is_server) {
            size_t len = sizeof(mine.addr);
            const int rc = fi_getname(&pep_->fid, mine.addr, &len);
            if (rc != 0) {
                return std::string("fi_getname(passive ep): ") + fi_strerror(-rc);
            }
            mine.addr_len = static_cast<uint32_t>(len);
        }
        hello_out_ = mine;

        // the same asymmetry the RDM path uses, and for the same
        // reason: both writing before either reads deadlocks the moment a hello outgrows a
        // socket buffer.
        if (cfg_.is_server) {
            if (!xfer_all(oob_fd_, &mine, sizeof(mine), true)) {
                return "sending hello failed";
            }
            if (!xfer_all(oob_fd_, &hello_in_, sizeof(hello_in_), false)) {
                return "receiving peer hello failed";
            }
        } else {
            if (!xfer_all(oob_fd_, &hello_in_, sizeof(hello_in_), false)) {
                return "receiving peer hello failed";
            }
            if (!xfer_all(oob_fd_, &mine, sizeof(mine), true)) {
                return "sending hello failed";
            }
        }
        if (hello_in_.magic != kHelloMagic) {
            return "peer hello magic mismatch -- the other side is a different build or program";
        }
        if (!cfg_.is_server && hello_in_.addr_len == 0) {
            return "the server published no listener address; it is not in FI_EP_MSG mode";
        }
        return check_geometry(hello_in_);
    }

    // registration keys, once both sides have a domain and an MR. Separate from
    // phase one because on the server the domain does not exist until the connection request
    // has arrived, and an MR key cannot precede its domain.
    std::string exchange_keys() {
        struct WireKeys {
            uint64_t magic;
            uint64_t region_base;
            uint64_t region_bytes;
            uint64_t mr_key;
        };
        WireKeys mine{kHelloMagic, reinterpret_cast<uint64_t>(region_), region_bytes_, fi_mr_key(mr_)};
        WireKeys theirs{};
        if (cfg_.is_server) {
            if (!xfer_all(oob_fd_, &mine, sizeof(mine), true) ||
                !xfer_all(oob_fd_, &theirs, sizeof(theirs), false)) {
                return "exchanging registration keys failed";
            }
        } else {
            if (!xfer_all(oob_fd_, &theirs, sizeof(theirs), false) ||
                !xfer_all(oob_fd_, &mine, sizeof(mine), true)) {
                return "exchanging registration keys failed";
            }
        }
        if (theirs.magic != kHelloMagic) {
            return "registration-key exchange out of sync with the peer";
        }
        peer_.region_base = theirs.region_base;
        peer_.region_bytes = theirs.region_bytes;
        peer_.mr_key = theirs.mr_key;
        peer_.host_id = hello_in_.host_id;
        peer_.chips_per_host = hello_in_.chips_per_host;
        peer_.grid_width = hello_in_.grid_width;
        peer_.cores_in_use = hello_in_.cores_in_use;
        peer_.provisioned_cores = hello_in_.provisioned_cores;
        peer_.arena_stride = hello_in_.arena_stride;
        peer_.arena_bytes = hello_in_.arena_bytes;
        return {};
    }

    // The geometry half of the hello check, factored out so the MSG path enforces exactly the
    // same rules. A disagreement here does not produce a bad offset -- it produces a VALID
    // offset naming a different physical core, which nothing downstream can detect.
    std::string check_geometry(const WireHello& theirs) const {
        if (theirs.arena_stride != kArenaStride || theirs.arena_bytes != kArenaBytes ||
            theirs.provisioned_cores != kProvisionedCores) {
            std::ostringstream m;
            m << "peer geometry disagrees: arena_stride " << theirs.arena_stride << " vs " << kArenaStride
              << ", arena_bytes " << theirs.arena_bytes << " vs " << kArenaBytes << ", provisioned_cores "
              << theirs.provisioned_cores << " vs " << kProvisionedCores
              << ". Both hosts must run builds from the same commit.";
            return m.str();
        }
        if (cfg_.grid_width != 0 && theirs.grid_width != 0 && theirs.grid_width != cfg_.grid_width) {
            std::ostringstream m;
            m << "peer grid_width " << theirs.grid_width << " != ours " << cfg_.grid_width
              << ". The same core index would name different physical cores on the two hosts.";
            return m.str();
        }
        if (theirs.chips_per_host != cfg_.chips_per_host) {
            std::ostringstream m;
            m << "peer chips_per_host " << theirs.chips_per_host << " != ours " << cfg_.chips_per_host
              << ". The UVA selector would decode to a different (host, chip, core) on each side.";
            return m.str();
        }
        return {};
    }

    std::string open_fabric() {
        fi_info* hints = fi_allocinfo();
        if (!hints) {
            return "fi_allocinfo failed";
        }
        // FI_EP_RDM: reliable datagram. Offered by both providers, and it needs no connection
        // state machine -- which matters for any provider with no connections to make.
        hints->ep_attr->type = FI_EP_RDM;
        hints->caps = FI_MSG | FI_RMA;
        hints->mode = FI_CONTEXT;
        // transfers on the endpoint (ep) must complete into the cq. Asked for as an
        // endpoint default as well as per-operation in rma_write(), because a suppressed
        // completion here is not a lost statistic -- it is a wait that never returns. See
        // rma_write() for the measured case: an RMA write under inject_size retiring
        // silently on a layered verbs provider.
        hints->tx_attr->op_flags = FI_COMPLETION;
        hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
        hints->domain_attr->threading = FI_THREAD_SAFE;
        hints->fabric_attr->prov_name = strdup(provider_name(cfg_.provider));

        // 1. FI_SOURCE REQUIRES A NODE OR A SERVICE. Passing FI_SOURCE with both NULL
        //    returns -FI_ENODATA from every provider on this box -- "No data available",
        //    which reads like "this provider is not installed" and is not. Every provider
        //    tried failed with both NULL and succeeded the moment a service string was
        //    supplied.
        //
        // 2. A SERVICE ALONE IS NOT ENOUGH: it binds INADDR_ANY, so fi_getname() hands back
        //    a 0.0.0.0 wildcard, and the peer's fi_av_insert() REFUSES it -- observed as
        //    "fi_av_insert did not accept the peer address" on the client while the server
        //    sat happily waiting. The server must therefore name a routable local address.
        //    Its own hostname is the default; --bind-addr overrides for a box whose
        //    hostname does not resolve to the interface the peer can reach.
        //
        // The client passes neither flag nor node: it needs no source address, it learns
        // the peer's from the address vector.
        std::string bind_node;
        if (cfg_.is_server) {
            if (!cfg_.bind_addr.empty()) {
                bind_node = cfg_.bind_addr;
            } else {
                char hn[256] = {0};
                if (gethostname(hn, sizeof(hn) - 1) == 0) {
                    bind_node = hn;
                }
            }
        }
        const uint64_t flags = cfg_.is_server ? FI_SOURCE : 0;
        const char* service = cfg_.is_server ? "0" : nullptr;
        const char* node = bind_node.empty() ? nullptr : bind_node.c_str();
        int rc = fi_getinfo(FI_VERSION(1, 18), node, service, flags, hints, &info_);
        if (rc != 0 && node != nullptr) {
            // A provider with no network address to bind finds a hostname meaningless.
            // Fall back rather than refuse one that never needed the node.
            rc = fi_getinfo(FI_VERSION(1, 18), nullptr, service, flags, hints, &info_);
        }
        if (rc != 0) {
            // Retry without FI_RMA. A provider may offer messaging only, and the
            // fallback is a supported mode of this transport -- so a provider that cannot
            // do RMA should downgrade rather than fail the run.
            hints->caps = FI_MSG;
            rc = fi_getinfo(FI_VERSION(1, 18), node, service, flags, hints, &info_);
            if (rc != 0 && node != nullptr) {
                rc = fi_getinfo(FI_VERSION(1, 18), nullptr, service, flags, hints, &info_);
            }
        }
        fi_freeinfo(hints);
        if (rc != 0) {
            return std::string("fi_getinfo(") + provider_name(cfg_.provider) + "): " + fi_strerror(-rc) +
                   ". `fi_info -l` lists the providers this libfabric was built with.";
        }

        // putting FI_ORDER_WAW in hints->tx_attr->msg_order would make fi_getinfo refuse a provider that cannot
        // guarantee it, and losing the provider entirely is worse than knowing it does not
        // order writes. inject_size decides which transfers can skip a completion, and
        // FI_ORDER_WAW decides whether a payload may skip one without losing the fence that
        // puts it ahead of its notice. Both are printed by describe().
        // FI_ORDER_RMA_WAW governs RMA writes against each other, which is the ordering this
        // protocol relies on; FI_ORDER_WAW alone is about messages. Ordered writes make the
        // completion redundant, unordered writes make it the only fence there is. All of that
        // lives in record_negotiated(), shared with the FI_EP_MSG path so the two cannot drift.
        record_negotiated();

        if ((rc = fi_fabric(info_->fabric_attr, &fabric_, nullptr)) != 0) {
            return std::string("fi_fabric: ") + fi_strerror(-rc);
        }
        if ((rc = fi_domain(fabric_, info_, &domain_, nullptr)) != 0) {
            return std::string("fi_domain: ") + fi_strerror(-rc);
        }

        fi_cq_attr cqa{};
        cqa.format = FI_CQ_FORMAT_DATA;
        cqa.wait_obj = FI_WAIT_NONE;  // spin; a waitset would add a syscall to every hop we time
        cqa.size = info_->tx_attr->size;
        if ((rc = fi_cq_open(domain_, &cqa, &cq_, nullptr)) != 0) {
            return std::string("fi_cq_open: ") + fi_strerror(-rc);
        }

        fi_av_attr ava{};
        ava.type = FI_AV_MAP;
        ava.count = 2;
        if ((rc = fi_av_open(domain_, &ava, &av_, nullptr)) != 0) {
            return std::string("fi_av_open: ") + fi_strerror(-rc);
        }
        if ((rc = fi_endpoint(domain_, info_, &ep_, nullptr)) != 0) {
            return std::string("fi_endpoint: ") + fi_strerror(-rc);
        }
        if ((rc = fi_ep_bind(ep_, &av_->fid, 0)) != 0) {
            return std::string("fi_ep_bind(av): ") + fi_strerror(-rc);
        }
        if ((rc = fi_ep_bind(ep_, &cq_->fid, FI_TRANSMIT | FI_RECV)) != 0) {
            return std::string("fi_ep_bind(cq): ") + fi_strerror(-rc);
        }
        if ((rc = fi_enable(ep_)) != 0) {
            return std::string("fi_enable: ") + fi_strerror(-rc);
        }
        // probed after enable, before any traffic: fi_atomicvalid needs a live endpoint,
        // and a run that needs the receive-slot pool should refuse at startup rather than
        // fail on its first message. caps says atomics EXIST; this says FI_SUM on u64 works.
        probe_atomics();
        return {};
    }

    std::string register_region() {
        // This is the registration the design is built
        // around: the same span PinnedMemory pinned for the device, pinned again for the
        // NIC. Both are refcounted references to the same pages, so nothing is copied and
        // nothing moves.
        const uint64_t access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
        const int rc = fi_mr_reg(domain_, region_, region_bytes_, access, 0, 0, 0, &mr_, nullptr);
        if (rc != 0) {
            std::ostringstream m;
            m << "fi_mr_reg over " << (region_bytes_ >> 20) << " MiB failed: " << fi_strerror(-rc)
              << ". If this is ENOMEM, RLIMIT_MEMLOCK must cover the region TWICE over -- the TT "
                 "driver's pin and the NIC's registration are independent pins of the same pages.";
            return m.str();
        }
        return {};
    }

    std::string exchange() {
        WireHello mine{};
        mine.magic = kHelloMagic;
        mine.region_base = reinterpret_cast<uint64_t>(region_);
        mine.region_bytes = region_bytes_;
        mine.mr_key = fi_mr_key(mr_);
        mine.arena_stride = kArenaStride;
        mine.arena_bytes = kArenaBytes;
        mine.provisioned_cores = kProvisionedCores;
        mine.host_id = cfg_.host_id;
        mine.chips_per_host = cfg_.chips_per_host;
        mine.grid_width = cfg_.grid_width;
        mine.cores_in_use = cfg_.cores_in_use;

        size_t addr_len = sizeof(mine.addr);
        const int rc = fi_getname(&ep_->fid, mine.addr, &addr_len);
        if (rc != 0) {
            return std::string("fi_getname: ") + fi_strerror(-rc);
        }
        mine.addr_len = static_cast<uint32_t>(addr_len);
        hello_out_ = mine;
        if (addr_len >= sizeof(sockaddr_in)) {
            const auto* sin = reinterpret_cast<const sockaddr_in*>(mine.addr);
            if (sin->sin_family == AF_INET) {
                char buf[INET_ADDRSTRLEN] = {0};
                inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
                local_addr_ = std::string(buf) + ":" + std::to_string(ntohs(sin->sin_port));
            }
        }

        // Server sends first, client sends first -- deliberately NOT symmetric ordering.
        // Both sides writing before either reads would deadlock the moment a hello grows
        // past the socket buffer, which is exactly the kind of bug that appears only when
        // someone adds a field.
        WireHello theirs{};
        if (cfg_.is_server) {
            if (!xfer_all(oob_fd_, &mine, sizeof(mine), true)) {
                return "sending hello failed";
            }
            if (!xfer_all(oob_fd_, &theirs, sizeof(theirs), false)) {
                return "receiving peer hello failed";
            }
        } else {
            if (!xfer_all(oob_fd_, &theirs, sizeof(theirs), false)) {
                return "receiving peer hello failed";
            }
            if (!xfer_all(oob_fd_, &mine, sizeof(mine), true)) {
                return "sending hello failed";
            }
        }

        // A server whose hostname resolves to 127.0.1.1 -- the default /etc/hosts mapping
        // on Debian-derived systems -- publishes that as its fabric address. The client
        // inserts it happily, and then every write it makes goes to its OWN loopback. The
        // failure is silent and asymmetric: the server's sends land, the client's vanish,
        // the server reports `delivered 0` and the client hangs waiting for traffic that
        // was never addressed to it. Measured exactly that way on this pair before this
        // check existed.
        //
        // Only refused when WE are not also on loopback: a genuine same-box run over
        // 127.0.0.1 is legitimate and must keep working.
        if (theirs.addr_len >= sizeof(sockaddr_in)) {
            const auto* peer_sin = reinterpret_cast<const sockaddr_in*>(theirs.addr);
            const auto* my_sin = reinterpret_cast<const sockaddr_in*>(hello_out_.addr);
            const bool peer_loop = peer_sin->sin_family == AF_INET &&
                                   (ntohl(peer_sin->sin_addr.s_addr) >> 24) == 127u;
            const bool my_loop = my_sin->sin_family == AF_INET &&
                                 (ntohl(my_sin->sin_addr.s_addr) >> 24) == 127u;
            if (peer_loop && !my_loop) {
                char buf[INET_ADDRSTRLEN] = {0};
                inet_ntop(AF_INET, &peer_sin->sin_addr, buf, sizeof(buf));
                std::ostringstream m;
                m << "peer published the loopback address " << buf
                  << " -- writes to it would land in OUR OWN memory, not on the peer. Its "
                     "hostname resolves to loopback (the usual /etc/hosts 127.0.1.1 entry). "
                     "Restart the server with --bind-addr <its routable IP>.";
                return m.str();
            }
        }

        if (theirs.magic != kHelloMagic) {
            return "peer hello magic mismatch -- the other side is a different build or a different program";
        }
        // THE GEOMETRY CHECK, on the wire rather than only in the region header. Two
        // hosts with different arena strides compute different destinations for the same
        // core and the write lands in a legal-looking wrong place.
        if (theirs.arena_stride != kArenaStride || theirs.arena_bytes != kArenaBytes ||
            theirs.provisioned_cores != kProvisionedCores) {
            std::ostringstream m;
            m << "peer geometry disagrees: arena_stride " << theirs.arena_stride << " vs " << kArenaStride
              << ", arena_bytes " << theirs.arena_bytes << " vs " << kArenaBytes << ", provisioned_cores "
              << theirs.provisioned_cores << " vs " << kProvisionedCores
              << ". Both hosts must run builds from the same commit.";
            return m.str();
        }

        // grid_width and chips_per_host get the same treatment as the arena geometry,
        // and for a sharper reason: a mismatch there does not produce a bad offset, it
        // produces a VALID offset naming a different physical core. Nothing downstream
        // can detect that, so it has to be refused here.
        if (cfg_.grid_width != 0 && theirs.grid_width != 0 && theirs.grid_width != cfg_.grid_width) {
            std::ostringstream m;
            m << "peer grid_width " << theirs.grid_width << " != ours " << cfg_.grid_width
              << ". The same core index would name different physical cores on the two hosts.";
            return m.str();
        }
        if (theirs.chips_per_host != cfg_.chips_per_host) {
            std::ostringstream m;
            m << "peer chips_per_host " << theirs.chips_per_host << " != ours " << cfg_.chips_per_host
              << ". The UVA selector would decode to a different (host, chip, core) on each side.";
            return m.str();
        }

        if (fi_av_insert(av_, theirs.addr, 1, &peer_addr_, 0, nullptr) != 1) {
            // Say WHAT was rejected. The usual cause is a wildcard source address on the
            // peer -- a server that bound INADDR_ANY publishes 0.0.0.0, which is a legal
            // sockaddr and an unroutable one.
            std::ostringstream m;
            m << "fi_av_insert refused the peer address (" << theirs.addr_len << " bytes: ";
            for (uint32_t i = 0; i < theirs.addr_len && i < 16; ++i) {
                m << std::hex << static_cast<int>(theirs.addr[i]) << " ";
            }
            m << std::dec << "). If that decodes to 0.0.0.0 the peer bound a wildcard "
                 "address -- give the server --bind-addr <its routable IP>.";
            return m.str();
        }

        peer_.region_base = theirs.region_base;
        peer_.region_bytes = theirs.region_bytes;
        peer_.mr_key = theirs.mr_key;
        peer_.host_id = theirs.host_id;
        peer_.chips_per_host = theirs.chips_per_host;
        peer_.grid_width = theirs.grid_width;
        peer_.cores_in_use = theirs.cores_in_use;
        peer_.provisioned_cores = theirs.provisioned_cores;
        peer_.arena_stride = theirs.arena_stride;
        peer_.arena_bytes = theirs.arena_bytes;

        // Only now, with the endpoint enabled and the peer in the AV, is there anything
        // to make progress on.
        start_progress();
        return probe_small_writes();
    }

    // measures if small rma write reports completion, vs inferring it from
    // inject_size. See kSuppressProbeOffset for why the target is a scratch line in the
    // header.
    //
    // Three outcomes, and they are three different facts:
    //   completion arrives  -> small writes are waitable; nothing needs padding.
    //   nothing arrives     -> this provider suppresses them. Payloads that small must be
    //                          padded above inject_size to keep the payload's completion,
    //                          which is the protocol's ordering fence.
    //   ERROR completion    -> RMA to the peer does not work at all. That is worth refusing
    //                          the run over: every later symptom would be a stall or a
    //                          silently unwritten arena, and this says it in one line before
    //                          any measurement is taken.
    std::string probe_small_writes() {
        if (!rma_) {
            return {};
        }
        // 8 bytes: small enough to be injectable on any provider that injects at all, so a
        // suppressing provider will suppress THIS write.
        auto* src = reinterpret_cast<uint64_t*>(region_ + kSuppressProbeOffset);
        *src = kHelloMagic;
        std::string acq_err;
        FabricOp* fop = acquire_op(~0ull /*tag: not a core*/, acq_err);
        if (fop == nullptr) {
            return acq_err;
        }
        const ssize_t rc =
            rma_write(src, sizeof(uint64_t), remote_addr(kSuppressProbeOffset), &fop->provider);
        if (rc != 0) {
            release_op(fop);
            return std::string("small-write probe: fi_writemsg: ") + fi_strerror(static_cast<int>(-rc));
        }
        posted_.fetch_add(1, std::memory_order_relaxed);

        OpHandle h;
        h.slot = fop->slot;
        const Completion c = wait(h, kProbeTimeoutMs);
        if (c.ok) {
            small_writes_complete_ = true;
            return {};
        }
        // Distinguish "no completion" from "the write failed". A failed op has an error text
        // from the CQ; a suppressed one times out with the transport's own wording.
        if (c.error.find("completion timeout") == std::string::npos) {
            return "small-write probe failed: " + c.error +
                   ". An 8-byte RMA write into the peer's header could not be completed, so no "
                   "payload will reach it either. Check the peer's mr_key and region_base in the "
                   "hello, and that both sides advertise the same mr_mode.";
        }
        small_writes_complete_ = false;
        return {};
    }

    TransportConfig cfg_;
    uint8_t* region_ = nullptr;
    uint64_t region_bytes_ = 0;

    fi_info* info_ = nullptr;
    fid_fabric* fabric_ = nullptr;
    fid_domain* domain_ = nullptr;
    fid_ep* ep_ = nullptr;
    fid_cq* cq_ = nullptr;
    fid_av* av_ = nullptr;
    fid_mr* mr_ = nullptr;
    // Connection-oriented path only (native verbs / FI_EP_MSG): an event queue for CM events
    // and, on the server, a passive endpoint that listens. Null on every RDM provider.
    fid_eq* eq_ = nullptr;
    fid_pep* pep_ = nullptr;
    bool msg_mode_ = false;
    fi_addr_t peer_addr_ = FI_ADDR_UNSPEC;

    bool rma_ = false;
    bool virt_addr_ = false;
    uint64_t inject_size_ = 0;
    bool atomics_ok_ = false;      // FI_SUM on FI_UINT64, probed at connect -- see record_negotiated()
    std::string out_err_scratch_;  // acquire_op's error, kept off the hot path's stack
    uint64_t tx_depth_ = 0;  // tx_attr->size; see Transport::tx_depth()
    bool waw_ordered_ = false;
    // Decided in open_fabric() from the granted msg_order, so the default is "no waits where
    // the fabric orders writes" and "waits where it does not" -- never a guess.
    bool wait_allowed_ = true;
    // Assumed true until the startup probe says otherwise, so a provider that never runs the
    // probe (message mode) behaves as it always did.
    bool small_writes_complete_ = true;
    std::atomic<bool> padding_announced_{false};
    std::string local_addr_;
    int oob_fd_ = -1;
    std::thread progress_;
    std::atomic<bool> progress_run_{false};

    // The operation pool. `ops_` and `by_context_` are built in the constructor and never
    // mutated again, which is what lets the progress thread look up a completion's owner
    // without taking a lock. Only the free list is contended.
    std::vector<std::unique_ptr<FabricOp>> ops_;
    std::unordered_map<const void*, FabricOp*> by_context_;
    std::vector<uint32_t> free_slots_;
    std::mutex pool_m_;
    std::atomic<uint64_t> posted_{0};
    std::atomic<uint64_t> retired_{0};
    std::atomic<uint64_t> unmatched_{0};

    // Post->completion samples, and the posts that could not be measured. See record_retire().
    const bool measure_retire_ = cfg_.measure_retire;
    mutable std::mutex retire_m_;
    uint64_t retire_n_ = 0;
    uint64_t retire_min_ = 0;
    uint64_t retire_max_ = 0;
    double retire_mean_ = 0.0;
    double retire_m2_ = 0.0;
    uint64_t retire_bytes_ = 0;
    std::atomic<uint64_t> retire_unmeasured_{0};
    // THE WARMUP GATE for the two counters above. Owned here rather than pointed at in the
    // driver, because this object outlives run_common's frame. True until told otherwise, so
    // a caller that never calls set_recording() measures the whole run as it always did.
    std::atomic<bool> recording_{true};
    std::atomic<uint64_t> abandoned_{0};
    // Transfers that carried their data inline and never entered the pool. Reported so
    // `posted` and `retired` can be reconciled: without it, an injected run looks like a run
    // whose completions all went missing.
    std::atomic<uint64_t> injected_{0};
    // Transfers the provider would not accept within the retry deadline.
    std::atomic<uint64_t> eagain_stalls_{0};

    // hints asks for FI_THREAD_SAFE and the code never checked what came back -- describe()
    // now prints it. Three threads use this endpoint: the sender (fi_writemsg), any scan
    // worker (fi_inject_write, via post_credit), and the progress thread (fi_cq_read). The
    // caller's own tx_mutex covers only the first.
    std::mutex fab_m_;

    mutable std::mutex err_m_;
    std::string progress_error_;
    std::string last_error_;
    uint32_t notice_seq_ = 0;
    PeerInfo peer_{};
    WireHello hello_out_{};
    // The peer's hello, kept because the MSG path needs its address bytes AFTER the exchange:
    // the client cannot call fi_getinfo until it knows where to connect.
    WireHello hello_in_{};
};

}  // namespace

std::unique_ptr<Transport> make_transport(const TransportConfig& cfg, std::string& error) {
    auto t = std::make_unique<FabricTransport>(cfg);
    error.clear();
    return t;
}

#endif  // HAS_LIBFABRIC


std::string connect_mesh(uint32_t num_hosts, uint32_t self, const std::string& addr_csv, uint16_t base_port,
                         const TransportConfig& base_cfg, uint8_t* region_base, uint64_t region_bytes,
                         std::vector<std::unique_ptr<Transport>>& owned, PeerTable& table) {
    const std::vector<std::string> addrs = split_csv(addr_csv);
    if (addrs.size() != num_hosts) {
        return "--peer lists " + std::to_string(addrs.size()) + " addresses but --host-num is " +
               std::to_string(num_hosts) + "; give one per host id, own slot included (it is ignored)";
    }
    table.configure(num_hosts, self);
    for (uint32_t j = 0; j < num_hosts; ++j) {
        if (j == self) {
            continue;
        }
        TransportConfig tc = base_cfg;
        tc.is_server = (self < j);  // the lower id listens
        tc.peer_host = addrs[j];
        tc.oob_port = pair_port(base_port, self, j, num_hosts);
        tc.host_id = self;
        std::string err;
        std::unique_ptr<Transport> t = make_transport(tc, err);
        if (!t) {
            return "transport for host " + std::to_string(j) + " unavailable: " + err;
        }
        std::cerr << "  peer " << j << "      "
                  << (tc.is_server ? std::string("listening") : "connecting to " + addrs[j]) << " on port "
                  << tc.oob_port << "\n";
        if (const std::string e = t->connect(region_base, region_bytes); !e.empty()) {
            return "connect to host " + std::to_string(j) + " failed: " + e;
        }
        const uint32_t said = t->peer().host_id;
        if (said != j) {
            return "host " + std::to_string(j) + " identifies itself as host " + std::to_string(said) +
                   "; the --peer list is miswired or two hosts share an ident";
        }
        if (const std::string e = table.connect_peer(said, t.get()); !e.empty()) {
            return e;
        }
        owned.push_back(std::move(t));
    }
    return {};
}

}  // namespace tt::tt_metal::experimental
