// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-host leg, over MPI one-sided RMA. See host_transport.hpp for the interface and
// for why remote completion is the one thing this backend cannot get for free.
//
// ONE WINDOW, MANY ENDPOINTS. MPI_Win_create is COLLECTIVE, so it cannot live in a per-peer
// Transport the way fi_mr_reg did: N-1 endpoints on each host would mean N-1 collective calls,
// and any rank that ordered them differently would deadlock. The window is therefore a
// process-wide object created exactly once over the region, and a Transport is a (window, peer
// rank) pair -- which is also why PeerInfo carries no base address or key. A window is addressed
// by displacement, and the displacement is the offset the caller already had.

#include "tt_metal/distributed/host_transport.hpp"

#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <sstream>
#include <vector>

#include "host_stats.hpp"  // now_ns()
#include "host_uva_layout.hpp"
#include "multihost/mpi_distributed_context.hpp"

#include <tt-metalium/distributed_context.hpp>

namespace tt::tt_metal::experimental {

namespace {

using tt::tt_metal::distributed::multihost::ContextPtr;
using tt::tt_metal::distributed::multihost::DistributedContext;
using tt::tt_metal::distributed::multihost::MPIContext;
using tt::tt_metal::distributed::multihost::Rank;
using tt::tt_metal::distributed::multihost::Tag;

// The hello's tag, in the context's tag space rather than a raw MPI one.
constexpr int kHelloTagValue = 0x7431;

constexpr uint32_t kMaxOutstandingOps = 256;

enum : uint32_t {
    kOpFree = 0,
    kOpPosted,
    kOpDone,
    kOpFailed,
};

// The staging word for post_word() and post_credit() lives in the tail of a notice staging slot.
// A notice is at most kNoticeStoreBytes (40) and a slot is kNoticeStageSlotBytes (64), so this is
// the first 8-aligned word past any notice the same slot could be carrying.
constexpr uint64_t kStageWordOffset = 48;
static_assert(kStageWordOffset >= kNoticeStoreBytes, "the staged word must not overlap a notice");
static_assert(kStageWordOffset + sizeof(uint64_t) <= kNoticeStageSlotBytes, "and must fit the slot");

constexpr uint64_t kHelloMagic = 0x54364855'56414831ull;  // "T6HUVAH1"

// Used to make a quorum before the window is instantiated. Everything here is either a compile-time
// constant of the layout or the region length that follows from it, so a disagreement means the
// two hosts are running different builds -- and it has to be caught before MPI_Win_create, not
// after: `same_size` is an assertion made by the caller, not something MPI verifies, and creating
// a window under a false one is erroneous rather than an error return.
struct WindowGeometry {
    uint64_t magic;
    uint64_t region_bytes;
    uint64_t arena_stride;
    uint64_t arena_bytes;
    uint32_t provisioned_cores;
    uint32_t pad;
};

// Sent pairwise once the window exists. No address and no key: see PeerInfo.
struct WireHello {
    uint64_t magic;
    uint64_t region_bytes;
    uint32_t host_id;
    uint32_t chips_per_host;
    uint32_t grid_width;
    uint32_t cores_in_use;
    uint32_t provisioned_cores;
    uint32_t pad;
    uint64_t arena_stride;
    uint64_t arena_bytes;
};

std::string mpi_error_text(const char* what, int rc) {
    if (rc == MPI_SUCCESS) {
        return {};
    }
    char buf[MPI_MAX_ERROR_STRING];
    int len = 0;
    if (MPI_Error_string(rc, buf, &len) != MPI_SUCCESS || len <= 0) {
        return std::string(what) + ": MPI error " + std::to_string(rc);
    }
    return std::string(what) + ": " + std::string(buf, static_cast<size_t>(len));
}

// Process-wide window
//
// Built on the first connect() and torn down when the last endpoint goes away. The lock_all is
// taken once and held for the whole run: passive target is the only epoch model that matches a
// design where the target never enters MPI on the data path, and re-taking a lock per message
// would be a synchronisation round trip per message on top of the flush.
class MpiWindow {
public:
    static MpiWindow& instance() {
        static MpiWindow w;
        return w;
    }

    // Idempotent across endpoints, collective across ranks. The second and later callers on this
    // host must present the same span: two different regions would need two windows, and the
    // offsets every caller computes assume there is only one.
    //
    // The window's MPI communicator comes from DistributedContext, not MPI_COMM_WORLD, and the difference
    // is a hang rather than a preference. get_current_world() is the SUB-context communicator
    // after an optional MPI_Comm_split via TT_RUN_SUBCONTEXT_ID. MPI_Win_create is collective
    // over its communicator, so building it on MPI_COMM_WORLD in a split job waits for ranks that
    // are in another subcontext and will never call this. An unsplit job cannot tell the
    // difference, which is exactly why it has to be got right here rather than discovered later.
    std::string ensure(uint8_t* base, uint64_t bytes) {
        std::lock_guard<std::mutex> g(m_);
        if (win_ != MPI_WIN_NULL) {
            if (base != base_ || bytes != bytes_) {
                return "the window is already open over a different region; one process has one "
                       "region and one window";
            }
            ++refs_;
            return {};
        }

        if (!DistributedContext::is_initialized()) {
            return "the distributed context is not initialized; DistributedContext::create() must "
                   "run before a host-to-host transport is opened";
        }

        // MPI_THREAD_MULTIPLE is not optional. The scan workers post credits while the sender thread
        // posts payloads, and osc/pt2pt -- the one component that would break this design by
        // requiring the target to enter MPI -- refuses to build a window at this level, which is
        // how the dangerous configuration excludes itself.
        int provided = MPI_THREAD_SINGLE;
        MPI_Query_thread(&provided);
        if (provided != MPI_THREAD_MULTIPLE) {
            return "MPI was initialized below MPI_THREAD_MULTIPLE; this transport posts from the "
                   "sender thread and the scan workers at the same time";
        }

        // duplicate() rather than the context's own communicator: the window then has a private
        // comm, so its collectives cannot interleave with the socket's own point-to-point or with
        // anything else the context is used for. Held for the window's lifetime.
        ctx_ = DistributedContext::get_current_world()->duplicate();
        const auto* mpi_ctx = dynamic_cast<const MPIContext*>(ctx_.get());
        if (mpi_ctx == nullptr) {
            return "the distributed context is not an MPI context; the host-to-host transport is "
                   "MPI one-sided RMA and has no other backend";
        }
        comm_ = mpi_ctx->comm();
        rank_ = *ctx_->rank();
        size_ = *ctx_->size();
        fault_tolerant_ = ctx_->supports_fault_tolerance();

        if (const std::string e = agree_on_geometry(bytes); !e.empty()) {
            comm_ = MPI_COMM_NULL;
            ctx_.reset();
            return e;
        }

        MPI_Info info = MPI_INFO_NULL;
        MPI_Info_create(&info);
        // Every rank contributes the same statically-sized region at the same displacement unit,
        // which lets a runtime skip the per-target size/unit table.
        MPI_Info_set(info, "same_size", "true");
        MPI_Info_set(info, "same_disp_unit", "true");
        // The only accumulate this transport issues is the ticket's fetch-and-add, so the runtime
        // is free to use the cheapest ordering that keeps same-location sums coherent.
        MPI_Info_set(info, "accumulate_ops", "same_op_no_op");

        // disp_unit 1: displacements are BYTES, so every offset host_uva_layout.hpp computes is
        // usable as a target displacement with no scaling.
        const int rc = MPI_Win_create(base, static_cast<MPI_Aint>(bytes), 1, info, comm_, &win_);
        MPI_Info_free(&info);
        if (rc != MPI_SUCCESS) {
            comm_ = MPI_COMM_NULL;
            ctx_.reset();
            win_ = MPI_WIN_NULL;
            return mpi_error_text("MPI_Win_create", rc) +
                   ". If this names a threading level, the runtime picked an osc component that "
                   "cannot do passive-target RMA under MPI_THREAD_MULTIPLE -- osc/pt2pt is the "
                   "usual one, and it is also the one that would deadlock this design.";
        }

        // Errors on RMA are returned rather than fatal, so a stalled endpoint reports itself
        // instead of taking the job down with a handler nobody installed.
        MPI_Win_set_errhandler(win_, MPI_ERRORS_RETURN);

        if (const std::string e = check_memory_model(); !e.empty()) {
            MPI_Win_free(&win_);
            MPI_Comm_free(&comm_);
            win_ = MPI_WIN_NULL;
            comm_ = MPI_COMM_NULL;
            return e;
        }

        if (const int lrc = MPI_Win_lock_all(MPI_MODE_NOCHECK, win_); lrc != MPI_SUCCESS) {
            MPI_Win_free(&win_);
            MPI_Comm_free(&comm_);
            win_ = MPI_WIN_NULL;
            comm_ = MPI_COMM_NULL;
            return mpi_error_text("MPI_Win_lock_all", lrc);
        }

        base_ = base;
        bytes_ = bytes;
        refs_ = 1;
        return {};
    }

    void release() {
        std::lock_guard<std::mutex> g(m_);
        if (win_ == MPI_WIN_NULL || --refs_ > 0) {
            return;
        }
        MPI_Win_unlock_all(win_);
        MPI_Win_free(&win_);
        // comm_ is NOT freed here: it belongs to ctx_, and dropping the context is what releases
        // it. Freeing it directly would leave the context holding a dead communicator.
        win_ = MPI_WIN_NULL;
        comm_ = MPI_COMM_NULL;
        ctx_.reset();
        base_ = nullptr;
        bytes_ = 0;
    }

    MPI_Win win() const { return win_; }
    MPI_Comm comm() const { return comm_; }
    int rank() const { return rank_; }
    int size() const { return size_; }
    const std::string& model() const { return model_; }
    bool fault_tolerant() const { return fault_tolerant_; }
    const ContextPtr& context() const { return ctx_; }

    // An unrecoverable window fault is not something a caller can route around: the region is
    // half-written on some peer and no error string returned up the stack changes that. Ending
    // the job through the context is the honest response, and it is what the context is for.
    [[noreturn]] void fatal(const std::string& why) const {
        std::cerr << "host-to-host transport: unrecoverable: " << why << "\n";
        if (ctx_) {
            ctx_->abort(1);
        }
        std::abort();
    }

private:
    MpiWindow() = default;

    // This runs BEFORE MPI_Win_create. `same_size` is an assertion the
    // caller makes and MPI does not check, so a window built while it is false is undefined --
    // not an error return, and not something a later per-peer check can undo. The pairwise hello
    // in exchange() used to carry these fields and verify them after the fact, which was a check
    // that could only ever report damage already done.
    //
    // Collective, and it has to be: every rank contributes to one window, so every rank must
    // agree, not just the pairs that happen to talk to each other.
    std::string agree_on_geometry(uint64_t bytes) {
        WindowGeometry mine{};
        mine.magic = kHelloMagic;
        mine.region_bytes = bytes;
        mine.arena_stride = kArenaStride;
        mine.arena_bytes = kArenaBytes;
        mine.provisioned_cores = kProvisionedCores;

        std::vector<WindowGeometry> all(static_cast<size_t>(size_));
        ctx_->all_gather(
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&mine), sizeof(mine)),
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(all.data()), all.size() * sizeof(WindowGeometry)));

        for (int r = 0; r < size_; ++r) {
            const WindowGeometry& t = all[static_cast<size_t>(r)];
            if (t.magic != kHelloMagic) {
                return "rank " + std::to_string(r) +
                       " sent a bad geometry magic; that rank is not running this protocol";
            }
            if (t.region_bytes == mine.region_bytes && t.arena_stride == mine.arena_stride &&
                t.arena_bytes == mine.arena_bytes && t.provisioned_cores == mine.provisioned_cores) {
                continue;
            }
            std::ostringstream m;
            m << "rank " << r << " disagrees about the window geometry: region_bytes " << t.region_bytes
              << " vs " << mine.region_bytes << ", arena_stride " << t.arena_stride << " vs "
              << mine.arena_stride << ", arena_bytes " << t.arena_bytes << " vs " << mine.arena_bytes
              << ", provisioned_cores " << t.provisioned_cores << " vs " << mine.provisioned_cores
              << ". Every rank contributes to one window and `same_size` is asserted on it, so this "
                 "must match everywhere. Both hosts must run builds from the same commit and "
                 "provision the same number of cores.";
            return m.str();
        }
        return {};
    }

    // Receiver reads from RX region with ordinary loads. That is only defined when the
    // window's public and private copies are the same memory. Under MPI_WIN_SEPARATE a peer's
    // put may never become visible to a local load, and the failure mode is a scanner that
    // silently never sees a payload -- so this refuses the run rather than discovering it as a
    // hang.
    std::string check_memory_model() {
        int* model = nullptr;
        int flag = 0;
        MPI_Win_get_attr(win_, MPI_WIN_MODEL, &model, &flag);
        if (flag == 0 || model == nullptr) {
            model_ = "unknown";
            return "the runtime did not report MPI_WIN_MODEL; this transport needs MPI_WIN_UNIFIED "
                   "and cannot verify it";
        }
        if (*model == MPI_WIN_UNIFIED) {
            model_ = "UNIFIED";
            return {};
        }
        model_ = "SEPARATE";
        return "this runtime gives MPI_WIN_SEPARATE windows. The receive path reads the region "
               "with ordinary loads, which only observes a peer's writes under MPI_WIN_UNIFIED. "
               "Select an RDMA-capable osc component (osc/rdma or osc/ucx).";
    }

    std::mutex m_;
    ContextPtr ctx_;  // owns comm_; outlives the window by construction
    MPI_Win win_ = MPI_WIN_NULL;
    MPI_Comm comm_ = MPI_COMM_NULL;
    uint8_t* base_ = nullptr;
    uint64_t bytes_ = 0;
    uint32_t refs_ = 0;
    int rank_ = 0;
    int size_ = 0;
    bool fault_tolerant_ = false;
    std::string model_ = "unknown";
};

struct MpiOp {
    MPI_Request request = MPI_REQUEST_NULL;
    std::atomic<uint32_t> state{kOpFree};
    uint32_t slot = 0;  // 1-based, matching OpHandle::slot
    uint64_t tag = 0;   // caller's core index, for attributing a stall
    uint64_t posted_ns = 0;
    uint64_t bytes = 0;
    std::string error;
};

class MpiRmaTransport final : public Transport {
public:
    explicit MpiRmaTransport(TransportConfig cfg) : cfg_(std::move(cfg)) {
        ops_.reserve(kMaxOutstandingOps);
        free_slots_.reserve(kMaxOutstandingOps);
        for (uint32_t i = 0; i < kMaxOutstandingOps; ++i) {
            auto op = std::make_unique<MpiOp>();
            op->slot = i + 1;
            ops_.push_back(std::move(op));
        }
        for (uint32_t i = kMaxOutstandingOps; i > 0; --i) {
            free_slots_.push_back(i);
        }
    }

    ~MpiRmaTransport() override {
        if (connected_) {
            MpiWindow::instance().release();
        }
    }

    std::string connect(uint8_t* region_base, uint64_t region_bytes) override {
        region_ = region_base;
        region_bytes_ = region_bytes;

        if (const std::string e = MpiWindow::instance().ensure(region_base, region_bytes); !e.empty()) {
            return e;
        }
        connected_ = true;

        MpiWindow& w = MpiWindow::instance();
        peer_rank_ = static_cast<int>(cfg_.peer_rank);
        if (peer_rank_ < 0 || peer_rank_ >= w.size()) {
            return "peer rank " + std::to_string(cfg_.peer_rank) + " is outside the " +
                   std::to_string(w.size()) + "-rank job";
        }
        if (peer_rank_ == w.rank()) {
            return "peer rank " + std::to_string(cfg_.peer_rank) +
                   " is THIS rank; routing should have taken the local arm before here";
        }
        return exchange();
    }

    std::string post(uint64_t local_offset, uint64_t remote_offset, uint64_t bytes, uint64_t tag,
                     OpHandle& op) override {
        op = OpHandle{};
        if (local_offset + bytes > region_bytes_) {
            return "post: local offset+len runs off the end of the region";
        }
        // MPI counts are int. An arena is 1.5 MiB so this cannot fire today, but it is the check
        // that turns a future arena growth into an error instead of a silent truncation.
        if (bytes > static_cast<uint64_t>(INT32_MAX)) {
            return "post: " + std::to_string(bytes) +
                   " bytes exceeds what an MPI_Rput count can carry; this needs the MPI-4 "
                   "large-count entry points";
        }

        std::string acq_err;
        MpiOp* mop = acquire_op(tag, acq_err);
        if (mop == nullptr) {
            return acq_err;
        }
        mop->bytes = bytes;

        int rc;
        {
            std::lock_guard<std::mutex> g(mpi_m_);
            rc = MPI_Rput(region_ + local_offset, static_cast<int>(bytes), MPI_BYTE, peer_rank_,
                          static_cast<MPI_Aint>(remote_offset), static_cast<int>(bytes), MPI_BYTE,
                          MpiWindow::instance().win(), &mop->request);
        }
        if (rc != MPI_SUCCESS) {
            // Never accepted, so no completion is coming and the slot is ours to take back.
            release_op(mop);
            return mpi_error_text("MPI_Rput(payload)", rc);
        }
        op.slot = mop->slot;
        posted_.fetch_add(1, std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
        return {};
    }

    // MPI_Win_flush is the only operation that promises the
    // bytes are at the target; the request MPI_Rput handed back promises only that the origin
    // buffer is reusable. There is no nonblocking form of this in MPI-4.1, which is why the
    // sender amortises it over a lap rather than overlapping it.
    std::string flush() override {
        std::lock_guard<std::mutex> g(mpi_m_);
        const int rc = MPI_Win_flush(peer_rank_, MpiWindow::instance().win());
        if (rc != MPI_SUCCESS) {
            std::string e = mpi_error_text("MPI_Win_flush", rc);
            set_last_error(e);
            // Failed flushes are non-recoverable; and it is the one error where continuing is
            // actively worse than stopping: it leaves the peer's region in a state nobody can
            // name -- some payloads landed, some did not, and the caller is about to arm triggers
            // over them. On a fault-tolerant context a caller could in principle revoke and
            // shrink; without one there is nothing to do but end the job while the evidence is
            // still true.
            if (!MpiWindow::instance().fault_tolerant()) {
                MpiWindow::instance().fatal(e);
            }
            return e;
        }
        // Cleared after the flush returns, and only on success: a flush that failed left the
        // writes exactly as unflushed as it found them, and clearing here would retire the one
        // signal that says so.
        dirty_.store(false, std::memory_order_release);
        return {};
    }

    bool needs_flush() const override { return dirty_.load(std::memory_order_acquire); }

    std::string post_notice(uint32_t dest_core, uint32_t rx_slot, uint64_t length,
                            uint32_t origin_selector, uint64_t elapsed_ns, bool reply,
                            uint32_t stage_slot, OpHandle& op, uint64_t dest_uva) override {
        op = OpHandle{};

        // single write, not four. The receiver cannot observe the trigger without the operands
        // because they are the same transfer -- which matters more here than it did under
        // libfabric: MPI places NO ordering between two separate puts, not even to the same
        // target, so splitting this would need a flush between the halves.
        const bool is_store = (dest_uva != 0);
        const uint32_t notice_bytes = is_store ? kNoticeStoreBytes : kNoticeBytes;

        const uint64_t stage_off = notice_stage_offset(stage_slot);
        if (stage_off + notice_bytes > region_bytes_) {
            return "post_notice: staging slot " + std::to_string(stage_slot) +
                   " lies outside the region";
        }

        auto* slot = reinterpret_cast<uint64_t*>(region_ + stage_off);
        slot[0] = ctrl_encode(is_store ? kOpRdmaWrite : kOpSendUva, 1u /*base, unused on this path*/, 3u,
                              kFlagStamped | kFlagRemoteNotice | (reply ? kFlagReply : 0ull),
                              notice_seq_++ % kCtrlSeqModulus);
        slot[1] = length;
        slot[2] = elapsed_ns;
        slot[3] = static_cast<uint64_t>(origin_selector);
        if (is_store) {
            slot[4] = dest_uva;
        }

        const uint64_t target = reg_offset(dest_core, rx_slot_reg(rx_slot));

        std::string acq_err;
        MpiOp* mop = acquire_op(dest_core, acq_err);
        if (mop == nullptr) {
            return acq_err;
        }
        mop->bytes = notice_bytes;

        int rc;
        {
            std::lock_guard<std::mutex> g(mpi_m_);
            rc = MPI_Rput(slot, static_cast<int>(notice_bytes), MPI_BYTE, peer_rank_,
                          static_cast<MPI_Aint>(target), static_cast<int>(notice_bytes), MPI_BYTE,
                          MpiWindow::instance().win(), &mop->request);
        }
        if (rc != MPI_SUCCESS) {
            release_op(mop);
            return mpi_error_text("MPI_Rput(notice)", rc);
        }
        op.slot = mop->slot;
        posted_.fetch_add(1, std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
        return {};
    }

    bool try_wait(OpHandle& handle, Completion& out) override {
        if (handle.inline_done) {
            out.ok = true;
            out.error.clear();
            handle = OpHandle{};
            return true;
        }
        if (handle.slot == 0 || handle.slot > kMaxOutstandingOps) {
            out.ok = false;
            out.error = "try_wait: handle names no operation";
            return true;
        }
        MpiOp& op = *ops_[handle.slot - 1];

        int done = 0;
        int rc;
        {
            std::lock_guard<std::mutex> g(mpi_m_);
            rc = MPI_Test(&op.request, &done, MPI_STATUS_IGNORE);
        }
        if (rc != MPI_SUCCESS) {
            out.ok = false;
            out.error = mpi_error_text("MPI_Test", rc);
            set_last_error(out.error);
            retire(op, /*ok=*/false);
            handle = OpHandle{};
            return true;
        }
        if (done == 0) {
            // Still outstanding: leave the handle UNTOUCHED so the caller can poll it again.
            return false;
        }
        record_retire(now_ns() - op.posted_ns, op.bytes);
        retire(op, /*ok=*/true);
        out.ok = true;
        out.error.clear();
        handle = OpHandle{};
        return true;
    }

    Completion wait(OpHandle& handle, uint32_t timeout_ms) override {
        Completion c;
        if (handle.inline_done) {
            handle = OpHandle{};
            c.ok = true;
            return c;
        }
        if (handle.slot == 0 || handle.slot > kMaxOutstandingOps) {
            c.error = "wait: handle names no operation";
            return c;
        }
        // MPI_Wait has no timeout, and blocking forever in a transport whose whole diagnostic
        // story is "which operation stalled" would throw that story away. Poll MPI_Test to the
        // caller's deadline instead, then report the tag so a stall names a core.
        const uint64_t deadline = now_ns() + static_cast<uint64_t>(timeout_ms ? timeout_ms : 30000) * 1000000ull;
        for (;;) {
            if (try_wait(handle, c)) {
                return c;
            }
            if (now_ns() > deadline) {
                MpiOp& op = *ops_[handle.slot - 1];
                c.ok = false;
                c.error = "operation on core " + std::to_string(op.tag) + " did not complete within " +
                          std::to_string(timeout_ms) + " ms";
                // Abandoned, not released: MPI still owns the request and reusing the slot would
                // hand a live request to the next caller.
                abandoned_.fetch_add(1, std::memory_order_relaxed);
                handle = OpHandle{};
                return c;
            }
            std::this_thread::yield();
        }
    }

    // Barrier by the context, which is collective over the window's communicator rather than
    // pairwise with this endpoint's peer.
    //
    // That is safe here because it is called once per endpoint and D2H2H2DSocket admits exactly
    // two hosts, so every rank calls it the same number of times. If the socket ever admits more,
    // this becomes a correctness coupling rather than a detail: N ranks each barriering N-1 times
    // agree only while N is uniform. The alternative -- ctx->send/recv with the peer -- is a
    // three-line change, and that comment is the reason to keep it in mind.
    std::string barrier(uint32_t timeout_ms) override {
        (void)timeout_ms;  // a collective has no timeout; a hung peer hangs here by design
        const ContextPtr& ctx = MpiWindow::instance().context();
        if (!ctx) {
            return "barrier: the transport is not connected";
        }
        ctx->barrier();
        return {};
    }

    // MPI always provides MPI_Fetch_and_op. Whether it is a hardware FetchAdd or an emulation is
    // the runtime's business; either way the semantics hold, so unlike libfabric there is no
    // capability here that a run could fail to negotiate.
    bool atomics_available() const override { return true; }

    std::string fetch_add(uint64_t remote_offset, uint64_t add, uint64_t& out) override {
        // The result buffer must live in the window: MPI does not require it, but keeping it
        // inside the registered span is what lets an RDMA osc component land the reply without
        // a bounce, exactly as the libfabric path needed.
        auto* result = reinterpret_cast<uint64_t*>(region_ + notice_stage_offset(0) + kStageWordOffset);
        uint64_t operand = add;

        std::lock_guard<std::mutex> g(mpi_m_);
        MPI_Win win = MpiWindow::instance().win();
        int rc = MPI_Fetch_and_op(&operand, result, MPI_UINT64_T, peer_rank_,
                                  static_cast<MPI_Aint>(remote_offset), MPI_SUM, win);
        if (rc != MPI_SUCCESS) {
            return mpi_error_text("MPI_Fetch_and_op", rc);
        }
        // The fetched value is NOT valid until the operation completes locally; the ticket is
        // needed before the payload can be placed, so there is nothing to overlap with anyway.
        rc = MPI_Win_flush_local(peer_rank_, win);
        if (rc != MPI_SUCCESS) {
            return mpi_error_text("MPI_Win_flush_local(fetch_add)", rc);
        }
        out = *result;
        return {};
    }

    std::string post_word(uint64_t remote_offset, uint64_t value) override {
        return staged_word(remote_offset, value, 0);
    }

    std::string post_credit(uint32_t core, uint32_t my_host, uint64_t count, uint32_t stage_slot) override {
        return staged_word(credit_word_offset(core, my_host), count, stage_slot);
    }

    const PeerInfo& peer() const override { return peer_; }

    RetireStats retire_stats() const override {
        RetireStats s;
        std::lock_guard<std::mutex> g(retire_m_);
        s.n = retire_n_;
        s.min_ns = retire_min_;
        s.max_ns = retire_max_;
        s.mean_ns = retire_mean_;
        s.m2 = retire_m2_;
        s.bytes = retire_bytes_;
        s.unmeasured = retire_unmeasured_.load(std::memory_order_relaxed);
        return s;
    }

    void set_recording(bool on) override { recording_.store(on, std::memory_order_relaxed); }

    std::string describe() const override {
        MpiWindow& w = MpiWindow::instance();
        char libver[MPI_MAX_LIBRARY_VERSION_STRING];
        int vlen = 0;
        if (MPI_Get_library_version(libver, &vlen) != MPI_SUCCESS || vlen <= 0) {
            std::snprintf(libver, sizeof(libver), "unknown MPI");
        } else if (char* nl = std::strchr(libver, '\n'); nl != nullptr) {
            *nl = '\0';
        }
        std::ostringstream o;
        o << "mpi-rma (one-sided MPI_Rput, window displacements, rank " << w.rank() << " -> "
          << peer_rank_ << " of " << w.size() << ") model=" << w.model() << " lib=" << libver;
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
            if (op->posted_ns < oldest) {
                oldest = op->posted_ns;
                d.oldest_tag = op->tag;
            }
        }
        d.last_error = last_error();
        return d;
    }

private:
    // An 8-byte put from a slot inside the window. MPI has no inject, so the value cannot be
    // passed from the caller's stack: the source must stay live until local completion, and this
    // path deliberately does not wait for one.
    std::string staged_word(uint64_t remote_offset, uint64_t value, uint32_t stage_slot) {
        const uint64_t off = notice_stage_offset(stage_slot) + kStageWordOffset;
        if (off + sizeof(uint64_t) > region_bytes_) {
            return "staged word: slot " + std::to_string(stage_slot) + " lies outside the region";
        }
        auto* staged = reinterpret_cast<uint64_t*>(region_ + off);

        // Staging write is owned by the lock's scope with the MPI_Put that sources it.
        //
        // my_stage_slot() hands out slots `% kNoticeStageSlots` (32), thread_local, so a run with
        // MORE THAN 32 SCAN WORKERS has two workers sharing a slot. With this store outside the
        // mutex, the second worker overwrote the first's value before the first reached its
        // MPI_Put, and that Put then sent the WRONG COUNT to its own core's credit word. A credit
        // is a monotonic count, so a count from another core stalls the reader permanently.
        //
        // Store and Put must be one critical section: the lock is already taken for every credit,
        // so moving an 8-byte store inside it costs nothing measurable. Raising the slot count
        // instead would only move the cliff to the next thread count.
        std::lock_guard<std::mutex> g(mpi_m_);
        *staged = value;
        MPI_Win win = MpiWindow::instance().win();
        int rc = MPI_Put(staged, sizeof(uint64_t), MPI_BYTE, peer_rank_,
                         static_cast<MPI_Aint>(remote_offset), sizeof(uint64_t), MPI_BYTE, win);
        if (rc != MPI_SUCCESS) {
            return mpi_error_text("MPI_Put(word)", rc);
        }
        // local completion, which is NOT the round trip the caller is trying to avoid: it says
        // the runtime is done reading `staged`, nothing about the peer. Without it the staging
        // slot -- one of kNoticeStageSlots, handed out round-robin -- could be rewritten by the
        // next caller while this put is still sourcing from it, which is a corrupted credit
        // rather than a late one.
        rc = MPI_Win_flush_local(peer_rank_, win);
        if (rc != MPI_SUCCESS) {
            return mpi_error_text("MPI_Win_flush_local(word)", rc);
        }
        posted_.fetch_add(1, std::memory_order_relaxed);
        injected_.fetch_add(1, std::memory_order_relaxed);
        // Remote arrival is still owed. needs_flush() is how the sender loop finds it.
        dirty_.store(true, std::memory_order_release);
        return {};
    }

    std::string exchange() {
        WireHello mine{};
        mine.magic = kHelloMagic;
        mine.region_bytes = region_bytes_;
        mine.host_id = cfg_.host_id;
        mine.chips_per_host = cfg_.chips_per_host;
        mine.grid_width = cfg_.grid_width;
        mine.cores_in_use = cfg_.cores_in_use;
        mine.provisioned_cores = kProvisionedCores;
        mine.arena_stride = kArenaStride;
        mine.arena_bytes = kArenaBytes;

        const ContextPtr& ctx = MpiWindow::instance().context();
        if (!ctx) {
            return "hello: the window has no context";
        }
        const Rank peer{peer_rank_};
        const Tag tag{kHelloTagValue};

        // sends posted before the snoop. Both ranks run this identical code, so a
        // blocking probe reached before anything is in flight deadlocks the pair on every run:
        // each side waits for a message the other has not posted. The old ordering -- snoop,
        // then irecv/isend -- never completed a two-rank connect. isend is nonblocking, so
        // posting it here costs nothing and gives the peer's probe something to match.
        auto tx = ctx->isend(
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&mine), sizeof(mine)), peer, tag);

        // sized before read. A peer built from a different commit can disagree about
        // sizeof(WireHello), and receiving that into this struct is a truncated read or an
        // overrun -- neither of which the magic check downstream would survive to report. The
        // snoop turns an ABI mismatch into a sentence naming both sizes.
        const std::size_t incoming = ctx->snoop_incoming_msg_size(peer, tag);
        if (incoming != sizeof(WireHello)) {
            // Our own hello is still in flight on this error path. Retiring it before returning
            // is what keeps an ABI mismatch a diagnosable message rather than an outstanding
            // request that surfaces later as a finalize-time complaint on an unrelated line.
            (void)tx->wait();
            return "peer's hello is " + std::to_string(incoming) + " B, ours is " +
                   std::to_string(sizeof(WireHello)) +
                   ". The two hosts are running builds that disagree about the wire struct.";
        }

        // irecv still goes up before either wait, so both directions are outstanding at once:
        // two blocking sends would deadlock the pair the moment the hello outgrows a runtime's
        // eager buffer.
        WireHello theirs{};
        auto rx = ctx->irecv(
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&theirs), sizeof(theirs)), peer, tag);
        (void)tx->wait();
        (void)rx->wait();

        if (theirs.magic != kHelloMagic) {
            return "peer sent a bad hello magic; the two sides are not the same protocol";
        }
        if (const std::string e = check_geometry(theirs); !e.empty()) {
            return e;
        }
        peer_.region_bytes = theirs.region_bytes;
        peer_.host_id = theirs.host_id;
        peer_.chips_per_host = theirs.chips_per_host;
        peer_.grid_width = theirs.grid_width;
        peer_.cores_in_use = theirs.cores_in_use;
        peer_.provisioned_cores = theirs.provisioned_cores;
        peer_.arena_stride = theirs.arena_stride;
        peer_.arena_bytes = theirs.arena_bytes;
        return {};
    }

    // A disagreement here does not produce a bad offset -- it produces a VALID offset naming a
    // different physical core, which nothing downstream can detect.
    std::string check_geometry(const WireHello& theirs) const {
        // arena_stride, arena_bytes, provisioned_cores and region_bytes are NOT re-checked here.
        // agree_on_geometry() settled them collectively before the window was created, which is
        // both earlier and stronger than a pairwise check; a second check that could disagree
        // with the first is how two verdicts about one fact end up in one program.
        if (cfg_.grid_width != 0 && theirs.grid_width != 0 && theirs.grid_width != cfg_.grid_width) {
            std::ostringstream m;
            m << "peer grid_width " << theirs.grid_width << " != ours " << cfg_.grid_width
              << ". The same core index would name different physical cores on the two hosts.";
            return m.str();
        }
        if (theirs.chips_per_host != cfg_.chips_per_host) {
            std::ostringstream m;
            m << "peer chips_per_host " << theirs.chips_per_host << " != ours " << cfg_.chips_per_host
              << ". UVA routing would send traffic to the wrong host.";
            return m.str();
        }
        return {};
    }

    MpiOp* acquire_op(uint64_t tag, std::string& err) {
        std::lock_guard<std::mutex> g(slots_m_);
        if (free_slots_.empty()) {
            unmatched_.fetch_add(1, std::memory_order_relaxed);
            err = "no free completion slot: " + std::to_string(kMaxOutstandingOps) +
                  " operations are already outstanding on this endpoint. MPI reports no queue "
                  "depth, so this table IS the flow-control limit.";
            return nullptr;
        }
        const uint32_t slot = free_slots_.back();
        free_slots_.pop_back();
        MpiOp* op = ops_[slot - 1].get();
        op->tag = tag;
        op->posted_ns = now_ns();
        op->bytes = 0;
        op->error.clear();
        op->request = MPI_REQUEST_NULL;
        op->state.store(kOpPosted, std::memory_order_release);
        return op;
    }

    void release_op(MpiOp* op) {
        op->state.store(kOpFree, std::memory_order_release);
        std::lock_guard<std::mutex> g(slots_m_);
        free_slots_.push_back(op->slot);
    }

    void retire(MpiOp& op, bool ok) {
        op.state.store(ok ? kOpDone : kOpFailed, std::memory_order_release);
        retired_.fetch_add(1, std::memory_order_relaxed);
        release_op(&op);
    }

    void record_retire(uint64_t ns, uint64_t bytes) {
        if (!cfg_.measure_retire || !recording_.load(std::memory_order_relaxed)) {
            return;
        }
        std::lock_guard<std::mutex> g(retire_m_);
        ++retire_n_;
        if (retire_n_ == 1 || ns < retire_min_) {
            retire_min_ = ns;
        }
        if (ns > retire_max_) {
            retire_max_ = ns;
        }
        const double d = static_cast<double>(ns) - retire_mean_;
        retire_mean_ += d / static_cast<double>(retire_n_);
        retire_m2_ += d * (static_cast<double>(ns) - retire_mean_);
        retire_bytes_ += bytes;
    }

    void set_last_error(std::string e) {
        std::lock_guard<std::mutex> g(err_m_);
        last_error_ = std::move(e);
    }
    std::string last_error() const {
        std::lock_guard<std::mutex> g(err_m_);
        return last_error_;
    }

    TransportConfig cfg_;
    uint8_t* region_ = nullptr;
    uint64_t region_bytes_ = 0;
    int peer_rank_ = -1;
    bool connected_ = false;
    PeerInfo peer_{};
    uint64_t notice_seq_ = 0;

    // One lock over the MPI calls. THREAD_MULTIPLE is required and provided, but concurrent RMA
    // on one window is the least-exercised corner of most runtimes, and the design already
    // funnels payloads through a single sender thread -- so the lock costs almost nothing and
    // removes a class of runtime bug from the picture.
    std::mutex mpi_m_;

    std::mutex slots_m_;
    std::vector<std::unique_ptr<MpiOp>> ops_;
    std::vector<uint32_t> free_slots_;

    std::atomic<uint64_t> posted_{0};
    std::atomic<uint64_t> retired_{0};
    std::atomic<uint64_t> unmatched_{0};
    std::atomic<uint64_t> abandoned_{0};
    std::atomic<uint64_t> injected_{0};
    std::atomic<bool> recording_{false};

    // Set by every one-sided write, cleared by a successful flush(). Written from the scan
    // workers (credits) and the sender thread (payloads), read by the sender thread.
    std::atomic<bool> dirty_{false};

    mutable std::mutex retire_m_;
    uint64_t retire_n_ = 0;
    uint64_t retire_min_ = 0;
    uint64_t retire_max_ = 0;
    double retire_mean_ = 0.0;
    double retire_m2_ = 0.0;
    uint64_t retire_bytes_ = 0;
    std::atomic<uint64_t> retire_unmeasured_{0};

    mutable std::mutex err_m_;
    std::string last_error_;
};

}  // namespace

std::unique_ptr<Transport> make_transport(const TransportConfig& cfg, std::string& error) {
    error.clear();
    return std::make_unique<MpiRmaTransport>(cfg);
}

bool transport_available() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    return initialized != 0;
}

// The window that carries the traffic was created collectively before any endpoint existed. What
// remains is one endpoint per peer rank and the geometry cross-check each one runs.
std::string connect_mesh(uint8_t* region_base, uint64_t region_bytes, const TransportConfig& base_cfg,
                         std::vector<std::unique_ptr<Transport>>& owned, PeerTable& table) {
    if (!DistributedContext::is_initialized()) {
        return "connect_mesh: the distributed context is not initialized";
    }
    const ContextPtr& ctx = DistributedContext::get_current_world();
    const uint32_t self = static_cast<uint32_t>(*ctx->rank());
    const uint32_t num_hosts = static_cast<uint32_t>(*ctx->size());
    table.configure(num_hosts, self);
    for (uint32_t j = 0; j < num_hosts; ++j) {
        if (j == self) {
            continue;
        }
        TransportConfig tc = base_cfg;
        tc.peer_rank = j;
        tc.host_id = self;
        std::string err;
        std::unique_ptr<Transport> t = make_transport(tc, err);
        if (!t) {
            return "transport for host " + std::to_string(j) + " unavailable: " + err;
        }
        if (const std::string e = t->connect(region_base, region_bytes); !e.empty()) {
            return "connect to host " + std::to_string(j) + " failed: " + e;
        }
        if (const std::string e = table.connect_peer(j, t.get()); !e.empty()) {
            return e;
        }
        owned.push_back(std::move(t));
    }
    return {};
}

}  // namespace tt::tt_metal::experimental
