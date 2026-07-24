// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "zmq_distributed_context.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string_view>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::distributed::multihost {

namespace {

// Wire framing. Sent as two frames on a DEALER -> ROUTER connection:
//   frame 0 (added by ROUTER): sender routing id (ignored; source is in the header)
//   frame 1: WireHeader (POD, memcpy'd)
//   frame 2: payload bytes
//
// NOTE (shortcut): fields are sent in host byte order. This assumes a
// homogeneous, little-endian (x86-64) cluster, which matches current
// deployments. A heterogeneous cluster would need explicit hton/ntoh.
struct WireHeader {
    std::uint32_t context_id;
    std::int32_t src_world_rank;
    std::int32_t tag;
    std::uint64_t nbytes;
};

// Parse "tcp://host:port" -> port string. Used to build the wildcard bind
// address ("tcp://*:port") from this rank's own endpoint.
std::string bind_address_from_endpoint(const std::string& endpoint) {
    const auto colon = endpoint.find_last_of(':');
    TT_FATAL(colon != std::string::npos, "ZMQ endpoint '{}' has no ':port' suffix", endpoint);
    return "tcp://*:" + endpoint.substr(colon + 1);
}

std::vector<std::string> parse_endpoints(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    for (std::string tok; std::getline(ss, tok, ',');) {
        if (!tok.empty()) {
            out.push_back(tok);
        }
    }
    return out;
}

}  // namespace

/* ============================= Exception ================================= */

ZmqDistributedException::ZmqDistributedException(Rank rank, int error_code, std::string msg) :
    rank_(rank), error_code_(error_code), message_(std::move(msg)), error_string_(zmq_strerror(error_code)) {
    message_ += ": " + error_string_;
}

Rank ZmqDistributedException::rank() const noexcept { return rank_; }
int ZmqDistributedException::error_code() const noexcept { return error_code_; }
const std::string& ZmqDistributedException::message() const noexcept { return message_; }
const std::string& ZmqDistributedException::error_string() const noexcept { return error_string_; }

/* ============================= Transport ================================= */

ZmqTransport::ZmqTransport(int world_rank, std::vector<std::string> endpoints) :
    world_rank_(world_rank), endpoints_(std::move(endpoints)), zmq_ctx_(/*io_threads=*/1) {
    const int n = static_cast<int>(endpoints_.size());
    TT_FATAL(n > 0, "ZmqTransport requires at least one endpoint");
    TT_FATAL(world_rank_ >= 0 && world_rank_ < n, "world_rank {} out of range [0,{})", world_rank_, n);

    // Receive socket: one ROUTER bound to our own endpoint's port.
    router_ = zmq::socket_t(zmq_ctx_, zmq::socket_type::router);
    router_.set(zmq::sockopt::linger, 0);
    router_.bind(bind_address_from_endpoint(endpoints_[world_rank_]));

    // Send sockets: one DEALER per peer, routing id set to our own rank so the
    // peer's ROUTER sees a stable identity (purely for debuggability).
    dealers_.resize(n);
    dealer_mtx_.resize(n);
    const std::string my_id = std::to_string(world_rank_);
    for (int r = 0; r < n; ++r) {
        dealer_mtx_[r] = std::make_unique<std::mutex>();
        if (r == world_rank_) {
            continue;  // self is handled via inbox loopback
        }
        zmq::socket_t d(zmq_ctx_, zmq::socket_type::dealer);
        d.set(zmq::sockopt::routing_id, my_id);
        d.set(zmq::sockopt::linger, 0);
        // High water mark generous enough that isend/buffered send does not block
        // for the small control messages this layer carries.
        d.set(zmq::sockopt::sndhwm, 0);  // 0 == unlimited
        d.connect(endpoints_[r]);
        dealers_[r] = std::move(d);
    }

    running_ = true;
    progress_ = std::thread([this] { progress_loop(); });
}

ZmqTransport::~ZmqTransport() {
    running_ = false;
    if (progress_.joinable()) {
        progress_.join();
    }
}

void ZmqTransport::progress_loop() {
    std::vector<zmq::pollitem_t> items{{router_.handle(), 0, ZMQ_POLLIN, 0}};
    while (running_.load(std::memory_order_relaxed)) {
        // Bounded poll so shutdown is observed within the timeout even with no traffic.
        zmq::poll(items, std::chrono::milliseconds(50));
        if ((items[0].revents & ZMQ_POLLIN) == 0) {
            continue;
        }
        // Drain everything currently queued.
        while (true) {
            zmq::message_t id_frame;
            auto got = router_.recv(id_frame, zmq::recv_flags::dontwait);
            if (!got) {
                break;  // nothing left
            }
            zmq::message_t hdr_frame;
            zmq::message_t payload;
            // Header and payload are guaranteed to follow the routing-id frame.
            (void)router_.recv(hdr_frame, zmq::recv_flags::none);
            (void)router_.recv(payload, zmq::recv_flags::none);

            TT_FATAL(
                hdr_frame.size() == sizeof(WireHeader), "ZMQ: malformed header frame ({} bytes)", hdr_frame.size());
            WireHeader h{};
            std::memcpy(&h, hdr_frame.data(), sizeof(h));
            TT_FATAL(payload.size() == h.nbytes, "ZMQ: payload {} bytes, header claims {}", payload.size(), h.nbytes);

            std::vector<std::byte> data(h.nbytes);
            if (h.nbytes != 0) {
                std::memcpy(data.data(), payload.data(), h.nbytes);
            }
            const Key key{h.context_id, h.src_world_rank, h.tag};
            {
                std::lock_guard<std::mutex> lk(inbox_mtx_);
                inbox_[key].push_back(std::move(data));
            }
            inbox_cv_.notify_all();
        }
    }
}

void ZmqTransport::post(int dst_world_rank, std::uint32_t context_id, int tag, ttsl::Span<std::byte> bytes) {
    WireHeader h{context_id, world_rank_, tag, static_cast<std::uint64_t>(bytes.size())};

    if (dst_world_rank == world_rank_) {
        // Loopback: deliver straight to our own inbox.
        std::vector<std::byte> data(bytes.begin(), bytes.end());
        const Key key{context_id, world_rank_, tag};
        {
            std::lock_guard<std::mutex> lk(inbox_mtx_);
            inbox_[key].push_back(std::move(data));
        }
        inbox_cv_.notify_all();
        return;
    }

    TT_FATAL(
        dst_world_rank >= 0 && dst_world_rank < world_size(),
        "ZMQ post: dst world rank {} out of range",
        dst_world_rank);

    std::lock_guard<std::mutex> lk(*dealer_mtx_[dst_world_rank]);
    dealers_[dst_world_rank].send(zmq::buffer(&h, sizeof(h)), zmq::send_flags::sndmore);
    dealers_[dst_world_rank].send(zmq::buffer(bytes.data(), bytes.size()), zmq::send_flags::none);
}

bool ZmqTransport::find_locked(const Key& key) const {
    const auto it = inbox_.find(key);
    return it != inbox_.end() && !it->second.empty();
}

Status ZmqTransport::recv_into(std::uint32_t context_id, int src_world_rank, int tag, ttsl::Span<std::byte> out) {
    const Key key{context_id, src_world_rank, tag};
    std::vector<std::byte> data;
    {
        std::unique_lock<std::mutex> lk(inbox_mtx_);
        inbox_cv_.wait(lk, [&] { return find_locked(key); });
        auto& dq = inbox_[key];
        data = std::move(dq.front());
        dq.pop_front();
        if (dq.empty()) {
            inbox_.erase(key);
        }
    }
    TT_FATAL(data.size() <= out.size(), "ZMQ recv: message {} bytes > buffer {} bytes", data.size(), out.size());
    if (!data.empty()) {
        std::memcpy(out.data(), data.data(), data.size());
    }
    return Status{Rank(src_world_rank), Tag(tag), static_cast<int>(data.size())};
}

std::optional<Status> ZmqTransport::try_recv_into(
    std::uint32_t context_id, int src_world_rank, int tag, ttsl::Span<std::byte> out) {
    const Key key{context_id, src_world_rank, tag};
    std::vector<std::byte> data;
    {
        std::lock_guard<std::mutex> lk(inbox_mtx_);
        if (!find_locked(key)) {
            return std::nullopt;
        }
        auto& dq = inbox_[key];
        data = std::move(dq.front());
        dq.pop_front();
        if (dq.empty()) {
            inbox_.erase(key);
        }
    }
    TT_FATAL(data.size() <= out.size(), "ZMQ recv: message {} bytes > buffer {} bytes", data.size(), out.size());
    if (!data.empty()) {
        std::memcpy(out.data(), data.data(), data.size());
    }
    return Status{Rank(src_world_rank), Tag(tag), static_cast<int>(data.size())};
}

std::size_t ZmqTransport::probe(std::uint32_t context_id, int src_world_rank, int tag) {
    const Key key{context_id, src_world_rank, tag};
    std::unique_lock<std::mutex> lk(inbox_mtx_);
    inbox_cv_.wait(lk, [&] { return find_locked(key); });
    return inbox_[key].front().size();
}

std::uint32_t ZmqTransport::reserve_context_ids(int n) {
    const std::uint32_t base = next_context_id_;
    next_context_id_ += static_cast<std::uint32_t>(n);
    return base;
}

/* ============================== Request ================================== */

ZmqRequest::ZmqRequest(Status done_status) : status_(done_status) {}

ZmqRequest::ZmqRequest(
    std::shared_ptr<ZmqTransport> transport,
    std::uint32_t context_id,
    int src_world_rank,
    int tag,
    ttsl::Span<std::byte> out) :
    transport_(std::move(transport)), context_id_(context_id), src_world_rank_(src_world_rank), tag_(tag), out_(out) {}

Status ZmqRequest::wait() {
    if (status_.has_value()) {
        return *status_;
    }
    TT_FATAL(!cancelled_, "ZmqRequest::wait called on a cancelled request");
    status_ = transport_->recv_into(context_id_, src_world_rank_, tag_, out_);
    return *status_;
}

std::optional<Status> ZmqRequest::test() {
    if (status_.has_value()) {
        return status_;
    }
    if (cancelled_) {
        return std::nullopt;
    }
    status_ = transport_->try_recv_into(context_id_, src_world_rank_, tag_, out_);
    return status_;
}

void ZmqRequest::cancel() {
    // Only a not-yet-matched receive can be cancelled; a completed/buffered send cannot.
    if (!status_.has_value()) {
        cancelled_ = true;
    }
}

bool ZmqRequest::active() const { return !status_.has_value() && !cancelled_; }

/* ============================== Context ================================== */

ZmqContext::ZmqContext(
    std::shared_ptr<ZmqTransport> transport, std::uint32_t context_id, std::vector<int> world_ranks, int local_rank) :
    transport_(std::move(transport)),
    context_id_(context_id),
    world_ranks_(std::move(world_ranks)),
    local_rank_(local_rank) {
    id_ = DistributedContext::generate_unique_id();
}

void ZmqContext::create(int /*argc*/, char** /*argv*/) {
    if (current_world_) {
        return;
    }
    // Initial state is supplied entirely by env vars (no launcher dependency):
    //   TT_ZMQ_RANK      -> this process's world rank
    //   TT_ZMQ_ENDPOINTS -> comma-separated "tcp://host:port" list, indexed by rank
    const char* rank_env = std::getenv("TT_ZMQ_RANK");
    const char* eps_env = std::getenv("TT_ZMQ_ENDPOINTS");
    TT_FATAL(rank_env != nullptr && rank_env[0] != '\0', "ZMQ backend selected but TT_ZMQ_RANK is unset");
    TT_FATAL(eps_env != nullptr && eps_env[0] != '\0', "ZMQ backend selected but TT_ZMQ_ENDPOINTS is unset");

    const int rank = std::stoi(rank_env);
    std::vector<std::string> endpoints = parse_endpoints(eps_env);
    TT_FATAL(!endpoints.empty(), "TT_ZMQ_ENDPOINTS parsed to an empty list");

    auto transport = std::make_shared<ZmqTransport>(rank, endpoints);

    std::vector<int> world_ranks(endpoints.size());
    std::iota(world_ranks.begin(), world_ranks.end(), 0);
    current_world_ = std::make_shared<ZmqContext>(std::move(transport), /*context_id=*/0, std::move(world_ranks), rank);
}

const ContextPtr& ZmqContext::get_current_world() {
    if (!current_world_) {
        create(0, nullptr);
    }
    return current_world_;
}

// No launcher-driven split in this draft, so the "full world" is the world context.
ContextPtr ZmqContext::get_world_context() { return get_current_world(); }

void ZmqContext::set_current_world(const ContextPtr& ctx) {
    TT_FATAL(
        ctx != nullptr && std::dynamic_pointer_cast<ZmqContext>(ctx) != nullptr,
        "ZmqContext::set_current_world: context is not a ZmqContext");
    current_world_ = ctx;
}

bool ZmqContext::is_initialized() { return current_world_ != nullptr; }

int ZmqContext::to_world(Rank local) const {
    const int r = *local;
    TT_FATAL(r >= 0 && r < static_cast<int>(world_ranks_.size()), "local rank {} out of range", r);
    return world_ranks_[r];
}

int ZmqContext::next_coll_tag() const { return -(1 + coll_seq_++); }

/* ---------------- basic info / sync ---------------- */

Rank ZmqContext::rank() const { return Rank(local_rank_); }
Size ZmqContext::size() const { return Size(static_cast<int>(world_ranks_.size())); }
bool ZmqContext::supports_fault_tolerance() const { return false; }
bool ZmqContext::is_revoked() { return false; }

void ZmqContext::barrier() const {
    const int n = static_cast<int>(world_ranks_.size());
    if (n <= 1) {
        return;
    }
    const int t = next_coll_tag();
    std::byte token{};
    ttsl::Span<std::byte> tok(&token, 1);
    if (local_rank_ == 0) {
        for (int r = 1; r < n; ++r) {
            transport_->recv_into(context_id_, world_ranks_[r], t, tok);
        }
        for (int r = 1; r < n; ++r) {
            transport_->post(world_ranks_[r], context_id_, t, tok);
        }
    } else {
        transport_->post(world_ranks_[0], context_id_, t, tok);
        transport_->recv_into(context_id_, world_ranks_[0], t, tok);
    }
}

/* ---------------- point-to-point ------------------- */

void ZmqContext::send(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const {
    TT_FATAL(*tag >= 0, "ZMQ user tags must be non-negative (negative tags are reserved), got {}", *tag);
    transport_->post(to_world(dest), context_id_, *tag, buf);
}

// SHORTCUT: ssend is implemented as a buffered send. ZMQ guarantees reliable,
// in-order delivery, so correctness (data + ordering) holds; what is *not*
// provided is the MPI "sender blocks until the receiver has matched" rendezvous
// guarantee. No production caller relies on that guarantee for correctness
// (only ordering), and buffered sends cannot deadlock. If strict synchronous
// semantics are later required, add an ack round-trip on a reserved tag.
void ZmqContext::ssend(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const { send(buf, dest, tag); }

void ZmqContext::recv(ttsl::Span<std::byte> buf, Rank source, Tag tag) const {
    TT_FATAL(*tag >= 0, "ZMQ user tags must be non-negative, got {}", *tag);
    transport_->recv_into(context_id_, to_world(source), *tag, buf);
}

RequestPtr ZmqContext::isend(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const {
    TT_FATAL(*tag >= 0, "ZMQ user tags must be non-negative, got {}", *tag);
    // ZMQ copies the payload into its own buffers on send, so the operation is
    // already complete from the caller's perspective.
    transport_->post(to_world(dest), context_id_, *tag, buf);
    return std::make_shared<ZmqRequest>(Status{dest, tag, static_cast<int>(buf.size())});
}

RequestPtr ZmqContext::irecv(ttsl::Span<std::byte> buf, Rank source, Tag tag) const {
    TT_FATAL(*tag >= 0, "ZMQ user tags must be non-negative, got {}", *tag);
    // SHORTCUT: matching is deferred to wait()/test(). If multiple irecvs are
    // posted for the same (source, tag), they are satisfied in wait()-call order
    // rather than post order. This matches all observed production usage
    // (topology_mapper posts then immediately waits), but differs from MPI's
    // strict post-order matching.
    return std::make_shared<ZmqRequest>(transport_, context_id_, to_world(source), *tag, buf);
}

/* ---------------- collectives ---------------------- */

void ZmqContext::broadcast(ttsl::Span<std::byte> buf, Rank root) const {
    const int n = static_cast<int>(world_ranks_.size());
    if (n <= 1) {
        return;
    }
    const int t = next_coll_tag();
    const int root_local = *root;
    if (local_rank_ == root_local) {
        for (int r = 0; r < n; ++r) {
            if (r != root_local) {
                transport_->post(world_ranks_[r], context_id_, t, buf);
            }
        }
    } else {
        transport_->recv_into(context_id_, world_ranks_[root_local], t, buf);
    }
}

void ZmqContext::all_gather(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf) const {
    const int n = static_cast<int>(world_ranks_.size());
    const std::size_t chunk = send_buf.size();
    TT_FATAL(
        recv_buf.size() == chunk * static_cast<std::size_t>(n),
        "all_gather: recv buffer {} bytes, expected {} (world {} x send {})",
        recv_buf.size(),
        chunk * n,
        n,
        chunk);

    // Place our own contribution.
    if (chunk != 0) {
        std::memcpy(recv_buf.data() + static_cast<std::size_t>(local_rank_) * chunk, send_buf.data(), chunk);
    }
    if (n <= 1) {
        return;
    }

    const int t = next_coll_tag();
    // Post all sends first (buffered, non-blocking), then receive from each peer.
    for (int r = 0; r < n; ++r) {
        if (r != local_rank_) {
            transport_->post(world_ranks_[r], context_id_, t, send_buf);
        }
    }
    for (int r = 0; r < n; ++r) {
        if (r != local_rank_) {
            ttsl::Span<std::byte> slot(recv_buf.data() + static_cast<std::size_t>(r) * chunk, chunk);
            transport_->recv_into(context_id_, world_ranks_[r], t, slot);
        }
    }
}

// ---- COLD collectives ----
// These have no production call sites (see usage analysis); they are exercised
// only by tests/tt_metal/multihost/single_host_mp_tests/test_context.cpp.
// all_reduce / reduce / reduce_scatter / scan can be layered on all_gather + a
// typed (DType x ReduceOp) local reduction; gather / scatter / all_to_all are
// direct p2p patterns. Left unimplemented to keep this draft focused.
void ZmqContext::all_reduce(ttsl::Span<std::byte>, ttsl::Span<std::byte>, ReduceOp, DType) const {
    TT_THROW("ZmqContext::all_reduce not yet implemented (COLD path).");
}
void ZmqContext::reduce(ttsl::Span<std::byte>, ttsl::Span<std::byte>, ReduceOp, DType, Rank) const {
    TT_THROW("ZmqContext::reduce not yet implemented (COLD path).");
}
void ZmqContext::gather(ttsl::Span<std::byte>, ttsl::Span<std::byte>, Rank) const {
    TT_THROW("ZmqContext::gather not yet implemented (COLD path).");
}
void ZmqContext::scatter(ttsl::Span<std::byte>, ttsl::Span<std::byte>, Rank) const {
    TT_THROW("ZmqContext::scatter not yet implemented (COLD path).");
}
void ZmqContext::all_to_all(ttsl::Span<std::byte>, ttsl::Span<std::byte>) const {
    TT_THROW("ZmqContext::all_to_all not yet implemented (COLD path).");
}
void ZmqContext::reduce_scatter(ttsl::Span<std::byte>, ttsl::Span<std::byte>, ReduceOp, DType) const {
    TT_THROW("ZmqContext::reduce_scatter not yet implemented (COLD path).");
}
void ZmqContext::scan(ttsl::Span<std::byte>, ttsl::Span<std::byte>, ReduceOp, DType) const {
    TT_THROW("ZmqContext::scan not yet implemented (COLD path).");
}

void ZmqContext::translate_ranks_to_other_ctx(
    ttsl::Span<int> ranks, const ContextPtr& other_ctx, ttsl::Span<int> translated_ranks) const {
    const auto* other = dynamic_cast<const ZmqContext*>(other_ctx.get());
    TT_FATAL(other != nullptr, "translate_ranks_to_other_ctx: other context is not a ZmqContext");
    TT_FATAL(ranks.size() == translated_ranks.size(), "translate_ranks: size mismatch");

    // world rank -> local rank in the other context
    std::map<int, int> world_to_other_local;
    for (int i = 0; i < static_cast<int>(other->world_ranks_.size()); ++i) {
        world_to_other_local[other->world_ranks_[i]] = i;
    }
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        const int world = to_world(Rank(ranks[i]));
        const auto it = world_to_other_local.find(world);
        translated_ranks[i] = (it == world_to_other_local.end()) ? RANK_UNDEFINED : it->second;
    }
}

/* ------------- communicator management ------------- */

ContextPtr ZmqContext::duplicate() const {
    // Collective: every rank advances the id counter identically.
    const std::uint32_t new_id = transport_->reserve_context_ids(1);
    return std::make_shared<ZmqContext>(transport_, new_id, world_ranks_, local_rank_);
}

ContextPtr ZmqContext::split(Color color, Key key) const {
    // Collective over the whole context. Exchange (color, key, world_rank) via
    // all_gather, then group by color, order by key (ties broken by world rank),
    // and assign new local ranks.
    struct Rec {
        std::int32_t color;
        std::int32_t key;
        std::int32_t world_rank;
    };
    const int n = static_cast<int>(world_ranks_.size());
    Rec mine{*color, *key, world_ranks_[local_rank_]};
    std::vector<Rec> all(n);
    all_gather(
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&mine), sizeof(Rec)),
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(all.data()), sizeof(Rec) * n));

    // Distinct, valid colors in ascending order -> deterministic id assignment.
    std::vector<int> colors;
    for (const auto& rec : all) {
        if (rec.color != SPLIT_COLOR_UNDEFINED) {
            colors.push_back(rec.color);
        }
    }
    std::sort(colors.begin(), colors.end());
    colors.erase(std::unique(colors.begin(), colors.end()), colors.end());

    // Reserve one id per distinct color (identical on every rank).
    const std::uint32_t base = transport_->reserve_context_ids(static_cast<int>(colors.size()));

    if (*color == SPLIT_COLOR_UNDEFINED) {
        return nullptr;  // excluded from all sub-contexts
    }

    std::vector<Rec> members;
    for (const auto& rec : all) {
        if (rec.color == *color) {
            members.push_back(rec);
        }
    }
    std::sort(members.begin(), members.end(), [](const Rec& a, const Rec& b) {
        return std::tie(a.key, a.world_rank) < std::tie(b.key, b.world_rank);
    });

    std::vector<int> member_world_ranks;
    int my_local = -1;
    for (int i = 0; i < static_cast<int>(members.size()); ++i) {
        member_world_ranks.push_back(members[i].world_rank);
        if (members[i].world_rank == world_ranks_[local_rank_]) {
            my_local = i;
        }
    }
    TT_FATAL(my_local >= 0, "split: this rank not found in its own color group");

    const auto color_idx = std::distance(colors.begin(), std::find(colors.begin(), colors.end(), *color));
    const std::uint32_t new_id = base + static_cast<std::uint32_t>(color_idx);
    return std::make_shared<ZmqContext>(transport_, new_id, std::move(member_world_ranks), my_local);
}

ContextPtr ZmqContext::create_sub_context(ttsl::Span<int> ranks) const {
    // Collective over the parent context: every rank must call with the SAME
    // `ranks` list (local ranks of this context). All ranks advance the id
    // counter by one identically; members build the context, non-members return
    // nullptr.
    const std::uint32_t new_id = transport_->reserve_context_ids(1);

    std::vector<int> member_world_ranks;
    member_world_ranks.reserve(ranks.size());
    int my_local = -1;
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        member_world_ranks.push_back(to_world(Rank(ranks[i])));
        if (ranks[i] == local_rank_) {
            my_local = static_cast<int>(i);
        }
    }
    if (my_local < 0) {
        return nullptr;  // not a member of this sub-context
    }
    return std::make_shared<ZmqContext>(transport_, new_id, std::move(member_world_ranks), my_local);
}

/* ------------- error handling / fault tolerance ---- */

// SHORTCUT: unlike MPI_Abort, this only terminates the local process. Peers are
// not signalled; the launcher / k8s pod lifecycle is responsible for tearing
// down the rest of the job.
void ZmqContext::abort(int error_code) const { std::exit(error_code); }

void ZmqContext::revoke_and_shrink() {
    TT_THROW("ZmqContext::revoke_and_shrink unsupported (no ULFM-equivalent fault tolerance).");
}

/* ------------- message snooping -------------------- */

std::size_t ZmqContext::snoop_incoming_msg_size(Rank source, Tag tag) const {
    TT_FATAL(*tag >= 0, "ZMQ user tags must be non-negative, got {}", *tag);
    return transport_->probe(context_id_, to_world(source), *tag);
}

}  // namespace tt::tt_metal::distributed::multihost
