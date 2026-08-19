// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emule_virtual_ranks.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/experimental/fabric/control_plane.hpp>

#if !defined(TT_METAL_USE_MPI)
#include "tt_metal/distributed/multihost/single_host_context.hpp"
#endif
#include "tt_metal/impl/context/metal_context.hpp"

namespace tt::tt_metal::emule {

using namespace tt::tt_metal::distributed::multihost;

namespace {

// Which virtual rank the calling thread speaks for. Per THREAD, because the whole point is that the
// K ranks are threads sharing one address space.
thread_local uint32_t t_virtual_rank = 0;

// Rendezvous + collective scratch for one set of ranks. create_sub_context is called once per
// PARTICIPATING RANK, each returning its own context object, so this state is looked up by member
// set rather than owned by the object — otherwise every rank would wait on a barrier of its own.
struct Gate {
    explicit Gate(size_t n) : members_n(n), scratch(n) {}

    // `faulted` is checked on every wake: an MPI rank that dies takes the job down with it, but a
    // dead THREAD is invisible, so without this its peers block here until the test timeout and the
    // real exception never surfaces.
    void wait(const std::atomic<bool>& faulted) {
        std::unique_lock<std::mutex> lk(mu);
        const uint64_t gen = generation;
        if (++arrived == members_n) {
            arrived = 0;
            ++generation;
            cv.notify_all();
        } else {
            cv.wait(lk, [&] { return generation != gen || faulted.load(std::memory_order_acquire); });
        }
        TT_FATAL(!faulted.load(std::memory_order_acquire), "virtual-rank barrier abandoned: a peer rank faulted");
    }

    const size_t members_n;
    std::mutex mu;
    std::condition_variable cv;
    size_t arrived = 0;
    uint64_t generation = 0;
    std::vector<std::vector<std::byte>> scratch;
    DistributedContextId id{0};
};

// One in-flight point-to-point message. `taken` is what makes ssend synchronous.
struct Message {
    std::vector<std::byte> data;
    std::shared_ptr<bool> taken = std::make_shared<bool>(false);
};

// The whole in-process world: p2p mailboxes plus every rank set that has been rendezvoused on.
struct World {
    // (context id, world src, world dst, tag). Scoped by context id so the same rank pair using the
    // same tag in a sub-context cannot collide with the parent's traffic.
    using MailKey = std::tuple<uint32_t, int, int, int>;

    std::atomic<bool> faulted{false};

    std::mutex mail_mu;
    std::condition_variable mail_cv;
    std::map<MailKey, std::deque<Message>> mail;

    std::mutex gates_mu;
    std::map<std::vector<int>, std::shared_ptr<Gate>> gates;

    // Release every rank parked in a rendezvous or a recv, so one thread's exception surfaces
    // instead of deadlocking the rest.
    void fault() {
        faulted.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(gates_mu);
            for (auto& [_members, gate] : gates) {
                std::lock_guard<std::mutex> glk(gate->mu);
                gate->cv.notify_all();
            }
        }
        std::lock_guard<std::mutex> lk(mail_mu);
        mail_cv.notify_all();
    }

    // A fresh world for each install: after a fault the gates hold partial arrival counts, and
    // reusing them would wedge the next run.
    void reset() {
        faulted.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(gates_mu);
            gates.clear();
        }
        std::lock_guard<std::mutex> lk(mail_mu);
        mail.clear();
    }

    // `fresh_id` is called under the lock: DistributedContext::generate_unique_id is documented as
    // not thread-safe, and all members of a rank set must agree on one id.
    std::shared_ptr<Gate> gate_for(
        const std::vector<int>& members, const std::function<DistributedContextId()>& fresh_id) {
        std::lock_guard<std::mutex> lk(gates_mu);
        auto& gate = gates[members];
        if (!gate) {
            gate = std::make_shared<Gate>(members.size());
            gate->id = fresh_id();
        }
        return gate;
    }
};

bool vrank_trace() {
    static const bool on = [] {
        const char* v = std::getenv("TT_EMULE_VRANK_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

[[maybe_unused]] World& world() {
    static World w;
    return w;
}

}  // namespace

// Derives from SingleHostContext because that is what the non-MPI
// DistributedContext::set_current_world accepts (it dynamic_casts). Only the pieces blaze actually
// exercises are implemented; the rest inherit SingleHostContext's throwing stubs, so an unsupported
// collective fails loudly instead of silently returning single-rank answers.
#if !defined(TT_METAL_USE_MPI)
class VirtualRankContext : public SingleHostContext {
public:
    // `members` are WORLD rank ids, in this context's rank order.
    explicit VirtualRankContext(std::vector<int> members) :
        members_(std::move(members)), gate_(world().gate_for(members_, [] { return generate_unique_id(); })) {
        id_ = gate_->id;
    }

    [[nodiscard]] Rank rank() const override { return Rank(static_cast<int>(my_index())); }
    [[nodiscard]] Size size() const override { return Size(static_cast<int>(members_.size())); }

    void barrier() const override { gate_->wait(world().faulted); }

    // Each rank contributes its slice; everyone leaves with the concatenation in rank order. Two
    // rendezvous: one so every contribution has landed before anyone reads, one so no rank races
    // ahead and overwrites the scratch while a straggler is still copying out.
    void all_gather(ttsl::Span<std::byte> send_buf, ttsl::Span<std::byte> recv_buf) const override {
        const size_t n = send_buf.size();
        const size_t k = members_.size();
        TT_FATAL(
            recv_buf.size() == n * k,
            "virtual-rank all_gather: recv buffer is {} bytes, expected {} ({} ranks x {})",
            recv_buf.size(),
            n * k,
            k,
            n);
        {
            std::lock_guard<std::mutex> lk(gate_->mu);
            gate_->scratch[my_index()].assign(send_buf.begin(), send_buf.end());
        }
        gate_->wait(world().faulted);
        for (size_t r = 0; r < k; ++r) {
            std::memcpy(recv_buf.data() + r * n, gate_->scratch[r].data(), n);
        }
        gate_->wait(world().faulted);
    }

    void broadcast(ttsl::Span<std::byte> buf, Rank root) const override {
        const auto root_idx = static_cast<size_t>(*root);
        if (my_index() == root_idx) {
            std::lock_guard<std::mutex> lk(gate_->mu);
            gate_->scratch[root_idx].assign(buf.begin(), buf.end());
        }
        gate_->wait(world().faulted);
        if (my_index() != root_idx) {
            std::memcpy(buf.data(), gate_->scratch[root_idx].data(), buf.size());
        }
        gate_->wait(world().faulted);
    }

    // --- point-to-point ---------------------------------------------------------------------
    // Buffered: the payload is copied into the mailbox so the sender can proceed, which is what
    // MPI_Send permits and what the MeshSocket descriptor handshake expects.
    void send(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const override { post(buf, dest, tag, /*sync=*/false); }

    // Synchronous: does not return until the peer has taken the message.
    void ssend(ttsl::Span<std::byte> buf, Rank dest, Tag tag) const override { post(buf, dest, tag, /*sync=*/true); }

    void recv(ttsl::Span<std::byte> buf, Rank source, Tag tag) const override {
        const auto key = mail_key(static_cast<size_t>(*source), my_index(), tag);
        auto& w = world();
        std::unique_lock<std::mutex> lk(w.mail_mu);
        if (vrank_trace()) {
            std::fprintf(stderr, "[VRANK] recv-wait ctx=%u %d->%d tag=%d want=%zu\n",
                         std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key), buf.size());
        }
        w.mail_cv.wait(lk, [&] {
            auto it = w.mail.find(key);
            return (it != w.mail.end() && !it->second.empty()) || w.faulted.load(std::memory_order_acquire);
        });
        TT_FATAL(!w.faulted.load(std::memory_order_acquire), "virtual-rank recv abandoned: a peer rank faulted");
        auto& queue = w.mail[key];
        Message msg = std::move(queue.front());
        queue.pop_front();
        TT_FATAL(
            msg.data.size() == buf.size(),
            "virtual-rank recv: message from rank {} tag {} is {} bytes, receiver expected {}",
            *source,
            *tag,
            msg.data.size(),
            buf.size());
        std::memcpy(buf.data(), msg.data.data(), msg.data.size());
        *msg.taken = true;
        w.mail_cv.notify_all();
    }

    std::size_t snoop_incoming_msg_size(Rank source, Tag tag) const override {
        const auto key = mail_key(static_cast<size_t>(*source), my_index(), tag);
        auto& w = world();
        std::unique_lock<std::mutex> lk(w.mail_mu);
        if (vrank_trace()) {
            std::fprintf(stderr, "[VRANK] snoop-wait ctx=%u %d->%d tag=%d\n",
                         std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key));
        }
        w.mail_cv.wait(lk, [&] {
            auto it = w.mail.find(key);
            return (it != w.mail.end() && !it->second.empty()) || w.faulted.load(std::memory_order_acquire);
        });
        TT_FATAL(!w.faulted.load(std::memory_order_acquire), "virtual-rank snoop abandoned: a peer rank faulted");
        return w.mail[key].front().data.size();
    }

    // --- communicator management ------------------------------------------------------------
    // `ranks` are ranks of THIS context; the sub-context stores them as world ranks so its p2p and
    // its parent's land in the same mailbox namespace. Only members call this (a rank with no socket
    // endpoint never reaches connect_with_peer), so there is no MPI_COMM_NULL case to model.
    [[nodiscard]] ContextPtr create_sub_context(ttsl::Span<int> ranks) const override {
        std::vector<int> members;
        members.reserve(ranks.size());
        for (int r : ranks) {
            TT_FATAL(
                r >= 0 && static_cast<size_t>(r) < members_.size(),
                "virtual-rank create_sub_context: rank {} is outside this context of size {}",
                r,
                members_.size());
            members.push_back(members_[static_cast<size_t>(r)]);
        }
        return std::make_shared<VirtualRankContext>(std::move(members));
    }

    void translate_ranks_to_other_ctx(
        ttsl::Span<int> ranks, const ContextPtr& other_ctx, ttsl::Span<int> translated_ranks) const override {
        const auto* other = dynamic_cast<const VirtualRankContext*>(other_ctx.get());
        TT_FATAL(other != nullptr, "virtual-rank translate_ranks_to_other_ctx: target is not a virtual-rank context");
        TT_FATAL(
            translated_ranks.size() == ranks.size(),
            "virtual-rank translate_ranks_to_other_ctx: {} ranks in, {} slots out",
            ranks.size(),
            translated_ranks.size());
        for (size_t i = 0; i < ranks.size(); ++i) {
            const int world_rank = members_.at(static_cast<size_t>(ranks[i]));
            const auto it = std::find(other->members_.begin(), other->members_.end(), world_rank);
            TT_FATAL(
                it != other->members_.end(),
                "virtual-rank translate_ranks_to_other_ctx: world rank {} is not in the target context",
                world_rank);
            translated_ranks[i] = static_cast<int>(std::distance(other->members_.begin(), it));
        }
    }

private:
    // This thread's rank within this context.
    size_t my_index() const {
        const auto it = std::find(members_.begin(), members_.end(), static_cast<int>(t_virtual_rank));
        TT_FATAL(
            it != members_.end(),
            "virtual rank {} called into a context it is not a member of — every thread that reaches a "
            "collective must be one of its ranks",
            t_virtual_rank);
        return static_cast<size_t>(std::distance(members_.begin(), it));
    }

    World::MailKey mail_key(size_t src_idx, size_t dst_idx, Tag tag) const {
        return {static_cast<uint32_t>(*id()), members_.at(src_idx), members_.at(dst_idx), *tag};
    }

    void post(ttsl::Span<std::byte> buf, Rank dest, Tag tag, bool sync) const {
        Message msg;
        msg.data.assign(buf.begin(), buf.end());
        auto taken = msg.taken;
        const auto key = mail_key(my_index(), static_cast<size_t>(*dest), tag);
        auto& w = world();
        std::unique_lock<std::mutex> lk(w.mail_mu);
        if (vrank_trace()) {
            std::fprintf(stderr, "[VRANK] send ctx=%u %d->%d tag=%d bytes=%zu sync=%d\n",
                         std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key), buf.size(), (int)sync);
        }
        w.mail[key].push_back(std::move(msg));
        w.mail_cv.notify_all();
        if (sync) {
            w.mail_cv.wait(lk, [&] { return *taken || w.faulted.load(std::memory_order_acquire); });
            TT_FATAL(*taken, "virtual-rank ssend abandoned: a peer rank faulted");
        }
    }

    std::vector<int> members_;
    std::shared_ptr<Gate> gate_;
};
#endif

// The ranks share one cluster and one control plane, so a rank-addressed MeshSocket has to be able
// to resolve each of them to a mesh; the control plane skips building any binding when the world it
// was constructed under had size 1. Devices are already open by the time ranks are installed.
void bind_ranks_in_control_plane(uint32_t k) {
    if (!MetalContext::instance_exists()) {
        return;  // no devices open yet; nothing addresses a mesh
    }
    MetalContext::instance().get_control_plane().bind_in_process_ranks(k);
}

void install_virtual_ranks(uint32_t k) {
    if (vrank_trace()) {
        std::fprintf(stderr, "[VRANK] install k=%u\n", k);
    }
#if defined(TT_METAL_USE_MPI)
    TT_FATAL(k <= 1, "in-process virtual ranks are unavailable in MPI builds");
#else
    world().reset();
    bind_ranks_in_control_plane(k);
    if (k <= 1) {
        DistributedContext::set_current_world(std::make_shared<SingleHostContext>());
        return;
    }
    std::vector<int> members(k);
    for (uint32_t r = 0; r < k; ++r) {
        members[r] = static_cast<int>(r);
    }
    DistributedContext::set_current_world(std::make_shared<VirtualRankContext>(std::move(members)));
#endif
}

void set_current_virtual_rank(uint32_t rank) {
#if defined(TT_METAL_USE_MPI)
    static_cast<void>(rank);
#else
    t_virtual_rank = rank;
#endif
}

uint32_t current_virtual_rank() {
#if defined(TT_METAL_USE_MPI)
    return 0;
#else
    return t_virtual_rank;
#endif
}

void fault_virtual_ranks() {
#if !defined(TT_METAL_USE_MPI)
    world().fault();
#endif
}

// Weak, so a rank's submesh is not pinned past the fixture that owns it.
static std::mutex& rank_mesh_mu() {
    static std::mutex mu;
    return mu;
}
static std::map<uint32_t, std::weak_ptr<distributed::MeshDevice>>& rank_meshes() {
    static std::map<uint32_t, std::weak_ptr<distributed::MeshDevice>> m;
    return m;
}

void register_virtual_rank_mesh(uint32_t rank, const std::shared_ptr<distributed::MeshDevice>& mesh) {
    if (vrank_trace()) {
        std::fprintf(stderr, "[VRANK] register mesh rank=%u mesh=%p\n", rank, static_cast<void*>(mesh.get()));
    }
    std::lock_guard<std::mutex> g(rank_mesh_mu());
    rank_meshes()[rank] = mesh;
}

std::shared_ptr<distributed::MeshDevice> virtual_rank_mesh(uint32_t rank) {
    std::shared_ptr<distributed::MeshDevice> mesh;
    {
        std::lock_guard<std::mutex> g(rank_mesh_mu());
        auto it = rank_meshes().find(rank);
        mesh = it == rank_meshes().end() ? nullptr : it->second.lock();
    }
    if (vrank_trace()) {
        std::fprintf(stderr, "[VRANK] lookup mesh rank=%u -> %p\n", rank, static_cast<void*>(mesh.get()));
    }
    return mesh;
}

uint32_t virtual_rank_count() {
#if defined(TT_METAL_USE_MPI)
    return 1;
#else
    if (!DistributedContext::is_initialized()) {
        return 1;
    }
    // Must answer "are the K ranks threads of THIS process", not "how big is the world" — under MPI
    // the world is also K, and a caller that confuses the two would partition per-process state that
    // MPI already partitions by process.
    auto world_ctx = std::dynamic_pointer_cast<const VirtualRankContext>(DistributedContext::get_current_world());
    return world_ctx ? static_cast<uint32_t>(*world_ctx->size()) : 1;
#endif
}

}  // namespace tt::tt_metal::emule
