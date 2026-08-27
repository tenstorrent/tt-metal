// SPDX-License-Identifier: Apache-2.0
//
// Definitions for the core API declared in tt/unified/api.h, targeting the
// Metal v1 programming model (Wormhole / Blackhole: 2 DM threads + a compute
// thread that metal splits three ways).
//
// Include <tt/unified/core>, not this header directly.

#pragma once

#include <new>

#include <tt/unified/api.h>

namespace tt {
namespace unified {

// --- Geometry ---
//
// Every body here names a data-movement-only symbol, so all of them are guarded.
// On a compute projection the coordinate types still exist and can be carried
// through a shared statement -- they just cannot be resolved to a NOC address.

inline PhysicalCoord PhysicalCoord::this_core() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    return PhysicalCoord::yx(my_y[noc_index], my_x[noc_index]);
#else
    return PhysicalCoord::yx(0, 0);
#endif
}

inline PhysicalCoord PhysicalCoord::origin() { return LogicalCoord::origin().to_physical(); }

inline uint64_t PhysicalCoord::get_noc_addr(uintptr_t l1_addr) const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    // Qualified: this class has a member of the same name.
    return ::get_noc_addr(x, y, static_cast<uint32_t>(l1_addr));
#else
    (void)l1_addr;
    return 0;
#endif
}

// Unguarded, unlike the rest of this section: a LOGICAL coordinate is one thing
// compute genuinely knows. get_relative_logical_* is declared for compute too
// (api/compute/common.h), and trisc.cc defines my_relative_x_/y_ and fills them
// from the launch message before calling the kernel -- so all five projections
// agree on where they are. Only the VIRTUAL mapping below is data-movement-only.
inline LogicalCoord LogicalCoord::this_core() {
    return LogicalCoord::yx(get_relative_logical_y(), get_relative_logical_x());
}

inline LogicalCoord LogicalCoord::origin() { return LogicalCoord::yx(0, 0); }

inline PhysicalCoord LogicalCoord::to_physical(uint32_t y_offset, uint32_t x_offset) const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    return PhysicalCoord::yx(
        worker_logical_row_to_virtual_row[y + y_offset], worker_logical_col_to_virtual_col[x + x_offset]);
#else
    (void)y_offset;
    (void)x_offset;
    return PhysicalCoord::yx(0, 0);
#endif
}

inline uint64_t LogicalCoord::get_noc_addr(uintptr_t l1_addr) const { return to_physical().get_noc_addr(l1_addr); }

// A multicast rectangle is carried in ascending virtual coordinates, which is
// what NOC 0 wants. NOC 1 runs the grid the other way round and wants the corners
// in ITS traversal order -- high corner first -- so they swap. Nothing downstream
// does this for us: get_noc_multicast_addr() maps each coordinate through
// DYNAMIC_NOC_X/Y, which is NOC_0_X/Y, the identity here (the mirroring variant
// is the separate NOC_0_X_PHYS_COORD). Handing NOC 1 an ascending rectangle
// silently drops part of the destination set, stranding whoever was dropped on
// the handshake.
//
// Spelled out at each call site rather than hidden behind a helper, because the
// same rectangle is ALSO the source of the sizes -- volume() and
// num_dests_excluding_sender() -- which must keep reading the ascending corners
// or they underflow. noc_index is constexpr in a kernel build, so the branch
// folds away.
inline uint64_t PhysicalMcast::get_noc_addr(uintptr_t l1_addr) const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if (noc_index == 1) {
        return ::get_noc_multicast_addr(end.x, end.y, start.x, start.y, static_cast<uint32_t>(l1_addr));
    }
    return ::get_noc_multicast_addr(start.x, start.y, end.x, end.y, static_cast<uint32_t>(l1_addr));
#else
    (void)l1_addr;
    return 0;
#endif
}

inline PhysicalMcast LogicalMcast::to_physical() const {
    return PhysicalMcast{coord.to_physical(), coord.to_physical(extent.h - 1, extent.w - 1)};
}

inline uint64_t LogicalMcast::get_noc_addr(uintptr_t l1_addr) const { return to_physical().get_noc_addr(l1_addr); }

// --- Semaphore ---

template <int thread>
Semaphore<thread>::Semaphore(uint32_t semaphore_id) :
    id(semaphore_id)
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    ,
    sem(semaphore_id)
#endif
{
}

template <int thread>
uintptr_t Semaphore<thread>::l1_addr() const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    // Recomputed from the id rather than read out of the wrapped semaphore:
    // metal keeps its own l1 address private. Same arithmetic it uses.
    return get_semaphore<ProgrammableCoreType::TENSIX>(id);
#else
    return 0;
#endif
}

template <int thread>
Semaphore<thread>& Semaphore<thread>::wait(uint32_t value) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        sem.wait(value);
    }
#endif
    (void)value;
    return *this;
}

template <int thread>
Semaphore<thread>& Semaphore<thread>::wait_min(uint32_t value) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        sem.wait_min(value);
    }
#endif
    (void)value;
    return *this;
}

template <int thread>
Semaphore<thread>& Semaphore<thread>::set(uint32_t value) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        sem.set(value);
    }
#endif
    (void)value;
    return *this;
}

template <int thread>
Semaphore<thread>& Semaphore<thread>::inc_remote(PhysicalCoord coord, uint32_t value) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        sem.up(Noc{}, coord.x, coord.y, value);
    }
#endif
    (void)coord;
    (void)value;
    return *this;
}

template <int thread>
Semaphore<thread>& Semaphore<thread>::inc_remote(LogicalCoord coord, uint32_t value) {
    return inc_remote(coord.to_physical(), value);
}

template <int thread>
Semaphore<thread>& Semaphore<thread>::inc_mcast(PhysicalMcast mcast, uint32_t value) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        // Sender-aware count, with no assertion that the issuer is the start
        // corner: multicast noc_core_write raises its arrival flag from OUTSIDE
        // the rectangle, where every core in range is a destination. set_mcast
        // below keeps the corner rule, because only the handshake paths use it.
        //
        // Corners in NOC order, sizes from the ascending rectangle. See
        // PhysicalMcast::get_noc_addr -- this path takes raw coordinates, so it
        // needs the same swap.
        const uint32_t dests = mcast.num_dests_excluding(PhysicalCoord::this_core());
        if (noc_index == 1) {
            sem.inc_multicast(Noc{}, mcast.end.x, mcast.end.y, mcast.start.x, mcast.start.y, value, dests);
        } else {
            sem.inc_multicast(Noc{}, mcast.start.x, mcast.start.y, mcast.end.x, mcast.end.y, value, dests);
        }
    }
#endif
    (void)mcast;
    (void)value;
    return *this;
}

template <int thread>
Semaphore<thread>& Semaphore<thread>::inc_mcast(LogicalMcast mcast, uint32_t value) {
    return inc_mcast(mcast.to_physical(), value);
}

template <int thread>
Semaphore<thread>& Semaphore<thread>::set_mcast(PhysicalMcast mcast) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
        ASSERT(PhysicalCoord::this_core() == mcast.start);  // as in inc_mcast above
#endif
        // Corners in NOC order, sizes from the ascending rectangle -- as in
        // inc_mcast above.
        const uint32_t dests = mcast.num_dests_excluding_sender();
        if (noc_index == 1) {
            sem.set_multicast(Noc{}, mcast.end.x, mcast.end.y, mcast.start.x, mcast.start.y, dests);
        } else {
            sem.set_multicast(Noc{}, mcast.start.x, mcast.start.y, mcast.end.x, mcast.end.y, dests);
        }
    }
#endif
    (void)mcast;
    return *this;
}

template <int thread>
Semaphore<thread>& Semaphore<thread>::set_mcast(LogicalMcast mcast) {
    return set_mcast(mcast.to_physical());
}

// --- Storage ---

// The one place a PAGE count is handed to something that reads it as a TILE count:
// the strategies index tiles (copy_tile, pack_tile, reduce_tile) once per page.
// True only while a page holds exactly one tile, which is how the harness builds
// every CB -- cb_page_bytes() is the tile size. A CB configured otherwise would
// need the two counts separated here.
template <typename S>
template <typename Node>
Block<S> Storage<S>::store(const Node& node) {
    // The destination must be exactly the shape the expression produces. This is the
    // check that replaces every hand-derived page count: a reduction's output, a
    // matmul's output block, a gather's stacked extent.
    static_assert(
        same_shape_v<node_shape_t<Node>, S>,
        "this Storage's shape is not the shape the expression produces -- compare the Storage<...> "
        "argument against the operands' shapes and the axis or geometry driving the op");
    Strategy<expr::kind_of_t<Node>>::run(node, cb_id, num_pages);
    return Block<S>(cb_id);
}

// --- Block ---

template <typename S>
Block<S>::Block(const Storage<S>& storage) : cb_id(storage.cb_id) {}

template <typename S>
Block<S>::Block(uint32_t cb_id) : cb_id(cb_id) {}

template <typename S>
Block<S>::Block(const Storage<S>& storage, Retained) : cb_id(storage.cb_id) {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    must_consume = false;
#endif
}

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
template <typename S>
Block<S>::~Block() {
    ASSERT(!must_consume || consumed);
}
#endif

// A move transfers the OBLIGATION rather than discharging it: the destination
// inherits must_consume/consumed unchanged, and only the source goes silent.
// Clearing the flag on the destination instead would let `Block b2 =
// std::move(b1);` launder the debt -- b2 could then be dropped without a pop
// and nothing would complain.
template <typename S>
Block<S>::Block(Block&& o) : cb_id(o.cb_id) {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(o.must_consume);  // a retained block belongs to its accumulator
    must_consume = o.must_consume;
    consumed = o.consumed;
    o.must_consume = false;
    o.consumed = true;
    o.cb_id = kMovedFrom;
#endif
}

template <typename S>
Block<S>& Block<S>::operator=(Block&& o) {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(o.must_consume);             // a retained block belongs to its accumulator
    ASSERT(!must_consume || consumed);  // do not drop pages this Block still owes
#endif
    cb_id = o.cb_id;
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    must_consume = o.must_consume;
    consumed = o.consumed;
    o.must_consume = false;
    o.consumed = true;
    o.cb_id = kMovedFrom;
#endif
    return *this;
}

template <typename S>
void Block<S>::consume() {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    // Catches a retained block reaching a consumer directly -- passing
    // accumulate(node, /*finish=*/false) straight into noc_store elides the
    // move, so the check in the move constructor never runs.
    ASSERT(must_consume);
    ASSERT(!consumed);
    consumed = true;
#endif
}

// --- RetainedBlock ---

template <typename S>
RetainedBlock<S>::RetainedBlock(Held&& block) {
    emplace(std::move(block));
}

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
template <typename S>
RetainedBlock<S>::~RetainedBlock() {
    // A block still held here was pushed and never waited on: a dropped output. Nothing to
    // destroy afterwards -- Block's own destructor only asserts, and this has already caught
    // the case.
    ASSERT(!held);
}
#endif

template <typename S>
RetainedBlock<S>& RetainedBlock<S>::operator=(Held&& in) {
    emplace(std::move(in));
    return *this;
}

template <typename S>
typename RetainedBlock<S>::Held RetainedBlock<S>::release() {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(held);
#endif
    Held out = std::move(get());
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    held = false;
#endif
    return out;
}

template <typename S>
void RetainedBlock<S>::emplace(Held&& in) {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    // Releasing first is the caller's job. Overwriting cannot un-push the old block's pages
    // -- the buffer would simply hold two and the next reader would get the stale one -- so
    // this is always a protocol bug rather than a bookkeeping slip.
    ASSERT(!held);
#endif
    ::new (static_cast<void*>(buf)) Held(std::move(in));
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    held = true;
#endif
}

template <typename S>
typename RetainedBlock<S>::Held& RetainedBlock<S>::get() {
    return *reinterpret_cast<Held*>(buf);
}

// --- Accumulator ---

template <typename S, AccumulatorMode Mode>
Accumulator<S, Mode>::Accumulator(const Storage<S>& acc_storage, const Storage<S>& out_storage) :
    acc_storage(acc_storage), out_storage(out_storage) {}

template <typename S, AccumulatorMode Mode>
template <typename Node, typename Epilogue>
Block<S> Accumulator<S, Mode>::accumulate(const Node& node, bool finish, Epilogue epilogue) {
    static_assert(is_fpu_fusion<Node>::value, "Accumulator drives FPU fusions");
    // The same conformance Storage::store enforces. The accumulator does NOT go through
    // store -- it drives the strategy directly -- so without this the two buffers it
    // was built from could disagree with the node's output block and nothing would say
    // so. A page-count match is not enough: Shape<1,2> and Shape<2,1> both hold two
    // pages, and a matmul mis-shaped that way ran correctly on device.
    static_assert(
        same_shape_v<node_shape_t<Node>, S>,
        "this Accumulator's shape is not the shape the matmul produces -- its two Storages must both be "
        "Shape<A_rows, B_cols>");

    if constexpr (std::is_same_v<Epilogue, std::nullptr_t>) {
        Strategy<expr::kind_of_t<Node>>::template run<Mode>(node, acc_storage.cb_id, out_storage.cb_id, reload, finish);
    } else {
        // Apply the epilogue to a bare-chain node of the same geometry to
        // recover just the ops it adds; those are the finish-only ones. The
        // node's own chain stays per-step.
        using Bare =
            MatmulNode<typename Node::lhs_shape, typename Node::rhs_shape, Node::transpose_b, expr::UnaryChain<>>;
        using Fused = decltype(epilogue(std::declval<Bare>()));
        static_assert(
            is_fpu_fusion<Fused>::value,
            "an epilogue must return an FPU fusion node -- it receives the matmul node and should "
            "extend its chain, e.g. [](auto mm) { return relu(mm); }");

        // EVALUATED, not just decltype'd, and that is the whole point. The chain is a
        // type and could be recovered without ever running the lambda -- but an operand
        // is not: `sum.bias(v)` puts v's circular buffer in a RUNTIME member of the node
        // it returns. Taking only the type discards it, which made
        //
        //     [&](auto sum) { return sum.bias(bias_row).relu(); }
        //
        // compile and quietly produce an UNBIASED matmul. It is the natural spelling --
        // bias belongs to the finished total, so it belongs here rather than on the
        // fusion -- and it was silently wrong. Measured at 0.49 max error on a bias of
        // +-0.5, i.e. the whole bias missing.
        //
        // The bare node is built with kNoBias rather than {}: a default-constructed one
        // would carry 0, which is a perfectly good circular buffer index, so an epilogue
        // that sets no bias would have added CB 0 to every output block.
        const Bare bare{{}, kNoBias, kNoBias, kNoBias, kNoBias};
        const auto fused = epilogue(bare);
        Strategy<expr::kind_of_t<Node>>::template run<Mode>(
            node, acc_storage.cb_id, out_storage.cb_id, reload, finish, typename Fused::chain{}, fused.bias_cb);
    }

    reload = !finish;
    // A mid-accumulation Block is RETAINED: its pages stay with the
    // accumulator, so it may be neither transferred nor consumed. Only the
    // finishing Block carries an obligation to reach a consumer.
    return finish ? Block<S>(out_storage) : Block<S>(acc_storage, typename Block<S>::Retained{});
}

template <typename S, AccumulatorMode Mode>
void Accumulator<S, Mode>::clear() {
    reload = false;
}

// --- ComputeBlock ---

template <typename S>
ComputeBlock<S>::ComputeBlock(Block<S> block) : cb_id(block.cb_id) {
    block.consume();
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    cb_wait_front(cb_id, num_pages);
#endif
}

template <typename S>
ComputeBlock<S>::~ComputeBlock() {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    cb_pop_front(cb_id, num_pages);
#endif
}

// --- Expression-leaf adaptors ---

// TileSource identifies a circular buffer, so this must be the cb id.
template <typename S>
TileSource<S> as_node(const ComputeBlock<S>& b) {
    return TileSource<S>{{}, b.get_cb_id()};
}

template <typename S>
TileSource<S> copy(const ComputeBlock<S>& b) {
    return as_node(b);
}

template <typename S>
auto relu(const ComputeBlock<S>& b) {
    return expr::Un<ReluOp, TileSource<S>>{{}, as_node(b)};
}

template <typename S>
auto silu(const ComputeBlock<S>& b) {
    return expr::Un<SiluOp, TileSource<S>>{{}, as_node(b)};
}

template <typename S>
auto exp_(const ComputeBlock<S>& b) {
    return expr::Un<ExpOp, TileSource<S>>{{}, as_node(b)};
}

template <typename S>
auto recip(const ComputeBlock<S>& b) {
    return expr::Un<RecipOp, TileSource<S>>{{}, as_node(b)};
}

template <typename S>
auto sqrt_(const ComputeBlock<S>& b) {
    return expr::Un<SqrtOp, TileSource<S>>{{}, as_node(b)};
}

template <typename S>
auto rsqrt(const ComputeBlock<S>& b) {
    return expr::Un<RsqrtOp, TileSource<S>>{{}, as_node(b)};
}

template <TransposeB Tr, typename SA, typename SB>
auto matmul(const ComputeBlock<SA>& a, const ComputeBlock<SB>& b) {
    return matmul<Tr>(as_node(a), as_node(b));
}

template <Axis A, typename S>
Broadcast<A, S> bcast(const ComputeBlock<S>& v) {
    return Broadcast<A, S>{v.get_cb_id()};
}

template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Sum, expr::UnaryChain<>> reduce_sum(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler) {
    return {{}, b.get_cb_id(), scaler.get_cb_id()};
}

template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Max, expr::UnaryChain<>> reduce_max(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler) {
    return {{}, b.get_cb_id(), scaler.get_cb_id()};
}

template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Avg, expr::UnaryChain<>> reduce_mean(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler) {
    return {{}, b.get_cb_id(), scaler.get_cb_id()};
}

// --- NocAsyncReadTx ---

template <int thread, typename S>
NocAsyncReadTx<thread, S>::NocAsyncReadTx(const Storage<S>& storage) : cb_id(storage.cb_id) {}

template <int thread, typename S>
NocAsyncReadTx<thread, S>::NocAsyncReadTx(uint32_t cb_id) : cb_id(cb_id) {}

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
template <int thread, typename S>
NocAsyncReadTx<thread, S>::~NocAsyncReadTx() {
    ASSERT(waited);
}
#endif

template <int thread, typename S>
Block<S> NocAsyncReadTx<thread, S>::wait() const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        noc_async_read_barrier();
        cb_push_back(cb_id, num_pages);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    return Block<S>(cb_id);
}

// --- NocAsyncWriteTx ---

template <int thread, typename S>
NocAsyncWriteTx<thread, S>::NocAsyncWriteTx(const Storage<S>& storage) : cb_id(storage.cb_id) {}

template <int thread, typename S>
NocAsyncWriteTx<thread, S>::NocAsyncWriteTx(uint32_t cb_id) : cb_id(cb_id) {}

template <int thread, typename S>
NocAsyncWriteTx<thread, S>::~NocAsyncWriteTx() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        // Writes have DEPARTED local L1 -- the release condition for the source
        // buffer. Not the same as having landed; see wait().
        noc_async_writes_flushed();
        cb_pop_front(cb_id, num_pages);
    }
#endif
}

template <int thread, typename S>
void NocAsyncWriteTx<thread, S>::wait() const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        noc_async_write_barrier();  // LANDED at the destination
    }
#endif
}

// --- NocAsyncReadCoreTx ---

template <int thread, typename D, typename S>
NocAsyncReadCoreTx<thread, D, S>::NocAsyncReadCoreTx(const Storage<D>& dst, const Block<S>& src) :
    dst_cb(dst.cb_id), src_cb(src.cb_id) {}

template <int thread, typename D, typename S>
NocAsyncReadCoreTx<thread, D, S>::~NocAsyncReadCoreTx() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        // The source is the peer's L1; the local Block is only a handle.
        cb_pop_front(src_cb, src_pages);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(waited);
#endif
#endif
}

template <int thread, typename D, typename S>
Block<D> NocAsyncReadCoreTx<thread, D, S>::wait() const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        noc_async_read_barrier();  // landed HERE, which is all a pull needs
        cb_push_back(dst_cb, dst_pages);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    return Block<D>(dst_cb);
}

// --- NocAsyncWriteCoreTx ---

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S>::NocAsyncWriteCoreTx(
    const Storage<D>& dst, const Block<S>& src, PhysicalMcast dst_range, uint32_t semaphore_id) :
    NocAsyncWriteCoreTx(dst, src, dst_range.contains(PhysicalCoord::this_core()), semaphore_id) {}

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S>::NocAsyncWriteCoreTx(
    const Storage<D>& dst, const Block<S>& src, bool reader, uint32_t semaphore_id) :
    dst_cb(dst.cb_id), src_cb(src.cb_id), arrived(semaphore_id), reader(reader) {}

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S>::~NocAsyncWriteCoreTx() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        noc_async_writes_flushed();
        // The arrival flag is an ATOMIC, and a write flush does not cover atomics.
        // Leaving one outstanding is an inter-kernel data race: the ack lands after
        // this kernel has finished, against whatever runs next. The watcher calls
        // it "kernel completing with pending NOC transactions".
        noc_async_atomic_barrier();
        cb_pop_front(src_cb, src_pages);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(waited);
#endif
#endif
}

template <int thread, typename D, typename S>
Block<D> NocAsyncWriteCoreTx<thread, D, S>::wait(uint32_t num_writers) const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        if (reader) {
            // Clearing after the count is a window: a writer already into the
            // next round has its increment erased here, hanging the round after.
            // Repeated pushes need the caller to keep the rounds apart --
            // synchronize_cores() between them is enough. See noc_core_write.
            arrived.wait(num_writers).set(0);
        }
        cb_push_back(dst_cb, dst_pages);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    (void)num_writers;  // read only by the reader's collect, above
    return Block<D>(dst_cb);
}

// --- Data movement ---

template <int thread, typename S>
Block<S> fill_reduce_scaler(const Storage<S>& scaler, uint32_t value_bits) {
    // One tile, and the body assumes it: it lays the pattern into a single page's four
    // faces. Previously the page count was simply hardcoded to 1 and a wider Storage
    // would have been silently under-filled.
    static_assert(same_shape_v<S, Shape<1, 1>>, "a reduce scaler is exactly one tile -- Shape<1, 1>");
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_reserve_back(scaler.cb_id, 1);

        const uint32_t words = cb_page_bytes(scaler.cb_id) / sizeof(uint32_t);
        volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(scaler.cb_id));

        // Everything the reduction must not pick up has to read as zero.
        for (uint32_t w = 0; w < words; ++w) {
            page[w] = 0;
        }

        // A tile is four 16x16 faces, so a quarter of the page each, and one row
        // is a sixteenth of a face.
        const uint32_t face_words = words / 4;
        const uint32_t row_words = face_words / 16;
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t w = 0; w < row_words; ++w) {
                page[f * face_words + w] = value_bits;
            }
        }

        cb_push_back(scaler.cb_id, 1);
    }
#else
    (void)value_bits;
#endif
    return Block<S>(scaler.cb_id);
}

// The built-in read, written as a custom routine. Every overload here funnels
// through the Fn form below, so the harness half of the protocol -- reserve, the
// write pointer, the page size, and via the handle the read barrier and push --
// exists in exactly one place, and the built-ins are held to the same contract
// they document for callers.
template <int thread, typename S, typename Accessor>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, const Accessor& acc, uint32_t block_idx) {
    const uint32_t first = block_idx * storage.num_pages;
    return noc_load<thread>(storage, [&](L1Pages pages) {
        for (uint32_t p = 0; p < pages.count; ++p) {
            noc_async_read(acc.get_noc_addr(first + p), pages.addr(p), pages.page_bytes);
        }
    });
}

template <int thread, typename S, typename Fn>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, Fn fn) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        {
            // Blocking here means the consumer has not freed a slot yet -- backpressure,
            // not work. Deeper CBs are what buy it down.
            TT_U_ZONE("LOAD-RESERVE");
            cb_reserve_back(storage.cb_id, storage.num_pages);
        }
        {
            // ISSUING the reads, not waiting for them: they are asynchronous and their
            // cost lands at the barrier.
            TT_U_ZONE("LOAD-ISSUE");
            fn(L1Pages{get_write_ptr(storage.cb_id), cb_page_bytes(storage.cb_id), storage.num_pages});
        }
    }
#else
    (void)fn;
#endif
    return NocAsyncReadTx<thread, S>(storage);
}

// The built-in drain, written as a custom routine, so the harness half -- the
// wait, the read pointer, the page size, and via the handle the flush and pop --
// lives once, in the Fn form below. `first` is read BEFORE the move: consuming
// the Block poisons its fields, and the move is what carries the obligation over
// so consume() still runs exactly once, down there rather than here.
template <int thread, typename S, typename Accessor>
NocAsyncWriteTx<thread, S> noc_store(Block<S> block, const Accessor& acc, uint32_t block_idx) {
    const uint32_t first = block_idx * block.num_pages;
    return noc_store<thread>(std::move(block), [&](L1Pages pages) {
        for (uint32_t p = 0; p < pages.count; ++p) {
            noc_async_write(pages.addr(p), acc.get_noc_addr(first + p), pages.page_bytes);
        }
    });
}

template <int thread, typename S, typename Fn>
NocAsyncWriteTx<thread, S> noc_store(Block<S> block, Fn fn) {
    block.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(block.cb_id, block.num_pages);
        fn(L1Pages{get_read_ptr(block.cb_id), cb_page_bytes(block.cb_id), block.num_pages});
    }
#else
    (void)fn;
#endif
    return NocAsyncWriteTx<thread, S>(block.cb_id);
}

template <int thread, typename S, typename Fn>
NocAsyncReadTx<thread, S> noc_load(
    const Storage<S>& storage,
    PhysicalMcast mcast,
    Semaphore<thread>& receivers_ready,
    Semaphore<thread>& data_sent,
    Fn fn) {
    // Also a custom routine. Every core reserves -- the sender fills its own copy
    // by reading, the receivers have theirs filled for them by the multicast --
    // and publishing is still left to wait(), same as every other load.
    //
    // `fn` is how the SENDER fills its copy, and it runs on the sender only. That is
    // what lets a multicast operand be GATHERED rather than read as one contiguous
    // block: a k-slice of a row-major activation, or a (k, n) tile of a wider weight
    // matrix, is strided in DRAM but perfectly ordinary once it is in L1, and the
    // broadcast that follows does not care how it got there. Same page count either
    // way -- the built-in read issues one per page too.
    return noc_load<thread>(storage, [&](L1Pages pages) {
        const uint32_t num_dests = mcast.volume() - 1;

        if (PhysicalCoord::this_core() == mcast.start) {
            fn(pages);

            // A one-core rectangle -- a 1xN or Nx1 grid gives one for the
            // degenerate axis -- has nobody to broadcast to. Read and publish,
            // skipping a handshake with no counterpart and a multicast to zero
            // destinations.
            if (num_dests == 0) {
                noc_async_read_barrier();
                return;
            }

            // Do not multicast into a receiver's buffer until it has told us the
            // buffer is free. Then clear the count so the next block starts from
            // zero -- leaving it set would let the next call skip the handshake.
            //
            // The three zones below split the sender's k-block into the only parts that can
            // be separately slow. Note that `fn` above merely ISSUES the reads: they are
            // asynchronous, so their cost lands in the barrier, not in the issue.
            {
                TT_U_ZONE("MCAST-READY");  // waiting for receivers to free their buffers
                receivers_ready.wait(num_dests);
                receivers_ready.set(0);
            }

            {
                TT_U_ZONE("MCAST-DRAM");   // the reads issued by fn actually landing
                noc_async_read_barrier();  // payload is in our L1 before we forward it
            }

            {
                TT_U_ZONE("MCAST-SEND");  // the broadcast itself, plus the flag and its flushes
                noc_async_write_multicast(pages.base, mcast.get_noc_addr(pages.base), pages.total_bytes(), num_dests);

                // The flag must not overtake the payload it describes.
                //
                // ttnn's matmul sender does NOT flush here: its payload and flag multicasts go
                // out on the same NOC, VC and command buffer (NOC_CMD_STATIC_VC), so they cannot
                // reorder, and it pays nothing. Ours cannot simply drop it -- removing both
                // flushes deadlocks the device -- because of the set(0) below, not because of
                // ordering. See there.
                noc_async_writes_flushed();

                data_sent.set(1);
                data_sent.set_mcast(mcast);

                // Back to 0 so BOTH semaphores read 0 on every core once this returns.
                // The flush is what makes that safe: set_mcast sources the value from
                // local L1, so the write must have departed before it is overwritten.
                // Otherwise the sender sits at 1 and anything else sharing the pair --
                // synchronize_cores() -- sees a stale release and skips its wait.
                //
                // These two flushes are the measured difference against ttnn's sender, which
                // does neither on Wormhole. They are not removable as they stand: taking both
                // out deadlocks the device, because THIS set(0) can overwrite the flag word
                // before set_mcast's write has sourced it, and the receivers then wait on a 1
                // that never arrives. The cost is two NOC round trips per broadcast, and there
                // are two broadcasts per k-block.
                //
                // The way out is the PROTOCOL, not the flush: a flag the sender never has to
                // reset in the same breath -- an incrementing counter the receiver compares
                // against a block number, say -- needs no set(0) and so no flush to protect it.
                // ttnn gets there by never rewriting the word it just multicast.
                noc_async_writes_flushed();
                data_sent.set(0);
            }
        } else {
            receivers_ready.inc_remote(mcast.start);
            data_sent.wait(1);
            data_sent.set(0);  // rearm for the next block
        }
    });
}

template <int thread, typename S, typename Accessor>
NocAsyncReadTx<thread, S> noc_load(
    const Storage<S>& storage,
    PhysicalMcast mcast,
    Semaphore<thread>& receivers_ready,
    Semaphore<thread>& data_sent,
    const Accessor& acc,
    uint32_t block_idx) {
    const uint32_t first = block_idx * storage.num_pages;
    return noc_load<thread>(storage, mcast, receivers_ready, data_sent, [&](L1Pages pages) {
        for (uint32_t p = 0; p < pages.count; ++p) {
            noc_async_read(acc.get_noc_addr(first + p), pages.addr(p), pages.page_bytes);
        }
    });
}

template <int thread, typename S, typename Accessor>
NocAsyncReadTx<thread, S> noc_load(
    const Storage<S>& storage,
    LogicalMcast mcast,
    Semaphore<thread>& receivers_ready,
    Semaphore<thread>& data_sent,
    const Accessor& acc,
    uint32_t block_idx) {
    return noc_load<thread>(storage, mcast.to_physical(), receivers_ready, data_sent, acc, block_idx);
}

// The handles are locals, not statics. With id-based semaphores the object is a
// handle -- an id plus sem_l1_base[..] + id * L1_ALIGNMENT -- while the counter
// itself lives in host-reserved L1, so static duration buys nothing. It is also
// not available: kernels build with -ftt-no-dyninit and sem_l1_base is a runtime
// extern, so a static Semaphore is rejected outright ("dynamic initialization of
// static-storage is disallowed in this environment").
template <int thread, int pair, typename S, typename Accessor>
NocAsyncReadTx<thread, S> noc_load(
    const Storage<S>& storage, PhysicalMcast mcast, const Accessor& acc, uint32_t block_idx) {
    static_assert(
        kMcastSemsReserved,
        "multicast needs its handshake semaphores reserved by the host: build the program through "
        "unified_program(), which reserves them and defines TT_UNIFIED_MCAST_SEM_BASE -- or pass your own pair "
        "to the six-argument noc_load()");
    Semaphore<thread> receivers_ready(kMcastReadySem<pair>);
    Semaphore<thread> data_sent(kMcastSentSem<pair>);
    return noc_load<thread>(storage, mcast, receivers_ready, data_sent, acc, block_idx);
}

template <int thread, int pair, typename S, typename Accessor>
NocAsyncReadTx<thread, S> noc_load(
    const Storage<S>& storage, LogicalMcast mcast, const Accessor& acc, uint32_t block_idx) {
    return noc_load<thread, pair>(storage, mcast.to_physical(), acc, block_idx);
}

template <int thread, int pair, typename S, typename Fn>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, PhysicalMcast mcast, Fn fn) {
    static_assert(
        kMcastSemsReserved,
        "multicast needs its handshake semaphores reserved by the host: build the program through "
        "unified_program(), which reserves them and defines TT_UNIFIED_MCAST_SEM_BASE -- or pass your own pair "
        "to the five-argument noc_load()");
    Semaphore<thread> receivers_ready(kMcastReadySem<pair>);
    Semaphore<thread> data_sent(kMcastSentSem<pair>);
    return noc_load<thread>(storage, mcast, receivers_ready, data_sent, fn);
}

template <int thread, int pair, typename S, typename Fn>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, LogicalMcast mcast, Fn fn) {
    return noc_load<thread, pair>(storage, mcast.to_physical(), fn);
}

// --- custom_compute ---

namespace detail {

// The routine is the LAST argument, which is where it reads best, so the leading pack has
// to be split off it. A pack cannot precede a named parameter and be deduced, so the whole
// thing arrives as one pack and the split happens here.
template <typename Tuple, std::size_t... I>
constexpr bool leading_all_compute_blocks(std::index_sequence<I...>) {
    return (is_compute_block<std::decay_t<std::tuple_element_t<I, Tuple>>>::value && ...);
}

template <typename Tuple, typename Fn, std::size_t... I>
void custom_compute_invoke(Tuple& packed, Fn& fn, std::index_sequence<I...>) {
    fn(std::get<I>(packed).get_cb_id()...);
}

}  // namespace detail

template <typename... Ts>
void custom_compute(Ts&&... ts) {
    static_assert(sizeof...(Ts) >= 1, "custom_compute takes the blocks and then the routine");
    constexpr std::size_t kBlocks = sizeof...(Ts) - 1;
    static_assert(
        detail::leading_all_compute_blocks<std::tuple<Ts...>>(std::make_index_sequence<kBlocks>{}),
        "custom_compute takes ComputeBlocks first and the routine LAST -- a Storage or a Block in a "
        "block position will not do, since only a ComputeBlock proves the buffer was waited on");

    auto packed = std::forward_as_tuple(ts...);
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    detail::custom_compute_invoke(packed, std::get<kBlocks>(packed), std::make_index_sequence<kBlocks>{});
#else
    // Compiled here, never called. See the contract in api.h.
    (void)packed;
#endif
}

// --- Runtime-argument sentinel ---

template <uint32_t Count>
inline void check_runtime_args() {
    // Every projection reads runtime args through the same get_arg_val, so this needs no
    // thread guard -- unlike the circular-buffer capacity check, whose cb_interface does
    // not link on a TRISC.
    ASSERT(get_arg_val<uint32_t>(Count) == kRuntimeArgSentinel);
}

// --- synchronize_cores: a barrier across CORES, for one data-movement thread ---

template <int thread>
void synchronize_cores(PhysicalMcast region) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        static_assert(
            kMcastSemsReserved,
            "synchronize_cores() needs the reserved handshake semaphores: build the program through "
            "unified_program(), which reserves them and defines TT_UNIFIED_MCAST_SEM_BASE");

        // Same pair the multicast handshake uses, in the same roles: count the
        // participants in, then release them.
        Semaphore<thread> arrived(kMcastReadySem<thread>);
        Semaphore<thread> release(kMcastSentSem<thread>);

        const uint32_t others = region.volume() - 1;
        if (others == 0) {
            return;  // nothing to synchronize with; a 0-destination mcast is not a thing
        }

        if (PhysicalCoord::this_core() == region.start) {
            arrived.wait(others);

            // Clear BEFORE releasing anyone. A core let go early can re-enter the
            // next barrier and increment `arrived` immediately; if the reset came
            // after the release, that arrival would be erased and the next
            // barrier would hang. While everyone is still parked on `release`,
            // no such increment is possible.
            arrived.set(0);

            release.set(1);
            release.set_mcast(region);
            noc_async_writes_flushed();
            release.set(0);  // leave the pair as we found it
        } else {
            arrived.inc_remote(region.start);
            // Same reason as in NocAsyncWriteCoreTx's destructor: an arrival is an
            // atomic, and nothing else here drains it. Parking on `release` does
            // not -- that spins on local L1.
            noc_async_atomic_barrier();
            release.wait(1);
            release.set(0);
        }
    }
#else
    (void)region;
#endif
}

template <int thread>
void synchronize_cores(LogicalMcast region) {
    synchronize_cores<thread>(region.to_physical());
}

template <int thread>
void synchronize_cores() {
    static_assert(
        kCoreGridKnown,
        "synchronize_cores() with no region needs the program's core grid: build the program through "
        "unified_program(), which defines TT_UNIFIED_CORE_GRID_H/W -- or pass a region explicitly");
    static_assert(
        kCoreGridExact,
        "synchronize_cores() with no region barriers the core grid's BOUNDING BOX, and this program's "
        "cores do not fill it -- so the barrier would wait on cores that were never launched, forever. "
        "Either launch on a rectangular core set, or pass the region this barrier actually means");
    synchronize_cores<thread>(LogicalMcast{LogicalCoord::yx(0, 0), Extent::hw(kCoreGridH, kCoreGridW)});
}

// --- Core-to-core movement ---

template <int thread, typename D, typename S>
NocAsyncReadCoreTx<thread, D, S> noc_core_read(
    const Storage<D>& dst, Block<S> src, PhysicalCoord coord, uint32_t byte_offset) {
    src.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(src.cb_id, src.num_pages);
        cb_reserve_back(dst.cb_id, dst.num_pages);
        const uint32_t bytes = cb_page_bytes(dst.cb_id);
        const uint64_t from = coord.get_noc_addr(get_read_ptr(src.cb_id) + byte_offset);
        noc_async_read(from, get_write_ptr(dst.cb_id), bytes * dst.num_pages);
    }
#else
    (void)coord;
    (void)byte_offset;
#endif
    return NocAsyncReadCoreTx<thread, D, S>(dst, src);
}

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S> noc_core_write(
    const Storage<D>& dst, Block<S> src, PhysicalCoord coord, bool write_predicate, uint32_t byte_offset) {
    src.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(src.cb_id, src.num_pages);
        cb_reserve_back(dst.cb_id, dst.num_pages);
        if (write_predicate) {
            const uint32_t bytes = cb_page_bytes(dst.cb_id);
            const uint64_t to = coord.get_noc_addr(get_write_ptr(dst.cb_id) + byte_offset);
            noc_async_write(get_read_ptr(src.cb_id), to, bytes * src.num_pages);

            Semaphore<thread> semaphore(kCopyArrivedSem<thread>);
            semaphore.inc_remote(coord);
        }
    }
#else
    (void)coord;
    (void)write_predicate;
    (void)byte_offset;
#endif
    return NocAsyncWriteCoreTx<thread, D, S>(dst, src, coord, kCopyArrivedSem<thread>);
}

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S> noc_core_write(
    const Storage<D>& dst, Block<S> src, PhysicalMcast mcast, bool write_predicate, uint32_t byte_offset) {
    static_assert(
        kMcastSemsReserved,
        "a multicast noc_core_write needs its arrival semaphore reserved by the host: build the program "
        "through unified_program(), which reserves it and defines TT_UNIFIED_MCAST_SEM_BASE");
    src.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(src.cb_id, src.num_pages);
        cb_reserve_back(dst.cb_id, dst.num_pages);

        if (write_predicate) {
            const uint32_t bytes = cb_page_bytes(dst.cb_id);
            const uint64_t to = mcast.get_noc_addr(get_write_ptr(dst.cb_id) + byte_offset);

            // A core inside the rectangle needs its own copy, which plain
            // multicast skips -- unless src and dst are already the same L1
            // address, where that copy would be onto itself.
            const bool same_local_addr = get_write_ptr(dst.cb_id) == get_read_ptr(src.cb_id);
            const bool loopback = !same_local_addr && mcast.contains(PhysicalCoord::this_core());

            // The two primitives count differently: plain multicast never writes
            // to self and wants self excluded, while the loopback variant does
            // write to self and wants it counted ("mcasting to an 8x8 grid that
            // includes self, num_dests should be 64" -- dataflow_api.h). Both
            // add num_dests to noc_nonposted_writes_acked at issue time, so the
            // wrong one leaves the write-ack counter skewed for the rest of the
            // kernel, not merely the transfer.
            const uint32_t num_dests =
                loopback ? mcast.volume() : mcast.num_dests_excluding(PhysicalCoord::this_core());

            if (loopback) {
                noc_async_write_multicast_loopback_src(get_read_ptr(src.cb_id), to, bytes * src.num_pages, num_dests);
            } else {
                noc_async_write_multicast(get_read_ptr(src.cb_id), to, bytes * src.num_pages, num_dests);
            }

            Semaphore<thread> semaphore(kCopyArrivedSem<thread>);
            semaphore.inc_mcast(mcast);
        }
    }
#else
    (void)mcast;
    (void)write_predicate;
    (void)byte_offset;
#endif
    return NocAsyncWriteCoreTx<thread, D, S>(dst, src, mcast, kCopyArrivedSem<thread>);
}

// The Logical forms translate and forward. `src` moves through, so consume() runs
// exactly once -- in the Physical overload, on its own copy.

template <int thread, typename D, typename S>
NocAsyncReadCoreTx<thread, D, S> noc_core_read(
    const Storage<D>& dst, Block<S> src, LogicalCoord coord, uint32_t byte_offset) {
    return noc_core_read<thread>(dst, std::move(src), coord.to_physical(), byte_offset);
}

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S> noc_core_write(
    const Storage<D>& dst, Block<S> src, LogicalCoord coord, bool write_predicate, uint32_t byte_offset) {
    return noc_core_write<thread>(dst, std::move(src), coord.to_physical(), write_predicate, byte_offset);
}

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S> noc_core_write(
    const Storage<D>& dst, Block<S> src, LogicalMcast mcast, bool write_predicate, uint32_t byte_offset) {
    return noc_core_write<thread>(dst, std::move(src), mcast.to_physical(), write_predicate, byte_offset);
}

}  // namespace unified
}  // namespace tt
