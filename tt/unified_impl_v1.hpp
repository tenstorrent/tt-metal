// SPDX-License-Identifier: Apache-2.0
//
// Definitions for the core API declared in tt/unified_api.h, targeting the
// Metal v1 programming model (Wormhole / Blackhole: 2 DM threads + a compute
// thread that metal splits three ways).
//
// Include <tt/unified>, not this header directly.

#pragma once

#include <tt/unified_api.h>

namespace tt {
namespace unified {

// --- Geometry ---
//
// Every body here names a data-movement-only symbol, so all of them are guarded.
// On a compute projection the coordinate types still exist and can be carried
// through a shared statement -- they just cannot be resolved to a NOC address.

inline PhysicalCoord PhysicalCoord::this_core() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    return PhysicalCoord{my_y[noc_index], my_x[noc_index]};
#else
    return PhysicalCoord{0, 0};
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
    return LogicalCoord{get_relative_logical_y(), get_relative_logical_x()};
}

inline LogicalCoord LogicalCoord::origin() { return LogicalCoord{0, 0}; }

inline PhysicalCoord LogicalCoord::to_physical(uint32_t y_offset, uint32_t x_offset) const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    return PhysicalCoord{
        worker_logical_row_to_virtual_row[y + y_offset], worker_logical_col_to_virtual_col[x + x_offset]};
#else
    (void)y_offset;
    (void)x_offset;
    return PhysicalCoord{0, 0};
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
    return PhysicalMcast{coord.to_physical(), coord.to_physical(shape.h - 1, shape.w - 1)};
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

template <typename Node>
Block Storage::store(const Node& node) {
    Strategy<expr::kind_of_t<Node>>::run(node, cb_id, num_tiles);
    return Block(cb_id, num_tiles);
}

// --- Block ---

inline Block::Block(const Storage& storage) : cb_id(storage.cb_id), num_tiles(storage.num_tiles) {}

inline Block::Block(uint32_t cb_id, uint32_t num_tiles) : cb_id(cb_id), num_tiles(num_tiles) {}

inline Block::Block(const Storage& storage, Retained) : cb_id(storage.cb_id), num_tiles(storage.num_tiles) {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    must_consume = false;
#endif
}

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
inline Block::~Block() { ASSERT(!must_consume || consumed); }
#endif

// A move transfers the OBLIGATION rather than discharging it: the destination
// inherits must_consume/consumed unchanged, and only the source goes silent.
// Clearing the flag on the destination instead would let `Block b2 =
// std::move(b1);` launder the debt -- b2 could then be dropped without a pop
// and nothing would complain.
inline Block::Block(Block&& o) : cb_id(o.cb_id), num_tiles(o.num_tiles) {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(o.must_consume);  // a retained block belongs to its accumulator
    must_consume = o.must_consume;
    consumed = o.consumed;
    o.must_consume = false;
    o.consumed = true;
    o.cb_id = kMovedFrom;
    o.num_tiles = kMovedFrom;
#endif
}

inline Block& Block::operator=(Block&& o) {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(o.must_consume);             // a retained block belongs to its accumulator
    ASSERT(!must_consume || consumed);  // do not drop pages this Block still owes
#endif
    cb_id = o.cb_id;
    num_tiles = o.num_tiles;
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    must_consume = o.must_consume;
    consumed = o.consumed;
    o.must_consume = false;
    o.consumed = true;
    o.cb_id = kMovedFrom;
    o.num_tiles = kMovedFrom;
#endif
    return *this;
}

inline void Block::consume() {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    // Catches a retained block reaching a consumer directly -- passing
    // accumulate(node, /*finish=*/false) straight into noc_store elides the
    // move, so the check in the move constructor never runs.
    ASSERT(must_consume);
    ASSERT(!consumed);
    consumed = true;
#endif
}

// --- Accumulator ---

template <AccumulatorMode Mode>
Accumulator<Mode>::Accumulator(const Storage& acc_storage, const Storage& out_storage) :
    acc_storage(acc_storage), out_storage(out_storage) {}

template <AccumulatorMode Mode>
template <typename Node, typename Epilogue>
Block Accumulator<Mode>::accumulate(const Node& node, bool finish, Epilogue epilogue) {
    static_assert(is_fpu_fusion<Node>::value, "Accumulator drives FPU fusions");

    if constexpr (std::is_same_v<Epilogue, std::nullptr_t>) {
        Strategy<expr::kind_of_t<Node>>::template run<Mode>(node, acc_storage.cb_id, out_storage.cb_id, reload, finish);
    } else {
        // Apply the epilogue to a bare-chain node of the same geometry to
        // recover just the ops it adds; those are the finish-only ones. The
        // node's own chain stays per-step.
        using Bare = MatmulNode<typename Node::geometry, expr::UnaryChain<>>;
        using Fused = decltype(epilogue(std::declval<Bare>()));
        static_assert(
            is_fpu_fusion<Fused>::value,
            "an epilogue must return an FPU fusion node -- it receives the matmul node and should "
            "extend its chain, e.g. [](auto mm) { return relu(mm); }");
        Strategy<expr::kind_of_t<Node>>::template run<Mode>(
            node, acc_storage.cb_id, out_storage.cb_id, reload, finish, typename Fused::chain{});
    }

    reload = !finish;
    // A mid-accumulation Block is RETAINED: its pages stay with the
    // accumulator, so it may be neither transferred nor consumed. Only the
    // finishing Block carries an obligation to reach a consumer.
    return finish ? Block(out_storage) : Block(acc_storage, Block::Retained{});
}

template <AccumulatorMode Mode>
void Accumulator<Mode>::clear() {
    reload = false;
}

// --- ComputeBlock ---

inline ComputeBlock::ComputeBlock(Block block) : cb_id(block.cb_id), num_tiles(block.num_tiles) {
    block.consume();
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    cb_wait_front(cb_id, num_tiles);
#endif
}

inline ComputeBlock::~ComputeBlock() {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    cb_pop_front(cb_id, num_tiles);
#endif
}

inline expr::Un<ExpOp, TileSource> ComputeBlock::exp() const { return {TileSource{cb_id}}; }

// --- Expression-leaf adaptors ---

// TileSource identifies a circular buffer, so this must be the cb id.
inline TileSource as_node(const ComputeBlock& b) { return TileSource{b.get_cb_id()}; }

inline auto relu(const ComputeBlock& b) { return expr::Un<ReluOp, TileSource>{as_node(b)}; }

inline auto exp_(const ComputeBlock& b) { return expr::Un<ExpOp, TileSource>{as_node(b)}; }

template <typename Geometry>
auto matmul(const ComputeBlock& a, const ComputeBlock& b) {
    return matmul<Geometry>(as_node(a), as_node(b));
}

template <typename Geometry, ReduceAxis Axis>
ReduceNode<Geometry, Axis, ReducePool::Sum, expr::UnaryChain<>> reduce_sum(
    const ComputeBlock& b, const Storage& scaler) {
    return {b.get_cb_id(), scaler.cb_id};
}

// --- NocAsyncReadTx ---

template <int thread>
NocAsyncReadTx<thread>::NocAsyncReadTx(const Storage& storage) : cb_id(storage.cb_id), num_tiles(storage.num_tiles) {}

template <int thread>
NocAsyncReadTx<thread>::NocAsyncReadTx(uint32_t cb_id, uint32_t num_tiles) : cb_id(cb_id), num_tiles(num_tiles) {}

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
template <int thread>
NocAsyncReadTx<thread>::~NocAsyncReadTx() {
    ASSERT(waited);
}
#endif

template <int thread>
Block NocAsyncReadTx<thread>::wait() const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        noc_async_read_barrier();
        cb_push_back(cb_id, num_tiles);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    return Block(cb_id, num_tiles);
}

// --- NocAsyncWriteTx ---

template <int thread>
NocAsyncWriteTx<thread>::NocAsyncWriteTx(const Storage& storage) : cb_id(storage.cb_id), num_tiles(storage.num_tiles) {}

template <int thread>
NocAsyncWriteTx<thread>::NocAsyncWriteTx(uint32_t cb_id, uint32_t num_tiles) : cb_id(cb_id), num_tiles(num_tiles) {}

template <int thread>
NocAsyncWriteTx<thread>::~NocAsyncWriteTx() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        // Writes have DEPARTED local L1 -- the release condition for the source
        // buffer. Not the same as having landed; see wait().
        noc_async_writes_flushed();
        cb_pop_front(cb_id, num_tiles);
    }
#endif
}

template <int thread>
void NocAsyncWriteTx<thread>::wait() const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        noc_async_write_barrier();  // LANDED at the destination
    }
#endif
}

// --- NocAsyncReadCoreTx ---

template <int thread>
NocAsyncReadCoreTx<thread>::NocAsyncReadCoreTx(const Storage& dst, const Block& src) :
    dst_cb(dst.cb_id), dst_tiles(dst.num_tiles), src_cb(src.cb_id), src_tiles(src.num_tiles) {}

template <int thread>
NocAsyncReadCoreTx<thread>::~NocAsyncReadCoreTx() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        // The source is the peer's L1; the local Block is only a handle.
        cb_pop_front(src_cb, src_tiles);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(waited);
#endif
#endif
}

template <int thread>
Block NocAsyncReadCoreTx<thread>::wait() const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        noc_async_read_barrier();  // landed HERE, which is all a pull needs
        cb_push_back(dst_cb, dst_tiles);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    return Block(dst_cb, dst_tiles);
}

// --- NocAsyncWriteCoreTx ---

template <int thread>
NocAsyncWriteCoreTx<thread>::NocAsyncWriteCoreTx(
    const Storage& dst, const Block& src, PhysicalMcast dst_range, uint32_t semaphore_id) :
    NocAsyncWriteCoreTx(dst, src, dst_range.contains(PhysicalCoord::this_core()), semaphore_id) {}

template <int thread>
NocAsyncWriteCoreTx<thread>::NocAsyncWriteCoreTx(
    const Storage& dst, const Block& src, bool reader, uint32_t semaphore_id) :
    dst_cb(dst.cb_id),
    dst_tiles(dst.num_tiles),
    src_cb(src.cb_id),
    src_tiles(src.num_tiles),
    arrived(semaphore_id),
    reader(reader) {}

template <int thread>
NocAsyncWriteCoreTx<thread>::~NocAsyncWriteCoreTx() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        noc_async_writes_flushed();
        // The arrival flag is an ATOMIC, and a write flush does not cover atomics.
        // Leaving one outstanding is an inter-kernel data race: the ack lands after
        // this kernel has finished, against whatever runs next. The watcher calls
        // it "kernel completing with pending NOC transactions".
        noc_async_atomic_barrier();
        cb_pop_front(src_cb, src_tiles);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(waited);
#endif
#endif
}

template <int thread>
Block NocAsyncWriteCoreTx<thread>::wait(uint32_t num_writers) const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        if (reader) {
            // Clearing after the count is a window: a writer already into the
            // next round has its increment erased here, hanging the round after.
            // Repeated pushes need the caller to keep the rounds apart --
            // synchronize_cores() between them is enough. See noc_core_write.
            arrived.wait(num_writers).set(0);
        }
        cb_push_back(dst_cb, dst_tiles);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    return Block(dst_cb, dst_tiles);
}

// --- Data movement ---

template <int thread>
void fill_reduce_scaler(const Storage& scaler, uint32_t value_bits) {
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
    (void)scaler;
    (void)value_bits;
#endif
}

template <int thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(const Storage& storage, const Accessor& acc, uint32_t block_idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_reserve_back(storage.cb_id, storage.num_tiles);
        uint32_t l1 = get_write_ptr(storage.cb_id);
        const uint32_t bytes = cb_page_bytes(storage.cb_id);
        const uint32_t first = block_idx * storage.num_tiles;
        for (uint32_t p = 0; p < storage.num_tiles; ++p) {
            noc_async_read(acc.get_noc_addr(first + p), l1, bytes);
            l1 += bytes;
        }
    }
#else
    (void)acc;
    (void)block_idx;
#endif
    return NocAsyncReadTx<thread>(storage);
}

template <int thread, typename Fn>
NocAsyncReadTx<thread> noc_load(const Storage& storage, Fn fn) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_reserve_back(storage.cb_id, storage.num_tiles);
        fn(get_write_ptr(storage.cb_id), cb_page_bytes(storage.cb_id));
    }
#else
    (void)fn;
#endif
    return NocAsyncReadTx<thread>(storage);
}

template <int thread, typename Accessor>
NocAsyncWriteTx<thread> noc_store(Block block, const Accessor& acc, uint32_t block_idx) {
    block.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(block.cb_id, block.num_tiles);
        uint32_t l1 = get_read_ptr(block.cb_id);
        const uint32_t bytes = cb_page_bytes(block.cb_id);
        const uint32_t first = block_idx * block.num_tiles;
        for (uint32_t p = 0; p < block.num_tiles; ++p) {
            noc_async_write(l1, acc.get_noc_addr(first + p), bytes);
            l1 += bytes;
        }
    }
#else
    (void)acc;
    (void)block_idx;
#endif
    return NocAsyncWriteTx<thread>(block.cb_id, block.num_tiles);
}

template <int thread, typename Fn>
NocAsyncWriteTx<thread> noc_store(Block block, Fn fn) {
    block.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(block.cb_id, block.num_tiles);
        fn(get_read_ptr(block.cb_id), cb_page_bytes(block.cb_id));
    }
#else
    (void)fn;
#endif
    return NocAsyncWriteTx<thread>(block.cb_id, block.num_tiles);
}

template <int thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(
    const Storage& storage,
    PhysicalMcast mcast,
    Semaphore<thread>& receivers_ready,
    Semaphore<thread>& data_sent,
    const Accessor& acc,
    uint32_t block_idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        const uint32_t tiles = storage.num_tiles;
        const uint32_t bytes = cb_page_bytes(storage.cb_id);

        // Every core reserves: the sender fills its own copy by reading, the
        // receivers have theirs filled for them by the multicast.
        cb_reserve_back(storage.cb_id, tiles);
        const uint32_t l1 = get_write_ptr(storage.cb_id);
        const uint32_t num_dests = mcast.volume() - 1;

        if (PhysicalCoord::this_core() == mcast.start) {
            uint32_t at = l1;
            const uint32_t first = block_idx * tiles;
            for (uint32_t p = 0; p < tiles; ++p) {
                noc_async_read(acc.get_noc_addr(first + p), at, bytes);
                at += bytes;
            }

            // A one-core rectangle -- a 1xN or Nx1 grid gives one for the
            // degenerate axis -- has nobody to broadcast to. Read and publish,
            // skipping a handshake with no counterpart and a multicast to zero
            // destinations.
            if (num_dests == 0) {
                noc_async_read_barrier();
                return NocAsyncReadTx<thread>(storage);
            }

            // Do not multicast into a receiver's buffer until it has told us the
            // buffer is free. Then clear the count so the next block starts from
            // zero -- leaving it set would let the next call skip the handshake.
            receivers_ready.wait(num_dests);
            receivers_ready.set(0);

            noc_async_read_barrier();  // payload is in our L1 before we forward it

            noc_async_write_multicast(l1, mcast.get_noc_addr(l1), tiles * bytes, num_dests);

            // The flag must not overtake the payload it describes.
            noc_async_writes_flushed();

            data_sent.set(1);
            data_sent.set_mcast(mcast);

            // Back to 0 so BOTH semaphores read 0 on every core once this returns.
            // The flush is what makes that safe: set_mcast sources the value from
            // local L1, so the write must have departed before it is overwritten.
            // Otherwise the sender sits at 1 and anything else sharing the pair --
            // synchronize_cores() -- sees a stale release and skips its wait.
            noc_async_writes_flushed();
            data_sent.set(0);
        } else {
            receivers_ready.inc_remote(mcast.start);
            data_sent.wait(1);
            data_sent.set(0);  // rearm for the next block
        }
    }
#else
    (void)mcast;
    (void)receivers_ready;
    (void)data_sent;
    (void)acc;
    (void)block_idx;
#endif
    // Publishing is left to wait(), same as every other load: the pages become
    // visible to the consumer when the caller says so, not as a side effect.
    return NocAsyncReadTx<thread>(storage);
}

template <int thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(
    const Storage& storage,
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
template <int thread, int pair, typename Accessor>
NocAsyncReadTx<thread> noc_load(const Storage& storage, PhysicalMcast mcast, const Accessor& acc, uint32_t block_idx) {
    static_assert(
        kMcastSemsReserved,
        "multicast needs its handshake semaphores reserved by the host: build the program through "
        "unified_program(), which reserves them and defines TT_UNIFIED_MCAST_SEM_BASE -- or pass your own pair "
        "to the six-argument noc_load()");
    Semaphore<thread> receivers_ready(kMcastReadySem<pair>);
    Semaphore<thread> data_sent(kMcastSentSem<pair>);
    return noc_load<thread>(storage, mcast, receivers_ready, data_sent, acc, block_idx);
}

template <int thread, int pair, typename Accessor>
NocAsyncReadTx<thread> noc_load(const Storage& storage, LogicalMcast mcast, const Accessor& acc, uint32_t block_idx) {
    return noc_load<thread, pair>(storage, mcast.to_physical(), acc, block_idx);
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
    synchronize_cores<thread>(LogicalMcast{LogicalCoord{0, 0}, Shape{kCoreGridH, kCoreGridW}});
}

// --- Core-to-core movement ---

template <int thread>
NocAsyncReadCoreTx<thread> noc_core_read(const Storage& dst, Block src, PhysicalCoord coord, uint32_t byte_offset) {
    src.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(src.cb_id, src.num_tiles);
        cb_reserve_back(dst.cb_id, dst.num_tiles);
        const uint32_t bytes = cb_page_bytes(dst.cb_id);
        const uint64_t from = coord.get_noc_addr(get_read_ptr(src.cb_id) + byte_offset);
        noc_async_read(from, get_write_ptr(dst.cb_id), bytes * dst.num_tiles);
    }
#else
    (void)coord;
    (void)byte_offset;
#endif
    return NocAsyncReadCoreTx<thread>(dst, src);
}

template <int thread>
NocAsyncWriteCoreTx<thread> noc_core_write(
    const Storage& dst, Block src, PhysicalCoord coord, bool write_predicate, uint32_t byte_offset) {
    src.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(src.cb_id, src.num_tiles);
        cb_reserve_back(dst.cb_id, dst.num_tiles);
        if (write_predicate) {
            const uint32_t bytes = cb_page_bytes(dst.cb_id);
            const uint64_t to = coord.get_noc_addr(get_write_ptr(dst.cb_id) + byte_offset);
            noc_async_write(get_read_ptr(src.cb_id), to, bytes * src.num_tiles);

            Semaphore<thread> semaphore(kCopyArrivedSem<thread>);
            semaphore.inc_remote(coord);
        }
    }
#else
    (void)coord;
    (void)byte_offset;
#endif
    return NocAsyncWriteCoreTx<thread>(dst, src, coord, kCopyArrivedSem<thread>);
}

template <int thread>
NocAsyncWriteCoreTx<thread> noc_core_write(
    const Storage& dst, Block src, PhysicalMcast mcast, bool write_predicate, uint32_t byte_offset) {
    static_assert(
        kMcastSemsReserved,
        "a multicast noc_core_write needs its arrival semaphore reserved by the host: build the program "
        "through unified_program(), which reserves it and defines TT_UNIFIED_MCAST_SEM_BASE");
    src.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(src.cb_id, src.num_tiles);
        cb_reserve_back(dst.cb_id, dst.num_tiles);

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
                noc_async_write_multicast_loopback_src(get_read_ptr(src.cb_id), to, bytes * src.num_tiles, num_dests);
            } else {
                noc_async_write_multicast(get_read_ptr(src.cb_id), to, bytes * src.num_tiles, num_dests);
            }

            Semaphore<thread> semaphore(kCopyArrivedSem<thread>);
            semaphore.inc_mcast(mcast);
        }
    }
#else
    (void)mcast;
    (void)byte_offset;
#endif
    return NocAsyncWriteCoreTx<thread>(dst, src, mcast, kCopyArrivedSem<thread>);
}

// The Logical forms translate and forward. `src` moves through, so consume() runs
// exactly once -- in the Physical overload, on its own copy.

template <int thread>
NocAsyncReadCoreTx<thread> noc_core_read(const Storage& dst, Block src, LogicalCoord coord, uint32_t byte_offset) {
    return noc_core_read<thread>(dst, std::move(src), coord.to_physical(), byte_offset);
}

template <int thread>
NocAsyncWriteCoreTx<thread> noc_core_write(
    const Storage& dst, Block src, LogicalCoord coord, bool write_predicate, uint32_t byte_offset) {
    return noc_core_write<thread>(dst, std::move(src), coord.to_physical(), write_predicate, byte_offset);
}

template <int thread>
NocAsyncWriteCoreTx<thread> noc_core_write(
    const Storage& dst, Block src, LogicalMcast mcast, bool write_predicate, uint32_t byte_offset) {
    return noc_core_write<thread>(dst, std::move(src), mcast.to_physical(), write_predicate, byte_offset);
}

}  // namespace unified
}  // namespace tt
