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

// ---------------------------------------------------------------------------
// Geometry
//
// Every body here names a data-movement-only symbol, so all of them are guarded.
// On a compute projection the coordinate types still exist and can be carried
// through a shared statement -- they just cannot be resolved to a NOC address.
// ---------------------------------------------------------------------------

inline PhysicalCoord PhysicalCoord::this_core() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    return PhysicalCoord{my_y[noc_index], my_x[noc_index]};
#else
    return PhysicalCoord{0, 0};
#endif
}

inline uint64_t PhysicalCoord::get_noc_addr(uintptr_t l1_addr) const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    // Qualified: this class has a member of the same name.
    return ::get_noc_addr(x, y, static_cast<uint32_t>(l1_addr));
#else
    (void)l1_addr;
    return 0;
#endif
}

inline LogicalCoord LogicalCoord::this_core() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    return LogicalCoord{get_relative_logical_y(), get_relative_logical_x()};
#else
    return LogicalCoord{0, 0};
#endif
}

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

inline uint64_t PhysicalMcast::get_noc_addr(uintptr_t l1_addr) const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
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

// ---------------------------------------------------------------------------
// Semaphore
// ---------------------------------------------------------------------------

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
        sem.inc_multicast(
            Noc{},
            mcast.start.x,
            mcast.start.y,
            mcast.end.x,
            mcast.end.y,
            value,
            mcast.num_dests_excluding(PhysicalCoord::this_core()));
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
        sem.set_multicast(
            Noc{},
            mcast.start.x,
            mcast.start.y,
            mcast.end.x,
            mcast.end.y,
            mcast.num_dests_excluding(PhysicalCoord::this_core()));
    }
#endif
    (void)mcast;
    return *this;
}

template <int thread>
Semaphore<thread>& Semaphore<thread>::set_mcast(LogicalMcast mcast) {
    return set_mcast(mcast.to_physical());
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

template <typename Node>
Block Storage::store(const Node& node) {
    Strategy<expr::kind_of_t<Node>>::run(node, cb_id, num_tiles);
    return Block(cb_id, num_tiles);
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Accumulator
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// ComputeBlock
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Expression-leaf adaptors
// ---------------------------------------------------------------------------

// TileSource identifies a circular buffer, so this must be the cb id.
inline TileSource as_node(const ComputeBlock& b) { return TileSource{b.get_cb_id()}; }

inline auto relu(const ComputeBlock& b) { return expr::Un<ReluOp, TileSource>{as_node(b)}; }

inline auto exp_(const ComputeBlock& b) { return expr::Un<ExpOp, TileSource>{as_node(b)}; }

template <typename Geometry>
auto matmul(const ComputeBlock& a, const ComputeBlock& b) {
    return matmul<Geometry>(as_node(a), as_node(b));
}

// ---------------------------------------------------------------------------
// NocAsyncReadTx
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// NocAsyncWriteTx
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// NocAsyncCopyTx
// ---------------------------------------------------------------------------

template <int thread, bool SrcIsLocal>
NocAsyncCopyTx<thread, SrcIsLocal>::NocAsyncCopyTx(const Storage& dst, const Block& src) :
    dst_cb(dst.cb_id), dst_tiles(dst.num_tiles), src_cb(src.cb_id), src_tiles(src.num_tiles) {}

template <int thread, bool SrcIsLocal>
NocAsyncCopyTx<thread, SrcIsLocal>::~NocAsyncCopyTx() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        if constexpr (SrcIsLocal) {
            // Writes have DEPARTED local L1 -- the release condition for a
            // source buffer. Not the same as having landed.
            noc_async_writes_flushed();
        }
        cb_pop_front(src_cb, src_tiles);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(waited);
#endif
#endif
}

template <int thread, bool SrcIsLocal>
Block NocAsyncCopyTx<thread, SrcIsLocal>::wait() const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        if constexpr (SrcIsLocal) {
            noc_async_write_barrier();  // LANDED at the destination
        } else {
            noc_async_read_barrier();
        }
        cb_push_back(dst_cb, dst_tiles);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    return Block(dst_cb, dst_tiles);
}

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

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

            // Do not multicast into a receiver's buffer until it has told us the
            // buffer is free. Then clear the count so the next block starts from
            // zero -- leaving it set would let the next call skip the handshake.
            receivers_ready.wait(num_dests);
            receivers_ready.set(0);

            noc_async_read_barrier();  // payload is in our L1 before we forward it

            noc_async_write_multicast(l1, mcast.get_noc_addr(l1), tiles * bytes, num_dests);

            // The flag must not overtake the payload it describes.
            noc_async_writes_flushed();

            // Held at 1 on the sender: this is the constant being broadcast, and
            // resetting it here would race the outbound read of its own L1. Each
            // receiver clears its own copy instead.
            data_sent.set(1);
            data_sent.set_mcast(mcast);
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
template <int thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(const Storage& storage, PhysicalMcast mcast, const Accessor& acc, uint32_t block_idx) {
    static_assert(
        kMcastSemsReserved,
        "multicast needs its handshake semaphores reserved by the host: build the program through "
        "unified_program(), which reserves them and defines TT_UNIFIED_MCAST_SEM_BASE -- or pass your own pair "
        "to the six-argument noc_load()");
    Semaphore<thread> receivers_ready(kMcastReadySem<thread>);
    Semaphore<thread> data_sent(kMcastSentSem<thread>);
    return noc_load<thread>(storage, mcast, receivers_ready, data_sent, acc, block_idx);
}

template <int thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(const Storage& storage, LogicalMcast mcast, const Accessor& acc, uint32_t block_idx) {
    return noc_load<thread>(storage, mcast.to_physical(), acc, block_idx);
}

// ---------------------------------------------------------------------------
// Core-to-core movement
// ---------------------------------------------------------------------------

template <int thread>
NocAsyncCopyTx<thread, /*SrcIsLocal=*/false> noc_read(
    const Storage& storage, Block block, LogicalCoord coord, uint32_t offset) {
    block.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(block.cb_id, block.num_tiles);
        cb_reserve_back(storage.cb_id, storage.num_tiles);
        const uint32_t bytes = cb_page_bytes(storage.cb_id);
        const uint64_t src = coord.get_noc_addr(get_read_ptr(block.cb_id) + offset);
        noc_async_read(src, get_write_ptr(storage.cb_id), bytes * storage.num_tiles);
    }
#else
    (void)coord;
    (void)offset;
#endif
    return NocAsyncCopyTx<thread, false>(storage, block);
}

template <int thread>
NocAsyncCopyTx<thread, /*SrcIsLocal=*/true> noc_write(
    const Storage& storage, Block block, LogicalCoord coord, uint32_t offset) {
    block.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(block.cb_id, block.num_tiles);
        cb_reserve_back(storage.cb_id, storage.num_tiles);
        const uint32_t bytes = cb_page_bytes(storage.cb_id);
        const uint64_t dst = coord.get_noc_addr(get_write_ptr(storage.cb_id) + offset);
        noc_async_write(get_read_ptr(block.cb_id), dst, bytes * block.num_tiles);
    }
#else
    (void)coord;
    (void)offset;
#endif
    return NocAsyncCopyTx<thread, true>(storage, block);
}

}  // namespace unified
}  // namespace tt
