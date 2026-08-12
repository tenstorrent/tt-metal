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

inline Block::Block(Block&& o) {
    cb_id = o.cb_id;
    num_tiles = o.num_tiles;
}

inline Block& Block::operator=(Block&& o) {
    cb_id = o.cb_id;
    num_tiles = o.num_tiles;
    return *this;
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
    return finish ? Block(out_storage) : Block(acc_storage);
}

template <AccumulatorMode Mode>
void Accumulator<Mode>::clear() {
    reload = false;
}

// ---------------------------------------------------------------------------
// ComputeBlock
// ---------------------------------------------------------------------------

inline ComputeBlock::ComputeBlock(Block block) : cb_id(block.cb_id), num_tiles(block.num_tiles) {
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

template <int thread, typename Accessor>
NocAsyncWriteTx<thread> noc_store(Block block, const Accessor& acc, uint32_t block_idx) {
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

// ---------------------------------------------------------------------------
// Core-to-core movement
// ---------------------------------------------------------------------------

template <int thread>
NocAsyncCopyTx<thread, /*SrcIsLocal=*/false> noc_read(
    const Storage& storage, Block block, Coord coord, uint32_t offset) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(block.cb_id, block.num_tiles);
        cb_reserve_back(storage.cb_id, storage.num_tiles);
        const uint32_t bytes = cb_page_bytes(storage.cb_id);
        const uint64_t src = get_noc_addr(coord.x, coord.y, get_read_ptr(block.cb_id) + offset);
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
    const Storage& storage, Block block, Coord coord, uint32_t offset) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait_front(block.cb_id, block.num_tiles);
        cb_reserve_back(storage.cb_id, storage.num_tiles);
        const uint32_t bytes = cb_page_bytes(storage.cb_id);
        const uint64_t dst = get_noc_addr(coord.x, coord.y, get_write_ptr(storage.cb_id) + offset);
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
