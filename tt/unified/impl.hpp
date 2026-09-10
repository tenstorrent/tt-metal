// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <new>

#include <tt/unified/api.h>

namespace tt {
namespace unified {

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
    return ::get_noc_addr(x, y, static_cast<uint32_t>(l1_addr));
#else
    (void)l1_addr;
    return 0;
#endif
}

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
        ASSERT(PhysicalCoord::this_core() == mcast.start);
#endif
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

template <typename S>
template <typename Node>
Block<S> Storage<S>::store(const Node& node) const {
    static_assert(
        same_shape_v<node_shape_t<Node>, S>,
        "this Storage's shape is not the shape the expression produces -- compare the Storage<...> "
        "argument against the operands' shapes and the axis or geometry driving the op");
    Strategy<expr::kind_of_t<Node>>::run(node, dfb_id, num_entries);
    return Block<S>(dfb_id);
}

template <typename S>
Block<S>::Block(const Storage<S>& storage) : dfb_id(storage.dfb_id) {}

template <typename S>
Block<S>::Block(uint32_t dfb_id) : dfb_id(dfb_id) {}

template <typename S>
Block<S>::Block(const Storage<S>& storage, Retained) : dfb_id(storage.dfb_id) {
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

template <typename S>
Block<S>::Block(Block&& o) : dfb_id(o.dfb_id) {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(o.must_consume);
    must_consume = o.must_consume;
    consumed = o.consumed;
    o.must_consume = false;
    o.consumed = true;
    o.dfb_id = kMovedFrom;
#endif
}

template <typename S>
Block<S>& Block<S>::operator=(Block&& o) {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(o.must_consume);
    ASSERT(!must_consume || consumed);
#endif
    dfb_id = o.dfb_id;
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    must_consume = o.must_consume;
    consumed = o.consumed;
    o.must_consume = false;
    o.consumed = true;
    o.dfb_id = kMovedFrom;
#endif
    return *this;
}

template <typename S>
void Block<S>::consume() {
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ASSERT(must_consume);
    ASSERT(!consumed);
    consumed = true;
#endif
}

template <typename S>
RetainedBlock<S>::RetainedBlock(Held&& block) {
    emplace(std::move(block));
}

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
template <typename S>
RetainedBlock<S>::~RetainedBlock() {
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

template <typename S, AccumulatorMode Mode>
Accumulator<S, Mode>::Accumulator(const Storage<S>& acc_storage, const Storage<S>& out_storage) :
    acc_storage(acc_storage), out_storage(out_storage) {}

template <typename S, AccumulatorMode Mode>
template <typename Node, typename Epilogue>
Block<S> Accumulator<S, Mode>::accumulate(const Node& node, bool finish, Epilogue epilogue) {
    static_assert(is_fpu_fusion<Node>::value, "Accumulator drives FPU fusions");
    static_assert(
        same_shape_v<node_shape_t<Node>, S>,
        "this Accumulator's shape is not the shape the matmul produces -- its two Storages must both be "
        "Shape<A_rows, B_cols>");

    if constexpr (std::is_same_v<Epilogue, std::nullptr_t>) {
        Strategy<expr::kind_of_t<Node>>::template run<Mode>(
            node, acc_storage.dfb_id, out_storage.dfb_id, reload, finish);
    } else {
        using Bare =
            MatmulNode<typename Node::lhs_shape, typename Node::rhs_shape, Node::transpose_b, expr::UnaryChain<>>;
        using Fused = decltype(epilogue(std::declval<Bare>()));
        static_assert(
            is_fpu_fusion<Fused>::value,
            "an epilogue must return an FPU fusion node -- it receives the matmul node and should "
            "extend its chain, e.g. [](auto mm) { return relu(mm); }");

        const Bare bare{{}, kNoBias, kNoBias, kNoBias, kNoBias};
        const auto fused = epilogue(bare);
        Strategy<expr::kind_of_t<Node>>::template run<Mode>(
            node, acc_storage.dfb_id, out_storage.dfb_id, reload, finish, typename Fused::chain{}, fused.bias_dfb);
    }

    reload = !finish;
    return finish ? Block<S>(out_storage) : Block<S>(acc_storage, typename Block<S>::Retained{});
}

template <typename S, AccumulatorMode Mode>
void Accumulator<S, Mode>::clear() {
    reload = false;
}

#if defined(TT_UNIFIED_CHECK_ALIASING) && TT_UNIFIED_CHECK_ALIASING && defined(ASSERT_ENABLED) && ASSERT_ENABLED
#define TT_U_ALIASING_CHECKED 1

inline uint32_t& live_compute_blocks() {
    static uint32_t live = 0;
    return live;
}

__attribute__((noinline)) inline void claim_compute_block(uint32_t dfb_id) {
    ASSERT((live_compute_blocks() & (1u << (dfb_id & 31u))) == 0);
    live_compute_blocks() |= (1u << (dfb_id & 31u));
}

__attribute__((noinline)) inline void release_compute_block(uint32_t dfb_id) {
    live_compute_blocks() &= ~(1u << (dfb_id & 31u));
}
#endif

template <typename S>
ComputeBlock<S, kNoDfb>::ComputeBlock(Block<S> block) : dfb_id(block.dfb_id) {
    block.consume();
#if defined(TT_U_ALIASING_CHECKED)
    claim_compute_block(dfb_id);
#endif
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    buffer(dfb_id).wait_front(num_entries);
#endif
}

template <typename S>
ComputeBlock<S, kNoDfb>::~ComputeBlock() {
#if defined(TT_U_ALIASING_CHECKED)
    release_compute_block(dfb_id);
#endif
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    buffer(dfb_id).pop_front(num_entries);
#endif
}

template <typename S>
TileSource<S> as_node(const ComputeBlock<S>& b) {
    return TileSource<S>{{}, b.get_dfb_id()};
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
    return Broadcast<A, S>{v.get_dfb_id()};
}

template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Sum, expr::UnaryChain<>> reduce_sum(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler) {
    return {{}, b.get_dfb_id(), scaler.get_dfb_id()};
}

template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Max, expr::UnaryChain<>> reduce_max(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler) {
    return {{}, b.get_dfb_id(), scaler.get_dfb_id()};
}

template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Avg, expr::UnaryChain<>> reduce_mean(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler) {
    return {{}, b.get_dfb_id(), scaler.get_dfb_id()};
}

template <int thread, typename S>
NocAsyncReadTx<thread, S>::NocAsyncReadTx(const Storage<S>& storage) : dfb_id(storage.dfb_id) {}

template <int thread, typename S>
NocAsyncReadTx<thread, S>::NocAsyncReadTx(uint32_t dfb_id) : dfb_id(dfb_id) {}

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
        Noc(noc_id).async_read_barrier();
        buffer(dfb_id).push_back(num_entries);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    return Block<S>(dfb_id);
}

template <int thread, typename S>
NocAsyncMcastTx<thread, S>::NocAsyncMcastTx(const Storage<S>& storage, uint32_t data_sent_id, bool sender) :
    dfb_id(storage.dfb_id), data_sent(data_sent_id), sender(sender) {}

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
template <int thread, typename S>
NocAsyncMcastTx<thread, S>::~NocAsyncMcastTx() {
    if constexpr (thread == TT_DM_THREAD_ID) {
        ASSERT(waited);
    }
}
#endif

template <int thread, typename S>
Block<S> NocAsyncMcastTx<thread, S>::wait() const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        Noc(noc_id).async_read_barrier();
        buffer(dfb_id).push_back(num_entries);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    return Block<S>(dfb_id);
}

#if defined(IS_DM_THREAD) && IS_DM_THREAD
namespace detail {
FORCE_INLINE void release_writes(uint8_t noc_id) {
    if constexpr (NOC_MODE == DM_DYNAMIC_NOC) {
        Noc(noc_id).async_write_barrier();
    } else {
        Noc(noc_id).async_writes_flushed();
    }
}
}  // namespace detail
#endif

template <int thread, typename S>
NocAsyncWriteTx<thread, S>::NocAsyncWriteTx(const Storage<S>& storage) : dfb_id(storage.dfb_id) {}

template <int thread, typename S>
NocAsyncWriteTx<thread, S>::NocAsyncWriteTx(uint32_t dfb_id) : dfb_id(dfb_id) {}

template <int thread, typename S>
NocAsyncWriteTx<thread, S>::~NocAsyncWriteTx() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        detail::release_writes(noc_id);
        buffer(dfb_id).pop_front(num_entries);
    }
#endif
}

template <int thread, typename S>
void NocAsyncWriteTx<thread, S>::wait() const {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        Noc(noc_id).async_write_barrier();
    }
#endif
}

template <int thread, typename D, typename S>
NocAsyncReadCoreTx<thread, D, S>::NocAsyncReadCoreTx(const Storage<D>& dst, const Block<S>& src) :
    dst_dfb(dst.dfb_id), src_dfb(src.dfb_id) {}

template <int thread, typename D, typename S>
NocAsyncReadCoreTx<thread, D, S>::~NocAsyncReadCoreTx() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        buffer(src_dfb).pop_front(src_entries);
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
        Noc(noc_id).async_read_barrier();
        buffer(dst_dfb).push_back(dst_entries);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    return Block<D>(dst_dfb);
}

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S>::NocAsyncWriteCoreTx(
    const Storage<D>& dst, const Block<S>& src, PhysicalMcast dst_range, uint32_t semaphore_id) :
    NocAsyncWriteCoreTx(dst, src, dst_range.contains(PhysicalCoord::this_core()), semaphore_id) {}

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S>::NocAsyncWriteCoreTx(
    const Storage<D>& dst, const Block<S>& src, bool reader, uint32_t semaphore_id) :
    dst_dfb(dst.dfb_id), src_dfb(src.dfb_id), arrived(semaphore_id), reader(reader) {}

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S>::~NocAsyncWriteCoreTx() {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        detail::release_writes(noc_id);
        Noc(noc_id).async_atomic_barrier();
        buffer(src_dfb).pop_front(src_entries);
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
            arrived.wait(num_writers).set(0);
        }
        buffer(dst_dfb).push_back(dst_entries);
    }
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    waited = true;
#endif
#endif
    (void)num_writers;
    return Block<D>(dst_dfb);
}

template <int T, uint32_t Id, typename S>
Block<S> fill_reduce_scaler(const Input<T, Id, S>& scaler, uint32_t value_bits) {
    return fill_reduce_scaler<T>(static_cast<const Storage<S>&>(scaler), value_bits);
}

template <int thread, typename S>
Block<S> fill_reduce_scaler(const Storage<S>& scaler, uint32_t value_bits) {
    static_assert(same_shape_v<S, Shape<1, 1>>, "a reduce scaler is exactly one tile -- Shape<1, 1>");
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        buffer(scaler.dfb_id).reserve_back(1);

        const uint32_t words = dfb_entry_bytes(scaler.dfb_id) / sizeof(uint32_t);
        volatile tt_l1_ptr uint32_t* page =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer(scaler.dfb_id).get_write_ptr());

        for (uint32_t w = 0; w < words; ++w) {
            page[w] = 0;
        }

        const uint32_t face_words = words / 4;
        const uint32_t row_words = face_words / 16;
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t w = 0; w < row_words; ++w) {
                page[f * face_words + w] = value_bits;
            }
        }

        buffer(scaler.dfb_id).push_back(1);
    }
#else
    (void)value_bits;
#endif
    return Block<S>(scaler.dfb_id);
}

namespace detail {

template <typename Accessor>
inline void check_entry_format(uint32_t dfb_id, const Accessor& acc) {
    ASSERT(dfb_entry_bytes(dfb_id) == acc.get_aligned_page_size());
}

}  // namespace detail

template <int thread, typename S, typename Accessor>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, const Accessor& acc, uint32_t block_idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    detail::check_entry_format(storage.dfb_id, acc);
#endif
    const uint32_t first = block_idx * storage.num_entries;
    return noc_load<thread>(storage, [&](L1Entries pages) {
        for (uint32_t p = 0; p < pages.count; ++p) {
            noc_async_read(acc.get_noc_addr(first + p), pages.addr(p), pages.entry_bytes);
        }
    });
}

template <int T, uint32_t Id, typename S, typename Accessor>
NocAsyncReadTx<T, S> noc_load(const Input<T, Id, S>& storage, const Accessor& acc, uint32_t block_idx) {
    return noc_load<T>(static_cast<const Storage<S>&>(storage), acc, block_idx);
}

namespace detail {

template <int thread, typename S, typename Fn>
inline void issue_load(const Storage<S>& storage, Fn fn) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        {
            TT_U_ZONE("LOAD-RESERVE");
            buffer(storage.dfb_id).reserve_back(storage.num_entries);
        }
        {
            TT_U_ZONE("LOAD-ISSUE");
            fn(L1Entries{buffer(storage.dfb_id).get_write_ptr(), dfb_entry_bytes(storage.dfb_id), storage.num_entries});
        }
    }
#else
    (void)storage;
    (void)fn;
#endif
}

}  // namespace detail

template <int thread, typename S, typename Fn>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, Fn fn) {
    detail::issue_load<thread>(storage, fn);
    return NocAsyncReadTx<thread, S>(storage);
}

template <int T, uint32_t Id, typename S, typename Fn>
NocAsyncReadTx<T, S> noc_load(const Input<T, Id, S>& storage, Fn fn) {
    return noc_load<T>(static_cast<const Storage<S>&>(storage), fn);
}

template <int thread, typename S, typename Accessor>
NocAsyncWriteTx<thread, S> noc_store(Block<S> block, const Accessor& acc, uint32_t block_idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    detail::check_entry_format(block.dfb_id, acc);
#endif
    const uint32_t first = block_idx * block.num_entries;
    return noc_store<thread>(std::move(block), [&](L1Entries pages) {
        for (uint32_t p = 0; p < pages.count; ++p) {
            noc_async_write(pages.addr(p), acc.get_noc_addr(first + p), pages.entry_bytes);
        }
    });
}

template <int thread, typename S, typename Fn>
NocAsyncWriteTx<thread, S> noc_store(Block<S> block, Fn fn) {
    block.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        buffer(block.dfb_id).wait_front(block.num_entries);
        fn(L1Entries{buffer(block.dfb_id).get_read_ptr(), dfb_entry_bytes(block.dfb_id), block.num_entries});
    }
#else
    (void)fn;
#endif
    return NocAsyncWriteTx<thread, S>(block.dfb_id);
}

template <int thread, typename S, typename Accessor, typename Node>
NocAsyncWriteTx<thread, S> noc_store(
    const Storage<S>& storage, const Node& node, const Accessor& acc, uint32_t block_idx) {
    return noc_store<thread>(storage.store(node), acc, block_idx);
}

template <int T, uint32_t Id, typename S, typename Accessor, typename Node>
NocAsyncWriteTx<T, S> noc_store(
    const Output<T, Id, S>& storage, const Node& node, const Accessor& acc, uint32_t block_idx) {
    return noc_store<T>(static_cast<const Storage<S>&>(storage), node, acc, block_idx);
}

template <int thread, typename S, typename Node, typename Fn>
NocAsyncWriteTx<thread, S> noc_store(const Storage<S>& storage, const Node& node, Fn fn) {
    return noc_store<thread>(storage.store(node), fn);
}

template <int T, uint32_t Id, typename S, typename Node, typename Fn>
NocAsyncWriteTx<T, S> noc_store(const Output<T, Id, S>& storage, const Node& node, Fn fn) {
    return noc_store<T>(static_cast<const Storage<S>&>(storage), node, fn);
}

template <int thread, typename S, typename Fn>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage,
    PhysicalMcast mcast,
    Semaphore<thread>& receivers_ready,
    Semaphore<thread>& data_sent,
    Fn fn) {
    bool is_sender = false;
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        is_sender = (PhysicalCoord::this_core() == mcast.start);
    }
#endif

    detail::issue_load<thread>(storage, [&](L1Entries pages) {
        const uint32_t num_dests = mcast.volume() - 1;

        if (PhysicalCoord::this_core() == mcast.start) {
            fn(pages);

            if (num_dests == 0) {
                Noc().async_read_barrier();
                return;
            }

            {
                TT_U_ZONE("MCAST-READY");
                receivers_ready.wait(num_dests);
                receivers_ready.set(0);
            }

            {
                TT_U_ZONE("MCAST-DRAM");
                Noc().async_read_barrier();
            }

            {
                TT_U_ZONE("MCAST-SEND");
                noc_async_write_multicast(pages.base, mcast.get_noc_addr(pages.base), pages.total_bytes(), num_dests);

                Noc().async_writes_flushed();

                data_sent.set(1);
                data_sent.set_mcast(mcast);

                Noc().async_writes_flushed();
                data_sent.set(0);
            }
        } else {
            receivers_ready.inc_remote(mcast.start);
            data_sent.wait(1);
            data_sent.set(0);
        }
    });
    return NocAsyncMcastTx<thread, S>(storage, data_sent.semaphore_id(), is_sender);
}

template <int thread, typename S, typename Accessor>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage,
    PhysicalMcast mcast,
    Semaphore<thread>& receivers_ready,
    Semaphore<thread>& data_sent,
    const Accessor& acc,
    uint32_t block_idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    detail::check_entry_format(storage.dfb_id, acc);
#endif
    const uint32_t first = block_idx * storage.num_entries;
    return noc_load<thread>(storage, mcast, receivers_ready, data_sent, [&](L1Entries pages) {
        for (uint32_t p = 0; p < pages.count; ++p) {
            noc_async_read(acc.get_noc_addr(first + p), pages.addr(p), pages.entry_bytes);
        }
    });
}

template <int thread, typename S, typename Accessor>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage,
    LogicalMcast mcast,
    Semaphore<thread>& receivers_ready,
    Semaphore<thread>& data_sent,
    const Accessor& acc,
    uint32_t block_idx) {
    return noc_load<thread>(storage, mcast.to_physical(), receivers_ready, data_sent, acc, block_idx);
}

template <int thread, int pair, typename S, typename Accessor>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage, PhysicalMcast mcast, const Accessor& acc, uint32_t block_idx) {
    static_assert(
        kMcastSemsReserved,
        "multicast needs its handshake semaphores reserved by the host: build the program through "
        "unified_program(), which reserves them and defines TT_UNIFIED_MCAST_SEM_BASE -- or pass your own pair "
        "to the six-argument noc_load()");
    static_assert(
        pair < 2,
        "there are only two multicast handshake pairs (the host reserves two per thread plus the "
        "copy flags); pair 2 would alias the noc_core_write arrival semaphore");
    Semaphore<thread> receivers_ready(kMcastReadySem<pair>);
    Semaphore<thread> data_sent(kMcastSentSem<pair>);
    return noc_load<thread>(storage, mcast, receivers_ready, data_sent, acc, block_idx);
}

template <int pair, int T, uint32_t Id, typename S, typename Accessor>
NocAsyncMcastTx<T, S> noc_load(
    const Input<T, Id, S>& storage, PhysicalMcast mcast, const Accessor& acc, uint32_t block_idx) {
    constexpr int p = (pair == kPairOfThread) ? T : pair;
    return noc_load<T, p>(static_cast<const Storage<S>&>(storage), mcast, acc, block_idx);
}

template <int thread, int pair, typename S, typename Accessor>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage, LogicalMcast mcast, const Accessor& acc, uint32_t block_idx) {
    return noc_load<thread, pair>(storage, mcast.to_physical(), acc, block_idx);
}

template <int pair, int T, uint32_t Id, typename S, typename Accessor>
NocAsyncMcastTx<T, S> noc_load(
    const Input<T, Id, S>& storage, LogicalMcast mcast, const Accessor& acc, uint32_t block_idx) {
    constexpr int p = (pair == kPairOfThread) ? T : pair;
    return noc_load<T, p>(static_cast<const Storage<S>&>(storage), mcast, acc, block_idx);
}

template <int thread, int pair, typename S, typename Fn>
NocAsyncMcastTx<thread, S> noc_load(const Storage<S>& storage, PhysicalMcast mcast, Fn fn) {
    static_assert(
        kMcastSemsReserved,
        "multicast needs its handshake semaphores reserved by the host: build the program through "
        "unified_program(), which reserves them and defines TT_UNIFIED_MCAST_SEM_BASE -- or pass your own pair "
        "to the five-argument noc_load()");
    static_assert(
        pair < 2,
        "there are only two multicast handshake pairs (the host reserves two per thread plus the "
        "copy flags); pair 2 would alias the noc_core_write arrival semaphore");
    Semaphore<thread> receivers_ready(kMcastReadySem<pair>);
    Semaphore<thread> data_sent(kMcastSentSem<pair>);
    return noc_load<thread>(storage, mcast, receivers_ready, data_sent, fn);
}

template <int pair, int T, uint32_t Id, typename S, typename Fn>
NocAsyncMcastTx<T, S> noc_load(const Input<T, Id, S>& storage, PhysicalMcast mcast, Fn fn) {
    constexpr int p = (pair == kPairOfThread) ? T : pair;
    return noc_load<T, p>(static_cast<const Storage<S>&>(storage), mcast, fn);
}

template <int thread, int pair, typename S, typename Fn>
NocAsyncMcastTx<thread, S> noc_load(const Storage<S>& storage, LogicalMcast mcast, Fn fn) {
    return noc_load<thread, pair>(storage, mcast.to_physical(), fn);
}

template <int pair, int T, uint32_t Id, typename S, typename Fn>
NocAsyncMcastTx<T, S> noc_load(const Input<T, Id, S>& storage, LogicalMcast mcast, Fn fn) {
    constexpr int p = (pair == kPairOfThread) ? T : pair;
    return noc_load<T, p>(static_cast<const Storage<S>&>(storage), mcast, fn);
}

namespace detail {

template <typename Tuple, std::size_t... I>
constexpr bool leading_all_compute_blocks(std::index_sequence<I...>) {
    return (is_compute_block<std::decay_t<std::tuple_element_t<I, Tuple>>>::value && ...);
}

template <typename Tuple, typename Fn, std::size_t... I>
void custom_compute_invoke(Tuple& packed, Fn& fn, std::index_sequence<I...>) {
    fn(std::get<I>(packed).get_dfb_id()...);
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
    (void)packed;
#endif
}

template <int thread>
uint32_t Semaphore<thread>::semaphore_id() const {
    return id;
}

template <int thread>
void synchronize_cores(PhysicalMcast region) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        static_assert(
            kMcastSemsReserved,
            "synchronize_cores() needs the reserved handshake semaphores: build the program through "
            "unified_program(), which reserves them and defines TT_UNIFIED_MCAST_SEM_BASE");

        Semaphore<thread> arrived(kMcastReadySem<thread>);
        Semaphore<thread> release(kMcastSentSem<thread>);

        const uint32_t others = region.volume() - 1;
        if (others == 0) {
            return;
        }

        if (PhysicalCoord::this_core() == region.start) {
            arrived.wait(others);

            arrived.set(0);

            release.set(1);
            release.set_mcast(region);
            Noc().async_writes_flushed();
            release.set(0);
        } else {
            arrived.inc_remote(region.start);
            Noc().async_atomic_barrier();
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

template <int thread, typename D, typename S>
NocAsyncReadCoreTx<thread, D, S> noc_core_read(
    const Storage<D>& dst, Block<S> src, PhysicalCoord coord, uint32_t byte_offset) {
    src.consume();
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        buffer(src.dfb_id).wait_front(src.num_entries);
        buffer(dst.dfb_id).reserve_back(dst.num_entries);
        const uint32_t bytes = dfb_entry_bytes(dst.dfb_id);
        const uint64_t from = coord.get_noc_addr(buffer(src.dfb_id).get_read_ptr() + byte_offset);
        noc_async_read(from, buffer(dst.dfb_id).get_write_ptr(), bytes * dst.num_entries);
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
        buffer(src.dfb_id).wait_front(src.num_entries);
        buffer(dst.dfb_id).reserve_back(dst.num_entries);
        if (write_predicate) {
            const uint32_t bytes = dfb_entry_bytes(dst.dfb_id);
            const uint64_t to = coord.get_noc_addr(buffer(dst.dfb_id).get_write_ptr() + byte_offset);
            noc_async_write(buffer(src.dfb_id).get_read_ptr(), to, bytes * src.num_entries);

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
        buffer(src.dfb_id).wait_front(src.num_entries);
        buffer(dst.dfb_id).reserve_back(dst.num_entries);

        if (write_predicate) {
            const uint32_t bytes = dfb_entry_bytes(dst.dfb_id);
            const uint64_t to = mcast.get_noc_addr(buffer(dst.dfb_id).get_write_ptr() + byte_offset);

            const bool same_local_addr = buffer(dst.dfb_id).get_write_ptr() == buffer(src.dfb_id).get_read_ptr();
            const bool loopback = !same_local_addr && mcast.contains(PhysicalCoord::this_core());

            const uint32_t num_dests =
                loopback ? mcast.volume() : mcast.num_dests_excluding(PhysicalCoord::this_core());

            if (loopback) {
                noc_async_write_multicast_loopback_src(
                    buffer(src.dfb_id).get_read_ptr(), to, bytes * src.num_entries, num_dests);
            } else {
                noc_async_write_multicast(buffer(src.dfb_id).get_read_ptr(), to, bytes * src.num_entries, num_dests);
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

template <int T, uint32_t Id, typename D, typename S>
NocAsyncWriteCoreTx<T, D, S> noc_core_write(
    const Input<T, Id, D>& dst, Block<S> src, LogicalCoord coord, bool write_predicate, uint32_t byte_offset) {
    return noc_core_write<T>(static_cast<const Storage<D>&>(dst), std::move(src), coord, write_predicate, byte_offset);
}

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S> noc_core_write(
    const Storage<D>& dst, Block<S> src, LogicalMcast mcast, bool write_predicate, uint32_t byte_offset) {
    return noc_core_write<thread>(dst, std::move(src), mcast.to_physical(), write_predicate, byte_offset);
}

}  // namespace unified
}  // namespace tt
