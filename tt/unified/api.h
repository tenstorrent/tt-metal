// SPDX-License-Identifier: Apache-2.0
//
// Core API of the unified programming model -- declarations only.
//
// A unified kernel is ONE source describing a whole Tensix pipeline. It is
// compiled once per baby RISC-V thread, and each statement below lowers to that
// thread's half of the circular-buffer protocol:
//
//   INPUT                    OUTPUT                   INTERMED
//        DM    Compute            DM    Compute            DM    Compute
//   reserve <- *               * -> reserve                   reserve
//     write                          write                      write
//      push ->    wait         wait <-  push                     push
//                read          read                              wait
//         * <-     pop          pop -> *                         read
//                                                                 pop
//
// Include <tt/unified/core>, not this header directly -- it selects an implementation
// and a backend binding, and documents the layering.

#pragma once

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <tt/unified/math.hpp>

namespace tt {
namespace unified {

template <typename S>
struct Block;
template <typename S>
class ComputeBlock;

// ---------------------------------------------------------------------------
// Geometry
//
// Core coordinates come in two flavours, kept as distinct types so the
// translation between them is explicit and checked. LOGICAL is what the host
// reasons about (0,0 is the first worker of the program's core range); PHYSICAL
// is what the NOC addresses. Mixing them silently targets the wrong core, which
// is why there is no implicit conversion.
//
// Members that touch NOC state are defined in the implementation header behind a
// data-movement guard: my_x/my_y, the logical->virtual tables and get_noc_addr
// are all data-movement-only names.
// ---------------------------------------------------------------------------

// Both coordinate types are built by NAME, never by position: LogicalCoord::xy(x, y)
// and LogicalCoord::yx(y, x) construct the same thing from arguments in opposite orders,
// and the call says which one you meant.
//
// The reason is that the two conventions in play disagree. Metal writes coordinates x
// first -- get_noc_addr(x, y), CoreCoord(x, y) -- while these structs store y first, and
// tensor code everywhere else (torch, and every framework that follows it) is row-major
// and reads y first too. A bare pair is therefore ambiguous to a reader and silently
// wrong when guessed, and the failure is not a crash: a multicast rectangle addressed
// through transposed corners still runs, on the wrong cores.
//
// The constructors are private so there is no positional form to guess at. That also
// makes these non-aggregates, so `LogicalCoord{1, 2}` no longer compiles -- which is the
// point, since that spelling was the ambiguous one.
struct PhysicalCoord {
    uint32_t y;
    uint32_t x;

    static constexpr PhysicalCoord yx(uint32_t y, uint32_t x) { return PhysicalCoord(y, x); }
    static constexpr PhysicalCoord xy(uint32_t x, uint32_t y) { return PhysicalCoord(y, x); }

    // This core's own physical coordinate, on this thread's NOC.
    //
    // DATA MOVEMENT ONLY. On a compute projection this returns the ORIGIN, not the
    // real coordinate: my_x/my_y are filled by risc_init(), which does not run on a
    // TRISC (risc_common.h guards it out), so metal gives compute no way to know
    // where it is. The consequence is a quiet one -- a statement whose COMPUTE side
    // branches on this behaves as though every core were (0,0), so circular-buffer
    // traffic guarded by it happens everywhere and the pushes go unmatched. Gate
    // NOC work on it, never CB work; LogicalCoord::this_core() is the one that
    // every projection answers correctly.
    static PhysicalCoord this_core();
    static PhysicalCoord origin();

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    // constexpr like the factories: a coordinate can now be built at compile time, so
    // comparing two of them should be answerable there as well.
    constexpr bool operator==(PhysicalCoord o) const { return y == o.y && x == o.x; }
    constexpr bool operator!=(PhysicalCoord o) const { return !(*this == o); }

private:
    // explicit as well as private: inside the class, `return {y, x}` would otherwise
    // still reach this constructor by list-initialisation, which is the positional form
    // the factories exist to remove. Now even the implementation has to name an order.
    explicit constexpr PhysicalCoord(uint32_t y_in, uint32_t x_in) : y(y_in), x(x_in) {}
};

struct LogicalCoord {
    uint32_t y;
    uint32_t x;

    // See PhysicalCoord above for why these are named rather than positional.
    static constexpr LogicalCoord yx(uint32_t y, uint32_t x) { return LogicalCoord(y, x); }
    static constexpr LogicalCoord xy(uint32_t x, uint32_t y) { return LogicalCoord(y, x); }

    // Correct on ALL projections, unlike PhysicalCoord::this_core(): compute is
    // told its logical position by the firmware even though it cannot resolve it
    // to a NOC address. Branch on this one, including in code compute runs.
    static LogicalCoord this_core();
    static LogicalCoord origin();

    PhysicalCoord to_physical(uint32_t y_offset = 0, uint32_t x_offset = 0) const;

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    // constexpr like the factories: a coordinate can now be built at compile time, so
    // comparing two of them should be answerable there as well.
    constexpr bool operator==(LogicalCoord o) const { return y == o.y && x == o.x; }
    constexpr bool operator!=(LogicalCoord o) const { return !(*this == o); }

private:
    // explicit as well as private: inside the class, `return {y, x}` would otherwise
    // still reach this constructor by list-initialisation, which is the positional form
    // the factories exist to remove. Now even the implementation has to name an order.
    explicit constexpr LogicalCoord(uint32_t y_in, uint32_t x_in) : y(y_in), x(x_in) {}
};

// The h x w extent of a core rectangle. Not a tile shape -- see Shape.
// Built by name for the same reason the coordinates are, and the confusion is the same
// one: this holds h before w, while metal's grid sizes go the other way -- a CoreCoord or
// compute_with_storage_grid_size is (x, y), meaning (w, h). hw(h, w) and wh(w, h) build the
// same extent from arguments in opposite orders, and the call says which was meant.
//
// A transposed extent is another silent failure: a 1 x N multicast row addressed as N x 1
// covers a column instead, which still runs and still writes, just to the wrong cores.
struct Extent {
    uint32_t h;
    uint32_t w;

    static constexpr Extent hw(uint32_t h, uint32_t w) { return Extent(h, w); }
    static constexpr Extent wh(uint32_t w, uint32_t h) { return Extent(h, w); }

    constexpr bool operator==(Extent o) const { return h == o.h && w == o.w; }
    constexpr bool operator!=(Extent o) const { return !(*this == o); }

private:
    explicit constexpr Extent(uint32_t h_in, uint32_t w_in) : h(h_in), w(w_in) {}
};

// A multicast rectangle, inclusive of both corners.
struct PhysicalMcast {
    PhysicalCoord start;
    PhysicalCoord end;

    // Declaring any constructor costs PhysicalMcast its aggregate status, so the
    // two-corner form has to be spelled out rather than left to brace-init.
    PhysicalMcast(PhysicalCoord start, PhysicalCoord end) : start(start), end(end) {}

    // Implicit: a single core is a 1x1 rectangle, which is what lets the unicast
    // noc_core_write hand its PhysicalCoord straight to the handle.
    PhysicalMcast(PhysicalCoord unit) : start(unit), end(unit) {}

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    uint32_t volume() const { return (end.y - start.y + 1) * (end.x - start.x + 1); }

    bool contains(PhysicalCoord c) const { return c.y >= start.y && c.y <= end.y && c.x >= start.x && c.x <= end.x; }

    // Destination counts. Metal's multicast primitives exclude the sender unless
    // NocOptions::MCAST_INCL_SRC is set, and num_dests must exclude it on exactly
    // the same terms -- but only when the sender is in range at all.
    //
    // Which to use depends on where the issuer sits, and the two cases are real:
    //
    //   ..._sender()  -- the issuer IS `start`. The handshake paths elect their
    //                    sender with `this_core() == start`, so containment is a
    //                    known fact and the count is a constant.
    //
    //   ..._excluding -- the issuer is wherever it is. A core pushing a block to
    //                    a rectangle is usually OUTSIDE it, in which case every
    //                    core in the rectangle is a destination. See noc_core_write.
    uint32_t num_dests_excluding_sender() const { return volume() - 1; }
    uint32_t num_dests_excluding(PhysicalCoord sender) const { return volume() - (contains(sender) ? 1 : 0); }
};

struct LogicalMcast {
    LogicalCoord coord;
    Extent extent;

    PhysicalMcast to_physical() const;

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    uint32_t volume() const { return extent.h * extent.w; }
};

// ---------------------------------------------------------------------------
// Storage -- a circular buffer
// ---------------------------------------------------------------------------

template <typename S>
struct Storage {
    using shape = S;

    explicit Storage(uint32_t cb_id) : cb_id(cb_id) {
        // The host sizes the circular buffer and the kernel names the Shape, and until
        // here nothing made the two meet. A buffer smaller than one block cannot ever
        // satisfy cb_reserve_back, so the kernel does not fail -- it waits forever, with
        // no assert and no output, and the device needs a reset. Checking it where the
        // Storage is built turns that into a stop at a source line.
        //
        // Greater-or-equal, not equal: a deeper buffer is how a reader runs ahead of
        // compute, and every prefetch depth in this work is exactly that.
        //
        // Assertion-only, and asserts are compiled out unless WATCHER_ENABLED or
        // LIGHTWEIGHT_KERNEL_ASSERTS is set -- see unified_api_hazards.md, which is why
        // the test harness turns them on.
        //
        // Data movement only: cb_interface does not link on a TRISC, so a live read of it
        // from a compute projection fails the build. One thread is enough -- the host
        // configures one circular buffer for the core, so every projection would be
        // checking the same number.
#if defined(IS_DM_THREAD) && IS_DM_THREAD
        ASSERT(cb_num_pages(cb_id) >= S::num_pages);
#endif
    }

    Storage(Storage&&) = delete;
    Storage(const Storage&) = delete;
    Storage& operator=(Storage&&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Evaluate a compute fusion into this buffer. The loop shape is chosen by the
    // fusion's kind; see Strategy in tt/unified/math.hpp.
    template <typename Node>
    Block<S> store(const Node& node);

    uint32_t cb_id;
    // PAGES, which is what the circular-buffer protocol counts -- reserve, push,
    // wait and pop are all in pages, and cb_page_bytes() sizes one. The compute
    // strategies then walk it as a TILE count, which holds only because this model
    // configures one tile per page; see Storage::store.
    //
    // Static now that the shape is: reading it off an instance still compiles, so
    // every `storage.num_pages` in the implementation is unchanged.
    static constexpr uint32_t num_pages = S::num_pages;
};

// This core's own pages, handed to a custom load or store routine. The harness has
// already reserved them (or waited on them), so these are the facts the routine
// cannot work out for itself: where the block starts now that the pointer has
// advanced, and how big a page is.
//
// L1, and named for it, because a routine juggles TWO page spaces at once and they
// are easy to confuse. A TensorAccessor's pages are indices into a tensor that
// mostly lives somewhere else; these are addresses in this core's L1:
//
//     noc_async_read(acc.get_noc_addr(tensor_page), pages.addr(i), pages.page_bytes);
//                                     ^ index, remote   ^ address, local
//
// `count` is the number the handle will push or pop whatever the routine actually
// touches, so looping on it -- rather than on a tile count the kernel re-derives
// from a compile-time arg -- is what keeps the two in step.
struct L1Pages {
    uint32_t base;
    uint32_t page_bytes;
    uint32_t count;

    // Address of page `i`, so a routine never writes the stride arithmetic itself.
    uint32_t addr(uint32_t i) const { return base + i * page_bytes; }

    uint32_t total_bytes() const { return count * page_bytes; }
};

// ---------------------------------------------------------------------------
// Block -- move-only evidence that a Storage was produced into
//
// Every Block comes from an operation that has already pushed, which is what
// makes it safe to hand one to a DM thread to drain. Move-only so it reaches
// exactly one consumer; consumers take it by value.
// ---------------------------------------------------------------------------

template <typename S, AccumulatorMode Mode = AccumulatorMode::Dst>
class Accumulator;

template <typename S>
struct Block {
    using shape = S;

    explicit Block(const Storage<S>& storage);
    explicit Block(uint32_t cb_id);
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ~Block();
#endif

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    Block(Block&& o);
    Block& operator=(Block&& o);

    // Part of the CONSUMER contract, not user API: every consumer that takes a
    // Block by value must call this once, to record that the pages were really
    // used. The destructor asserts on a Block that owed consumption and never got
    // it, which is how a dropped output block is caught.
    //
    // It cannot be folded into the move, because C++17 guaranteed elision means a
    // prvalue handed straight to a by-value parameter initializes it directly and
    // no move ever runs. Only the consumer knows consumption happened.
    //
    // Compiles to nothing when asserts are off.
    void consume();

    uint32_t cb_id;
    static constexpr uint32_t num_pages = S::num_pages;

private:
    // A RETAINED block: one the Accumulator hands back mid-accumulation. Its pages
    // still belong to the accumulator, so it must neither be transferred to
    // another thread nor consumed -- only the next accumulate() call may touch
    // them. Only Accumulator can make one.
    struct Retained {};
    Block(const Storage<S>& storage, Retained);

    template <typename S2, AccumulatorMode M>
    friend class Accumulator;

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    // Two independent facts, deliberately not folded into one flag:
    //   must_consume -- this Block owes a consumer (false for retained blocks)
    //   consumed     -- a consumer has taken it
    // A moved-from Block has must_consume=false and consumed=true, so it is silent
    // at destruction and asserts if used again.
    bool must_consume = true;
    bool consumed = false;

    // Poison stamped into a moved-from Block's cb_id. num_pages is part of the
    // type now, so there is one field left to poison -- which is enough, since any
    // use of a moved-from Block goes through cb_id.
    static constexpr uint32_t kMovedFrom = ~uint32_t(0);
#endif
};

// ---------------------------------------------------------------------------
// RetainedBlock -- a Block that outlives the statement that produced it
//
// For a value carried across a loop: written in one iteration, read in the next. The running
// maximum, sum and output of an online softmax are the case this exists for.
//
// A Block cannot simply be left lying around. It owes a consumer, and ~Block asserts if it
// never reached one, which is how a dropped output is caught. A ComputeBlock is not the
// answer either: it waits in its constructor and POPS in its destructor, so using the value
// you just wrote also consumes it -- the state is gone before the next iteration looks for
// it, and on device that is a hang.
//
// So the obligation is MOVED here rather than discharged. That distinction is the whole
// point: a `retain(block)` that just called consume() would switch off the very check worth
// keeping, whereas ~RetainedBlock asserting that it is empty says "you pushed pages and
// nobody ever waited on them" -- the same diagnostic, relocated.
//
//     RetainedBlock<Vec> m;                       // OUTSIDE the loop: that is the lifetime
//     for (uint32_t j = 0; j < chunks; ++j) {
//         if (j == 0) {
//             m = m_storage.store(first(...));    // moves the obligation into the slot
//         } else {
//             ComputeBlock<Vec> prev = m.release();   // the consumer: waits, then pops
//             m = m_storage.store(update(prev, ...));
//         }
//     }
//
// Costs nothing in a release build: every member below is assertion-only, so the slot is
// byte-for-byte a Block.
// ---------------------------------------------------------------------------

template <typename S>
class RetainedBlock {
public:
    using Held = Block<S>;

    // The premise that lets the occupancy flag and the destructor be assertion-only: with
    // assertions off Block has no user-declared destructor, so there is nothing to run and
    // nothing to track. If Block ever acquires a resource this breaks the build rather than
    // leaking quietly.
#if !(defined(ASSERT_ENABLED) && ASSERT_ENABLED)
    static_assert(
        std::is_trivially_destructible<Held>::value,
        "RetainedBlock does not destroy what it holds in a release build, because Block's "
        "destructor is assertion-only. Block now has a real one, so this has to track "
        "occupancy in every build.");
#endif

    RetainedBlock() = default;
    explicit RetainedBlock(Held&& block);

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ~RetainedBlock();
#endif

    // A slot is a fixed place, not a value: it names where the state lives for the whole loop.
    RetainedBlock(const RetainedBlock&) = delete;
    RetainedBlock& operator=(const RetainedBlock&) = delete;
    RetainedBlock(RetainedBlock&&) = delete;
    RetainedBlock& operator=(RetainedBlock&&) = delete;

    // Take ownership of a freshly stored block. The MOVE is the mechanism: it transfers the
    // obligation and leaves the source silent. Copying the cb id into a second Block would
    // leave two obligations for one push, and the source would then assert as it died.
    RetainedBlock& operator=(Held&& in);

    // Hand the block to its consumer, leaving the slot empty. By value, so the caller
    // move-constructs from the returned temporary.
    Held release();

private:
    void emplace(Held&& in);
    Held& get();

    // Manual storage because Block has no default constructor -- which is also what lets the
    // slot exist before the first value does.
    alignas(Held) unsigned char buf[sizeof(Held)];

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    bool held = false;
#endif
};

// ---------------------------------------------------------------------------
// Accumulator -- multi-block matmul
//
// The k-loop belongs to the kernel, because the operand CBs have to be waited and
// popped per block so the reader can stream them. The Accumulator holds the state
// that loop would otherwise carry: which buffer is the running total, which is
// the destination, and whether there is anything to reload yet.
//
//     Accumulator acc(partials_storage, out_storage);
//     for (uint32_t k = 0; k < Geom::num_blocks; ++k) {
//         ComputeBlock a = noc_load<1>(in0_storage, in0, k).wait();
//         ComputeBlock b = noc_load<1>(in1_storage, in1, k).wait();
//         Block result = acc.accumulate(matmul<Geom>(a, b), k == Geom::num_blocks - 1);
//         if (k == Geom::num_blocks - 1) noc_store<0>(std::move(result), out, 0);
//     }
//
// The two Storages must be DIFFERENT circular buffers. Intermediate blocks are
// pushed to the accumulation buffer and re-consumed by the next call; if that
// were also the output buffer, the DM writer would drain the first intermediate
// as though it were the answer, and two threads would be popping one CB (see the
// warning in api/compute/cb_api.h).
// ---------------------------------------------------------------------------

template <typename S, AccumulatorMode Mode>
class Accumulator {
public:
    using shape = S;

    Accumulator(const Storage<S>& acc_storage, const Storage<S>& out_storage);

    // Fold one k-block into the running total. `finish` selects the pack target:
    // the accumulation buffer, or the output buffer on the last block.
    //
    // The two ways of attaching SFPU work mean DIFFERENT things:
    //
    //   accumulate(relu(mm), finish)                          per-step: relu runs
    //     on every k-block, so the accumulator carries the transformed value.
    //
    //   accumulate(mm, finish, [](auto n){ return relu(n); })  finish-only: relu
    //     runs once, on the completed accumulator.
    //
    // The lambda receives the node and returns one with a longer chain; only the
    // ops *it* adds are deferred. See Strategy<FPUFusion>::run for what "per-step"
    // sees in each mode -- the contribution alone in L1 mode, the running total in
    // Dst mode.
    //
    // Only the Block returned on the finishing call is meaningful; earlier ones
    // describe the accumulation buffer, which the next call re-consumes.
    template <typename Node, typename Epilogue = std::nullptr_t>
    Block<S> accumulate(const Node& node, bool finish, Epilogue epilogue = nullptr);

    // Reset between output blocks.
    void clear();

private:
    const Storage<S>& acc_storage;
    const Storage<S>& out_storage;
    bool reload = false;
};

// ---------------------------------------------------------------------------
// ComputeBlock -- compute-side consumption of a Block, and an expression leaf
// ---------------------------------------------------------------------------

template <typename S>
class ComputeBlock : public expr::Fluent<ComputeBlock<S>> {
public:
    using shape = S;

    ComputeBlock(Block<S> block);
    ~ComputeBlock();

    ComputeBlock(const ComputeBlock&) = delete;
    ComputeBlock& operator=(const ComputeBlock&) = delete;
    ComputeBlock(ComputeBlock&&) = delete;
    ComputeBlock& operator=(ComputeBlock&&) = delete;

    uint32_t get_cb_id() const { return cb_id; }
    static constexpr uint32_t get_num_pages() { return S::num_pages; }

private:
    uint32_t cb_id;
    static constexpr uint32_t num_pages = S::num_pages;
};

// ---------------------------------------------------------------------------
// Adaptors letting a ComputeBlock stand in for an expression leaf. These are the
// hooks tt/unified/math.hpp declares; they live here because this is the only
// place the math layer needs to know about a core type.
// ---------------------------------------------------------------------------

// Without this the operator+ in tt/unified/math.hpp is SFINAE'd out and
// `lhs + rhs` does not resolve.
template <typename S>
struct is_operand<ComputeBlock<S>> : std::true_type {};

template <typename S>
TileSource<S> as_node(const ComputeBlock<S>& b);

// The identity expression: materialise a block into another buffer. `store` takes an
// expression, and sometimes the expression is just "this block" -- seeding a running value, or
// copying a scratch result into the buffer that will carry it to the next iteration.
//
// as_node does the same thing, but it is the hook the math layer reaches through rather than
// something a kernel should name.
template <typename S>
TileSource<S> copy(const ComputeBlock<S>& b);

template <typename S>
auto relu(const ComputeBlock<S>& b);
template <typename S>
auto silu(const ComputeBlock<S>& b);
template <typename S>
auto exp_(const ComputeBlock<S>& b);
template <typename S>
auto recip(const ComputeBlock<S>& b);
template <typename S>
auto sqrt_(const ComputeBlock<S>& b);
template <typename S>
auto rsqrt(const ComputeBlock<S>& b);

// The geometry is DERIVED from the operands -- see MatmulGeometry in
// tt/unified/math.hpp. A must be rt x kt tiles and B kt x ct, and their agreement on
// kt is a compile error rather than silent garbage.
//
// `Tr` selects whether B's TILES are read transposed. It is not a B-transpose on its
// own -- the tile grid is the reader's job -- and it must match the argument given to
// matmul_init. See TransposeB in tt/unified/math.hpp, which spells out both halves.
template <TransposeB Tr = TransposeB::No, typename SA, typename SB>
auto matmul(const ComputeBlock<SA>& a, const ComputeBlock<SB>& b);

// Mark a ComputeBlock as a BROADCAST operand along `A`, for use as the right-hand side
// of +, - or *:
//
//     u::ComputeBlock m = m_storage.store(u::reduce_max<u::Axis::Cols>(x, one));
//     e_storage.store((x - u::bcast<u::Axis::Cols>(m)).exp());          // exp(x - rowmax)
//
// The axis is DECLARED because a Shape counts tiles and cannot express it: one tile
// holding a row, a column, or a lone value at [0, 0] is Shape<1, 1> in every case. The
// vector's shape is then CHECKED against the axis, so the two carry different halves of
// the requirement and neither is guessed. See bcast_vec_shape in tt/unified/math.hpp.
//
// The same axis names a reduction's collapse, so a reduction and the broadcast undoing it
// agree by construction: reduce over Cols yields Shape<rows, 1>, which is exactly what
// bcast<Axis::Cols> demands.
//
// A ComputeBlock and not a Storage, for the reason reduce_* takes one: its constructor
// holds the cb_wait_front that makes reading the buffer legal, and its destructor holds
// the matching pop. Hold it at the scope the vector must survive -- for a resident vector
// re-read by every block, that is kernel scope.
template <Axis A, typename S>
Broadcast<A, S> bcast(const ComputeBlock<S>& v);

// Reduce `b`'s tile grid down one axis, within and across tiles. `Axis` says which
// dimension collapses; the input grid comes from `b`'s own shape -- see ReduceAxis in
// tt/unified/math.hpp for what each axis leaves behind.
//
//     using In = u::Shape<4, 4>;
//     u::Storage<u::reduce_shape<In, u::ReduceAxis::Rows>> out(kCbOut);   // Shape<1, 4>
//     out.store(u::reduce_sum<u::ReduceAxis::Rows>(block, scaler));
//
// The destination's shape is checked against what the reduction yields, so a
// mis-sized output buffer does not compile.
//
// `scaler` comes from fill_reduce_scaler and must be held at KERNEL scope: every
// reduce_tile re-reads it, so it must not be popped until the kernel ends. Taking
// a ComputeBlock rather than a Storage is what says so -- and proves the buffer
// was actually filled and waited on.
template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Sum, expr::UnaryChain<>> reduce_sum(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler);

// Same shape, different fold. The SCALER differs though, and silently: metal folds
// it into every reduce_tile, so
//
//   sum, max   scaler 1               -- kReduceScalerOne
//   mean       scaler 1/N             -- bf16_pair(1.0f / ReduceGeometry<In>::elements(Axis))
//              (1/sqrt(N) when both axes collapse)
//
// A mean fed a scaler of 1 is just a sum, with nothing to say so. The scaler is
// the kernel's to fill, so the kernel has to match it to the fold it asks for.
template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Max, expr::UnaryChain<>> reduce_max(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler);

template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Avg, expr::UnaryChain<>> reduce_mean(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler);

// ---------------------------------------------------------------------------
// custom_compute -- the escape hatch
//
// The compute-side counterpart of noc_load's and noc_store's Fn forms: for a pass
// this model does not express, call the LLK directly.
//
//     custom_compute(a, b, [&](uint32_t a_cb, uint32_t b_cb) {
//     #if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
//         // any llk / compute api, on those two buffers
//     #endif
//     });
//
// Takes any number of ComputeBlocks and a routine taking that many circular-buffer
// ids, in the same order. Anything else in a block position is a compile error.
//
// WHY THE GUARD IS YOURS. The routine is only CALLED on the compute projection --
// this does nothing on the three data-movement ones -- but its body is COMPILED on
// all five, because a lambda's body is compiled where it is written. So every name
// it mentions has to resolve on a data-movement build too, and most LLK entry
// points do not. Hence the `#if` inside. Exactly the contract noc_load's Fn form
// has, from the other side.
//
// WHAT THE HARNESS DOES: waits the blocks (each ComputeBlock's constructor did) and
// pops them at the end of the enclosing scope (its destructor). That is all.
//
// WHAT IT DOES NOT DO, and each of these is yours:
//
//   * DST registers. No tile_regs_acquire/commit/wait/release around the routine --
//     unlike Storage::store, whose strategies bracket every pass. If the routine
//     uses DST it must bracket itself.
//   * The output. The routine gets INPUT buffers. To produce a block, reach the
//     destination's `Storage::cb_id`, do the reserve/pack/push by hand, and then
//     `Block<Out>{out_storage}` is the handle a data-movement thread can drain --
//     that constructor only records the buffer, it does not push.
//   * PUTTING THE UNITS BACK. This is the one that bites. Unpacker, math and packer
//     configuration is per-kernel state, and whatever the routine leaves set, the
//     next unified op inherits. The library's own passes already live with this --
//     every matmul re-runs matmul_block_init because a broadcast or a reduction
//     before it reconfigured the units. A routine that reconfigures anything should
//     leave it as it found it, or the next op returns garbage with nothing to say so.
// ---------------------------------------------------------------------------

template <typename T>
struct is_compute_block : std::false_type {};

template <typename S>
struct is_compute_block<ComputeBlock<S>> : std::true_type {};

template <typename... Ts>
void custom_compute(Ts&&... ts);

// ---------------------------------------------------------------------------
// Reserved multicast semaphores
//
// The multicast handshake needs two counters that mean nothing to the caller, so
// the harness reserves them and passes their base id in as a define. Two PER DM
// THREAD, at base + 2*thread: multicasts on one NOC serialize in hardware anyway,
// so a pair per NOC is the natural granularity, and giving each thread its own
// pair keeps a NOC-0 and a NOC-1 broadcast from sharing handshake state.
//
// The base sits above any semaphore the caller allocated, so user ids are
// unconstrained.
// ---------------------------------------------------------------------------

#if defined(TT_UNIFIED_MCAST_SEM_BASE)
inline constexpr bool kMcastSemsReserved = true;
inline constexpr uint32_t kMcastSemBase = TT_UNIFIED_MCAST_SEM_BASE;
#else
inline constexpr bool kMcastSemsReserved = false;
inline constexpr uint32_t kMcastSemBase = 0;
#endif

// ---------------------------------------------------------------------------
// Buffer slots
//
//     constexpr uint32_t kCbIn = get_named_compile_time_arg_val("cb_in");
//     u::Storage<Block1D> in_storage(kCbIn);
//
// A kernel cannot get its buffers' slot numbers from the `dfb::` binding tokens the way a
// single-projection kernel would. A token is emitted only into the kernels that bind that
// buffer (genfiles.cpp:129), a buffer's two endpoint roles are both spoken for, and a
// unified kernel declares every Storage on every projection -- so `dfb::out` does not exist
// on the build that only reads `in`. See unified_metal2_spec.md 7.1.
//
// So the slot arrives as a named compile-time VALUE, exactly like every other scalar the
// kernel is given, under the name `cb_<buffer>`. The harness predicts it from metal's
// allocator rule (lowest free slot, declaration order) and unified_program_spec() documents
// that prediction.
//
// The prediction is not separately checked, and the reason it need not be is that all five
// projections read the SAME value: a wrong one has every thread agreeing on a buffer the
// host allocated to something else, which is a hang or wrong data on the first run rather
// than anything silent. That is unlike the endpoint ROLES, which really are silent when
// wrong on Gen1 -- see derive_roles() in unified_harness.py, which is why those are read off
// the kernel rather than restated.
// ---------------------------------------------------------------------------

// Under Metal 2.0 the base is a PREDICTION, and this is where it gets checked.
//
// Semaphores reach a 2.0 kernel as `sem::<name>` ids the host assigned, and the harness has
// to predict them because everything below derives its ids from one base by arithmetic --
// which needs the reserved run to be contiguous and to start where the harness thinks it
// does. The host is the only party that knows either fact, and a token is the only thing
// that reports it back, so the harness passes the FIRST and LAST reserved names as token
// expressions and the arithmetic is checked against them here.
//
// Checking both ends is what makes it airtight rather than indicative: metal cannot issue a
// duplicate id, so six distinct ids whose smallest is `base` and whose largest is `base + 5`
// can only be the contiguous run.
//
// Nothing to check on the legacy path, where the harness allocates the ids itself.
#if defined(TT_UNIFIED_MCAST_SEM_FIRST) && defined(TT_UNIFIED_MCAST_SEM_LAST)
static_assert(
    kMcastSemBase == static_cast<uint32_t>(TT_UNIFIED_MCAST_SEM_FIRST),
    "the harness's predicted multicast semaphore base does not match the id the host assigned");
static_assert(
    kMcastSemBase + 2 * 2 + 2 - 1 == static_cast<uint32_t>(TT_UNIFIED_MCAST_SEM_LAST),
    "the reserved multicast semaphores are not contiguous -- every id below is derived from "
    "kMcastSemBase by arithmetic, so a gap in the run silently retargets a handshake");
#endif

// Ids of the pair belonging to `thread`.
template <int thread>
inline constexpr uint32_t kMcastReadySem = kMcastSemBase + 2 * thread;
template <int thread>
inline constexpr uint32_t kMcastSentSem = kMcastSemBase + 2 * thread + 1;

// One more per thread, above the two pairs: the arrival flag a multicast
// noc_core_write raises on its receivers. It gets its own slot rather than
// borrowing the pair above, because a core-to-core push and a broadcast (or a
// synchronize_cores) can legitimately be in flight on one thread at once.
template <int thread>
inline constexpr uint32_t kCopyArrivedSem = kMcastSemBase + 4 + thread;

// The program's core grid, so a whole-program barrier needs no arguments. Also
// supplied by the harness, which is the only place that knows the core range.
#if defined(TT_UNIFIED_CORE_GRID_H) && defined(TT_UNIFIED_CORE_GRID_W)
inline constexpr bool kCoreGridKnown = true;
inline constexpr uint32_t kCoreGridH = TT_UNIFIED_CORE_GRID_H;
inline constexpr uint32_t kCoreGridW = TT_UNIFIED_CORE_GRID_W;
#else
inline constexpr bool kCoreGridKnown = false;
inline constexpr uint32_t kCoreGridH = 1;
inline constexpr uint32_t kCoreGridW = 1;
#endif

// Whether that grid is the WHOLE story: H x W cores, all of them running this program.
//
// It is not always. The grid above is the core range's BOUNDING BOX, and a range set need
// not fill it -- twelve cores laid out row-major are eight in row 0 and four in row 1,
// whose bounding box is 2 x 8 = sixteen. A rectangle is the only thing a multicast can
// address, so anything derived from the bounding box then addresses four cores that were
// never launched, and a barrier waits on them forever.
//
// The harness knows both numbers and defines this only when they agree, which turns that
// hang into a compile error at the one call that cannot take a region argument.
#if defined(TT_UNIFIED_CORE_GRID_EXACT)
inline constexpr bool kCoreGridExact = true;
#else
inline constexpr bool kCoreGridExact = false;
#endif

// Two bfloat16 1.0 values in one 32-bit word -- the scaler a SUM reduction wants.
// A float32 CB would want a single 0x3F800000 instead.
inline constexpr uint32_t kReduceScalerOne = 0x3F803F80u;

// The same word for any value: bfloat16 is the top half of a float32, twice over.
// A mean needs it, because its scaler is 1/N rather than 1 -- see reduce_mean.
inline uint32_t bf16_pair(float v) {
    uint32_t bits = 0;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    const uint32_t half = bits >> 16;
    return (half << 16) | half;
}

// ---------------------------------------------------------------------------
// synchronize_cores -- barrier across the CORES of a region
//
// Every participating core runs the same statement; the region's start corner is
// the rendezvous point. Reuses the reserved multicast handshake pair, which is
// why every operation touching those semaphores leaves both at 0 on every core.
//
// It synchronizes CORES, not the five threads on a core: only DM thread `thread`
// participates, and the other projections run straight past. Two threads can
// barrier independently, since each has its own reserved pair.
//
// The no-argument form spans the program's whole core grid.
// ---------------------------------------------------------------------------

template <int thread>
void synchronize_cores(PhysicalMcast region);

template <int thread>
void synchronize_cores(LogicalMcast region);

template <int thread>
void synchronize_cores();

// ---------------------------------------------------------------------------
// Semaphore -- a host-allocated L1 counter, projected onto one DM thread
//
// The storage belongs to the HOST: a SemaphoreDescriptor on the program reserves
// one slot per core in a range and stamps its initial value, and `semaphore_id`
// is the index into that reservation. This is what makes cross-core signalling
// work at all -- every core resolves the same id to the same L1 offset, and the
// offset is independent of which RISC is running.
//
// Do NOT give a Semaphore its own storage instead. A member or local would sit at
// an address that only happens to agree across cores (and not at all across the
// BRISC/NCRISC binaries), and its initial value would be set by a constructor
// that runs once per binary load rather than once per program launch.
//
// `thread` selects the owning DM thread, exactly as for the Noc*Tx handles: every
// operation is a no-op on the other projections, so one shared statement means
// "thread N does this".
template <int thread>
class Semaphore {
public:
    explicit Semaphore(uint32_t semaphore_id);

    // The reserved id. A handle that outlives this object has to carry the id rather
    // than a reference: every pair-derived multicast builds its two semaphores as
    // LOCALS inside noc_load, and a reference would dangle the moment it returned.
    uint32_t semaphore_id() const;

    // Local: spin until this core's copy reaches (or reaches at least) `value`.
    Semaphore& wait(uint32_t value);
    Semaphore& wait_min(uint32_t value);

    // Local: overwrite this core's copy.
    Semaphore& set(uint32_t value);

    // Remote: atomically add to the SAME semaphore on another core.
    Semaphore& inc_remote(PhysicalCoord coord, uint32_t value = 1);
    Semaphore& inc_remote(LogicalCoord coord, uint32_t value = 1);

    // Remote: atomically add to the same semaphore on every core of a rectangle.
    // Must be issued by the rectangle's start corner, which is excluded from the
    // destinations.
    Semaphore& inc_mcast(PhysicalMcast mcast, uint32_t value = 1);
    Semaphore& inc_mcast(LogicalMcast mcast, uint32_t value = 1);

    // Remote: copy this core's value into the same semaphore on every core of a
    // rectangle. Named for what it does -- do not call it `mcast`, or a parameter
    // of the same name shadows the overload set.
    Semaphore& set_mcast(PhysicalMcast mcast);
    Semaphore& set_mcast(LogicalMcast mcast);

private:
    // Backs semaphore_id(). Metal's Semaphore keeps its own L1 address private and
    // exposes no id, so a handle that has to outlive this object carries this instead
    // -- and there is deliberately no l1_addr() accessor recomputing that address by
    // hand: a routine addressing the semaphore directly should take the Semaphore.
    uint32_t id;

#if defined(IS_DM_THREAD) && IS_DM_THREAD
    // Metal's own semaphore. Spelled ::Semaphore because this class shadows it.
    ::Semaphore<ProgrammableCoreType::TENSIX> sem;
#endif
};

// Optional per-instance profiler zones inside the multicast sender, for splitting its
// k-block into waiting-for-receivers, waiting-for-DRAM, and broadcasting. Off unless
// TT_UNIFIED_MCAST_ZONES is defined, because a zone per block per core is a lot of records
// and the profiler drops them silently once its buffer fills.
#if defined(TT_UNIFIED_MCAST_ZONES) && defined(IS_DM_THREAD) && IS_DM_THREAD
#define TT_U_ZONE(name) DeviceZoneScopedN(name)
#else
#define TT_U_ZONE(name) ((void)0)
#endif

// ---------------------------------------------------------------------------
// NOC transaction handles
//
// Reads: wait() is mandatory -- you need the data, and it is what publishes the
// destination. A forgotten wait() is caught by the destructor assert.
//
// Writes: fire and forget. The destructor completes them correctly, so there is
// nothing at the call site to forget. wait() is there for the rare case that
// needs *landed* rather than *departed*.
// ---------------------------------------------------------------------------
// NocAsyncMcastTx -- the handle a MULTICAST load returns
//
// A multicast load is not a plain read: the sender fills its own copy from DRAM and
// then broadcasts it, while every receiver has its copy filled for it and learns so
// from a flag. NocAsyncReadTx cannot express the second half, because it has no idea
// which role this core plays or which semaphore carries the flag.
//
// So this carries both -- the `data_sent` id and the role -- and today does nothing
// with them. `wait()` is byte for byte what NocAsyncReadTx's does: the read barrier
// and the push. The receiver's flag wait is still inside noc_load, where it has
// always been.
//
// THAT IS DELIBERATE. Sinking the flag wait into wait() is a behaviour change with a
// precondition attached -- the flag is a 0/1 the sender rewrites every round, so a
// receiver holding an uncleared 1 cannot tell round b from b+1 -- and it belongs in
// its own step. See unified_mcast_handle_spec.md. This step is the shape only, and
// its checkpoint is that every suite is unchanged.
// ---------------------------------------------------------------------------

template <int thread, typename S>
struct NocAsyncMcastTx {
    using shape = S;

    NocAsyncMcastTx(const Storage<S>& storage, uint32_t data_sent_id, bool sender);

    NocAsyncMcastTx(const NocAsyncMcastTx&) = delete;
    NocAsyncMcastTx& operator=(const NocAsyncMcastTx&) = delete;
    NocAsyncMcastTx(NocAsyncMcastTx&&) = delete;
    NocAsyncMcastTx& operator=(NocAsyncMcastTx&&) = delete;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ~NocAsyncMcastTx();
#endif

    // Publishes the block. No argument, and returns Block<S>, so every existing call
    // site -- all of them `noc_load(...).wait()` -- compiles unchanged.
    Block<S> wait() const;

    uint32_t cb_id;
    static constexpr uint32_t num_pages = S::num_pages;

    // The NOC this transaction was ISSUED on, so wait() can barrier on that one.
    //
    // Not decoration. noc_async_read_barrier and noc_async_writes_flushed are per-NOC, and a
    // barrier on the wrong one returns immediately -- the push then publishes pages that have
    // not landed, with no hang and no assert. So the NOC has to travel with the handle rather
    // than be re-derived at the barrier, which is why this member exists before anything can
    // request a NOC other than the thread's own. See unified_explicit_noc_spec.md, step 1.
    //
    // Stored as the INDEX rather than as a Noc: the handle types exist on every projection and
    // Noc is declared only off-TRISC, so a Noc member would not compile on compute. It is
    // reconstituted where it is used, inside data-movement-guarded code.
    uint8_t noc_id = noc_index;

    // Carried for the step that moves the flag wait here. Unused today.
    mutable Semaphore<thread> data_sent;
    bool sender;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    mutable bool waited = false;
#endif
};

// ---------------------------------------------------------------------------

template <int thread, typename S>
struct NocAsyncReadTx {
    using shape = S;

    explicit NocAsyncReadTx(const Storage<S>& storage);
    explicit NocAsyncReadTx(uint32_t cb_id);

    NocAsyncReadTx(const NocAsyncReadTx&) = delete;
    NocAsyncReadTx& operator=(const NocAsyncReadTx&) = delete;
    NocAsyncReadTx(NocAsyncReadTx&&) = delete;
    NocAsyncReadTx& operator=(NocAsyncReadTx&&) = delete;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ~NocAsyncReadTx();
#endif

    // Completes the read and publishes the destination.
    Block<S> wait() const;

    uint32_t cb_id;
    static constexpr uint32_t num_pages = S::num_pages;

    // The NOC this transaction was ISSUED on, so wait() can barrier on that one.
    //
    // Not decoration. noc_async_read_barrier and noc_async_writes_flushed are per-NOC, and a
    // barrier on the wrong one returns immediately -- the push then publishes pages that have
    // not landed, with no hang and no assert. So the NOC has to travel with the handle rather
    // than be re-derived at the barrier, which is why this member exists before anything can
    // request a NOC other than the thread's own. See unified_explicit_noc_spec.md, step 1.
    //
    // Stored as the INDEX rather than as a Noc: the handle types exist on every projection and
    // Noc is declared only off-TRISC, so a Noc member would not compile on compute. It is
    // reconstituted where it is used, inside data-movement-guarded code.
    uint8_t noc_id = noc_index;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    mutable bool waited = false;
#endif
};

template <int thread, typename S>
struct NocAsyncWriteTx {
    using shape = S;

    explicit NocAsyncWriteTx(const Storage<S>& storage);
    explicit NocAsyncWriteTx(uint32_t cb_id);

    NocAsyncWriteTx(const NocAsyncWriteTx&) = delete;
    NocAsyncWriteTx& operator=(const NocAsyncWriteTx&) = delete;
    NocAsyncWriteTx(NocAsyncWriteTx&&) = delete;
    NocAsyncWriteTx& operator=(NocAsyncWriteTx&&) = delete;

    // Releases the source: flush, then pop.
    ~NocAsyncWriteTx();

    // Optional: block until the data has LANDED at the destination.
    void wait() const;

    uint32_t cb_id;
    static constexpr uint32_t num_pages = S::num_pages;

    // The NOC this transaction was ISSUED on, so wait() can barrier on that one.
    //
    // Not decoration. noc_async_read_barrier and noc_async_writes_flushed are per-NOC, and a
    // barrier on the wrong one returns immediately -- the push then publishes pages that have
    // not landed, with no hang and no assert. So the NOC has to travel with the handle rather
    // than be re-derived at the barrier, which is why this member exists before anything can
    // request a NOC other than the thread's own. See unified_explicit_noc_spec.md, step 1.
    //
    // Stored as the INDEX rather than as a Noc: the handle types exist on every projection and
    // Noc is declared only off-TRISC, so a Noc member would not compile on compute. It is
    // reconstituted where it is used, inside data-movement-guarded code.
    uint8_t noc_id = noc_index;
};

// A core-to-core copy has both halves: a local source Block to release and a
// destination Storage to publish. The destination follows the read rule (explicit
// wait()) and the source follows the write rule (the destructor).
//
// Pull: the source is the PEER's L1 and the local Block is only a handle, so the
// destructor pops it bare, and this core's own read barrier is proof the data
// landed -- it landed here.
template <int thread, typename D, typename S>
struct NocAsyncReadCoreTx {
    // A core-to-core copy is not required to fill its destination: a GATHER has n
    // writers each depositing its own source at its own byte_offset, so the
    // destination is n times the source. What must hold is that the source fits and
    // tiles the destination evenly -- one whole slot per writer. Equality would be
    // wrong; nothing checked either fact before, since the two page counts were
    // independent runtime fields.
    static_assert(
        S::num_pages <= D::num_pages,
        "a core-to-core copy's source does not fit its destination -- the source Block has more pages "
        "than the destination Storage");
    static_assert(
        D::num_pages % S::num_pages == 0,
        "a core-to-core copy's destination is not a whole multiple of its source -- a gather deposits "
        "one source-sized slot per writer, so a ragged destination cannot be addressed by byte_offset");

    NocAsyncReadCoreTx(const Storage<D>& dst, const Block<S>& src);

    NocAsyncReadCoreTx(const NocAsyncReadCoreTx&) = delete;
    NocAsyncReadCoreTx& operator=(const NocAsyncReadCoreTx&) = delete;
    NocAsyncReadCoreTx(NocAsyncReadCoreTx&&) = delete;
    NocAsyncReadCoreTx& operator=(NocAsyncReadCoreTx&&) = delete;

    ~NocAsyncReadCoreTx();

    Block<D> wait() const;

    uint32_t dst_cb;
    static constexpr uint32_t dst_pages = D::num_pages;
    uint32_t src_cb;
    static constexpr uint32_t src_pages = S::num_pages;

    // The NOC this transaction was ISSUED on, so wait() can barrier on that one.
    //
    // Not decoration. noc_async_read_barrier and noc_async_writes_flushed are per-NOC, and a
    // barrier on the wrong one returns immediately -- the push then publishes pages that have
    // not landed, with no hang and no assert. So the NOC has to travel with the handle rather
    // than be re-derived at the barrier, which is why this member exists before anything can
    // request a NOC other than the thread's own. See unified_explicit_noc_spec.md, step 1.
    //
    // Stored as the INDEX rather than as a Noc: the handle types exist on every projection and
    // Noc is declared only off-TRISC, so a Noc member would not compile on compute. It is
    // reconstituted where it is used, inside data-movement-guarded code.
    uint8_t noc_id = noc_index;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    mutable bool waited = false;
#endif
};

// Push: this core's L1 is the source, so the NOC must have finished reading it
// before the pop and the destructor flushes first.
//
// The write side also carries the arrival handshake, which is why it is a
// separate type. A write barrier only tells the SENDER its data landed -- it
// waits on the destination's acks, which the receiving core cannot observe. So
// for a push to a rectangle, wait() splits by role: the sender barriers and then
// raises `arrived` across the rectangle, and every core inside waits on it before
// publishing its own copy of `dst`.
//
// `arrived` is a member rather than a parameter: it is protocol plumbing on a
// host-reserved slot (kCopyArrivedSem), the same argument that makes the reserved
// broadcast pair preferable to a caller-supplied one. It sits unused on the
// unicast form, where construction costs only an L1 address.
template <int thread, typename D, typename S>
struct NocAsyncWriteCoreTx {
    // A core-to-core copy is not required to fill its destination: a GATHER has n
    // writers each depositing its own source at its own byte_offset, so the
    // destination is n times the source. What must hold is that the source fits and
    // tiles the destination evenly -- one whole slot per writer. Equality would be
    // wrong; nothing checked either fact before, since the two page counts were
    // independent runtime fields.
    static_assert(
        S::num_pages <= D::num_pages,
        "a core-to-core copy's source does not fit its destination -- the source Block has more pages "
        "than the destination Storage");
    static_assert(
        D::num_pages % S::num_pages == 0,
        "a core-to-core copy's destination is not a whole multiple of its source -- a gather deposits "
        "one source-sized slot per writer, so a ragged destination cannot be addressed by byte_offset");

    NocAsyncWriteCoreTx(const Storage<D>& dst, const Block<S>& src, PhysicalMcast dst_range, uint32_t semaphore_id);
    NocAsyncWriteCoreTx(const Storage<D>& dst, const Block<S>& src, bool reader, uint32_t semaphore_id);

    NocAsyncWriteCoreTx(const NocAsyncWriteCoreTx&) = delete;
    NocAsyncWriteCoreTx& operator=(const NocAsyncWriteCoreTx&) = delete;
    NocAsyncWriteCoreTx(NocAsyncWriteCoreTx&&) = delete;
    NocAsyncWriteCoreTx& operator=(NocAsyncWriteCoreTx&&) = delete;

    ~NocAsyncWriteCoreTx();

    Block<D> wait(uint32_t num_writers) const;

    uint32_t dst_cb;
    static constexpr uint32_t dst_pages = D::num_pages;
    uint32_t src_cb;
    static constexpr uint32_t src_pages = S::num_pages;

    // The NOC this transaction was ISSUED on, so wait() can barrier on that one.
    //
    // Not decoration. noc_async_read_barrier and noc_async_writes_flushed are per-NOC, and a
    // barrier on the wrong one returns immediately -- the push then publishes pages that have
    // not landed, with no hang and no assert. So the NOC has to travel with the handle rather
    // than be re-derived at the barrier, which is why this member exists before anything can
    // request a NOC other than the thread's own. See unified_explicit_noc_spec.md, step 1.
    //
    // Stored as the INDEX rather than as a Noc: the handle types exist on every projection and
    // Noc is declared only off-TRISC, so a Noc member would not compile on compute. It is
    // reconstituted where it is used, inside data-movement-guarded code.
    uint8_t noc_id = noc_index;

    // mutable: wait() is const across the whole API, and signalling is what a
    // wait on this handle does.
    mutable Semaphore<thread> arrived;
    bool reader;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    mutable bool waited = false;
#endif
};

// ---------------------------------------------------------------------------
// Data movement. Each is pinned to a DM thread by its `thread` argument and
// compiles away entirely on every other thread.
//
// `thread` ALSO picks the NOC, since a DM thread is bound to one by its index,
// and the two NOCs are not interchangeable: for DRAM reads NOC 0 is much the
// faster of them. Measured on the blocked matmul, where flipping which NOC
// carries the large operand is worth 1.4x on its own and the whole arrangement
// spans 2.6x; rmsnorm 2.4x; flash attention 1.18x, it being latency-bound
// rather than bandwidth-bound. So the rule these kernels follow is READS ON
// THREAD 0, writes on 1 -- and where there are two read streams, the BIG one
// takes thread 0 and the other takes 1 so they still overlap. Getting this
// backwards costs more than any other single choice in these kernels; see
// unified_llama_prefill.md.
// ---------------------------------------------------------------------------

// Reads `storage.num_pages` pages into the buffer, starting at page
// `block_idx * storage.num_pages`. The returned handle publishes them.
template <int thread, typename S, typename Accessor>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, const Accessor& acc, uint32_t block_idx);

// Custom load, for routines the built-in overload cannot express. The harness
// keeps the circular-buffer protocol -- cb_reserve_back, the write pointer, and
// (via the returned handle) the read barrier and cb_push_back -- and `fn` owns
// the traffic. It is called as
//
//     fn(L1Pages pages)
//
// and must fill pages.count consecutive pages from pages.base: that is the
// count the handle pushes, whatever `fn` actually wrote, so loop on pages.count.
//
// The built-in overloads above are written this way too, so this path carries the
// same weight as they do rather than being a side door.
//
// `fn` must issue ONLY READS, and only on this thread's assigned NOC. The handle
// releases with noc_async_read_barrier(), which covers reads on a single NOC --
// reads issued on the other NOC, or writes, are not covered, and the push would
// then publish pages that have not landed.
//
// `fn` is only CALLED on the owning data-movement thread, but its body is
// COMPILED on all five projections, so the intrinsics it names have to resolve
// everywhere; see tt/unified/adaptor.hpp.
template <int thread, typename S, typename Fn>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, Fn fn);

// Multicast load: one core in the rectangle reads the block from `acc` and
// multicasts it into the SAME circular buffer on every core of the rectangle.
// Every core runs this same statement; which side of the handshake it takes is a
// runtime decision on its own coordinate.
//
// Two semaphores are required, and both must be reserved by the host so all cores
// agree on their offsets:
//   receivers_ready -- receivers count themselves in; the sender waits for them
//   data_sent       -- the sender announces the payload has been multicast
//
// The call is repeatable without host intervention, which takes deliberate resets
// -- a semaphore that keeps its count lets the NEXT call fall straight through
// the handshake. `receivers_ready` is cleared by the sender once it has counted
// everyone in; `data_sent` is cleared by each receiver after it observes it.
template <int thread, typename S, typename Accessor>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage,
    PhysicalMcast mcast,
    Semaphore<thread>& receivers_ready,
    Semaphore<thread>& data_sent,
    const Accessor& acc,
    uint32_t block_idx);

template <int thread, typename S, typename Accessor>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage,
    LogicalMcast mcast,
    Semaphore<thread>& receivers_ready,
    Semaphore<thread>& data_sent,
    const Accessor& acc,
    uint32_t block_idx);

// Same, with the handshake semaphores supplied by the harness's reservation.
// Prefer these: the pair is protocol plumbing, and having callers allocate it
// invites the initial-value and reset mistakes the explicit form makes possible.
//
// `pair` selects which reserved pair to use, defaulting to the driving thread's.
// Two broadcasts must never share a pair -- their ready counters would interleave
// and noc_semaphore_wait, which waits for EQUALITY, would miss its target. The
// default is right when they run on different threads (the usual case: one per
// NOC, overlapping). Name the pair explicitly to put two broadcasts on ONE thread
// and still keep them apart.
template <int thread, int pair = thread, typename S, typename Accessor>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage, PhysicalMcast mcast, const Accessor& acc, uint32_t block_idx);

// Multicast load with a CUSTOM fill, the same relationship the plain noc_load's Fn form has
// to its accessor form. `fn` runs on the sender only and fills its copy however it likes;
// the broadcast that follows does not care how the bytes arrived.
//
// This is what lets a multicast operand be gathered rather than read as one contiguous
// block. A k-slice of a row-major activation, or a (k, n) tile of a wider weight matrix, is
// strided in DRAM and contiguous nowhere -- but it is an ordinary block once in L1. Costs no
// extra traffic: the built-in read issues one request per page too.
template <int thread, int pair = thread, typename S, typename Fn>
NocAsyncMcastTx<thread, S> noc_load(const Storage<S>& storage, PhysicalMcast mcast, Fn fn);
template <int thread, int pair = thread, typename S, typename Fn>
NocAsyncMcastTx<thread, S> noc_load(const Storage<S>& storage, LogicalMcast mcast, Fn fn);

template <int thread, int pair = thread, typename S, typename Accessor>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage, LogicalMcast mcast, const Accessor& acc, uint32_t block_idx);

// Fill a one-page Storage with the constant metal's reduce folds in: the value in
// the first row of each of the tile's four 16x16 faces, zero everywhere else.
// Call it ONCE, before the first reduction; it pushes the page and nothing ever
// pops it, because every reduce_tile re-reads the same tile.
//
// `value_bits` is written as raw 32-bit words, so its packing follows the CB's
// format: for bfloat16 one word is TWO values, which is what kReduceScalerOne is.
// Sum wants 1.0; an average wants 1/N (1/sqrt(N) reducing both axes).
// Returns the page as a Block, so a scaler is held the same way a fused bias is:
// as a ComputeBlock at KERNEL scope. That is what makes the wait happen once, in
// its constructor, and the pop happen at the end of the kernel rather than after
// the first reduction.
template <int thread, typename S>
Block<S> fill_reduce_scaler(const Storage<S>& scaler, uint32_t value_bits = kReduceScalerOne);

// Drains a Block to a tensor. Takes the Block by value: this call consumes it.
template <int thread, typename S, typename Accessor>
NocAsyncWriteTx<thread, S> noc_store(Block<S> block, const Accessor& acc, uint32_t block_idx);

// Custom store: the mirror of the custom noc_load. `fn` is called as
// fn(L1Pages pages) over `block`'s pages -- pages.count is the number the handle
// pops.
//
// `fn` must issue ONLY WRITES, and only on this thread's assigned NOC. The handle
// releases the source buffer with noc_async_writes_flushed() and pops, which
// covers writes departing local L1 on a single NOC. Reads issued here, or writes
// on the other NOC, are not covered, so the pop can hand the pages back while
// they are still being sourced.
template <int thread, typename S, typename Fn>
NocAsyncWriteTx<thread, S> noc_store(Block<S> block, Fn fn);

// ---------------------------------------------------------------------------
// Core-to-core movement: pull a peer's block into this core's Storage
// (noc_core_read), or push this core's block into a peer's Storage
// (noc_core_write). `byte_offset` shifts the PEER-side address within its buffer.
//
// The Physical overloads are the real ones; the Logical ones translate and
// forward. Only the write side takes a rectangle: pushing one block to many peers
// is meaningful, pulling from many is not.
//
// NOTE: reserve/push act on the *local* view of the destination CB. For a genuine
// peer buffer the far side's pointers have to be advanced too -- see
// api/remote_circular_buffer.h (remote_cb_reserve_back /
// remote_cb_push_back_and_write_pages, asymmetric between sender and receiver) or
// the explicit semaphore handshake the matmul mcast kernels use.
// ---------------------------------------------------------------------------

template <int thread, typename D, typename S>
NocAsyncReadCoreTx<thread, D, S> noc_core_read(
    const Storage<D>& dst, Block<S> src, PhysicalCoord coord, uint32_t byte_offset = 0);

// EVERY core in the exchange runs one statement and takes its side from its own
// coordinate and predicate:
//
//   `dst_range`         the cores being written INTO. A core inside it is a
//                       reader: it takes delivery and publishes its own copy of
//                       `dst`.
//   `write_predicate`   whether THIS core writes. Its destinations are `dst_range`.
//
// `wait(num_writers)` is the count the reader collects: how many cores had
// write_predicate true. Nothing checks it against reality -- too high hangs, too
// low publishes short.
//
// WHAT THIS GIVES YOU is arrival notification for ONE push: the writers raise
// `arrived` after their payload, and the reader will not publish `dst` until it
// has counted them all. That is the part a write barrier cannot do, since it
// tells only the writer that its data landed.
//
// WHAT IT DOES NOT GIVE YOU is a repeatable channel. Two things are the caller's:
//
//   1. dst has to be FREE before the writers write. Nothing here tells a writer
//      that the reader is done with the previous round's contents, and the
//      reader's `arrived.set(0)` after collecting is a window in which a writer
//      already into the next round loses its increment -- which hangs the round
//      after. Put a synchronize_cores() between pushes, or otherwise establish
//      that every reader is finished, and both go away.
//
//   2. Addressing. The destination is computed from the WRITER's local view of
//      `dst`, so the copies have to stay in step (see the NOTE above).
template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S> noc_core_write(
    const Storage<D>& dst, Block<S> src, PhysicalCoord coord, bool write_predicate, uint32_t byte_offset = 0);

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S> noc_core_write(
    const Storage<D>& dst, Block<S> src, PhysicalMcast dst_range, bool write_predicate, uint32_t byte_offset = 0);

template <int thread, typename D, typename S>
NocAsyncReadCoreTx<thread, D, S> noc_core_read(
    const Storage<D>& dst, Block<S> src, LogicalCoord coord, uint32_t byte_offset = 0);

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S> noc_core_write(
    const Storage<D>& dst, Block<S> src, LogicalCoord coord, bool write_predicate, uint32_t byte_offset = 0);

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S> noc_core_write(
    const Storage<D>& dst, Block<S> src, LogicalMcast dst_range, bool write_predicate, uint32_t byte_offset = 0);

}  // namespace unified
}  // namespace tt
