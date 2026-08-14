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
// Include <tt/unified>, not this header directly -- it selects an implementation
// and a backend binding, and documents the layering.

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include <tt/unified_math.hpp>

namespace tt {
namespace unified {

struct Block;
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

struct PhysicalCoord {
    uint32_t y;
    uint32_t x;

    // This core's own physical coordinate, on this thread's NOC.
    static PhysicalCoord this_core();

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    bool operator==(PhysicalCoord o) const { return y == o.y && x == o.x; }
    bool operator!=(PhysicalCoord o) const { return !(*this == o); }
};

struct LogicalCoord {
    uint32_t y;
    uint32_t x;

    static LogicalCoord this_core();

    PhysicalCoord to_physical(uint32_t y_offset = 0, uint32_t x_offset = 0) const;

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    bool operator==(LogicalCoord o) const { return y == o.y && x == o.x; }
    bool operator!=(LogicalCoord o) const { return !(*this == o); }
};

struct Shape {
    uint32_t h;
    uint32_t w;
};

// A multicast rectangle, inclusive of both corners.
struct PhysicalMcast {
    PhysicalCoord start;
    PhysicalCoord end;

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
    Shape shape;

    PhysicalMcast to_physical() const;

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    uint32_t volume() const { return shape.h * shape.w; }
};

// ---------------------------------------------------------------------------
// Storage -- a circular buffer
// ---------------------------------------------------------------------------

struct Storage {
    Storage(uint32_t cb_id, uint32_t num_tiles) : cb_id(cb_id), num_tiles(num_tiles) {}

    Storage(Storage&&) = delete;
    Storage(const Storage&) = delete;
    Storage& operator=(Storage&&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Evaluate a compute fusion into this buffer. The loop shape is chosen by the
    // fusion's kind; see Strategy in tt/unified_math.hpp.
    template <typename Node>
    Block store(const Node& node);

    uint32_t cb_id;
    uint32_t num_tiles;  // could eventually be N dimensional, here and below
};

// ---------------------------------------------------------------------------
// Block -- move-only evidence that a Storage was produced into
//
// Every Block comes from an operation that has already pushed, which is what
// makes it safe to hand one to a DM thread to drain. Move-only so it reaches
// exactly one consumer; consumers take it by value.
// ---------------------------------------------------------------------------

template <AccumulatorMode Mode = AccumulatorMode::Dst>
class Accumulator;

struct Block {
    explicit Block(const Storage& storage);
    Block(uint32_t cb_id, uint32_t num_tiles);
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
    uint32_t num_tiles;

private:
    // A RETAINED block: one the Accumulator hands back mid-accumulation. Its pages
    // still belong to the accumulator, so it must neither be transferred to
    // another thread nor consumed -- only the next accumulate() call may touch
    // them. Only Accumulator can make one.
    struct Retained {};
    Block(const Storage& storage, Retained);

    template <AccumulatorMode M>
    friend class Accumulator;

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    // Two independent facts, deliberately not folded into one flag:
    //   must_consume -- this Block owes a consumer (false for retained blocks)
    //   consumed     -- a consumer has taken it
    // A moved-from Block has must_consume=false and consumed=true, so it is silent
    // at destruction and asserts if used again.
    bool must_consume = true;
    bool consumed = false;

    // Poison value stamped into a moved-from Block's fields.
    static constexpr uint32_t kMovedFrom = ~uint32_t(0);
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

template <AccumulatorMode Mode>
class Accumulator {
public:
    Accumulator(const Storage& acc_storage, const Storage& out_storage);

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
    Block accumulate(const Node& node, bool finish, Epilogue epilogue = nullptr);

    // Reset between output blocks.
    void clear();

private:
    const Storage& acc_storage;
    const Storage& out_storage;
    bool reload = false;
};

// ---------------------------------------------------------------------------
// ComputeBlock -- compute-side consumption of a Block, and an expression leaf
// ---------------------------------------------------------------------------

class ComputeBlock {
public:
    ComputeBlock(Block block);
    ~ComputeBlock();

    ComputeBlock(const ComputeBlock&) = delete;
    ComputeBlock& operator=(const ComputeBlock&) = delete;
    ComputeBlock(ComputeBlock&&) = delete;
    ComputeBlock& operator=(ComputeBlock&&) = delete;

    uint32_t get_cb_id() const { return cb_id; }
    uint32_t get_num_tiles() const { return num_tiles; }

    expr::Un<ExpOp, TileSource> exp() const;

private:
    uint32_t cb_id;
    uint32_t num_tiles;
};

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

    // L1 offset, for handing to a routine that addresses the semaphore directly.
    uintptr_t l1_addr() const;

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
    // Kept so l1_addr() can recompute the offset; metal's Semaphore keeps its own
    // address private.
    uint32_t id;

#if defined(IS_DM_THREAD) && IS_DM_THREAD
    // Metal's own semaphore. Spelled ::Semaphore because this class shadows it.
    ::Semaphore<ProgrammableCoreType::TENSIX> sem;
#endif
};

// ---------------------------------------------------------------------------
// Adaptors letting a ComputeBlock stand in for an expression leaf. These are the
// hooks tt/unified_math.hpp declares; they live here because this is the only
// place the math layer needs to know about a core type.
// ---------------------------------------------------------------------------

// Without this the operator+ in tt/unified_math.hpp is SFINAE'd out and
// `lhs + rhs` does not resolve.
template <>
struct is_operand<ComputeBlock> : std::true_type {};

TileSource as_node(const ComputeBlock& b);

auto relu(const ComputeBlock& b);
auto exp_(const ComputeBlock& b);

template <typename Geometry>
auto matmul(const ComputeBlock& a, const ComputeBlock& b);

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

template <int thread>
struct NocAsyncReadTx {
    explicit NocAsyncReadTx(const Storage& storage);
    NocAsyncReadTx(uint32_t cb_id, uint32_t num_tiles);

    NocAsyncReadTx(const NocAsyncReadTx&) = delete;
    NocAsyncReadTx& operator=(const NocAsyncReadTx&) = delete;
    NocAsyncReadTx(NocAsyncReadTx&&) = delete;
    NocAsyncReadTx& operator=(NocAsyncReadTx&&) = delete;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ~NocAsyncReadTx();
#endif

    // Completes the read and publishes the destination.
    Block wait() const;

    uint32_t cb_id;
    uint32_t num_tiles;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    mutable bool waited = false;
#endif
};

template <int thread>
struct NocAsyncWriteTx {
    explicit NocAsyncWriteTx(const Storage& storage);
    NocAsyncWriteTx(uint32_t cb_id, uint32_t num_tiles);

    NocAsyncWriteTx(const NocAsyncWriteTx&) = delete;
    NocAsyncWriteTx& operator=(const NocAsyncWriteTx&) = delete;
    NocAsyncWriteTx(NocAsyncWriteTx&&) = delete;
    NocAsyncWriteTx& operator=(NocAsyncWriteTx&&) = delete;

    // Releases the source: flush, then pop.
    ~NocAsyncWriteTx();

    // Optional: block until the data has LANDED at the destination.
    void wait() const;

    uint32_t cb_id;
    uint32_t num_tiles;
};

// A core-to-core copy has both halves: a local source Block to release and a
// destination Storage to publish. The destination follows the read rule (explicit
// wait()) and the source follows the write rule (the destructor).
//
// Pull: the source is the PEER's L1 and the local Block is only a handle, so the
// destructor pops it bare, and this core's own read barrier is proof the data
// landed -- it landed here.
template <int thread>
struct NocAsyncReadCoreTx {
    NocAsyncReadCoreTx(const Storage& dst, const Block& src);

    NocAsyncReadCoreTx(const NocAsyncReadCoreTx&) = delete;
    NocAsyncReadCoreTx& operator=(const NocAsyncReadCoreTx&) = delete;
    NocAsyncReadCoreTx(NocAsyncReadCoreTx&&) = delete;
    NocAsyncReadCoreTx& operator=(NocAsyncReadCoreTx&&) = delete;

    ~NocAsyncReadCoreTx();

    Block wait() const;

    uint32_t dst_cb;
    uint32_t dst_tiles;
    uint32_t src_cb;
    uint32_t src_tiles;

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
template <int thread>
struct NocAsyncWriteCoreTx {
    NocAsyncWriteCoreTx(const Storage& dst, const Block& src);
    NocAsyncWriteCoreTx(const Storage& dst, const Block& src, PhysicalMcast rect, bool receiving);

    NocAsyncWriteCoreTx(const NocAsyncWriteCoreTx&) = delete;
    NocAsyncWriteCoreTx& operator=(const NocAsyncWriteCoreTx&) = delete;
    NocAsyncWriteCoreTx(NocAsyncWriteCoreTx&&) = delete;
    NocAsyncWriteCoreTx& operator=(NocAsyncWriteCoreTx&&) = delete;

    ~NocAsyncWriteCoreTx();

    Block wait() const;

    uint32_t dst_cb;
    uint32_t dst_tiles;
    uint32_t src_cb;
    uint32_t src_tiles;

    // Meaningful only when `broadcast`.
    PhysicalMcast rect{};
    bool broadcast = false;
    bool receiving = false;

    // mutable: wait() is const across the whole API, and signalling is what a
    // wait on this handle does.
    mutable Semaphore<thread> arrived;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    mutable bool waited = false;
#endif
};

// ---------------------------------------------------------------------------
// Data movement. Each is pinned to a DM thread by its `thread` argument and
// compiles away entirely on every other thread.
// ---------------------------------------------------------------------------

// Reads `storage.num_tiles` pages into the buffer, starting at page
// `block_idx * storage.num_tiles`. The returned handle publishes them.
template <int thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(const Storage& storage, const Accessor& acc, uint32_t block_idx);

// Custom load, for routines the built-in overload cannot express. The harness
// keeps the circular-buffer protocol -- cb_reserve_back, the write pointer, and
// (via the returned handle) the read barrier and cb_push_back -- and `fn` owns
// the traffic. It is called as
//
//     fn(uint32_t l1_addr, uint32_t page_bytes)
//
// with the address of the first page and the CB's page size, and must fill
// exactly storage.num_tiles consecutive pages from there: that is the count the
// handle pushes, whatever `fn` actually wrote.
//
// `fn` must issue ONLY READS, and only on this thread's assigned NOC. The handle
// releases with noc_async_read_barrier(), which covers reads on a single NOC --
// reads issued on the other NOC, or writes, are not covered, and the push would
// then publish pages that have not landed.
//
// `fn` is only CALLED on the owning data-movement thread, but its body is
// COMPILED on all five projections, so the intrinsics it names have to resolve
// everywhere; see tt/unified_adaptor_v1.hpp.
template <int thread, typename Fn>
NocAsyncReadTx<thread> noc_load(const Storage& storage, Fn fn);

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
template <int thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(
    const Storage& storage,
    PhysicalMcast mcast,
    Semaphore<thread>& receivers_ready,
    Semaphore<thread>& data_sent,
    const Accessor& acc,
    uint32_t block_idx);

template <int thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(
    const Storage& storage,
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
template <int thread, int pair = thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(const Storage& storage, PhysicalMcast mcast, const Accessor& acc, uint32_t block_idx);

template <int thread, int pair = thread, typename Accessor>
NocAsyncReadTx<thread> noc_load(const Storage& storage, LogicalMcast mcast, const Accessor& acc, uint32_t block_idx);

// Drains a Block to a tensor. Takes the Block by value: this call consumes it.
template <int thread, typename Accessor>
NocAsyncWriteTx<thread> noc_store(Block block, const Accessor& acc, uint32_t block_idx);

// Custom store: the mirror of the custom noc_load. `fn` is called with the
// address of the first page of `block`, and covers block.num_tiles consecutive
// pages -- the count the handle pops.
//
// `fn` must issue ONLY WRITES, and only on this thread's assigned NOC. The handle
// releases the source buffer with noc_async_writes_flushed() and pops, which
// covers writes departing local L1 on a single NOC. Reads issued here, or writes
// on the other NOC, are not covered, so the pop can hand the pages back while
// they are still being sourced.
template <int thread, typename Fn>
NocAsyncWriteTx<thread> noc_store(Block block, Fn fn);

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

template <int thread>
NocAsyncReadCoreTx<thread> noc_core_read(const Storage& dst, Block src, PhysicalCoord coord, uint32_t byte_offset = 0);

template <int thread>
NocAsyncWriteCoreTx<thread> noc_core_write(
    const Storage& dst, Block src, PhysicalCoord coord, uint32_t byte_offset = 0);

// Push to a rectangle of peers. EVERY core in the exchange runs this one
// statement and takes its side from its own coordinate: the core outside `mcast`
// sends, the cores inside receive. The handshake rides on the handle's own
// reserved semaphore, so there is nothing to pass and nothing to reset.
//
// Fixes arrival notification, not addressing: the destination is still computed
// from the SENDER's local view of `dst`, so a repeated push needs the far-side
// pointers kept in step (see the NOTE above).
template <int thread>
NocAsyncWriteCoreTx<thread> noc_core_write(
    const Storage& dst, Block src, PhysicalMcast mcast, uint32_t byte_offset = 0);

template <int thread>
NocAsyncReadCoreTx<thread> noc_core_read(const Storage& dst, Block src, LogicalCoord coord, uint32_t byte_offset = 0);

template <int thread>
NocAsyncWriteCoreTx<thread> noc_core_write(const Storage& dst, Block src, LogicalCoord coord, uint32_t byte_offset = 0);

template <int thread>
NocAsyncWriteCoreTx<thread> noc_core_write(const Storage& dst, Block src, LogicalMcast mcast, uint32_t byte_offset = 0);

}  // namespace unified
}  // namespace tt
