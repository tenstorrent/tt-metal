// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <tt/unified/math.hpp>

namespace tt {
namespace unified {

// Public interface for unified kernels. Include <tt/unified/core> rather than
// this file directly so the appropriate intrinsic binding is installed first.
template <typename S>
struct Block;

constexpr uint32_t kNoDfb = ~0u;

template <typename S, uint32_t DfbId = kNoDfb>
class ComputeBlock;

// Physical coordinates address the NOC grid. Use xy()/yx() to make argument
// order explicit. this_core() is meaningful only on data-movement threads.
struct PhysicalCoord {
    uint32_t y;
    uint32_t x;

    static constexpr PhysicalCoord yx(uint32_t y, uint32_t x) { return PhysicalCoord(y, x); }
    static constexpr PhysicalCoord xy(uint32_t x, uint32_t y) { return PhysicalCoord(y, x); }

    static PhysicalCoord this_core();
    static PhysicalCoord origin();

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    constexpr bool operator==(PhysicalCoord o) const { return y == o.y && x == o.x; }
    constexpr bool operator!=(PhysicalCoord o) const { return !(*this == o); }

private:
    explicit constexpr PhysicalCoord(uint32_t y_in, uint32_t x_in) : y(y_in), x(x_in) {}
};

// Logical coordinates are relative to the program's worker grid and are valid
// on every projection. Conversion to a physical coordinate requires a
// data-movement thread.
struct LogicalCoord {
    uint32_t y;
    uint32_t x;

    static constexpr LogicalCoord yx(uint32_t y, uint32_t x) { return LogicalCoord(y, x); }
    static constexpr LogicalCoord xy(uint32_t x, uint32_t y) { return LogicalCoord(y, x); }

    static LogicalCoord this_core();
    static LogicalCoord origin();

    PhysicalCoord to_physical(uint32_t y_offset = 0, uint32_t x_offset = 0) const;

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    constexpr bool operator==(LogicalCoord o) const { return y == o.y && x == o.x; }
    constexpr bool operator!=(LogicalCoord o) const { return !(*this == o); }

private:
    explicit constexpr LogicalCoord(uint32_t y_in, uint32_t x_in) : y(y_in), x(x_in) {}
};

// Height and width of a rectangular core region.
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

// Inclusive physical multicast rectangle. A single coordinate converts to a
// one-core rectangle.
struct PhysicalMcast {
    PhysicalCoord start;
    PhysicalCoord end;

    PhysicalMcast(PhysicalCoord start, PhysicalCoord end) : start(start), end(end) {}

    PhysicalMcast(PhysicalCoord unit) : start(unit), end(unit) {}

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    uint32_t volume() const { return (end.y - start.y + 1) * (end.x - start.x + 1); }

    bool contains(PhysicalCoord c) const { return c.y >= start.y && c.y <= end.y && c.x >= start.x && c.x <= end.x; }

    uint32_t num_dests_excluding_sender() const { return volume() - 1; }
    uint32_t num_dests_excluding(PhysicalCoord sender) const { return volume() - (contains(sender) ? 1 : 0); }
};

// Logical multicast rectangle described by its top-left coordinate and extent.
struct LogicalMcast {
    LogicalCoord coord;
    Extent extent;

    PhysicalMcast to_physical() const;

    uint64_t get_noc_addr(uintptr_t l1_addr) const;

    uint32_t volume() const { return extent.h * extent.w; }
};

template <uint32_t DfbId>
struct DfbTag {
    static constexpr uint32_t id = DfbId;
};

template <uint32_t DfbId>
inline constexpr DfbTag<DfbId> dfb{};

// Shape-bearing base for a dataflow buffer. Callers normally declare one of
// Input, Output, or Intermediate instead. store() evaluates a compute
// expression into this buffer and returns the resulting ownership token.
template <typename S>
struct Storage {
    using shape = S;

protected:
    template <uint32_t DfbId>
    explicit Storage(DfbTag<DfbId>) : dfb_id(DfbId) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
        ASSERT(dfb_num_entries(DfbId) >= S::num_entries);
#endif

#if defined(TT_U_HAVE_DFB_TILE_GEOMETRY)
        static_assert(
            dfb_tile_rows(DfbId) == S::tile::rows,
            "this Storage's tile HEIGHT is not the one the host configured for the buffer. The "
            "kernel's Shape and the launcher's dfb(..., tile=) are two ends of one contract: a "
            "plain Shape against a sub-tile buffer computes correctly and then divides a "
            "reduce_mean by the wrong number, because elements() reads the TYPE. Wrap the shape "
            "in Tiled<Tile<rows, 32>, ...> to match, or fix the launcher");
        static_assert(
            dfb_tile_cols(DfbId) == S::tile::cols,
            "this Storage's tile WIDTH is not the one the host configured for the buffer");
#endif
    }

public:
    Storage(Storage&&) = delete;
    Storage(const Storage&) = delete;
    Storage& operator=(Storage&&) = delete;
    Storage& operator=(const Storage&) = delete;

    template <typename Node>
    Block<S> store(const Node& node) const;

    uint32_t dfb_id;

    static constexpr uint32_t num_entries = S::num_entries;
};

enum class Role : uint8_t { Input, Output, Intermediate };

inline constexpr int kNoDmThread = -1;

inline constexpr int kPairOfThread = -1;

template <Role R, int DmThread, uint32_t DfbId, typename S>
struct Endpoint : Storage<S> {
    static constexpr Role role = R;
    static constexpr int dm_thread = DmThread;
    static constexpr uint32_t dfb_id_v = DfbId;

protected:
    Endpoint() : Storage<S>(dfb<DfbId>) {}
};

// Buffer produced by the selected data-movement thread and consumed by compute.
template <int DmThread, uint32_t DfbId, typename S>
struct Input : Endpoint<Role::Input, DmThread, DfbId, S> {
    Input() = default;

    template <typename Node>
    Block<S> store(const Node& node) const = delete;
};

// Buffer produced by compute and drained by the selected data-movement thread.
template <int DmThread, uint32_t DfbId, typename S>
struct Output : Endpoint<Role::Output, DmThread, DfbId, S> {
    Output() = default;
};

// Buffer whose producer and consumer are both compute.
template <uint32_t DfbId, typename S>
struct Intermediate : Endpoint<Role::Intermediate, kNoDmThread, DfbId, S> {
    Intermediate() = default;
};

// Consecutive local-L1 pages supplied to custom load and store callbacks.
// count is the number of pages the enclosing transaction publishes or consumes.
struct L1Entries {
    uint32_t base;
    uint32_t entry_bytes;
    uint32_t count;

    uint32_t addr(uint32_t i) const { return base + i * entry_bytes; }

    uint32_t total_bytes() const { return count * entry_bytes; }
};

template <typename S, AccumulatorMode Mode = AccumulatorMode::Dst>
class Accumulator;

// Move-only evidence that a buffer has been produced. A Block must be passed to
// exactly one consumer, such as ComputeBlock, noc_store(), or noc_core_*().
template <typename S>
struct Block {
    using shape = S;

    explicit Block(const Storage<S>& storage);
    explicit Block(uint32_t dfb_id);
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ~Block();
#endif

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    Block(Block&& o);
    Block& operator=(Block&& o);

    void consume();

    uint32_t dfb_id;
    static constexpr uint32_t num_entries = S::num_entries;

private:
    struct Retained {};
    Block(const Storage<S>& storage, Retained);

    template <typename S2, AccumulatorMode M>
    friend class Accumulator;

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    bool must_consume = true;
    bool consumed = false;

    static constexpr uint32_t kMovedFrom = ~uint32_t(0);
#endif
};

// Holds a produced block across scopes or loop iterations. release() transfers
// it back to a normal Block for consumption.
template <typename S>
class RetainedBlock {
public:
    using Held = Block<S>;

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

    RetainedBlock(const RetainedBlock&) = delete;
    RetainedBlock& operator=(const RetainedBlock&) = delete;
    RetainedBlock(RetainedBlock&&) = delete;
    RetainedBlock& operator=(RetainedBlock&&) = delete;

    RetainedBlock& operator=(Held&& in);

    Held release();

private:
    void emplace(Held&& in);
    Held& get();

    alignas(Held) unsigned char buf[sizeof(Held)];

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
    bool held = false;
#endif
};

// Carries partial results across a multi-block matmul. accumulate() writes an
// intermediate result unless finish is true, in which case it writes the output.
// The optional epilogue transforms only the completed result.
template <typename S, AccumulatorMode Mode>
class Accumulator {
public:
    using shape = S;

    Accumulator(const Storage<S>& acc_storage, const Storage<S>& out_storage);

    template <typename Node, typename Epilogue = std::nullptr_t>
    Block<S> accumulate(const Node& node, bool finish, Epilogue epilogue = nullptr);

    void clear();

private:
    const Storage<S>& acc_storage;
    const Storage<S>& out_storage;
    bool reload = false;
};

// Compute-side view of a produced block. Construction waits for its pages and
// destruction releases them. Its lifetime must cover every expression using it.
template <typename S>
class ComputeBlock<S, kNoDfb> : public expr::Fluent<ComputeBlock<S, kNoDfb>> {
public:
    using shape = S;

    ComputeBlock(Block<S> block);
    ~ComputeBlock();

    ComputeBlock(const ComputeBlock&) = delete;
    ComputeBlock& operator=(const ComputeBlock&) = delete;
    ComputeBlock(ComputeBlock&&) = delete;
    ComputeBlock& operator=(ComputeBlock&&) = delete;

    uint32_t get_dfb_id() const { return dfb_id; }
    static constexpr uint32_t get_num_entries() { return S::num_entries; }

private:
    uint32_t dfb_id;
    static constexpr uint32_t num_entries = S::num_entries;
};

template <typename T, typename = void>
struct is_storable : std::false_type {};
template <typename T>
struct is_storable<T, std::void_t<node_shape_t<T>>> : std::true_type {};

template <typename S, uint32_t DfbId>
class ComputeBlock : public ComputeBlock<S, kNoDfb> {
public:
#if defined(TT_U_HAVE_DFB_TILE_GEOMETRY)
    static_assert(
        dfb_tile_rows(DfbId) == S::tile::rows,
        "this ComputeBlock's tile HEIGHT is not the one the host configured for the buffer -- "
        "the kernel's Shape and the launcher's dfb(..., tile=) disagree");
    static_assert(
        dfb_tile_cols(DfbId) == S::tile::cols,
        "this ComputeBlock's tile WIDTH is not the one the host configured for the buffer");
#endif

    template <typename Node, typename = std::enable_if_t<is_storable<Node>::value>>
    ComputeBlock(const Node& node) : ComputeBlock<S, kNoDfb>(Intermediate<DfbId, S>().store(node)) {}

    ComputeBlock(Block<S>) = delete;
};

template <typename S, uint32_t D>
struct is_operand<ComputeBlock<S, D>> : std::true_type {};

template <typename S>
ComputeBlock(Block<S>) -> ComputeBlock<S, kNoDfb>;

template <typename S>
struct is_operand<ComputeBlock<S>> : std::true_type {};

template <typename S>
TileSource<S> as_node(const ComputeBlock<S>& b);

// Build compute expressions from live ComputeBlocks. Expressions execute when
// passed to Storage::store(), noc_store(), or a buffer-owning ComputeBlock.
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

template <TransposeB Tr = TransposeB::No, typename SA, typename SB>
auto matmul(const ComputeBlock<SA>& a, const ComputeBlock<SB>& b);

template <Axis A, typename S>
Broadcast<A, S> bcast(const ComputeBlock<S>& v);

template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Sum, expr::UnaryChain<>> reduce_sum(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler);

template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Max, expr::UnaryChain<>> reduce_max(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler);

template <ReduceAxis Axis, typename SB, typename SC>
ReduceNode<SB, Axis, ReducePool::Avg, expr::UnaryChain<>> reduce_mean(
    const ComputeBlock<SB>& b, const ComputeBlock<SC>& scaler);

template <typename T>
struct is_compute_block : std::false_type {};

template <typename S>
struct is_compute_block<ComputeBlock<S>> : std::true_type {};

// Invoke a custom compute callback with the DFB ids of its leading
// ComputeBlock arguments. The callback owns all raw compute setup and cleanup.
template <typename... Ts>
void custom_compute(Ts&&... ts);

#if defined(TT_UNIFIED_MCAST_SEM_BASE)
inline constexpr bool kMcastSemsReserved = true;
inline constexpr uint32_t kMcastSemBase = TT_UNIFIED_MCAST_SEM_BASE;
#else
inline constexpr bool kMcastSemsReserved = false;
inline constexpr uint32_t kMcastSemBase = 0;
#endif

#if defined(TT_UNIFIED_MCAST_SEM_FIRST) && defined(TT_UNIFIED_MCAST_SEM_LAST)
static_assert(
    kMcastSemBase == static_cast<uint32_t>(TT_UNIFIED_MCAST_SEM_FIRST),
    "the harness's predicted multicast semaphore base does not match the id the host assigned");
static_assert(
    kMcastSemBase + 2 * 2 + 2 - 1 == static_cast<uint32_t>(TT_UNIFIED_MCAST_SEM_LAST),
    "the reserved multicast semaphores are not contiguous -- every id below is derived from "
    "kMcastSemBase by arithmetic, so a gap in the run silently retargets a handshake");
#endif

template <int thread>
inline constexpr uint32_t kMcastReadySem = kMcastSemBase + 2 * thread;
template <int thread>
inline constexpr uint32_t kMcastSentSem = kMcastSemBase + 2 * thread + 1;

template <int thread>
inline constexpr uint32_t kCopyArrivedSem = kMcastSemBase + 4 + thread;

#if defined(TT_UNIFIED_CORE_GRID_H) && defined(TT_UNIFIED_CORE_GRID_W)
inline constexpr bool kCoreGridKnown = true;
inline constexpr uint32_t kCoreGridH = TT_UNIFIED_CORE_GRID_H;
inline constexpr uint32_t kCoreGridW = TT_UNIFIED_CORE_GRID_W;
#else
inline constexpr bool kCoreGridKnown = false;
inline constexpr uint32_t kCoreGridH = 1;
inline constexpr uint32_t kCoreGridW = 1;
#endif

#if defined(TT_UNIFIED_CORE_GRID_EXACT)
inline constexpr bool kCoreGridExact = true;
#else
inline constexpr bool kCoreGridExact = false;
#endif

inline constexpr uint32_t kReduceScalerOne = 0x3F803F80u;

inline uint32_t bf16_pair(float v) {
    uint32_t bits = 0;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    const uint32_t half = bits >> 16;
    return (half << 16) | half;
}

// Barrier all cores in region on the selected data-movement thread. The
// zero-argument overload uses the configured worker grid.
template <int thread>
void synchronize_cores(PhysicalMcast region);

template <int thread>
void synchronize_cores(LogicalMcast region);

template <int thread>
void synchronize_cores();

// Thread-bound wrapper around a local NOC semaphore. wait() requires equality;
// wait_min() accepts any value at least as large as the requested value.
template <int thread>
class Semaphore {
public:
    explicit Semaphore(uint32_t semaphore_id);

    uint32_t semaphore_id() const;

    Semaphore& wait(uint32_t value);
    Semaphore& wait_min(uint32_t value);

    Semaphore& set(uint32_t value);

    Semaphore& inc_remote(PhysicalCoord coord, uint32_t value = 1);
    Semaphore& inc_remote(LogicalCoord coord, uint32_t value = 1);

    Semaphore& inc_mcast(PhysicalMcast mcast, uint32_t value = 1);
    Semaphore& inc_mcast(LogicalMcast mcast, uint32_t value = 1);

    Semaphore& set_mcast(PhysicalMcast mcast);
    Semaphore& set_mcast(LogicalMcast mcast);

private:
    uint32_t id;

#if defined(IS_DM_THREAD) && IS_DM_THREAD
    ::Semaphore<ProgrammableCoreType::TENSIX> sem;
#endif
};

#if defined(TT_UNIFIED_MCAST_ZONES) && defined(IS_DM_THREAD) && IS_DM_THREAD
#define TT_U_ZONE(name) DeviceZoneScopedN(name)
#else
#define TT_U_ZONE(name) ((void)0)
#endif

// Asynchronous transaction handles are single-use synchronization objects.
// Read and multicast handles return a produced Block from wait(); write handles
// release their source pages when wait() completes.
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

    Block<S> wait() const;

    uint32_t dfb_id;
    static constexpr uint32_t num_entries = S::num_entries;

    uint8_t noc_id = noc_index;

    mutable Semaphore<thread> data_sent;
    bool sender;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    mutable bool waited = false;
#endif
};

template <int thread, typename S>
struct NocAsyncReadTx {
    using shape = S;

    explicit NocAsyncReadTx(const Storage<S>& storage);
    explicit NocAsyncReadTx(uint32_t dfb_id);

    NocAsyncReadTx(const NocAsyncReadTx&) = delete;
    NocAsyncReadTx& operator=(const NocAsyncReadTx&) = delete;
    NocAsyncReadTx(NocAsyncReadTx&&) = delete;
    NocAsyncReadTx& operator=(NocAsyncReadTx&&) = delete;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    ~NocAsyncReadTx();
#endif

    Block<S> wait() const;

    uint32_t dfb_id;
    static constexpr uint32_t num_entries = S::num_entries;

    uint8_t noc_id = noc_index;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    mutable bool waited = false;
#endif
};

template <int thread, typename S>
struct NocAsyncWriteTx {
    using shape = S;

    explicit NocAsyncWriteTx(const Storage<S>& storage);
    explicit NocAsyncWriteTx(uint32_t dfb_id);

    NocAsyncWriteTx(const NocAsyncWriteTx&) = delete;
    NocAsyncWriteTx& operator=(const NocAsyncWriteTx&) = delete;
    NocAsyncWriteTx(NocAsyncWriteTx&&) = delete;
    NocAsyncWriteTx& operator=(NocAsyncWriteTx&&) = delete;

    ~NocAsyncWriteTx();

    void wait() const;

    uint32_t dfb_id;
    static constexpr uint32_t num_entries = S::num_entries;

    uint8_t noc_id = noc_index;
};

template <int thread, typename D, typename S>
struct NocAsyncReadCoreTx {
    static_assert(
        S::num_entries <= D::num_entries,
        "a core-to-core copy's source does not fit its destination -- the source Block has more pages "
        "than the destination Storage");
    static_assert(
        D::num_entries % S::num_entries == 0,
        "a core-to-core copy's destination is not a whole multiple of its source -- a gather deposits "
        "one source-sized slot per writer, so a ragged destination cannot be addressed by byte_offset");

    NocAsyncReadCoreTx(const Storage<D>& dst, const Block<S>& src);

    NocAsyncReadCoreTx(const NocAsyncReadCoreTx&) = delete;
    NocAsyncReadCoreTx& operator=(const NocAsyncReadCoreTx&) = delete;
    NocAsyncReadCoreTx(NocAsyncReadCoreTx&&) = delete;
    NocAsyncReadCoreTx& operator=(NocAsyncReadCoreTx&&) = delete;

    ~NocAsyncReadCoreTx();

    Block<D> wait() const;

    uint32_t dst_dfb;
    static constexpr uint32_t dst_entries = D::num_entries;
    uint32_t src_dfb;
    static constexpr uint32_t src_entries = S::num_entries;

    uint8_t noc_id = noc_index;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    mutable bool waited = false;
#endif
};

template <int thread, typename D, typename S>
struct NocAsyncWriteCoreTx {
    static_assert(
        S::num_entries <= D::num_entries,
        "a core-to-core copy's source does not fit its destination -- the source Block has more pages "
        "than the destination Storage");
    static_assert(
        D::num_entries % S::num_entries == 0,
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

    uint32_t dst_dfb;
    static constexpr uint32_t dst_entries = D::num_entries;
    uint32_t src_dfb;
    static constexpr uint32_t src_entries = S::num_entries;

    uint8_t noc_id = noc_index;

    mutable Semaphore<thread> arrived;
    bool reader;

#if defined(IS_DM_THREAD) && IS_DM_THREAD && defined(ASSERT_ENABLED) && ASSERT_ENABLED
    mutable bool waited = false;
#endif
};

// Load one shape-sized block from tensor pages into storage. block_idx selects
// the block in the accessor. Input overloads derive the DM thread from storage.
template <int thread, typename S, typename Accessor>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, const Accessor& acc, uint32_t block_idx);

template <int T, uint32_t Id, typename S, typename Accessor>
NocAsyncReadTx<T, S> noc_load(const Input<T, Id, S>& storage, const Accessor& acc, uint32_t block_idx);

// Custom load. fn(L1Entries) must issue reads that fill every supplied page on
// the selected thread's NOC. The returned handle completes and publishes them.
template <int thread, typename S, typename Fn>
NocAsyncReadTx<thread, S> noc_load(const Storage<S>& storage, Fn fn);

template <int T, uint32_t Id, typename S, typename Fn>
NocAsyncReadTx<T, S> noc_load(const Input<T, Id, S>& storage, Fn fn);

// Multicast load. One core reads the block and distributes it to every core in
// mcast. Explicit semaphore overloads use caller-managed handshake semaphores;
// pair-based overloads use the reserved pair selected by the template argument.
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

template <int thread, int pair = thread, typename S, typename Accessor>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage, PhysicalMcast mcast, const Accessor& acc, uint32_t block_idx);

template <int pair = kPairOfThread, int T, uint32_t Id, typename S, typename Accessor>
NocAsyncMcastTx<T, S> noc_load(
    const Input<T, Id, S>& storage, PhysicalMcast mcast, const Accessor& acc, uint32_t block_idx);

template <int thread, int pair = thread, typename S, typename Fn>
NocAsyncMcastTx<thread, S> noc_load(const Storage<S>& storage, PhysicalMcast mcast, Fn fn);

template <int pair = kPairOfThread, int T, uint32_t Id, typename S, typename Fn>
NocAsyncMcastTx<T, S> noc_load(const Input<T, Id, S>& storage, PhysicalMcast mcast, Fn fn);
template <int thread, int pair = thread, typename S, typename Fn>
NocAsyncMcastTx<thread, S> noc_load(const Storage<S>& storage, LogicalMcast mcast, Fn fn);
template <int pair = kPairOfThread, int T, uint32_t Id, typename S, typename Fn>
NocAsyncMcastTx<T, S> noc_load(const Input<T, Id, S>& storage, LogicalMcast mcast, Fn fn);

template <int thread, int pair = thread, typename S, typename Accessor>
NocAsyncMcastTx<thread, S> noc_load(
    const Storage<S>& storage, LogicalMcast mcast, const Accessor& acc, uint32_t block_idx);

template <int pair = kPairOfThread, int T, uint32_t Id, typename S, typename Accessor>
NocAsyncMcastTx<T, S> noc_load(
    const Input<T, Id, S>& storage, LogicalMcast mcast, const Accessor& acc, uint32_t block_idx);

// Fill and publish the persistent one-page scaler used by reductions.
// value_bits contains the packed value in the buffer's element format.
template <int thread, typename S>
Block<S> fill_reduce_scaler(const Storage<S>& scaler, uint32_t value_bits = kReduceScalerOne);

template <int T, uint32_t Id, typename S>
Block<S> fill_reduce_scaler(const Input<T, Id, S>& scaler, uint32_t value_bits = kReduceScalerOne);

// Store a produced block to tensor pages. block_idx selects the destination
// block. Storage-leading overloads evaluate node into storage before writing.
template <int thread, typename S, typename Accessor>
NocAsyncWriteTx<thread, S> noc_store(Block<S> block, const Accessor& acc, uint32_t block_idx);

template <int thread, typename S, typename Accessor, typename Node>
NocAsyncWriteTx<thread, S> noc_store(
    const Storage<S>& storage, const Node& node, const Accessor& acc, uint32_t block_idx);

template <int T, uint32_t Id, typename S, typename Accessor, typename Node>
NocAsyncWriteTx<T, S> noc_store(
    const Output<T, Id, S>& storage, const Node& node, const Accessor& acc, uint32_t block_idx);

// Custom store. fn(L1Entries) must issue writes for the supplied source pages
// on the selected thread's NOC. The returned handle completes the writes and
// releases the source pages.
template <int thread, typename S, typename Fn>
NocAsyncWriteTx<thread, S> noc_store(Block<S> block, Fn fn);

template <int thread, typename S, typename Node, typename Fn>
NocAsyncWriteTx<thread, S> noc_store(const Storage<S>& storage, const Node& node, Fn fn);

template <int T, uint32_t Id, typename S, typename Node, typename Fn>
NocAsyncWriteTx<T, S> noc_store(const Output<T, Id, S>& storage, const Node& node, Fn fn);

// Copy blocks between cores. coord or dst_range identifies the peer side;
// byte_offset is within the peer buffer. For writes, write_predicate selects
// participating writers and wait(num_writers) must match that count. Callers
// must keep peer buffer pointers synchronized across repeated exchanges.
template <int thread, typename D, typename S>
NocAsyncReadCoreTx<thread, D, S> noc_core_read(
    const Storage<D>& dst, Block<S> src, PhysicalCoord coord, uint32_t byte_offset = 0);

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

template <int T, uint32_t Id, typename D, typename S>
NocAsyncWriteCoreTx<T, D, S> noc_core_write(
    const Input<T, Id, D>& dst, Block<S> src, LogicalCoord coord, bool write_predicate, uint32_t byte_offset = 0);

template <int thread, typename D, typename S>
NocAsyncWriteCoreTx<thread, D, S> noc_core_write(
    const Storage<D>& dst, Block<S> src, LogicalMcast dst_range, bool write_predicate, uint32_t byte_offset = 0);

}  // namespace unified
}  // namespace tt
