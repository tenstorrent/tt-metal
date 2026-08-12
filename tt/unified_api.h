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
// Include <tt/unified>, not this header directly -- it selects an
// implementation and a backend binding.
//
// Layering:
//   tt/unified_expr.hpp       -- domain-free expression tree + DST allocator
//   tt/unified_math.hpp       -- leaves, ops, fusion kinds, driver strategies
//   tt/unified_api.h          -- this file: the core API surface
//   tt/unified_impl_v1.hpp    -- its definitions
//   tt/unified_adaptor_v1.hpp -- metal binding

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
// ---------------------------------------------------------------------------

struct Coord {
    uint32_t y;
    uint32_t x;
};

struct Shape {
    uint32_t h;
    uint32_t w;
};

struct Mcast {
    Coord coord;
    Shape shape;
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

    // Evaluate a compute fusion into this buffer. The loop shape is chosen by
    // the fusion's kind; see Strategy in tt/unified_math.hpp.
    template <typename Node>
    Block store(const Node& node);

    uint32_t cb_id;
    uint32_t num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

// ---------------------------------------------------------------------------
// Block -- move-only evidence that a Storage was produced into
//
// Every Block comes from an operation that has already pushed, which is what
// makes it safe to hand one to a DM thread to drain. Move-only so it reaches
// exactly one consumer; consumers take it by value.
// ---------------------------------------------------------------------------

struct Block {
    explicit Block(const Storage& storage);
    Block(uint32_t cb_id, uint32_t num_tiles);

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    // TODO: does not disengage the source, so a moved-from Block is
    // indistinguishable from a live one and a second consumer silently issues a
    // duplicate cb_pop_front.
    Block(Block&& o);
    Block& operator=(Block&& o);

    uint32_t cb_id;
    uint32_t num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

// ---------------------------------------------------------------------------
// Accumulator -- multi-block matmul
//
// The k-loop belongs to the kernel, because the operand CBs have to be waited
// and popped per block so the reader can stream them. The Accumulator holds the
// state that loop would otherwise have to carry: which buffer is the running
// total, which is the destination, and whether there is anything to reload yet.
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
// as though it were the answer, and two threads would be popping one CB (see
// the warning in api/compute/cb_api.h).
// ---------------------------------------------------------------------------

enum class AccumulatorMode {
    Dst,  // reload partials into DST and matmul on top
    L1,   // let the packer accumulate into L1 (not yet implemented)
};

template <AccumulatorMode Mode = AccumulatorMode::Dst>
class Accumulator {
public:
    Accumulator(const Storage& acc_storage, const Storage& out_storage) :
        acc_storage(acc_storage), out_storage(out_storage) {}

    // Fold one k-block into the running total. `finish` selects the pack
    // target: the accumulation buffer, or the output buffer on the last block.
    //
    // An epilogue rides on the node itself -- relu(matmul<Geom>(a, b)) appends
    // to the node's chain, and the chain is applied in place on DST on the final
    // block, when the accumulator is complete. No separate argument needed.
    //
    // Only the Block returned on the finishing call is meaningful; earlier ones
    // describe the accumulation buffer, which the next call re-consumes.
    template <typename Node>
    Block accumulate(const Node& node, bool finish) {
        static_assert(is_fpu_fusion<Node>::value, "Accumulator drives FPU fusions");
        static_assert(
            Mode == AccumulatorMode::Dst || Node::chain::empty,
            "an epilogue chain is only available in Dst mode; in L1 mode the result is assembled "
            "by the packer and never sits complete in DST for the chain to fold into");
        static_assert(Mode == AccumulatorMode::Dst, "AccumulatorMode::L1 is not implemented yet");

        Strategy<expr::kind_of_t<Node>>::run(node, acc_storage.cb_id, out_storage.cb_id, reload, finish);
        reload = !finish;
        return finish ? Block(out_storage) : Block(acc_storage);
    }

    // Reset between output blocks.
    void clear() { reload = false; }

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
    uint32_t num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

// ---------------------------------------------------------------------------
// Adaptors letting a ComputeBlock stand in for an expression leaf. These are
// the hooks tt/unified_math.hpp declares; they live here because this is the
// only place the math layer needs to know about a core type.
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
    uint32_t num_tiles;  // This could eventually be N dimensional (maybe via template params?)

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
    uint32_t num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

// A core-to-core copy has both halves: a local source Block to release and a
// destination Storage to publish. The destination follows the read rule
// (explicit wait()) and the source follows the write rule (the destructor).
//
// `SrcIsLocal` is true when this core's L1 is the data source, i.e. a push to a
// peer -- the NOC must have finished reading it before the pop, so the
// destructor flushes first. For a pull the source is the peer's L1 and the local
// Block is only a handle, so a bare pop is right.
template <int thread, bool SrcIsLocal>
struct NocAsyncCopyTx {
    NocAsyncCopyTx(const Storage& dst, const Block& src);

    NocAsyncCopyTx(const NocAsyncCopyTx&) = delete;
    NocAsyncCopyTx& operator=(const NocAsyncCopyTx&) = delete;
    NocAsyncCopyTx(NocAsyncCopyTx&&) = delete;
    NocAsyncCopyTx& operator=(NocAsyncCopyTx&&) = delete;

    ~NocAsyncCopyTx();

    Block wait() const;

    uint32_t dst_cb;
    uint32_t dst_tiles;
    uint32_t src_cb;
    uint32_t src_tiles;

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

template <int thread, typename Accessor>
Block noc_load_mcast(const Storage& storage, Mcast mcast, const Accessor& acc, uint32_t block_idx);

// Drains a Block to a tensor. Takes the Block by value: this call consumes it.
template <int thread, typename Accessor>
NocAsyncWriteTx<thread> noc_store(Block block, const Accessor& acc, uint32_t block_idx);

// ---------------------------------------------------------------------------
// Core-to-core movement: pull a peer's block into this core's Storage
// (noc_read), or push this core's block into a peer's Storage (noc_write).
//
// NOTE: reserve/push act on the *local* view of the destination CB. For a
// genuine peer buffer the far side's pointers have to be advanced too -- see
// api/remote_circular_buffer.h (remote_cb_reserve_back /
// remote_cb_push_back_and_write_pages, asymmetric between sender and receiver)
// or the explicit semaphore handshake the matmul mcast kernels use.
// ---------------------------------------------------------------------------

template <int thread>
NocAsyncCopyTx<thread, /*SrcIsLocal=*/false> noc_read(
    const Storage& storage, Block block, Coord coord, uint32_t offset);

template <int thread>
NocAsyncCopyTx<thread, /*SrcIsLocal=*/true> noc_write(
    const Storage& storage, Block block, Coord coord, uint32_t offset);

}  // namespace unified
}  // namespace tt
