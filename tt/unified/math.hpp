// SPDX-License-Identifier: Apache-2.0
//
// Compute fusions: what a leaf is, what each op emits, and the per-kind driver
// strategies. See <tt/unified/core> for the layering.
//
// This header deliberately does not depend on the core types. Leaves and nodes
// carry raw circular-buffer ids, so the dependency runs one way: tt/unified/api.h
// includes this and supplies the thin ComputeBlock adaptors.
//
// A fusion KIND selects the driver strategy -- the shape of the enclosing loop,
// and which hardware unit owns the DST register file. That is the axis along
// which fusions genuinely differ, not the op set.
//
//   SFPUFusion -- the SFPU indexes DST freely, so an arbitrary expression tree
//                 can be allocated across it. One pass per tile, pack each result.
//
//   FPUFusion  -- the FPU maximises DST. matmul_block self-increments dst_index
//                 from 0 across out_subblock_num_tiles, so there is nothing left
//                 to allocate: only a *unary* epilogue can fuse, applied in place
//                 on the final accumulation step. The hardware says the same --
//                 matmul's activation is compiled out when FUSE_BIAS is set, and
//                 bias spills through an intermediate CB instead
//                 (bmm_large_block_zm_fused_bias_activation.cpp:384).
//
//   ReduceFusion -- metal's reduce folds a whole tile grid down an axis, within
//                 and across tiles, accumulating into ONE DST slot. So there is
//                 nothing to allocate either, and again only a unary epilogue can
//                 fuse. Unlike the other two it has no DST budget to check.
//
// Adding an FPU op later means adding a node type that declares
// `using fusion_kind = FPUFusion;` -- Strategy<FPUFusion> is reused as-is.

#pragma once

#include <cstdint>
#include <type_traits>

#include <tt/unified/expr.hpp>
#include <tt/unified/shape.hpp>

// Every op body below is guarded on IS_COMPUTE_THREAD, which a binding defines.
// Without one they would all silently compile to nothing, so refuse instead.
#if !defined(IS_COMPUTE_THREAD) && !defined(IS_DM_THREAD)
#error "include <tt/unified/core> (or a binding) before tt/unified/math.hpp"
#endif

namespace tt {
namespace unified {

// Usable DST tiles per acquire.
//
// DST holds 16 tiles, but under the default DstSync::SyncHalf the register file
// is banked in two and only ONE half is addressable between a tile_regs_acquire
// and its release -- so the budget is 8. ttnn's own matmul never picks a larger
// subblock: SUBBLOCK_HW_CHOICES tops out at 8 ({4,2},{2,4},{8,1},{1,8}).
//
// Exceeding it is not a clean failure. A 16-tile subblock still round-trips in
// Dst mode, because math writes and pack read the same wrong mapping and it
// cancels; L1 mode exposes it, because the packer's read-modify-write of L1
// depends on the absolute DST->L1 mapping and only the upper half is right --
// measured as tiles 0..7 overwritten instead of accumulated.
//
// Halves again to 4 under fp32 accumulate (see reg_api.h), which this model does
// not enable; raising it to 16 would require dst_full_sync_en.
inline constexpr uint32_t kMaxDstTiles = 8;

// --- Leaves and ops ---

// One tile out of a circular buffer, copied into a DST slot. The allocator picks
// the slot, not the caller -- that is what keeps operands from clobbering
// intermediates.
template <typename S>
struct TileSource : expr::Fluent<TileSource<S>> {
    using is_expr_node = std::true_type;
    using shape = S;
    static constexpr uint32_t need = 1;

    uint32_t cb_id;

    // `reconfigure` re-points the unpacker's srcA data format at THIS leaf's buffer.
    // Without it a tree whose leaves live in circular buffers of different formats is
    // silently wrong: copy_tile does not carry a format, and
    // copy_tile_to_dst_init_short explicitly "does not reconfigure the unpacker data
    // types". The one-argument form is used because it needs no previous operand, so a
    // leaf stays self-sufficient and nothing has to track hardware state across the
    // walk. ttnn's binary_ng threads the previous operand instead, which lets it use
    // the cheaper conditional form -- possible there because it batches all of one
    // operand's tiles together, which this per-tile loop does not.
    //
    // Uniform TILE GEOMETRY is a separate assumption, and one the model already makes:
    // every circular buffer here holds exactly one 32x32 tile per page.
    void emit(uint32_t dst, uint32_t tile, bool reconfigure) const {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        if (reconfigure) {
            ckernel::reconfig_data_format_srca(cb_id);
            ckernel::copy_tile_to_dst_init_short(cb_id);
        }
        ckernel::copy_tile(cb_id, tile, dst);
#else
        (void)dst;
        (void)tile;
        (void)reconfigure;
#endif
    }
};

struct AddOp {
    static void apply(uint32_t lhs, uint32_t rhs, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        ckernel::add_binary_tile_init();
        ckernel::add_binary_tile(lhs, rhs, out);
#else
        (void)lhs;
        (void)rhs;
        (void)out;
#endif
    }
};

// Sub and Div are NOT commutative, and nothing downstream reorders them: the
// allocator evaluates lhs into `base` and rhs into `base + 1` in that order, and
// metal's idst0/idst1 are its first and second operand. See the note on the
// heavier-child optimisation in tt/unified/expr.hpp -- it is left out precisely
// because it would need commutativity.
//
// No apply_in_place: a binary has two operands, so it cannot be a link in a
// unary epilogue chain the way relu and exp can.

struct SubOp {
    static void apply(uint32_t lhs, uint32_t rhs, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        ckernel::sub_binary_tile_init();
        ckernel::sub_binary_tile(lhs, rhs, out);
#else
        (void)lhs;
        (void)rhs;
        (void)out;
#endif
    }
};

struct MulOp {
    static void apply(uint32_t lhs, uint32_t rhs, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        ckernel::mul_binary_tile_init();
        ckernel::mul_binary_tile(lhs, rhs, out);
#else
        (void)lhs;
        (void)rhs;
        (void)out;
#endif
    }
};

struct DivOp {
    static void apply(uint32_t lhs, uint32_t rhs, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        ckernel::div_binary_tile_init();
        ckernel::div_binary_tile(lhs, rhs, out);
#else
        (void)lhs;
        (void)rhs;
        (void)out;
#endif
    }
};

struct ExpOp {
    static void apply(uint32_t src, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;  // == out; SFPU unaries work in place
        ckernel::exp_tile_init();
        ckernel::exp_tile(out);
#else
        (void)src;
        (void)out;
#endif
    }
    static void apply_in_place(uint32_t slot) { apply(slot, slot); }
};

// NOTE: a reduction is not here among the ops, and not by omission: it collapses
// the tile loop instead of running inside it, so it is a third KIND. See
// ReduceNode and Strategy<ReduceFusion> below.

// A unary usable in either kind: a node in an SFPU tree, or a link in an FPU
// node's epilogue chain. Both run on the SFPU against DST, so one implementation
// serves both.
//
// The `*_tile_init()` calls are inline rather than hoisted: they are cheap, and
// metal kernels routinely re-init per use (see SFPU_OP_CHAIN_0 in
// tests/.../compute/eltwise_sfpu.cpp). Worth hoisting if it shows in a profile.
struct ReluOp {
    static void apply(uint32_t src, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;  // == out; SFPU unaries work in place
        ckernel::relu_tile_init();
        ckernel::relu_tile(out);
#else
        (void)src;
        (void)out;
#endif
    }

    static void apply_in_place(uint32_t slot) { apply(slot, slot); }
};

// recip is what softmax normalises with, rsqrt what RMSNorm does. All three
// take metal's own defaults for the approximation and legacy_compat template
// parameters, the way exp and relu do. recip is full-accuracy on Float32,
// Float16_b and Bfp8_b only.
struct RecipOp {
    static void apply(uint32_t src, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;  // == out; SFPU unaries work in place
        ckernel::recip_tile_init();
        ckernel::recip_tile(out);
#else
        (void)src;
        (void)out;
#endif
    }

    static void apply_in_place(uint32_t slot) { apply(slot, slot); }
};

struct SqrtOp {
    static void apply(uint32_t src, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;  // == out; SFPU unaries work in place
        ckernel::sqrt_tile_init();
        ckernel::sqrt_tile(out);
#else
        (void)src;
        (void)out;
#endif
    }

    static void apply_in_place(uint32_t slot) { apply(slot, slot); }
};

struct RsqrtOp {
    static void apply(uint32_t src, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;  // == out; SFPU unaries work in place
        ckernel::rsqrt_tile_init();
        ckernel::rsqrt_tile(out);
#else
        (void)src;
        (void)out;
#endif
    }

    static void apply_in_place(uint32_t slot) { apply(slot, slot); }
};

// --- Kinds and FPU nodes ---

using SFPUFusion = expr::TreeKind;

struct FPUFusion {};

struct ReduceFusion {};

// How B's tiles are read. This is the ONE matmul property that is not derivable from
// the operand shapes, so it is the only thing left to state.
//
// It is a PER-TILE transpose and nothing more: each 32x32 tile of B is transposed as it
// is unpacked, and the tile GRID is untouched. Metal's own wording is "transpose
// operation on tiles in B", and ttnn's SDPA relies on exactly this -- its indexing is
// byte-identical with the flag on or off.
//
// So this alone does NOT give you B-transpose. A true Bt needs both halves:
//
//   per-tile   this flag
//   tile grid  the READER, placing page (r, c) of B at slot (c, r)
//
// For Q@Kt that means K arrives grid-transposed -- so its shape here is genuinely
// kt x ct and the geometry below infers correctly -- and this flag supplies the rest.
// Verified on silicon: the pair matches torch's q @ k.T to 0.006-0.007 across four
// shapes, while the flag alone matches neither q@k nor q@k.T once B is wider than one
// tile. An A-transpose is NOT symmetric with this and has no flag: ttnn does it with a
// separate materialised pass into another circular buffer.
enum class TransposeB { No, Yes };

// The geometry a matmul runs at, DERIVED from its operands rather than declared.
// Names follow matmul_block's own parameters: A is rt_dim x kt_dim tiles, B is
// kt_dim x ct_dim, C is rt_dim x ct_dim.
//
// Nothing writes this type; it falls out of the two shapes. What used to be five
// template parameters is now zero:
//
//   rt_dim, ct_dim, kt_dim  read off the operands, and their agreement is checked
//   num_blocks              never used here -- it was only ever a kernel loop bound
//   in1_row_stride          B's own width, which is what it always was
//
// `in1_row_stride` is how far to step in B's circular buffer to move down one k row.
// It is B's full block width, which equals ct_dim as long as B holds exactly the
// output's columns. Subblocking B -- several output subblocks side by side in one
// buffer -- would make them differ, and would then be expressed by giving B a wider
// shape than the output takes.
template <typename SA, typename SB, TransposeB Tr = TransposeB::No>
struct MatmulGeometry {
    static_assert(
        SA::cols == SB::rows, "matmul inner dimension disagrees: operand A's columns must equal operand B's rows");
    static_assert(SA::leading == SB::leading, "matmul operands disagree on their leading (batch) extent");

    static constexpr uint32_t rt_dim = SA::rows;  // output rows  (A rows)
    static constexpr uint32_t ct_dim = SB::cols;  // output cols  (B cols)
    static constexpr uint32_t kt_dim = SA::cols;  // inner dim
    static constexpr uint32_t in1_row_stride = SB::cols;
    static constexpr uint32_t out_subblock_num_tiles = rt_dim * ct_dim;

    // What metal's matmul_block wants: any non-zero value means transpose.
    static constexpr uint32_t transpose = (Tr == TransposeB::Yes) ? 1u : 0u;

    // The output block, with any leading extent carried through from A.
    using out_shape = with_hw<SA, rt_dim, ct_dim>;
};

// No bias. A real cb id could be 0, so the sentinel has to be out of range.
inline constexpr uint32_t kNoBias = ~uint32_t(0);

template <typename SA, typename SB, TransposeB Tr, typename Chain>
struct MatmulNode : expr::Fluent<MatmulNode<SA, SB, Tr, Chain>> {
    using fusion_kind = FPUFusion;
    using lhs_shape = SA;
    using rhs_shape = SB;
    static constexpr TransposeB transpose_b = Tr;
    using geometry = MatmulGeometry<SA, SB, Tr>;
    using chain = Chain;
    using shape = typename geometry::out_shape;

    // Fuse a bias, added ONCE to the finished total -- never per k-block, which
    // would scale it by the block count. It lands before the epilogue chain, so
    // matmul(a, b).bias(v).relu() is relu(A@B + v), the usual fusion.
    //
    // `operand` is duck-typed rather than named: this header does not know the
    // core types, and all it needs is the circular buffer behind one. Pass a
    // ComputeBlock held at KERNEL scope -- the bias is read by every finishing
    // block and must not be popped until the kernel ends. See unified_kernels.
    template <typename Operand>
    auto bias(const Operand& operand) const {
        // A bias is one row broadcast down the output block, so its shape is fixed by
        // the geometry. This was a runtime ASSERT on the page count, which could only
        // fire in an asserts-enabled build and only once the kernel ran.
        static_assert(
            same_shape_v<typename Operand::shape, Shape<1, geometry::ct_dim>>,
            "a fused bias must be Shape<1, ct_dim> -- one row of the output block's width");
        // The bias is ct_dim tiles, one per output column. Fewer and the finishing
        // pass reads past what was pushed -- whatever is next in that buffer, with
        // nothing to notice; more and the kernel and the geometry disagree about
        // the shape. Checked here rather than in the strategy because this is where
        // the operand's page count is still in hand.
        MatmulNode<SA, SB, Tr, Chain> out{{}, in0_cb, in1_cb, operand.get_cb_id()};
        return out;
    }

    uint32_t in0_cb;
    uint32_t in1_cb;
    uint32_t bias_cb = kNoBias;
};

// --- Reduction ---
//
// A reduction is not an op: it collapses the tile loop rather than running inside
// it, and metal wants its dimension as a template argument. So it gets its own
// kind and driver, reducing WITHIN each tile and ACROSS the block at once.
//
// The axis names say which dimension the op ACTS ON: the one a reduction collapses,
// and the one a broadcast expands again. One name for both halves, so a reduction and
// the broadcast that undoes it read alike and cannot drift apart:
//
//     reduce_max<Axis::Cols>(x, one)      collapses Cols  ->  Shape<Ht, 1>
//     x - bcast<Axis::Cols>(m)            expands Cols    ->  Shape<Ht, Wt>
//
// Metal names the survivor instead for reductions, which is the usual source of error,
// so the mapping is written out:
//
//   Rows -> REDUCE_COL     Ht x Wt -> 1 x Wt, each column's value in row 0
//   Cols -> REDUCE_ROW     Ht x Wt -> Ht x 1, each row's value in column 0
//   Both -> REDUCE_SCALAR  Ht x Wt -> 1 x 1,  the value at [0, 0]
//
// reduce_init programs the packer's edge masks so every datum that is NOT part of
// the result is written out as zero -- so a 4x4-tile block reduced over Rows
// leaves one valid row spread across 1x4 tiles, and nothing else.
enum class Axis { Rows, Cols, Both };

// The name this had when reductions were the only op with an axis.
using ReduceAxis = Axis;

// Ours rather than metal's PoolType, because this names a template argument of a
// type that appears in shared kernel code, and PoolType only exists on a compute
// build.
enum class ReducePool { Sum, Avg, Max };

// The shape a reduction leaves behind: the collapsing axis becomes 1 and everything
// else is preserved, leading extents included. This is exactly what the destination
// Storage must hold.
template <typename S, Axis A>
using reduce_shape = with_hw<S, (A == Axis::Cols ? S::rows : 1), (A == Axis::Rows ? S::cols : 1)>;

// The shape a broadcast along `A` requires of its vector, given the block it applies to.
// Identical to what a reduction along the same axis PRODUCES, which is what makes a
// reduce and the broadcast that undoes it check each other:
//
//   Rows -> Shape<1, cols>    a row,    replicated down the rows
//   Cols -> Shape<rows, 1>    a column, replicated across the columns
//   Both -> Shape<1, 1>       a scalar
//
// The axis has to be DECLARED rather than read off the vector, because a Shape counts
// TILES and the distinction is inside one: a single tile holding a row, a column, or a
// lone value at [0, 0] is Shape<1, 1> in all three cases. The shapes only differ when the
// block's tile extents do, so inferring the axis would work for a 4x6 block and collide
// for 1x6, 4x1 or 1x1.
template <typename SB, Axis A>
using bcast_vec_shape = reduce_shape<SB, A>;

// Never true, but only once a template argument is substituted -- so a static_assert
// using it fires on instantiation rather than on definition. Used by the guards that
// reject an operand a hardware path cannot take.
template <typename T>
struct always_false : std::false_type {};

// --- Broadcast ---
//
// A broadcast reads a BLOCK and a VECTOR from two circular buffers and expands the vector
// along one axis as it goes. That is why it cannot be an expression-tree node: a tree
// leaf copies whole tiles into DST, and what is needed here is the replication of one row
// or one column WITHIN each tile, which only the unpacker's broadcast mode does.
//
// Its precedent is already in this file: Strategy<FPUFusion>::bias_finish is a tile loop
// of add_tiles_bcast_rows, and a fused bias is the Rows case of exactly this op.

struct BcastFusion {};

// Which metal call each (op, axis) pair lowers to. Spelled out rather than composed,
// because the init_short names are NOT uniform: add's scalar form omits `tiles_` while
// sub's and mul's include it.
//
// The ROW/COL wording is metal's and describes the VECTOR's shape, not the axis walked --
// established by the fused bias, which passes a row through add_tiles_bcast_rows and adds
// it per column. Metal's own prose contradicts itself on this (its COL paragraph says
// both "a filled 0-column" and "C[h,w] = A[h,w] + B[w]"), so the mapping below is what
// test_unified_bcast.py measures rather than assumes.
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
template <typename Op, Axis A>
struct BcastOps;

#define TT_UNIFIED_BCAST_OPS(OpType, rows_init, rows_op, cols_init, cols_op, sc_init, sc_op) \
    template <>                                                                              \
    struct BcastOps<OpType, Axis::Rows> {                                                    \
        static void init(uint32_t b, uint32_t v) { ckernel::rows_init(b, v); }               \
        static void apply(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {    \
            ckernel::rows_op(b, v, bt, vt, d);                                               \
        }                                                                                    \
    };                                                                                       \
    template <>                                                                              \
    struct BcastOps<OpType, Axis::Cols> {                                                    \
        static void init(uint32_t b, uint32_t v) { ckernel::cols_init(b, v); }               \
        static void apply(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {    \
            ckernel::cols_op(b, v, bt, vt, d);                                               \
        }                                                                                    \
    };                                                                                       \
    template <>                                                                              \
    struct BcastOps<OpType, Axis::Both> {                                                    \
        static void init(uint32_t b, uint32_t v) { ckernel::sc_init(b, v); }                 \
        static void apply(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {    \
            ckernel::sc_op(b, v, bt, vt, d);                                                 \
        }                                                                                    \
    };

TT_UNIFIED_BCAST_OPS(
    AddOp,
    add_bcast_rows_init_short,
    add_tiles_bcast_rows,
    add_bcast_cols_init_short,
    add_tiles_bcast_cols,
    add_bcast_scalar_init_short,
    add_tiles_bcast_scalar)
TT_UNIFIED_BCAST_OPS(
    SubOp,
    sub_bcast_rows_init_short,
    sub_tiles_bcast_rows,
    sub_bcast_cols_init_short,
    sub_tiles_bcast_cols,
    sub_tiles_bcast_scalar_init_short,
    sub_tiles_bcast_scalar)
TT_UNIFIED_BCAST_OPS(
    MulOp,
    mul_bcast_rows_init_short,
    mul_tiles_bcast_rows,
    mul_bcast_cols_init_short,
    mul_tiles_bcast_cols,
    mul_tiles_bcast_scalar_init_short,
    mul_tiles_bcast_scalar)

#undef TT_UNIFIED_BCAST_OPS
#endif

// The marker that carries the axis. `bcast<Axis::Cols>(v)` says what the vector IS, which
// is the one thing its shape cannot say; the shape is then checked against it in
// BcastNode. Not an operand -- is_operand stays false for it -- so the SFPU operators
// cannot swallow one and the broadcast overloads are unambiguous.
template <Axis A, typename S>
struct Broadcast {
    static constexpr Axis axis = A;
    using shape = S;

    uint32_t cb_id;
};

template <typename SB, Axis A>
struct BcastNodeChecks {
    static_assert(SB::leading == 1, "broadcasting a shape with a leading (batch) extent is not implemented");
};

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
struct BcastNode : expr::Fluent<BcastNode<Op, A, SB, SV, Chain>>, BcastNodeChecks<SB, A> {
    using fusion_kind = BcastFusion;
    using op = Op;
    static constexpr Axis axis = A;
    using block_shape = SB;
    using vec_shape = SV;
    using chain = Chain;

    // A broadcast is shape-preserving, so Storage::store's conformance check and
    // node_shape's primary template both work with nothing added.
    using shape = SB;

    static_assert(
        same_shape_v<SV, bcast_vec_shape<SB, A>>,
        "the broadcast vector's shape does not match the axis it declares: Axis::Rows needs "
        "Shape<1, cols>, Axis::Cols needs Shape<rows, 1>, Axis::Both needs Shape<1, 1>, all "
        "relative to the block");

    // Which vector tile pairs with block tile `t`, the block being row-major.
    static constexpr uint32_t vec_tile(uint32_t t) {
        return A == Axis::Rows ? t % SB::cols : (A == Axis::Cols ? t / SB::cols : 0);
    }

    uint32_t block_cb;
    uint32_t vec_cb;
};

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
// Ours -> metal's, in the one place metal's enums are nameable. Note the axis
// crossover: collapsing rows is REDUCE_COL, because metal names what survives.
constexpr ckernel::PoolType metal_pool(ReducePool p) {
    return p == ReducePool::Sum   ? ckernel::PoolType::SUM
           : p == ReducePool::Avg ? ckernel::PoolType::AVG
                                  : ckernel::PoolType::MAX;
}

constexpr ckernel::ReduceDim metal_dim(ReduceAxis a) {
    return a == ReduceAxis::Rows   ? ckernel::ReduceDim::REDUCE_COL
           : a == ReduceAxis::Cols ? ckernel::ReduceDim::REDUCE_ROW
                                   : ckernel::ReduceDim::REDUCE_SCALAR;
}
#endif

// The INPUT tile grid, row-major: tile (h, w) is at index h * wt + w.
// Derived from the input shape, not declared: every member below is a pure function of
// (rows, cols, axis), so the operand's shape IS the geometry. Nothing writes this type.
template <typename S>
struct ReduceGeometry {
    static_assert(
        S::leading == 1,
        "reducing a shape with a leading (batch) extent is not implemented -- the strategy walks one "
        "2-D grid, so run the reduction per batch from the kernel's own loop");

    static constexpr uint32_t ht = S::rows;
    static constexpr uint32_t wt = S::cols;
    static constexpr uint32_t num_tiles = ht * wt;

    // Tiles the result occupies, which is what the destination Storage must hold.
    static constexpr uint32_t out_tiles(ReduceAxis axis) {
        return axis == ReduceAxis::Rows ? wt : (axis == ReduceAxis::Cols ? ht : 1);
    }

    // Elements folded into ONE output value -- what an average divides by. A tile
    // is 32x32, so collapsing rows folds ht*32 of them.
    static constexpr uint32_t elements(ReduceAxis axis) {
        return axis == ReduceAxis::Rows ? ht * 32 : (axis == ReduceAxis::Cols ? wt * 32 : ht * wt * 32 * 32);
    }

    // Input tiles feeding one output tile.
    static constexpr uint32_t group(ReduceAxis axis) {
        return axis == ReduceAxis::Rows ? ht : (axis == ReduceAxis::Cols ? wt : num_tiles);
    }

    // Index of the g'th contributor to output tile `o`.
    static constexpr uint32_t contributor(ReduceAxis axis, uint32_t o, uint32_t g) {
        return axis == ReduceAxis::Rows ? g * wt + o : (axis == ReduceAxis::Cols ? o * wt + g : g);
    }
};

// `scaler_cb` holds the constant reduce_tile folds in: see fill_reduce_scaler.
template <typename S, ReduceAxis Axis, ReducePool Pool, typename Chain>
struct ReduceNode : expr::Fluent<ReduceNode<S, Axis, Pool, Chain>> {
    using fusion_kind = ReduceFusion;
    using in_shape = S;
    using geometry = ReduceGeometry<S>;
    using chain = Chain;
    using shape = reduce_shape<S, Axis>;
    static constexpr ReduceAxis axis = Axis;
    static constexpr ReducePool pool = Pool;

    uint32_t in_cb;
    uint32_t scaler_cb;
};

template <typename S, ReduceAxis A, ReducePool P, typename Chain>
auto relu(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, ReluOp>>{{}, r.in_cb, r.scaler_cb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto relu(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, ReluOp>>{{}, {}, b.block_cb, b.vec_cb};
}

template <typename S, ReduceAxis A, ReducePool P, typename Chain>
auto exp_(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, ExpOp>>{{}, r.in_cb, r.scaler_cb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto exp_(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, ExpOp>>{{}, {}, b.block_cb, b.vec_cb};
}

template <typename S, ReduceAxis A, ReducePool P, typename Chain>
auto recip(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, RecipOp>>{{}, r.in_cb, r.scaler_cb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto recip(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, RecipOp>>{{}, {}, b.block_cb, b.vec_cb};
}

template <typename S, ReduceAxis A, ReducePool P, typename Chain>
auto sqrt_(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, SqrtOp>>{{}, r.in_cb, r.scaler_cb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto sqrt_(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, SqrtOp>>{{}, {}, b.block_cb, b.vec_cb};
}

template <typename S, ReduceAxis A, ReducePool P, typename Chain>
auto rsqrt(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, RsqrtOp>>{{}, r.in_cb, r.scaler_cb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto rsqrt(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, RsqrtOp>>{{}, {}, b.block_cb, b.vec_cb};
}

// --- Node shapes ---
//
// The shape an expression produces. Lives here rather than in tt/unified/shape.hpp
// because it walks the op tree, and not in tt/unified/expr.hpp because that layer is
// deliberately ignorant of shapes as well as of ops -- so the tree shapes themselves
// carry no shape, and this reads it back out of them.
//
// The Bin case is where strictness bites: two operands must have the SAME shape, not
// merely the same page count. Shape<1, 4> and Shape<4> hold four pages each and are
// different shapes, and before this the difference was invisible.

template <typename Node>
struct node_shape {
    using type = typename Node::shape;
};

template <typename Node>
using node_shape_t = typename node_shape<Node>::type;

template <typename Op, typename C>
struct node_shape<expr::Un<Op, C>> {
    using type = node_shape_t<C>;
};

template <typename Op, typename L, typename R>
struct node_shape<expr::Bin<Op, L, R>> {
    static_assert(
        same_shape_v<node_shape_t<L>, node_shape_t<R>>,
        "the two sides of a binary op must have the SAME shape -- equal page counts are not enough, "
        "since e.g. Shape<1, 4> and Shape<4> both hold four pages");
    using type = node_shape_t<L>;
};

// An FPU fusion's shape is its output block: rows from A, columns from B.
// --- Operand plumbing ---
//
// `is_operand` and `as_node` are the extension points the core header hooks into:
// it specialises is_operand<ComputeBlock> and overloads as_node(ComputeBlock),
// which the templates below pick up by ADL.

template <typename T>
struct is_operand : expr::is_expr<T> {};

template <typename Node, typename = std::enable_if_t<expr::is_expr<Node>::value>>
const Node& as_node(const Node& n) {
    return n;
}

template <typename A, typename B, typename = std::enable_if_t<is_operand<A>::value && is_operand<B>::value>>
auto operator+(const A& a, const B& b) {
    using LN = std::decay_t<decltype(as_node(a))>;
    using RN = std::decay_t<decltype(as_node(b))>;
    return expr::Bin<AddOp, LN, RN>{{}, as_node(a), as_node(b)};
}

template <typename A, typename B, typename = std::enable_if_t<is_operand<A>::value && is_operand<B>::value>>
auto operator-(const A& a, const B& b) {
    using LN = std::decay_t<decltype(as_node(a))>;
    using RN = std::decay_t<decltype(as_node(b))>;
    return expr::Bin<SubOp, LN, RN>{{}, as_node(a), as_node(b)};
}

template <typename A, typename B, typename = std::enable_if_t<is_operand<A>::value && is_operand<B>::value>>
auto operator*(const A& a, const B& b) {
    using LN = std::decay_t<decltype(as_node(a))>;
    using RN = std::decay_t<decltype(as_node(b))>;
    return expr::Bin<MulOp, LN, RN>{{}, as_node(a), as_node(b)};
}

template <typename A, typename B, typename = std::enable_if_t<is_operand<A>::value && is_operand<B>::value>>
auto operator/(const A& a, const B& b) {
    using LN = std::decay_t<decltype(as_node(a))>;
    using RN = std::decay_t<decltype(as_node(b))>;
    return expr::Bin<DivOp, LN, RN>{{}, as_node(a), as_node(b)};
}

// relu() on a tree wraps it; relu() on an FPU node folds into that node's
// epilogue chain instead. This per-kind dispatch is what a CRTP `Derived`
// parameter would otherwise be threading through every combinator.
//
// A trailing underscore where the name would shadow a <cmath> function, and only
// there: exp_ and sqrt_ carry one, relu, recip and rsqrt do not. The METHOD
// spelling is unshadowed either way, so it stays bare -- x.exp(), x.sqrt().
template <typename N, typename = std::enable_if_t<expr::is_expr<N>::value>>
auto relu(const N& n) {
    return expr::Un<ReluOp, N>{{}, n};
}

template <typename N, typename = std::enable_if_t<expr::is_expr<N>::value>>
auto exp_(const N& n) {
    return expr::Un<ExpOp, N>{{}, n};
}

template <typename N, typename = std::enable_if_t<expr::is_expr<N>::value>>
auto recip(const N& n) {
    return expr::Un<RecipOp, N>{{}, n};
}

template <typename N, typename = std::enable_if_t<expr::is_expr<N>::value>>
auto sqrt_(const N& n) {
    return expr::Un<SqrtOp, N>{{}, n};
}

template <typename N, typename = std::enable_if_t<expr::is_expr<N>::value>>
auto rsqrt(const N& n) {
    return expr::Un<RsqrtOp, N>{{}, n};
}

template <typename SA, typename SB, TransposeB Tr, typename Chain>
auto relu(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, ReluOp>>{{}, m.in0_cb, m.in1_cb, m.bias_cb};
}

template <typename SA, typename SB, TransposeB Tr, typename Chain>
auto exp_(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, ExpOp>>{{}, m.in0_cb, m.in1_cb, m.bias_cb};
}

template <typename SA, typename SB, TransposeB Tr, typename Chain>
auto recip(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, RecipOp>>{{}, m.in0_cb, m.in1_cb, m.bias_cb};
}

template <typename SA, typename SB, TransposeB Tr, typename Chain>
auto sqrt_(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, SqrtOp>>{{}, m.in0_cb, m.in1_cb, m.bias_cb};
}

template <typename SA, typename SB, TransposeB Tr, typename Chain>
auto rsqrt(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, RsqrtOp>>{{}, m.in0_cb, m.in1_cb, m.bias_cb};
}

// A broadcast is spelled with the ordinary operators, the marker on the right telling
// them apart from an SFPU tree. Dispatch is on Broadcast's TYPE -- something the caller
// wrote -- not on a shape mismatch, so strict shape equality between two plain blocks is
// untouched. is_operand<Broadcast> is false, which is what keeps the SFPU operators from
// swallowing one.
//
// The marker has to be the RIGHT operand: metal's broadcast ops read the vector from in1.

// A broadcast's block operand has to be a stored buffer. TileSource is what as_node()
// yields for one; an expression yields itself, and an expression lives in DST where the
// FPU cannot read it as an operand.
template <typename T, typename = void>
struct bcast_block_shape {
    using type = Shape<1, 1>;  // a placeholder, so the static_assert below is what reports
    static constexpr bool ok = false;
};

template <typename S>
struct bcast_block_shape<TileSource<S>> {
    using type = S;
    static constexpr bool ok = true;
};

#define TT_UNIFIED_BCAST_OPERATOR(sym, OpType)                                                            \
    template <typename B, Axis A, typename SV, typename = std::enable_if_t<is_operand<B>::value>>         \
    auto operator sym(const B& block, Broadcast<A, SV> vec) {                                             \
        using BN = std::decay_t<decltype(as_node(block))>;                                                \
        static_assert(                                                                                    \
            bcast_block_shape<BN>::ok,                                                                    \
            "a broadcast's left operand must be a stored buffer, not an expression -- the FPU "           \
            "reads both operands from circular buffers while an expression lives in DST, so "             \
            "store it to a Storage first");                                                               \
        using SB = typename bcast_block_shape<BN>::type;                                                  \
        return BcastNode<OpType, A, SB, SV, expr::UnaryChain<>>{{}, {}, as_node(block).cb_id, vec.cb_id}; \
    }                                                                                                     \
    template <typename R, Axis A, typename SV>                                                            \
    auto operator sym(Broadcast<A, SV>, const R&) {                                                       \
        static_assert(                                                                                    \
            always_false<R>::value,                                                                       \
            "a broadcast has to be the RIGHT operand -- metal reads the broadcast vector from in1, "      \
            "so write `block " #sym " bcast<Axis::...>(vec)`");                                           \
    }

TT_UNIFIED_BCAST_OPERATOR(+, AddOp)
TT_UNIFIED_BCAST_OPERATOR(-, SubOp)
TT_UNIFIED_BCAST_OPERATOR(*, MulOp)

#undef TT_UNIFIED_BCAST_OPERATOR

// The hooks expr.hpp's Fluent calls. Unqualified inside, so ordinary lookup finds
// the overloads above for the expression shapes, and ADL finds the ones declared
// later for the core types -- ComputeBlock's live in tt/unified/api.h.
namespace expr {
template <typename N>
auto fluent_relu(const N& n) {
    return relu(n);
}
template <typename N>
auto fluent_exp(const N& n) {
    return exp_(n);
}
template <typename N>
auto fluent_recip(const N& n) {
    return recip(n);
}
template <typename N>
auto fluent_sqrt(const N& n) {
    return sqrt_(n);
}
template <typename N>
auto fluent_rsqrt(const N& n) {
    return rsqrt(n);
}
}  // namespace expr

// The operands ARE the geometry now, so there is nothing left to state twice -- and
// the inner-dimension agreement that nothing checked before is a static_assert inside
// MatmulGeometry, reached by simply naming it.
template <TransposeB Tr = TransposeB::No, typename SA, typename SB>
auto matmul(TileSource<SA> a, TileSource<SB> b) {
    return MatmulNode<SA, SB, Tr, expr::UnaryChain<>>{{}, a.cb_id, b.cb_id, kNoBias};
}

// An FPU fusion cannot be an operand of a binary op: it already owns every DST
// slot, so there is nowhere to materialise the other side. Keyed on the *kind*,
// so future FPU ops inherit the rule without another overload.

template <typename T>
struct is_fpu_fusion : std::is_same<expr::kind_of_t<T>, FPUFusion> {};

// The message lives in one place so the four guards below cannot drift.
template <typename A>
void reject_fpu_operand() {
    static_assert(
        always_false<A>::value,
        "an FPU fusion consumes all of DST, so it cannot be an operand of a binary op; "
        "store it to an intermediate Storage first, then combine");
}

template <typename A, typename B, typename = std::enable_if_t<is_fpu_fusion<A>::value || is_fpu_fusion<B>::value>>
void operator+(const A&, const B&) {
    reject_fpu_operand<A>();
}

template <typename A, typename B, typename = std::enable_if_t<is_fpu_fusion<A>::value || is_fpu_fusion<B>::value>>
void operator-(const A&, const B&) {
    reject_fpu_operand<A>();
}

template <typename A, typename B, typename = std::enable_if_t<is_fpu_fusion<A>::value || is_fpu_fusion<B>::value>>
void operator*(const A&, const B&) {
    reject_fpu_operand<A>();
}

template <typename A, typename B, typename = std::enable_if_t<is_fpu_fusion<A>::value || is_fpu_fusion<B>::value>>
void operator/(const A&, const B&) {
    reject_fpu_operand<A>();
}

// --- Hardware startup ---
//
// These are MMIO writes and must run once, at kernel entry, before any other
// compute API call -- which is why they are the kernel's job rather than the
// strategy's. Which one you call depends on the fusion kind you are about to
// use; they configure the ALU differently and are not interchangeable.
//
// Both self-guard: on a data-movement build the body preprocesses away, so
// kernels call them unconditionally.

// SFPU path: configures unpack/pack for one input/output CB pair.
inline void compute_init(uint32_t in_cb, uint32_t out_cb) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    ckernel::init_sfpu(in_cb, out_cb);
#else
    (void)in_cb;
    (void)out_cb;
#endif
}

// FPU path: matmul needs SrcOrder::Reverse -- in0 lands in SrcA's partner SrcB
// and in1 in SrcA -- plus the block dimensions programmed up front. Calling
// compute_init() instead leaves the ALU configured for SFPU work, and matmul then
// runs against a state it cannot use.
// The transpose here MUST match the one every matmul() in the kernel uses. This call
// programs the MOP once at startup; matmul_block then passes its own flag per call, and
// a disagreement leaves the unpacker configured for the other arrangement. Nothing can
// check it across two separate calls -- the fix is to name it once in the kernel and
// pass that constant to both. ttnn has the same coupling and resolves it the same way.
template <typename SA, typename SB, TransposeB Tr = TransposeB::No>
inline void matmul_init(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    using Geometry = MatmulGeometry<SA, SB, Tr>;
    ckernel::compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(in0_cb, in1_cb, out_cb);
    ckernel::matmul_block_init(
        in0_cb, in1_cb, Geometry::transpose, Geometry::ct_dim, Geometry::rt_dim, Geometry::kt_dim);
#else
    (void)in0_cb;
    (void)in1_cb;
    (void)out_cb;
#endif
}

// How a multi-block FPU fusion carries its running total.
//
//   Dst -- the partial is reloaded from a separate buffer into DST before each
//          matmul, which then accumulates on top. Costs a DST round-trip and two
//          format reconfigs per block. DST holds the *running total*, so a
//          finish-only epilogue is meaningful and a per-step chain sees the total
//          so far.
//
//   L1  -- the packer accumulates into L1 instead. No reload, and DST only ever
//          holds one block's product -- so a per-step chain sees that block's
//          contribution alone, but a finish-only epilogue is impossible, since
//          the total never sits in DST.
enum class AccumulatorMode {
    Dst,
    L1,
};

// --- Driver strategies ---
//
// The loop shape *is* the strategy. Storage::store dispatches on the root node's
// kind; everything above this point only decides what gets emitted.

template <typename Kind>
struct Strategy;

// SFPU: one pass of the whole expression per tile, packing each result.
template <>
struct Strategy<SFPUFusion> {
    template <typename Node>
    static void run(const Node& node, uint32_t cb_id, uint32_t num_tiles) {
        static_assert(
            expr::need_v<Node> <= kMaxDstTiles,
            "SFPU expression needs more DST slots than the hardware has; "
            "split it across an intermediate Storage");
        // Who pays for a format reconfig. With ONE leaf the unpacker stays pointed at
        // that buffer for the whole loop, so pointing it there once is enough and the
        // common unary and single-operand cases cost nothing. With more than one leaf
        // the leaves alternate, so every one has to re-point -- that is the price of a
        // mixed-format tree being correct rather than quietly wrong.
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        constexpr bool kEveryTile = expr::leaf_count_v<Node> > 1;
        cb_reserve_back(cb_id, num_tiles);
        for (uint32_t i = 0; i < num_tiles; ++i) {
            ckernel::tile_regs_acquire();
            expr::emit(node, i, kEveryTile || i == 0);
            ckernel::tile_regs_commit();
            ckernel::tile_regs_wait();
            ckernel::pack_tile(expr::result_slot_v<Node>, cb_id);
            ckernel::tile_regs_release();
        }
        cb_push_back(cb_id, num_tiles);
#else
        (void)node;
        (void)cb_id;
        (void)num_tiles;
#endif
    }
};

// FPU: one k-block per call. The kernel owns the k-loop (see Accumulator in
// tt/unified/api.h), because the operand CBs must be waited and popped per block
// so the reader can stream them.
//
// Mirrors bmm_large_block_zm_fused_bias_activation.cpp:
//   acquire -> [reload partials into DST] -> matmul_block across k
//           -> [epilogue on DST] -> commit -> pack to partials, or to out on the
//              final block.
template <>
struct Strategy<FPUFusion> {
    // The finishing pass when a bias is fused. Both modes converge here: the
    // total is in acc_cb, so this adds the broadcast bias into DST, applies the
    // epilogue, and packs to out_cb.
    //
    // It has to be a second pass over an intermediate, because metal's bcast add
    // reads BOTH operands from circular buffers -- there is no "add a buffer tile
    // into a DST slot". L1 mode pays nothing for that: this replaces the copy-out
    // it already did. Dst mode pays one extra pack, into acc_cb, which it leaves
    // idle at finish anyway -- so neither mode needs a new buffer.
    template <typename Node, typename EpilogueChain>
    static void bias_finish(const Node& node, uint32_t acc_cb, uint32_t out_cb, EpilogueChain) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using G = typename Node::geometry;
        constexpr uint32_t kAccTiles = G::out_subblock_num_tiles;
        constexpr uint32_t kTranspose = G::transpose;

        ckernel::tile_regs_acquire();
        ckernel::reconfig_data_format(acc_cb, node.bias_cb);
        ckernel::add_bcast_rows_init_short(acc_cb, node.bias_cb);

        cb_wait_front(acc_cb, kAccTiles);
        for (uint32_t t = 0; t < kAccTiles; ++t) {
            // Bias is 1 x ct_dim tiles broadcast DOWN the rows, so the tile for
            // output (r, c) is c -- and the output block is row-major.
            ckernel::add_tiles_bcast_rows(acc_cb, node.bias_cb, t, t % G::ct_dim, t);
        }
        cb_pop_front(acc_cb, kAccTiles);

        if constexpr (!EpilogueChain::empty) {
            for (uint32_t t = 0; t < kAccTiles; ++t) {
                EpilogueChain::apply_in_place(t);
            }
        }

        ckernel::tile_regs_commit();
        cb_reserve_back(out_cb, kAccTiles);
        ckernel::tile_regs_wait();
        ckernel::pack_block(0, out_cb, kAccTiles);
        ckernel::tile_regs_release();
        cb_push_back(out_cb, kAccTiles);

        // Put back what matmul_block needs, so the next output block can run.
        ckernel::reconfig_data_format_srca(acc_cb, node.in1_cb);
        ckernel::matmul_block_init(node.in0_cb, node.in1_cb, kTranspose, G::ct_dim, G::rt_dim, G::kt_dim);
#else
        (void)node;
        (void)acc_cb;
        (void)out_cb;
#endif
    }

    // Single-shot: one k-block, no accumulation buffer. This is the shape
    // Storage::store() uses, so `out.store(matmul<Geom>(a, b))` still works for a
    // one-round matmul. With reload=false and finish=true the accumulation buffer
    // is never touched, so passing the destination for both is safe.
    template <typename Node>
    static void run(const Node& node, uint32_t cb_id, uint32_t /*num_tiles*/) {
        run<AccumulatorMode::Dst>(node, /*acc_cb=*/cb_id, /*out_cb=*/cb_id, /*reload=*/false, /*finish=*/true);
    }

    // `Node::chain` is the PER-STEP chain, run on every call; `EpilogueChain` runs
    // only on the finishing call, against the completed accumulator.
    //
    // What a per-step chain sees differs by mode. In Dst mode the reload happens
    // before the matmul, so DST already holds the running total and the chain sees
    // f(total-so-far), not f(this contribution) -- isolating the contribution
    // would need a second rt*ct-sized scratch region, which does not fit. L1 mode
    // gets it for free: the packer does the summing, so DST only ever holds one
    // block's product.
    template <AccumulatorMode Mode, typename Node, typename EpilogueChain = expr::UnaryChain<>>
    static void run(const Node& node, uint32_t acc_cb, uint32_t out_cb, bool reload, bool finish, EpilogueChain = {}) {
        using G = typename Node::geometry;
        constexpr uint32_t kAccTiles = G::out_subblock_num_tiles;

        static_assert(
            kAccTiles <= kMaxDstTiles,
            "matmul rt_dim * ct_dim exceeds the per-acquire DST budget (8 tiles under half-sync). "
            "Split the output block into subblocks of at most 8 tiles -- 4x4 is not a legal subblock, "
            "4x2 / 2x4 / 8x1 / 1x8 are the largest that are.");
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using Chain = typename Node::chain;
        constexpr uint32_t kTranspose = G::transpose;

        // Program the block dimensions here rather than trusting matmul_init to still be in
        // force. A broadcast, a reduction or an SFPU pass reconfigures the unpack and math
        // units for itself, so a matmul that FOLLOWS one -- as attention's second matmul
        // does -- would otherwise run against another op's state and return garbage.
        //
        // matmul_init still has to run once at kernel entry, for the hardware startup it
        // carries, and that part must NOT be repeated: compute_kernel_hw_startup is MMIO
        // plus a pack-sync init, and calling it a second time mid-kernel hangs the device.
        ckernel::matmul_block_init(node.in0_cb, node.in1_cb, kTranspose, G::ct_dim, G::rt_dim, G::kt_dim);

        ckernel::tile_regs_acquire();

        if constexpr (Mode == AccumulatorMode::Dst) {
            if (reload) {
                // Partials L1 -> DST, then restore the state matmul_block needs.
                ckernel::copy_tile_to_dst_init_short_with_dt(node.in1_cb, acc_cb);
                cb_wait_front(acc_cb, kAccTiles);
                ckernel::copy_block(acc_cb, 0, 0, kAccTiles);
                cb_pop_front(acc_cb, kAccTiles);
                ckernel::reconfig_data_format_srca(acc_cb, node.in1_cb);
                ckernel::matmul_block_init(node.in0_cb, node.in1_cb, kTranspose, G::ct_dim, G::rt_dim, G::kt_dim);
            }
        }

        // This block's product. In Dst mode it lands on top of the reloaded
        // partial; in L1 mode DST holds it alone.
        uint32_t in0_index = 0;
        uint32_t in1_index = 0;
        for (uint32_t k = 0; k < G::kt_dim; ++k) {
            ckernel::matmul_block(
                node.in0_cb,
                node.in1_cb,
                in0_index,
                in1_index,
                /*idst=*/0,
                kTranspose,
                G::ct_dim,
                G::rt_dim,
                G::kt_dim);
            in0_index += 1;
            in1_index += G::in1_row_stride;
        }

        if constexpr (!Chain::empty) {
            for (uint32_t t = 0; t < kAccTiles; ++t) {
                Chain::apply_in_place(t);
            }
        }

        // In L1 mode the total is not in DST yet, so the epilogue runs in the
        // copy-out stage below instead. A fused bias moves it later in Dst mode
        // too: the epilogue has to see the BIASED total, so bias_finish applies
        // it and this must not, or the chain runs twice and the first run sees
        // A@B without the bias -- relu(relu(A@B) + v) instead of relu(A@B + v).
        if constexpr (Mode == AccumulatorMode::Dst) {
            if constexpr (!EpilogueChain::empty) {
                if (finish && node.bias_cb == kNoBias) {
                    for (uint32_t t = 0; t < kAccTiles; ++t) {
                        EpilogueChain::apply_in_place(t);
                    }
                }
            }
        }

        ckernel::tile_regs_commit();

        if constexpr (Mode == AccumulatorMode::Dst) {
            // A fused bias needs the total in a buffer to add against, so the
            // finishing pack goes to acc_cb and bias_finish carries it to out_cb.
            const bool via_bias = finish && node.bias_cb != kNoBias;
            const uint32_t dest = (finish && !via_bias) ? out_cb : acc_cb;
            cb_reserve_back(dest, kAccTiles);
            ckernel::tile_regs_wait();
            ckernel::pack_block(0, dest, kAccTiles);
            ckernel::tile_regs_release();
            cb_push_back(dest, kAccTiles);
            if (via_bias) {
                bias_finish(node, acc_cb, out_cb, EpilogueChain{});
            }
        } else {
            // L1: the packer adds this block's product into what is already at the
            // destination, so the running total lives in L1 and never occupies DST.
            //
            // The push/pop pair is load-bearing, not bookkeeping. pack_block
            // advances the CB's fifo_wr_tile_ptr itself and cb_push_back is the
            // only thing that resets it (llk_io_pack.h), so a pack without a
            // matching push lands one block further along each round instead of on
            // top of the previous one. Pushing then popping a CB sized to exactly
            // one block wraps both pointers back to the base address -- which
            // still holds the partial, since a pop does not erase.
            cb_reserve_back(acc_cb, kAccTiles);
            ckernel::tile_regs_wait();
            ckernel::pack_reconfig_l1_acc(reload ? 1 : 0);
            ckernel::pack_block(0, acc_cb, kAccTiles);
            ckernel::tile_regs_release();
            cb_push_back(acc_cb, kAccTiles);
            ckernel::pack_reconfig_l1_acc(0);  // leave the packer as we found it

            if (!finish) {
                cb_wait_front(acc_cb, kAccTiles);
                cb_pop_front(acc_cb, kAccTiles);
            } else if (node.bias_cb != kNoBias) {
                // The copy-out below, with the bias folded into it -- same wait,
                // same pop, same pack, one op different.
                bias_finish(node, acc_cb, out_cb, EpilogueChain{});
            } else {
                // Move the completed total into the output buffer. Copying it
                // through DST rather than letting the DM writer drain acc_cb keeps
                // one popper per CB -- compute owns acc_cb, the writer owns out_cb
                // -- and gives the finish-only epilogue the whole total in DST,
                // exactly as in Dst mode.
                ckernel::tile_regs_acquire();
                ckernel::copy_tile_to_dst_init_short_with_dt(node.in1_cb, acc_cb);
                cb_wait_front(acc_cb, kAccTiles);
                ckernel::copy_block(acc_cb, 0, 0, kAccTiles);
                cb_pop_front(acc_cb, kAccTiles);

                if constexpr (!EpilogueChain::empty) {
                    for (uint32_t t = 0; t < kAccTiles; ++t) {
                        EpilogueChain::apply_in_place(t);
                    }
                }

                ckernel::tile_regs_commit();
                cb_reserve_back(out_cb, kAccTiles);
                ckernel::tile_regs_wait();
                ckernel::pack_block(0, out_cb, kAccTiles);
                ckernel::tile_regs_release();
                cb_push_back(out_cb, kAccTiles);

                // Restore the state matmul_block needs, so the accumulator can be
                // cleared and driven again for the next output block.
                ckernel::reconfig_data_format_srca(acc_cb, node.in1_cb);
                ckernel::matmul_block_init(node.in0_cb, node.in1_cb, kTranspose, G::ct_dim, G::rt_dim, G::kt_dim);
            }
        }
#else
        (void)node;
        (void)acc_cb;
        (void)out_cb;
        (void)reload;
        (void)finish;
#endif
    }
};

// Reduce: metal's reduce, folding the input grid down one axis. Every contributor
// to an output tile accumulates into DST slot 0 -- reduce_tile adds into idst
// rather than overwriting it -- so this costs ONE slot whatever the geometry.
// BcastFusion: the block and the vector both come from circular buffers, and the
// unpacker's broadcast mode replicates the vector's one row or one column across each
// tile as it is read. So there is nothing to allocate -- one DST slot holds the result of
// one tile, whatever the block's size.
//
// Per TILE rather than per block, unlike bias_finish which packs the whole block. That
// caps a block at the DST budget of 8 tiles, and attention's score block is 16.
//
// Neither operand is popped here. Both are ComputeBlocks whose destructors pop them,
// which is the same contract Strategy<SFPUFusion> follows for its leaves.
template <>
struct Strategy<BcastFusion> {
    template <typename Node>
    static void run(const Node& node, uint32_t cb_id, uint32_t num_tiles) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using Chain = typename Node::chain;
        using Ops = BcastOps<typename Node::op, Node::axis>;

        // Point the unpacker at this pair, then program the broadcast mode. Both are
        // hoisted: neither buffer changes across the loop.
        ckernel::reconfig_data_format(node.block_cb, node.vec_cb);
        Ops::init(node.block_cb, node.vec_cb);

        cb_reserve_back(cb_id, num_tiles);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            ckernel::tile_regs_acquire();
            Ops::apply(node.block_cb, node.vec_cb, t, Node::vec_tile(t), 0);
            if constexpr (!Chain::empty) {
                Chain::apply_in_place(0);
            }
            ckernel::tile_regs_commit();
            ckernel::tile_regs_wait();
            ckernel::pack_tile(0, cb_id);
            ckernel::tile_regs_release();
        }
        cb_push_back(cb_id, num_tiles);
#else
        (void)node;
        (void)cb_id;
        (void)num_tiles;
#endif
    }
};

template <>
struct Strategy<ReduceFusion> {
    template <typename Node>
    static void run(const Node& node, uint32_t cb_id, uint32_t /*num_tiles*/) {
        using G = typename Node::geometry;
        constexpr ReduceAxis kAxis = Node::axis;
        constexpr uint32_t kOut = G::out_tiles(kAxis);
        constexpr uint32_t kGroup = G::group(kAxis);
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using Chain = typename Node::chain;
        constexpr ckernel::PoolType kPool = metal_pool(Node::pool);
        constexpr ckernel::ReduceDim kDim = metal_dim(kAxis);

        // No check that the destination is the shape the axis leaves behind: that was a
        // runtime ASSERT on the page count here, and Storage::store now static_asserts
        // full shape identity -- unconditionally, rather than only in an
        // asserts-enabled build, and on the shape rather than just its page count.

        // No cb_wait_front on the scaler: it is a ComputeBlock at kernel scope, so
        // its constructor waited once and nothing pops it until the kernel ends.

        if constexpr (kDim == ckernel::ReduceDim::REDUCE_ROW && kPool != ckernel::PoolType::MAX) {
            // SUM/AVG along a row is an MVMUL with the operands swapped, so the
            // scaler has to be SrcA before init -- see reduce_init's own note.
            ckernel::reconfig_data_format(node.scaler_cb, node.in_cb);
        }
        ckernel::reduce_init<kPool, kDim>(node.in_cb, node.scaler_cb, cb_id);

        cb_reserve_back(cb_id, kOut);
        for (uint32_t o = 0; o < kOut; ++o) {
            ckernel::tile_regs_acquire();
            for (uint32_t g = 0; g < kGroup; ++g) {
                ckernel::reduce_tile<kPool, kDim>(node.in_cb, node.scaler_cb, G::contributor(kAxis, o, g), 0, 0);
            }
            if constexpr (!Chain::empty) {
                Chain::apply_in_place(0);
            }
            ckernel::tile_regs_commit();
            ckernel::tile_regs_wait();
            ckernel::pack_tile(0, cb_id);
            ckernel::tile_regs_release();
        }
        cb_push_back(cb_id, kOut);

        // Mandatory, not tidiness: reduce_init left the packer masking every datum
        // outside the result to zero, and the next op inherits that until it is
        // reset.
        ckernel::reduce_uninit(node.in_cb);
#else
        (void)node;
        (void)cb_id;
        (void)kOut;
        (void)kGroup;
#endif
    }
};

}  // namespace unified
}  // namespace tt
