// SPDX-License-Identifier: Apache-2.0
//
// Compute fusions for unified.hpp.
//
// Layering:
//   tt/unified_expr.hpp       -- domain-free tree shapes + register allocator
//   tt/unified_math.hpp       -- this file: what a leaf is, what each op emits,
//                                and the per-kind driver strategies
//   tt/unified_api.h          -- core API declarations
//   tt/unified_impl_v1.hpp    -- core API definitions
//   tt/unified_adaptor_v1.hpp -- metal binding
//
// This header deliberately does not depend on the core types. Leaves and nodes
// carry raw circular-buffer ids, so the dependency runs one way: unified.hpp
// includes this, and supplies the thin ComputeBlock adaptors at the bottom of
// its own file.
//
// ---------------------------------------------------------------------------
// FUSION KINDS
//
// A kind selects the *driver strategy*: the shape of the enclosing loop, and
// which hardware unit owns the DST register file. That is the axis along which
// fusions genuinely differ -- not the op set.
//
//   SFPUFusion -- the SFPU indexes DST freely, so an arbitrary expression tree
//                 can be allocated across it. One pass per tile, pack each
//                 result.
//
//   FPUFusion  -- the FPU maximises DST. matmul_block self-increments dst_index
//                 from 0 across out_subblock_num_tiles, so there is nothing left
//                 to allocate: only a *unary* epilogue can fuse, applied in
//                 place on the final accumulation step. The hardware says the
//                 same thing -- matmul's activation is compiled out when
//                 FUSE_BIAS is set, and bias spills through an intermediate CB
//                 instead (bmm_large_block_zm_fused_bias_activation.cpp:384).
//
// Adding an FPU op later means adding a node type that declares
// `using fusion_kind = FPUFusion;` -- Strategy<FPUFusion> is reused as-is.

#pragma once

#include <type_traits>

#include <tt/unified_expr.hpp>

// Every op body below is guarded on IS_COMPUTE_THREAD, which a binding defines.
// Without one they would all silently compile to nothing, so refuse instead.
#if !defined(IS_COMPUTE_THREAD) && !defined(IS_DM_THREAD)
#error "include <tt/unified> (or a binding) before tt/unified_math.hpp"
#endif

namespace tt {
namespace unified {

// Max DST tiles. Halves under fp32 accumulate; see reg_api.h.
inline constexpr int kMaxDstTiles = 16;

// ---------------------------------------------------------------------------
// Leaves and ops
// ---------------------------------------------------------------------------

// One tile out of a circular buffer, copied into a DST slot. Which slot is
// chosen by the allocator, not the caller -- that is what keeps operands from
// clobbering intermediates.
struct TileSource {
    using is_expr_node = std::true_type;
    static constexpr int need = 1;

    int cb_id;

    void emit(int dst, int tile) const {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        ckernel::copy_tile(cb_id, tile, dst);
#else
        (void)dst;
        (void)tile;
#endif
    }
};

struct AddOp {
    static void apply(int lhs, int rhs, int out) {
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

struct ExpOp {
    static void apply(int src, int out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;  // == out; SFPU unaries work in place
        ckernel::exp_tile_init();
        ckernel::exp_tile(out);
#else
        (void)src;
        (void)out;
#endif
    }
    static void apply_in_place(int slot) { apply(slot, slot); }
};

// NOTE: a cross-tile reduction is deliberately absent. It is not an op -- it
// accumulates across the tile loop and packs once, so it wants a third
// Strategy alongside SFPUFusion/FPUFusion.

// A unary usable in either kind. The two entry points are genuinely different
// hardware paths: `apply`/`apply_in_place` run on the SFPU during math, while
// `apply_from_pack` is the packer-side epilogue the FPU strategy uses -- it
// replaces tile_regs_wait(). An op that offers only the former simply fails to
// compile in an FPU chain, which is the intent.
//
// The `*_tile_init()` calls are inline rather than hoisted: they are cheap, and
// metal kernels routinely re-init per use (see SFPU_OP_CHAIN_0 in
// tests/.../compute/eltwise_sfpu.cpp). Worth hoisting if it shows in a profile.
struct ReluOp {
    static void apply(int src, int out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;  // == out; SFPU unaries work in place
        ckernel::relu_tile_init();
        ckernel::relu_tile(out);
#else
        (void)src;
        (void)out;
#endif
    }

    static void apply_in_place(int slot) { apply(slot, slot); }

    // Templated so it is only instantiated when an FPU chain actually uses it;
    // the pack-side epilogue is not yet bound to metal (see unified_metal.hpp).
    template <int = 0>
    static void apply_from_pack(int base, int count) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        relu_from_pack(base, count);
#else
        (void)base;
        (void)count;
#endif
    }
};

// ---------------------------------------------------------------------------
// Kinds and FPU nodes
// ---------------------------------------------------------------------------

using SFPUFusion = expr::TreeKind;

struct FPUFusion {};

// Compile-time geometry, so the strategy can unroll and the DST budget is
// checkable with a static_assert.
template <int OutSubblockH, int OutSubblockW, int In0BlockW, int NumBlocks>
struct MatmulGeometry {
    static constexpr int out_subblock_h = OutSubblockH;
    static constexpr int out_subblock_w = OutSubblockW;
    static constexpr int in0_block_w = In0BlockW;
    static constexpr int num_blocks = NumBlocks;
    static constexpr int out_subblock_num_tiles = OutSubblockH * OutSubblockW;
};

template <typename Geometry, typename Chain>
struct MatmulNode {
    using fusion_kind = FPUFusion;
    using geometry = Geometry;
    using chain = Chain;

    int in0_cb;
    int in1_cb;
};

// ---------------------------------------------------------------------------
// Operand plumbing
//
// `is_operand` and `as_node` are the extension points the core header hooks
// into: unified.hpp specialises is_operand<ComputeBlock> and overloads
// as_node(ComputeBlock), which the templates below pick up by ADL.
// ---------------------------------------------------------------------------

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
    return expr::Bin<AddOp, LN, RN>{as_node(a), as_node(b)};
}

// relu() on a tree wraps it; relu() on an FPU node folds into that node's
// epilogue chain instead. This per-kind dispatch is what a CRTP `Derived`
// parameter would otherwise be threading through every combinator.
template <typename N, typename = std::enable_if_t<expr::is_expr<N>::value>>
auto relu(const N& n) {
    return expr::Un<ReluOp, N>{n};
}

template <typename N, typename = std::enable_if_t<expr::is_expr<N>::value>>
auto exp_(const N& n) {
    return expr::Un<ExpOp, N>{n};
}

template <typename Geometry, typename Chain>
auto relu(const MatmulNode<Geometry, Chain>& m) {
    return MatmulNode<Geometry, expr::chain_append_t<Chain, ReluOp>>{m.in0_cb, m.in1_cb};
}

template <typename Geometry>
auto matmul(TileSource a, TileSource b) {
    return MatmulNode<Geometry, expr::UnaryChain<>>{a.cb_id, b.cb_id};
}

// An FPU fusion cannot be an operand of a binary op: it already owns every DST
// slot, so there is nowhere to materialise the other side. Keyed on the *kind*,
// so future FPU ops inherit the rule without another overload.
template <typename T>
struct always_false : std::false_type {};

template <typename T>
struct is_fpu_fusion : std::is_same<expr::kind_of_t<T>, FPUFusion> {};

template <typename A, typename B, typename = std::enable_if_t<is_fpu_fusion<A>::value || is_fpu_fusion<B>::value>>
void operator+(const A&, const B&) {
    static_assert(
        always_false<A>::value,
        "an FPU fusion consumes all of DST, so it cannot be an operand of a binary op; "
        "store it to an intermediate Storage first, then combine");
}

// ---------------------------------------------------------------------------
// Driver strategies
//
// The loop shape *is* the strategy. Storage::store dispatches on the root
// node's kind; everything above this point only decides what gets emitted.
// ---------------------------------------------------------------------------

template <typename Kind>
struct Strategy;

// SFPU: one pass of the whole expression per tile, packing each result.
template <>
struct Strategy<SFPUFusion> {
    template <typename Node>
    static void run(const Node& node, int cb_id, int num_tiles) {
        static_assert(
            expr::need_v<Node> <= kMaxDstTiles,
            "SFPU expression needs more DST slots than the hardware has; "
            "split it across an intermediate Storage");
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_reserve_back(cb_id, num_tiles);
        for (int i = 0; i < num_tiles; ++i) {
            ckernel::tile_regs_acquire();
            expr::emit(node, i);
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

// FPU: the node owns the loop. DST accumulates across the inner-k blocks and is
// packed once, on the final block, after the unary epilogue runs from pack.
template <>
struct Strategy<FPUFusion> {
    template <typename Node>
    static void run(const Node& node, int cb_id, int /*num_tiles*/) {
        using G = typename Node::geometry;
        static_assert(
            G::out_subblock_num_tiles <= kMaxDstTiles,
            "matmul out_subblock_h * out_subblock_w exceeds the DST register file");
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using Chain = typename Node::chain;
        for (int block = 0; block < G::num_blocks; ++block) {
            const bool last_out = (block == G::num_blocks - 1);
            ckernel::tile_regs_acquire();
            // matmul_block self-increments dst_index from 0, so the whole
            // subblock is the accumulator -- nothing here is allocatable.
            ckernel::matmul_block(node.in0_cb, node.in1_cb, G::out_subblock_h, G::out_subblock_w, G::in0_block_w);
            ckernel::tile_regs_commit();
            if (last_out) {
                cb_reserve_back(cb_id, G::out_subblock_num_tiles);
                if constexpr (Chain::empty) {
                    ckernel::tile_regs_wait();
                } else {
                    // The pack-side epilogue replaces tile_regs_wait().
                    Chain::apply_from_pack(0, G::out_subblock_num_tiles);
                }
                ckernel::pack_block(0, cb_id, G::out_subblock_num_tiles);
                cb_push_back(cb_id, G::out_subblock_num_tiles);
            } else {
                ckernel::tile_regs_wait();
                // TODO: real kernels spill/reload partials through an
                // intermediate CB here (mm_partials). Omitted from the sketch.
            }
            ckernel::tile_regs_release();
        }
#else
        (void)node;
        (void)cb_id;
#endif
    }
};

}  // namespace unified
}  // namespace tt
