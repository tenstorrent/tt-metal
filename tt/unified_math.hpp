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

#include <cstdint>
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
inline constexpr uint32_t kMaxDstTiles = 16;

// ---------------------------------------------------------------------------
// Leaves and ops
// ---------------------------------------------------------------------------

// One tile out of a circular buffer, copied into a DST slot. Which slot is
// chosen by the allocator, not the caller -- that is what keeps operands from
// clobbering intermediates.
struct TileSource {
    using is_expr_node = std::true_type;
    static constexpr uint32_t need = 1;

    uint32_t cb_id;

    void emit(uint32_t dst, uint32_t tile) const {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        ckernel::copy_tile(cb_id, tile, dst);
#else
        (void)dst;
        (void)tile;
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

    // Templated so it is only instantiated when an FPU chain actually uses it;
    // the pack-side epilogue is not yet bound to metal (see unified_metal.hpp).
    template <int = 0>
    static void apply_from_pack(uint32_t base, uint32_t count) {
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
// checkable with a static_assert. Names follow matmul_block's own parameters:
//
//   A is rt_dim x kt_dim tiles, B is kt_dim x ct_dim, C is rt_dim x ct_dim.
//
// `in1_row_stride` is how far to step in B's CB to move down one k row -- B's
// full block width, which is not necessarily ct_dim when B holds several
// subblocks side by side.
template <uint32_t RtDim, uint32_t CtDim, uint32_t KtDim, uint32_t NumBlocks = 1, uint32_t In1RowStride = CtDim>
struct MatmulGeometry {
    static constexpr uint32_t rt_dim = RtDim;  // output rows  (A rows)
    static constexpr uint32_t ct_dim = CtDim;  // output cols  (B cols)
    static constexpr uint32_t kt_dim = KtDim;  // inner dim
    static constexpr uint32_t num_blocks = NumBlocks;
    static constexpr uint32_t in1_row_stride = In1RowStride;
    static constexpr uint32_t out_subblock_num_tiles = RtDim * CtDim;
};

template <typename Geometry, typename Chain>
struct MatmulNode {
    using fusion_kind = FPUFusion;
    using geometry = Geometry;
    using chain = Chain;

    uint32_t in0_cb;
    uint32_t in1_cb;
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
// Hardware startup
//
// These are MMIO writes and must run once, at kernel entry, before any other
// compute API call -- which is why they are the kernel's job rather than the
// strategy's. Which one you call depends on the fusion kind you are about to
// use; they configure the ALU differently and are not interchangeable.
//
// Both are self-guarding: on a data-movement build the body preprocesses away,
// so kernels call them unconditionally and no no-op counterpart is needed.
// ---------------------------------------------------------------------------

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
// compute_init() instead leaves the ALU configured for SFPU work and matmul
// then runs against a state it cannot use.
template <typename Geometry>
inline void matmul_init(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t transpose = 0) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    ckernel::compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(in0_cb, in1_cb, out_cb);
    ckernel::matmul_block_init(in0_cb, in1_cb, transpose, Geometry::ct_dim, Geometry::rt_dim, Geometry::kt_dim);
#else
    (void)in0_cb;
    (void)in1_cb;
    (void)out_cb;
    (void)transpose;
#endif
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
    static void run(const Node& node, uint32_t cb_id, uint32_t num_tiles) {
        static_assert(
            expr::need_v<Node> <= kMaxDstTiles,
            "SFPU expression needs more DST slots than the hardware has; "
            "split it across an intermediate Storage");
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_reserve_back(cb_id, num_tiles);
        for (uint32_t i = 0; i < num_tiles; ++i) {
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

// FPU: the node owns the loop. matmul_block accumulates into DST across the
// inner (k) dimension, and the result is packed once, on the final block, after
// the unary epilogue runs from pack.
//
// Mirrors bmm_large_block_zm_fused_bias_activation.cpp: matmul_block_init once,
// then a k-loop stepping in0 right by one tile and in1 down by one row.
template <>
struct Strategy<FPUFusion> {
    template <typename Node>
    static void run(const Node& node, uint32_t cb_id, uint32_t /*num_tiles*/) {
        using G = typename Node::geometry;
        static_assert(
            G::out_subblock_num_tiles <= kMaxDstTiles, "matmul rt_dim * ct_dim exceeds the DST register file");
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using Chain = typename Node::chain;
        constexpr uint32_t kTranspose = 0;

        // Startup is the kernel's job -- see matmul_init() above.

        for (uint32_t block = 0; block < G::num_blocks; ++block) {
            const bool last_out = (block == G::num_blocks - 1);
            ckernel::tile_regs_acquire();

            // Accumulate across the inner dimension. matmul_block internally
            // advances dst_index, so the whole rt_dim x ct_dim subblock is the
            // accumulator -- there is nothing here for the allocator to hand out.
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
                in0_index += 1;                  // step right along A
                in1_index += G::in1_row_stride;  // step down one row of B
            }

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
                // intermediate CB here (mm_partials). Omitted -- so num_blocks
                // > 1 currently drops everything but the last block.
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
