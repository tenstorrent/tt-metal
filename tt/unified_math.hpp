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

// Usable DST tiles per acquire.
//
// DST holds 16 tiles, but under the default DstSync::SyncHalf the register file
// is banked in two and only ONE half is addressable between a tile_regs_acquire
// and its release -- so the budget is 8, not 16. ttnn's own matmul never picks a
// larger subblock: SUBBLOCK_HW_CHOICES tops out at 8 ({4,2},{2,4},{8,1},{1,8})
// in ttnn/.../matmul/device/config/matmul_program_config.cpp.
//
// Exceeding it is not a clean failure. A 16-tile subblock still round-trips in
// Dst mode, because math writes and pack read the same wrong mapping and it
// cancels; L1 mode exposes it, because the packer's read-modify-write of L1
// depends on the absolute DST->L1 mapping and only the upper half is right --
// measured as tiles 0..7 overwritten instead of accumulated.
//
// Halves again to 4 under fp32 accumulate (see reg_api.h), which this model does
// not enable; raising it to 16 would require dst_full_sync_en on the compute
// config.
inline constexpr uint32_t kMaxDstTiles = 8;

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

// A unary usable in either kind: as a node in an SFPU expression tree, or as a
// link in an FPU node's epilogue chain. Both run on the SFPU against DST, so
// one implementation serves both -- `apply_in_place` is just apply(s, s).
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

template <typename Geometry, typename Chain>
auto exp_(const MatmulNode<Geometry, Chain>& m) {
    return MatmulNode<Geometry, expr::chain_append_t<Chain, ExpOp>>{m.in0_cb, m.in1_cb};
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

// How a multi-block FPU fusion carries its running total.
//
//   Dst -- the partial is reloaded from a separate buffer into DST before each
//          matmul, which then accumulates on top. Costs a DST round-trip and two
//          format reconfigs per block. DST holds the *running total*, so a
//          finish-only epilogue is meaningful and a per-step chain sees the
//          total so far.
//
//   L1  -- the packer accumulates into L1 instead. No reload, and DST only ever
//          holds one block's product -- so a per-step chain sees that block's
//          contribution alone, but a finish-only epilogue is impossible, since
//          the total never sits in DST.
enum class AccumulatorMode {
    Dst,
    L1,
};

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

// FPU: one k-block per call. The kernel owns the k-loop (see Accumulator in
// tt/unified_api.h), because the operand CBs must be waited and popped per
// block so the reader can stream them.
//
// Mirrors bmm_large_block_zm_fused_bias_activation.cpp:
//   acquire -> [reload partials into DST] -> matmul_block across k
//           -> [epilogue on DST] -> commit -> pack to partials, or to out on
//              the final block.
template <>
struct Strategy<FPUFusion> {
    // Single-shot: one k-block, no accumulation buffer. This is the shape
    // Storage::store() uses, so `out.store(matmul<Geom>(a, b))` still works for
    // a one-round matmul -- and for any future FPU op that does not accumulate.
    // With reload=false and finish=true the accumulation buffer is never
    // touched, so passing the destination for both is safe.
    template <typename Node>
    static void run(const Node& node, uint32_t cb_id, uint32_t /*num_tiles*/) {
        run<AccumulatorMode::Dst>(node, /*acc_cb=*/cb_id, /*out_cb=*/cb_id, /*reload=*/false, /*finish=*/true);
    }

    // `Node::chain` is the PER-STEP chain: it runs on every call. `EpilogueChain`
    // runs only on the finishing call, against the completed accumulator.
    //
    //     accumulate(relu(mm), finish)                   -> per-step
    //     accumulate(mm, finish, [](auto n){return relu(n);}) -> finish only
    //
    // What a per-step chain sees differs by mode. In Dst mode the reload happens
    // before the matmul, so DST already holds the running total and the chain
    // sees f(total-so-far), not f(this contribution) -- isolating the
    // contribution would need a second rt*ct-sized scratch region, which does
    // not fit. L1 mode gets it for free: the packer does the summing, so DST
    // only ever holds one block's product and a per-step chain is a true
    // per-contribution f.
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
        constexpr uint32_t kTranspose = 0;

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

        // Per-step chain: every call.
        if constexpr (!Chain::empty) {
            for (uint32_t t = 0; t < kAccTiles; ++t) {
                Chain::apply_in_place(t);
            }
        }

        // Epilogue chain: the finishing call only, against the completed total.
        // In L1 mode the total is not in DST yet, so it runs in the copy-out
        // stage below instead.
        if constexpr (Mode == AccumulatorMode::Dst) {
            if constexpr (!EpilogueChain::empty) {
                if (finish) {
                    for (uint32_t t = 0; t < kAccTiles; ++t) {
                        EpilogueChain::apply_in_place(t);
                    }
                }
            }
        }

        ckernel::tile_regs_commit();

        if constexpr (Mode == AccumulatorMode::Dst) {
            const uint32_t dest = finish ? out_cb : acc_cb;
            cb_reserve_back(dest, kAccTiles);
            ckernel::tile_regs_wait();
            ckernel::pack_block(0, dest, kAccTiles);
            ckernel::tile_regs_release();
            cb_push_back(dest, kAccTiles);
        } else {
            // L1: the packer adds this block's product into what is already at
            // the destination, so the running total lives in L1 and never
            // occupies DST.
            //
            // The push/pop pair is load-bearing, not bookkeeping. pack_block
            // advances the CB's fifo_wr_tile_ptr itself and cb_push_back is the
            // only thing that resets it (llk_io_pack.h), so a pack without a
            // matching push lands one block further along each round instead of
            // on top of the previous one. Pushing then popping a CB sized to
            // exactly one block wraps both pointers back to the base address --
            // which still holds the partial, since a pop does not erase.
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
            } else {
                // Move the completed total into the output buffer. Copying it
                // through DST rather than letting the DM writer drain acc_cb
                // keeps one popper per CB -- compute owns acc_cb, the writer
                // owns out_cb -- and gives the finish-only epilogue the whole
                // total in DST, exactly as in Dst mode.
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

                // Restore the state matmul_block needs, so the accumulator can
                // be cleared and driven again for the next output block.
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

}  // namespace unified
}  // namespace tt
