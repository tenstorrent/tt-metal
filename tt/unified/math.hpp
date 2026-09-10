// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include <tt/unified/expr.hpp>
#include <tt/unified/shape.hpp>

#if !defined(IS_COMPUTE_THREAD) && !defined(IS_DM_THREAD)
#error "include <tt/unified/core> (or a binding) before tt/unified/math.hpp"
#endif

namespace tt {
namespace unified {

#if !defined(TT_UNIFIED_DST_32BIT)
#define TT_UNIFIED_DST_32BIT 0
#endif
inline constexpr uint32_t kMaxDstTiles = TT_UNIFIED_DST_32BIT ? 4 : 8;

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
static_assert(
    DST_ACCUM_MODE == (TT_UNIFIED_DST_32BIT != 0),
    "TT_UNIFIED_DST_32BIT disagrees with metal's generated DST_ACCUM_MODE: the host's "
    "enable_32_bit_dest and the define it emits to the kernels have come apart");
#endif

namespace detail {
inline constexpr uint32_t kPackUnset = ~uint32_t(0);
inline uint32_t g_pack_configured = kPackUnset;

inline uint32_t g_unpack_geometry_a = kPackUnset;
inline uint32_t g_unpack_geometry_b = kPackUnset;
inline uint32_t g_pack_geometry = kPackUnset;
}  // namespace detail

inline void unpack_geometry_to(uint32_t dfb_id) {
#if defined(TT_U_HAVE_DFB_TILE_GEOMETRY)
    const uint32_t geometry = unpack_tile_geometry(dfb_id);
    if (geometry != detail::g_unpack_geometry_a || geometry != detail::g_unpack_geometry_b) {
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(dfb_id)));
        detail::g_unpack_geometry_a = geometry;
        detail::g_unpack_geometry_b = geometry;
    }
#else
    (void)dfb_id;
#endif
}

inline void unpack_geometry_to(uint32_t dfb0, uint32_t dfb1) {
#if defined(TT_U_HAVE_DFB_TILE_GEOMETRY)
    const uint32_t geometry_a = unpack_tile_geometry(dfb0);
    const uint32_t geometry_b = unpack_tile_geometry(dfb1);
    if (geometry_a != detail::g_unpack_geometry_a || geometry_b != detail::g_unpack_geometry_b) {
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(dfb0, dfb1)));
        detail::g_unpack_geometry_a = geometry_a;
        detail::g_unpack_geometry_b = geometry_b;
    }
#else
    (void)dfb0;
    (void)dfb1;
#endif
}

inline void pack_geometry_to(uint32_t dfb_id) {
#if defined(TT_U_HAVE_PACK_TILE_GEOMETRY)
    const uint32_t geometry = pack_tile_geometry(dfb_id);
    if (geometry != detail::g_pack_geometry) {
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(dfb_id)));
        PACK((llk_pack_init(dfb_id)));
        detail::g_pack_geometry = geometry;
    }
#else
    (void)dfb_id;
#endif
}

inline void unpack_geometry_assume(uint32_t dfb_a, uint32_t dfb_b) {
#if defined(TT_U_HAVE_DFB_TILE_GEOMETRY)
    detail::g_unpack_geometry_a = unpack_tile_geometry(dfb_a);
    detail::g_unpack_geometry_b = unpack_tile_geometry(dfb_b);
#else
    (void)dfb_a;
    (void)dfb_b;
#endif
}

inline void pack_geometry_assume(uint32_t dfb_id) {
#if defined(TT_U_HAVE_PACK_TILE_GEOMETRY)
    detail::g_pack_geometry = pack_tile_geometry(dfb_id);
#else
    (void)dfb_id;
#endif
}

__attribute__((noinline)) inline void pack_one(uint32_t dst, uint32_t dfb_id) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    ckernel::pack_tile(dst, dfb_id);
#else
    (void)dst;
    (void)dfb_id;
#endif
}

inline void pack_to(uint32_t dfb_id) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    if (detail::g_pack_configured == dfb_id) {
        return;
    }
    if (detail::g_pack_configured == detail::kPackUnset) {
        ckernel::pack_reconfig_data_format(dfb_id);
    } else {
        ckernel::pack_reconfig_data_format(detail::g_pack_configured, dfb_id);
    }
    pack_geometry_to(dfb_id);
    detail::g_pack_configured = dfb_id;
#else
    (void)dfb_id;
#endif
}

inline void pack_to_forget() {
    detail::g_pack_configured = detail::kPackUnset;
    detail::g_pack_geometry = detail::kPackUnset;
}

constexpr uint32_t largest_divisor_at_most(uint32_t dim, uint32_t cap) {
    for (uint32_t f = cap; f > 0; --f) {
        if (dim % f == 0) {
            return f;
        }
    }
    return 1;
}

struct DstSubblock {
    uint32_t rows;
    uint32_t cols;

    constexpr uint32_t tiles() const { return rows * cols; }
};

constexpr DstSubblock dst_subblock(uint32_t rt_dim, uint32_t ct_dim, uint32_t capacity = kMaxDstTiles) {
    const uint32_t cols = largest_divisor_at_most(ct_dim, capacity);
    const uint32_t remaining = (cols == ct_dim) ? capacity / cols : 1;
    return DstSubblock{largest_divisor_at_most(rt_dim, remaining), cols};
}

constexpr uint32_t block_tile_index(uint32_t r0, uint32_t c0, uint32_t t, uint32_t sub_cols, uint32_t ct_dim) {
    return (r0 + t / sub_cols) * ct_dim + c0 + t % sub_cols;
}

template <typename S>
struct TileSource : expr::Fluent<TileSource<S>> {
    using is_expr_node = std::true_type;
    using shape = S;
    static constexpr uint32_t need = 1;

    uint32_t dfb_id;

    uint32_t source_dfb() const { return dfb_id; }

    void emit(uint32_t dst, uint32_t tile, bool reconfigure) const {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        if (reconfigure) {
            unpack_geometry_to(dfb_id);
            ckernel::reconfig_data_format_srca(dfb_id);
            ckernel::copy_tile_to_dst_init_short(dfb_id);
        }
        ckernel::copy_tile(dfb_id, tile, dst);
#else
        (void)dst;
        (void)tile;
        (void)reconfigure;
#endif
    }
};

enum class FpuOp { Add, Sub, Mul };

template <FpuOp TheOp>
struct FpuBinary {
    static constexpr bool fpu_capable = true;
    static constexpr FpuOp fpu_op = TheOp;

    static void fpu_seed_init(uint32_t dfb0, uint32_t dfb1) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        unpack_geometry_to(dfb0, dfb1);
        ckernel::reconfig_data_format(dfb0, dfb1);
        if constexpr (TheOp == FpuOp::Add) {
            ckernel::add_tiles_init(dfb0, dfb1);
        } else if constexpr (TheOp == FpuOp::Sub) {
            ckernel::sub_tiles_init(dfb0, dfb1);
        } else {
            ckernel::mul_tiles_init(dfb0, dfb1);
        }
#else
        (void)dfb0;
        (void)dfb1;
#endif
    }

    static void fpu_seed_apply(uint32_t dfb0, uint32_t dfb1, uint32_t t0, uint32_t t1, uint32_t dst) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        if constexpr (TheOp == FpuOp::Add) {
            ckernel::add_tiles(dfb0, dfb1, t0, t1, dst);
        } else if constexpr (TheOp == FpuOp::Sub) {
            ckernel::sub_tiles(dfb0, dfb1, t0, t1, dst);
        } else {
            ckernel::mul_tiles(dfb0, dfb1, t0, t1, dst);
        }
#else
        (void)dfb0;
        (void)dfb1;
        (void)t0;
        (void)t1;
        (void)dst;
#endif
    }

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    static constexpr ckernel::EltwiseBinaryType kType = TheOp == FpuOp::Add   ? ckernel::EltwiseBinaryType::ELWADD
                                                        : TheOp == FpuOp::Sub ? ckernel::EltwiseBinaryType::ELWSUB
                                                                              : ckernel::EltwiseBinaryType::ELWMUL;
    template <bool DstIsLhs>
    static constexpr ckernel::EltwiseBinaryReuseDestType kDir =
        DstIsLhs ? ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA
                 : ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB;
#endif

    template <bool DstIsLhs>
    static void fpu_reuse_init(uint32_t dfb) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        unpack_geometry_to(dfb);
        if constexpr (DstIsLhs) {
            ckernel::reconfig_data_format_srcb(dfb);
        } else {
            ckernel::reconfig_data_format_srca(dfb);
        }
        ckernel::binary_dest_reuse_tiles_init<kType, kDir<DstIsLhs>>(dfb);
#else
        (void)dfb;
#endif
    }

    template <bool DstIsLhs>
    static void fpu_reuse_apply(uint32_t dfb, uint32_t tile, uint32_t dst) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        ckernel::binary_dest_reuse_tiles<kType, kDir<DstIsLhs>>(dfb, tile, dst);
#else
        (void)dfb;
        (void)tile;
        (void)dst;
#endif
    }
};

struct AddOp : FpuBinary<FpuOp::Add> {
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

struct SubOp : FpuBinary<FpuOp::Sub> {
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

struct MulOp : FpuBinary<FpuOp::Mul> {
#if defined(TT_UNIFIED_SFPU_MUL)
    static constexpr bool fpu_capable = false;
#endif
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

struct MaxOp {
    static constexpr bool fpu_capable = false;
    static void apply(uint32_t lhs, uint32_t rhs, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        ckernel::binary_max_tile_init();
        ckernel::binary_max_tile(lhs, rhs, out);
#else
        (void)lhs;
        (void)rhs;
        (void)out;
#endif
    }
};

struct DivOp {
    static constexpr bool fpu_capable = false;
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

#if defined(TRISC_MATH)
constexpr bool kMathApprox = APPROX;
#else
constexpr bool kMathApprox = false;
#endif

struct ExpOp {
    static void apply(uint32_t src, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;
        ckernel::exp_tile_init<kMathApprox>();
        ckernel::exp_tile<kMathApprox>(out);
#else
        (void)src;
        (void)out;
#endif
    }
    static void apply_in_place(uint32_t slot) { apply(slot, slot); }
};

struct SiluOp {
    static void apply(uint32_t src, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;
        ckernel::silu_tile_init();
        ckernel::silu_tile(out);
#else
        (void)src;
        (void)out;
#endif
    }

    static void apply_in_place(uint32_t slot) { apply(slot, slot); }
};

struct ReluOp {
    static void apply(uint32_t src, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;
        ckernel::relu_tile_init();
        ckernel::relu_tile(out);
#else
        (void)src;
        (void)out;
#endif
    }

    static void apply_in_place(uint32_t slot) { apply(slot, slot); }
};

struct RecipOp {
    static void apply(uint32_t src, uint32_t out) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        (void)src;
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
        (void)src;
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
        (void)src;
        ckernel::rsqrt_tile_init();
        ckernel::rsqrt_tile(out);
#else
        (void)src;
        (void)out;
#endif
    }

    static void apply_in_place(uint32_t slot) { apply(slot, slot); }
};

using SFPUFusion = expr::TreeKind;

struct FPUFusion {};

using FpuEltwiseFusion = expr::FpuTreeKind;

struct ReduceFusion {};

enum class TransposeB { No, Yes };

template <typename SA, typename SB, TransposeB Tr = TransposeB::No>
struct MatmulGeometry {
    static_assert(
        SA::cols == SB::rows, "matmul inner dimension disagrees: operand A's columns must equal operand B's rows");
    static_assert(SA::leading == SB::leading, "matmul operands disagree on their leading (batch) extent");

    static_assert(
        SB::tile::rows <= SA::tile::cols,
        "matmul tile shapes are incompatible: operand B's tile is TALLER than operand A's tile is wide, "
        "so the inner dimension reads columns of A that its tile does not have. Equal extents are the "
        "clean case; A wider than B is tall is the padded one, and needs A's surplus columns to be "
        "zero. This is neither");
    static_assert(
        SA::tile::cols == SB::tile::cols,
        "matmul tile shapes are incompatible: the output inherits operand A's tile, and its COLUMNS "
        "come from operand B, so the two tiles must be the same width or the result claims an extent "
        "it does not hold");

    static constexpr uint32_t rt_dim = SA::rows;
    static constexpr uint32_t ct_dim = SB::cols;
    static constexpr uint32_t kt_dim = SA::cols;
    static constexpr uint32_t in1_row_stride = SB::cols;
    static constexpr uint32_t out_subblock_num_tiles = rt_dim * ct_dim;

    static constexpr uint32_t transpose = (Tr == TransposeB::Yes) ? 1u : 0u;

    using out_shape = with_hw<SA, rt_dim, ct_dim>;
};

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

    template <typename Operand>
    auto add(const Operand& operand) const {
        static_assert(
            same_shape_v<typename Operand::shape, typename geometry::out_shape>,
            "a fused addend must have the matmul's OUTPUT shape -- for one row broadcast "
            "down the block, that is bias(), not add()");
        MatmulNode<SA, SB, Tr, Chain> out{{}, in0_dfb, in1_dfb, bias_dfb, operand.get_dfb_id()};
        return out;
    }

    template <typename Operand>
    auto bias(const Operand& operand) const {
        static_assert(
            same_shape_v<typename Operand::shape, Shape<1, geometry::ct_dim>>,
            "a fused bias must be Shape<1, ct_dim> -- one row of the output block's width");
        MatmulNode<SA, SB, Tr, Chain> out{{}, in0_dfb, in1_dfb, operand.get_dfb_id(), addend_dfb};
        return out;
    }

    uint32_t in0_dfb;
    uint32_t in1_dfb;
    uint32_t bias_dfb = kNoBias;
    uint32_t addend_dfb = kNoBias;
};

enum class Axis { Rows, Cols, Both };

using ReduceAxis = Axis;

enum class ReducePool { Sum, Avg, Max };

template <typename S, Axis A>
using reduce_shape = with_hw<S, (A == Axis::Cols ? S::rows : 1), (A == Axis::Rows ? S::cols : 1)>;

template <typename SB, Axis A>
using bcast_vec_shape = reduce_shape<SB, A>;

template <typename T>
struct always_false : std::false_type {};

struct FPUFusion;

template <typename T>
struct is_fpu_fusion : std::is_same<expr::kind_of_t<T>, FPUFusion> {};

struct BcastFusion {};

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

template <Axis A, typename S>
struct Broadcast {
    static constexpr Axis axis = A;
    using shape = S;

    uint32_t dfb_id;
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

    using shape = SB;

    static_assert(
        same_shape_v<SV, bcast_vec_shape<SB, A>>,
        "the broadcast vector's shape does not match the axis it declares: Axis::Rows needs "
        "Shape<1, cols>, Axis::Cols needs Shape<rows, 1>, Axis::Both needs Shape<1, 1>, all "
        "relative to the block");

    static constexpr uint32_t vec_tile(uint32_t t) {
        return A == Axis::Rows ? t % SB::cols : (A == Axis::Cols ? t / SB::cols : 0);
    }

    uint32_t block_dfb;
    uint32_t vec_dfb;
};

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
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

template <typename S>
struct ReduceGeometry {
    static_assert(
        S::leading == 1,
        "reducing a shape with a leading (batch) extent is not implemented -- the strategy walks one "
        "2-D grid, so run the reduction per batch from the kernel's own loop");

    static constexpr uint32_t ht = S::rows;
    static constexpr uint32_t wt = S::cols;
    static constexpr uint32_t num_tiles = ht * wt;

    static constexpr uint32_t out_tiles(ReduceAxis axis) {
        return axis == ReduceAxis::Rows ? wt : (axis == ReduceAxis::Cols ? ht : 1);
    }

    static constexpr uint32_t elements(ReduceAxis axis) {
        return axis == ReduceAxis::Rows
                   ? logical_rows_v<S>
                   : (axis == ReduceAxis::Cols ? logical_cols_v<S> : logical_rows_v<S> * logical_cols_v<S>);
    }

    static constexpr uint32_t group(ReduceAxis axis) {
        return axis == ReduceAxis::Rows ? ht : (axis == ReduceAxis::Cols ? wt : num_tiles);
    }

    static constexpr uint32_t contributor(ReduceAxis axis, uint32_t o, uint32_t g) {
        return axis == ReduceAxis::Rows ? g * wt + o : (axis == ReduceAxis::Cols ? o * wt + g : g);
    }
};

template <typename S, ReduceAxis Axis, ReducePool Pool, typename Chain>
struct ReduceNode : expr::Fluent<ReduceNode<S, Axis, Pool, Chain>> {
    using fusion_kind = ReduceFusion;
    using in_shape = S;
    using geometry = ReduceGeometry<S>;
    using chain = Chain;
    using shape = reduce_shape<S, Axis>;
    static constexpr ReduceAxis axis = Axis;
    static constexpr ReducePool pool = Pool;

    uint32_t in_dfb;
    uint32_t scaler_dfb;
};

template <typename S, ReduceAxis A, ReducePool P, typename Chain>
auto silu(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, SiluOp>>{{}, r.in_dfb, r.scaler_dfb};
}

template <typename S, Axis A, ReducePool P, typename Chain>
auto relu(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, ReluOp>>{{}, r.in_dfb, r.scaler_dfb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto silu(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, SiluOp>>{{}, {}, b.block_dfb, b.vec_dfb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto relu(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, ReluOp>>{{}, {}, b.block_dfb, b.vec_dfb};
}

template <typename S, ReduceAxis A, ReducePool P, typename Chain>
auto exp_(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, ExpOp>>{{}, r.in_dfb, r.scaler_dfb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto exp_(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, ExpOp>>{{}, {}, b.block_dfb, b.vec_dfb};
}

template <typename S, ReduceAxis A, ReducePool P, typename Chain>
auto recip(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, RecipOp>>{{}, r.in_dfb, r.scaler_dfb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto recip(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, RecipOp>>{{}, {}, b.block_dfb, b.vec_dfb};
}

template <typename S, ReduceAxis A, ReducePool P, typename Chain>
auto sqrt_(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, SqrtOp>>{{}, r.in_dfb, r.scaler_dfb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto sqrt_(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, SqrtOp>>{{}, {}, b.block_dfb, b.vec_dfb};
}

template <typename S, ReduceAxis A, ReducePool P, typename Chain>
auto rsqrt(const ReduceNode<S, A, P, Chain>& r) {
    return ReduceNode<S, A, P, expr::chain_append_t<Chain, RsqrtOp>>{{}, r.in_dfb, r.scaler_dfb};
}

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
auto rsqrt(const BcastNode<Op, A, SB, SV, Chain>& b) {
    return BcastNode<Op, A, SB, SV, expr::chain_append_t<Chain, RsqrtOp>>{{}, {}, b.block_dfb, b.vec_dfb};
}

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

template <typename A, typename B, typename = std::enable_if_t<is_operand<A>::value && is_operand<B>::value>>
auto max_(const A& a, const B& b) {
    static_assert(
        !is_fpu_fusion<A>::value && !is_fpu_fusion<B>::value,
        "an FPU fusion consumes all of DST, so it cannot be an operand of max_; store it to an "
        "intermediate Storage first");
    using LN = std::decay_t<decltype(as_node(a))>;
    using RN = std::decay_t<decltype(as_node(b))>;
    return expr::Bin<MaxOp, LN, RN>{{}, as_node(a), as_node(b)};
}

template <typename N, typename = std::enable_if_t<expr::is_expr<N>::value>>
auto relu(const N& n) {
    return expr::Un<ReluOp, N>{{}, n};
}

template <typename N, typename = std::enable_if_t<expr::is_expr<N>::value>>
auto silu(const N& n) {
    return expr::Un<SiluOp, N>{{}, n};
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
auto silu(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, SiluOp>>{
        {}, m.in0_dfb, m.in1_dfb, m.bias_dfb, m.addend_dfb};
}

template <typename SA, typename SB, TransposeB Tr, typename Chain>
auto relu(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, ReluOp>>{
        {}, m.in0_dfb, m.in1_dfb, m.bias_dfb, m.addend_dfb};
}

template <typename SA, typename SB, TransposeB Tr, typename Chain>
auto exp_(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, ExpOp>>{
        {}, m.in0_dfb, m.in1_dfb, m.bias_dfb, m.addend_dfb};
}

template <typename SA, typename SB, TransposeB Tr, typename Chain>
auto recip(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, RecipOp>>{
        {}, m.in0_dfb, m.in1_dfb, m.bias_dfb, m.addend_dfb};
}

template <typename SA, typename SB, TransposeB Tr, typename Chain>
auto sqrt_(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, SqrtOp>>{
        {}, m.in0_dfb, m.in1_dfb, m.bias_dfb, m.addend_dfb};
}

template <typename SA, typename SB, TransposeB Tr, typename Chain>
auto rsqrt(const MatmulNode<SA, SB, Tr, Chain>& m) {
    return MatmulNode<SA, SB, Tr, expr::chain_append_t<Chain, RsqrtOp>>{
        {}, m.in0_dfb, m.in1_dfb, m.bias_dfb, m.addend_dfb};
}

template <typename T, typename = void>
struct bcast_block_shape {
    using type = Shape<1, 1>;
    static constexpr bool ok = false;
};

template <typename S>
struct bcast_block_shape<TileSource<S>> {
    using type = S;
    static constexpr bool ok = true;
};

#define TT_UNIFIED_BCAST_OPERATOR(sym, OpType)                                                              \
    template <typename B, Axis A, typename SV, typename = std::enable_if_t<is_operand<B>::value>>           \
    auto operator sym(const B& block, Broadcast<A, SV> vec) {                                               \
        using BN = std::decay_t<decltype(as_node(block))>;                                                  \
        static_assert(                                                                                      \
            bcast_block_shape<BN>::ok,                                                                      \
            "a broadcast's left operand must be a stored buffer, not an expression -- the FPU "             \
            "reads both operands from dataflow buffers while an expression lives in DST, so "               \
            "store it to a Storage first");                                                                 \
        using SB = typename bcast_block_shape<BN>::type;                                                    \
        return BcastNode<OpType, A, SB, SV, expr::UnaryChain<>>{{}, {}, as_node(block).dfb_id, vec.dfb_id}; \
    }                                                                                                       \
    template <typename R, Axis A, typename SV>                                                              \
    auto operator sym(Broadcast<A, SV>, const R&) {                                                         \
        static_assert(                                                                                      \
            always_false<R>::value,                                                                         \
            "a broadcast has to be the RIGHT operand -- metal reads the broadcast vector from in1, "        \
            "so write `block " #sym " bcast<Axis::...>(vec)`");                                             \
    }

TT_UNIFIED_BCAST_OPERATOR(+, AddOp)
TT_UNIFIED_BCAST_OPERATOR(-, SubOp)
TT_UNIFIED_BCAST_OPERATOR(*, MulOp)

#undef TT_UNIFIED_BCAST_OPERATOR

namespace expr {
template <typename N>
auto fluent_relu(const N& n) {
    return relu(n);
}
template <typename N>
auto fluent_silu(const N& n) {
    return silu(n);
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

template <TransposeB Tr = TransposeB::No, typename SA, typename SB>
auto matmul(TileSource<SA> a, TileSource<SB> b) {
    return MatmulNode<SA, SB, Tr, expr::UnaryChain<>>{{}, a.dfb_id, b.dfb_id, kNoBias};
}

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

inline void compute_init(uint32_t in_dfb, uint32_t out_dfb) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    ckernel::init_sfpu(in_dfb, out_dfb);
    unpack_geometry_assume(in_dfb, in_dfb);
    pack_geometry_assume(out_dfb);
#else
    (void)in_dfb;
    (void)out_dfb;
#endif
}

template <typename SA, typename SB, TransposeB Tr = TransposeB::No>
inline void matmul_init(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t out_dfb) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    using Geometry = MatmulGeometry<SA, SB, Tr>;
    ckernel::compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(in0_dfb, in1_dfb, out_dfb);
    unpack_geometry_assume(in1_dfb, in0_dfb);
    pack_geometry_assume(out_dfb);
    ckernel::matmul_block_init(
        in0_dfb, in1_dfb, Geometry::transpose, Geometry::ct_dim, Geometry::rt_dim, Geometry::kt_dim);
#else
    (void)in0_dfb;
    (void)in1_dfb;
    (void)out_dfb;
#endif
}

enum class AccumulatorMode {
    Dst,
    L1,
};

template <typename Kind>
struct Strategy;

template <>
struct Strategy<SFPUFusion> {
    template <typename Node>
    static void run(const Node& node, uint32_t dfb_id, uint32_t num_tiles) {
        static_assert(
            expr::need_v<Node> <= kMaxDstTiles,
            "SFPU expression needs more DST slots than the hardware has; "
            "split it across an intermediate Storage");
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        constexpr uint32_t kLeaves = expr::leaf_slots_v<Node>;
        constexpr bool kLeafOuter = kLeaves > 1 && kLeaves * 2 <= kMaxDstTiles;
        pack_to(dfb_id);
        buffer(dfb_id).reserve_back(num_tiles);
        if constexpr (kLeafOuter) {
            constexpr uint32_t kPerAcquire = kMaxDstTiles / kLeaves;
            for (uint32_t base = 0; base < num_tiles; base += kPerAcquire) {
                const uint32_t remaining = num_tiles - base;
                const uint32_t count = remaining < kPerAcquire ? remaining : kPerAcquire;
                ckernel::tile_regs_acquire();
                expr::load_leaves(node, base, count);
                for (uint32_t k = 0; k < count; ++k) {
                    expr::apply_ops(node, k * kLeaves);
                }
                ckernel::tile_regs_commit();
                ckernel::tile_regs_wait();
                for (uint32_t k = 0; k < count; ++k) {
                    pack_one(k * kLeaves + expr::leaf_result_ofs_v<Node>, dfb_id);
                }
                ckernel::tile_regs_release();
            }
        } else {
            constexpr bool kEveryTile = expr::leaf_count_v<Node> > 1;
            for (uint32_t i = 0; i < num_tiles; ++i) {
                ckernel::tile_regs_acquire();
                expr::emit(node, i, kEveryTile || i == 0);
                ckernel::tile_regs_commit();
                ckernel::tile_regs_wait();
                pack_one(expr::result_slot_v<Node>, dfb_id);
                ckernel::tile_regs_release();
            }
        }
        buffer(dfb_id).push_back(num_tiles);
#else
        (void)node;
        (void)dfb_id;
        (void)num_tiles;
#endif
    }
};

template <>
struct Strategy<FpuEltwiseFusion> {
    template <typename Node>
    static void run(const Node& node, uint32_t dfb_id, uint32_t num_tiles) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        constexpr uint32_t kPerAcquire = kMaxDstTiles;
        pack_to(dfb_id);
        buffer(dfb_id).reserve_back(num_tiles);
        for (uint32_t base = 0; base < num_tiles; base += kPerAcquire) {
            const uint32_t remaining = num_tiles - base;
            const uint32_t count = remaining < kPerAcquire ? remaining : kPerAcquire;
            ckernel::tile_regs_acquire();
            expr::fpu_stages(node, base, count);
            ckernel::tile_regs_commit();
            ckernel::tile_regs_wait();
            for (uint32_t k = 0; k < count; ++k) {
                pack_one(k, dfb_id);
            }
            ckernel::tile_regs_release();
        }
        buffer(dfb_id).push_back(num_tiles);
#else
        (void)node;
        (void)dfb_id;
        (void)num_tiles;
#endif
    }
};

template <>
struct Strategy<FPUFusion> {
    template <typename Node, typename EpilogueChain>
    static void bias_finish(const Node& node, uint32_t acc_dfb, uint32_t out_dfb, uint32_t bias_dfb, EpilogueChain) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using G = typename Node::geometry;
        constexpr uint32_t kTranspose = G::transpose;

        constexpr DstSubblock kSub = dst_subblock(G::rt_dim, G::ct_dim);
        constexpr uint32_t kSubTiles = kSub.tiles();

        ckernel::reconfig_data_format(acc_dfb, bias_dfb);
        ckernel::add_bcast_rows_init_short(acc_dfb, bias_dfb);
        pack_to(out_dfb);

        for (uint32_t r0 = 0; r0 < G::rt_dim; r0 += kSub.rows) {
            for (uint32_t c0 = 0; c0 < G::ct_dim; c0 += kSub.cols) {
                ckernel::tile_regs_acquire();
                buffer(acc_dfb).wait_front(kSubTiles);
                for (uint32_t t = 0; t < kSubTiles; ++t) {
                    ckernel::add_tiles_bcast_rows(acc_dfb, bias_dfb, t, c0 + t % kSub.cols, t);
                }
                buffer(acc_dfb).pop_front(kSubTiles);

                if constexpr (!EpilogueChain::empty) {
                    for (uint32_t t = 0; t < kSubTiles; ++t) {
                        EpilogueChain::apply_in_place(t);
                    }
                }

                ckernel::tile_regs_commit();
                buffer(out_dfb).reserve_back(kSubTiles);
                ckernel::tile_regs_wait();
                ckernel::pack_block(0, out_dfb, kSubTiles);
                ckernel::tile_regs_release();
                buffer(out_dfb).push_back(kSubTiles);
            }
        }

        unpack_geometry_to(node.in1_dfb, node.in0_dfb);
        ckernel::reconfig_data_format_srca(acc_dfb, node.in1_dfb);
        ckernel::matmul_block_init(node.in0_dfb, node.in1_dfb, kTranspose, kSub.cols, kSub.rows, G::kt_dim);
#else
        (void)node;
        (void)acc_dfb;
        (void)out_dfb;
        (void)bias_dfb;
#endif
    }

    template <typename Node>
    static void run(const Node& node, uint32_t dfb_id, uint32_t) {
        using G = typename Node::geometry;
        if constexpr (G::out_subblock_num_tiles <= kMaxDstTiles) {
            run<AccumulatorMode::Dst>(node, dfb_id, dfb_id, false, true);
        } else {
            run_banded(node, dfb_id);
        }
    }

    template <typename Node>
    static void run_banded(const Node& node, uint32_t out_dfb) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using G = typename Node::geometry;
        using Chain = typename Node::chain;
        constexpr uint32_t kTranspose = G::transpose;
        constexpr DstSubblock kSub = dst_subblock(G::rt_dim, G::ct_dim);
        constexpr uint32_t kSubTiles = kSub.tiles();
        constexpr uint32_t kTotalTiles = G::out_subblock_num_tiles;

        unpack_geometry_to(node.in1_dfb, node.in0_dfb);
        ckernel::matmul_block_init(node.in0_dfb, node.in1_dfb, kTranspose, kSub.cols, kSub.rows, G::kt_dim);

        buffer(out_dfb).reserve_back(kTotalTiles);
        for (uint32_t r0 = 0; r0 < G::rt_dim; r0 += kSub.rows) {
            for (uint32_t c0 = 0; c0 < G::ct_dim; c0 += kSub.cols) {
                ckernel::tile_regs_acquire();
                uint32_t in0_index = r0 * G::kt_dim;
                uint32_t in1_index = c0;
                for (uint32_t k = 0; k < G::kt_dim; ++k) {
                    ckernel::matmul_block(
                        node.in0_dfb,
                        node.in1_dfb,
                        in0_index,
                        in1_index,
                        0,
                        kTranspose,
                        kSub.cols,
                        kSub.rows,
                        G::kt_dim);
                    in0_index += 1;
                    in1_index += G::in1_row_stride;
                }
                if (node.addend_dfb != kNoBias) {
                    AddOp::fpu_reuse_init<true>(node.addend_dfb);
                    for (uint32_t t = 0; t < kSubTiles; ++t) {
                        AddOp::fpu_reuse_apply<true>(
                            node.addend_dfb, block_tile_index(r0, c0, t, kSub.cols, G::ct_dim), t);
                    }
                    ckernel::matmul_block_init(node.in0_dfb, node.in1_dfb, kTranspose, kSub.cols, kSub.rows, G::kt_dim);
                }

                if (node.bias_dfb != kNoBias) {
                    AddOp::fpu_reuse_init<true>(node.bias_dfb);
                    for (uint32_t t = 0; t < kSubTiles; ++t) {
                        AddOp::fpu_reuse_apply<true>(node.bias_dfb, c0 + t % kSub.cols, t);
                    }
                    ckernel::matmul_block_init(node.in0_dfb, node.in1_dfb, kTranspose, kSub.cols, kSub.rows, G::kt_dim);
                }

                if constexpr (!Chain::empty) {
                    for (uint32_t t = 0; t < kSubTiles; ++t) {
                        Chain::apply_in_place(t);
                    }
                }
                ckernel::tile_regs_commit();
                ckernel::tile_regs_wait();
                pack_to(out_dfb);
                ckernel::pack_block(0, out_dfb, kSubTiles);
                ckernel::tile_regs_release();
            }
        }
        buffer(out_dfb).push_back(kTotalTiles);
#else
        (void)node;
        (void)out_dfb;
#endif
    }

    template <AccumulatorMode Mode, typename Node, typename EpilogueChain = expr::UnaryChain<>>
    static void run(
        const Node& node,
        uint32_t acc_dfb,
        uint32_t out_dfb,
        bool reload,
        bool finish,
        EpilogueChain = {},
        uint32_t epi_bias_dfb = kNoBias) {
        using G = typename Node::geometry;
        constexpr uint32_t kAccTiles = G::out_subblock_num_tiles;

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using Chain = typename Node::chain;
        constexpr uint32_t kTranspose = G::transpose;
        constexpr DstSubblock kSub = dst_subblock(G::rt_dim, G::ct_dim);
        constexpr uint32_t kSubTiles = kSub.tiles();

        unpack_geometry_to(node.in1_dfb, node.in0_dfb);
        ckernel::matmul_block_init(node.in0_dfb, node.in1_dfb, kTranspose, kSub.cols, kSub.rows, G::kt_dim);

        constexpr bool kBiasFolded = (Mode == AccumulatorMode::Dst);
        ASSERT(epi_bias_dfb == kNoBias || node.bias_dfb == kNoBias);
        const uint32_t bias_dfb = (epi_bias_dfb != kNoBias) ? epi_bias_dfb : node.bias_dfb;
        const bool via_bias = !kBiasFolded && finish && bias_dfb != kNoBias;

        for (uint32_t r0 = 0; r0 < G::rt_dim; r0 += kSub.rows) {
            for (uint32_t c0 = 0; c0 < G::ct_dim; c0 += kSub.cols) {
                ckernel::tile_regs_acquire();

                if constexpr (Mode == AccumulatorMode::Dst) {
                    if (reload) {
                        ckernel::copy_tile_to_dst_init_short_with_dt(node.in1_dfb, acc_dfb);
                        buffer(acc_dfb).wait_front(kSubTiles);
                        ckernel::copy_block(acc_dfb, 0, 0, kSubTiles);
                        buffer(acc_dfb).pop_front(kSubTiles);
                        unpack_geometry_to(node.in1_dfb, node.in0_dfb);
                        ckernel::reconfig_data_format_srca(acc_dfb, node.in1_dfb);
                        ckernel::matmul_block_init(
                            node.in0_dfb, node.in1_dfb, kTranspose, kSub.cols, kSub.rows, G::kt_dim);
                    }
                }

                uint32_t in0_index = r0 * G::kt_dim;
                uint32_t in1_index = c0;
                for (uint32_t k = 0; k < G::kt_dim; ++k) {
                    ckernel::matmul_block(
                        node.in0_dfb,
                        node.in1_dfb,
                        in0_index,
                        in1_index,
                        0,
                        kTranspose,
                        kSub.cols,
                        kSub.rows,
                        G::kt_dim);
                    in0_index += 1;
                    in1_index += G::in1_row_stride;
                }

                if (node.addend_dfb != kNoBias) {
                    AddOp::fpu_reuse_init<true>(node.addend_dfb);
                    for (uint32_t t = 0; t < kSubTiles; ++t) {
                        AddOp::fpu_reuse_apply<true>(
                            node.addend_dfb, block_tile_index(r0, c0, t, kSub.cols, G::ct_dim), t);
                    }
                    ckernel::matmul_block_init(node.in0_dfb, node.in1_dfb, kTranspose, kSub.cols, kSub.rows, G::kt_dim);
                }

                if constexpr (kBiasFolded) {
                    if (finish && bias_dfb != kNoBias) {
                        AddOp::fpu_reuse_init<true>(bias_dfb);
                        for (uint32_t t = 0; t < kSubTiles; ++t) {
                            AddOp::fpu_reuse_apply<true>(bias_dfb, c0 + t % kSub.cols, t);
                        }
                        ckernel::matmul_block_init(
                            node.in0_dfb, node.in1_dfb, kTranspose, kSub.cols, kSub.rows, G::kt_dim);
                    }
                }

                if constexpr (!Chain::empty) {
                    for (uint32_t t = 0; t < kSubTiles; ++t) {
                        Chain::apply_in_place(t);
                    }
                }

                if constexpr (Mode == AccumulatorMode::Dst) {
                    if constexpr (!EpilogueChain::empty) {
                        if (finish && (kBiasFolded || bias_dfb == kNoBias)) {
                            for (uint32_t t = 0; t < kSubTiles; ++t) {
                                EpilogueChain::apply_in_place(t);
                            }
                        }
                    }
                }

                ckernel::tile_regs_commit();

                if constexpr (Mode == AccumulatorMode::Dst) {
                    const uint32_t dest = (finish && !via_bias) ? out_dfb : acc_dfb;
                    pack_to(dest);
                    buffer(dest).reserve_back(kSubTiles);
                    ckernel::tile_regs_wait();
                    ckernel::pack_block(0, dest, kSubTiles);
                    ckernel::tile_regs_release();
                    buffer(dest).push_back(kSubTiles);
                } else {
                    pack_to(acc_dfb);
                    buffer(acc_dfb).reserve_back(kSubTiles);
                    ckernel::tile_regs_wait();
                    ckernel::pack_reconfig_l1_acc(reload ? 1 : 0);
                    ckernel::pack_block(0, acc_dfb, kSubTiles);
                    ckernel::tile_regs_release();
                    buffer(acc_dfb).push_back(kSubTiles);
                    ckernel::pack_reconfig_l1_acc(0);
                }
            }
        }

        if constexpr (Mode == AccumulatorMode::Dst) {
            if (via_bias) {
                bias_finish(node, acc_dfb, out_dfb, bias_dfb, EpilogueChain{});
            }
        } else {
            if (!finish) {
                buffer(acc_dfb).wait_front(kAccTiles);
                buffer(acc_dfb).pop_front(kAccTiles);
            } else if (!kBiasFolded && bias_dfb != kNoBias) {
                bias_finish(node, acc_dfb, out_dfb, bias_dfb, EpilogueChain{});
            } else {
                ckernel::copy_tile_to_dst_init_short_with_dt(node.in1_dfb, acc_dfb);
                pack_to(out_dfb);
                for (uint32_t sb = 0; sb < kAccTiles; sb += kSubTiles) {
                    ckernel::tile_regs_acquire();
                    buffer(acc_dfb).wait_front(kSubTiles);
                    ckernel::copy_block(acc_dfb, 0, 0, kSubTiles);
                    buffer(acc_dfb).pop_front(kSubTiles);

                    if constexpr (!EpilogueChain::empty) {
                        for (uint32_t t = 0; t < kSubTiles; ++t) {
                            EpilogueChain::apply_in_place(t);
                        }
                    }

                    ckernel::tile_regs_commit();
                    buffer(out_dfb).reserve_back(kSubTiles);
                    ckernel::tile_regs_wait();
                    ckernel::pack_block(0, out_dfb, kSubTiles);
                    ckernel::tile_regs_release();
                    buffer(out_dfb).push_back(kSubTiles);
                }

                ckernel::reconfig_data_format_srca(acc_dfb, node.in1_dfb);
                ckernel::matmul_block_init(node.in0_dfb, node.in1_dfb, kTranspose, kSub.cols, kSub.rows, G::kt_dim);
            }
        }
#else
        (void)node;
        (void)acc_dfb;
        (void)out_dfb;
        (void)reload;
        (void)finish;
        (void)epi_bias_dfb;
        (void)kAccTiles;
#endif
    }
};

template <>
struct Strategy<BcastFusion> {
    template <typename Node>
    static void run(const Node& node, uint32_t dfb_id, uint32_t num_tiles) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using Chain = typename Node::chain;
        using Ops = BcastOps<typename Node::op, Node::axis>;

        unpack_geometry_to(node.block_dfb, node.vec_dfb);
        ckernel::reconfig_data_format(node.block_dfb, node.vec_dfb);
        Ops::init(node.block_dfb, node.vec_dfb);

        pack_to(dfb_id);
        buffer(dfb_id).reserve_back(num_tiles);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            ckernel::tile_regs_acquire();
            Ops::apply(node.block_dfb, node.vec_dfb, t, Node::vec_tile(t), 0);
            if constexpr (!Chain::empty) {
                Chain::apply_in_place(0);
            }
            ckernel::tile_regs_commit();
            ckernel::tile_regs_wait();
            pack_one(0, dfb_id);
            ckernel::tile_regs_release();
        }
        buffer(dfb_id).push_back(num_tiles);
#else
        (void)node;
        (void)dfb_id;
        (void)num_tiles;
#endif
    }
};

template <>
struct Strategy<ReduceFusion> {
    template <typename Node>
    static void run(const Node& node, uint32_t dfb_id, uint32_t) {
        using G = typename Node::geometry;
        constexpr ReduceAxis kAxis = Node::axis;
        constexpr uint32_t kOut = G::out_tiles(kAxis);
        constexpr uint32_t kGroup = G::group(kAxis);
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        using Chain = typename Node::chain;
        constexpr ckernel::PoolType kPool = metal_pool(Node::pool);
        constexpr ckernel::ReduceDim kDim = metal_dim(kAxis);

        if constexpr (kDim == ckernel::ReduceDim::REDUCE_ROW && kPool != ckernel::PoolType::MAX) {
            ckernel::reconfig_data_format(node.scaler_dfb, node.in_dfb);
        }
        ckernel::reduce_init<kPool, kDim>(node.in_dfb, node.scaler_dfb, dfb_id);

        pack_to(dfb_id);
        buffer(dfb_id).reserve_back(kOut);
        for (uint32_t o = 0; o < kOut; ++o) {
            ckernel::tile_regs_acquire();
            for (uint32_t g = 0; g < kGroup; ++g) {
                ckernel::reduce_tile<kPool, kDim>(node.in_dfb, node.scaler_dfb, G::contributor(kAxis, o, g), 0, 0);
            }
            if constexpr (!Chain::empty) {
                Chain::apply_in_place(0);
            }
            ckernel::tile_regs_commit();
            ckernel::tile_regs_wait();
            pack_one(0, dfb_id);
            ckernel::tile_regs_release();
        }
        buffer(dfb_id).push_back(kOut);

        ckernel::reduce_uninit(node.in_dfb);
#else
        (void)node;
        (void)dfb_id;
        (void)kOut;
        (void)kGroup;
#endif
    }
};

}  // namespace unified
}  // namespace tt
