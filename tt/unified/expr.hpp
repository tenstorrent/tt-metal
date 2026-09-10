// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

namespace tt {
namespace unified {
namespace expr {

template <typename N>
auto fluent_relu(const N& n);
template <typename N>
auto fluent_silu(const N& n);
template <typename N>
auto fluent_exp(const N& n);
template <typename N>
auto fluent_recip(const N& n);
template <typename N>
auto fluent_sqrt(const N& n);
template <typename N>
auto fluent_rsqrt(const N& n);

template <typename Self>
struct Fluent {
    auto relu() const { return fluent_relu(self()); }
    auto silu() const { return fluent_silu(self()); }
    auto exp() const { return fluent_exp(self()); }
    auto recip() const { return fluent_recip(self()); }
    auto sqrt() const { return fluent_sqrt(self()); }
    auto rsqrt() const { return fluent_rsqrt(self()); }

private:
    const Self& self() const { return static_cast<const Self&>(*this); }
};

template <typename Op, typename Lhs, typename Rhs>
struct Bin : Fluent<Bin<Op, Lhs, Rhs>> {
    Lhs lhs;
    Rhs rhs;
};

template <typename Op, typename Child>
struct Un : Fluent<Un<Op, Child>> {
    Child child;
};

template <typename T, typename = void>
struct is_expr : std::false_type {};

template <typename T>
struct is_expr<T, std::void_t<typename T::is_expr_node>> : std::true_type {};

template <typename Op, typename L, typename R>
struct is_expr<Bin<Op, L, R>> : std::true_type {};

template <typename Op, typename C>
struct is_expr<Un<Op, C>> : std::true_type {};

struct FpuTreeKind {};

template <typename Node>
struct IsLeaf : std::true_type {};
template <typename Op, typename C>
struct IsLeaf<Un<Op, C>> : std::false_type {};
template <typename Op, typename L, typename R>
struct IsLeaf<Bin<Op, L, R>> : std::false_type {};

template <typename Node>
inline constexpr bool is_leaf_v = IsLeaf<Node>::value;

template <typename Node>
struct FpuFusable : std::false_type {};

template <typename Op, typename L, typename R>
struct FpuFusable<Bin<Op, L, R>> {
    static constexpr bool value =
        Op::fpu_capable && ((is_leaf_v<L> && is_leaf_v<R>) || (!is_leaf_v<L> && is_leaf_v<R> && FpuFusable<L>::value) ||
                            (is_leaf_v<L> && !is_leaf_v<R> && FpuFusable<R>::value));
};

template <typename Op, typename C>
struct FpuFusable<Un<Op, C>> {
    static constexpr bool value = !is_leaf_v<C> && FpuFusable<C>::value;
};

#if defined(TT_UNIFIED_NO_FPU_ELTWISE)
template <typename Node>
inline constexpr bool fpu_fusable_v = false;
#else
template <typename Node>
inline constexpr bool fpu_fusable_v = FpuFusable<Node>::value;
#endif

template <typename Node>
struct FpuStages;

template <typename Op, typename L, typename R>
struct FpuStages<Bin<Op, L, R>> {
    static void run(const Bin<Op, L, R>& n, uint32_t base_tile, uint32_t count) {
        if constexpr (is_leaf_v<L> && is_leaf_v<R>) {
            Op::fpu_seed_init(n.lhs.source_dfb(), n.rhs.source_dfb());
            for (uint32_t k = 0; k < count; ++k) {
                Op::fpu_seed_apply(n.lhs.source_dfb(), n.rhs.source_dfb(), base_tile + k, base_tile + k, k);
            }
        } else if constexpr (is_leaf_v<R>) {
            FpuStages<L>::run(n.lhs, base_tile, count);
            Op::template fpu_reuse_init<true>(n.rhs.source_dfb());
            for (uint32_t k = 0; k < count; ++k) {
                Op::template fpu_reuse_apply<true>(n.rhs.source_dfb(), base_tile + k, k);
            }
        } else {
            FpuStages<R>::run(n.rhs, base_tile, count);
            Op::template fpu_reuse_init<false>(n.lhs.source_dfb());
            for (uint32_t k = 0; k < count; ++k) {
                Op::template fpu_reuse_apply<false>(n.lhs.source_dfb(), base_tile + k, k);
            }
        }
    }
};

template <typename Op, typename C>
struct FpuStages<Un<Op, C>> {
    static void run(const Un<Op, C>& n, uint32_t base_tile, uint32_t count) {
        FpuStages<C>::run(n.child, base_tile, count);
        for (uint32_t k = 0; k < count; ++k) {
            Op::apply_in_place(k);
        }
    }
};

template <typename Node>
void fpu_stages(const Node& node, uint32_t base_tile, uint32_t count) {
    FpuStages<Node>::run(node, base_tile, count);
}

struct TreeKind {};

template <typename Node, typename = void>
struct kind_of {
    using type = TreeKind;
};

template <typename Node>
struct kind_of<Node, std::void_t<typename Node::fusion_kind>> {
    using type = typename Node::fusion_kind;
};

template <typename Op, typename L, typename R>
struct kind_of<Bin<Op, L, R>, void> {
    using type = std::conditional_t<fpu_fusable_v<Bin<Op, L, R>>, FpuTreeKind, TreeKind>;
};

template <typename Op, typename C>
struct kind_of<Un<Op, C>, void> {
    using type = std::conditional_t<fpu_fusable_v<Un<Op, C>>, FpuTreeKind, TreeKind>;
};

template <typename Node>
using kind_of_t = typename kind_of<Node>::type;

template <typename... Ops>
struct UnaryChain {
    static constexpr bool empty = (sizeof...(Ops) == 0);
    static constexpr uint32_t size = sizeof...(Ops);

    static void apply_in_place(uint32_t slot) { (Ops::apply_in_place(slot), ...); }
};

template <typename Chain, typename Op>
struct chain_append;

template <typename... Ops, typename Op>
struct chain_append<UnaryChain<Ops...>, Op> {
    using type = UnaryChain<Ops..., Op>;
};

template <typename Chain, typename Op>
using chain_append_t = typename chain_append<Chain, Op>::type;

template <typename Node>
struct LeafCount {
    static constexpr uint32_t value = 1;
};

template <typename Op, typename C>
struct LeafCount<Un<Op, C>> {
    static constexpr uint32_t value = LeafCount<C>::value;
};

template <typename Op, typename L, typename R>
struct LeafCount<Bin<Op, L, R>> {
    static constexpr uint32_t value = LeafCount<L>::value + LeafCount<R>::value;
};

template <typename Node>
constexpr uint32_t leaf_count_v = LeafCount<Node>::value;

template <typename Node>
struct Need {
    static constexpr uint32_t value = Node::need;
};

template <typename Op, typename C>
struct Need<Un<Op, C>> {
    static constexpr uint32_t value = Need<C>::value;
};

template <typename Op, typename L, typename R>
struct Need<Bin<Op, L, R>> {
    static constexpr uint32_t left = Need<L>::value;
    static constexpr uint32_t right = 1 + Need<R>::value;
    static constexpr uint32_t value = left > right ? left : right;
};

template <uint32_t Base, typename Node>
struct Emit {
    static constexpr uint32_t result = Base;
    static void run(const Node& n, uint32_t tile, bool reconfigure) { n.emit(Base, tile, reconfigure); }
};

template <uint32_t Base, typename Op, typename C>
struct Emit<Base, Un<Op, C>> {
    static constexpr uint32_t result = Base;
    static void run(const Un<Op, C>& n, uint32_t tile, bool reconfigure) {
        Emit<Base, C>::run(n.child, tile, reconfigure);
        Op::apply(Emit<Base, C>::result, Base);
    }
};

template <uint32_t Base, typename Op, typename L, typename R>
struct Emit<Base, Bin<Op, L, R>> {
    static constexpr uint32_t result = Base;
    static void run(const Bin<Op, L, R>& n, uint32_t tile, bool reconfigure) {
        Emit<Base, L>::run(n.lhs, tile, reconfigure);
        Emit<Base + 1, R>::run(n.rhs, tile, reconfigure);
        Op::apply(Base, Base + 1, Base);
    }
};

template <uint32_t Stride, uint32_t J, typename Node>
struct LoadLeaves {
    static void run(const Node& n, uint32_t base_tile, uint32_t count) {
        for (uint32_t k = 0; k < count; ++k) {
            n.emit(k * Stride + J, base_tile + k, k == 0);
        }
    }
};

template <uint32_t Stride, uint32_t J, typename Op, typename C>
struct LoadLeaves<Stride, J, Un<Op, C>> {
    static void run(const Un<Op, C>& n, uint32_t base_tile, uint32_t count) {
        LoadLeaves<Stride, J, C>::run(n.child, base_tile, count);
    }
};

template <uint32_t Stride, uint32_t J, typename Op, typename L, typename R>
struct LoadLeaves<Stride, J, Bin<Op, L, R>> {
    static void run(const Bin<Op, L, R>& n, uint32_t base_tile, uint32_t count) {
        LoadLeaves<Stride, J, L>::run(n.lhs, base_tile, count);
        LoadLeaves<Stride, J + LeafCount<L>::value, R>::run(n.rhs, base_tile, count);
    }
};

template <uint32_t J, typename Node>
struct ApplyOps {
    static constexpr uint32_t result_ofs = J;
    static void run(const Node&, uint32_t) {}
};

template <uint32_t J, typename Op, typename C>
struct ApplyOps<J, Un<Op, C>> {
    static constexpr uint32_t result_ofs = ApplyOps<J, C>::result_ofs;
    static void run(const Un<Op, C>& n, uint32_t base) {
        ApplyOps<J, C>::run(n.child, base);
        Op::apply(base + result_ofs, base + result_ofs);
    }
};

template <uint32_t J, typename Op, typename L, typename R>
struct ApplyOps<J, Bin<Op, L, R>> {
    static constexpr uint32_t kRightJ = J + LeafCount<L>::value;
    static constexpr uint32_t result_ofs = ApplyOps<J, L>::result_ofs;
    static void run(const Bin<Op, L, R>& n, uint32_t base) {
        ApplyOps<J, L>::run(n.lhs, base);
        ApplyOps<kRightJ, R>::run(n.rhs, base);
        Op::apply(base + result_ofs, base + ApplyOps<kRightJ, R>::result_ofs, base + result_ofs);
    }
};

template <typename Node>
constexpr uint32_t need_v = Need<Node>::value;

template <typename Node>
constexpr uint32_t result_slot_v = Emit<0, Node>::result;

template <typename Node>
void emit(const Node& node, uint32_t tile, bool reconfigure) {
    Emit<0, Node>::run(node, tile, reconfigure);
}

template <typename Node>
constexpr uint32_t leaf_slots_v = LeafCount<Node>::value;

template <typename Node>
constexpr uint32_t leaf_result_ofs_v = ApplyOps<0, Node>::result_ofs;

template <typename Node>
void load_leaves(const Node& node, uint32_t base_tile, uint32_t count) {
    LoadLeaves<leaf_slots_v<Node>, 0, Node>::run(node, base_tile, count);
}

template <typename Node>
void apply_ops(const Node& node, uint32_t base) {
    ApplyOps<0, Node>::run(node, base);
}

}  // namespace expr
}  // namespace unified
}  // namespace tt
