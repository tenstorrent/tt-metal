// SPDX-License-Identifier: Apache-2.0
//
// Expression-tree register allocation.
//
// Deliberately op-agnostic: it knows nothing about circular buffers, the NOC, or
// Tensix, and does not name a single op. It provides the tree shapes, a
// compile-time register allocator, the emission walk, and the method spelling of
// the ops; tt/unified/math.hpp supplies the policies.
//
// A compute expression is a tree encoded in its own type, e.g.
//     x + y + z   ==>   Bin<AddOp, Bin<AddOp, Leaf, Leaf>, Leaf>
// so there is no tree to *build* -- only to walk.
//
// POLICIES the domain header must satisfy:
//
//   Leaf node L:   static constexpr uint32_t need = 1;            // DST slots
//                  void emit(uint32_t dst, uint32_t tile, bool reconfigure) const;
//                    tile -> dst. `reconfigure` asks the leaf to re-point the hardware
//                    at its own source before copying; the driver decides when that is
//                    necessary, since only it knows whether another leaf has intervened.
//   Binary op Op:  static void apply(uint32_t lhs_dst, uint32_t rhs_dst, uint32_t out_dst);
//   Unary  op Op:  static void apply(uint32_t src_dst, uint32_t out_dst);
//   Method hooks:  fluent_<op>(node), one per op the mixin spells as a method
//                  -- see Fluent below.
//
// ALLOCATION is Sethi-Ullman numbering. A binary node evaluates its left child
// into `base`; once that finishes only the child's *result* is live, so the right
// child may start at `base + 1` and reuse everything above it. The op then folds
// in place back into `base`:
//
//     need(leaf)   = L::need
//     need(unary)  = need(child)
//     need(binary) = max(need(L), 1 + need(R))
//
// The consequence worth knowing: a left-associated chain of any length costs two
// slots, because each intermediate is consumed immediately.
//
//     x + y + z + w   ->  2 slots
//     x + (y + (z+w)) ->  4 slots
//
// (Evaluating the heavier child first would flatten the second case too, but that
// requires commutativity, so it is left out.)
//
// Every slot number is a template parameter, so the emitted code contains only
// compile-time constants -- no base-offset arithmetic at run time.

#pragma once

#include <cstdint>
#include <type_traits>

namespace tt {
namespace unified {
namespace expr {

// ---------------------------------------------------------------- shapes ---

// Method syntax for the ops, so `relu(x)` and `x.relu()` are the same call: the
// method delegates, so the two spellings cannot drift and per-kind behaviour comes
// along unasked.
//
// Which ops exist is domain knowledge, so this layer does not name them. It calls
// a hook declared here and defined by the domain (tt/unified/math.hpp). That
// indirection is not decoration -- it is what frees the mixin from the domain's
// declaration ORDER. The hook's body is looked up where a method is used, by which
// time every overload is in; a direct qualified call would instead bind at the
// point Fluent is defined, which would force it below the last op declaration,
// halfway down the public header, and would keep any concrete node defined before
// that point from carrying it at all.
template <typename N>
auto fluent_relu(const N& n);
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
    auto exp() const { return fluent_exp(self()); }
    auto recip() const { return fluent_recip(self()); }
    auto sqrt() const { return fluent_sqrt(self()); }
    auto rsqrt() const { return fluent_rsqrt(self()); }

private:
    const Self& self() const { return static_cast<const Self&>(*this); }
};

// Aggregates with a base, so brace-init leads with {} for it.
template <typename Op, typename Lhs, typename Rhs>
struct Bin : Fluent<Bin<Op, Lhs, Rhs>> {
    Lhs lhs;
    Rhs rhs;
};

template <typename Op, typename Child>
struct Un : Fluent<Un<Op, Child>> {
    Child child;
};

// Tag for "this type participates in compute expressions", to keep the operator
// overloads from swallowing unrelated types.
template <typename T, typename = void>
struct is_expr : std::false_type {};

template <typename T>
struct is_expr<T, std::void_t<typename T::is_expr_node>> : std::true_type {};

template <typename Op, typename L, typename R>
struct is_expr<Bin<Op, L, R>> : std::true_type {};

template <typename Op, typename C>
struct is_expr<Un<Op, C>> : std::true_type {};

// ------------------------------------------------------------- kinds ------
//
// A fusion "kind" selects the driver strategy: how the enclosing loop is shaped
// and who owns the DST register file. Trees default to TreeKind (freely
// allocatable DST); a node may override by declaring `fusion_kind`.

struct TreeKind {};

template <typename Node, typename = void>
struct kind_of {
    using type = TreeKind;
};

template <typename Node>
struct kind_of<Node, std::void_t<typename Node::fusion_kind>> {
    using type = typename Node::fusion_kind;
};

template <typename Node>
using kind_of_t = typename kind_of<Node>::type;

// ------------------------------------------------------- unary chains -----
//
// An ordered list of unary ops applied *in place* to slots someone else owns.
// Used by fusion kinds whose hardware unit consumes the whole register file, so
// there is nothing to allocate and nothing to copy: the ops rewrite the
// accumulator where it already sits. Ops must provide `apply_in_place(slot)`.

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

// ------------------------------------------------------------ leaf count ---
//
// How many leaves the tree emits. Structural, like Need, and used for the same kind of
// reason: with a single leaf the hardware stays pointed at one source for the whole
// loop, so it only has to be pointed there once.

template <typename Node>
struct LeafCount {
    static constexpr uint32_t value = 1;  // leaf
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

// ------------------------------------------------------ register demand ---

template <typename Node>
struct Need {
    static constexpr uint32_t value = Node::need;  // leaf
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

// ------------------------------------------------------------ emission ---
//
// Emit<Base, Node>::result is the slot holding the node's value;
// Emit<Base, Node>::run(node, tile) emits the ops for one tile.

template <uint32_t Base, typename Node>
struct Emit {  // leaf
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
        Emit<Base, L>::run(n.lhs, tile, reconfigure);      // left result lands in Base
        Emit<Base + 1, R>::run(n.rhs, tile, reconfigure);  // right starts above it
        Op::apply(Base, Base + 1, Base);                   // fold in place
    }
};

// ------------------------------------------------- leaf-outer emission ---
//
// The walk above interleaves loads and ops, which means the unpacker is re-pointed at
// a different circular buffer between every pair of leaves -- once per leaf per TILE.
// Measured on flash, that reconfiguration is 9.17us of a 51.6us kernel, and the
// expensive half of it (reprogramming the unpacker MOP) has no conditional form to
// hide behind. See unified_llama_prefill.md.
//
// So there is a second emission order: hoist the tile loop INSIDE the leaf walk. Load
// every tile of leaf 0, then every tile of leaf 1, and only then apply the ops. The
// reconfiguration count drops from leaves*tiles to leaves per group.
//
// The price is slots. `Emit` REUSES them -- a Bin folds into its left operand, so a
// leaf's slot is overwritten by an op before a later leaf is read -- which is why this
// order cannot share its allocation. Here every leaf gets its own slot, so a group of
// G tiles needs G*leaf_count of them, against need_v for the interleaved walk.
//
// The allocation that makes the op phase work is: leaf j of tile k lives at
// k*leaf_count + j, and a subtree folds into the slot of its LEFTMOST leaf. Subtrees
// own disjoint leaf ranges, so folding left can never land on a slot another subtree
// still needs. `J` threads that leftmost-leaf index down the walk.

// Loads every tile of every leaf, leaf-outer, with ONE reconfigure per leaf.
template <uint32_t Stride, uint32_t J, typename Node>
struct LoadLeaves {  // leaf
    static void run(const Node& n, uint32_t base_tile, uint32_t count) {
        for (uint32_t k = 0; k < count; ++k) {
            // Only the first tile of this leaf re-points the unpacker; the rest are
            // already pointed at the right buffer. This is the whole point.
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

// Applies the ops for ONE tile whose leaves are already resident. `base` is the tile's
// slot base and is a runtime value, which it can be because slots are runtime
// arguments to emit and apply -- only the leaf OFFSETS have to be compile-time.
template <uint32_t J, typename Node>
struct ApplyOps {  // leaf: already loaded, nothing to do
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

// -------------------------------------------------------------- driver ---

// DST slots this expression needs.
template <typename Node>
constexpr uint32_t need_v = Need<Node>::value;

// Slot the result ends up in.
template <typename Node>
constexpr uint32_t result_slot_v = Emit<0, Node>::result;

// Emit the whole expression for one tile index. `reconfigure` is passed down to every
// leaf; see the leaf policy above and Strategy<SFPUFusion> for who decides it.
template <typename Node>
void emit(const Node& node, uint32_t tile, bool reconfigure) {
    Emit<0, Node>::run(node, tile, reconfigure);
}

// Slots one tile of the leaf-outer layout occupies: one per leaf.
template <typename Node>
constexpr uint32_t leaf_slots_v = LeafCount<Node>::value;

// Where a tile's result lands, relative to its slot base. The whole tree's leftmost
// leaf is index 0, so this is 0 -- named rather than assumed.
template <typename Node>
constexpr uint32_t leaf_result_ofs_v = ApplyOps<0, Node>::result_ofs;

// Load `count` tiles from `base_tile` on, leaf-outer. See LoadLeaves.
template <typename Node>
void load_leaves(const Node& node, uint32_t base_tile, uint32_t count) {
    LoadLeaves<leaf_slots_v<Node>, 0, Node>::run(node, base_tile, count);
}

// Apply the ops for the tile whose slots start at `base`.
template <typename Node>
void apply_ops(const Node& node, uint32_t base) {
    ApplyOps<0, Node>::run(node, base);
}

}  // namespace expr
}  // namespace unified
}  // namespace tt
