// SPDX-License-Identifier: Apache-2.0
//
// Expression-tree register allocation.
//
// Deliberately domain-free: it knows nothing about circular buffers, the NOC, or
// Tensix. It provides the tree shapes, a compile-time register allocator, and the
// emission walk; tt/unified_math.hpp supplies the policies.
//
// A compute expression is a tree encoded in its own type, e.g.
//     x + y + z   ==>   Bin<AddOp, Bin<AddOp, Leaf, Leaf>, Leaf>
// so there is no tree to *build* -- only to walk.
//
// POLICIES the domain header must satisfy:
//
//   Leaf node L:   static constexpr uint32_t need = 1;            // DST slots
//                  void emit(uint32_t dst, uint32_t tile) const;  // tile -> dst
//   Binary op Op:  static void apply(uint32_t lhs_dst, uint32_t rhs_dst, uint32_t out_dst);
//   Unary  op Op:  static void apply(uint32_t src_dst, uint32_t out_dst);
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

template <typename Op, typename Lhs, typename Rhs>
struct Bin {
    Lhs lhs;
    Rhs rhs;
};

template <typename Op, typename Child>
struct Un {
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
    static void run(const Node& n, uint32_t tile) { n.emit(Base, tile); }
};

template <uint32_t Base, typename Op, typename C>
struct Emit<Base, Un<Op, C>> {
    static constexpr uint32_t result = Base;
    static void run(const Un<Op, C>& n, uint32_t tile) {
        Emit<Base, C>::run(n.child, tile);
        Op::apply(Emit<Base, C>::result, Base);
    }
};

template <uint32_t Base, typename Op, typename L, typename R>
struct Emit<Base, Bin<Op, L, R>> {
    static constexpr uint32_t result = Base;
    static void run(const Bin<Op, L, R>& n, uint32_t tile) {
        Emit<Base, L>::run(n.lhs, tile);      // left result lands in Base
        Emit<Base + 1, R>::run(n.rhs, tile);  // right starts above it
        Op::apply(Base, Base + 1, Base);      // fold in place
    }
};

// -------------------------------------------------------------- driver ---

// DST slots this expression needs.
template <typename Node>
constexpr uint32_t need_v = Need<Node>::value;

// Slot the result ends up in.
template <typename Node>
constexpr uint32_t result_slot_v = Emit<0, Node>::result;

// Emit the whole expression for one tile index.
template <typename Node>
void emit(const Node& node, uint32_t tile) {
    Emit<0, Node>::run(node, tile);
}

}  // namespace expr
}  // namespace unified
}  // namespace tt
