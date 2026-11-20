// =====================================================================================================
// LLK tensor proposals, report 1: compile-time / run-time (CT/RT) polymorphism
// =====================================================================================================
// Plain C++17: no class-type NTTPs, no -ftt-* extensions. The helpers are defined in
// 01_ctrt_polymorphism_appendix.cpp, which this file includes. Builds with the host and the device compiler:
//   g++ | riscv-tt-elf-g++  -std=c++17 -pedantic -Wall -Wextra -Werror -c 01_ctrt_polymorphism.cpp
//   g++ -std=c++17 -DSELF_TEST 01_ctrt_polymorphism.cpp -o r1 && ./r1

#ifndef LLK_PROPOSAL_CTRT
#define LLK_PROPOSAL_CTRT

#include "00_ctrt_polymorphism_appendix.cpp"

namespace llk::ctrt {

// ==== 1. What this report is about ===================================================================
// Kernel quantities such as tile counts, extents and data formats are sometimes known when a kernel is compiled
// and sometimes only when it runs: weights have fixed shapes, activations often have a run-time row count.
// blaze's rt_ct_arg<F>() already picks CT or RT per argument, but it returns a std::uint32_t either way, so whether
// anything downstream folds depends on the optimizer. Code that needs a constant has to choose: TT_MOP everywhere
// (a store to the instruction buffer even when the count is a constant), or TTI_MOP on a function parameter, which
// compiles only when optimization has folded the parameter to a constant (at -O0 it fails, even with always_inline).
// This report proposes writing each LLK once and getting:
//   * every compile-time value used as a constant, guaranteed by the type system rather than by the optimizer;
//   * run-time values handled by the same source;
//   * code that needs a constant (TTI_ immediates, MOP counts, config values, static_assert, template arguments)
//     choosing its implementation from what is known.
// Mechanism: whether a value is known at compile time, its *kind*, is part of its type. A polymorphic struct
// (pStruct) declares per field whether it is static or run time. Functions take pStructs as deduced template
// parameters, constrained to the struct they expect, and return results whose types carry their kinds, so kinds
// survive any chain of calls.

// ==== 2. Helper concepts (defined in the appendix) ===================================================
//   Static<V>         a value known at compile time. An empty type: the value is in the type and nothing is
//                     stored. V is a number or an enum. Numbers are always std::uint32_t, written U32<N>:
//                     Static<4> (an int) does not compile, and neither does Static<4u> on riscv-tt-elf, where
//                     std::uint32_t is unsigned long.
//   run-time value    a plain std::uint32_t (alias Dyn) or an enum value.
//   operators         + - * / % of two Statics give a Static; anything with a run-time value gives a run-time
//                     value. < <= == follow the same rule and give std::bool_constant or bool. Enums can be
//                     compared but have no arithmetic.
//   is_static_v<T>    the kind as a constant, for `if constexpr`.
//   as_dyn(v)         the value as a run-time value. It is the only way to drop a kind: a Static does not convert
//                     to std::uint32_t or to its enum.
//   check(c)          a static_assert when c is a std::bool_constant, a run-time assert when it is a bool.
//   Requires<C>       constrains a function template to the types for which C holds (section 5), e.g.
//                     template <class B, Requires<is_block_v<B>> = 0> constexpr auto num_tiles(const B& b).
//   is_number_v<T>    true for U32<N> and for integers: a number of either kind, for value parameters.
//   DYN               marks a run-time field in a pStruct's template arguments. It converts to any field type
//                     through that type's reserved value, dyn_value<T>: 0xFFFFFFFF for numbers, and one unused
//                     value per enum, declared next to the enum.
//   CtRt<Vs...>       the generic pStruct base. It stores the DYN fields and provides the constructors and get<I>(),
//                     which returns Static<V> for a static field and the stored value for a DYN one.
//                     Storage, two options (select one): per-field empty bases (every pStruct written out by hand),
//                     or one array of only the DYN fields in the generic base (reused by every pStruct). The
//                     appendix uses the array.
//   static_or_dyn<T>  for deduction guides: the template argument for a constructor argument of type T, which is V
//                     for Static<V> and DYN for a run-time value.
//   TTI_MOP, TT_MOP   host stand-ins with the device contract of ckernel_ops.h: TTI_ only compiles for a constant,
//                     TT_ encodes at run time and stores the instruction to the instruction buffer.

// ==== 3. Defining a pStruct ==========================================================================
// A pStruct is a class template whose template parameters are its fields: a value makes a field static, DYN makes
// it run time. It derives from CtRt<fields...> and inherits its constructors, names the fields with accessors, and
// adds one deduction guide so that P(a, b, c) deduces the kinds. A field can be a number or any enum that has a
// reserved value. Example: a block of tiles in one data format, with an enum field and two number fields.
template <> struct dyn_value<DataFormat> {  // once per enum, next to it; 0xFE is unused in tensix_types.h
    static constexpr auto value = static_cast<DataFormat>(0xFE);
};

template <DataFormat Format, std::uint32_t Rows, std::uint32_t Cols>
struct Block : CtRt<Format, Rows, Cols> {
    using CtRt<Format, Rows, Cols>::CtRt;
    constexpr auto format() const { return this->template get<0>(); }
    constexpr auto rows() const { return this->template get<1>(); }
    constexpr auto cols() const { return this->template get<2>(); }
};
template <class F, class R, class C>
Block(F, R, C) -> Block<static_or_dyn<F>, static_or_dyn<R>, static_or_dyn<C>>;

// ==== 4. Constructing pStructs =======================================================================
//   P<a, b, c>{}        all static: the values are template arguments, nothing is stored
//   P(a, b, c)          all run time: the deduction guide makes every field DYN
//   P<a, DYN, c>(b)     mixed: one constructor argument per DYN field, in field order
//   P(a, U32<8>{}, c)   deduced per argument: a Static<V> makes its field static, a run-time value makes it DYN
// Only DYN fields are stored, one word each. In a kernel the run-time values come from kernel arguments, e.g.
// `const Block<DataFormat::Float16_b, DYN, 8> act(get_arg_val<uint32_t>(0));`. The examples here are constexpr
// only so that the static_asserts can evaluate them.
constexpr Block<DataFormat::Bfp8_b, 4, 8> weights{};                         // all static
constexpr Block dynamic(DataFormat::Float16_b, 6u, 8u);                      // all run time
constexpr Block<DataFormat::Float16_b, DYN, 8> act(6u);                      // mixed: only the row count is run time
constexpr Block act_deduced(Static<DataFormat::Float16_b>{}, 6u, U32<8>{});  // deduced: the same type as act

static_assert(std::is_same_v<decltype(dynamic), const Block<DYN, DYN, DYN>> &&
              std::is_same_v<decltype(act_deduced), decltype(act)>);
static_assert(sizeof(weights) == 1 && sizeof(act) == 4 && sizeof(dynamic) == 12, "only DYN fields are stored");
static_assert(std::is_same_v<decltype(act.rows()), Dyn> && std::is_same_v<decltype(act.cols()), U32<8>> &&
              std::is_same_v<decltype(dynamic.format()), DataFormat> && act.rows() == 6);

// ==== 5. Strongly typed functions ====================================================================
// Accessors alone are not a type: report 1.5's TileShape also has rows() and cols(), in datums rather than tiles,
// so a duck-typed num_tiles would accept it and count datums as tiles. Each pStruct has a trait that holds for all
// of its specializations, whatever their kinds. A function requires the trait of every pStruct it takes, and
// is_number_v of every value it takes. Requires<C> drops the function from overload resolution unless C holds:
// a wrong argument fails at the call, and functions on different structs can share a name. The constraint names
// the struct, not its kinds, so functions stay polymorphic and passing a pStruct on keeps its kinds. (Writing
// the parameter as Block<F, R, C> also works, but ties every function to the struct's parameter list; in C++20
// the trait becomes a concept.)
template <class T> inline constexpr bool is_block_v = false;  // true for every Block, whatever its kinds
template <DataFormat F, std::uint32_t R, std::uint32_t C> inline constexpr bool is_block_v<Block<F, R, C>> = true;

// ==== 6. A function that works on a pStruct ==========================================================
// An ordinary function template with the pStruct as a deduced, constrained parameter. One body serves every
// combination of kinds, and the kind of the result follows from the kinds of the inputs. A function can return a
// value or a new pStruct; a new pStruct is built through the deduction guide, so its kinds are deduced too.
template <class B, Requires<is_block_v<B>> = 0>
constexpr auto num_tiles(const B& b) { return b.rows() * b.cols(); }

template <class B, class N, Requires<is_block_v<B> && is_number_v<N>> = 0>
constexpr auto first_rows(const B& b, N n) {
    check(n <= b.rows());  // a static_assert when both are static, a run-time assert otherwise
    return Block(b.format(), n, b.cols());
}

static_assert(std::is_same_v<decltype(num_tiles(weights)), U32<32>>, "static * static is static");
static_assert(std::is_same_v<decltype(num_tiles(act)), Dyn> && num_tiles(act) == 48, "run time * static is run time");
static_assert(std::is_same_v<decltype(first_rows(act, U32<2>{})), Block<DataFormat::Float16_b, 2, 8>>,
              "the format and column count stay static, and the new row count is static");
// first_rows(weights, U32<5>{}) does not compile: the block has 4 rows.

// ==== 7. A function that passes a pStruct on =========================================================
// A function that receives a pStruct can call another function with it, or do some work on it first and pass on
// the result. The callee sees exactly the kinds the caller had, or the kinds the work produced. Three rules make
// this hold without relying on the optimizer:
//   1. take pStructs as deduced, constrained template parameters (section 5), never as a concrete type that would
//      fix the kinds;
//   2. keep derived values in `auto` or pass them straight on. The compiler enforces this one: a Static does not
//      convert to std::uint32_t or to its enum, so dropping a kind takes an explicit as_dyn();
//   3. work that changes a pStruct returns a new pStruct, so the next call is instantiated for the new kinds.
// Static information can appear on the way (the first two rows of a run-time-height block have a static height)
// or disappear (a row count chosen at run time makes the height run time). Each combination of kinds is its own
// instantiation; as_dyn() trades constant folding for code size where that matters.
template <class B, Requires<is_block_v<B>> = 0>
constexpr auto head_tiles(const B& b) { return num_tiles(first_rows(b, U32<2>{})); }

template <class B, Requires<is_block_v<B>> = 0>
constexpr auto tiles_in_rows(const B& b, Dyn rows) { return num_tiles(first_rows(b, rows)); }

static_assert(std::is_same_v<decltype(head_tiles(act)), U32<16>>, "static again: 2 static rows of 8 columns");
static_assert(std::is_same_v<decltype(tiles_in_rows(weights, 2u)), Dyn> && tiles_in_rows(weights, 2u) == 16);

// ==== 8. A function with CT/RT dependent side effects: TTI_ vs TT_ ===================================
// A function that needs a constant (an instruction immediate, a MOP count baked at init, a config value, a
// static_assert, a template argument) branches on the kind with `if constexpr`. A static MOP count is encoded
// into the instruction when the kernel is compiled (TTI_MOP: the instruction word is an immediate, no run-time
// work). A run-time count is encoded when the kernel runs and stored to the instruction buffer (TT_MOP). The
// branch is chosen by the count's type, so it does not depend on the optimizer: at -O0 a static count still
// gives TTI_MOP, and TTI_MOP never sees a value that is only constant after optimization.
template <class Count, Requires<is_number_v<Count>> = 0>
inline void run_mop([[maybe_unused]] Count count) {
    if constexpr (is_static_v<Count>) {
        static_assert(Count::value >= 1 && Count::value <= 128, "the MOP loop count field is 7 bits");
        TTI_MOP(0, Count::value - 1, 0);
    } else {
        TT_MOP(0, count - 1, 0);
    }
}

// Callers keep the choice through any chain: one body, and the block's kinds pick the instruction.
template <class B, Requires<is_block_v<B>> = 0>
inline void unpack_tiles(const B& b) { run_mop(num_tiles(b)); }

}  // namespace llk::ctrt

#if defined(SELF_TEST)
#include <cstdio>

int main() {  // both calls issue the same MOP instruction, one encoded at compile time and one at run time
    using namespace llk::ctrt;
    volatile std::uint32_t rows = 4;  // genuinely run time
    const Block<DataFormat::Bfp8_b, DYN, 8> act_rt(rows);
    unpack_tiles(weights);  // 4 x 8 tiles, all static
    const sim::Insn from_static = sim::last;
    unpack_tiles(act_rt);   // 4 x 8 tiles, rows at run time
    const sim::Insn from_run_time = sim::last;
    const bool ok = from_static.immediate && !from_run_time.immediate &&
                    from_static.encoding == from_run_time.encoding && head_tiles(act_rt) == U32<16>{};
    std::puts(ok ? "report 1 self-test: OK" : "report 1 self-test: FAILED");
    return ok ? 0 : 1;
}
#endif

#endif  // LLK_PROPOSAL_CTRT
