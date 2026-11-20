// =====================================================================================================
// Appendix to report 1 (01_ctrt_polymorphism.cpp): definitions of the helpers the report describes.
// =====================================================================================================

#ifndef LLK_PROPOSAL_CTRT_APPENDIX
#define LLK_PROPOSAL_CTRT_APPENDIX

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace llk::ctrt {

// ---- kinds ------------------------------------------------------------------------------------------
template <auto V>
struct Static {
    static_assert(!std::is_integral_v<decltype(V)> || std::is_same_v<decltype(V), std::uint32_t>,
                  "numbers are std::uint32_t: write U32<N>");
    static constexpr auto value = V;
};
template <std::uint32_t N> using U32 = Static<N>;
using Dyn = std::uint32_t;

template <class T> struct is_static : std::false_type {};
template <auto V> struct is_static<Static<V>> : std::true_type {};
template <class T> inline constexpr bool is_static_v = is_static<std::decay_t<T>>::value;
template <class T> inline constexpr bool is_scalar_v = std::is_integral_v<T> || std::is_enum_v<T>;

template <auto V> constexpr auto as_dyn(Static<V>) { return V; }
template <class T, std::enable_if_t<is_scalar_v<T>, int> = 0> constexpr T as_dyn(T v) { return v; }

// ---- operators: Static o Static is Static, anything with a run-time value is run time ----------------
#define CTRT_ARITH(op)                                                                                     \
    template <std::uint32_t A, std::uint32_t B>                                                            \
    constexpr U32<std::uint32_t(A op B)> operator op(Static<A>, Static<B>) { return {}; }                  \
    template <std::uint32_t A> constexpr Dyn operator op(Static<A>, Dyn b) { return A op b; }             \
    template <std::uint32_t B> constexpr Dyn operator op(Dyn a, Static<B>) { return a op B; }
#define CTRT_CMP(op)                                                                                       \
    template <auto A, auto B>                                                                              \
    constexpr std::bool_constant<(A op B)> operator op(Static<A>, Static<B>) { return {}; }                \
    template <auto A> constexpr bool operator op(Static<A>, decltype(A) b) { return A op b; }             \
    template <auto B> constexpr bool operator op(decltype(B) a, Static<B>) { return a op B; }
CTRT_ARITH(+) CTRT_ARITH(-) CTRT_ARITH(*) CTRT_ARITH(/) CTRT_ARITH(%)
CTRT_CMP(<) CTRT_CMP(<=) CTRT_CMP(==)
#undef CTRT_ARITH
#undef CTRT_CMP

template <bool B>
constexpr void check(std::bool_constant<B>) { static_assert(B, "ctrt::check failed at compile time"); }
constexpr void check(bool b) { b ? static_cast<void>(0) : __builtin_trap(); }  // on device: the LLK assert

// ---- strongly typed functions -----------------------------------------------------------------------
template <bool C> using Requires = std::enable_if_t<C, int>;  // template <class T, Requires<is_x_v<T>> = 0>
template <class T> inline constexpr bool is_number_v = std::is_integral_v<T> && !std::is_same_v<T, bool>;
template <std::uint32_t N> inline constexpr bool is_number_v<U32<N>> = true;  // a number of either kind

// ---- DYN and pStruct fields -------------------------------------------------------------------------
template <class T> struct dyn_value;  // the reserved value that means "run time", one per field type
template <> struct dyn_value<std::uint32_t> { static constexpr std::uint32_t value = 0xFFFFFFFFu; };
inline constexpr struct DynTag {
    template <class T> constexpr operator T() const { return dyn_value<T>::value; }
} DYN{};
template <auto V> inline constexpr bool is_dyn_v = V == dyn_value<decltype(V)>::value;

template <class T> struct StaticOrDyn : DynTag {};  // a run-time argument: DYN
template <auto V> struct StaticOrDyn<Static<V>> {   // a Static<V> argument: V, only for a field of V's type
    template <class F, std::enable_if_t<std::is_same_v<F, decltype(V)>, int> = 0>
    constexpr operator F() const { return V; }
};
template <class T> inline constexpr StaticOrDyn<T> static_or_dyn{};

template <class T, auto V> inline constexpr bool is_static_of_v = false;  // T is exactly Static<V>
template <auto V> inline constexpr bool is_static_of_v<Static<V>, V> = true;
template <class T, class F>  // a run-time argument fits a field: same type, or any integer for a number field
inline constexpr bool fits_v = std::is_same_v<T, F> || (std::is_integral_v<F> && std::is_integral_v<T>);

// ---- CtRt<Vs...>: the generic pStruct base; stores only the DYN fields, one word each -----------------
template <std::size_t N> struct DynStorage { std::uint32_t dyn[N]; };
template <> struct DynStorage<0> {};  // all static: an empty base

template <auto... Vs>
struct CtRt : DynStorage<(is_dyn_v<Vs> + ... + 0)> {
    static constexpr std::size_t kDyn = (is_dyn_v<Vs> + ... + 0);
    static constexpr bool kIsDyn[] = {is_dyn_v<Vs>...};
    template <std::size_t I> using field_t = std::tuple_element_t<I, std::tuple<decltype(Vs)...>>;

    // one value per DYN field, in field order
    template <class... Ts, std::enable_if_t<sizeof...(Ts) == kDyn && (is_scalar_v<Ts> && ...), int> = 0>
    constexpr CtRt(Ts... v) : DynStorage<kDyn>{static_cast<std::uint32_t>(v)...} {
        static_assert(fit<Ts...>(std::make_index_sequence<kDyn>{}), "an argument does not match its field's type");
    }
    // every field, each a Static<V> or a run-time value; keeps the DYN ones
    template <class... Ts, std::enable_if_t<sizeof...(Ts) == sizeof...(Vs) && kDyn != sizeof...(Vs) &&
                                                ((is_static_v<Ts> || is_scalar_v<Ts>) && ...), int> = 0>
    constexpr CtRt(Ts... all)
        : CtRt(pick(std::make_index_sequence<kDyn>{}, static_cast<std::uint32_t>(as_dyn(all))...)) {
        static_assert(((is_dyn_v<Vs> || is_static_of_v<Ts, Vs>) && ...), "a static field needs its own Static<V>");
    }

    template <std::size_t I>
    constexpr auto get() const {
        constexpr auto v = std::get<I>(std::make_tuple(Vs...));
        if constexpr (!kIsDyn[I]) return Static<v>{};
        else return static_cast<field_t<I>>(this->dyn[slot(I)]);
    }

private:
    static constexpr std::size_t slot(std::size_t field) {  // storage index of a DYN field
        std::size_t n = 0;
        for (std::size_t j = 0; j < field; ++j) n += kIsDyn[j];
        return n;
    }
    static constexpr std::size_t field_of(std::size_t k) {  // field index of the k-th DYN field
        for (std::size_t i = 0;; ++i) if (kIsDyn[i] && k-- == 0) return i;
    }
    template <class... Ts, std::size_t... K>
    static constexpr bool fit(std::index_sequence<K...>) { return (fits_v<Ts, field_t<field_of(K)>> && ...); }
    template <std::size_t... K, class... Ws>
    static constexpr CtRt pick(std::index_sequence<K...>, Ws... words) {
        const std::uint32_t w[] = {words...};
        static_cast<void>(w);  // unused when every field is static
        return CtRt(static_cast<field_t<field_of(K)>>(w[field_of(K)])...);
    }
};

// ---- stand-ins so the report builds without ckernel -------------------------------------------------
enum class DataFormat : std::uint8_t { Float32 = 0, Float16 = 1, Float16_b = 5, Bfp8_b = 6, Bfp4_b = 7 };  // subset

// Same contract as ckernel_ops.h: TTI_INSN only accepts a constant (on device an "n" asm constraint, here a
// template argument); TT_INSN computes the encoding at run time and stores it (on device to instrn_buffer[0]).
namespace sim {
struct Insn { std::uint32_t encoding; bool immediate; };
inline Insn last{};
template <std::uint32_t Encoding> void tti_insn() { last = {Encoding, true}; }
inline void tt_insn(std::uint32_t encoding) { last = {encoding, false}; }
}  // namespace sim

}  // namespace llk::ctrt

#define TT_OP(opcode, params) ((opcode << 24) + params)
#define TT_OP_MOP(mop_type, loop_count, zmask) \
    TT_OP(0x01, (((mop_type) << 23) + ((loop_count) << 16) + ((zmask) << 0)))
#define TTI_INSN(ENCODING) ::llk::ctrt::sim::tti_insn<unsigned(ENCODING)>()
#define TT_INSN(ENCODING) ::llk::ctrt::sim::tt_insn(unsigned(ENCODING))
#define TTI_MOP(mop_type, loop_count, zmask) TTI_INSN(TT_OP_MOP(mop_type, loop_count, zmask))
#define TT_MOP(mop_type, loop_count, zmask) TT_INSN(TT_OP_MOP(mop_type, loop_count, zmask))

#endif  // LLK_PROPOSAL_CTRT_APPENDIX
