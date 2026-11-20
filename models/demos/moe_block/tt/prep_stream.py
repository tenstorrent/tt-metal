# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""prep_stream - the fused head's dispatch prep (combine_rf/dispatch_prep.py's two outputs, made inside the head's
program): the route word of every owned token (its destination chips - every chip holding one of its K experts, the
own chip excluded - the tree order and the link; sparse_all_gather_2d's format) computed on the router cores from the
joint rows already in their L1, and the owned rows written row-major into this chip's slots of the dispatch buffer."""

from ttnn.bringup.combine_rf.sparse_all_gather_2d import route_words, span_subs

# dispatch_prep's _ROUTE per token, as a function of the token's ids (sparse_all_gather_2d's _COMMON span helpers)
_ROUTE_FN = r"""
// (dispatch prep's route words: combine_rf/dispatch_prep.py's _ROUTE, one token at a time)
namespace rw_ {
constexpr uint32_t R = __R__, C = __C__, ME = __ME__, K = __K__, RW = __RW__;
constexpr bool WRAP_X = __WRAPX__, WRAP_Y = __WRAPY__;
constexpr uint8_t XRING[WRAP_X ? (1u << C) * 2 : 1] = {__XRING__};
constexpr uint8_t YRING[WRAP_Y ? (1u << R) * 2 : 1] = {__YRING__};
constexpr uint8_t CHIP_OF[__E__] = {__CHIPS__};
inline uint32_t tie_of(uint32_t t) { return ((t * 2654435761u) >> 30) & 1; }
inline uint32_t hi_bit(uint32_t m) { uint32_t b = 0; while (m >>= 1) ++b; return b; }
inline uint32_t lo_bit(uint32_t m) { uint32_t b = 0; while (!(m & 1)) m >>= 1, ++b; return b; }
template <uint32_t n, bool wrap>
inline uint32_t span_of(const uint8_t* ring, uint32_t pos, uint32_t m, uint32_t t) {
    m &= ~(1u << pos) & ((1u << n) - 1);
    if (m == 0) return 0;
    if constexpr (wrap) {
        return ring[(((m >> pos) | (m << (n - pos))) & ((1u << n) - 1)) * 2 + tie_of(t)];
    } else {
        const uint32_t hi = hi_bit(m), lo = lo_bit(m);
        return (hi > pos ? hi - pos : 0) | ((lo < pos ? pos - lo : 0) << 4);
    }
}
inline uint32_t xspan(uint32_t pos, uint32_t cm, uint32_t t) { return span_of<C, WRAP_X>(XRING, pos, cm, t); }
inline uint32_t yspan(uint32_t pos, uint32_t rm, uint32_t t) { return span_of<R, WRAP_Y>(YRING, pos, rm, t); }
inline uint32_t links(uint32_t s) { return (s & 15) + (s >> 4); }
inline uint32_t cols_of(uint32_t m) {
    uint32_t cm = 0;
    for (uint32_t r = 0; r < R; ++r) cm |= (m >> (r * C)) & ((1u << C) - 1);
    return cm;
}
inline uint32_t rows_of(uint32_t m) {
    uint32_t rm = 0;
    for (uint32_t r = 0; r < R; ++r) rm |= (((m >> (r * C)) & ((1u << C) - 1)) != 0) << r;
    return rm;
}
inline uint32_t rows_in_col(uint32_t m, uint32_t c) {
    uint32_t rm = 0;
    for (uint32_t r = 0; r < R; ++r) rm |= ((m >> (r * C + c)) & 1) << r;
    return rm;
}
inline uint32_t cols_in_row(uint32_t m, uint32_t r) { return (m >> (r * C)) & ((1u << C) - 1); }
// token t's route word(s) (RW 1: mask | order << 16 | link << 17; RW 2: mask, order | link << 1) from its K expert
// ids, at o[RW t]
inline void put(volatile uint32_t* o, uint32_t t, uint32_t m, uint32_t flags) {
    if constexpr (RW == 2) {
        o[2 * t] = m;
        o[2 * t + 1] = flags;
    } else {
        o[t] = m | (flags << 16);
    }
}
inline void word(volatile uint32_t* o, uint32_t i, volatile const uint16_t* ids, uint32_t t) {
    const uint32_t r0 = ME / C, c0 = ME % C;
    uint32_t m = 0;
    for (uint32_t k = 0; k < K; ++k) m |= 1u << CHIP_OF[ids[k]];
    m &= ~(1u << ME);
    if (m == 0) return put(o, i, 0, 0);
    const uint32_t cm = cols_of(m), rm = rows_of(m);
    uint32_t hx = links(xspan(c0, cm & ~(1u << c0), t)), hy = links(yspan(r0, rm & ~(1u << r0), t));
    for (uint32_t c = 0; c < C; ++c) {
        if (((cm | (1u << c0)) >> c) & 1) hx += links(yspan(r0, rows_in_col(m, c) & ~(1u << r0), t));
    }
    for (uint32_t r = 0; r < R; ++r) {
        if (((rm | (1u << r0)) >> r) & 1) hy += links(xspan(c0, cols_in_row(m, r) & ~(1u << c0), t));
    }
    const uint32_t hsh = t * 2654435761u;
    const uint32_t order = hx < hy ? 0 : (hy < hx ? 1 : (hsh >> 31) & 1);
    put(o, i, m, order | (((hsh >> 29) & 1) << 1));
}
}  // namespace rw_
"""


def route_src(mesh_shape, me, *, E, EPC, K, torus=(False, False)):
    """The route-word function (namespace rw_) of chip me (r C + c, dispatch_prep's numbering): expert e on chip e //
    EPC; torus (wrap X, wrap Y) where cabled (a wrap needs more than two chips, as sparse_all_gather_2d)."""
    R, C = mesh_shape
    src = _ROUTE_FN
    for k, v in span_subs(R, C, bool(torus[0]) and C > 2, bool(torus[1]) and R > 2).items():
        src = src.replace(k, v)
    return (
        src.replace("__R__", str(R))
        .replace("__C__", str(C))
        .replace("__ME__", str(me))
        .replace("__K__", str(K))
        .replace("__RW__", str(route_words(R * C)))
        .replace("__E__", str(E))
        .replace("__CHIPS__", ", ".join(str(e // EPC) for e in range(E)))
    )
