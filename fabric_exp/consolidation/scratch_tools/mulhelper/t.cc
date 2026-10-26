#include <cstdint>
// 1. plain constant multiply: GCC decides mul vs shift/add by its cached cost search
uint32_t plain(uint32_t x) { return x * 17; }
// 2. hand-written shift/add: GCC canonicalizes it back to x * 17 first
uint32_t handwritten(uint32_t x) { return (x << 4) + x; }
// 3. the router's encoding: (t << 4) | t with t < 16 is also x * 17 to GCC
uint32_t encoding(uint32_t t) { t &= 0xF; return 0x100 | (t << 4) | t; }
// 4. struct array indexed at run time: a hidden multiply by sizeof
struct S40 { uint32_t w[10]; };
uint32_t index40(const S40* a, uint32_t i) { return a[i].w[3]; }
// 5. helper: hide the constant's value -> always one mul (no cost search)
template <uint32_t C> inline uint32_t mul_by(uint32_t x) { uint32_t c = C; asm("" : "+r"(c)); return x * c; }
uint32_t helper_mul(uint32_t x) { return mul_by<17>(x); }
// 6. helper: force shift+add (barrier stops re-folding into a multiply)
inline uint32_t shl4_add(uint32_t x) { uint32_t t = x << 4; asm("" : "+r"(t)); return t + x; }
uint32_t helper_shadd(uint32_t x) { return shl4_add(x); }
// 7. division by a constant: becomes a multiply-high by a magic number
uint32_t div7(uint32_t x) { return x / 7; }
