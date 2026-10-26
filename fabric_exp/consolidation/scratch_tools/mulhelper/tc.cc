#include <cstdint>
struct St { uint32_t a, b, c, d; };
using Fn = void (*)(St*, uint32_t, uint32_t, uint32_t);
extern Fn table[4];
__attribute__((noinline)) void step_b(St* s, uint32_t x, uint32_t y, uint32_t z);
void step_a(St* s, uint32_t x, uint32_t y, uint32_t z) {
    x += s->a * 17;               // a multiply in this handler only
    [[gnu::musttail]] return table[x & 3](s, x, y, z);
}
