# sfpi: `&&` abandons CC chaining as soon as one term needs a helper instruction

**Labels:** `sfpi`, `perf`
**Kind:** missed optimization
**Worth:** 6 Tensix instructions per predicate
**Compiler:** `riscv-tt-elf-g++ (tenstorrent/sfpi:7.69.0[822]) 15.1.0`, `-O2 -mcpu=tt-bh-tensix`
**Context:** ask 4 of 7 from [the umbrella writeup](../ISSUE.md). Origin: [tt-metal#52932](https://github.com/tenstorrent/tt-metal/pull/52932).

`&&` is currently a performance trap whose cost depends on whether an unrelated constant
happened to fit an immediate field.

## What we have today

A three-term `&&` where every term lowers to exactly one CC-setting instruction chains
perfectly — 9 instructions per row, no `SFPPUSHC` anywhere
(`p_three_term_small_imm` in the probes, where the widest constant is `100` and fits
`SFPIADD`'s 12-bit immediate).

Change one term so it needs a helper instruction first, and the same predicate jumps to
**21**:

```cpp
// verified: 04_and_lowering.cc :: q_three_term_preloaded
extern "C" void q_three_term_preloaded(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = 0x7F800000;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 0.0f;
        v_if(a < b && sum != 0.0f && as<vInt>(sum) <= inf_bits) {
            dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f;
        }
        v_endif;
    }
}
```

`0x7F800000` does not fit the immediate, so it lives in an LReg and the third term needs an
`SFPMOV` before its `SFPIADD`. At that point sfpi gives up on CC chaining entirely and
materialises the partial predicate as a *value*:

```
SFPSETCC L2, 0, 0
SFPSETCC L1, 0, 2
SFPLOADI L2, 1, 4        # materialise the partial predicate as a value
SFPPUSHC 0
SFPMOV   L3, L0, 2
SFPIADD  L3, L1, 0, 10
SFPSETCC L3, 0, 2
SFPCOMPC
SFPMOV   L1, L2, 2
SFPLOADI L1, 0, 4        # LV:L1
SFPPOPC  0
SFPSETCC L1, 0, 6
```

Hoisting the constant out of the loop does not help — that is already what the code above
does. The helper instruction just becomes an `SFPMOV` instead of an `SFPLOADI`, and the
cliff is identical either way (21 both ways).

## What we need

Spelling the *identical* predicate as nested `v_if`/`v_endif` gets it back to **15**:

```cpp
// verified: 04_and_lowering.cc :: r_nested_vif
extern "C" void r_nested_vif(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
    vInt inf_bits = 0x7F800000;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        vFloat sum = as<vFloat>(setsgn(as<vUInt>(a), 0)) + as<vFloat>(setsgn(as<vUInt>(b), 0));
        dst_reg[io * TS] = 0.0f;
        v_if(a < b) {
            v_if(sum != 0.0f) {
                v_if(as<vInt>(sum) <= inf_bits) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}
```

The six instructions that disappear are exactly the materialisation machinery —
`SFPLOADI`, `SFPPUSHC`, `SFPMOV`, `SFPLOADI`, `SFPPOPC`, `SFPSETCC`. The nested form
hoists the third term's `SFPMOV` ahead of the chain and then just chains `SFPSETCC`s,
which is what we wanted from `&&` in the first place.

**The ask:** lower `A && B && C` the way nested `v_if` already does — hoist each term's
helper instructions ahead of the chain, then chain the `SFPSETCC`s. The cheap lowering
demonstrably exists and is reachable; `&&` just does not reach it.

## Why it matters

`&&` is the natural way to write a guarded comparison, and it is how all three fp32
families in the conversion were written. The cost model is invisible from the source: the
same three-term predicate is 9 instructions or 21 depending on the *magnitude of a
constant* in one of the terms. Nothing warns, and a reviewer has no way to see it.

Nesting `v_if` three deep to get the fast path is also a real readability loss — it is the
opposite of why we want to move these kernels to sfpi.

## Repro

```sh
git fetch origin ldjurovic/sfpi-perf-gap-investigation
git checkout ldjurovic/sfpi-perf-gap-investigation
./sfpi_perf_investigation/run.sh
```

Compare `p_three_term_small_imm` (9, chains fine), `q_three_term_preloaded` (21, falls off
the cliff) and `r_nested_vif` (15, same predicate spelled as nesting). See
[the umbrella writeup](../ISSUE.md) §3 for the counting method.
