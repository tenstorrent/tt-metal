# sfpi: `vSMag` / `DataLayout::SM32` compares lower to a two's-complement subtract, which cannot order sign-magnitude values

**Labels:** `sfpi`, `bug`
**Kind:** **likely bug** — please confirm
**Worth:** correctness; and fixing it would deliver most of [ask 5](05-total-order-float-compare.md) as a side effect
**Compiler:** `riscv-tt-elf-g++ (tenstorrent/sfpi:7.69.0[822]) 15.1.0`, `-O2 -mcpu=tt-bh-tensix`
**Context:** ask 7 of 7 from [the umbrella writeup](../ISSUE.md). Origin: [tt-metal#52932](https://github.com/tenstorrent/tt-metal/pull/52932).

## What we have today

`vSMag` is sfpi's sign-magnitude type and `SFPXCMP_MOD1_TYPE_SMAG` exists in the mod
encoding, so we expected sign-magnitude compares to be exactly where `SFPGT`/`SFPLE` would
show up. They are not:

```cpp
// verified: 03_compare_costs.cc :: h_smag_lt
extern "C" void h_smag_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vSMag a = as<vSMag>(vUInt(dst_reg[i0 * TS].mode<sfpi::DataLayout::U32>()));
        vSMag b = as<vSMag>(vUInt(dst_reg[i1 * TS].mode<sfpi::DataLayout::U32>()));
        dst_reg[io * TS] = 0.0f;
        v_if(vBool(vBool::LT, a, b)) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}
```

emits

```
SFPLOAD   L0, a0, 4, 7
SFPLOAD   L1, a3, 4, 7
SFPSTORE  L9, a2, 0, 7
SFPIADD   L0, L1, 0, 2     # <-- two's-complement subtract, same as vInt
SFPSTORE  L10, a4, 0, 6
SFPENCC   3, 10
```

That `SFPIADD ..., 0, 2` is the identical two's-complement subtract sfpi emits for `vInt`
(see [ask 6](06-integer-compare-overflow.md)). The sign-magnitude type made no difference
to the lowering.

Loading DEST as `DataLayout::SM32` instead produces byte-identical code — the compiler
tail-merged the two functions, which is how we noticed:

```cpp
// verified: 03_compare_costs.cc :: n_sm32_lt
extern "C" void n_sm32_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vSMag a = dst_reg[i0 * TS].mode<sfpi::DataLayout::SM32>();
        vSMag b = dst_reg[i1 * TS].mode<sfpi::DataLayout::SM32>();
        dst_reg[io * TS] = 0.0f;
        v_if(vBool(vBool::LT, a, b)) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}
```

```
n_sm32_lt:
        tail    h_smag_lt
```

## Why this is wrong

A two's-complement subtract cannot order sign-magnitude values. Counterexample:

| value | sign-magnitude bits |
|---|---|
| −1 | `0x80000001` |
| −2 | `0x80000002` |

`0x80000001 − 0x80000002 = −1`, sign set, so the compare answers **−1 < −2**. Wrong.

More generally, for two negative sign-magnitude operands the correct order is the
*reverse* of magnitude order, and a plain subtract on the raw patterns never reverses it.

## What we need

Please confirm whether this is a lowering bug rather than an intended limitation. We would
expect `vBool(Cond, vSMag, vSMag)` to lower to `SFPGT` / `SFPLE`, which compare
sign-magnitude bit patterns natively and in a single instruction — that is precisely what
those instructions are for.

If it were fixed that way, [ask 5](05-total-order-float-compare.md) would largely be solved
as a side effect: `vSMag` is the natural type in which to spell a total-order float
compare, and

```cpp
v_if(vBool(vBool::LT, as<vSMag>(a), as<vSMag>(b))) { ... } v_endif;
```

would become the total-order compare we currently have no way to reach from `v_if`. That
makes this the highest-leverage of the two, since it needs no new API surface at all — only
a corrected lowering for a type that already exists.

## Repro

```sh
git fetch origin ldjurovic/sfpi-perf-gap-investigation
git checkout ldjurovic/sfpi-perf-gap-investigation
./sfpi_perf_investigation/run.sh   # h_smag_lt, and n_sm32_lt reported as identical codegen
```

The counting script reports `n_sm32_lt` as "identical codegen to `h_smag_lt`" because it
detects the tail call. See [the umbrella writeup](../ISSUE.md) §3 for the counting method.
