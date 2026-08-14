# sfpi: integer compare against a loop-invariant costs 4 instructions where a commute would cost 1

**Labels:** `sfpi`, `perf`
**Kind:** missed optimization
**Worth:** 3 Tensix instructions per compare-against-a-constant
**Compiler:** `riscv-tt-elf-g++ (tenstorrent/sfpi:7.69.0[822]) 15.1.0`, `-O2 -mcpu=tt-bh-tensix`
**Context:** ask 1 of 7 from [the umbrella writeup](../ISSUE.md). Origin: [tt-metal#52932](https://github.com/tenstorrent/tt-metal/pull/52932).

This is the single largest missed optimization we found, and it appears in six of the
eight SFPU comparison kernel families we tried to convert — every NaN guard is a
`|x|` against `+inf` compare.

## What we have today

`+inf`'s bit pattern does not fit `SFPIADD`'s 12-bit immediate, so it has to live in an
LReg, hoisted out of the loop. Written the way it reads:

```cpp
// verified: 06_commuted_compares.cc :: s_le_form
extern "C" void s_le_form() {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(abs_bits <= inf_bits) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}
```

9 Tensix instructions per DEST row. Four of them are the compare:

```
SFPMOV    L2, L0, 2      # copy the loop invariant, because SFPIADD will clobber it
SFPIADD   L2, L1, 0, 10
SFPSETCC  L2, 0, 2
SFPCOMPC                 # because there is no ">0" CC, `x <= k` becomes !(x > k)
```

## What we need

Commuting the comparison to `inf_bits >= abs_bits` — the *same predicate*, just the
other operand order — collapses all four into one:

```cpp
// verified: 06_commuted_compares.cc :: t_ge_form
extern "C" void t_ge_form() {
    vInt inf_bits = INFB;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(inf_bits >= abs_bits) { dst_reg[0].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}
```

```
SFPIADD   L1, L0, 0, 10
```

6 instructions per row instead of 9. The commuted form picks the operand order in which
the invariant survives, so the defensive `SFPMOV` is unnecessary, and it lands on a CC
polarity `SFPIADD` can fuse, so the `SFPSETCC` and the `SFPCOMPC` both disappear.

This is exactly what the raw-TTI original did by hand:

```cpp
TTI_SFPIADD(0, INF, ABS_V, SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | SFPIADD_MOD1_CC_GTE0);
```

**The ask:** canonicalise integer relational compares so that

1. the operand direction is chosen to keep read-only / loop-invariant operands live,
2. the polarity is chosen from the set `SFPIADD`'s CC field can express, and
3. `SFPIADD`'s fused CC test is used instead of a following `SFPSETCC`.

Never emit `SFPCOMPC` for a predicate that could have been commuted instead.

## Why it matters

Two of the eight kernel families reach **exact instruction parity with raw TTI** on the
strength of this one rewrite:

| kernel family | raw TTI | as written | commuted |
|---|---|---|---|
| float `ltz`/`gtz` | 8 | 11 (+38%) | **8 (par)** |
| float `lez`/`gez` | 10 | 12 (+20%) | **10 (par)** |

The compare in isolation goes 9 → 6.

Nothing about the fast form is discoverable. `a <= b` and `b >= a` are the same predicate
to any reader, they are the same predicate to the type system, and one of them is three
instructions more expensive with no diagnostic saying so.

## Repro

```sh
git fetch origin ldjurovic/sfpi-perf-gap-investigation
git checkout ldjurovic/sfpi-perf-gap-investigation
./sfpi_perf_investigation/run.sh          # look for s_le_form vs t_ge_form
```

`run.sh` locates the toolchain and prints Tensix instructions executed per DEST row for
every probe, modelling the replay buffer. See [the umbrella writeup](../ISSUE.md) §3 for
the counting method.
