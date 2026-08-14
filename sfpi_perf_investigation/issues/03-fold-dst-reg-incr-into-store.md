# sfpi: `dst_reg++` always emits a separate `TTINCRWC` instead of folding into the preceding store

**Labels:** `sfpi`, `perf`
**Kind:** missed optimization
**Worth:** 1 Tensix instruction per DEST row
**Compiler:** `riscv-tt-elf-g++ (tenstorrent/sfpi:7.69.0[822]) 15.1.0`, `-O2 -mcpu=tt-bh-tensix`
**Context:** ask 3 of 7 from [the umbrella writeup](../ISSUE.md). Origin: [tt-metal#52932](https://github.com/tenstorrent/tt-metal/pull/52932).

This is the cheapest fix on our list and it closes an entire kernel family on its own.

## What we have today

`impl_::DstRegFile::operator++` unconditionally emits `TTINCRWC 0, 2, 0, 0`
(`sfpi_classes.h:400`), and `dst_reg[k] = v` always stores with `addr_mode` 7, meaning no
increment. So the canonical loop shape pays a whole instruction to advance DEST:

```cpp
// verified: 01_idiomatic_baseline.cc :: probe_int32_lt
extern "C" void probe_int32_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vInt a = dst_reg[i0 * TS].mode<sfpi::DataLayout::I32>();
        vInt b = dst_reg[i1 * TS].mode<sfpi::DataLayout::I32>();
        vInt fold = a - as<vInt>(setsgn(as<vUInt>(b), 0));
        a = a ^ b;
        a = a | fold;
        a = a ^ b;
        dst_reg[io * TS].mode<sfpi::DataLayout::I32>() = as<vInt>(as<vUInt>(a) >> 31);
        dst_reg++;
    }
}
```

10 Tensix instructions per DEST row:

```
SFPLOAD   L0, a0, 4, 7
SFPLOAD   L1, a3, 4, 7
SFPSETSGN L2, L1, 0, 1
SFPIADD   L2, L0, 0, 6
SFPXOR    L0, L1
SFPOR     L0, L0, L2, 1
SFPXOR    L0, L1
SFPSHFT   L0, L0, -31, 5
SFPSTORE  L0, a4, 4, 7    # addr_mode 7: no increment
TTINCRWC  0, 2, 0, 0      # ... so the advance costs a separate instruction
```

The raw-TTI kernel this replaced put the dest advance in the final store's address mod,
where it is free — 9 instructions, no `TTINCRWC`.

**This is the entire int32 regression.** The two bodies are instruction-for-instruction
identical; sfpi is 10 against raw TTI's 9 solely because of the `TTINCRWC`. That is
+11.1% by instruction count, against +11.5% measured in CI.

## What we need

Writing the store with an explicit `addr_mode` and dropping the `dst_reg++` reaches
parity:

```cpp
// verified: 02_single_workarounds.cc :: a_int32_lt_addrmod
extern "C" void a_int32_lt_addrmod(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vInt a = dst_reg[i0 * TS].mode<sfpi::DataLayout::I32>();
        vInt b = dst_reg[i1 * TS].mode<sfpi::DataLayout::I32>();
        vInt fold = a - as<vInt>(setsgn(as<vUInt>(b), 0));
        a = a ^ b;
        a = a | fold;
        a = a ^ b;
        dst_reg[io * TS].mode<sfpi::DataLayout::I32>(AM6) = as<vInt>(as<vUInt>(a) >> 31);
    }
}
```

9 instructions — the last one becomes `SFPSTORE L0, a4, 4, 6` and the `TTINCRWC` is gone.
The emitted sequence is now byte-for-byte the raw-TTI sequence, in the same order.

**The ask:** when a loop body's last DEST access is a store immediately followed by
`dst_reg++`, fold the increment into that store's `addr_mode` and drop the `TTINCRWC`.

## Why the workaround is not good enough

We do not want to write `mode<...>(AM6)` in kernels, for two reasons.

First, nobody found it. It is one line of comment in `sfpi.h`, and the addr_mod constants
are not in `sfpi_constants.h`. The conversion PR was written, reviewed and merged without
anyone noticing the `TTINCRWC`.

Second, `6` is `ADDR_MOD_6`, a **metal** convention — the kernels' `*_init()` functions
program it with `dest.incr = 2`. Passing that number through `mode()` leaks a metal
register index into what should be an sfpi-level concern, and it silently produces wrong
addressing if the caller did not happen to program that addr_mod the same way. It is a
correctness footgun traded for one instruction.

## Repro

```sh
git fetch origin ldjurovic/sfpi-perf-gap-investigation
git checkout ldjurovic/sfpi-perf-gap-investigation
./sfpi_perf_investigation/run.sh   # probe_int32_lt (10) vs a_int32_lt_addrmod (9)
```

See [the umbrella writeup](../ISSUE.md) §3 for the counting method.
