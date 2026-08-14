# sfpi: `vInt` and `vUInt` relational compares are wrong over the full 32-bit range

**Labels:** `sfpi`, `correctness`
**Kind:** **correctness** — the cheap idiomatic form is the broken one, and there is no diagnostic
**Worth:** unblocks `calculate_binary_comp_uint`; removes a 6-instruction hand-rolled fold
**Compiler:** `riscv-tt-elf-g++ (tenstorrent/sfpi:7.69.0[822]) 15.1.0`, `-O2 -mcpu=tt-bh-tensix`
**Context:** ask 6 of 7 from [the umbrella writeup](../ISSUE.md). Origin: [tt-metal#52932](https://github.com/tenstorrent/tt-metal/pull/52932).

## What we have today

`v_if(a < b)` on `vInt` emits a single instruction:

```
SFPIADD  L0, L1, 0, 2
```

That is a two's-complement subtract with the condition code taken from the sign of the
difference, so it wraps:

- **`vInt`**: `INT32_MAX − (−1) = 0x80000000`, sign set, so **`INT32_MAX > -1` answers false**.
- **`vUInt`**: `0u − 0xC0000000u = 0x40000000`, sign clear, so **`0u < 0xC0000000u` answers false**.

`vUInt` is worth spelling out, because one might expect the unsigned type to select an
unsigned compare. It does not — the emitted instruction is the identical signed
`SFPIADD ..., 0, 2`:

```cpp
// verified: 03_compare_costs.cc :: m_uint32_lt
extern "C" void m_uint32_lt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt a = dst_reg[i0 * TS].mode<sfpi::DataLayout::U32>();
        vUInt b = dst_reg[i1 * TS].mode<sfpi::DataLayout::U32>();
        vUInt r = 0;
        v_if(a < b) { r = 1; }
        v_endif;
        dst_reg[io * TS].mode<sfpi::DataLayout::U32>(AM6) = r;
    }
}
```

## This has already shipped as a bug twice

This is not a hypothetical. It is the bug that #27829 and #28397 originally fixed, it is
what #51097 rolled back, and it re-broke `ttnn.lt` / `ttnn.gt` on int32 the moment
[#52932](https://github.com/tenstorrent/tt-metal/pull/52932) wrote the compare
idiomatically.

`calculate_binary_comp_int32` on `main` therefore carries a hand-rolled 6-instruction
branchless sign-fold and a "do **NOT** write this as `v_if(a < b)`" comment.
`calculate_binary_comp_uint` was left in raw TTI entirely for the same reason.

## What we need

The correct full-range compare, which is what we are forced to write today:

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

Six of those nine instructions — `SFPSETSGN`, `SFPIADD`, `SFPXOR`, `SFPOR`, `SFPXOR`,
`SFPSHFT` — exist only to replace the one `SFPIADD` that sfpi would have emitted.

**The ask**, in descending order of preference:

1. Lower integer relational compares overflow-safely by default — `SFPGT`/`SFPLE` on a
   sign-flipped operand, or the sign-fold above, whichever the backend prefers.
2. Failing that, reject or warn on the unsafe form, and expose an explicit full-range
   compare that kernels can call.

Silently answering `INT32_MAX > -1` as false is the worst of the three options.

## Note the shape of the trap

| form | instructions per DEST row | correct? |
|---|---|---|
| `v_if(a < b)` on `vInt` | 8 | **no** |
| hand-rolled branchless sign-fold | 9 | yes |

sfpi's compare is **cheap and wrong**, so the fast path is the broken one, the correct path
costs more, and nothing in between warns you. That combination is why this keeps getting
reintroduced: someone converts a kernel, the instruction count goes down, CI perf looks
better, and the range bug comes back.

## Repro

```sh
git fetch origin ldjurovic/sfpi-perf-gap-investigation
git checkout ldjurovic/sfpi-perf-gap-investigation
./sfpi_perf_investigation/run.sh
```

Relevant probes: `probe_int32_lt_naive` (8, wrong), `probe_int32_lt` (10) and
`a_int32_lt_addrmod` (9) for the correct fold, and `m_uint32_lt` (7) for the `vUInt` case.
`repro.cc :: int32_lt_wrong` is the minimal standalone version. See
[the umbrella writeup](../ISSUE.md) §3 for the counting method.
