# sfpi: no total-order float compare reachable from `v_if` — `SFPGT`/`SFPLE` are never emitted

**Labels:** `sfpi`, `perf`, `correctness`
**Kind:** **missing feature** — the only ask on our list that needs new API surface
**Worth:** 1–2 Tensix instructions on the compare itself, plus 4 more in workarounds it lets us delete
**Compiler:** `riscv-tt-elf-g++ (tenstorrent/sfpi:7.69.0[822]) 15.1.0`, `-O2 -mcpu=tt-bh-tensix`
**Context:** ask 5 of 7 from [the umbrella writeup](../ISSUE.md). Origin: [tt-metal#52932](https://github.com/tenstorrent/tt-metal/pull/52932).

This is the only item of the seven that is a genuine feature gap rather than a missed
optimization, and it is the only one that also fixes a correctness gap we currently
**cannot close in source at all**.

## What we have today

`SFPGT` and `SFPLE` compare sign-magnitude bit patterns directly. That gives the total
order over finite values, ±0 and ±inf in a single instruction, with no arithmetic. sfpi
never emits either of them for a `v_if`. Instead every `vFloat` relational operator
becomes a subtract plus a sign test:

| sfpi source | emitted | count | with `SFPGT`/`SFPLE` |
|---|---|---|---|
| `v_if(a < b)`, `vFloat` | `SFPMAD` + `SFPSETCC(LT0)` | 2 | 1 |
| `v_if(a > b)`, `vFloat` | `SFPMAD` + `SFPSETCC(GTE0)` + `SFPSETCC(NE0)` | 3 | 1 |

```cpp
// verified: 03_compare_costs.cc :: i_float_lt_only
extern "C" void i_float_lt_only(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        dst_reg[io * TS] = 0.0f;
        v_if(a < b) { dst_reg[io * TS].mode<sfpi::DataLayout::Default>(AM6) = 1.0f; }
        v_endif;
    }
}
```

The instructions do exist in the compiler, as `__builtin_rvtt_sfpgt` and
`__builtin_rvtt_sfple` (`tensix_builtins.def:123–126`), and calling one directly really
does emit `SFPGT L0, L1, 0, 8`:

```cpp
// verified: 02_single_workarounds.cc :: d_raw_sfpgt
extern "C" void d_raw_sfpgt(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * TS];
        vFloat b = dst_reg[i1 * TS];
        // SFPGT_MOD1_SET_VD == 8: vd = (a > b) ? -1 : 0, total order on the bits.
        vUInt mask = as<vUInt>(vFloat(__builtin_rvtt_sfpgt(a.get(), b.get(), 8)));
        dst_reg[io * TS].mode<sfpi::DataLayout::U32>() = mask >> 31;
        dst_reg++;
    }
}
```

But that only produces a **vector** result (`SET_VD` — a 0/−1 lane mask), and
`sfpi::vBool` is constructible only from `sfpxfcmpv` / `sfpxfcmps` / `sfpxicmps` /
`sfpxicmpv`, every one of which the backend lowers to arithmetic plus `SFPSETCC`. There is
no path from a raw `SFPGT` result into `v_if`. Round-tripping through the mask with
`v_if(mask != 0)` costs the `SFPSETCC` straight back, so it exactly breaks even.

## The expensive part is the semantics, not the instruction count

Subtract-then-test-sign is not a total order, and every consequence had to be worked
around in kernel source:

- **`inf − inf = NaN`** with a clear sign bit, so `inf == inf` answered "unordered" and
  `inf <= inf` answered false. We fixed it by adding an explicit bitwise-equality clause:
  `SFPIADD` + `SFPSETCC` on `eq`/`ne` (**+2**), and `SFPIADD` + `SFPSETCC` + `SFPSTORE` +
  `SFPENCC` on `le`/`ge` (**+4**). That is pure workaround cost — the raw-TTI
  `SFPGT`/`SFPLE` sequences needed none of it.

- **`a − b` underflows into the flushed denormal range**, so operands differing by less
  than 2⁻¹²⁶ compare equal. Adjacent normals near the bottom of the exponent range are
  reachable, so this is not theoretical. This one is **still not fixed**: it is pinned by
  an `xfail` (`test_binary_comp_fp32_denormal_window_ties`) and is a live behaviour
  regression against raw `SFPGT`/`SFPLE`, which were exact there. We cannot close it in
  source at all.

## What we need

Either lower `vBool(Cond, vFloat, vFloat)` to `SFPGT`/`SFPLE` with `SET_CC` when the
requested predicate is a total-order one, or expose an explicit spelling that `v_if`
accepts:

```cpp
// Illustrative -- neither of these exists today.
v_if(total_order_lt(a, b)) { ... } v_endif;

// or, once the vSMag lowering bug (ask 7) is fixed, simply:
v_if(vBool(vBool::LT, as<vSMag>(a), as<vSMag>(b))) { ... } v_endif;
```

The second spelling is the one we would prefer — `vSMag` is already sfpi's sign-magnitude
type, `SFPXCMP_MOD1_TYPE_SMAG` already exists in the mod encoding, and a total-order
compare is exactly what a sign-magnitude compare *means*. See
[ask 7](07-vsmag-compare-lowering-bug.md): fixing that bug would deliver most of this
issue as a side effect.

## Why it matters

With a total-order compare available to `v_if`, both remaining non-parity families reach
raw-TTI parity:

| kernel family | raw TTI | best sfpi today | with a total-order compare |
|---|---|---|---|
| fp32 `le`/`ge` | 13 | 19 (+46%) | **13 (par)** |
| fp32 `lt`/`gt` | 11 | 12 (+9%) | **11 (par)** |
| fp32 `eq`/`ne` | 14 | 16 (+14%) | **14 (par)** |

How that last column is obtained matters, so to be explicit about what is measured and
what is accounted for. We can only *approximate* the ask today, via the
`__builtin_rvtt_sfpgt` escape hatch above: take its 0/−1 mask and test it. That test is an
`SFPSETCC`, so the round-trip breaks even against the arithmetic compare it replaces.
Measured that way, `le`/`ge` is 14 (`ideal_fp32_le` against `best2_fp32_le` at 19) and
`lt`/`gt` is 12 (`ideal_fp32_lt`, i.e. no gain — the round-trip exactly cancels).

A compare that sets the CC **directly**, which is what this issue asks for, removes that
round-trip `SFPSETCC` and takes both rows to the values above. So each is one measured
instruction plus one instruction of arithmetic that is accounted for exactly, not
estimated. The `eq`/`ne` row is today's 16 minus the two-instruction bitwise-equality
clause that exists only to make `inf == inf` answer true, which a total order answers
directly.

This gap keeps resurfacing outside the PR that prompted this writeup.
`ckernel_sfpu_recip.h` already carries the comment *"Equivalently, we could use
`v_if (t >= 2.0)` instead, but SFPI doesn't support SFPLE/GT at the moment."*, and
`tt-llk`'s `perf_sfpu_comp.py` notes that the float `calculate_comp` *"is deliberately
retained as hand-tuned TTI (the SFPI form measured slower on Wormhole)"*.

## Repro

```sh
git fetch origin ldjurovic/sfpi-perf-gap-investigation
git checkout ldjurovic/sfpi-perf-gap-investigation
./sfpi_perf_investigation/run.sh
```

Relevant probes: `i_float_lt_only` (7) and `j_float_gt_only` (8) for the isolated compare
cost, `d_raw_sfpgt` (6) for the builtin being reachable but unusable from `v_if`,
`h_smag_lt` (6) for the sign-magnitude path, and `best2_fp32_le` (19) against
`ideal_fp32_le` (14) for the end-to-end effect on `le`/`ge`. See
[the umbrella writeup](../ISSUE.md) §3 for the counting method.
