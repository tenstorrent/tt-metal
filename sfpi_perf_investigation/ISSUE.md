# sfpi: idiomatic SFPU code costs 30–90% more Tensix instructions than the raw TTI it replaces

**Labels:** `sfpi`, `perf`
**Compiler:** `riscv-tt-elf-g++ (tenstorrent/sfpi:7.69.0[822]) 15.1.0`, `-O2 -mcpu=tt-bh-tensix`
**Origin:** [tt-metal#52932](https://github.com/tenstorrent/tt-metal/pull/52932) — closed, not merged, because every kernel it converted got slower.

---

## 1. TL;DR

We tried to convert the Blackhole SFPU comparison kernels (`ckernel_sfpu_comp.h`,
`ckernel_sfpu_binary_comp.h`) from hand-written TTI to sfpi. Every one of the eight
converted kernel families regressed, by +11% to +98% cycles. Nothing got faster, so
we closed the PR and kept the raw TTI.

We then disassembled every variant we could think of. The result splits cleanly into
two very different problems:

**Problem A — sfpi's *optimizer* leaves 30–90% on the table for idiomatic code.**
Four of the seven gaps below are missed optimizations, not missing features. Once we
rewrote the same sfpi source in the *shape* the raw TTI happened to use — predicated
store instead of predicated register assignment, dest increment folded into the store's
`addr_mode`, nested `v_if` instead of `&&`, and integer compares commuted to the polarity
`SFPIADD`'s condition-code field can express — four of the eight families reached
**exact instruction parity** with raw TTI, and a fifth came within one instruction.
The fast form and the natural form are the same algorithm; only the spelling differs.
That spelling is undiscoverable, undocumented, and it is why the PR failed.

**Problem B — one primitive is genuinely missing: a total-order float compare usable
from `v_if`.** `SFPGT`/`SFPLE` compare sign-magnitude bit patterns in a single
instruction. sfpi never emits them. Every `vFloat` relational operator lowers to
`SFPMAD` (a − b) plus `SFPSETCC` on the sign, which is 2–3 instructions instead of 1
**and has different semantics**: `inf − inf = NaN`, and `a − b` flushes to zero for
operands closer than 2⁻¹²⁶. Working around the semantic difference costs 4 more
instructions on `le`/`ge` and 2 on `eq`/`ne`, and one behaviour difference is still
open in `main` behind an `xfail`.

**What we need, in priority order** (§5 has details):

| # | Ask | Kind | Worth |
|---|-----|------|-------|
| 1 | Commute integer compares to a CC polarity `SFPIADD` can fuse, instead of copy + subtract + `SFPSETCC` + `SFPCOMPC` | missed opt | 3 instr per compare-against-a-constant |
| 2 | Lower "select between two constants under a CC, then store" to two predicated stores | missed opt | 2 instr per predicated result |
| 3 | Fold a trailing `dst_reg++` into the preceding store's `addr_mode` | missed opt | 1 instr per DEST row |
| 4 | Make `A && B && C` lower like nested `v_if` even when a term needs a helper instruction | missed opt | up to 8 instr per predicate |
| 5 | Expose a total-order (sign-magnitude) compare that `v_if` can consume | **missing feature** | 2 instr, plus 4 more in dropped workarounds |
| 6 | Make `vInt`/`vUInt` relational compares correct over the full range | **correctness** | unblocks `calculate_binary_comp_uint`; removes a 6-instr hand-rolled fold |
| 7 | `vSMag` / `DataLayout::SM32` relational compares lower to a two's-complement subtract, which cannot order sign-magnitude values | **likely bug** | — |

Asks 1–4 are the bulk of the regression and none of them require new ISA surface or new
API — they are all "sfpi already knows how to emit this, just not from this source form."

---

## 2. What was converted, and what it cost

PR #52932 restored the Blackhole comparison portions of #49926 that #51097 had rolled
back. Two headers, eight kernel families. The CI cycle deltas measured on
`bh_p150b_civ2`:

| kernel family | cycle cost vs raw TTI |
|---|---|
| fp32 `le`/`ge` | **+84%** |
| fp32 `lt`/`gt` | **+82%** |
| fp32 `eq`/`ne` | **+46%** |
| float `ltz`/`gtz` | **+91%** (bf16) / **+98%** (fp32) |
| float `lez`/`gez` | **+56%** (bf16) / **+59%** (fp32) |
| float `eqz`/`nez` | **+31%** (bf16) / **+34%** (fp32) |
| int32 `lt`/`gt`/`le`/`ge` | **+11.5%** |
| uint16/uint32 `eqz`/`nez` — already sfpi on `main`, control | 0.0% |

This is not a Blackhole-only or a one-off observation. `tt-llk`'s
`perf_sfpu_comp.py` already carries the note *"the float `calculate_comp` is deliberately
retained as hand-tuned TTI (the SFPI form measured slower on Wormhole)"*, and
`ckernel_sfpu_recip.h` already carries *"Equivalently, we could use `v_if (t >= 2.0)`
instead, but SFPI doesn't support SFPLE/GT at the moment."* The same gaps keep
surfacing in different kernels.

---

## 3. Method

Each `TTI_*`/`TT_*` macro in the raw-TTI kernels is exactly one Tensix instruction, so
the raw baseline is a source count. For the sfpi versions we compiled each loop body
standalone and simulated the replay buffer over the emitted stream: `TTREPLAY
start,len,_,1` records and executes `len` instructions; `TTREPLAY start,len,0,0`
re-executes them. Total executed Tensix instructions ÷ 8 unrolled iterations = cost per
DEST row. This is the quantity `PerfRunType.MATH_ISOLATE` measures, and it tracks the
CI cycle deltas above closely.

Cross-check that the counting is sound: the int32 fold compiles to
`SFPLOAD, SFPLOAD, SFPSETSGN, SFPIADD, SFPXOR, SFPOR, SFPXOR, SFPSHFT, SFPSTORE` —
the identical nine-instruction sequence, in the identical order, as the raw-TTI source
it replaced. §7 has a self-contained repro.

---

## 4. The numbers

Tensix instructions per DEST row. "PR" is what #52932 shipped, written the way one
would naturally write sfpi. "Best today" is the same algorithm rewritten with all of
the workarounds in §5.1–§5.4 stacked. "With ask 5" adds a total-order compare.

| kernel family | raw TTI | PR #52932 | best sfpi today | with ask 5 |
|---|---|---|---|---|
| float `eqz`/`nez` | 6 | 8 (+33%) | **6 (par)** | 6 |
| float `ltz`/`gtz` | 8 | 14 (+75%) | **8 (par)** | 8 |
| float `lez`/`gez` | 10 | 14 (+40%) | **10 (par)** | 10 |
| int32 `lt`/`gt`/`le`/`ge` | 9 | 10 (+11%) | **9 (par)** | 9 |
| fp32 `lt`/`gt` | 11 | 19 (+73%) | 12 (+9%) | **11 (par)** |
| fp32 `eq`/`ne` | 14 | 20 (+43%) | 16 (+14%) | **14 (par)** |
| fp32 `le`/`ge` | 13 | 23 (+77%) | 19 (+46%) | **13–14 (par)** |

Read the last two columns together: **there is no case where raw TTI is fundamentally
cheaper than sfpi.** The whole regression is (a) the optimizer not recognising four
idioms, and (b) one absent primitive. Both are fixable inside sfpi.

---

## 5. The seven gaps

### 5.1 Integer compares against a constant cost 4 instructions where 1 suffices — and the fix is a commute

This is the single largest missed optimization, and it appears in six of the eight
kernel families (every NaN guard is `|x| vs +inf`).

`+inf`'s bit pattern does not fit `SFPIADD`'s 12-bit immediate, so it has to live in an
LReg. Written the obvious way:

```cpp
vInt inf_bits = 0x7F800000;   // hoisted outside the loop
...
v_if(abs_bits <= inf_bits) { dst_reg[0] = 1.0f; }
v_endif;
```

emits, per DEST row:

```
SFPMOV    L2, L0, 2      # copy the loop invariant, because SFPIADD will clobber it
SFPIADD   L2, L1, 0, 10
SFPSETCC  L2, 0, 2
SFPCOMPC                 # because there is no ">0" CC, `x <= k` becomes !(x > k)
```

Commuting the comparison to `inf_bits >= abs_bits` — the same predicate — emits:

```
SFPIADD   L1, L0, 0, 10
```

**One instruction.** The commuted form picks the operand order in which the invariant
survives, and lands on a CC polarity `SFPIADD` can fuse, so the copy, the `SFPSETCC`
and the `SFPCOMPC` all disappear. This is exactly what the raw-TTI original did by
hand: `TTI_SFPIADD(0, INF, ABS_V, SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | SFPIADD_MOD1_CC_GTE0)`.

Measured effect of this one change: `ltz`/`gtz` **9 → 6** in isolation, and the whole
`ltz` kernel **11 → 8**, i.e. from +38% to exact parity. `lez`/`gez` **12 → 10**, also
exact parity.

**Ask:** canonicalise integer relational compares so that (a) the direction is chosen to
keep read-only operands live, (b) the polarity is chosen from the set `SFPIADD`'s CC
field can express, and (c) `SFPIADD`'s fused CC test is used instead of a following
`SFPSETCC`. Never emit `SFPCOMPC` for a predicate that can be commuted instead.

### 5.2 A predicated result costs 2 extra instructions unless you predicate the *store*

```cpp
vFloat result = 0.0f;
v_if(cond) { result = 1.0f; }
v_endif;
dst_reg[0] = result;
```

emits `SFPSETCC; SFPMOV L0,L9,2; SFPMOV L0,L10,0 # LV:L0; SFPENCC; SFPSTORE` — the
constant is materialised into an allocatable LReg, plus a second `SFPMOV` for the
liveness merge (the `# LV:` annotation).

The equivalent, storing under the predicate:

```cpp
dst_reg[0] = 0.0f;
v_if(cond) { dst_reg[0] = 1.0f; }
v_endif;
```

emits `SFPSTORE L9; SFPSETCC; SFPSTORE L10; SFPENCC` — **two instructions cheaper**, and
it stores straight out of the constant registers `L9`/`L10` with no LReg traffic at all.
So sfpi already knows how to reach the good form; it just will not get there from the
first spelling. On `eqz` this is 8 → 7. (`vConst0`/`vConst1` do not help and are
deprecated — we tried.)

**Ask:** recognise "assign one of two constants under a CC, then store the result" and
lower it to the two-predicated-store form. Failing that, drop the redundant liveness
copy when the predicated assignment's only consumer is a store.

### 5.3 `dst_reg++` emits a separate `TTINCRWC`; the raw kernels folded it into the store

`impl_::DstRegFile::operator++` unconditionally emits `TTINCRWC 0, 2, 0, 0`
(`sfpi_classes.h:400`), and `dst_reg[k] = v` always stores with `addr_mode` 7 (no
increment). The raw-TTI kernels instead put the dest advance in the final store's
address mod, which is free.

This is the **entire** int32 regression: the sfpi and raw-TTI bodies are
instruction-for-instruction identical, and sfpi is 10 vs 9 solely because of the
`TTINCRWC`. Writing the store as `dst_reg[k].mode<DataLayout::I32>(6) = v` and dropping
the `dst_reg++` gets it to 9 — parity.

Two problems with that workaround. First, nobody found it: it is one line of comment in
`sfpi.h` and the addr_mod constants are not in `sfpi_constants.h`. Second, `6` is
`ADDR_MOD_6`, a *metal* convention programmed by the kernels' `*_init()` functions with
`dest.incr = 2`; passing it through `mode()` leaks a metal register number into what
should be an sfpi-level concern, and silently produces wrong addressing if the caller
did not program that addr_mod.

**Ask:** when a loop body's last DEST access is a store immediately followed by
`dst_reg++`, fold the increment into that store's `addr_mode`. This is the
cheapest fix on the list and it fixes an entire kernel family on its own.

### 5.4 `&&` falls off a cliff as soon as one term needs a helper instruction

A three-term `&&` where every term lowers to exactly one CC-setting instruction chains
perfectly — 9 instructions per row, no `SFPPUSHC` in sight. Change one term so it needs
an `SFPLOADI` or `SFPMOV` first, and the same predicate jumps to **21**:

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

sfpi abandons CC chaining entirely and materialises the boolean into an LReg — about
eight wasted instructions. Hoisting the constant out of the loop does not help; the
helper instruction becomes an `SFPMOV` instead of an `SFPLOADI` and the cliff is the
same (21 either way).

Spelling the identical predicate as nested `v_if`/`v_endif` gets it back to **15**. So
the cheap lowering exists and is reachable — `&&` just does not reach it.

**Ask:** lower `A && B && C` the way nested `v_if` does: hoist each term's helper
instructions ahead of the chain and chain the `SFPSETCC`s. The current behaviour means
`&&` is a performance trap whose cost depends on whether an unrelated constant happened
to fit an immediate field.

### 5.5 No total-order float compare reachable from `v_if` — the one genuine feature gap

`SFPGT`/`SFPLE` compare sign-magnitude bit patterns directly, giving the total order over
finite values, ±0 and ±inf in one instruction with no arithmetic. sfpi never emits them
for a `v_if`. Instead:

| sfpi source | emitted | count | `SFPGT`/`SFPLE` |
|---|---|---|---|
| `v_if(a < b)`, `vFloat` | `SFPMAD` + `SFPSETCC(LT0)` | 2 | 1 |
| `v_if(a > b)`, `vFloat` | `SFPMAD` + `SFPSETCC(GTE0)` + `SFPSETCC(NE0)` | 3 | 1 |

The instructions exist in the compiler as `__builtin_rvtt_sfpgt` / `__builtin_rvtt_sfple`
(`tensix_builtins.def:123–126`), and calling one directly does emit
`SFPGT L0, L1, 0, 8`. But they only produce a *vector* result (`SET_VD`, a 0/−1 lane
mask), and `sfpi::vBool` is constructible only from `sfpxfcmpv`/`sfpxfcmps`/
`sfpxicmps`/`sfpxicmpv` — all of which the backend lowers to arithmetic + `SFPSETCC`.
There is no way to get a raw `SFPGT` result into `v_if`. Round-tripping through the mask
(`v_if(mask != 0)`) costs the `SFPSETCC` back, so it breaks even.

**The semantic difference is the expensive part.** Subtract-then-test-sign is not a total
order, and every consequence had to be worked around in source:

- `inf − inf = NaN` with a clear sign bit, so `inf == inf` answered "unordered" and
  `inf <= inf` answered false. Fixed in the PR by adding an explicit bitwise-equality
  clause — `SFPIADD` + `SFPSETCC` on `eq`/`ne` (+2), and `SFPIADD` + `SFPSETCC` +
  `SFPSTORE` + `SFPENCC` on `le`/`ge` (+4). Pure workaround cost; the raw-TTI
  `SFPGT`/`SFPLE` sequences needed none of it.
- `a − b` underflows into the flushed denormal range, so operands differing by less than
  2⁻¹²⁶ compare equal. Adjacent normals near the bottom of the exponent range are
  reachable. This one is **still not fixed** — it is pinned by an `xfail`
  (`test_binary_comp_fp32_denormal_window_ties`) and is a behaviour regression against
  raw `SFPGT`/`SFPLE`, which were exact there. We cannot close it in source at all.

Measured: with a total-order compare available to `v_if`, `le`/`ge` goes 19 → 13–14
(2 from the compare, 4 from dropping the inf-tie clause) and `eq`/`ne` goes 16 → 14 —
both to raw-TTI parity.

**Ask:** lower `vBool(Cond, vFloat, vFloat)` to `SFPGT`/`SFPLE` with `SET_CC`, or add an
explicit spelling (`total_order_lt(a, b)`, or `vBool(Cond, vSMag, vSMag)` once §5.7 is
fixed) that `v_if` accepts. This is the only item on the list that needs new API surface,
and it is the only item that also fixes a correctness gap we currently cannot close.

### 5.6 `vInt` and `vUInt` relational compares are wrong over the full range

`v_if(a < b)` on `vInt` emits a single `SFPIADD L0, L1, 0, 2` — a two's-complement
subtract with the CC taken from the sign of the difference. That wraps:

- **`vInt`**: `INT32_MAX − (−1) = 0x80000000`, sign set, so `INT32_MAX > -1` answers
  **false**.
- **`vUInt`** (`m_uint32_lt` in the repro): `0u − 0xC0000000u = 0x40000000`, sign clear,
  so `0u < 0xC0000000u` answers **false**.

This is not a theoretical concern. It is the bug #27829 and #28397 originally fixed, it
is what #51097 rolled back, and it re-broke `ttnn.lt`/`ttnn.gt` on int32 the moment
#52932 wrote the compare idiomatically. `calculate_binary_comp_int32` therefore carries a
hand-rolled 6-instruction branchless sign-fold plus a "do NOT write this as `v_if(a < b)`"
comment, and `calculate_binary_comp_uint` was left in raw TTI entirely for the same
reason.

Note the shape of the trap: the *incorrect* idiomatic form is 8 instructions and the
*correct* hand-rolled form is 9. sfpi's compare is cheap and wrong, so the fast path is
the broken one and there is no diagnostic.

**Ask:** lower integer relational compares overflow-safely — `SFPGT`/`SFPLE` on a
sign-flipped operand, or the sign-fold, whichever the backend prefers — or, at minimum,
reject/warn on the unsafe form and expose an explicit full-range compare. Silently
answering `INT32_MAX > -1` as false is the worst of the three options.

### 5.7 `vSMag` and `DataLayout::SM32` compares lower to a two's-complement subtract — likely a bug

`vSMag` is sfpi's sign-magnitude type and `SFPXCMP_MOD1_TYPE_SMAG` exists in the mod
encoding, so we expected sign-magnitude compares to be exactly where `SFPGT`/`SFPLE`
would show up. They are not: `vBool(vBool::LT, vSMag, vSMag)` emits
`SFPIADD L0, L1, 0, 2`, the same two's-complement subtract as `vInt`. Loading DEST as
`DataLayout::SM32` produces byte-identical code (the compiler tail-merged the two
functions).

A two's-complement subtract cannot order sign-magnitude values. Counterexample:
smag `−1` is `0x80000001` and smag `−2` is `0x80000002`; `0x80000001 − 0x80000002 = −1`,
sign set, so the compare answers `−1 < −2`, which is wrong.

**Ask:** please confirm whether this is a lowering bug. If `vSMag` compares were lowered
to `SFPGT`/`SFPLE`, §5.5 would largely be solved as a side effect — `vSMag` is the
natural type for a total-order float compare.

---

## 6. Why this matters beyond one PR

We would like to convert the SFPU kernels to sfpi. It is more readable, more portable
across architectures, and far easier to review than hand-allocated LReg sequences. The
blocker is not that sfpi is fundamentally slower — §4 shows raw TTI has no inherent
advantage in any of these eight kernels.

The blocker is that hitting sfpi's own best codegen currently requires knowing four
undocumented rewrites, none of which a reviewer would flag and none of which have a
diagnostic. A conversion written the way the documentation suggests is 30–90% slower,
which means every conversion needs a full CI perf run to discover it regressed, and
several rounds of disassembly to discover why. That is the loop we just spent a PR on.

Asks 1–4 would collapse most of that. They need no new API and no new ISA surface — they
make the natural spelling as fast as the tuned one. Ask 5 is the only new primitive, and
it is also the only way to close a correctness gap that is currently `xfail`'d in `main`.

---

## 7. Repro

The full harness — every probe quoted above, plus a runner that locates the toolchain
and prints the whole table — is on branch
[`ldjurovic/sfpi-perf-gap-investigation`](https://github.com/tenstorrent/tt-metal/tree/ldjurovic/sfpi-perf-gap-investigation/sfpi_perf_investigation)
under `sfpi_perf_investigation/`:

```sh
git fetch origin ldjurovic/sfpi-perf-gap-investigation
git checkout ldjurovic/sfpi-perf-gap-investigation
./sfpi_perf_investigation/run.sh
```

If you would rather not check anything out, the two files below are self-contained and
reproduce the four claims that carry the argument.

`shim.h`:

```cpp
#pragma once
#include <cstdint>

namespace ckernel {
extern volatile std::uint32_t instrn_buffer[];
}

#include "sfpi.h"
```

`repro.cc`:

```cpp
// Minimal repro for the four claims that carry the argument in ISSUE.md.
// The 0*.cc probes cover the full matrix; start here.
#include <cstdint>
#include "shim.h"

using namespace sfpi;

constexpr int AM6 = 6;  // metal ADDR_MOD_6: dest.incr = 2

// Idiomatic. This is the shape PR #52932 shipped.  -> 14 instr/row
extern "C" void ltz_idiomatic() {
    vInt inf_bits = 0x7F800000;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        vFloat result = 0.0f;
        v_if(v < 0.0f && abs_bits != 0) { result = 1.0f; }
        v_endif;
        v_if(abs_bits > inf_bits) { result = 0.0f; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

// Same algorithm with all four workarounds applied: predicated store (ISSUE 5.2),
// addr_mode fold (5.3), nested v_if (5.4), commuted compare (5.1).
// -> 8 instr/row, i.e. exact parity with the raw TTI this replaces.
extern "C" void ltz_tuned() {
    vInt inf_bits = 0x7F800000;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(v < 0.0f) {
            v_if(abs_bits != 0) {
                v_if(inf_bits >= abs_bits) { dst_reg[0].mode<DataLayout::Default>(AM6) = 1.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}

// ISSUE 5.6: emits a bare `SFPIADD L0, L1, 0, 2`, a two's-complement subtract
// with the condition code taken from the sign, so INT32_MAX > -1 answers false.
extern "C" void int32_lt_wrong(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vInt a = dst_reg[i0 * 32].mode<DataLayout::I32>();
        vInt b = dst_reg[i1 * 32].mode<DataLayout::I32>();
        vInt r = 0;
        v_if(a < b) { r = 1; }
        v_endif;
        dst_reg[io * 32].mode<DataLayout::I32>() = r;
        dst_reg++;
    }
}

// ISSUE 5.5: SFPGT is reachable as a builtin and does emit `SFPGT L0, L1, 0, 8`,
// but only as a 0/-1 lane mask. sfpi::vBool cannot be built from it, so v_if can
// never consume a total-order compare.
extern "C" void sfpgt_reachable(std::uint32_t i0, std::uint32_t i1, std::uint32_t io) {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[i0 * 32];
        vFloat b = dst_reg[i1 * 32];
        vUInt mask = as<vUInt>(vFloat(__builtin_rvtt_sfpgt(a.get(), b.get(), 8)));
        dst_reg[io * 32].mode<DataLayout::U32>() = mask >> 31;
        dst_reg++;
    }
}
```

```sh
SFPI=/path/to/tt-metal/runtime/sfpi
$SFPI/compiler/bin/riscv-tt-elf-g++ -O2 -mcpu=tt-bh-tensix \
    -I$SFPI/include -std=c++17 -S -o repro.s repro.cc
```

Per-DEST-row costs quoted in §4 come from simulating the replay buffer over the emitted
stream: a `TTREPLAY start,len,_,1` records and executes `len` Tensix instructions, a
`TTREPLAY start,len,0,0` re-executes them; total executed ÷ 8 unrolled iterations.

Kernels under discussion, on `main`:
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h`,
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_comp.h`.
The attempted conversion is on branch `ldjurovic/bh_sfpu_complex_reland`.
