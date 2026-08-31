# SFPU kernels: accuracy-neutral instruction-count experiments

Four ways to cut SFPU instruction count **without changing numerical results** — three of them
bit-exact, one neutral on the ordinary domain. Sections 0 and 5 record which have been
implemented and measured on hardware.

References below to "the companion document" mean a separate audit of literal *encoding* —
shaving one `SFPLOADI` off a constant by moving its value onto the bf16 or fp16a grid, which
always costs a little accuracy. That audit is not checked in; the ideas here are independent of
it and cost no accuracy at all.

**Method.** Static numbers were produced by compiling the expression with the repo-pinned toolchain and counting emitted SFPU instructions — no device involved. **Ideas §1 and §3 have since been implemented and measured on Wormhole n300 hardware** via the tt-llk perf suite; see §0. Where a static prediction and a hardware measurement disagree, the hardware number is authoritative and the static model is corrected in place. Toolchain `tt_metal/tt-llk/tests/sfpi` 7.72.0 build 873; flags copied from `TestConfig.setup_compilation_options` (`-O3 -std=c++17 -ffast-math -fno-finite-math-only -fsigned-zeros -fno-associative-math -fno-exceptions -fno-rtti`); `-mcpu=tt-wh-tensix`, `-mcpu=tt-bh-tensix`, `-mcpu=tt-qsr32-tensix`. "slots/element" counts `SFP*` plus `TTINCRWC`/`TTREPLAY` inside the per-element replayed body, which is what actually consumes issue bandwidth.

**Why per-element counts are the right metric.** The compiler records the loop body into the replay buffer and re-issues it per element (`TTREPLAY`), so anything inside the body — including loop-invariant constant loads — costs on every element. Nothing is hoisted. See §6 of the companion document for the evidence.

## Headline

Measured on one realistic kernel shape (exp-style 5-term Horner, `r = r*x + c` ×4), per element:

| variant | WH | BH | accuracy change |
|---|---|---|---|
| as commonly written today (fp32 literals, 1 elem/body) | 18.0 | 14.0 | — |
| **§1** three coefficients in the program CREGs | 13.0 (−28 %) | 8.0 (−43 %) | **none, bit-exact** |
| **§1 + §2** plus 2 elements per iteration | **8.5 (−53 %)** | **7.5 (−46 %)** | **none, bit-exact** |

And on a real shipped kernel, `_relu_max_impl_` (§3): **WH 13 → 6 slots/element (−54 %), BH 12 → 6 (−50 %)**.

| # | idea | typical saving | neutrality |
|---|---|---|---|
| 1 | Park hot fp32 constants in `vConstFloatPrgm0/1/2` | **1–2 slots** per constant, per element — see §1's latency caveat | **bit-exact** |
| 2 | Two elements per loop iteration, hand-interleaved | **−18.3 %** on `i0` on top of §1 *(hardware)* | **bit-exact** (reordering only) |
| 3 | Branch-free idioms instead of `v_if` predication | 2–9 slots per site | bit-exact, NaN and −0.0 included — the SFPU compare is the *same* sign-magnitude total order as `SFPSWAP` (§3) |
| 4 | Reclaiming LREG11 as a 4th constant | 2 slots per element | **unsafe on WH — documented as a trap** |

---

## 0. Implementation status — what landed, and what it actually measured

Three kernels changed on branch `ldjurovic/sfpu_instr_count_reduction`, commit `209a89138fc`.
Measured with the tt-llk perf suite on **Wormhole n300**: `MATH_ISOLATE`, Float16_b→Float16_b,
`tile_cnt=8`, `loop_factor=16`, `iterations=32`, ELF cache wiped between variants, each op
A/B'd against `main`.

| kernel | idea | `KERNEL` before → after | delta | `INIT` |
|---|---|---|---|---|
| `_relu_max_body_` / `_relu_max_impl_` (LLK) | §3 branch-free clamp | 62850 → **38205** | **−39.2 %** | 245 → 239 |
| `ckernel_sfpu_i0.h` (MTL) | §1 program CREGs | 180856 → 168760 | −6.7 % | 240 → 257 |
| `ckernel_sfpu_i0.h` (MTL) | **§2 2-way interleave**, on top of §1 | 168760 → **137858** | **−18.3 %** | 257 → 256 |
| `ckernel_sfpu_softplus.h` (MTL) | §1 program CREGs | 198231 → **185561** | **−6.4 %** | 239 → 257 |

`i0` cumulative against `main`: 180856 → **137858, −23.8 %**. §1 and §2 compound on it exactly as
predicted — §1 converts coefficient loads into `SFPNOP`s, §2 fills them.

`TILE_LOOP` tracks `KERNEL` to within 0.1 % in all three. For `relu_max`, `L1_TO_L1` also fell
63720 → 38935, so the win is not confined to the isolated math thread. Static check on the real
math ELF: `relu_max` 14 → 8 SFPU instructions, `SFPSETCC`/`SFPENCC` gone.

**Correctness:** 527 passed / 9 skipped in `test_eltwise_unary_sfpu.py` for
`mathop ∈ {I0, Softplus, ReluMax, Hardsigmoid, Silu}`, including the fp32 `_edges`
variants that feed NaN, ±inf and −0.0. The skips are the pre-existing ReluMin gate
(tt-llk#1120), not a consequence of these changes.

### Four assumptions in this document that were wrong

1. **§3's clamp rewrite had the operand order backwards.** Corrected in §3.
2. **§1 saves ~1 slot per coefficient in a long dependent chain, not 2.** Corrected in §1.
   The loads were doing useful work — filling `SFPMAD` latency.
3. **Whole-ELF static instruction counts cannot measure §1 at all.** Corrected in §6.
   The loads move from the body into init, so the total does not change.
4. **§2 does not work by restructuring the loop — the interleave must be written out
   statement by statement.** GCC will not interleave two independent expression trees.
   Corrected in §2, and it is the difference between −18 % and nothing.

### Not implemented, and why

- **`tanh_derivative`** — `tanh_derivative_init` already programs `l_reg[LReg0/1/2]` for the
  tanh LUT, and there are two init variants (`tanh_derivative_init`,
  `tanh_derivative_sech2_init`). Establishing which one the sech² polynomial path actually
  runs needs more validation than the ~3-slot payoff justifies.
- **`logsigmoid`, `binary_pow`, LLK `cdf` / `silu` / `expm1_cw`** — not reachable from the
  tt-llk unary perf sweep (`Logsigmoid` and `BinaryPow` are not even `MathOperation`
  members), so they can be neither measured nor regression-tested in this harness. §1's
  candidate table now flags measurability per kernel.
- **§2 on `softplus`** — the same rewrite `i0` got. It still carries 6 `SFPNOP` per element after
  §1. Same `PolynomialEvaluator::eval` shape, so the same statement-level interleave applies.
- **§2 on the kernels ranked in §2's survey table** — `erfinv` (21 % of its SFPU instructions are
  `SFPNOP`), `acosh`, `asin`, `asinh` are the densest remaining.

---

## 1. Park hot fp32 constants in the program constant registers

### Mechanism

`SFPMAD`, `SFPMUL` and `SFPADD` can name a constant register directly in an operand field. So a constant that lives in a CREG costs **zero** instructions to use — and, unlike the fp16a/bf16 rewrites in the companion document, it keeps its exact fp32 bit pattern. There is no accuracy question to answer.

sfpi exposes three programmable slots — `vConstFloatPrgm0/1/2` → CREG 12/13/14 (`sfpi_constants.h:316-318`) — plus the hardwired `0.0f` (CREG 9), `1.0f` (CREG 10), `-1.0f` (CREG 11) and `0.8373f` (CREG 8).

### Probe

```cpp
#define P0 vConstFloatPrgm0
#define P1 vConstFloatPrgm1
#define P2 vConstFloatPrgm2

// (a) as commonly written: coefficients are fp32 literals
void lit() {
#pragma GCC unroll 8
  for (int d=0; d<8; d++) { vFloat x = dst_reg[0];
    vFloat r = 8.37312452e-3f;
    r = r*x + 4.16695364e-2f;
    r = r*x + 1.66664720e-1f;
    r = r*x + 0.5f;
    r = r*x + 1.0f;
    dst_reg[0] = r; dst_reg++; }
}
// (b) same three fp32 coefficients parked at init; 0.5 folds to SFPADDI, 1.0 is CREG 10
void prgm_init() { vConstFloatPrgm0 = 8.37312452e-3f;
                   vConstFloatPrgm1 = 4.16695364e-2f;
                   vConstFloatPrgm2 = 1.66664720e-1f; }
void prgm() {
#pragma GCC unroll 8
  for (int d=0; d<8; d++) { vFloat x = dst_reg[0];
    vFloat r = P0;
    r = r*x + P1;
    r = r*x + P2;
    r = r*x + 0.5f;
    r = r*x + 1.0f;
    dst_reg[0] = r; dst_reg++; }
}
```

### Result

| | WH | BH |
|---|---|---|
| (a) fp32 literals | 18 slots/elem, 6 `SFPLOADI`, 4 `SFPNOP` | 14 slots/elem, 6 `SFPLOADI` |
| (b) program CREGs | **13 slots/elem, 0 `SFPLOADI`**, 5 `SFPNOP` | **8 slots/elem, 0 `SFPLOADI`** |

All six `SFPLOADI` disappear. On a simpler 3-coefficient chain the effect is even starker: WH 12 → 7, BH 11 → 5.

Emitted body for (b), showing the CREGs used inline as MAD operands (`L12`/`L13`/`L14` are Prgm0/1/2, `L10` is the hardwired 1.0):

```
	SFPLOAD	L1, 0, 0, 3
	SFPMAD	L0, L12, L1, L13, 0      <- r = P0*x + P1, no loads
	SFPNOP
	SFPMAD	L0, L0, L1, L14, 0       <- r = r*x + P2
	SFPNOP
	SFPMUL	L0, L0, L1, 0
	SFPNOP
	SFPADDI	L0, 16128, 0             <- + 0.5f folds to an immediate
	SFPMAD	L0, L0, L1, L10, 0       <- + 1.0f from CREG 10
	SFPSTORE	L0, 0, 0, 3
	TTINCRWC	0, 2, 0, 0
```

### Correction (hardware): the loads were filling `SFPMAD` latency

The probe above predicts −5 slots for three parked coefficients (six `SFPLOADI` removed, one
`SFPNOP` gained). On the real kernels the gain per coefficient is closer to **1 slot, not 2**,
and this is why.

Disassembling `softplus` as shipped on `main`, the Horner chain looks like this — note there is
**no `SFPNOP` anywhere in it**:

```
	sfploadi	L0,11169,2
	sfploadi	L0,-18429,8
	sfpmad	L0,L2,L0,L3,0
	sfploadi	L3,10480,2
	sfploadi	L3,-17676,8
	sfpmad	L0,L2,L0,L3,0
	sfploadi	L3,103,2
	sfploadi	L3,-17323,8
	sfpmad	L0,L2,L0,L3,0
	...
```

Each dependent `SFPMAD` needs a latency slot before its result can be consumed, and on Wormhole
the *next* coefficient's two `SFPLOADI` land exactly there. **The coefficient loads are free —
they are filling a slot that would otherwise be an `SFPNOP`.** Draining them into CREGs does not
remove three instructions per step, it converts them into one exposed `SFPNOP`:

| | per Horner step, WH | `SFPNOP` in body |
|---|---|---|
| fp32 literal coefficient | `LOADI` + `LOADI` + `MAD` = 3 slots | 0 |
| CREG coefficient | `MAD` + `SFPNOP` = 2 slots | 1 |

Measured on the real kernels: parking three coefficients added exactly three `SFPNOP`
(`softplus` 3 → 6, `i0` 6 → 9), for a net **−3 slots per element**, which is the −6.4 % / −6.7 %
seen on hardware.

So the saving per parked coefficient is **2 minus whatever NOP it exposes**, i.e. between 1 and
2 slots. Which end you land on depends on whether the scheduler has other independent work to
fill the slot:

- **Short chains with free constants** (the probe: five terms, two of them `0.5f`/`1.0f`) have
  independent work available → closer to 2 slots each (−5 for three coefficients, −28 %).
- **Long, purely dependent chains** (`softplus` degree-6, `i0` degree-10) have none → 1 slot
  each (−3 for three coefficients, ≈ −6.5 %).

**This makes §2 the natural partner, not an alternative.** Interleaving two elements per body
supplies exactly the independent work these exposed NOPs need. §1 alone converts loads into
NOPs; §1 + §2 removes both.

Two practical consequences:

- Do not park coefficients *and* stop there in a long dependent chain expecting −2 each.
- Do not park *every* coefficient. The optimum leaves enough loads in the chain to cover the
  MAD latency; beyond that point each extra CREG buys a NOP instead of a saving. `i0`'s
  implementation deliberately parks three of eleven for this reason.

### Cost

Writing a Prgm register costs 3 instructions once (`SFPLOADI` ×2 + `SFPCONFIG`), so 9 for all three, executed once in the kernel's init hook rather than per element:

```
prgm_init:  SFPLOADI SFPLOADI SFPCONFIG  SFPLOADI SFPLOADI SFPCONFIG  SFPLOADI SFPLOADI SFPCONFIG
```

Break-even is under two elements. Irrelevant at tile scale.

### Which kernels actually have slots free

This is the part that needs care: availability is **transitive**. A kernel's init hook often calls a shared helper that claims the CREGs without the kernel file mentioning them. `digamma_init` documents exactly this:

```cpp
// ckernel_sfpu_digamma.h:88
void digamma_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // sfpu_reciprocal_init programs vConstFloatPrgm0/1/2 with the reciprocal's Newton seed;
    // all three stay reserved for it, so digamma must not repurpose Prgm1/Prgm2.
    sfpu_reciprocal_init();
}
```

This bit me twice, in two different ways, and both are worth recording because a naive scan
gets the central table of this document wrong:

1. A per-file grep for `vConstFloatPrgm` reports `erf`, `i1`, `digamma` and `gelu` as having
   three free slots. They have **zero** — all four reach `sfpu_reciprocal_init`.
2. Resolving init calls transitively is not enough on its own if the parser only recognises
   `inline` function definitions. `void i1_init() {` and `void erf_init() {` are declared
   without `inline`, so they were skipped, the recursion never ran, and those kernels came
   back as "free" again — this time with an authoritative-looking transitive analysis behind
   the wrong answer. Fixing the parser also moved `lgamma` and `exp` from "free" to "claimed".

`prgm_avail.py` now unions direct writes with a transitive walk over all definition forms.
Ground truth it should reproduce: `erf`, `i1`, `digamma`, `gelu`, `erfc`, `lgamma`,
`trigonometry` and `exp` all report **0 free**. Of 142 kernel headers, these have free slots
*and* an init hook *and* enough per-element fp32 constants to use them:

`slots/elem recoverable` is the *upper bound* (2 per parked coefficient). Read it together with
the latency correction above: in a long dependent chain expect about half of it. `measurable`
says whether the tt-llk unary perf sweep can drive the op at all — if not, the change can be
neither benchmarked nor regression-tested in this harness.

| kernel | Prgm claimed | free | distinct fp32 consts in hot body | upper bound slots/elem | measurable | init hook | status |
|---|---|---|---|---|---|---|---|
| MTL `softplus` | — | 3 | 25 | 6 (got 3) | yes | `softplus_init` | **done, −6.4 %** |
| MTL `i0` | — | 3 | 9 | 6 (got 3) | yes | `i0_init` | **done, −6.7 %** |
| MTL `tanh_derivative` | — | 3 | 17 | 6 | yes | `tanh_derivative_init`, `tanh_derivative_sech2_init` | deferred — init already programs `l_reg[LReg0/1/2]` for the tanh LUT, and two init variants to disambiguate |
| MTL `logsigmoid` | — | 3 | 9 | 6 | **no** — not a `MathOperation` member | `logsigmoid_init` | blocked on harness coverage |
| MTL `binary_pow` | 0,1 | 1 | 13 | 2 | **no** — not a `MathOperation` member | `sfpu_binary_pow_init` | blocked on harness coverage |

A further four have three free slots but no init hook of their own, so they would need one
plumbed before this applies, and none is reachable from the unary perf sweep: LLK `expm1_cw`
(11 constants), LLK `cdf` (7), LLK `silu` (6), MTL `piecewise_rational` (6).

**Verify the init hook actually runs on the path you are optimising.** For `softplus` and `i0`
it does, but not obviously: the tt-llk harness's `call_unary_sfpu_operation_init()` has no
per-op arm for either, and falls through to
`llk_math_eltwise_unary_sfpu_init<OPERATION, is_fp32_dest_acc_en>()`, which runs
`_llk_math_eltwise_unary_sfpu_init_` — and *that* is the `switch` on `SfpuType` that calls
`sfpu::softplus_init()` / `sfpu::i0_init()`. A missing init would not fail to compile; it would
silently read whatever the CREGs last held.

`softplus_init` is the clearest case — it is currently nothing but a counter reset:

```cpp
// ckernel_sfpu_softplus.h:34
inline void softplus_init() { math::reset_counters(p_setrwc::SET_ABD_F); }
```

25 distinct fp32 constants in its per-element body, three CREGs idle, and a dispatched init hook already in place. Six slots per element, bit-exact.

Full per-kernel availability table in Appendix A.

### Caveats

- **Pick the three most-executed constants**, not the first three. A constant used only inside a rarely-taken `v_if` branch is worth less than one on the main path. The counts above are *distinct constants*, not dynamic frequency — that ranking still needs doing per kernel.
- **Fused/back-to-back SFPU ops.** If two SFPU ops that both program the CREGs are chained without re-running init between them, the second will read the first's values. This is the failure mode `sfpu_reciprocal_init` is guarding against. Any change here must confirm the init hook actually runs on the path being optimised.
- **`lgamma` looks like a candidate and is not.** It has 18 fp32 constants and its own file never mentions `vConstFloatPrgm`, but `lgamma_stirling_init` reaches the log/reciprocal init, which claims all three. It also calls `_calculate_log_body_no_init_` — the no-init log variant — deliberately. Leave it alone; it is the best example of why the transitive check matters.

---

## 2. Two elements per loop iteration, hand-interleaved

### Mechanism

On Wormhole a dependent `SFPMAD` must be followed by `SFPNOP` before its result can be consumed. A Horner chain is maximally dependent, so it pays that tax on every step. Two *independent* element chains in the same body give the scheduler something to put in those slots. It also halves the `TTINCRWC` and loop overhead, which is why Blackhole — which has no NOP tax — still gains a little.

The section title said "replayed body" in an earlier draft. That describes the probes, which use `TTREPLAY`; the real kernels this was applied to do not (see limit 2 below). The idea is the same either way — two elements per loop iteration.

Several kernels already interleave *within* one element for this reason, e.g. `ckernel_sfpu_tanh.h:48`: `s = f * f;  // hide SFPMAD latency`. What is missing is interleaving *across* elements.

### Probe

```cpp
// (a) one element per replayed body — today's shape
void seq1() {
#pragma GCC unroll 8
  for (int d=0; d<8; d++) { vFloat x = dst_reg[0];
    vFloat r = P0; r = r*x + P1; r = r*x + P2;
    dst_reg[0] = r; dst_reg++; }
}
// (b) two elements per body — independent chains
void ilv2() {
#pragma GCC unroll 4
  for (int d=0; d<4; d++) {
    vFloat x0 = dst_reg[0], x1 = dst_reg[1];
    vFloat r0 = P0,          r1 = P0;
    r0 = r0*x0 + P1;   r1 = r1*x1 + P1;
    r0 = r0*x0 + P2;   r1 = r1*x1 + P2;
    dst_reg[0] = r0; dst_reg[1] = r1; dst_reg += 2; }
}
```

### Probe result — 3-coefficient chain, constants in CREGs

| | body | per element | `SFPNOP` |
|---|---|---|---|
| WH, 1 elem/body | 7 | 7.00 | 2 |
| WH, 2 elem/body | 9 | **4.50 (−36 %)** | **0** |
| BH, 1 elem/body | 5 | 5.00 | 0 |
| BH, 2 elem/body | 9 | **4.50 (−10 %)** | 0 |

Every NOP is gone on WH. The emitted body shows why — the two chains' MADs alternate:

```
	SFPLOAD	L3, 0, 0, 3
	SFPLOAD	L2, 2, 0, 3
	SFPMAD	L1, L12, L3, L13, 0
	SFPMAD	L0, L12, L2, L13, 0     <- fills what was a SFPNOP
	SFPMAD	L1, L1, L3, L14, 0
	SFPMAD	L0, L0, L2, L14, 0      <- fills what was a SFPNOP
	SFPSTORE	L1, 0, 0, 3
	SFPSTORE	L0, 2, 0, 3
	TTINCRWC	0, 4, 0, 0              <- one increment for two elements
```

### Probe result — 5-coefficient chain (the exp-style shape)

| | WH | BH |
|---|---|---|
| CREG constants, 1 elem/body | 13.0 | 8.0 |
| CREG constants, 2 elem/body | **8.5 (−35 %)** | **7.5 (−6 %)** |

### Ordering matters: do §1 first

With the coefficients still as **fp32 literals**, interleaving duplicates the loads, and most of the benefit evaporates:

| 5-coefficient chain, fp32 literals | WH | BH |
|---|---|---|
| 1 elem/body | 18.0 | 14.0 |
| 2 elem/body | 14.5 (−19 %) | 13.5 (−4 %) — and 12 `SFPLOADI` instead of 6 |

So §1 and §2 compound, but only in that order. Interleaving a literal-heavy body mostly buys duplicated `SFPLOADI`.

### The interleave must be written out statement by statement

**This is the whole ballgame, and it is not obvious.** GCC will **not** interleave two independent
expression trees. Handing it two adjacent elements and letting it schedule does nothing:

```cpp
// DOES NOT WORK -- measured: 18 SFPNOP per two elements, i.e. 9 per element, unchanged.
dst_reg[0] = I0_SERIES(x0);
dst_reg[1] = I0_SERIES(x1);
```

That form emitted the two chains back to back and left every stall in place; only `TTINCRWC`
amortised, for 37.5 slots/element against 38.0 — nothing. The win appears only when the two
chains advance in lockstep in the *source*:

```cpp
// WORKS -- SFPNOP 9 -> 3 per element.
#define I0_STEP2(c)      \
    do {                 \
        r0 = r0 * t0 + (c); \
        r1 = r1 * t1 + (c); \
    } while (0)

vFloat r0 = C_hi, r1 = C_hi;
I0_STEP2(C9); I0_STEP2(C8); ... I0_STEP2(C0);
```

So §2 is not a loop-restructuring change that can be applied mechanically — it is a rewrite of the
kernel's polynomial evaluation, per kernel. A `POLYVAL`/`PolynomialEvaluator::eval` macro call has
to be expanded into explicit alternating steps. Budget accordingly.

The earlier probe in this section happened to be hand-alternated, which is why this constraint went
unnoticed until `i0` was actually implemented.

### Result on a real kernel: `i0`

| | slots/element | `SFPNOP`/element |
|---|---|---|
| after §1 (CREGs), 1 element/iteration | 38.0 | 9 |
| naive 2-way (two `POLYVAL10` calls) | 37.5 | 9 |
| **2-way, hand-interleaved** | **31.5** | **3** |

Hardware: `MATH_ISOLATE` `KERNEL` 168760 → **137858, −18.3 %**; `TILE_LOOP` −18.4 %. Bit-exact,
14/14 correctness tests including the fp32 `_edges` variants.

The 9 stalls came from exactly where §1's correction predicts — the back half of the chain, where
the coefficients are CREGs or bf16-exact immediates and there is nothing left to fill the latency:

```
sfploadi / sfploadi / sfpmad   x5   <- fp32 literal loads fill the slot, no NOP
sfpnop / sfpmad L14                 <- CREG operand: nothing to fill with
sfpnop / sfpmad L13
sfpnop / sfpmad L12
sfpnop / sfpmul ... sfpnop / sfpaddi ...
```

### Where else to apply it

`SFPNOP` density in the compiled math ELF is the direct signal. Whole-ELF counts, so read it as a
ranking rather than a per-element figure:

| op | `SFPNOP` | total `sfp*` | NOP share |
|---|---|---|---|
| `erfinv` | 42 | 199 | **21.1 %** |
| `acosh` | 20 | 122 | 16.4 % |
| `asin` | 11 | 73 | 15.1 % |
| `asinh` | 24 | 166 | 14.5 % |
| `atanh` | 10 | 100 | 10.0 % |
| `celu` | 6 | 60 | 10.0 % |
| `exponential` | 3 | 31 | 9.7 % |
| `digamma` | 12 | 137 | 8.8 % |
| `erf` | 5 | 63 | 7.9 % |
| `cosh` | 1 | 50 | 2.0 % |

`softplus` still carries 6 per element after §1 and has the same `PolynomialEvaluator::eval` shape
as `i0`, so it is the cheapest next one.

### Three measured limits

1. **`dst_reg[k]` immediate window is `[-8, 7]` half-rows** — i.e. `dst_reg[-4]` … `dst_reg[3]`. A 4-way interleave with `dst_reg += 4` overflows it once GCC unrolls, failing to compile outright:
   ```
   sfpi_classes.h:404: error: argument 2 '8' is out of range [-8, 7]
   ```
   **2-way is the practical shape.** 4-way needs an extra `TTINCRWC` mid-body, which gives back part of what it won.
2. **Replay body caps at 32 instructions — but check whether replay is in use at all.** In the
   probes (fixed trip count, `#pragma GCC unroll 8`) GCC records the body and emits `TTREPLAY 0,
   32, 1, 1`, splitting into shorter segments beyond 32. **The real kernels do not do this.** `i0`
   carries an explicit `#pragma GCC unroll 0` and compiles to a rolled RISC-V loop with no
   `TTREPLAY` anywhere, so the 32-instruction ceiling never binds and the interleaved 63-slot body
   is fine. Check the disassembly for `TTREPLAY` before treating the cap as a constraint; on a
   rolled loop the binding limit is register pressure alone.
3. **Register pressure — the real limit.** 8 general-purpose LRegs (L0–L7); CREGs occupy 8–15.
   `i0`'s 2-way interleave holds 2 × (`t`, `r`) plus two loads = 6 of the 8, with the coefficients
   in CREGs; a third element spills. Deep chains with several live temporaries may not fit even at
   2-way, and the compiler starts emitting `SFPMOV` spill-shuffles — that is the signal to stop.

### Caveat

This is a pure reordering: same operations, same operands, same order within each element's own
dependency chain. Bit-exact — including the trailing operations a `POLYVAL` macro adds, which must
be carried over faithfully when the macro is expanded by hand.

Two risks, neither numerical. If register pressure forces spills, check the emitted count rather
than assuming a win. And because the polynomial evaluation is being rewritten rather than
mechanically transformed, each application needs its own bit-exactness argument and correctness
run — this is not a sweep.

---

## 3. Branch-free idioms instead of `v_if` predication

### Mechanism

A `v_if` block costs `SFPSETCC` to set the predicate and `SFPENCC` to restore it, plus a load for the guard's constant, plus whatever the body does. Where the whole block is a saturate, a sign transplant or an absolute value, the SFPU has a single instruction for it.

### Probe and result

| pattern | predicated form | slots WH / BH | branch-free form | slots WH / BH | saving |
|---|---|---|---|---|---|
| two-sided clamp | `v_if(x>hi){x=hi;} v_endif; v_if(x<lo){x=lo;} v_endif;` | 15 / 13 | `max(min(x,hi),lo)` — **order matters, see below** | 6 / 6 | **−9 / −7** |
| sign transplant | `v_if(x<0){r=-r;} v_endif;` | 6 / 6 | `copysgn(r,x)` | 4 / 4 | −2 / −2 |
| floor at zero | `v_if(x<0){x=0;} v_endif;` | 5 / 5 | `max(x,0.0f)` | 3 / 3 | −2 / −2 |
| absolute value | `v_if(x<0){x=-x;} v_endif;` | 5 / 5 | `setsgn(x,0)` | 3 / 3 | −2 / −2 |

Emitted forms, showing what replaces the predicate machinery:

```
clamp_vif  (WH, 15): SFPLOAD SFPLOADI SFPMAD SFPNOP SFPSETCC SFPSETCC SFPLOADI SFPENCC
                     SFPLOADI SFPMAD SFPNOP SFPSETCC SFPLOADI SFPENCC SFPSTORE
clamp_mm   (WH,  6): SFPLOAD SFPLOADI SFPSWAP SFPLOADI SFPSWAP SFPSTORE

relu_vif   (WH,  5): SFPLOAD SFPSETCC SFPMOV SFPENCC SFPSTORE
relu_max   (WH,  3): SFPLOAD SFPSWAP SFPSTORE          <- 0.0f is CREG 9, no load at all

abs_vif    (WH,  5): SFPLOAD SFPSETCC SFPMOV SFPENCC SFPSTORE
abs_ss     (WH,  3): SFPLOAD SFPSETSGN SFPSTORE
```

Note the two-sided clamp is worse than it looks in source: the guard `x > 3.0f` costs `SFPLOADI` + `SFPMAD` + `SFPSETCC`, *and* the body's `x = 3.0f` costs another `SFPLOADI` — four loads across the two blocks, versus two for `min`/`max`.

#### The clamp order is not free to choose

An earlier version of this document wrote the rewrite as `min(max(x, lo), hi)`. **That is wrong
whenever the two bounds can cross**, and one of the four real sites has a runtime bound that can.
`min`/`max` compose in the order you write them, so the rewrite has to apply the bounds in the
same order the predicated original did:

```
original:  v_if (x > hi) { x = hi; } v_endif;   // high bound applied FIRST
           v_if (x < lo) { x = lo; } v_endif;   // low bound applied SECOND
rewrite:   x = max(min(x, hi), lo);             // same order, inside out
```

With literal bounds and `lo <= hi` both orders agree, which is why the probe above measures the
same either way. With `lo = 0` and a *runtime* `hi = threshold` that can go negative, they do not:

| x | threshold | original | `min(max(x,0),t)` | `max(min(x,t),0)` |
|---|---|---|---|---|
| −5 | 6 | 0 | 0 | 0 |
| 9 | 6 | 6 | 6 | 6 |
| 3 | −2 | **0** | **−2** ✗ | **0** ✓ |
| −5 | −2 | **0** | **−2** ✗ | **0** ✓ |

`min(max(..))` disagrees with the original on 6 of 24 sampled `(x, threshold)` combinations, all
of them at negative thresholds. `max(min(..))` disagrees on none. The shipped fix uses
`max(min(..))`.

### Case study: `_relu_max_impl_`

`tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h:53-70` is exactly this shape, on one of the hottest ops there is:

```cpp
for (int d = 0; d < iterations; d++) {
    VecType result = sfpi::dst_reg[0];
    v_if (result > threshold) { result = threshold; } v_endif;
    v_if (result < 0)         { result = 0; }         v_endif;
    ...
```

Rewritten as `result = max(min(result, threshold), 0.0f);` (high bound first — see above):

| | WH | BH |
|---|---|---|
| as shipped | 13 slots/elem | 12 |
| `max(min(...))` | **6 (−54 %)** | **6 (−50 %)** |

```
before (WH): SFPLOAD SFPLOAD SFPMAD SFPNOP SFPSETCC SFPSETCC SFPMOV SFPENCC
             SFPSETCC SFPMOV SFPENCC SFPSTORE TTINCRWC
after  (WH): SFPLOAD SFPLOAD SFPSWAP SFPSWAP SFPSTORE TTINCRWC
```

`threshold` is a runtime value here, so none of this is about constant encoding — it is pure
predication overhead. And because it is a runtime value that *can* be negative, this is the site
that forces the `max(min(..))` order.

**Confirmed on hardware:** real math ELF 14 → 8 SFPU instructions, `MATH_ISOLATE` 62850 → 38205
(**−39.2 %**), `L1_TO_L1` 63720 → 38935. The isolated-probe prediction of −54 % overstates it —
the shipped kernel carries threshold plumbing the probe did not model — but the direction and
scale hold.

**Only the `vFloat` path was changed.** `sfpi::min` / `sfpi::max` accept `vFloat` and `vSMag` on
Wormhole and Blackhole; the `vInt` overloads are gated behind `#if __riscv_xtttensixqsr`
(Quasar only), so `max(vInt, 0)` does not even compile the way you would expect — it resolves
through the `vUInt` overload and returns the wrong type. `SFPSWAP` also orders operands as
sign+magnitude, so a genuine `vInt` clamp would need the 2's-complement → sign-magnitude
conversion that `_relu_min_` already does by hand. `_relu_max_impl_` is templated on
`VecType ∈ {vFloat, vInt}`, so the shipped change guards the rewrite with
`if constexpr (std::is_same_v<VecType, sfpi::vFloat>)` and leaves the `vInt` arm predicated.

### Inventory

332 `v_if`/`v_elseif` sites across the WH trees. 128 have the single-assignment shape; of those, 74 map onto a specific branch-free idiom (Appendix B).

The highest-value subset is genuine **two-sided clamp pairs** — adjacent opposite-direction guards on the same variable. There are **four**, all in `relu_max`:

| site | current | collapses to | status |
|---|---|---|---|
| `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h:42-47` | `v_if(result > threshold){...} v_if(result < 0.0f){...}` | `max(min(result, threshold), 0.0f)` | **done** |
| `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h:61-66` | same, templated on `VecType` | `max(min(result, threshold), 0.0f)`, `vFloat` arm only | **done** |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h:125-127` | `v_if(x<0){x=0;} v_if(x>t){x=t;}` | low bound first here: `min(max(x, 0), t)` | not yet |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h:164-166` | `v_if(a>threshold){...} v_if(a<0.0f){...}` | `max(min(a, threshold), 0.0f)` | not yet |

Note the third row applies its bounds in the opposite order from the other three, so it takes
`min(max(..))`. Read the order off the source every time rather than pattern-matching the shape.

My pair detector initially reported six. Two were false positives, worth recording so nobody re-finds them:

- `ckernel_sfpu_relu.h:67-70` (MTL) — the two guards are the two arms of an `if constexpr (IS_LOWER_BOUND) / else`. Only one ever compiles. Not a clamp.
- `ckernel_sfpu_hardtanh.h:34-41` (LLK) — there is a `val += p1;` between the two blocks, so they are sequential operations on different values, not a clamp.

### The NaN caveat, resolved — and why it dissolved

This section previously flagged the edge domain as the one place these rewrites might not be
bit-exact, with `SFPSWAP`'s NaN ordering as the specific unknown. Hardware settled it, and the
reasoning is worth keeping because it generalises to the whole idiom class.

**The SFPU's comparison is not IEEE.** `v_if (x > threshold)` compiles to a compare that orders
operands as a **sign-magnitude total order**, exactly like `SFPSWAP`. A NaN has the maximum
exponent, so as a magnitude it is larger than any finite value — meaning the *predicated original*
already treats `NaN > threshold` as **true** and replaces NaN with the threshold. It never passed
NaN through, contrary to what IEEE semantics would suggest.

The tt-llk golden documents this in `SPECIALS_READY_OPS`:

> `_relu_max_body_`: a total-order `> threshold` replaces a NaN with the threshold, and the relu
> clamp then sees a finite value.

Since `min`/`max` lower to `SFPSWAP` and use the *same* total order, both forms agree on NaN by
construction. `min(NaN, threshold) = threshold`, then `max(threshold, 0) = threshold` — identical
to the predicated path. Signed zero agrees for the same reason.

**Verified, not argued:** the `_edges` variants for `ReluMax` feed exactly
`{0, −0, 5, +inf, −inf, NaN}` on the fp32→fp32 pipeline where specials survive to L1
(`specials_safe(Float32, Float32, ·) == True`), and all of them pass after the rewrite.

So for this op the rewrite is unconditionally bit-exact. The general lesson is not "NaN is fine"
but: **on the SFPU, predicated compares and `SFPSWAP` share one total order, so swapping between
them does not change edge behaviour.** What *would* change it is assuming IEEE semantics in
either direction. Two things still deserve a check per site:

- Ops where the golden defines a *different* non-finite contract than the total order implies —
  check `SPECIALS_READY_OPS` for the op's note before rewriting.
- `setsgn(x,0)` is unconditionally `|x|`, which is a genuine difference from
  `v_if(x<0){x=-x;}` only if the surrounding code depends on a NaN's sign bit surviving.

Validate with `test_eltwise_unary_sfpu.py`'s `_edges` variants on an fp32→fp32 format pair,
which is where specials actually reach L1 — not with a uniform exhaustive bf16 sweep.

---

## 4. A fourth constant register — why not to

There are only three `vConstFloatPrgm` slots, and §1 shows they are worth two slots per element each. So the obvious next question is whether CREG 11 — currently holding the constant −1.0 — can be reclaimed. Hardware-wise, yes:

```cpp
// sfpi_constants.h:304, 315, 319
constexpr unsigned int SFPCONFIG_DEST_LREG11 = 11;
constexpr unsigned int CREG_IDX_PRGM0        = 11;   // <- it IS a programmable slot
constexpr unsigned int CREG_IDX_NEG_1        = CREG_IDX_PRGM0;
```

and `_sfpu_load_config32_(dest, upper16, lower16)` in `ckernel_sfpu_load_config.h:28` will write any LREG including 11.

**Do not do this on Wormhole.** WH has no subtract instruction, so sfpi implements every subtraction as a multiply-add against the −1.0 CREG:

```
// vFloat a - b, -mcpu=tt-wh-tensix
	SFPLOAD	L1, 0, 0, 3
	SFPLOAD	L0, 2, 0, 3
	SFPMAD	L0, L0, L11, L1, 0      <- reads CREG 11, must be exactly -1.0
	SFPNOP
	SFPSTORE	L0, 0, 0, 3
```

`a -= b` is the same. Clobbering LREG11 silently corrupts every subtraction in the kernel — including subtractions the compiler introduces inside sfpi's own helpers, which no amount of reading the kernel source will reveal.

Blackhole does not have this dependency; it has a native `SFPADD` with a negate modifier:

```
// vFloat a - b, -mcpu=tt-bh-tensix
	SFPADD	L0, L0, L1, 1           <- mod 1 = negate second operand, no CREG involved
```

Negation and multiply-by-−1 are safe on both (`SFPMOV` with the negate modifier, no CREG read):

```
	SFPLOAD	L0, 0, 0, 3
	SFPMOV	L0, L0, 1
```

**Verdict:** viable on Blackhole/Quasar only, and only after auditing the kernel for every `-` including those inside inlined helpers. Given §1 still has three genuinely free slots in four kernels with an existing init hook, this is not where to spend risk budget. Recorded here so the idea gets rejected on evidence rather than re-litigated.

---

## 5. Combined effect and suggested sequencing

The three usable ideas compose, and the order is not arbitrary:

```
   fp32 literals, 1 elem/body           WH 18.0   BH 14.0   slots/element
       |
       |  §1  park 3 coefficients in vConstFloatPrgm0/1/2      (bit-exact)
       v
   CREG constants, 1 elem/body          WH 13.0   BH  8.0     -28% / -43%
       |
       |  §2  two elements per replayed body                   (bit-exact)
       v
   CREG constants, 2 elem/body          WH  8.5   BH  7.5     -53% / -46% cumulative
```

§2 before §1 gets you WH 18.0 → 14.5 instead of → 8.5, because interleaving duplicates literal loads. **§1 first.**

That chain is the probe's. On the real kernels the §1 step is smaller than the probe suggests
(§1's latency correction) and the §2 step is correspondingly *larger*, because §1 leaves exposed
`SFPNOP`s for §2 to fill. The composition argument survives; the split between the two steps
moves toward §2.

Remaining order of work (items 1–2 are **done** — see §0):

1. ~~`relu_max` clamp rewrite (§3)~~ — **done, −39.2 % on hardware.** Bit-exact, including NaN.
2. ~~`softplus` and `i0` CREGs (§1)~~ — **done, −6.4 % / −6.7 %.** Bit-exact.
3. ~~2-way interleave (§2) on `i0`~~ — **done, −18.3 % on top of §1** (−23.8 % cumulative). The
   compounding worked as predicted. Note it required expanding `POLYVAL10` into explicit
   alternating steps; see §2.
4. **2-way interleave (§2) on `softplus`** — still 6 `SFPNOP` per element after §1, same
   `PolynomialEvaluator::eval` shape as `i0`. Cheapest next win.
5. **2-way interleave on the densest remaining kernels** — `erfinv` (21 % NOP share), `acosh`,
   `asin`, `asinh`; see §2's survey table.
6. **`tanh_derivative` CREGs** (§1) — needs the `l_reg[LReg0/1/2]` LUT interaction and the two
   init variants resolved first. ~3 slots/element once that is understood.
7. **`relu.h:125-127` and `:164-166`** (§3) — the two remaining two-sided clamp pairs, in the
   Metal tree. Mind the bound order: `:125` is low-bound-first and takes `min(max(..))`.
8. **`logsigmoid`, `binary_pow`** (§1) — blocked on harness coverage, not on the idea. Either add
   them to the perf sweep or measure them another way; do not land unmeasurable changes.
9. The remaining ~70 single-sided `v_if` sites (§3, Appendix B) — mechanical, but each needs its
   own bound-order and `SPECIALS_READY_OPS` check, so batch them by idiom rather than by file.

### How this compares to the literal-encoding work

Static probe figures are marked *(probe)*; figures confirmed on Wormhole n300 are marked
*(hardware)* and are the ones to quote.

| | approach | best measured | accuracy cost |
|---|---|---|---|
| companion doc, T1 | round literal onto the 16-bit grid | −2 slots, one site *(probe)* | ~5 fp32 ULP |
| companion doc, T3 | round approximation coefficients | −17 % on the `silu` body *(probe)* | +8 % on an 8.7e-3 error |
| **§3 here** | branch-free clamp | **−39.2 %** `relu_max` *(hardware)* | **none, NaN included** |
| **§1 here** | constants into CREGs | **−6.4 % / −6.7 %** `softplus` / `i0` *(hardware)* | **none, bit-exact** |
| **§2 here** | 2 elements per body, hand-interleaved | **−18.3 %** `i0`, on top of §1 *(hardware)* | **none, bit-exact** |
| **§1+§2 here** | both | **−23.8 %** `i0` *(hardware)* | **none, bit-exact** |

The hardware numbers reorder the priorities from the original draft. §3 is the largest win where
it applies, but it applies to few sites. §1 alone is much smaller than it looked. **§2 is the one
with broad remaining headroom** — it is the only idea here that attacks the `SFPNOP` tax directly,
it compounds with §1, and §2's survey table shows most polynomial kernels still paying 8–21 % of
their SFPU instructions to stalls. Its cost is that each application is a real rewrite, not a
mechanical edit.

The literal-encoding work is still worth doing — it applies to kernels that have no free CREGs,
which is most of them — but it should not be the first thing attempted on the kernels in §1's
table.

---

## 6. Reproducing

All static probes are instruction-count only; no device required. The probe sources are the code
blocks in §1, §2, §3 and §4 — each is a complete file once you prepend the boilerplate below.

Write the probe as `probe.cc`:

```cpp
#include <cstdint>
// sfpi's builtins reference ckernel::instrn_buffer, so it must be declared first.
namespace ckernel {
extern volatile std::uint32_t __instrn_buffer[];
constexpr inline volatile std::uint32_t (&instrn_buffer)[] = __instrn_buffer;
}
#include <sfpi.h>
using namespace sfpi;

// ... a "before" and an "after" function from one of the sections above ...
```

Then, from the tt-metal checkout root:

```bash
GXX=./tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-g++
FLAGS="-O3 -std=c++17 -ffast-math -fno-finite-math-only -fsigned-zeros \
       -fno-associative-math -fno-exceptions -fno-rtti -Itt_metal/tt-llk/tests/sfpi/include"
$GXX -mcpu=tt-wh-tensix $FLAGS -S -o out.s probe.cc

# per-function instruction counts
awk '/^_Z/{gsub(/:/,"");n=$0} /^\t(SFP|TTI)/{c[n]++} END{for(k in c) print c[k], k}' out.s | sort -rn
```

Run all three `-mcpu` values. WH emits `SFPNOP` padding the others do not, which is exactly why some rewrites change sign between architectures.

Two probe-writing constraints, both learned the hard way:

- Declare `ckernel::instrn_buffer` before `#include <sfpi.h>`, or every builtin fails to compile.
- `vFloat` cannot cross a function boundary — the SFPU register file has no spill path. Take inputs from `dst_reg[]` and write results back to `dst_reg[]`, otherwise: `error: cannot read SFPU object from memory`.

Appendices A and B were produced by throwaway scanners over the kernel headers rather than by a
checked-in tool. If you regenerate them, the two rules that matter are:

- **Appendix A (CREG availability)** — a kernel's claimed `vConstFloatPrgm` slots are the union of
  direct `vConstFloatPrgm* =` writes in its file *and* everything reachable transitively from its
  init hooks. Resolve callees recursively, and match function definitions with **and without**
  `inline`: `void i1_init() {` and `void erf_init() {` carry no `inline`, and skipping them makes
  `erf`, `i1`, `digamma` and `gelu` all look like they have three free slots when they have none.
- **Appendix B (predication inventory)** — match `v_if (<guard>) { <lhs> = <rhs>; } v_endif` where
  the body is a single assignment, then classify by guard direction and whether `<rhs>` equals the
  guard's bound. Adjacent opposite-direction guards on the same variable are the two-sided clamps;
  verify by eye that nothing sits between the two blocks and that they are not the arms of an
  `if constexpr`/`else` (both produced false positives — see §3).

### Measuring on hardware

This is the loop actually used for the §0 results. Both halves matter — the static count alone
gets §1 wrong.

```bash
cd tt_metal/tt-llk/tests/python_tests          # from the tt-metal checkout root
rm -rf "${TMPDIR:-/tmp}/tt-llk-build"          # MUST wipe: the cache ignores header content
../.venv/bin/python -m pytest perf_eltwise_unary_sfpu.py -q -p no:randomly -k "Softplus"
# results land in tt_metal/tt-llk/perf_data/runs/local-<timestamp>/*.parquet
```

Read `mean(MATH_ISOLATE)` **split by the `marker` column** — `INIT`, `KERNEL`, `TILE_LOOP`. The
split is not optional for §1: the whole point of that change is moving work from `KERNEL` into
`INIT`, and an aggregate hides it. The perf sweep drives Float16_b→Float16_b, so a kernel with
`#ifdef INP_FLOAT32` branches is exercising its **bf16** path.

Toggle the kernel header between `main` and the change, wipe the cache each time, and diff by
`(marker, mathop, approx_mode)`. Beware that two parametrisations can share those keys (approx
Yes/No often compile identically), which turns a naive merge into a cross-join — pair the sorted
values instead.

### Why whole-ELF static instruction counts cannot measure §1

The obvious static check — objdump the math ELF and count `sfp*` mnemonics — reports **no change
at all** for the CREG work, and reports it as a *regression* if you count naively:

| | `softplus` before → after | `i0` before → after |
|---|---|---|
| total `sfp*` | 45 → 51 | 42 → 48 |
| `sfploadi` | 16 → **16** | 18 → **18** |
| `sfpnop` | 3 → 6 | 6 → 9 |
| hardware `MATH_ISOLATE` | **−6.4 %** | **−6.7 %** |

`sfploadi` is unchanged because the six loads did not disappear — they **moved out of the
per-element body into the init function**, and a whole-ELF count cannot tell a once-per-kernel
instruction from a once-per-element one. The total *rises* because the init's three `SFPCONFIG`
writes are added while the exposed `SFPNOP`s appear. Both effects are real; neither is the thing
you wanted to measure.

Use whole-ELF static counts for §3 (where the change is entirely inside the body — `relu_max`
14 → 8 is a true reading) and the `INIT`/`KERNEL` marker split for §1.

### Validating correctness

- §1 and §2 are bit-exact, so a byte-comparison of output tiles against the pre-change build is a
  complete test — no error metrics needed. Stronger validation than anything available for the
  literal-rounding work, and it should be used.
- §3 needs edge-domain coverage: ±0.0, ±inf, NaN, and values exactly at the clamp bounds. Run
  `test_eltwise_unary_sfpu.py`'s `_edges` variants and make sure an **fp32→fp32** pair is in the
  selection — that is the only configuration where specials survive to L1
  (`specials_safe()`), so a bf16-only run silently skips the interesting cases.
- Do **not** validate any of this with a uniform exhaustive bf16 sweep — uniform aggregates over
  all 2^16 values weight subnormals and huge exponents absurdly and can rank a worse result as
  better.
- Check the op's entry in `SPECIALS_READY_OPS` before rewriting a predicated block: it records
  the golden's non-finite contract, and for `ReluMax` it already documented the total-order
  NaN behaviour that §3's caveat had treated as an open question.

---

## Appendix A — program constant register availability, all WH kernels

`Prgm claimed` is the union of direct `vConstFloatPrgm* =` writes in the file and everything reachable transitively from the file's init hooks. `fp32 consts in hot body` counts *distinct* fp32 literals outside init that are neither bf16-exact nor a hardwired CREG value — i.e. the candidates that currently cost two `SFPLOADI` each. A kernel with `free > 0` and a high constant count is an opportunity; `free = 0` means the slots are already spoken for.

| kernel | Prgm claimed | free | fp32 consts in hot body | reads Prgm | init entry points |
|---|---|---|---|---|---|
| MTL `softplus` | — | 3 | 25 | — | softplus_init |
| MTL `tanh_derivative` | — | 3 | 17 | — | tanh_derivative_init, tanh_derivative_sech2_init |
| LLK `expm1_cw` | — | 3 | 11 | — | — |
| MTL `i0` | — | 3 | 9 | — | i0_init |
| MTL `logsigmoid` | — | 3 | 9 | — | logsigmoid_init |
| LLK `cdf` | — | 3 | 7 | — | — |
| LLK `silu` | — | 3 | 6 | — | — |
| MTL `piecewise_rational` | — | 3 | 6 | — | — |
| LLK `rsqrt_compat` | — | 3 | 1 | — | — |
| MTL `isclose` | 0 | 2 | 0 | 0 | isclose_init |
| MTL `unary_max_min` | 0 | 2 | 0 | — | unary_max_min_init, unary_max_min_int32_init |
| MTL `binary_pow` | 0,1 | 1 | 13 | 0,1 | sfpu_binary_pow_init |
| LLK `mul_int` | 0,1 | 1 | 0 | — | _init_mul_int_ |
| MTL `activations` | 0,1 | 1 | 0 | 0,1 | hardsigmoid_init |
| MTL `fmod` | 0,1 | 1 | 0 | 0,1 | init_fmod |
| MTL `lcm` | 0,1 | 1 | 0 | — | calculate_sfpu_gcd_init, calculate_sfpu_lcm_init |
| MTL `remainder` | 0,1 | 1 | 0 | 0,1 | init_remainder |
| MTL `trigonometry` | 0,1,2 | 0 | 69 | 0,1,2 | acos_init, asin_acos_init, asin_init |
| MTL `digamma` | 0,1,2 | 0 | 56 | 0 | _calculate_log_body_no_init_, digamma_init, sfpu_reciprocal_init |
| MTL `gelu` | 0,1,2 | 0 | 51 | 0 | gelu_derivative_polynomial_init, gelu_init, gelu_tanh_init |
| MTL `i1` | 0,1,2 | 0 | 28 | — | i1_init |
| MTL `erf` | 0,1,2 | 0 | 24 | — | erf_init |
| MTL `erfc` | 0,1,2 | 0 | 20 | — | erfc_init |
| MTL `unary_power` | 0,1,2 | 0 | 20 | 0,1,2 | power_init, sfpu_unary_pow_init |
| MTL `lgamma` | 0,1,2 | 0 | 18 | — | _calculate_log_body_no_init_, lgamma_stirling_init |
| MTL `tanhshrink` | 0,1,2 | 0 | 15 | — | tanhshrink_init |
| MTL `exp` | 0,1,2 | 0 | 13 | 0,1 | _init_sfpu_config_reg, exp_init |
| MTL `atan2` | 0,1,2 | 0 | 10 | — | calculate_sfpu_atan2_init |
| MTL `snake_beta` | 0,1,2 | 0 | 9 | 0 | _init_sfpu_config_reg, snake_beta_init |
| MTL `xielu` | 0,1,2 | 0 | 9 | 0,1,2 | xielu_init |
| MTL `log` | 0,1,2 | 0 | 7 | 0,1,2 | log_init |
| MTL `log1p` | 0,1,2 | 0 | 7 | 0,1,2 | log1p_init |
| LLK `log` | 0,1,2 | 0 | 6 | 0,1,2 | _calculate_log_body_no_init_, _init_log_ |
| MTL `exp2` | 0,1,2 | 0 | 6 | 0,1,2 | exp2_init |
| MTL `tanh` | 0,1,2 | 0 | 6 | 0,1,2 | tanh_init |
| MTL `binary` | 0,1,2 | 0 | 5 | — | sfpu_binary_init |
| MTL `expm1` | 0,1,2 | 0 | 5 | 0,1,2 | expm1_init |
| MTL `cbrt` | 0,1,2 | 0 | 2 | 0,1,2 | cube_root_init |
| MTL `erfinv` | 0,1,2 | 0 | 2 | — | erfinv_init |
| MTL `polygamma` | 0,1,2 | 0 | 1 | — | polygamma_init |
| LLK `binary_bcast` | 0,1,2 | 0 | 0 | — | _init_sfpu_config_reg, _sfpu_binary_bcast_init_ |
| LLK `topk` | 0,1,2 | 0 | 0 | 0 | _init_topk |
| MTL `addcdiv` | 0,1,2 | 0 | 0 | — | init_addcdiv |
| MTL `binary_fmod` | 0,1,2 | 0 | 0 | — | fmod_binary_init, fmod_int32_init |
| MTL `binary_remainder` | 0,1,2 | 0 | 0 | 0,1,2 | remainder_binary_init, remainder_int32_init, remainder_uint32_init |
| MTL `div_int32` | 0,1,2 | 0 | 0 | — | div_init |
| MTL `div_int32_floor` | 0,1,2 | 0 | 0 | 0,1,2 | div_floor_init, div_trunc_init |
| MTL `mac` | 0,1,2 | 0 | 0 | — | mac_init |
| MTL `max_pool_indices` | 0,1,2 | 0 | 0 | — | init_max_pool_with_indices |
| MTL `mish` | 0,1,2 | 0 | 0 | — | mish_init |
| MTL `mul_int32` | 0,1,2 | 0 | 0 | — | mul_int32_init |
| MTL `quant` | 0,1,2 | 0 | 0 | — | dequant_init, quant_init, requant_init |
| MTL `rdiv` | 0,1,2 | 0 | 0 | — | rdiv_init |
| MTL `recip` | 0,1,2 | 0 | 0 | 0,1,2 | _init_sfpu_config_reg, recip_init, sfpu_reciprocal_init |
| MTL `reduce` | 0,1,2 | 0 | 0 | 0 | _init_sfpu_config_reg, init_reduce, init_reduce_max_min |
| MTL `rsqrt` | 0,1,2 | 0 | 0 | — | rsqrt_init |
| MTL `sigmoid` | 0,1,2 | 0 | 0 | — | sigmoid_appx_init, sigmoid_init |
| MTL `silu` | 0,1,2 | 0 | 0 | — | silu_init |
| MTL `softsign` | 0,1,2 | 0 | 0 | — | init_softsign |
| MTL `sqrt` | 0,1,2 | 0 | 0 | 0,1,2 | sqrt_init |
| MTL `topk` | 0,1,2 | 0 | 0 | — | _init_topk, topk_init |
| MTL `typecast` | 0,1,2 | 0 | 0 | 0 | init_typecast_fp32_to_fp16b, init_typecast_fp32_to_uint16, init_typecast_fp32_to_uint8 |
| MTL `llk_math_eltwise_unary_sfpu_init` | 0,1,2 | 0 | 0 | — | abs_init, acos_init, alt_complex_rotate90_init |

## Appendix B — predication inventory

Every `v_if` site whose body is a single assignment and which maps onto a specific branch-free idiom. Each still needs an individual edge-case check and a bound-order check (see §3) — this is a candidate list, not a patch set.

| tree | kernel:line | current | collapses to |
|---|---|---|---|
| LLK | `hardtanh:34` | `v_if (val < 0.0f) { val = 0.0f; } v_endif` | `val = max(val, 0.0f)` |
| LLK | `hardtanh:41` | `v_if (val >= 0.0f) { val = 0.0f; } v_endif` | `val = min(val, 0.0f)` |
| LLK | `relu:42` | `v_if (result > threshold) { result = threshold; } v_endif` | `result = min(result, threshold)` |
| LLK | `relu:47` | `v_if (result < 0.0f) { result = 0.0f; } v_endif` | `result = max(result, 0.0f)` |
| LLK | `relu:61` | `v_if (result > threshold) { result = threshold; } v_endif` | `result = min(result, threshold)` |
| LLK | `relu:66` | `v_if (result < 0) { result = 0; } v_endif` | `result = max(result, 0.0f)` |
| LLK | `rsqrt_compat:126` | `v_if (in < 0.0) { out = -out; } v_endif` | `copysgn(...)` / `setsgn(...)` |
| LLK | `rsqrt_compat:159` | `v_if (in < 0.0) { out = -out; } v_endif` | `copysgn(...)` / `setsgn(...)` |
| MTL | `binary_fmod:31` | `v_if(a_signed < 0) { r = -r; } v_endif` | `copysgn(...)` / `setsgn(...)` |
| MTL | `binary_pow:76` | `v_if(z_f32 < low_threshold) { z_f32 = low_threshold; } v_endif` | `z_f32 = max(z_f32, low_threshold)` |
| MTL | `binary_remainder:114` | `v_if(r < 0) { tmp = -tmp; } v_endif` | `copysgn(...)` / `setsgn(...)` |
| MTL | `expm1:108` | `v_if(jm2 < 0.0f) { r = -0.5f; } v_endif` | `copysgn(...)` / `setsgn(...)` |
| MTL | `relu:67` | `v_if(x < t) { x = t; } v_endif` | `x = max(x, t)` |
| MTL | `relu:70` | `v_if(x > t) { x = t; } v_endif` | `x = min(x, t)` |
| MTL | `relu:101` | `v_if(x < t) { x = t; } v_endif` | `x = max(x, t)` |
| MTL | `relu:111` | `v_if(x < t) { x = t; } v_endif` | `x = max(x, t)` |
| MTL | `relu:125` | `v_if(x < 0) { x = 0; } v_endif` | `x = max(x, 0.0f)` |
| MTL | `relu:127` | `v_if(x > t) { x = t; } v_endif` | `x = min(x, t)` |
| MTL | `relu:150` | `v_if(a < threshold) { a = threshold; } v_endif` | `a = max(a, threshold)` |
| MTL | `relu:164` | `v_if(a > threshold) { a = threshold; } v_endif` | `a = min(a, threshold)` |
| MTL | `relu:166` | `v_if(a < 0.0f) { a = 0.0f; } v_endif` | `a = max(a, 0.0f)` |
| MTL | `remainder:58` | `v_if(r < 0) { tmp = -tmp; } v_endif` | `copysgn(...)` / `setsgn(...)` |
| MTL | `unary_power:78` | `v_if(z_f32 < low_threshold) { z_f32 = low_threshold; } v_endif` | `z_f32 = max(z_f32, low_threshold)` |

## Appendix C — raw measurement table

Every number in this document, as emitted instruction counts. Reproduce with the commands in §6.

| probe | function | WH | BH | QSR32 |
|---|---|---|---|---|
| `exp_prgm.cc` | `lit3` — 3 fp32 literal coefficients | 12 | 11 | 11 |
| `exp_prgm.cc` | `prgm3` — same, in CREGs | **7** | **5** | 5 |
| `exp_prgm.cc` | `lit5` — 5-coefficient chain, literals | 18 | 14 | 14 |
| `exp_prgm.cc` | `prgm5` — same, 3 in CREGs | **13** | **8** | 8 |
| `exp_prgm.cc` | `prgm_init` — one-time cost of 3 CREG writes | 9 | 9 | 9 |
| `exp_interleave.cc` | `creg_1elem` (1 elem/body) | 13 → 13.0/elem | 8 → 8.0/elem | — |
| `exp_interleave.cc` | `creg_2elem` (2 elem/body) | 17 → **8.5/elem** | 15 → **7.5/elem** | — |
| `exp_interleave.cc` | `lit_1elem` (1 elem/body) | 18 → 18.0/elem | 14 → 14.0/elem | — |
| `exp_interleave.cc` | `lit_2elem` (2 elem/body) | 29 → 14.5/elem | 27 → 13.5/elem | — |
| `predication.cc` | `clamp_vif` | 15 | 13 | 13 |
| `predication.cc` | `clamp_mm` — `min(max(x,lo),hi)`; literal `lo < hi`, so order is immaterial here | **6** | **6** | 6 |
| `predication.cc` | `sign_vif` | 6 | 6 | 6 |
| `predication.cc` | `sign_cs` — `copysgn` | **4** | **4** | 4 |
| `predication.cc` | `relu_vif` | 5 | 5 | 5 |
| `predication.cc` | `relu_max` — `max(x,0.0f)` | **3** | **3** | 3 |
| `predication.cc` | `abs_vif` | 5 | 5 | 5 |
| `predication.cc` | `abs_ss` — `setsgn(x,0)` | **3** | **3** | 3 |
| `relu_max.cc` | `relu_max_before` (pre-change) | 13 | 12 | — |
| `relu_max.cc` | `relu_max_after` — `max(min(..))`, high bound first | **6** | **6** | — |
| `neg1_creg.cc` | `sub_ab` — `a - b` | 5, **reads L11** | 4, native `SFPADD` | — |
| `neg1_creg.cc` | `subassign` — `a -= b` | 5, **reads L11** | 4, native `SFPADD` | — |
| `neg1_creg.cc` | `negate` / `mulneg1` | 3, `SFPMOV` | 3, `SFPMOV` | — |

Replay-body ceiling, from `replay_cap.sh` (2-way interleaved body, growing MAD count, WH). The recorded body tracks the chain 1:1 until it saturates at 32, after which the excess is emitted outside the recorded segment and eventually the loop is split into several segments:

| interleaved MADs | emitted | recorded replay body | replay segments |
|---|---|---|---|
| 2 | 9 | 9 | 4 |
| 4 | 13 | 13 | 4 |
| 6 | 17 | 17 | 4 |
| 8 | 21 | 21 | 4 |
| 12 | 29 | 29 | 4 |
| 16 | 52 | **32 (ceiling)** | 4 |
| 24 | 40 | 22 | 11 (split) |

Practical reading: keep an interleaved body under ~32 instructions. Past that the compiler still produces correct code, but the excess stops being amortised by the replay mechanism and the emitted count jumps (16 MADs → 52 emitted for a 32-instruction recorded body).

### Hardware measurements (Wormhole n300)

tt-llk perf suite, `MATH_ISOLATE`, Float16_b→Float16_b, `tile_cnt=8`, `loop_factor=16`,
`iterations=32`; ELF cache wiped between variants. These supersede the probe figures wherever
the two disagree.

| kernel | marker | before (main) | after | delta |
|---|---|---|---|---|
| `relu_max` | INIT | 245 | 239 | −2.4 % |
| `relu_max` | KERNEL | 62850 | **38205** | **−39.2 %** |
| `relu_max` | TILE_LOOP | 62398 | 37760 | −39.5 % |
| `relu_max` | `L1_TO_L1` | 63720 | 38935 | −38.9 % |
| `softplus` | INIT | 239 | 257 | +7.5 % |
| `softplus` | KERNEL | 198231 | **185561** | **−6.4 %** |
| `softplus` | TILE_LOOP | 197786 | 185102 | −6.4 % |
| `i0` (§1 CREGs) | INIT | 240 | 257 | +7.1 % |
| `i0` (§1 CREGs) | KERNEL | 180856 | 168760 | −6.7 % |
| `i0` (§1 CREGs) | TILE_LOOP | 180421 | 168302 | −6.7 % |
| `i0` (§2 interleave, on top) | INIT | 257 | 256 | −0.4 % |
| `i0` (§2 interleave, on top) | KERNEL | 168760 | **137858** | **−18.3 %** |
| `i0` (§2 interleave, on top) | TILE_LOOP | 168302 | 137391 | −18.4 % |
| `i0` (§1+§2 vs main) | KERNEL | 180856 | **137858** | **−23.8 %** |

The `INIT` rise on `softplus` and `i0` is the three `SFPCONFIG` CREG writes (+17 cycles); it pays
back inside the first tile. `relu_max`'s `INIT` fall is incidental.

Real math ELF, static, `relu_max`: 14 → 8 `sfp*` instructions; `sfpsetcc` 3 → 0,
`sfpencc` 2 → 0, `sfpswap` 0 → 2.

## Appendix D — negative results and dead ends

Recorded so they are not rediscovered.

| idea | outcome |
|---|---|
| Wrap literals in `sFloat16a`/`sFloat16b` to force a single load | Unnecessary — the compiler already folds grid-aligned literals, and `sFloat16a(float)` is `= delete`d for exactly this reason. See companion doc §1. |
| Reclaim CREG 11 (the −1.0 constant) as a 4th program slot | **Unsafe on WH** — every `a - b` compiles to `SFPMAD(a, L11, b)`. BH-only, and not worth the risk while §1 slots remain. §4. |
| Interleave elements without first moving constants to CREGs | Duplicates the `SFPLOADI` (6 → 12); WH gains only 18.0 → 14.5 instead of → 8.5. §2. |
| 4-way element interleave | `dst_reg[k]` immediate window is `[-8,7]` half-rows; overflows and fails to compile once unrolled. §2. |
| Round the Horner-tail constant `4.99999851e-1f` to `0.5f` | **Regresses WH** (+1): the bf16 immediate splits the fused `SFPMAD` into `SFPMULI`+`SFPADD` and the added `SFPNOP`s cost more than the two saved loads. Companion doc §4, T1-d. |
| Assume `erf`/`i1`/`digamma`/`gelu` have free CREG slots | Wrong — all reach `sfpu_reciprocal_init`, which claims all three. Availability must be resolved transitively. §1. |
| Apply §2 by writing two element expressions back to back and letting GCC schedule them | Does nothing — GCC will not interleave two independent expression trees. Measured 18 `SFPNOP` per two elements on `i0`, identical per-element to the 1-way version. The interleave has to alternate at statement level. §2. |
| Treat the 32-instruction replay cap as a constraint on every kernel | The real kernels with `#pragma GCC unroll 0` compile to rolled RISC-V loops with no `TTREPLAY` at all, so the cap never binds. `i0`'s interleaved body is 63 slots and fine. Check the disassembly first. §2. |
| Resolve CREG availability transitively but only parse `inline` function definitions | Still wrong, and worse because it looks rigorous — `void i1_init() {` and `void erf_init() {` are not `inline`, so the recursion never ran. Also mis-reported `lgamma` and `exp` as having free slots. §1. |
| Two-sided clamp pairs at `relu.h:67-70` (MTL) and `hardtanh.h:34-41` (LLK) | False positives — an `if constexpr`/`else` pair, and two unrelated ops separated by `val += p1`. §3. |
| Rewrite a predicated two-sided clamp as `min(max(x, lo), hi)` | Wrong when the bounds can cross. `min`/`max` compose in written order, so it must match the order the predicated original applied them. Disagrees with `relu_max` on 6 of 24 sampled cases, all at negative thresholds. Use `max(min(x, hi), lo)` where the original clamped high first. §3. |
| Expect §1 to save 2 slots per parked coefficient | On Wormhole a dependent Horner chain emits `LOADI/LOADI/MAD` with **no** `SFPNOP` — the loads were filling the MAD latency slot. Parking exposes it, so the net is 1–2 slots and tends to 1 in a long chain. Measured: `softplus` and `i0` each gained exactly 3 `SFPNOP` for 3 parked coefficients. §1. |
| Park *every* fp32 coefficient of a chain in CREGs | Past the point where the remaining loads no longer cover MAD latency, each further CREG buys an `SFPNOP` instead of a saving. `i0` parks 3 of 11 deliberately. §1. |
| Measure §1 by counting `sfp*` in the math ELF | Reports zero change (`sfploadi` 16→16, 18→18) and a rising total, while hardware shows −6.4 %/−6.7 %. The loads move from body to init and a whole-ELF count cannot separate per-element from once-per-kernel. Use the `INIT`/`KERNEL` marker split. §6. |
| Assume `v_if (x > threshold)` passes NaN through, per IEEE | The SFPU compare is a sign-magnitude **total order**, so NaN compares greater than any finite value and the predicated form already mapped NaN → threshold. `SFPSWAP` shares that order, which is why the branch-free rewrite is bit-exact on NaN rather than merely close. `SPECIALS_READY_OPS` documented this the whole time. §3. |
| Use `sfpi::min`/`max` on `vInt` | The `vInt` overloads are gated behind `#if __riscv_xtttensixqsr` — Quasar only. On WH/BH `max(vInt, 0)` resolves through the `vUInt` overload and returns the wrong type; `SFPSWAP` also orders sign+magnitude, so a real `vInt` clamp needs the 2's-complement conversion `_relu_min_` does by hand. §3. |
| Assume a kernel's init hook runs just because it exists | The tt-llk harness has no per-op init arm for `softplus` or `i0`; both reach their init only through the fallback `llk_math_eltwise_unary_sfpu_init<OPERATION, ...>()` → `_llk_math_eltwise_unary_sfpu_init_`'s `SfpuType` switch. A missing init compiles fine and silently reads stale CREGs. Verify the path. §1. |
| Plan §1 work for `logsigmoid` / `binary_pow` | Not `MathOperation` members, so unreachable from the tt-llk unary perf sweep — unmeasurable and untestable in this harness regardless of how many CREGs are free. §1. |
