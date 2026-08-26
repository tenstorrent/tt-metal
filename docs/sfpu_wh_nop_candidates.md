# Wormhole SFPU: NOP and pipeline-stall candidates

A survey of the Wormhole B0 SFPU kernels for instruction-scheduling wins — places where an
`SFPNOP` could be replaced with useful work, or a stall removed by reshuffling.

**Status: attempted. See §0 for what actually survived contact.** One of the six candidates
produced a win; four turned out to have no headroom and one is untestable. The original
per-candidate analysis is kept below, corrected in place, because the reasons the dead ends
are dead are the useful part.

**Scope: Wormhole B0 only.** Blackhole does not need these NOPs at all — it interlocks —
so none of this applies there. That is also the reason the list is worth having: WH kernels
carry a scheduling tax that the BH versions of the same kernels do not.

---

## 0. Outcome

Worked through on branch `ldjurovic/sfpu_wh_nop_overlap`.

| Candidate | Outcome |
|---|---|
| `ema` | **Done — 17 → 11 slots**, bit-identical over 172032 outputs |
| `quant` / `requant` / `dequant` | **Blocked** — not wired into the LLK harness, no test to validate against |
| `lcm` | **No headroom** — all 8 LREGs in use, and the one "free" slot is already filled |
| `binary_bcast` | **Already done** by its author; residual NOPs have no independent work left |
| `reduce` | **Already done**; the 3 remaining NOPs are end-of-chain drains |
| `cumsum` | **Wash**, as predicted in §3.6 |

The recurring blocker is not cleverness, it is **register pressure**. Filling a latency slot
needs an independent instruction, an independent instruction needs its own registers, and
these kernels mostly already use all eight LREGs. `ema` was winnable precisely because it did
*not* need another register — the win came from restructuring the arithmetic so that
independent work already in the block could be dealt into the slots.

**The generalisable lesson:** look for kernels where independent work already exists in the
block and is merely mis-scheduled, or where an algebraic reassociation creates independent
work without new registers. Do not look for kernels with big NOP counts.

A second, cheaper lesson: **check for a test before analysing a kernel.** `quant` was ranked
first on the merits and then turned out to be unverifiable on the LLK harness, which is a
hard stop — the whole point of these changes is that they are bit-neutral, and that is not a
claim to make without a way to check it.

---

## 1. The one thing to read before starting

There are **two kinds of NOP** in these kernels and only one of them is removable.

**Data-hazard NOPs.** `SFPMAD`, `SFPADD`, `SFPSHFT2` and `SFPIADD` have a 2-cycle write
latency, so a consumer of their result cannot issue in the next slot. Any *independent*
instruction can fill that slot. **These are the fillable ones.**

**Structural-hazard NOPs.** `SFPSWAP` takes 2 cycles and is **not pipelined** — the unit
cannot accept a new SWAP in the following slot no matter which registers are involved.
Register rotation does nothing here. **These are dead ends.**

The evidence that this distinction is real, and not a guess:

- `ckernel_sfpu_exp.h:527-580` (approximate path) already rotates **4-way** through
  LREG0-3 via macro sequence registers 4/5/6/7, so consecutive `SFPLOADMACRO`s are
  register-independent — and it *still* carries an `SFPNOP` after every one. The comment at
  line 533 says why: "NOP is necessary because the SWAP operation takes 2 cycles and
  unfortunately is not pipelined."
- `ckernel_sfpu_unary_max_min.h:41` states it as a schedule table:
  `1 | nop | swap_minmax([a], v)`.

This single observation removes most of the apparent opportunity in the tree — see §4.

Also relevant, and cheap to forget: `SFPCAST`, `SFPEXMAN`, `SFPEXEXP` and `SFP_STOCH_RND`
results **are** readable by the very next instruction. Do not insert NOPs after those.

---

## 2. Method

```sh
# census
cd tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu
for f in *.h; do n=$(grep -c TTI_SFPNOP "$f"); [ "$n" -gt 0 ] && echo "$n $f"; done | sort -rn
# and the same over tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/
```

Two traps when reading the results:

1. A NOP inside `lltt::record` / `TTI_REPLAY(..., 1, 1)` is **per-element** even though it is
   not textually inside a loop. Naive "is it in a `for`?" classification gets these backwards.
2. NOPs *after* a `SFPLOADMACRO` loop are one-time pipeline drain and cost nothing per
   element. `ckernel_sfpu_typecast.h` has 34 NOPs and only 2 of them are per-element.

What matters is **NOPs per element in the steady-state body**, which is what the table uses.

---

## 3. Candidates, ranked

| # | Kernel | Per-element body | NOPs | Hazard | Fill material available? |
|---|---|---|---|---|---|
| 1 | `ckernel_sfpu_quant.h` (dequant) | 8 slots | **2 (25 %)** | data | **yes — next element's loads** |
| 1 | `ckernel_sfpu_quant.h` (quant) | 6 slots | 1 (17 %) | data | **yes** |
| 1 | `ckernel_sfpu_quant.h` (requant) | 7 slots | 1 (14 %) | data | **yes** |
| 2 | `ckernel_sfpu_ema.h` | 16 slots / 4 rows | **8 (50 %)** | data | only across independent sequences |
| 3 | `ckernel_sfpu_binary_bcast.h` | — | 8 | data (`SFPSHFT2`) | needs analysis |
| 4 | `ckernel_sfpu_lcm.h` | — | 3 | data | partly done already |
| 5 | `ckernel_sfpu_exp.h` (accurate TTI) | 17 slots | 3 | data + 1 struct | no free register |
| 6 | `ckernel_sfpu_cumsum.h` | 16 slots | **8 (50 %)** | data | **no — see below** |

### 1. `quant` / `requant` / `dequant` — BLOCKED: no test

> **Outcome: not attempted.** These kernels are reachable only from ttnn
> (`tt_metal/hw/inc/api/compute/quantization.h`). They are **not** wired into
> `tests/helpers/include/sfpu_operations.h` and no python test drives them, so there is no
> way to show a reshuffle is bit-neutral on this harness. Wiring up coverage is the
> prerequisite, and is probably worth more than the reshuffle.
>
> Also noted while reading: the kernels program `ADDR_MOD_6` with the dest increment
> (`_quant_kernels_configure_dest_incr_addrmod_`) but the stores issue against `ADDR_MOD_2`,
> which is shared state configured by datacopy/A2D. That may be fine and may be a latent
> bug; either way it needs resolving before anyone restructures the addressing.

On the merits this is still the best-shaped candidate — fill material and register headroom
both confirmed:

The recorded bodies (`ckernel_sfpu_quant.h:46-56`) are:

```
QUANT_REPLAY_LEN   = 3    MAD, NOP, STOCH_RND
REQUANT_REPLAY_LEN = 4    CAST, MAD, NOP, STOCH_RND
DEQUANT_REPLAY_LEN = 5    CAST, ADD, NOP, MUL, NOP
```

and each is wrapped per element by two `TT_SFPLOAD`s and one `TT_SFPSTORE`:

```c
for (d...) {
    TT_SFPLOAD(LREG0, ...);   // operand A
    TT_SFPLOAD(LREG1, ...);   // operand B — scaler
    lltt::replay(...);        // <- the NOP(s) live in here
    TT_SFPSTORE(LREG0, ...);
}
```

**Why it is fillable:** the two loads for element `d+1` do not depend on element `d`'s
MAD/ADD/MUL. Rotating the operand registers by one element lets the next load issue into the
latency slot, which is exactly the trick `ckernel_sfpu_binary_max_min.h` already uses on its
macro path.

**Register headroom is there:** quant touches only LREG0/1/2/4 and dequant only LREG0/1, so
LREG5/6 are free for the rotation at no cost.

`dequant` is the one to do first: 2 NOPs in 8 slots, and both are plain MAD-class latency.

### 2. `ema` — DONE, 17 → 11 slots

> **Outcome: fixed.** The math block went from 8 MADs + 8 NOPs + carry MOV (17 slots) to
> 8 MADs + 2 NOPs + MOV (11 slots), and the per-row critical path from two MADs to one.
> Bit-identical over 172032 outputs (8 seeds x 3 amplitudes x 3 tile counts).
>
> **The analysis below was wrong about how.** It claimed the fix needed "a *second*
> independent EMA sequence to interleave". It did not. Scaling the inputs by beta up front
> turns each row into a single fused `row_i = alpha*row_{i-1} + beta_scaled_i`, and the four
> beta multiplies are mutually independent, so three of them deal straight into the chain's
> latency slots. No second sequence, no extra register — the independent work was already
> inside the block, just expressed in a form that kept it on the critical path.
>
> The reassociation is not bit-neutral in fp32 (~2^-24), but DEST for this kernel is
> bfloat16, whose resolution is 2^-9, so nothing survives to the output. **That headroom is
> what made it safe** and is worth checking first on any similar reassociation: compare the
> fp32 perturbation against the output format's ULP.

8 MADs, 8 NOPs, every one annotated "Next cycle cannot read from LREGn (2-cycle operation)".
All data hazards. `_compute_ema_math_()` was a serial recurrence over 4 rows:

```
LREG7 = alpha * LREG4          (carry in from the previous block)
LREG0 = beta * LREG0 + LREG7
LREG7 = alpha * LREG0
LREG1 = beta * LREG1 + LREG7
...
```

Row `i` genuinely depends on row `i-1`, and blocks chain through LREG4, so the whole tile is
one dependency chain. Filling these NOPs means finding a *second* independent EMA sequence to
interleave (a batch dimension, say) — a restructuring, not a reshuffle. Worth scoping before
committing to it, because it may simply not be available.

### 3. `binary_bcast` — already optimised by its author

> **Outcome: no change.** `_broadcast_stage3_with_data_prefetch_` already applies exactly
> this technique, and says so: the 4 `SFPSHFT2` latency slots are filled by interleaving the
> 4 independent data `SFPLOAD`s. The 4 per-slot binops are likewise deliberately pipelined,
> and the trailing drain is documented. The 6 NOPs left in `_record_broadcast_replay_`
> (stages 1-2) have **no independent work available** — the only independent instructions in
> the block are those 4 loads and stage 3 has already claimed them, so any redeal just moves
> the NOPs. `SFPSHFT2` offers only `SHFLROR1` (no ROR2/ROR4), so the double-rotate in stage 2
> cannot collapse either. Cross-band pipelining is the only remaining angle and there is ~1
> free LREG for it.


8 NOPs around a `MUL` / `SFPSHFT2` / `ADD` shuffle-reduce chain. `SFPSHFT2` is 2-cycle, so
these are data hazards and fillable in principle, but it is a log-reduction where each round
consumes the previous one's output. Needs someone to look at whether rounds overlap.

### 4. `lcm` — no headroom

> **Outcome: no change, and the suggestion below was wrong.** It proposed migrating the
> `SFPEXEXP` at line 71 into a later NOP slot. That instruction is already doing exactly that
> job: the MAD at line 70 writes LREG0, line 74 reads it, and line 71 is the filler between
> them. Moving it would create the stall it was placed to avoid.
>
> Nothing else is available. `calculate_sfpu_mul_u16_to_u32_body` uses **all eight** of
> LREG0-7, so there is no room to overlap a second element, and within one element the three
> remaining NOPs sit in a strictly serial Newton-Raphson chain.

3 per-element NOPs across two serial Newton-Raphson reciprocal iterations. The author had
already filled the fourth slot: line 82 drops an independent `SFPIADD` (exponent bookkeeping)
straight after the last MAD.

### 5. `exp` accurate TTI path — known, blocked

`_sfpu_exp_21f_bf16_tti_` runs 17 slots with 3 NOPs. Analysed in depth already; see
`docs/sfpu_exp21f_optimization.md`. Going further needs software-pipelining two Dest
elements, which needs a register that is not free. One of its NOPs is the non-pipelined
`SFPSWAP` and is structural.

Two WH-specific facts worth carrying over from that work, because they are not obvious and
cost real time to rediscover:

- **`SFPSTOCHRND_RND_ZERO` is silently ignored on WH B0.** The encoding is bit-identical to
  Blackhole and sfpi defines the constant unguarded, but the silicon rounds to nearest
  anyway. Anything relying on it for `floor` is wrong by a factor of 2 wherever the
  fractional part is >= 0.5.
- **`SFPGT` does not exist on WH** (4 references on BH, zero on WH), so BH's SFPGT/SFPAND
  mask idiom for cheap predication cannot be ported.

### 6. `cumsum` — the trap

Worst ratio in the tree at 8 NOPs in a 16-instruction recorded body (50 %), and **the obvious
fix does not pay**. Listed last deliberately so nobody burns a week on it.

`cumsum_init()` (`ckernel_sfpu_cumsum.h:150-168`) records a serial prefix sum across
LREG7 → LREG0 → LREG1 → ... → LREG7, one ADD per lane, each dependent on the last.

Two restructurings, both dead:

- **Hillis-Steele / Kogge-Stone scan.** Three rounds of mutually independent adds, so zero
  NOPs — but 7 + 6 + 4 = **17 adds against 16 slots today**. A wash, very slightly worse.
- **Interleave the two `SFPTRANSP` blocks.** They are independent, but each needs all 8
  LREGs, so there is no register headroom to hold both.

If someone wants this one, the angle is reducing the *number* of adds, not hiding their
latency.

---

## 4. Ruled out — do not spend time here

**The whole `SFPLOADMACRO` + NOP family.** These look like the biggest prize by raw count and
are structural (§1), so rotation cannot help:

| Kernel | Note |
|---|---|
| `ckernel_sfpu_unary_max_min.h` | macro wraps `SFPSWAP`; schedule table at line 41 says `nop` outright |
| `ckernel_sfpu_binary_max_min.h` | already 4-way pipelined (a0/b0/a1/b1); remaining NOPs are drain |
| `ckernel_sfpu_typecast.h` | 34 NOPs but only 2 per-element; clamp path wraps `SFPSWAP` |
| `ckernel_sfpu_signbit.h` | all 5 are one-time drain |
| `ckernel_sfpu_exp.h` approx path | already 4-way rotated and still needs the NOP — the proof case |

**Already optimised, leave alone.** `ckernel_sfpu_reduce.h` — line 1318 reads "Cover A3
latency (A1 is already available)" and line 330 documents deliberate `SFPSHFT2` interleaving.
Someone has done this work; the 3 remaining per-element NOPs are the residue.

**One-time drain only, no per-element cost:** `cumsum` (outside the recorded body), `topk`,
`quant` init, `welfords`, `add_top_row`, `mul_int`, `generalized_moe_gate_topk_single_face`,
`reshuffle_rows` (1 NOP, and the following `SFPTRANSP` needs the result).

---

## 5. Before landing any of this

1. **Measure first.** `tests/run_llk_perf_wormhole.sh`, or
   `perf_eltwise_unary_sfpu.py` with `MATH_ISOLATE` on the `TILE_LOOP` marker, which is the
   methodology `docs/sfpu_exp21f_optimization.md` used. Build baseline and branch from
   separate clean roots.
2. **Gate on bit-exactness, not on the test suite passing.** The sweeps have tolerances that
   absorb single-ULP movement. Dump raw output bits for a dense input sweep plus edges
   (overflow knee, negative flush, `±inf`, NaN) before and after, and diff them. A
   reshuffle that only reorders independent instructions should be bit-identical by
   construction; if it is not, something is wrong with the reshuffle.
3. **Check `BODY_LEN` against the emitted instruction count** for anything inside a replay
   body. It must match exactly or the replay buffer misaligns, and the failure mode is not a
   clean one.
4. **Expect the win to shrink.** On the `exp` WH work, a change that looked like
   −3 instructions collapsed to −1 once bit-identity was required, because the extra
   instruction needed to preserve the rounding brought its own latency NOP with it.
