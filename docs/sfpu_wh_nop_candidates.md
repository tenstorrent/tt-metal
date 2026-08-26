# Wormhole SFPU: NOP and pipeline-stall candidates

A survey of the Wormhole B0 SFPU kernels for instruction-scheduling wins — places where an
`SFPNOP` could be replaced with useful work, or a stall removed by reshuffling.

**Status: static analysis only. Nothing here has been measured.** Instruction counts are an
upper bound on the win, not the win. Treat every number below as "worth measuring", not
"worth landing".

**Scope: Wormhole B0 only.** Blackhole does not need these NOPs at all — it interlocks —
so none of this applies there. That is also the reason the list is worth having: WH kernels
carry a scheduling tax that the BH versions of the same kernels do not.

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

### 1. `quant` / `requant` / `dequant` — start here

Best effort-to-payoff by a wide margin, and the only entry where both the fill material and
the register headroom are already confirmed.

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

### 2. `ema` — highest ceiling, hardest

8 MADs, 8 NOPs, every one annotated "Next cycle cannot read from LREGn (2-cycle operation)".
All data hazards, so in principle all fillable.

The obstacle is the algorithm, not the schedule. `_compute_ema_math_()` is a serial
recurrence over 4 rows:

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

### 3. `binary_bcast` — unassessed

8 NOPs around a `MUL` / `SFPSHFT2` / `ADD` shuffle-reduce chain. `SFPSHFT2` is 2-cycle, so
these are data hazards and fillable in principle, but it is a log-reduction where each round
consumes the previous one's output. Needs someone to look at whether rounds overlap.

### 4. `lcm` — small, low-risk, already half-solved

3 per-element NOPs across two serial Newton-Raphson reciprocal iterations. Note that the
author **already** filled one slot: line 82 drops an independent `SFPIADD` (exponent
bookkeeping) straight after the last MAD. The `SFPEXEXP` at line 71 is likewise independent
of the Newton chain and could migrate into one of the remaining slots. Modest win, contained
blast radius, good first exercise for anyone learning this.

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
