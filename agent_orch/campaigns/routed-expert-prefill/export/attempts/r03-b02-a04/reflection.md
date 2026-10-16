# r03-b02-a04 result: 1.5717 (ok). A regression vs the parent (1.6103, -2.4%), entirely on the shapes where column streaming engaged

## What happened vs expected
- **Correctness:**
  - All 9 shapes are valid and bit-identical to the parent and the baseline (PCC, max_abs and mean_abs equal to every digit).
  - So the ct < rt LLK mode, the lazy re-inits around the owner tilize and the last block, and the pinned TILE_SIZE_B are all correct.
  - No hang and no JIT error on the first and only run. No kernel-config-buffer overflow: the program is smaller than the parent's (local .o sum -0.5..-1.0 KB).
- **Speed.** I expected 20-35% off GU/DN on the per_core_M >= 3 cores. Instead those cores got slower:

| shape | parent µs | this µs | Δ | mean core end parent → this | CS used? |
|---|---|---|---|---|---|
| minimax-m3 | 365.5 | 392.9 | **+7.5%** | 344 → 362 | yes (pcM 3) |
| gpt-oss | 261.8 | 275.0 | **+5.0%** | 238 → 248 | yes (pcM 3-4) |
| mistral | 185.9 | 194.6 | **+4.7%** | 166 → 174 | yes (pcM 3) |
| glm | 451.2 | 459.5 | +1.8% | 429 → 438 | phase A (pcM 3) |
| dsv3 | 514.0 | 521.1 | +1.4% | 493 → 497 | phase A (pcM 3) |
| kimi-k2.7 | 729.5 | 736.5 | +1.0% | 719 → 728 | some 5-tile experts (pcM 3) |
| kimi-k3 | 1365.8 | 1378.3 | +0.9% | 1273 → 1270 | no (pcM 2) |
| dsv4-flash | 316.9 | 318.1 | +0.4% | 298 → 295 | few |
| dsv4-pro | 1070.9 | 1067.4 | -0.3% | 1059 → 1058 | no (pcM 1-2) |

The shapes that never or rarely take the CS path (kimi-k3, dsv4-pro, dsv4-flash) are flat in the mean core. So the out-of-line helper refactor (runtime ct/rt, one matmul_block copy) costs nothing measurable on the 1 x W path.

## Why (profiler evidence)
MATH perf counters (/tmp/r03b02a03/hw.py on reports/r03-b02-a04, parent numbers in brackets):

| shape / phase | cyc/mult | FPU busy | UNP0 (SrcA) busy | UNP1 (SrcB) busy | SrcA wr / mult | SrcB wr / mult |
|---|---|---|---|---|---|---|
| mistral GU | 32.4 [31.0] | 49.7% [51.9] | 29.9% [57.3] | 50.4% [26.8] | 3.6 [17.5] | 7.6 [2.9] |
| mistral DN | 31.3 [29.1] | 51.4% [55.5] | 27.7% [62.2] | 57.4% [25.4] | 3.0 [17.5] | 8.4 |
| minimax-m3 GU | 31.6 [30.3] | 50.9% [53.0] | 29.1% [59.1] | 53.2% [56.3] | 3.3 [17.7] | 7.9 |
| gpt-oss DN | 32.3 [30.9] | 49.9% [52.0] | 30.1% [60.2] | 56.4% [55.3] | 4.7 [18.0] | 11.9 |
| dsv3 GU | 33.0 [32.6] | 48.7% | 38.1% [54.3] | 39.2% [23.8] | 8.7 [17.5] | 5.8 |

- **The mechanism did exactly what it was built to do.**
  - The bf4 weight is now unpacked once per column of per_core_M mults: SrcA writes per mult fell ~5x on mistral / mm3, and UNP0 busy halved.
  - The bf8 x / activated tiles took over as the per-mult stream: UNP1 busy doubled.
- **But the tile-mult did not get cheaper.**
  - FPU busy is still 16.1 cycles/mult and still only ~50% of the window.
  - The streamed bf8 tile costs ~16 SrcB-unpacker cycles per mult (UNP1 50-57% of ~32 cycles).
  - That unpack is just as serialized with the FPU as the bf4 SrcA unpack was: FPU busy + UNP1 busy ≈ 100-108% on mistral.
- **So the parent's reading was half right.**
  - A mult costs FPU (16) + one streamed-tile unpack (~15-17), serialized.
  - But that holds for whichever operand streams and whatever its format (bf4 SrcA or bf8 SrcB). It is not a property of the SrcA/bf4 path that switching operands could escape.
  - The premise that "big BH GEMMs overlap SrcB streaming" does not carry over to this kernel.
- **Why it got slower, not neutral:**
  - CS does more unpacks per mult: (1 + rt) / rt per column, e.g. 1.33 at pcM 3 with ct 2 → 2 context rounds per 6 mults. The 1 x 6 walk needs 1.17.
  - Each weight-tile unpack is still ~17-29 UNP0 cycles, adding to the serialized chain at every column switch.
  - Net ~+1-2 cycles/mult on the CS cores: +4..+7% on the one-expert-per-group compute-bound shapes, +1-2% on dsv3/glm phase A.

## Classification
flawed idea. The implementation is correct and bit-identical. Swapping which operand is streamed does not break the unpack→FPU serialization; it only adds context rounds.

## What a child of this node should try next
1. **Start from the parent's fused_swiglu.cpp** (r03-b02-a03), or set `kCsMinRows` > 8 here to disable CS.
   - The out-of-line `mm_init` / `mm_kloop` / `mm_group` / `mm_block` helpers are worth keeping only for kernel-config-buffer headroom. With CS they are net -0.5..-1.0 KB vs the parent, and they cost nothing measurable on the 1 x W path (kimi-k3 / dsv4 mean cores flat).
   - With CS disabled at compile time, UNPACK would shrink further, because ct/rt become effectively constant for reuse_a.
2. **The real question is why the src double-buffering doesn't overlap unpack with MVMUL in this kernel.** The FPU sits at 16 cycles per mult, ~50% duty, whichever operand streams.
   - Candidates to check in llk_unpack_AB_matmul.h's per-tile replay: UNPACR → RDCFG → ADDDMAREG → STALLWAIT(STALL_CFG, THCON) → WRCFG of the *active* context's base address → NOP. If the WRCFG to the in-use context's base register stalls until that UNPACR finishes, the unpacker cannot run ahead.
   - An op-local copy of the unpack MOP could pre-compute tile addresses, or alternate the two config contexts per tile, so tile i+1's address is never written into the context tile i is using. That is a direct, measurable test with these counters.
   - Copying LLK code into the op directory is allowed by the rules (shared helpers must be copied first).
   - Measure on mistral GU first: success looks like cyc/mult → ~18-20 with FPU busy → 80%+.
3. Don't retry: any operand-swap / subblock-geometry variant (this node, r01-b02-a04, the taller-subblock notes), or FW-throttle theories.
4. The trimmed RE-HW instrumentation (FPU + TDMA_UNPACK banks; stall / valid / throttle slots now emit 0) still parses with /tmp/r03b02a03/hw.py.
