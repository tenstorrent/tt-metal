# WELFORD side-by-side: hand impl-1 (record/replay) vs clean semantic impl

Lane BB (agent/welford-clean-rewrite), 2026-08-18.  Both disassemblies are
the PERF variant (TRACE_N=0, all 32 rows live, WELFORD_BODY zone; NOT the
trace build), compiled by the pinned toolchain (cc1plus ac81ede3827d…).

| artifact | flags | math.elf .text sha256 |
|---|---|---|
| hand impl-1 `device_profile[1-handwritten_replay]` | sweep ON set (== OFF set byte-identically) | `214e3fd4fc45f09b…` |
| clean semantic impl-2 `device_profile[2-vfloat_direct]` | sweep ON set (== OFF set byte-identically: `cef74778e53d…` both legs) | `cef74778e53de80c…` |

Byte facts: the hand binaries are OFF/ON byte-identical, and the clean
semantic perf binary is ALSO OFF/ON byte-identical — no sweep-ON mechanism
fires on this shape (no SFPLOADMACRO, no compiler replay wrap, no
dst-autoincr rewrite of the explicit fp32-parked scratch traffic).  The only
welford binary the ON set changes at all is the unwired diagnostic impl-4
(`vfloat_manual_early_fold`, latency-schedule class).

## 1. High-level source (the clean semantic body)

Per row (impl 2, `VFLOAT_DIRECT`; `recip` is a constexpr `1.0f / N`):

```cpp
sfpi::vFloat delta     = x - mean;
sfpi::vFloat next_mean = mean + delta * recip;
sfpi::vFloat next_m2   = m2 + delta * (x - next_mean);
mean = next_mean;  m2 = next_m2;
```

Per block (4 rows, 8 blocks; `base = I*16 + 2*J`):

```cpp
sfpi::dst_reg[VF_SCRATCH_MEAN].mode<F32>() = mean;   // park (see section 4)
sfpi::dst_reg[VF_SCRATCH_M2].mode<F32>()   = m2;
sfpi::vFloat x0 = sfpi::dst_reg[base + 0];           // rows r..r+3, even cols
sfpi::vFloat x1 = sfpi::dst_reg[base + 1];           // rows r..r+3, odd cols
sfpi::vFloat x2 = sfpi::dst_reg[base + 8];           // face+1, even cols
sfpi::vFloat x3 = sfpi::dst_reg[base + 9];           // face+1, odd cols
sfpi::subvec_transp(x0, x1, x2, x3);                 // -> one tile row per vector
mean = sfpi::dst_reg[VF_SCRATCH_MEAN].mode<F32>();   // unpark
m2   = sfpi::dst_reg[VF_SCRATCH_M2].mode<F32>();
vfloat_step<Base+1>(x0, mean, m2);  ... vfloat_step<Base+4>(x3, mean, m2);
```

The body ends with two fp32 stores of the final mean/m2 to the scratch slots
— its output contract (and what keeps the last block alive in the TRACE_N=0
build; without it the compiler dead-code-eliminated block 8, an
under-measurement this lane caught in disassembly review).

## 2. Hand source (pristine `ckernel_sfpu_welfords.h`)

Row program, recorded once at init into replay slots 0-23
(`_program_welfords_replay_buffer_`, 6 instructions per input LREG):

```cpp
TTI_SFPMAD(LREG11 /*-1*/, LREG4, input, LREG6, 0);  // alpha = x - mean
TTI_SFPMAD(LREG6, LREG7, LREG4, LREG6, 0);          // mean' = alpha*recip + mean
TTI_SFPMAD(LREG11, LREG4, input, LREG4, 0);         // alpha again (LREG6 now mean')
TTI_SFPMAD(LREG11, LREG6, input, input, 0);         // beta = x - mean'
TTI_SFPMAD(LREG4, input, LREG5, LREG5, 0);          // m2 += alpha*beta
TTI_SFPMOV(0, LREG6, LREG4, 0);                     // mean <- mean'
```

Block gather (`_welfords_load_block_`): `TTI_SFPTRANSP; 4x TTI_SFPLOAD(L0..L3,
FMT_SRCB, ADDR_MOD_7, {o, o+2, o+16, o+18}); TTI_SFPTRANSP` — the
TRANSP/TRANSP pair is an involution on the L4-L7 bank (mean/m2/scratch/recip
survive scrambled-then-descrambled) while the loads replace L0-L3 between,
so the net effect is "transpose only the fresh loads".  Recips are pushed at
runtime per row: `TT_SFPLOADI(LREG7, UPPER/LOWER, bits(1/N))` — two
RISC-pushed instruction words per row (the test instantiates the no-LUT
path).  Impl-1 executes each row as `lltt::replay(6*input_lreg, 6)`.

## 3. Annotated disassembly

### 3a. Hand impl-1, one steady-state block (rows 5-8; `b5a8..b5e0` + 2 more rows)

```
b5a8: sfpload  L0,4,0,7      ; face0 rows 4-7, even cols   (prev block's 2nd TRANSP at b5a4)
b5ac: sfpload  L1,6,0,7      ; face0 rows 4-7, odd cols
b5b0: sfpload  L2,20,0,7     ; face1 rows 4-7, even cols
b5b4: sfpload  L3,22,0,7     ; face1 rows 4-7, odd cols
b5b8: sfptransp               ; 2nd half of the involution: L0-L3 -> one row each,
                              ;   L4-L7 (mean/m2/...) descrambled back
b5bc: lui  a7,0x717ad         ; RISC: build bits(1/5)          } per-row recip
b5c0: addi a2,a4,-436         ; RISC: SFPLOADI UPPER word      } delivery =
b5c4: sw   a2,0(a5)           ; push SFPLOADI(L7,UPPER,...)    } 2 pushed words
b5c8: addi t4,a7,-819
b5cc: sw   t4,0(a5)           ; push SFPLOADI(L7,LOWER,...)
b5d0: ttreplay 0,6,0,0        ; row 5: replay slots 0-5 (input L0) = 6 executed MADs/MOV
b5d4: addi a2,a4,-470
b5d8: sw   a2,0(a5)           ; 1/6 upper
b5dc: sw   t1,0(a5)           ; 1/6 lower (reused reg)
b5e0: ttreplay 6,6,0,0        ; row 6: replay slots 6-11 (input L1)
...                           ; rows 7, 8 likewise (slots 12-17, 18-23)
```

The first TRANSP of each block's involution pair sits at the END of the
previous row group (`b5a4`), so the steady-state stream reads
`TRANSP | 4 loads | TRANSP | 4 x (2 pushed loadi + 1 replay launch)`.

### 3b. Clean semantic impl-2, one steady-state block (rows 9-12; `b634..b6b4`)

```
b634: sfpstore L0,320,3,7    ; PARK mean  (fp32, physical dst tile 10)
b638: sfpstore L1,328,3,7    ; PARK m2
b63c: sfpload  L0,8,0,7      ; block gather, same four offsets as hand
b640: sfpload  L1,10,0,7
b644: sfpload  L2,24,0,7
b648: sfpload  L3,26,0,7
b64c: sfptransp               ; SINGLE transpose (no live values in L4-L7:
                              ;   the accumulators are parked in Dst)
b650: sfpload  L7,320,3,7    ; UNPARK mean
b654: sfpload  L5,328,3,7    ; UNPARK m2
; ---- row 9 (recip 1/9: not a power of two -> in-stream SFPLOADI pair)
b658: sfpadd   L6,L7,L0,1    ; delta = x0 - mean
b65c: sfploadi L4,-29127,2   ; bits(1/9) lower
b660: sfploadi L4,15843,8    ; bits(1/9) upper
b664: sfpmad   L4,L6,L4,L7,0 ; mean' = delta*recip + mean
b668: sfpadd   L0,L4,L0,1    ; x0 - mean'
b66c: sfpmad   L6,L6,L0,L5,0 ; m2 += delta*(x0-mean')   (delta retained in L6)
; ---- row 10 (1/10)
b670: sfpadd   L0,L4,L1,1    ; delta = x1 - mean
b674: sfploadi L5,-13107,2
b678: sfploadi L5,15820,8
b67c: sfpmad   L5,L0,L5,L4,0 ; mean'
b680: sfpadd   L4,L5,L1,1    ; x1 - mean'
b684: sfpmad   L1,L0,L4,L6,0 ; m2
; ---- rows 11, 12 identical shape (1/11, 1/12)
...
b6b4: (last mad)             ; then next block's PARK stores
```

Power-of-two rows (N = 2,4,8,16,32) drop the SFPLOADI pair for
`SFPMOV; SFPMULI(imm)` (the recip constant-folds into the 16-bit muli
immediate); the N=1 row folds away entirely (mean1 = x0, m2 = 0), which is
why block 1 is 31 instructions instead of 33.

## 4. Why the parks exist (the involution the compiler cannot spell)

Architectural SFPTRANSP permutes BOTH four-register banks.  The hand code
protects its live L4-L7 state with the raw TRANSP/TRANSP involution around
the loads.  Typed sfpi's `subvec_transp` lowers to the 4-operand
`rvtt_sfptransp_int` tuple, which sfpi-gcc's own rvtt.md marks DELIBERATELY
UNAUDITED because it under-states the write set (the L4-L7 companion writes
are not SETs of the PARALLEL); the audited complete form
(`rvtt_sfptransp8_int`) does not return the companion bank's results to C.
Consequence: any typed value the allocator leaves in the companion bank
across `subvec_transp` is silently lane-scrambled — this lane reproduced the
corruption on the corrected BH sim (welford trace n=5: mean's even columns
survive as the bank-transpose diagonal, odd columns swap with m2).  The
clean body therefore keeps mean/m2 in Dst scratch across every transpose:
2 fp32 stores + 2 fp32 loads per block of executed dst traffic.  Filed as a
compiler work item in AUDIT-DELREG.md; when a sound full-bank transpose
spelling exists, the parks (9 slots/block = 74 total, minus the 2 output
stores which stay) can be elided and the old ~230-slot body shape returns.

## 5. Per-row / issue-slot table (WELFORD_BODY zone, PERF variant)

Issue slots = instructions in the Tensix instruction stream (or pushed to
it); executed ops = what the SFPU actually retires.

| item (per 4-row block, steady state) | hand impl-1 | clean semantic impl-2 |
|---|---|---|
| block gather | 2 TRANSP + 4 LOAD = 6 issued/executed | 1 TRANSP + 4 LOAD = 5 issued/executed |
| accumulator park/unpark | 0 (raw involution protects L4-L7) | 2 STORE + 2 LOAD = 4 issued/executed |
| per-row compute | 1 `ttreplay` launch slot -> 6 executed (5 MAD + 1 MOV) | 5-6 issued=executed (2 ADD + 2 MAD + recip delivery) |
| per-row recip delivery | 2 RISC-pushed SFPLOADI words (+3-4 RISC lui/addi/sw off-stream) | 2 in-stream SFPLOADI (26 rows) / MOV+MULI (5 rows) / folded (row 1) |
| block total issued | 6 + 4x3 = **18** issued (of which 8 pushed) | **33** issued (31 in block 1) |
| block total executed SFPU ops | 6 + 4x8 = **38** | **33** |

| whole body (8 blocks, 32 rows) | hand impl-1 | clean semantic impl-2 |
|---|---|---|
| Tensix-issued static instructions | 80 (32 LOAD, 16 TRANSP, 32 replay launches) | 264 (48 LOAD, 18 STORE, 8 TRANSP, 70 ADD, 58 MAD, 52 LOADI, 5 MULI, 5 MOV) |
| RISC-pushed instruction words | 64 (SFPLOADI pairs; + ~58 RISC lui/addi and 64 sw on the scalar core) | 0 |
| executed SFPU ops | 304 (192 replayed + 32 LOAD + 16 TRANSP + 64 pushed LOADI) | 264 |
| replay-buffer usage | slots 0-23 recorded once at init, launched 32x | none in the body (init's record is linked for all impls, never launched here) |
| measured device anchor | 326 cycles (p150 baseline, 3x reproduced) | old l_reg body: 323; clean body: RE-MEASURE (pre-registered ~350-365) |

## 6. Honest account of the delivery structures (corrections to the earlier trace-build read)

The earlier read ("inline software-pipelined delta-retaining MADs with
constant-folded recips; the recorded block in its ELF is the HAND init's,
linked but never launched by sem") is corrected against the real perf body:

* **CONFIRMED — delta-retaining MADs:** the m2 update consumes the retained
  `delta` register (e.g. `b66c: sfpmad L6,L6,L0,L5`), exactly the
  `m2 += delta*(x - mean')` fusion; 2 ADD + 2 MAD per row.
* **CONFIRMED — the recorded block is the hand init's:** the sem perf ELF
  contains exactly one `ttreplay 0,24,0,1` (record-only, exec=0) from
  `_llk_math_welfords_sfpu_init_`, and ZERO replay launches in the semantic
  body.
* **CORRECTED — "constant-folded recips" is only 6/32 true:** only the
  power-of-two recips fold into `SFPMULI` immediates (5 rows) or vanish
  (row 1); the other 26 rows deliver bits(1/N) as an in-stream SFPLOADI
  pair — the same two words the hand path pushes from the RISC, just
  issued from the instruction stream instead of the push port.
* **CORRECTED — "software-pipelined" was an overstatement:** the perf body
  is straight-line per-row with only adjacent-row overlap from ordinary
  scheduling (the next row's delta ADD issues before the previous m2 MAD's
  consumer); there is no cross-block rotation.  The trace build (TRACE_N
  variants) additionally shows compiler replay COMPRESSION (`ttreplay
  24,4,1,1` record-exec + `ttreplay 24,4,0,0` relaunches, allocated after
  the hand init's slots 0-23) — that compression does NOT appear in the
  perf body.
* **NEW since the trace-build read — the parks:** the clean rewrite adds
  2 park stores + 2 unpark loads per block plus 2 final output stores
  (section 4); the old l_reg body had none (its accumulator pinning was the
  hand-ism this rewrite removes).  This is the entire expected perf
  regression vs the 323 record.

Structural summary: the hand kernel is a DELIVERY-optimized form — 18
issued slots/block, with 6-op row bodies amortized through the replay
buffer and recips streamed from the scalar core.  The clean semantic form
is an EXECUTION-transparent form — every op visible to and proven by the
compiler, fewer executed SFPU ops (264 vs 304: it never recomputes alpha
and moves no mean register), but 3.3x the issued slots, currently paying 9
extra dst slots/block for the transpose-bank hazard.  Closing the issued
gap needs (a) the full-bank transpose spelling (park elision, −74 slots),
and (b) compiler replay compression firing on the PERF body the way it
already does on the trace bodies (−~100 slots) — both generic mechanisms,
no welford-specific work.

## 7. Lane BF addendum (2026-08-18): the park elision landed

The compiler work item of section 4 is implemented (sfpi-gcc
agent/welford-win): `-mtt-tensix-optimize-transp-involution` forms the
TRANSP / 4x SFPLOAD / TRANSP involution as ONE atomic
companion-preserving instruction (the rename-through-permutation idea
composed to pi*pi = identity — a single transpose scatters whole-register
values at quarter-register granularity, so the involution is the only
whole-value-usable composition), proves or FORCES the all-lanes state a
sound involution requires (one materialized SFPENCC), and forwards all 16
Dst park store/load pairs through registers (bit-exact-format +
never-denormal producer audits; the final deposit stores survive).
Separately, the recording-epoch raw-word closure un-poisons replay
formation on the perf body (the hand init's raw record payload used to
refuse formation function-wide), and compression now fires there (two
4-slot record-execs in the free slots 24-31, relaunched in blocks 4-8).

Census at the full new ON set (this source, unchanged): 251 issued slots /
~267 executed SFPU ops (was parked 264/264; the OLD l_reg body — 323
measured — was 264/264).  Both mechanisms CRAQ 15/15 bit-exact at OFF and
ON.  Pre-registered prediction: generated ≈320, band 310-330, expected
small WIN vs hand 326 — the old 323-record class recovered on the clean
body.  The decisive-further-win route stays the parameterized counted-row
formation (recips excluded from the record and delivered between
launches; register-rotation via MVE-in-capture): the sequence matcher is
word-exact and only 8 replay slots remain (the hand init owns 0-23, and
stealing recorded-but-unlaunched-in-function slots would be unsound).
Evidence: ~/sfpi-uplift/laneBF-evidence-20260818/.
