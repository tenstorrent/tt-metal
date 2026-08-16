# AddInt/SubInt handwritten/generated A/B (conversion-queue row 10)

## Result

**LOSS, but a different mechanism than MulInt32.** The fresh typed-C++
implementation is correct on WH and BH CRAQ and on physical BH; its BH
math-isolate device time is +17.4% (add) / +26.1% (sub) versus the handwritten
sign-magnitude path. The compiler flag stack (dst-autoincr) closes -6.9%/-6.5%
causally; the macro planner refuses these rows byte-identically.

| BH TILE_LOOP MATH_ISOLATE cycles/tile (3 fresh procs) | handwritten | sem OFF | sem ON | causal | ON vs hand |
|---|---:|---:|---:|---:|---:|
| AddInt (Int32, sign-magnitude Dst) | 367.875 | 463.703125 | 431.78125 | -6.88% | **+17.37%** |
| SubInt (Int32, sign-magnitude Dst) | 367.921875 | 495.760417 (mean) | 463.773438 (mean) | -6.45% | **+26.05%** |

Handwritten OFF/ON `.text` is byte-identical (raw TTI path), so one physical
hand run fills both hand cells (identity recorded in the evidence manifest).
All samples deterministic across the three processes except two sub-cycle
tails on SubInt (495.75/495.765625; 463.796875/463.7265625).

## Where the gap actually is (falsifiable slot accounting)

Per unrolled row over the sign-magnitude Int32 Dst the production harness
drives (`_add_int_`/`_sub_int_` with `InstrModLoadStore::INT32`,
`SIGN_MAGNITUDE_FORMAT=true`):

- **Hand (BH raw TTI)**: 10 issue slots — TT_SFPLOAD, SFPCAST+SFPSETSGN,
  TT_SFPLOAD, SFPCAST+SFPSETSGN, TTI_SFPIADD, SFPCAST+SFPSETSGN, TT_SFPSTORE —
  three of them RISC-pushed (runtime dst addresses), plus one TTINCRWC.
  No replay, no SFPLOADMACRO.
- **Fresh typed (BH, `DataLayout::SM32`)**: 13 slots (add) / 14 slots (sub, one
  extra register-shuffle SFPMOV) delivered as one 13/14-slot REPLAY record per
  call and one launch per row. `sfpi::impl_::smag_to_int` lowers to a
  4-slot predicated sequence (SFPSETCC, SFPSETSGN, SFPIADD, SFPENCC) because
  `sfpi_lib.h` refuses BH's SFPCAST for the SM32->INT32 direction, while the
  hand path uses the 2-slot SFPCAST+SFPSETSGN form (the tt-llk-bh#16 errata
  workaround). ON leg: dst-autoincr fires (store mode 7->6, three owned
  SETC16 per call, all 32 per-row TTINCRWC eliminated) — exactly the measured
  ~32 cycles/tile causal gain. Zero SFPLOADI in the capture window.

**Contrast with MulInt32 (+98.16%)**: there the hand advantage is four
preprogrammed SFPLOADMACRO delayed templates (delivery). Here the loss is
dominated by *typed-API conversion lowering*, not macro delivery: with a
2-slot smag<->int conversion the fresh row is 9-10 slots, at or under the
hand row, before any macro formation.

## Mechanisms fired / refused (dumps archived)

- dst-autoincr: FIRED (both ops; the whole causal delta).
- Baseline replay formation: fires in both legs (13/14-slot row bodies).
- Macro planner: REFUSED byte-identically, named refusals
  `cc-template-unsupported` (rows contain SFPSETCC/SFPENCC from the SM32
  software conversion; same named refusal family as TTNN Where) and
  `row-not-closed`. Dump: `-fdump-rtl-rvtt_macro_planner` archives in the
  evidence directory.
- replay-hoist/invariant-loadi/latency-schedule/dst-iteration-fusion: no
  byte effect beyond the above on these bodies.

## Ordered generic next steps (no kernel-name peepholes)

1. **sfpi BH `smag_to_int` lowering** (`sfpi_lib.h`): adopt the proven 2-slot
   SFPCAST(SM32->INT32)+SFPSETSGN errata form instead of the 4-slot predicated
   sequence. Generic library/ISel change, value-independent; predicts
   near-parity rows for every sign-magnitude integer op on BH, and removes the
   CC writes that currently trigger the planner's `cc-template-unsupported`.
2. **Errata-scope note**: sfpi's BH `int_to_smag` uses bare SFPCAST with *no*
   SFPSETSGN workaround while the hand LLK applies it in both directions;
   correctness passed exactly on CRAQ and p150 silicon for the default Int32
   stimulus range, but the asymmetry should be resolved against the
   tt-llk-bh#16 errata scope before wider integer promotion.
3. **Planner CC-template extension** (shared with the TTNN Where close): only
   needed for macro-forming rows that keep CC writes; step 1 makes these rows
   CC-free first.

## Correctness and provenance

- Contract: exact integer equality (suite integer gate), identical seeded
  stimuli/golden per impl pair.
- CRAQ: 16/16 PASS ({add,sub} x {hand,fresh} x {OFF,ON} x {BH,WH}), generic
  descriptor-driven libttsim (bh
  `1fb30514fcab808539c8e4ae637c8fc94639c5c6345a85d65169afa90234081e`, wh
  `e043a0ed98bc61ed3c5afeab1e243282bcee49d01c614684752cf9b40aa22c78`).
- Physical BH (p150b): 6/6 correctness cells PASS; 24 serialized flocked
  device jobs total.
- Compiler: recalibrated pin, cc1plus sha256
  `33221397ebb22eefeecb91e9c579bdfe6336c86a97287b0205facdbcb68f5ce1`.
- OFF/ON flag sets: canonical post-WP8 (corpus README); planner leg
  `-mtt-tensix-macro-planner`.
- Evidence archive: `convert-addsub-evidence-20260816` (ELFs, `.text` hashes,
  disassemblies, planner dumps, CRAQ + silicon logs, raw/post CSVs,
  SHA256SUMS) on tt-quietbox-0.
- No production LLK file is modified; the selectors are test-only
  (`FRESH_CPP_IMPL`), and default builds are byte-identical (proven pre/post
  on all three TRISC ELFs).
