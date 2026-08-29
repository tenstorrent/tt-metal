# AUDIT-DELREG — corpus-wide de-lreg audit of the semantic bodies under tests/

Lane BB (agent/welford-clean-rewrite), 2026-08-18.  Pin context: tt-metal
eed9c37961 base, cc1plus ac81ede3827d…, corrected BH sim 32489dda….

## Charter

After the LLK-pristine ruling (conf R7, tt-metal 78cf0f8581) reverted the
trusted `__builtin_rvtt_sfprawlreg_access` annotation out of the handwritten
`ckernel_sfpu_welfords.h`, the semantic welford variant silently went
numerically wrong: its vfloat body read `sfpi::l_reg[LReg0..3]` populated by
the hand load-block's raw TTI SFPLOADs, and pinned its accumulators in
`l_reg[LReg4/5]`.  Without the annotation the compiler has no reason to
believe L0-L3 are live-defined across the raw boundary and clobbers them.
This audit sweeps every semantic body under `tests/` for the same hand-ism
classes:

* `sfpi::l_reg` reads/writes,
* `p_sfpu::LREG*` references inside semantic bodies,
* raw `TTI_*`/`TT_*` instructions inside semantic bodies,
* hand-helper calls that impose register contracts on typed code,
* leftover annotation-class builtins (`__builtin_rvtt_sfprawlreg_access`).

Classification per item: (i) hand-ism -> rewrite to plain typed C++;
(ii) boundary contract with hand context that cannot be removed without LLK
edits -> documented here with the exact contract; (iii) executed-instruction
semantics -> keep, justified.

Scan method: pattern grep over `tests/sources/*.cpp`, `tests/helpers/include/*.h`,
`tests/corpus/*.cpp` (BH scope; `sources/quasar/` excluded — no BH sweep rows),
then manual read of every hit in context.  Hand-implementation code paths
(impl selectors 0/1, raw wrapper headers such as `helpers/include/sfpu_operations.h`)
are hand code under test, not semantic bodies, and are out of scope by
definition.

## Inventory

### (i) hand-isms — REWRITTEN (this lane)

1. **`sources/sfpu_welford_prefix_snapshot.cpp` — vfloat impls 2/3/4** (the
   `welford` sweep row's generated arm).
   *Was:* inputs read from `sfpi::l_reg[LReg0..3]` behind the hand
   `_welfords_load_block_`'s raw TTI SFPLOADs; running mean/m2 pinned in
   `l_reg[LReg4/5]`; capture via hand `TTI_SFPSTORE(p_sfpu::LREGn, ...)`.
   *Now:* inputs gathered with `sfpi::dst_reg[]` at the algorithm's own Dst
   offsets + typed `sfpi::subvec_transp`; mean/m2 are plain `vFloat` locals
   carried across the whole body; `1/N` is a `constexpr` per-row constant;
   the trace capture stores the typed locals themselves to the same trace
   Dst slots; the body ends by depositing mean/m2 in Dst scratch (its output
   contract — also keeps the perf build's last block live).
   *Gate:* CRAQ 15/15 PASS at pinned OFF and full ON flag sets (corrected BH
   sim); vfloat numerics now IDENTICAL to the hand impls at every traced n.
   *Bytes:* generated arm's binaries changed (by design) — the row is listed
   for re-measurement below.  Hand impls byte-identical (corr exact; perf
   differs only in the two WELFORD_BODY zone-id `lui` immediates from the
   source-line shift).

   **Residual (justified, class iii):** the accumulators are parked in
   private Dst scratch (fp32 addr space 320/328 = physical dst tile 10)
   across every `subvec_transp`, because architectural SFPTRANSP permutes
   BOTH lreg banks while sfpi's 4-operand `subvec_transp` models only its
   operands — sfpi-gcc `rvtt.md` marks the 4-operand tuple "DELIBERATELY
   UNAUDITED" for exactly this under-claimed write set.  The hand code
   survives the same instruction with a raw TRANSP/TRANSP involution that
   typed sfpi cannot spell (`__builtin_rvtt_sfptransp8` models all eight
   writes but returns only the first bank's results to C).  The parks are
   executed dst traffic, not compiler hints.
   **Compiler work item (filed):** give typed sfpi a sound full-bank
   transpose spelling (return the second bank's results, or auto-emit the
   involution pair around a live companion bank), then elide the welford
   parks.  Until then any typed kernel that keeps values live across
   `subvec_transp` in the companion bank is silently corrupted — checked by
   CRAQ only if the traced build exercises it (the welford n>=5 traces do).

### (ii) boundary contracts with hand context — DOCUMENTED, not removable without LLK edits or a dedicated re-measured rewrite

2. **`sources/sfpu_reduce_sdpa_test.cpp` — `calculate_reduce_max_col_subblock_4x2_generated`**
   (the `reduce-sdpa` pinpair row's generated arm — the flagship measured
   win, 834 vs 840).
   Exact contract: the pristine LLK helpers
   `sfpu_reduce_max_col_subblock_4x2_load_initial_values()` (raw TTI, loads
   -inf into L4-L7), `_reduce_max_col_subblock_4x2_prologue_/epilogue_` and
   `_move_to_next_subblock_4x2_` (raw) bracket a typed middle that carries
   the four column accumulators in `l_reg[LReg4..7]`
   (`generated_load_max<Accumulator, Offset>`), plus raw
   `TTI_SFPSWAP(0, LREG0/1, LREG4, 1)` result folds where L0/L1 carry
   prologue state.  The register placement IS the ABI with the un-editable
   hand phases, so the l_reg pinning cannot be removed without either LLK
   edits (forbidden) or absorbing prologue/epilogue/load/swap into the test
   source (a full rewrite of a measured flagship row — separate lane +
   silicon re-measurement required; NOT done here).
   **ESCALATION — trusted annotations still live in tests/:** this file
   carries three `__builtin_rvtt_sfprawlreg_access` annotations
   (`(0, 0xf3)` twice, `(0xff, 0)` once).  They are annotation-class trusted
   claims ("L0/L1 live-in, L4-L7 raw-defined"), the same builtin the
   LLK-pristine revert removed from `welfords.h`, surviving because they
   live under tests/ where R7 does not reach.  They are load-bearing:
   deleting them without the rewrite breaks this row exactly the way welford
   broke.  Under the standing "no trusted annotations anywhere — the
   compiler proves or refuses" rule these must eventually be retired by
   either (a) the D2-class compiler derivation of raw-TTI LREG defs
   (rtt effects on decoded raw words — Lane BD's decode direction), or
   (b) the dedicated rewrite-and-remeasure lane.  Owner decision required;
   flagged, not silently patched.

3. **`sources/sfpu_binary_bcast_test.cpp` — `generated_binary_lregs` /
   `generated_col_band`** (the `binary-bcast` pinpair row's generated arm,
   measured exact parity 608/608).
   Exact contract: a typed arithmetic island reads
   `l_reg[{LReg1,LReg3,LReg4,LReg5}]` (data) and `l_reg[LReg0]` (broadcast
   scalar) that were populated by raw `TT_SFPLOAD` + an `lltt::replay` of
   the hand broadcast stage, writes results back to the same lregs, and pins
   them live through the raw `TT_SFPSTORE` consumers with
   `sfpi::l_reg[..].in_use()`.  The row note (Lane AZ) already books this
   deliberately as hand-structured (all four selectors OFF/ON
   byte-identical; kind=pinpair for that reason).  The fixture comment
   states the design intent: "addressing, transpose, replay and stores
   remain architectural LLK operations".  A de-lreg rewrite is possible
   entirely inside the test (loads/stores are test-source raw ops, not LLK),
   but converts a measured-parity row into new bytes -> conversion-queue
   item with mandatory re-measurement; not done in this lane.
   No annotation builtins here; the `.in_use()` liveness pins are sfpi's
   documented fixed-lreg liveness API (register-contract class, but visible
   to and honored by the compiler, not a trusted effect claim).

4. **`corpus/topk_typed_index_tracking_probe.cpp`** — l_reg quad
   (`LReg0..3`) around `subvec_transp`/sorting plus `.in_use()` pins on
   `LReg4..7`.  Unwired diagnostic probe (no sweep row); its conversion
   blocker is already documented in `corpus/TOPK_TYPED_CONVERSION_BLOCKER.md`.
   Note: as a transpose-adjacent l_reg user it is also in the blast radius
   of the `subvec_transp` companion-bank hazard documented in item 1 —
   whoever wires it must CRAQ with live companion-bank state.

### (iii) executed-instruction semantics — KEPT, justified

5. **RESOLVED 2026-08-29 (lane IS, owner-ratified F1 honest fix)** —
   the `__builtin_rvtt_sfppushc(0); __builtin_rvtt_sfppopc(0);` pairs
   formerly kept here are DELETED from all 11 sites (this file's original
   5 in `fresh_cpp_operations.h`, the typecast driver, and the 5
   fresh_cpp headers the idiom had spread to: gcd, isclose, lcm,
   lcm_legacy, mul_int32_limb2).  This item's original justification
   ("executed semantics, not hints … establishing the all-lanes
   predicate boundary") was adjudicated FALSE by the lane-IQ adversarial
   audit (AUDIT-IQ F1, laneIQ-evidence-20260829): the pinned compiler
   NEVER emits the pair (pass_rvtt_cc lowers the outermost pair to one
   all-lanes SFPENCC), a mod1=0 push+pop is architecturally a pure
   identity on lane predication (it could not establish all-lanes even
   if executed), and a shim A/B proved the lowered SFPENCC was a
   LOAD-BEARING COMPILER SIGNAL gating macro-planner formation and
   blocking dst-iteration fusion.  Owner ratification
   (review_records-local/OWNER-RATIFICATION-F1-honest-fix.md, remedy a):
   the compiler now derives the fn-entry ambient-all-lanes fact itself
   and synthesizes the canonical enable where a formation needs one
   (sfpi-gcc agent/f1-ambient-entry: entry-ambient kill-aware walk +
   immediate-delta row admission), the sources carry NO marker, and
   every affected booked row was re-measured 3-rep same-leg and booked
   as it measured (lane IS evidence, laneIS-evidence-20260829).  The
   fresh_cpp/README "no markers" contract now holds on its face.

6. **`lltt::setrwc<...>()` typed face boundaries** in
   `sources/eltwise_unary_typecast_test.cpp` (adopted probe form) — typed
   builtin spelling of the SETRWC face advance; RWC effect is
   compiler-visible.  Executed semantics.

7. **`::_llk_math_eltwise_sfpu_inc_dst_face_addr_()` calls** in
   `helpers/include/fresh_cpp_operations.h` (binary max/min, add/sub int
   face loops).  Post-revert these LLK wrappers are raw TTI face advances
   (SETRWC/SETC16 class) — real Dst counter movement the algorithm needs.
   The compiler currently treats them as opaque run separators; Lane BD's
   raw-boundary-decode (architectural field decode of the raw face-advance
   words) is the standing recovery item.  No register contract is imposed
   on typed values (the bodies' locals do not cross the boundary).

8. **Harness scaffolding raw TTIs outside semantic bodies** —
   `TTI_CLEARDVALID(1,0)` in `sources/sfpu_reduce_sdpa_perf.cpp` (SrcA
   valid-flag drop between measured iterations), `TTI_STALLWAIT` fences in
   test run_kernel prologues, the welford test's hand capture
   `TTI_SFPSTORE(p_sfpu::LREGn, …)` (impl 0/1 trace ABI — it snapshots the
   HAND implementation's registers; that is its purpose).  All are either
   hand-code-under-test or thread-sync scaffolding, not semantic bodies.

### Clean (scanned, zero findings)

`helpers/include/fresh_cpp_operations.h` bodies (exp, sigmoid cubic,
sigmoid tree, signbit, binary max/min, unary max/min float/int, add/sub
int, addcmul) — no l_reg, no raw TTI, no annotations;
`sources/sfpu_ternary_test.cpp` `calculate_where_generated` (pure typed
U16/U32 bitwise-select, v_if/v_endif);
`sources/eltwise_unary_typecast_test.cpp` semantic body (typed converts);
`sources/sfpu_sdpa_exp_unclamped_test.cpp` semantic impl (typed, calls the
typed `_sfpu_exp_21f_bf16_*` math from the pristine experimental header);
`sources/sfpu_binary_test.cpp`, `sources/eltwise_binary_sfpu_perf.cpp`,
`sources/eltwise_unary_sfpu_perf.cpp`, `sources/sfpu_ternary_perf.cpp`,
`sources/eltwise_unary_typecast_perf.cpp`.

## Counts

* hand-isms rewritten: **1** (welford vfloat impls 2/3/4 — the assigned fix).
* boundary contracts documented: **3** (reduce-sdpa generated arm,
  binary-bcast generated arm, topk probe).
* kept-with-justification executed-instruction sites: **3 classes**
  (lltt::setrwc, inc_dst_face_addr wrappers, harness scaffolding), plus
  welford's subvec_transp + dst parks.  (The former 4th class —
  pushc/popc pairs — was RESOLVED-BY-DELETION 2026-08-29; see item
  (iii).5.)
* leftover trusted annotations found: **3** (all in
  `sfpu_reduce_sdpa_test.cpp`; escalated above, deliberately not deleted —
  removal without rewrite reproduces the welford breakage on the flagship
  row).

## Rows needing re-measurement (bytes changed by this lane)

* `welford` (generated arm only): corr + perf binaries changed.  p150
  baseline `generated` cells (323) measured the old l_reg-coupled body.
  Pre-registered expectation for the clean body: ~350-365 cycles (steady
  state 33 issued slots/block x 8 + 2, all in-stream; the +9 slots/block
  park+unpark overhead is the price of the subvec_transp hazard) — a LOSS
  vs hand 326 until the compiler work item lands.  First re-measure books
  via reviewed baseline update; old cells untouched.  Hand cells need no
  re-measurement (timed stream byte-identical; zone-id lui shift only,
  which correctly invalidates the perf cache).

No other row's binaries were touched by this lane (hand welford impls
byte-identity-proven; no other source edited).
