# REVIEW_RECORD-8e87fba0e35f — silicon authorization for the pin-13 build

Pin: cc1plus sha256 `8e87fba0e35f2a2b4a80981310afc3601a1ce34131518dea1b62c6afc1b030d5`
Built from: sfpi-gcc `8ae4a2d6b01` (staging/pin13 union, base 7b4e4d96fb6) via sfpi
(`scripts/build.sh`; log `~/sfpi-uplift/toolchain-rebuild-pin13.log`; stage2 stamp +
build dir removed first, cc1plus sha VERIFIED CHANGED from pin-12 `352aee85ec92…`;
all three new flags accepted by the installed driver).
Date: `2026-08-19`
Reviewer: orchestrator session (Claude, session a64aef93, operated by nkapre@tenstorrent.com)
Independence: NOT independent — the reviewer spawned and reviewed all six lanes and the
batch merger. The batch merger ran the full adversarial union gates independently of the
lanes. An independent re-review should supersede this record.

## Reviewed commits/branches
- sfpi-gcc `8ae4a2d6b01` = pin-12 base 7b4e4d96fb6 + merges (in order): BV
  reduce-sdpa re-record pricing split (3d8c1fc1aef), BW counted-row
  multidef refusal + replay-owner word-exact barrier (f071f551eb5), CC
  upward-IMS carrier former (67830bb5896, flag OUT of the measured ON
  set — no real-kernel fire), CD crossloop-hoist (03549943022), CA
  drain-backedge elision + init-hoist (3af885d073c), CF const-residency
  CC-canonical peel + TU value-identical reuse + SFPIADD-imm/SFPDIVP2
  audits incl. the repeat-raw-claim soundness fix (66332ec60dd).
- tt-metal batch: coupled conf branches merged (crossloop-hoist-conf,
  drain-completion-conf); ON set = 25 flags (adds crossloop-hoist,
  init-hoist); pin values + PIN HISTORY #13 + witness-table update
  (counted-row books 6 launch sites — the pin-12 drift re-reviewed and
  resolved, BW verified welford math .text byte-identical pin-vs-fixed).

## Gates checked (batch merger, evidence ~/sfpi-uplift/pin13-evidence-20260819/ + SHA256SUMS)
- Full rvtt.exp at the union (fresh drivers): 3419 PASS; unexpected-FAIL
  set byte-identical to the frozen-9 (one run-1 shim artifact fixed and
  archived).
- Focused families 0-FAIL: replay-hoist 123, counted-row 26, crossloop 45,
  ims-carrier 41, drain-backedge 33, init-hoist 47, const-residency 56,
  prgm-const 120, invariant 201, macro-planner 975.
- Corpus flags-off byte-identity vs pristine same-recipe base: 5462/5462
  identical (frozen worktree at 7e8319be1f).
- Union ON inventory: 163 changed / 5462 across 29 rows — ALL attributed
  per lane (incl. 8 pin-12 ICE victims now compiling); CF shard-00
  ruling: attributed byte-exactly. Zero unattributed.
- Fire witnesses verbatim on the union build: BV benefit-620 hoist, BW
  counted-row 6-launch-site + multidef refusal, CD crossloop hoist on
  exp (+ expm1_cw/exp2/i0), CA init-hoist stage-2 + drain-backedge
  elision, CF residency peel + log 23->17 replay; regression witnesses
  (prgm-const, ccmask, crosscall, drain-schedule, planner, residency)
  all present. BVxCF interaction: hardsigmoid/isclose pricing refusals
  preserved on the union.
- CRAQ (pinned BH sim 32489dda): 53/53 PASS at OFF and union-ON.
- Installed-binary verification (this record): cc1plus sha changed,
  driver cf7c8f15d13e…, three new flags accepted; witness preflight at
  the installed binary runs with the enforcement suite before the sweep.
- Model note carried: CF's lcm near-threshold replay fire (modeled
  +135 cs, band 64..148) measured +0.34% silicon-neutral — MIN_BENEFIT
  band review queued.
- NOT checked: no pin-13 silicon exists yet (this record authorizes the
  first run); pre-registered expectations: reduce-sdpa ~832.75 restored,
  minmax ~16.72 WIN, where ~154.5 WIN, exp ~73.8, sdpa ~928, log ~74.3,
  sqrt ~110.0, rsqrt ~114.1, welford 322 WIN reproduced, tanh/sdpafw/
  sigmoid/celu/elu/mish/selu first cells at the union ON set.
