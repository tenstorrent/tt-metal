# REVIEW_RECORD — pin 27

cc1plus sha256: 981d8af93ff044da5cb564df76b2a1e4907c9780501cdb8f1c68f1f17defd8bd
driver (g++) sha256: 6f79329d0aa7ac26a7c9d9c5f36295d4a57ce8d26434eb6e6010edffb30e1b96
(CORRECTION 2026-08-23: the first ceremony commit wrongly carried pin-26's
driver sha as "unchanged" — the driver rebuilds with cc1plus (embedded
checksum) and the pin-27 install manifest records the new sha above; the
error was caught by the weekly preflight's DRIVER SHA MISMATCH fail-closed
guard on the first ON-28 weekly launch and corrected in the same-day
follow-up commit. Ceremony rule banked: read the driver sha from the
CURRENT install manifest at every pin, never assert "unchanged".)
source: sfpi-gcc nkapre/sfpi 0045d296318 (merge of lane GA
agent/hardsigmoid-residency bcf2ff308e9 + lane FZ agent/lcm-pricing
e08f9c7b291, both off pin-26 97f38861a94; clean merges). Built in
gcc-build-laneFR (build-pin27.log rc=0); installed via pin-install-fast
with loud --expect-cc1plus verification; no live sweeps at install; no
sfpi header changes; no .opt changes (both lanes ride existing flags).

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

- Lane GA (FX-F1 hardsigmoid residency regression, closing report
  2026-08-23): interfering mechanism = 3-step chain — EL's
  cc-restore-loadi widening hoists the loop constants to the preheader;
  const-residency's fusion + loop classes recognize only IN-LOOP
  constant loads (placement-sensitive recognition hole); combine's
  deliberate addi/muli immediate folds then eat the hoisted FLOATB
  sfploadi so the mad rule cannot fuse (loop decays 11 -> 12
  words/row). Drop-one isolation at pin 26 reproduced FX's pin-15
  witness BYTE-EXACT on drop-invariant-loadi; all other drops inert.
  Fix = MAD-PAIR class in gimple-rvtt-prgm-const.cc: pair-atomic
  re-claim of fold-vulnerable (SFPLOADI-FLOATB) single-use mul+add
  constants into PRGM registers at the hoisted materialization
  (pressure-class programming discipline + cc-region reach proof);
  recognition-only — the unchanged mad rule fuses; non-vulnerable
  chains left in LREGs (claiming would be a pure +1-word loss).
  Refusals: madpair-shared-constant, trip-count-single-trip,
  cc-region-unproven, madpair-prgm-exhausted (+ pair-atomicity
  witnessed, no-op twin). MAD-PAIR pricing audited in rvtt-cost.md.
- Lane FZ (lcm ON-28 pricing regression, closing report 2026-08-23):
  mechanism (a) PRICING chosen; (b) discharge REFUTED — the refusing
  dst-autoincr clause is lane ES's silicon-witnessed hang window
  (ES hung Tensix on lcm-fresh itself at 2-word distance; BH has no
  REPLAY functional model to admit anything inside the window), so
  narrowing it needs new silicon science, not an audit. Fix:
  rtl-rvtt-dst-autoincr.cc exports
  rvtt_dst_autoincr_hoist_capture_composition_p (exact mirror of the
  guard's window semantics; forward folded candidate scan; replay-row
  leads deliberately un-mirrored, documented scope bound);
  rtl-rvtt-replay.cc hoist_preheader re-record path refuses
  record-hoist-downstream-fallback-unprofitable when the hoist would
  force the mod-write fallback. No new constants (W_drain + word
  accounting reused from their single source); RECORD-HOIST x
  MOD-WRITE COMPOSITION entry in rvtt-cost.md.
- FZ-F1 harness finding (follow-up owed): corpus_leg_store.py keys
  entries by caller-env cc1plus but compiles with GEP from the
  --compiler realpath — hybrid legs MUST pass --compiler with the
  hybrid driver; the store should refuse on divergence. FZ's first
  fix legs were pin bytes under a fix key (caught by the vanished
  revert).

## Gates

- Union full rvtt.exp (dejagnu-pin27, srcdir at merged tip): 5876
  PASS (= pin-26's 5795 + GA's 48 + FZ's 33 exactly); FAIL set 16
  rows LINE-IDENTICAL to frozen (diff vs dejagnu-pin26/fail-set.txt
  empty).
- Lane GA gates (its build, closing report): dg madpair 48/48 +
  const-residency/prgm-const/invariant/cc-restore families 452/452;
  full rvtt.exp 5843 frozen-16 identical; corpus OFF + TRUE-DEFAULT
  3249/3249 identical; ON-28 delta = exactly ONE TU (sigmoid_appx
  fresh_cpp, adjudicated win: mul+sfpaddi -> sfpmad, CRAQ PASS); all
  EL/invariant-loadi win TUs byte-identical (keep-controls by
  identity). SILICON (BH p150, <=0.003% spread): hardsigmoid-fresh
  KERNEL 66359 -> 58420 (-12.0%), vs-hand +14.60 -> +0.89 (beats
  pin-15); sigmoid_appx 46135 -> 38199 (-17.2%); controls softplus +
  tanhderivlut cycle-exact before==after.
- Lane FZ gates (final build e49855142a77, closing report): dg
  record-hoist + dst-autoincr + autoincr families 536/0 (3 refusal
  twins incl. WH + renamed-varied, covered-fire >= W_drain control,
  no-autoincr gate control; FHD-5 re-keyed with coverage preserved);
  full rvtt.exp 5828 frozen-16 identical; corpus TD/OFF/ON-25
  3249/3249 identical; ON-28 delta = exactly ONE TU (lcm math.elf
  reverts to base-ON25 bytes fd8c5ac4 -> 08d62bac); 12 FY-pre extras
  + both sfpu_reduce_sdpa pack TUs byte-identical (v1 over-refusal
  caught by this gate and fixed); CRAQ lcm PASS pinned sim. SILICON
  (headline-laneFZ-20260823): lcm-fresh vs-hand +6.61 / causal -0.37
  — the FY regression fully reverted, gate <= +6.61 met exactly;
  controls blaze-sdpareducerow max-t8 +0.97 / sum-t8 +1.54 hold FY/FW.
- Ceremony extras cleared: lgamma-fitted v2 baseline anchors
  refreshed from lane GC's reviewed run (headline-laneGC-fitted-
  20260823, KERNEL-DELTA.md reviewed; 909375/874449 -> 479293/456635,
  vs-hand +233.35 -> +74.07); FINAL-BOARD bookings: hardsigmoid-fresh
  LOSS +14.60 -> PARITY +0.89 (lane GA), sigmoidappx +59.94 -> +32.42
  (lane GA bonus), lcm-fresh +7.55 -> +6.61 (lane FZ revert). Board
  tally 66W/21P/60L -> 66W/22P/59L.
- Evidence: ~/sfpi-uplift/laneGA-evidence-20260823/ (SHA256SUMS 271
  files), ~/sfpi-uplift/laneFZ-evidence-20260823/ (SHA256SUMS 35
  files), ~/sfpi-uplift/dejagnu-pin27/.
