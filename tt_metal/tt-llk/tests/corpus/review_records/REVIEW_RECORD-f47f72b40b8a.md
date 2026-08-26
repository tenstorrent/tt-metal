# REVIEW_RECORD — pin 29 (the ceremonied pin 29)

cc1plus sha256: f47f72b40b8a5de3896c8c292d53d6062ecf290184604e8c35ce24a22f2967d6
driver (g++) sha256: a08186799e97b69187055faa195279e557ab9fc5697e75f41c21853dd79e93f2
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 3f5af960548 = post-revert reconciled tip
d69d54fbd6e + five lane merges (GP agent/crossrow-pairing 81268d8febf,
GQ agent/crosscall-record-dstspill 7cce3f3255c, GU agent/fp16-6entry-lut
c8db4f9d26c0, GW agent/isa-unlocks-arecip-gtle 35f5cd0e6ea, GV
agent/pressure-licm-remat 41a8e30da5b). Companions: sfpi be48aa0 (GU-F1
P1 ICE fix — >=5-deep v_elseif chains; header STAGED into the install
include/ by rm-then-cp, compile-verified), tt-metal knobs through
85219a4aa5 (three knob-table both-append conflicts union-resolved,
py-parse + conf-lint verified each time). Built in gcc-build-laneFR
(build-pin29.log rc=0); all five new flags smoke-accepted together
(OPTCHECK); installed via pin-install-fast with loud --expect-cc1plus;
no live sweeps at install. NOTE: this is the CEREMONIED pin 29; the
failed CANDIDATE 29 (45ba7169, codex) remains audit history above the
entry — unratified, never canonically installed, reverted per lanes
GN/GO.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed commits/branches

Five lanes, each with its own closing report, evidence dir (+SHA256SUMS)
and REVIEW_RECORD/memory: GP (crossrow pairing, 12 twins, wrong-code
defect #3 in the codex prototype found+fixed by its own paired CRAQ),
GQ (record-hoist-peel, 5 twins, hazard shape never forms by
construction; lcm Dst-spill refused-by-scope with gate arithmetic),
GU (fp16 6-entry LUT: 6 encoding-table mode rows, 5-region matcher
flag-gated byte-identical off, exact re-encode, all-2^32 certs x3,
BH all-affine bit-exact unlicensed; GU-F1 P1 ICE fixed in sfpi; GU-F2
GS-census ADL artifact corrected), GW (SFPARECIP EXP/COND_RECIP with
the latent VB=0 wrong-code fix, doc==sim==silicon 4/4 bit-exact,
craq-sim ec15220f sim-gap fix = SIM PIN-BUMP CEREMONY OWED, pinned
oracle unchanged; native GT/LE compare 2^32x2-proven), GV
(pressure-park at the prgm-const layer, post-CC admission + LREG tier;
GT's callee-in-loop-constants gap closed for parking).

## Gates checked

- Union rvtt.exp WITH the full SFPI env (dejagnu-pin29): 6213 PASS,
  FAIL set 16 rows LINE-IDENTICAL to the pin-28 frozen baseline (diff
  empty). All five flags smoke-accepted together on the union driver.
- Per-lane corpus byte-gates vs the pin-28 stores: OFF/TD/ON-28 all
  3252/3252 .text-identical for every lane (Init(0) proven); knob
  deltas fully adjudicated + CRAQ green: GP exactly 2 TUs, GQ 29 TUs
  (1629/0 CRAQ), GU 9 TUs (= the 3 licensed bodies), GW 60 TUs
  (1629/0), GV exactly 4 TUs (3/4 sim-corr pass, 1 pre-existing
  flag-independent skip).
- Per-lane silicon (BH p150, 3-rep, corr-first, dual flocks; controls
  hold everywhere): sigmoidappx-fresh -9.44 W (GP); recip math-zone
  -34.28 / KERNEL neutral booked honestly (GQ); sigmoidlut -0.94 W,
  geluappx +6.25, tanhlut +0.56 P at 3.6x accuracy (GU); threshold
  -19.49 / hardshrink -17.76 knob-causal w/ booking convention
  deferred (GW); gelu-fresh -2.79 W, softplus +0.68, softsign +1.27
  (GV).
- Install: sha-verified 2a71feada1d9... -> f47f72b40b8a...; driver
  from manifest; sfpi include/ staged (GU-F1) and compile-verified.
- Board at cut: 75W/33P/50L (gelu-fresh W, sigmoidlut W, sigmoidappx
  family W booked this wave; geluappx/softplus/softsign/tanhlut
  improved-in-class).
- Open owner/ceremony items: sim pin-bump (ec15220f), GW booking
  convention at promotion, GQ-F1 gcd x unroll timeout triage, tanh-fresh
  arm preference.
- Evidence: laneG{P,Q,U,W,V}-evidence-20260825/ + dejagnu-pin29/.
