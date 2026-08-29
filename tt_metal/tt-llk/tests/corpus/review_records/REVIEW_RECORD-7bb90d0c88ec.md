# REVIEW_RECORD — pin 46 (the F1 honest-fix pin)

cc1plus sha256: 7bb90d0c88ec055c3a6666492188f14f735119add84c687e45db70f4f19ad468
driver (g++) sha256: cc2123de744a5e99113b4807e3254994ac4e536f76892452429d57834210d91d
(read from the CURRENT PIN-INSTALL-MANIFEST entry; the driver rebuilt
with cc1plus and REPRODUCED IDENTICAL to pin-45's driver)
source: sfpi-gcc nkapre/sfpi d8c4c71264a = pin-45 union tip 18318a7b5e4
+ lane IP's audit twins (agent/audit-ip-twins 88e9734614, incl the
IP-1 dead-twin repair) + lane IS (agent/f1-ambient-entry
f8b9c3deca3). Companions: tt-metal chain through a7e7c2e1af (IS
marker removal + DELREG correction) + ca92f915e3 (IP-5 dup fix) +
9ea702df9392 (IT same-leg re-books). KNOB_MODES dup grep: NONE (the
last benign-looking duplicate eliminated). No sfpi include/ changes.
Built in gcc-build-laneFR (build-pin46.log rc=0); OPTCHECK smoke;
installed via pin-install-fast with loud --expect-cc1plus; no live
sweeps at install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

THE F1 HONEST FIX (owner ratification
review_records-local/OWNER-RATIFICATION-F1-honest-fix.md, executed by
lane IS). The IQ audit's P1: the empty sfppushc(0)/sfppopc(0) pair
in 11 semantic sites lowered to one all-lanes SFPENCC doing TWO
load-bearing compiler jobs — the macro-planner's ambient all-lanes
enable AND a per-iteration CC write blocking dst-iteration fusion —
while also being an executed ~4.1k-cycle tax in rolled int loops.
The fix (ON-behavior, reviewed default pipeline):
entry_ambient_all_lanes_p (kill-aware backwards walk; fn entry
ambient per the structured-CC lowering contract; word-exact
all-lanes SFPENCC kills; calls/asm/opaque dirty, fail-closed; on
success formation SYNTHESIZES the canonical enable, refusal
all-lanes-proof-missing (ambient-entry-unproven)); immediate-delta
fused-row admission (absolute-progression proof, absorbed-stride
calendar, emission normalized word-identical to unfused); and the
opcode_l16_target_proven gate closing a LATENT derive-core
wrong-code (an LReg16-staged SFPABS template producing device corr
FAIL — pre-existing, unreachable pre-F1, caught by the lane's own
corr-first discipline; the re-derived VD-direct calendar passes
device+oracle and is faster). All 11 marker sites deleted; the
false AUDIT-DELREG (iii).5 exemption corrected. 18 IS twins + 5 IP
adversarial twins (incl the shared-reload fingerprint probe: fires
with every production coefficient replaced).

PER-ROW RE-MEASURES (anchors reproduced the booked cells at pin-45
first; 3-rep corr-first same-leg):
- SURVIVE EXACT (byte-identical streams): signbit -5.69,
  unarymaxmin-max/min -41.90 x2, mulint32-fresh -4.65.
- DEEPEN: isclose-fresh -14.37 -> -18.19 (its pin drift
  re-anchored), gcd-fresh -19.15 -> -19.53, subint -34.60 -> -49.97
  and binarybitwise -19.26 -> -33.17 (new fused-row admissions),
  lcm comp leg +4.42 -> +3.79.
- FLIP: absint32 int-abs knob leg PARITY +0.23 -> WIN -5.31 (after
  the L16 guard).
- DIE HONESTLY: minmax-max/min -5.03 -> LOSS +36.0 x2 (formation
  admissible but refuses on pricing without the marker; the macro
  form itself measures 17451 vs the 25000 now booked — successor:
  init-hoist-aware run pricing, a recoverable pure pricing gap);
  typecast -5.10 -> LOSS +13.27 (ambient-entry-unproven at the
  opaque LLK init — successor: audited-TU walk transparency).
BOARD 83W/36P/15L -> 81W/35P/18L: the three new losses carry NAMED
SUCCESSORS (not certificates) — the zero-loss-or-cert state is
honestly re-opened, and every surviving win is clean of the marker
signal.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin46; pinned-install
  env): 7159 PASS; FAIL set 16 rows LINE-IDENTICAL to the pin-45
  frozen baseline; dg ERROR count == 0 (the IP-1 rule now part of
  the gate). Flags smoke-accepted (OPTCHECK).
- Corpus: base-vs-fix deltas = exactly the 17 marker TUs + 3 new
  admissions (7 marker TUs byte-identical at ON-36 = the exact
  survival set); the L16 guard footprint = 1 TU; paired CRAQ 20/20
  + 8/8 + finals on the pinned sims (bh 1d162f0adf67); device corr
  extras 8/8; the 84-node screen deltas all adjudicated (typecast
  hand = profiler zone-id LUIs only).
- Board: 84032325c0f3 -> 876d448f80c3 (13 rows booked by IS).
- Evidence: laneIS-evidence-20260829 (+SHA256SUMS 15426), with the
  audit trio's records in laneIP/IQ/IR-evidence-20260829.
- Install: sha-verified 1f2b3baf48b4... -> 7bb90d0c88ec...; driver
  reproduced identical (manifest-read).
- ON set UNCHANGED at 36 (the fix is reviewed default-pipeline
  behavior — the full ON-delta adjudication above is its review).

## Witness re-seat (the pin-11 rule, exercised honestly)

Three R9 witnesses (macro-planner, drain-schedule, init-hoist) rode
the binary max_min node — the exact row the F1 fix honestly killed —
and went RED at the first preflight. Re-seated: macro-planner onto
the surviving unarymaxmin node; drain-schedule and init-hoist onto
the mulint32 node (positive control confirmed both lines fire there
at full ON). All three re-verified TWO-SIDED: line present at the
union, ABSENT at union-minus-flag (rc=0 compiles, non-vacuous — a
first vacuous-negative attempt with rc=4 was caught and discarded).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary, after the witness re-seat (outputs in
~/sfpi-uplift/sweep-2x2/pin46-ceremony/).
