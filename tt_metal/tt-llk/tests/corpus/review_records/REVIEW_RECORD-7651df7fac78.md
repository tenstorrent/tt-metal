# REVIEW_RECORD — pin 43

cc1plus sha256: 7651df7fac781bad57ec68f375eeaf26284b98c5b867766e562c9a3e75bef59e
driver (g++) sha256: 57a1fac8fcc094c2aaa081d155de99130190bcbb8fe9bd0ca77a344aff16f874
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi d1e0ae6565a = pin-42 union tip 6e819875c01
+ lane IL merge (agent/lcm-window-density, fast-forward). Companions:
tt-metal chain through c2e5e6def125 (IL knob registration + measured
note). KNOB_MODES dup grep clean (only the known benign
lut-select-fp16 pair). No sfpi include/ changes. Built in
gcc-build-laneFR (build-pin43.log rc=0); flags smoke-accepted
(OPTCHECK, mcpu=tt-bh); installed via pin-install-fast with loud
--expect-cc1plus; no live sweeps at install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

One lane (closing report, evidence dir + SHA256SUMS 104, memory
file): IL RECORD-HOIST PLACEMENT LIFT
(-mtt-tensix-optimize-record-hoist-lift Init(0), composing on the
reviewed-ON record-hoist). The autopsy corrected the mission
premise: lcm's Stein loop is COUNTED (15 certified rounds; GIMPLE
fully unrolls it and the replay former already captures it
exec-while-record) — the window-density gap is the RE-RECORD
CADENCE: the sem body re-recorded 15 issue words every row (32
rows/call) while hand records once per kernel; the blocker was the
FZ downstream-fallback oracle being tried at the innermost preheader
only (record-hoist-downstream-fallback-unprofitable, distance 3<7).
The key fact was read from the oracle's own code: its
hazard-distance walk runs UPSTREAM and a path reaching function
entry is proven separated — so an outer dedicated preheader is
outside the refuted no-exec x mod-write class by the guard's own
distance semantics (exactly the hand init-record discipline). The
lift walks the placement outward and commits the UNCHANGED no-exec
hoist at the outermost admissible oracle-clean level. Hazard proofs:
every crossed loop proves replay-preserving under the record-hoist
interval walk (slot liveness across trips); placements must be
dedicated, recording-state-clean, and oracle-clean; the payload is
storeless by construction (the Dst-store mirror refuses first, so
the ES/FJ sweep rule-1 class never applies); FS
dominance/non-reachability holds on the preheader chain; sweep rule
2 re-audits the final placement; a failing level stops by name and
no-admissible-level keeps bytes. Pricing unchanged (the floor). A
first exec-conversion/junk-exec design with DF-liveness proofs was
retired once the oracle's direction was established — no new
execution semantics were introduced. The uncounted
(data-dependent-trips) admission rides lane FW's structural
trips>=1 proof, twinned with a runtime-trip do-while fire whose loop
control stays in-body per trip. 7 dg twins (fire, renamed-varied,
WH, uncounted-fire, alllevels-refuse near-miss, loopaudit-stop
refuse, default-off); record-hoist family 318/0.

RESULT: lcm's kernel entry carries ttreplay 0,14,0,1 with its 14
swallowed words ONCE PER KERNEL CALL; each row is 7 contiguous
launches + trim (per-row Stein issue 25 -> 11 words). SILICON (BH
p150, corr-first, 3 flocked reps, all rc=0): sem off 680400.0 x3 =
exact booked repro; knob 678213.0 x3 cycle-identical (-0.32%); hand
649518.0 x3 exact and byte-inert. Booked: lcm-fresh +4.75 -> +4.42
(still LOSS). RESIDUAL SPLIT NAMED, no closure cert: ~<=1.5pp
window-sizing successor (pick_replay's (clones-1)x(length-1)
preference — sem issues 7 launches/row of a 14-slot window vs hand's
4 of a 28-slot + partial-launch trim; 18 slots free, unlike IH's
slot-bound case) + ~3pp round-shape EXECUTION class (executed words
near-parity ~144 vs ~142/row; ctz-chain SFPSHFT+IADD-imm vs
SFPSHFT2, MAD shadows — lane IJ rename/schedule territory).

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin43; pinned-install
  env): 6987 PASS = pin-42's 6927 + IL's 60 exactly; FAIL set 16
  rows LINE-IDENTICAL to the pin-42 frozen baseline (diff empty; the
  3 pre-existing ERROR warts identical). Flags smoke-accepted
  (OPTCHECK).
- Corpus (corpus-legs-laneIL = the first pin-42 base store):
  base-vs-fix OFF/TD/ON-36 3300/3300 x3 .text-identical; knob delta
  = exactly 1 TU (the lcm corr TU), paired-CRAQ at both flagsets on
  the pinned sims + device corr; the lcm PERF node stays corr-gated
  per the HU long-perf precedent (stated in evidence).
- Preservation: topk-perf's booked leg (ON-36 + launch-flatten, both
  arms) BYTE-IDENTICAL under the knob; gcd-fresh BYTE-IDENTICAL;
  the 84-node headline screen exactly-one-changed.
- Board: 8eede5fe5915 -> 564b0693ea; tally 84W/36P/14L unchanged.
- Evidence: laneIL-evidence-20260828 (+SHA256SUMS 104).
- Install: sha-verified a7d856a79a6c... -> 7651df7fac78...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36 (record-hoist-lift registers as an on-plus
  booking knob).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin43-ceremony/).
