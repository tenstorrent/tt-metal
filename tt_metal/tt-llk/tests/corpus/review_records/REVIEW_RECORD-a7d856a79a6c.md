# REVIEW_RECORD — pin 42

cc1plus sha256: a7d856a79a6c4a3a2ef786f1fffe297df3d5dfcbb65357348e2b332a24c51791
driver (g++) sha256: f0ed73491fd46ec0bfa44c2065e6796b1eec8dff99543a4257fd88e753738d4c
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 6e819875c01 = pin-41 union tip ccfc4ccabee
+ two lane merges (IK agent/binopscalar-addrmod 2e207b59987, IJ
agent/trig-chain e12d8998e26). Companions: tt-metal chain through
ffb37a6b6b (IK knob 54674d12895a + IJ knob registration c82652ca3b +
measured notes). KNOB_MODES dup grep clean (only the known benign
lut-select-fp16 pair). No sfpi include/ changes. Built in
gcc-build-laneFR (build-pin42.log rc=0); flags smoke-accepted
(OPTCHECK, mcpu=tt-bh); installed via pin-install-fast with loud
--expect-cc1plus; no live sweeps at install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

Two lanes, each with its own closing report, evidence dir
(+SHA256SUMS) and memory file:

IK (-mtt-tensix-optimize-crosscall-addrmod Init(0)): once-per-kernel
ADDR_MOD slot programming. ISA adjudication first: ThreadConfig
ADDR_MOD rows are per-thread and writable only by same-thread SETC16
(the WRCFG/CFGSHIFTMASK/RMWCIB functional models each exclude
ThreadConfig), so the slot program is kernel-lifetime stable modulo
a same-thread SETC16 write; BH's 3-bit modifier selects slots
directly, WH reaches slot 6 only via ADDR_MOD_SET_Base=1 — handled
as a refuse-only watch row. The contract reuses lane CA's init-hoist
machinery on the ADDR_MOD face plus lane HC's residency walk (a
3-level lift to run_kernel entry fires on the real TU), with
stage-2-or-refuse, a preheader-occupancy belt, and
statement-identity contract-call admission (the constprop-clone
mis-refusal found and fixed). The callee side fires only when all
groups refuse under lane IA's pricing, single-stride, explicit rows,
a WHOLE-CALLEE slot-clobber census clean (covering the callee tail =
the next call's clobber), the entry-distance guard, and rows
exceeding the call-boundary W_drain charge (6-refuse/7-fire boundary
twins); fired groups price config at zero (preheader-class per IA's
placement split). binopscalar +1.93 -> PARITY +0.08 (sem 20781.0 x3
cycle-identical vs hand 20764.0; anchors exact; TILE_LOOP 159.98 <
hand 160.04 < base 162.95 — the fired form beats both arms per
tile; TTSETC16 18,0/34,2/53,0 once per kernel; callee 27->24
words/call). 14 dg twins incl the slot-clobber near-miss;
fill/copydest family byte-identical incl the load-carrier
composition; 70-leg screen exactly-one-changed.

IJ (-mtt-tensix-optimize-cyclic-region-schedule Init(0)): the trig
chain-bound arithmetic (the DT/EI oracle style) proved EQUAL
recurrence circuits (sem RecMII>=88 == hand's) — the +1.66 gap was
scheduling (sem achieved-II>=94 vs hand's >=92; the sem-only stalls
sit in one interior region with the natural fillers
register-serialized on L5; every in-tree pass's inability to fire is
dump-proven by name). Fix: interior regions of a multi-region
self-loop row are re-list-scheduled under the established
vocabulary, accepted only on a STRICT WHOLE-ROW CYCLIC-II DECREASE
over every issued word (barrier words fixed, bit-exact by
construction; raw CP-greedy acceptance would have regressed 94->95 —
the cyclic acceptance is load-bearing). Delivered oracle II 94->93 =
the region-local model floor (a 6000-trial search agrees).
trigonometry-fresh +1.66 -> +0.63 (387641 vs booked hand 385199,
causal -4.69; the composition matrix is monotone with the triple
best; the same-leg hand 381105 noted per convention) with a
CONSTRAINED-FLOOR certificate: the last slot vs hand requires a
rename with no free LREG (the 8-LREG wall) or a value-changing
circuit restructure (muli+add -> fused mad = licensed class or a
fresh-source edit) — named successors. 6 dg twins incl a WH fire
twin; knob delta = exactly 12 corr TUs, all attributed via
keep-build and paired-CRAQ PASS at hash-verified binaries.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin42; pinned-install
  env): 6927 PASS = pin-41's 6834 + 93 new checks, ALL in the IK/IJ
  twin files (solo-branch counts were 79 + 12; the union emits 2
  additional conditional checks within those same files —
  pass-list diff verified zero strays); FAIL set 16 rows
  LINE-IDENTICAL to the pin-41 frozen baseline (diff empty). Flags
  smoke-accepted (OPTCHECK).
- Both lanes: corpus OFF/TD/ON-36 base-vs-fix 3300/3300 x3
  .text-identical; 71-leg loss+WIN screens and the 9-pair seed
  screen 0 CHANGED; paired CRAQ green on the pinned sims (bh
  1d162f0adf67); device corr corr-first every session.
- Silicon (BH p150, 3-rep cycle-identical): binopscalar PARITY
  +0.08 (IK); trigonometry-fresh +0.63 (IJ). All anchors reproduced
  exactly; IK's booking and IJ's booking serialized cleanly
  (pre-state re-asserted on the mid-lane board move).
- Board: dfed90cb3f3a -> 8eede5fe5915; tally 84W/36P/14L.
- Evidence: laneIK-evidence-20260828 (+SHA256SUMS 674),
  laneIJ-evidence-20260828 (+SHA256SUMS 1002).
- Install: sha-verified d8046e29c2b3... -> a7d856a79a6c...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36 (both new flags register as on-plus
  booking knobs).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin42-ceremony/).
