# REVIEW_RECORD — pin 39

cc1plus sha256: 287a307f4836556b0a017205d64133f14c6747af6371b0b3cef428343c9519cc
driver (g++) sha256: 0d9551fb06b8eea53439e448fbbcfb01ab310186c8477dda0e8f7cf6857eae0e
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 07835727b3d = pin-38 union tip 9290941b633
+ lane IF merge (agent/dst-autoincr-load-carrier, fast-forward).
Companions: tt-metal chain through 2623bcb601d1 (IF knob registration;
IE's blaze twins 8e04a1c9ed merged earlier). KNOB_MODES dup grep clean
(only the known benign lut-select-fp16 pair). No sfpi include/
changes. Built in gcc-build-laneFR (build-pin39.log rc=0); flags
smoke-accepted (OPTCHECK, mcpu=tt-bh); installed via
pin-install-fast with loud --expect-cc1plus; no live sweeps at
install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

One lane (closing report, evidence dir + SHA256SUMS 406, memory
file): IF dst-autoincr LOAD-CARRIER
(-mtt-tensix-optimize-dst-autoincr-load-carrier Init(0)). The
counter-probe moved the mission premise: LOAD-terminated rows
already fire in record-free functions (classify_access has carried
the load builtin all along) — the true blocker was
occupies_replay_slot_p counting raw .ttinsn asm words as ZERO replay
slots, so the LLK envelope datacopy record's 16-raw-word shadow
overran its block and bailed the whole function before candidates
ran. Fix: the audited rvtt_raw_ttinsn_word extraction wired into the
slot counter — one raw constant word = one replay slot = one
frontend word, COUNTING ONLY (raw words stay AIC_FOREIGN, never
payload-rewritable/gap-legal/config-window-legal; all lane-IA
pricing, placement-split, and payload-family rules untouched and
composed). Replay soundness PROVEN with no per-launch reset needed:
per the WH REPLAY/INCRWC/RWCs functional models (the BH doc gap
defers to the laneFS silicon model + pinned sim), RWC/ADDR_MOD
effects are per-execution cumulative — the only skew mechanism is
executions != removed increments, which is exactly the pass's
existing fail-closed payload-coverage refusal (the walk-skew twin
exercises it on a LOAD payload). Capturing TTINCRWC words inside
replay windows was examined and REFUSED by design (it would steal
the autoincr fold and reproduce lane IE's measured-worse envelope
family). rvtt-cost.md carries the LOAD-CARRIER audited section.
Five dg twins (rawrecord-fire bh+wh, rawrecord-bail-refuse knob-off,
walk-skew refuse, renamed-varied). ACCEPTANCE on BLAZE_IMPL 8:
compile EXACT (raw-word shadow counted 16/16; 32 raw TTINCRWC -> 32
encoding-identical carried mode-6 loads; sum 126->97 / max 128->99
words/tile). SILICON: the carried walk is real (sum useq 1854 ->
1596 = causal -13.9%; max SFPSWAP-bound -0.5%) but the class is
honestly REFUSED-BY-MEASUREMENT on the blaze pair — the twin does
not beat the booked straight-push lift (anchors 1472/1495 and
1758/1775 reproduced EXACT), so A9/A10 stay lift-booked and the
successor is bounded honestly (post-autoincr window re-formation,
<=23/8 cy/tile recoverable on sum, must beat the envelope law).

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin39; pinned-install
  env): 6771 PASS; FAIL set 16 rows LINE-IDENTICAL to the pin-38
  frozen baseline (diff empty). Flags smoke-accepted (OPTCHECK).
- IF gates: dg focused 417/417; blaze compile screen 174/174; paired
  CRAQ 8/8 both arms on the pinned sims (bh 1d162f0adf67, incl the
  useq t32 trip hazard); device corr 6/6 + 4/4; blaze preservation
  8/8 byte-identical; corpus OFF/TD/ON-36 base-vs-fix 3300/3300 x3
  .text-identical; knob corpus delta = ZERO TUs (perf-vehicle fire
  only, the HO-F2 pattern).
- Silicon (BH p150, 3-rep cycle-identical, corr-first, 26 sessions
  rc=0): cells above; board notes-only 90a83c4e07d2 -> 71665b30bc39,
  tally 78W/35P/21L unchanged.
- Evidence: laneIF-evidence-20260827 (+SHA256SUMS 406). Incidents
  banked with recoveries: the zsh subscript TSV bug -> device wedge
  -> flush.sh recovery + health check; mid-leg farm commits discard
  corpus legs (second confirmation of the head-guard gotcha).
- Install: sha-verified e8226c223427... -> 287a307f4836...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36 (load-carrier registers as an on-plus
  booking knob).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin39-ceremony/).
