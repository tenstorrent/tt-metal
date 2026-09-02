# REVIEW_RECORD — pin 54 (knob promotion round 5 + wordfact deletion)

cc1plus sha256: eaccb93b0e98f26eaf027169634b0d88ac03d1c74904ff02324133f2ec9473c6
driver (g++) sha256: dded27bb7d1d726cfc9f1d07a92c00c3aee2493797caf341d10bd9ba28eb1626 (UNCHANGED — the deletion adds no options)
source: sfpi-gcc nkapre/sfpi 6191f71fa91 = pin-53 ae333a1e145 + the
laneKR wordfact one-pin deletion (agent-executed, blame-verified,
-539 lines, zero references remain, 0 dg checks removed).  tt-metal
canon: + agent/laneKV-promotion 276703a9d944 (ff) — ON SET 37 -> 39.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

KNOB PROMOTION ROUND 5 (laneKV, conf-only at the pin-53 binary; full
record laneKV-evidence-20260901/RESULTS.md): ims + mve-expand PROMOTED
on measured obligations (9 moved rows adjudicated; 15 moved-row device
A/Bs, zero WIN/PARITY regressions; erfinv-fresh -41.08 -> -42.19 and
silu re-books; CRAQ verdict-identical at the new sims; two-sided
witnesses; preflight ALL GREEN at ON-39).  FOUR candidates REFUSED BY
MEASUREMENT — the promotion gates caught two silicon wrong-code
defects and a P0 ICE in the wave-4/5 rename/CC stack (chains,
temporal, cc-region-general; rename-cc-region by dependency).  Fix
lanes KZ/LA/LB closed with root causes named (DF-liveness trust in
opaque functions; dest-reuse-writer temporal admission; peel-entry
anchoring — the last also SILENT wrong code on 2 ATAN2 corpus rows,
device-proven); their branches merge at pin 55.  The trig licensed
WIN stands BY BYTES at this pin's ON-39; laneLA's fix moves those
bytes — re-measure owed at pin 55, nothing booked.

WORDFACT DELETION: laneKR's item-#4 one-pin equality phase served; the
audited word-fact table + accessors are the sole spelling; no one-pin
scaffolding remains anywhere in tt/ (grep empty).

## Gates checked
- Union rvtt.exp (dejagnu-pin54): 7708 PASS = pin-53's 7708 + 0; FAIL
  set 16 LINE-IDENTICAL; dg ERROR 0.
- Corpus (sweep-2x2/pin54-ceremony/corpus/): pinned-53 vs union ON-39
  3300/3300 BYTE-IDENTICAL; chkon rc=0, ZERO ICEs, .text == ON.
- conf_lint GREEN; witness_preflight ALL GREEN at ON-39 (result below).
- Board 85W/35P/14L (KV re-books in place, post 938bb7d144c2afbb).
- ON 37 -> 39; KNOB_MODES 54 -> 56.
- Install: sha-verified 46d3116c469d -> eaccb93b0e98; CEREMONY LESSON:
  the installer refused under the fix lanes' live compiles (correct);
  the first conf edit raced it and was reverted — conf edits only
  after a verified install.
- PIN-55 QUEUE: KZ/LA/LB merges (dg expect 7722), trig licensed-cell
  re-measure, addint/sem-corr-wh archaeology standing.
