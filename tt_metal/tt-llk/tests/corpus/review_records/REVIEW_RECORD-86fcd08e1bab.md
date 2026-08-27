# REVIEW_RECORD — pin 36

cc1plus sha256: 86fcd08e1babc0878bafb9f6eb33d4d8abac9b0a41d9c3fc744d285e148b136c
driver (g++) sha256: 312466219738b108b81638aa92838c1c473876179f5269e880101cf457d592d2
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi ff2199b67d5 = pin-35 union tip 0befdec8214
+ two lane merges (HZ agent/stochrnd-store-fold bc1b5339ca3, IA
agent/binopscalar-autoincr-pricing d4a1b6cf38f). Companions: tt-metal
chain through 6d6db62e998 (HZ licensed-knob registration d2a9069224 +
IA registration; KNOB_MODES dup grep clean — only the known benign
lut-select-fp16 pair). No sfpi include/ changes. Built in
gcc-build-laneFR (build-pin36.log rc=0); flags smoke-accepted
(OPTCHECK, mcpu=tt-bh); installed via pin-install-fast with loud
--expect-cc1plus; no live sweeps at install. CRAQ in both lanes ran
on the NEW pinned sims (lane HU re-pin: bh 1d162f0adf67, wh
f22bc917a4ef).

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

Two lanes, each with its own closing report, evidence dir
(+SHA256SUMS) and memory file:

HZ (stochrnd-into-store fold, LICENSED value-changing token
-mtt-tensix-optimize-stochrnd-store-fold Init(0)): ISA adjudication
first — SFPSTORE's BF16/FP16 conversion truncates toward zero with
no in-store rounding mode (BH+WH SFPSTORE.md; SFPSTOCHRND
RND_NEAREST = ties-away + specials normalized), so the fold is NOT
bit-exact; the standing laneEK 2^32 sweep
tt/proofs/stochrnd-store-round is the divergence certificate (BF16
2,155,741,184/2^32 differing, FP16 268,435,456/2^32) and the
bit-exact cut keeps its named refusal byte-identically. License
authority (EJ discipline in full): the folded stream is
instruction-for-instruction the hand kernel's own store path, so
matching hand's exact bits matches the golden per laneCX — confirmed
corr-first on device (3x) and the new pinned sim. Design call: the
token gates the store-fold pass by itself (the composed S1 leg
reshaped the HAND arm 25766->19498 — a named against-interest
successor finding). binary-float +6.45 -> WIN -17.37 (sem
27428->21291, hand byte- and cycle-identical; 4 slots/row = hand
window parity; causal -36.34). 11 dg twins incl the required
stochastic-mode near-miss refusal; license-only knob census 64
changed variants all adjudicated licensed fires; delta CRAQ 36/36
runnable ids PASS both legs on bh 1d162f0adf67.

IA (dst-autoincr admission pricing, same flag, no new flag): the
semantic straight-line 8-row callee re-emitted its 3-SETC16 ADDR_MOD
slot program on every invocation (512 calls/KERNEL) because the old
admission priced 3 config WORDS against 8 removed rows; the true
per-execution cost (3 x audited rvtt_issue_cfg 2-slot config class +
min_config_distance drain residual on the jal-drained entry) = 8 >=
8 -> refuse; silicon bracketed the fire at ~+1.5cy/call. Fix =
per-execution config pricing + payload-family joint pricing (groups
sharing a rewritten capture payload price all-or-nothing — an 8-row
orphan otherwise poisoned the rdiv/sqrt/cbrt hand 32-launch streams)
+ placement split (preheader programs keep laneEP's measured word
pricing; non-preheader pay 3x2+2 per execution) — the uniform
intermediate pricing regressed lcm 692423->694979 and relu-hand
45744->49330, counterfactual silicon archived, the split restores
both byte-identically. rvtt-cost.md carries the new audited section;
boundary twin pair (8-row refuse / 9-row fire). binopscalar causal
+3.61 -> 0.00 EXACT (fix sem 21164.0 == the sem_off anchor 3-rep);
row +5.61 -> +1.93; residual named (crosscall once-per-kernel
ADDR_MOD programming successor). ON-delta = exactly the flip TU + 5
HE-varmap-named TUs, all paired-CRAQ PASS both arms; 72-leg
loss+WIN screen: all 25 WIN rows byte-identical (the HY lesson
applied); the touched log-fresh hand-arm TU silicon 3-rep
cycle-identical with WIN cells reproduced EXACT. dg focused 395/395
+ sfpi 9/9.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin36; pinned-install
  env): 6695 PASS; FAIL set 16 rows LINE-IDENTICAL to the pin-35
  frozen baseline (diff empty). Flags smoke-accepted (OPTCHECK).
- Per-lane corpus byte-gates vs pin-35: HZ OFF/TD/ON-36 all
  3300/3300 identical (license-only census 64 variants adjudicated);
  IA OFF/TD 3252/3252 identical + ON-36 delta = exactly 5
  HE-varmap-named TUs + the flip TU, CRAQ-adjudicated.
- CRAQ green in both lanes on the NEW pinned sims; device corr
  corr-first everywhere.
- Silicon (BH p150, 3-rep cycle-identical): binary-float WIN -17.37
  (HZ; control leg reproduced HX's booked cells exactly);
  binopscalar +1.93 with causal 0.00 exact (IA; lcm/relu/log-fresh
  anchors reproduced exactly).
- Evidence: laneHZ-evidence-20260827, laneIA-evidence-20260827
  (SHA256SUMS in each). Board tally 78W/35P/22L.
- Install: sha-verified 133619cf6b77... -> 86fcd08e1bab...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36 (stochrnd-store-fold = third LICENSED_KNOBS
  entry; the IA fix is same-flag pricing inside the reviewed
  dst-autoincr).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin36-ceremony/).
