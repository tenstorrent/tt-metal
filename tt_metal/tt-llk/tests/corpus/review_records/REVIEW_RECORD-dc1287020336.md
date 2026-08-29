# REVIEW_RECORD — pin 47 (the F1 recovery pair)

cc1plus sha256: dc12870203364a44b04bb3dce082086b5c5c3c1f15490ef293a2006cdde80fb6
driver (g++) sha256: 38af15bd3390404351149ab60dd971dd46ec91bbe8943556507af55e9ac956ca
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 7190c2460a3 = pin-46 union tip d8c4c71264a
+ two lane merges (IU agent/minmax-run-pricing 28a57951438, IV
agent/typecast-walk-transparency 4ebac00bc9e). Both fixes are
compiler-only ON-behavior (no tt-metal knob changes; tt-metal canon
stays 0259d0d203). KNOB_MODES dup grep: NONE. No sfpi include/
changes. Built in gcc-build-laneFR (build-pin47.log rc=0); OPTCHECK
smoke; installed via pin-install-fast with loud --expect-cc1plus; no
live sweeps at install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

THE F1 RECOVERY PAIR — both honest-fix casualties recovered through
real compiler machinery, closing the adversarial-audit campaign
(charter -> IP/IQ/IR audits -> IS honest fix + IT same-leg re-books
-> IU/IV recoveries):

IU INIT-HOIST-AWARE RUN PRICING. The minmax autopsy priced the
death to the word: the frozen conservative-per-run formation gate
charged config_prefix_cost (17 issue words: 1 enable + 3 SETC16 +
13 descriptor materializations) to every run — 44 > 32 ->
unprofitable x4 — when the stage-2 crosscall init-hoist provably
removes that prefix from the callee (a ~37% phantom charge
bracketing the measured 30% gap; the marker era had merely padded
the explicit side past the broken arithmetic). Fix:
rvtt_crosscall_init_hoist grew a commit flag — the identical proof
chain runs PROOF-ONLY ahead of the profitability gate, filling the
stage and the proven caller-loop profile trip weight; under a proven
stage-2 contract run_profitable_p prices config*E +
(rows*ii+drain)*B < rows*explicit*B (the rvtt-cost.md
INIT-HOIST-AWARE RUN PRICING derivation; refusal-biased —
stage-1/no-weight/off keep the frozen pricing); the commit remains
LAST among refusal points behind a fail-closed
init-hoist-commit-diverged guard; IMS arbitration amortizes its
formed-side prefix identically; composes with lane-IA's
per-execution pricing (disjoint words). minmax-max +36.03 -> WIN
-5.04 (17451.0 x3 / 18377.3 — the sem cell EQUALS lane IS's macro
measurement exactly); minmax-min +36.04 -> WIN -5.03. 5 twins
(30/30) incl the mandated multisite near-miss where the
unprofitable refusal returns; planner+init-hoist suite 1225/1225.

IV AUDITED-REGION WALK TRANSPARENCY. The typecast dirt anatomized:
the inlined LLK envelope init is ~45 raw .ttinsn words
(SETC16/sync/SETRWC/SFPLOADI/SFPCONFIG incl the LaneConfig
default-reset and two load-macro template captures) plus
template-programming store asm, one MOP run word, and empty-asm
barriers — ZERO calls and NO SFPENCC anywhere, so the derived
verdict is the no-CC-writer arm: the fn-entry all-lanes ambient
fact carries THROUGH the init rather than being killed or dirtied
(the two capture words are CC-inert under both backdoor readings —
SFPCONFIG.md DISABLE_BACKDOOR_LOAD + the pinned sim's capture arm;
every SFPCAST/SFPSTOCHRND mode writes only LReg[VD]). Fix:
rvtt_raw_cc_word_class (an audited CC/lane-enable opcode-class
table whose proven classes are AMBIENT-PRESERVING ONLY, never kills
— a raw word can sit swallowed inside a REPLAY record window, the
lane-HS soundness argument) + a TU-wide CC audit riding the
existing prgm-const scan (fail-closed when uncomputed) + the
entry-ambient walk's asm arm (decode / barrier / TU-lean; calls and
everything undecodable stay DIRTY under the unchanged named
refusal, with dirty-insn+word diagnostics). typecast +13.27 -> WIN
-5.10, CYCLE-EXACT the pre-F1 booked cell (sem 558.0 x3; hand
byte-identical; OFF untouched). 17 twins incl the mandated
real-CC-writer near-miss staying dirty.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin47; pinned-install
  env): 7206 PASS; FAIL set 16 rows LINE-IDENTICAL to the pin-46
  frozen baseline; dg ERROR count == 0. Flags smoke-accepted
  (OPTCHECK).
- Per-lane: IU corpus OFF/TD 3300/3300 identical + ON-36 delta =
  exactly the 2 minmax sem corr TUs (paired CRAQ + device corr
  PASS); IV corpus OFF/TD identical + ON-36 delta = exactly 1 TU
  (a NEW formation on the negative production corr TU, adjudicated
  + CRAQ-paired PASS, its second region still refusing fail-closed
  in the same TU); 84-node screens = exactly the target TUs; R9
  witness nodes (unarymaxmin, mulint32) silicon-exact in BOTH
  lanes; the IS survival set byte-preserved.
- Silicon (BH p150, 3-rep cycle-identical, corr-first, anchors
  first): minmax pair WIN -5.04/-5.03; typecast WIN -5.10
  cycle-exact the pre-F1 cell.
- Board: 876d448f80c3 -> 3afc304149f3 (IU) -> ec2119b5942f (IV);
  TALLY 81W/35P/18L -> 84W/35P/15L — both F1 casualties recovered,
  absint32's P->W kept: ONE WIN BETTER than pre-audit, every win
  clean of the marker signal, every loss certified or floor-named.
- Evidence: laneIU-evidence-20260829 (+SHA256SUMS 6881),
  laneIV-evidence-20260829 (+SHA256SUMS 5666).
- Install: sha-verified 7bb90d0c88ec... -> dc1287020336...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36.

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin47-ceremony/).
