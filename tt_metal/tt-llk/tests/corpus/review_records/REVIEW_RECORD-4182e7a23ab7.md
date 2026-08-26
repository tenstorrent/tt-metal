# REVIEW_RECORD — pin 31

cc1plus sha256: 4182e7a23ab770c4864f5b302f452a96f6a936bb0669125d17596ece7b399786
driver (g++) sha256: 1aef36035d0093c0a5e02ed15505b1c43cad95b0dc4a3e9ad9a4a176d2c843f4
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 657413d694b = pin-30 union tip ca8e9e19386
+ three lane merges (HM agent/rlu-tanh-wrongcode a0d991a3f4e, HL
agent/store-sink-license fd6aa3662de, HJ agent/el-composition-audit
203618b66c7). Companions: tt-metal chain through ba4dcda5ae (HK
certified-floor notes f4bc319c43 + owner ratification 957d33babb + HI
knob rows c1238453bf + HL licensed-knob registration 5db46f88bb +
HJ madpair-vocab knob 6bfbedac8f; one conf both-append conflict
union-resolved, py-parse + conf-lint verified). No sfpi include/
changes since pin 29. Built in gcc-build-laneFR (build-pin31.log
rc=0); new flags smoke-accepted together (OPTCHECK, mcpu=tt-bh:
madpair-vocabulary + store-sink + prior-wave knobs); installed via
pin-install-fast with loud --expect-cc1plus; no live sweeps at
install (the single pgrep hit was the checking shell itself).

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

Three lanes, each with its own closing report, evidence dir
(+SHA256SUMS) and memory file:

HM (P1 laneHI-F1 adjudicated verdict (A) COMPILER WRONG-CODE, the EV
class: counted-row canonicalization moved excluded loadis and VACATED
the nop inserter's discharged SFPMUL->SFPSWAP delay shadow — the
SFPMAD.md CAUTION erratum, BH scoreboard blind to SFPSWAP 1st-cycle
reads — so recorded windows replayed the pair unpadded; numeric
closure: all 22 device escape values equal the unclamped polynomial
at bf16 inputs in [3.375, 5.0]; silicon protocol ruled out poisoning
(post-reset rep) and control leakage; fix = crf_shadow_contract_ok
re-verifies the whole delay contract over the plan's final order with
the nop inserter's own dependence-and-erratum test, undischarged
pairs refuse by name counted-row-vacated-delay-shadow; the fixed tanh
window is DENSER than the broken form; four dg twins; board-wide
811-ELF erratum-adjacency audit: the tanh rlu pair was the ONLY
instance, nothing was booked on defective bytes; rlu-on-tanh is
measured-no-help once sound so tanh's +24.14 stands).

HL (owner-ratified store-sink license, EJ discipline in full:
-mtt-tensix-optimize-store-sink Init(0) admits the store-fold S2 sink
at the store-fold-sink-format-canonicalizing refusal, shape-general,
float pairs only, WH INT32_SM refuses even licensed, mixed pairs stay
format-unproven; accuracy gate VERIFIED not inherited — sunk == hand
== 0 error vs raw torch on all 254 BF16 denormal flush witnesses, and
at these rows' constants the licensed output is bit-identical to
baseline for every representable input; second composition fix
store-source-encoding-ceiling drops store-consumed constants to the
pressure-park LREG tier, erasing the RA SFPMOV copy tax
(SFPSTORE sources L0-L11 only); threshold 6->5 words = hand,
hardshrink 7->6 = hand; 7 new dg twins 24/24; HL-F1 filed for the
unlicensed copy-tax generalization; new adjudication surface named:
harness golden applies _apply_ftz vs raw torch keeping subnormals).

HJ (systematic EL-composition audit: 28 losses + 22 positive-residual
parity rows dump-compiled at pin-30, 104 byte-verified legs +
instrumented exact-obligation cc1plus + on34-vs-noel A/B; census:
HF's counting-phantom class has exactly ONE remaining instance —
the geluappx licensed body's dead-pre-entry config pair, LATENT-MASKED
by lut-select's exempt re-placement at zero delivery cost, named
deliberately not fixed — and 88 other EL pressure refusals proved
GENUINE; GA's silent recognition-miss class had two live instances
fixed by -mtt-tensix-optimize-madpair-vocabulary Init(0), widening
MAD-PAIR discovery to the combine's own vocabulary (sfpadd_lv,
single-use SFPMOV-COMPLEMENT wrapper) with admission/refusals/pricing
unchanged and flag-off byte-identical delegation; 4 dg twins;
softplus-fresh EL-vs-pressure-park ORDERING composition named for a
successor lane).

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin31; env =
  ~/sfpi-uplift/sfpi/build/sfpi — the pinned install; the laneFP env
  dir is stale post-pin-30, lane-HJ finding): 6404 PASS, FAIL set 16
  rows LINE-IDENTICAL to the pin-30 frozen baseline (diff empty).
  New flags smoke-accepted together on the union (OPTCHECK).
- Per-lane corpus byte-gates vs the pin-30 stores: OFF/TD/ON-34 all
  3252/3252 byte-identical in every lane; HM's ON34+rlu knob legs
  base-vs-fix 0 delta TUs; HL's 242-row knob census fully adjudicated
  (228 = laneEK S1 word-neutral class w/ zero licensed fires, 8
  licensed sinks dump-witnessed, 6 = the two target TUs); HJ's knob
  corpus-inert 0/3300 (single-trip tile loops, named trip-count
  refusal, HB precedent).
- CRAQ on the pinned sim green in every lane (HM counted-row suite
  42/42 + fixed-tanh CRAQ PASS; HL paired 8/8 + moe_gate 89/89 +
  binary 414 both legs; HJ paired 4/4). Device corr corr-first
  everywhere (HM 3/3 fixed tanh; HL 24/24; HJ 2/2).
- Silicon (BH p150, 3-rep, cycle-identical reps): threshold-fresh
  +0.03 P / hardshrink-fresh -0.35 P same-leg (HL); smoothstep-fresh
  WIN -5.28, KERNEL 78887 -> 70955 (HJ); tanh rlu sound form
  measured-no-help, +24.14 cells stand (HM).
- Evidence: laneHM-evidence-20260826, laneHL-evidence-20260826,
  laneHJ-evidence-20260826 (SHA256SUMS in each).
- Install: sha-verified 106e98daddc7... -> 4182e7a23ab7...; driver
  read from the fresh manifest entry; no sfpi include/ staging owed.
- ON set UNCHANGED at 34 (madpair-vocabulary and store-sink register
  as on-plus knobs; store-sink is the second LICENSED_KNOBS entry).
  BOARD at cut: 75W/35P/25L.

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-34 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin31-ceremony/).
