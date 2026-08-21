# SEM-ONLY AUDIT — does a hand kernel exist for every kind=semantic row? (lane ED, 2026-08-20)

Owner challenge: the dashboard books ~53 "SEM-ONLY = no hand kernel" rows — "surely they exist,
are we getting confused?"  This audit settles it op by op for **every kind=semantic row** in
`sweep_2x2_ops.tsv` (66 rows at base `origin/nkapre/sfpi` = `b532e99582`), against the
byte-untouched production kernel sources: the pristine tt-llk library
(`tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/`) and the metal LLK tree
(`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/`) that the test harness actually
dispatches (`tests/helpers/include/sfpu_operations.h` includes BOTH trees; the corpus manifest
`sfpu_corpus_v2.tsv` records the audited node->function mapping per id).

## Classes

- **(A)** the upstream production kernel exists and is ALREADY plain typed-SFPI C++
  identical-in-kind to a semantic body: hand == sem SOURCE, a vs-hand comparison would be a
  self-comparison, and causal-only (passes OFF vs ON) is the honest axis.  **55 / 66 rows.**
- **(B)** no upstream kernel exists at all (op added purely by corpus/fitter work): **0 / 66 rows.**
  Every semantic row was built FROM a production header (the corpus manifest maps each
  corpus_id to its `header_bh`), so this class is structurally empty — the dashboard's
  "sem-only" label never meant "we invented the op".
- **(C)** MISCLASSIFIED — a genuinely distinct handwritten/intrinsic-style kernel exists
  upstream that the row's sem cells never race as a hand arm: **11 / 66 rows.**  These are the
  confusions the owner suspected.  Two shapes exist:
  - **C-as-sem** (8 rows): the row's "semantic" cells themselves compile a raw-TTI/LUT hand
    kernel — the row note claimed "production typed-SFPI body" or "no distinct hand LLK", and
    that claim is FALSE (comp, gcd, lcm, ema, softmaxk, moegatetopk, sfpureduce,
    tanhderivative-lut).
  - **C-shadow** (3 rows): the row's own cells are honestly typed, but the same op carries a
    distinct hand LUT kernel upstream that NO node anywhere raced (tanh approx-mode SFPLUT
    body; the legacy 6-segment SFPLUTFP32 sigmoid, manifest class D-ABSENT; the metal
    6-segment SFPLUTFP32 `calculate_gelu_appx`).

## Verdict on the owner's question

The dashboard label is *mostly* right and *specifically* wrong: for 55/66 rows "no hand kernel"
means "the production kernel IS the semantic-style source" (hand==sem, nothing to race), and for
8 more the hand kernel is raced — but as the row's SEM body, mislabeled.  Three hand LUT kernels
(tanh-approx, legacy sigmoid-lut6, gelu-appx) were never raced by anything.  All 11 (C) rows are
now converted, refused-by-name, or note-fixed on this branch (`agent/semonly-audit`).

## Actions taken on this branch

| action | rows |
|---|---|
| row FLIPPED semantic -> full2x2 (S4 relu precedent; SEM-CELL BASELINE BREAK) | `comp` (NEZ vehicle; fresh `fresh_cpp/comp.h` bodies for all five remaining float comparisons) |
| NEW full2x2 conversion rows (hand LUT kernel = hand arm, fresh body = sem arm) | `tanhlut-fresh`, `sigmoidlut-fresh` (new impl-3 selector + `_init_sigmoid_` override), `geluappx-fresh` |
| NEW machine-readable named refusals (kind=skip; softmaxk-fresh precedent) | `ema-fresh`, `moegatetopk-fresh` (SKIP_NOT_FEASIBLE: cross-lane TRANSP/rotate mechanisms), `sfpureduce-fresh` (SKIP_BLOCKED: S4 subvec_shflror1 ruling) |
| base-row NOTE FIXES (false "typed body"/"no distinct hand LLK" claims corrected) | `comp` (via flip), `gcd`, `lcm`, `ema`, `softmaxk`, `moegatetopk`, `sfpureduce`, `tanhderivative-lut`, `tanh`, `sigmoid`, `gelu` |

Conversion discipline: byte-untouched production kernels (both trees untouched; the legacy LUT
sigmoid is reached by a test-side include + impl-3 selector only), identical stimuli/golden/
tolerance per row (tolerance derivations in the test docstrings), collect-verified nodes,
compile-only classify proven (below).  NO device runs this lane: the new rows are
CORRECTNESS-UNVERIFIED-ON-SILICON until the first weekly books or refuses them — each row note
says so.

## Full table (66 kind=semantic rows at the audit base)

| op | class | upstream production kernel (BH) | verdict / action |
|---|---|---|---|
| `activations` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_activations.h :: calculate_activation | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `cbrt` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h :: calculate_cube_root | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `clamp` | A | tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_clamp.h :: _calculate_clamp_ | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `comp` | C | metal llk_sfpu/ckernel_sfpu_comp.h calculate_comp (ALL float comparison nodes) | raw TTI hand kernel raced under the sem label; CONVERTED this lane: comp row flipped full2x2 (NEZ vehicle, fresh_cpp/comp.h sem arm); eqz already raced by eqz-fresh |
| `digamma` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_digamma.h :: calculate_digamma | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `divint32floor` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_div_int32_floor.h :: calculate_div_int32_floor, calculate_div_int32_trunc | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `elu` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_elu.h :: calculate_elu | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `ema` | C | legacy tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_ema.h _calculate_ema_tile_ | raw TTI hand kernel (SFPTRANSP-bracketed MADs, raw-LREG carry ABI); named refusal ADDED this lane (ema-fresh SKIP_NOT_FEASIBLE); base-row note FIXED |
| `erf` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_erf.h :: calculate_erf | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `erfc` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_erfc.h :: calculate_erfc | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `erfinv` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h :: calculate_erfinv | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `exp2` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp2.h :: calculate_exp2 | plain typed SFPI body; hand==sem source; causal-only axis honest (typed body with isolated __builtin_rvtt_sfpmad fused-MAD idiom — value-semantic, per the laneDM/S4 near-clean rulings) |
| `expm1` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_expm1.h :: calculate_expm1 | plain typed SFPI body; hand==sem source; causal-only axis honest (typed body with isolated __builtin_rvtt_sfpmad fused-MAD idiom — value-semantic, per the laneDM/S4 near-clean rulings) |
| `expm1cw` | A | tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_expm1_cw.h :: expm1_cw_clamped | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `fill` | A | tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_fill.h :: _calculate_fill_ | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `fmod` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_fmod.h :: calculate_fmod | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `gcd` | C | metal llk_sfpu/ckernel_sfpu_gcd.h calculate_sfpu_gcd | raw TTI + TTI_REPLAY binary-GCD hand kernel raced under the sem label; already converted (gcd-fresh); base-row note FIXED this lane |
| `gelu` | C | metal llk_sfpu/ckernel_sfpu_gelu.h calculate_gelu_appx (6-segment SFPLUTFP32 lut2) | distinct hand LUT kernel; only ever compiled as a causal-only node under the sem label; CONVERTED this lane: new geluappx-fresh full2x2 row; base row note added |
| `hardmish` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardmish.h :: hardmish | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `hardshrink` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardshrink.h :: calculate_hardshrink | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `hardtanh` | A | tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h :: _calculate_hardtanh_ | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `heaviside` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_heaviside.h :: calculate_heaviside | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `i0` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_i0.h :: calculate_i0 | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `i1` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_i1.h :: calculate_i1 | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `identity` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_identity.h :: calculate_identity | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `isclose` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_isclose.h :: calculate_sfpu_isclose | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `isinfisnan` | A | tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_isinf_isnan.h :: _calculate_sfpu_isinf_isnan_ | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `lcm` | C | metal llk_sfpu/ckernel_sfpu_lcm.h calculate_sfpu_lcm | raw TTI hand kernel (SFPMUL24 + GCD replay + raw Newton recip); already converted (lcm-fresh); base-row note FIXED this lane |
| `lerp` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_lerp.h :: (see manifest) | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `lgamma` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h :: calculate_lgamma_stirling | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `log` | A | tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_log.h :: _calculate_log_ | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `log1p` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_log1p.h :: calculate_log1p | plain typed SFPI body; hand==sem source; causal-only axis honest (typed body with isolated __builtin_rvtt_sfpmad fused-MAD idiom — value-semantic, per the laneDM/S4 near-clean rulings) |
| `logicalnot` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_logical_not.h :: calculate_logical_not | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `logsigmoid` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_logsigmoid.h :: calculate_logsigmoid | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `mask` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_mask.h :: calculate_mask | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `mish` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_mish.h :: calculate_mish | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `moegatetopk` | C | legacy .../sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk.h _generic_moe_gate_topk_ | raw TTI bitonic-sort hand kernel (116 TTI_SFP sites); named refusal ADDED this lane (moegatetopk-fresh SKIP_NOT_FEASIBLE); base-row note FIXED |
| `negative` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_negative.h :: _calculate_negative_ | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `polygamma` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_polygamma.h :: calculate_polygamma | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `prelu` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h :: calculate_prelu | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `remainder` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_remainder.h :: calculate_remainder | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `rsqrt` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rsqrt.h :: calculate_rsqrt | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `rsqrtcompat` | A | tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h :: (see manifest) | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `rsubint32` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rsub_int32.h :: calculate_rsub_int | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `sampling` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sampling.h :: calculate_sampling_binary_comp_first_column | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `sdpafw` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa_fw.h :: calculate_exponential_first_column | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `sfpureduce` | C | metal llk_sfpu/ckernel_sfpu_reduce.h calculate_reduce | raw TTI reduce family (298 TTI_SFP + 30 replay sites); BLOCKED refusal ADDED this lane (sfpureduce-fresh, cites the S4 subvec_shflror1 ruling); base-row note FIXED |
| `sigmoid` | C | legacy tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sigmoid.h _calculate_sigmoid_ (6-segment SFPLUTFP32 lut2) | distinct hand LUT kernel, manifest class D-ABSENT: never included/dispatched by ANY test; CONVERTED this lane: new sigmoidlut-fresh full2x2 row (impl-3 selector + init override); base row note added |
| `silu` | A | tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_silu.h :: _calculate_silu_ | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `softmaxk` | C | legacy .../sfpu/experimental/ckernel_sfpu_softmax_k.h _softmax_k_ | raw TTI hand kernel (65 TTI_SFP sites); refusal already existed (softmaxk-fresh); base-row note FIXED this lane |
| `softplus` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softplus.h :: calculate_softplus | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `softshrink` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h :: calculate_softshrink | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `softsign` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h :: calculate_softsign | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `sqrt` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h :: calculate_sqrt | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `sqrtcustom` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt_custom.h :: sfpu_sqrt_custom | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `square` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_square.h :: calculate_square | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `tanh` | C | metal llk_sfpu/ckernel_sfpu_tanh.h calculate_tanh<APPROX=true> (3-region SFPLUT; byte-identical pristine twin legacy _calculate_tanh_) | distinct hand LUT kernel NEVER raced by any node (zero approx:Yes nodes; generic sweep skips Tanh approx); CONVERTED this lane: new tanhlut-fresh full2x2 row; base row cells stay typed-poly (honest) + audit note |
| `tanhderivative` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh_derivative.h :: calculate_tanh_derivative_sech2 | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `tanhderivative-lut` | C | legacy .../sfpu/ckernel_sfpu_tanh_derivative.h _calculate_tanh_derivative_ | LUT l_reg-choreography hand kernel raced under the sem label; already converted (tanhderivlut-fresh); base-row note FIXED this lane |
| `tanhshrink` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_tanhshrink.h :: calculate_tanhshrink | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `threshold` | A | tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_threshold.h :: _calculate_threshold_ | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `trigonometry` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h :: calculate_acos, calculate_acosh, calculate_asin, calculate_asinh, calculate_atan, calculate_atanh, calculate_cosh, calculate_cosine, calculate_sine, calculate_s | plain typed SFPI body; hand==sem source; causal-only axis honest (typed body with isolated __builtin_rvtt_sfpmad fused-MAD idiom — value-semantic, per the laneDM/S4 near-clean rulings) |
| `unarycomp` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_comp.h :: calculate_unary_eq, calculate_unary_ge, calculate_unary_gt, calculate_unary_le, calculate_unary_lt, calculate_unary_ne | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `unarypower` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_power.h :: calculate_unary_power | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `unaryshift` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_shift.h :: calculate_left_shift, calculate_right_shift | plain typed SFPI body; hand==sem source; causal-only axis honest |
| `xielu` | A | tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_xielu.h :: calculate_xielu | plain typed SFPI body; hand==sem source; causal-only axis honest |

## Fitted-row reference verification (mission item 3)

Claim checked: "each X-fitted row's hand column is our compiled semantic X."  From the ops.tsv
row config, ALL 21 `-fitted` rows take their hand cells from `test_fitted_cpp[X-production]` /
`test_perf_fitted_cpp[...fresh_cpp_impl:0]` — i.e. **the production dispatch at approx_mode:No**,
not a separate "compiled semantic" artifact.

- For **20/21 rows** the claim holds **by identity**: the impl-0 production body for that mathop
  is the typed hand==sem source (class A above), so "production" and "compiled semantic X" are
  the same compiled body.  (tanh-fitted's reference is the TYPED polynomial branch — not the
  approx LUT; gelu-fitted's is typed `calculate_gelu` — not `calculate_gelu_appx`.)
- **exp-fitted is the one discrepancy**: its impl-0 reference is production
  `calculate_exponential` — the DISTINCT handwritten exp that the `exp` full2x2 row races as its
  HAND arm.  The fitted row's reference is therefore the hand kernel, NOT "our compiled semantic
  exp" (the semantic exp body is `calculate_exp_fresh_cpp`, impl 1, raced only by the exp row).
  Any dashboard text claiming the fitted references are all "compiled semantic X" must carve out
  exp-fitted.

## Findings out of row scope (recorded, no action)

- `legacy__ckernel_sfpu_sqrt` / `legacy__ckernel_sfpu_exp` raw kernels exist but are
  **QSR-only** (no BH/WH header) — out of this BH sweep's scope.
- Distinct-op raw legacy kernels with no corpus row at all (not misclassifications of existing
  rows; potential future rows): `swiglu`, `rope`, `dropout`, `rand`, `quant`, `cumsum`,
  `max_pool_indices`, `reshuffle_rows`, `cdf` (manifest class B-TRANSITIVE).
- The metal tree carries typed twins of several raced legacy kernels (`metal ckernel_sfpu_log`
  with one fused-MAD builtin, `metal silu/hardtanh/clamp/relu`) — typed, so class-A-equivalent
  shadows, not hand kernels; recorded in the manifest as unmapped ids.

Evidence: `~/sfpi-uplift/laneED-evidence-20260821/` (SHA256SUMS; classify logs, selftest logs,
lint log).  Branch: tt-metal `agent/semonly-audit` off `origin/nkapre/sfpi` @ `b532e99582`
(the mission brief's `59608255c8` tip is a local-only `conf-ceremony` commit, not on origin).
