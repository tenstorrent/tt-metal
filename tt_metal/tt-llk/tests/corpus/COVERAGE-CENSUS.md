# SFPU COVERAGE CENSUS — every kernel header vs the raced corpus (lane EU, 2026-08-21)

Owner order: expand the raced-LLK pool to the full SFPU surface.  This census enumerates
EVERY SFPU kernel header on the BH+WH surfaces and diffs it against the corpus's covered
`corpus_id` set (`sweep_2x2_ops.tsv` col 2), then classifies and disposes of every
uncovered header.  Base: `origin/nkapre/sfpi` @ `f3f86df857` (ops v3, pin 18).
Branch: `agent/llk-coverage-expansion`.  Compile-only lane: NO device runs — every
conversion row is CORRECTNESS-UNVERIFIED-ON-SILICON until the first weekly books or
refuses it (lane ET's e2e weekly owns the device).

## Method

Enumerated trees (id convention = the manifest's `legacy__`/`metal__` prefix + header stem;
one id covers both arches when the stem exists on both):

| tree | headers |
|---|---|
| `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/` (incl. `experimental/`) | 41 |
| `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/` | 32 |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/` | 109 |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/` | 109 |
| `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/experimental/llk_sfpu/` | 13 + 5 |

**Counts: 163 distinct ids enumerated; 114 covered by ops.tsv rows at base (all 114
tsv-covered ids exist on disk — zero stale references); 49 uncovered.**

Uncovered disposition (this lane):

| class | count | disposition |
|---|---|---|
| REAL-OP | **11** | CONVERTED — full2x2 rows (S4/ED pattern; fresh body + byte-untouched hand arm) |
| HELPER | **18** | census-only (included by ops, not raceable alone) |
| TWIN / SHADOWED | **11** | census-only (the OP is already raced through the other-surface header; racing the twin re-races the same op) |
| INFEASIBLE | **9** | machine-readable named refusals — kind=skip rows (SKIP_NOT_FEASIBLE / SKIP_BLOCKED) |

Cross-check: 11 + 18 + 11 + 9 = 49.  After this lane every enumerated id is either
raced, named-refused in ops.tsv, or classified here as helper/twin.

## Blaze check (owner question, for the record)

**No blaze-named LLK kernels exist in tt-llk.**  `grep -ri blaze` over all of
`tt_metal/tt-llk/` (*.h, *.cpp, *.py, *.md) hits exactly two COMMENTS, both in
experimental `llk_lib` (not SFPU kernels): `llk_math_hadamard.h:229` and
`llk_math_deepseek_moe_gate_eltwise_binary.h:77` — each explains that a no-op/unused
parameter is kept so the signature "keeps matching blaze's" (i.e. these are kernels
ported FROM a blaze-original codebase; nothing in-tree is a blaze kernel or carries a
blaze name).  Zero blaze-named files, functions, or ops.

## REAL-OP conversions (11 rows, kind=full2x2, schedule=weekly)

Vehicle: new test-side pair `sources/sfpu_coverage_test.cpp` / `sources/sfpu_coverage_perf.cpp`
(+ `test_sfpu_coverage.py` / `perf_sfpu_coverage.py`), dispatch by `COVERAGE_OP` /
`COVERAGE_SUBOP` / `FRESH_CPP_IMPL` defines — zero edits to any LLK tree (R7); binarypow
rides the existing `test_sfpu_binary.py` / `perf_eltwise_binary_sfpu.py` vehicle via new
impl-1/impl-3 selectors.  Fresh bodies in `tests/helpers/include/fresh_cpp/<op>.h`.

| row | corpus_id | production kernel (hand arm) | notes |
|---|---|---|---|
| `rotate90-fresh` | metal__ckernel_sfpu_alt_complex_rotate90 | calculate_alt_complex_rotate90 | interleaved complex i*z; exact gate |
| `unarybitwise-fresh` | metal__ckernel_sfpu_bitwise | calculate_sfpu_unary_bitwise | XOR row vehicle; corr covers AND/OR/XOR; exact int32 |
| `addrsqrt-fresh` | metal__ckernel_sfpu_add_rsqrt | calculate_add_rsqrt (experimental) | rsqrt(x+eps), eps=0.5 fixed |
| `smoothstep-fresh` | metal__ckernel_sfpu_smoothstep | smoothstep_tile_face (experimental) | edges (-0.5, 0.5) fixed |
| `tiledprod-fresh` | metal__ckernel_sfpu_tiled_prod | calculate_tiled_prod | 9-row running product (production's documented ITERATIONS+1 walk) |
| `zeropad-fresh` | legacy__ckernel_sfpu_zero_pad | _zero_pad_tile_ (experimental) | rows [24,32) scrubbed; exact gate |
| `sparsekfilter-fresh` | legacy__ckernel_sfpu_sparse_k_filter | _sparse_k_filter_tile_ (experimental) | fixed bank-field geometry; exact int32 |
| `customadd-fresh` | metal__ckernel_sfpu_custom_add | my_add_tile_face (experimental) | two-Dst-tile add (buffer_B -> tile 1) |
| `copydest-fresh` | metal__ckernel_sfpu_copy_dest_values | copy_dest_value (raw TT_SFPLOAD/STORE) | tile 0 -> tile 1 move; packs tile 1; exact gate |
| `intsum-fresh` | metal__ckernel_sfpu_int_sum | calculate_sum_int_col/_row | COL row vehicle; corr covers COL+ROW; exact int32 |
| `binarypow-fresh` | metal__ckernel_sfpu_binary_pow | calculate_sfpu_binary_pow | the header's own entry had ZERO nodes (tested POW routes through calculate_sfpu_binary); impl 3 = hand, impl 1 = fresh |

Gates run this lane (installed pin-18 toolchain, cc1plus `664bbf81b2ca…`): all 50 new
nodes compile-classify PASS (26 corr + 20 coverage perf + 4 binarypow); sweep batched
classify over the 20 new rows = 88 compile legs, 0 COMPILE_FAIL, REPORT GREEN (dry-run,
silicon correctly refused without --allow-hardware); per-arm .text hashes distinct for
every hand/fresh pair (corr and perf); conf-lint GREEN (R7 LLK trees pristine); 13 corpus
selftests + 18 test_sfpu_corpus unittests GREEN.  Evidence:
`~/sfpi-uplift/laneEU-evidence-20260821/` (SHA256SUMS).

## INFEASIBLE — named refusals (9 rows, kind=skip)

| row | corpus_id | refusal |
|---|---|---|
| `dropout-fresh` | metal__ckernel_sfpu_dropout | SKIP_NOT_FEASIBLE: hardware XNOR-LFSR PRNG stream (SFPMOV special-source 9) — no torch-defineable pointwise golden (statistical contract) |
| `rand-fresh` | metal__ckernel_sfpu_rand | SKIP_NOT_FEASIBLE: hardware PRNG + bijective finalizer — no pointwise golden |
| `cumsum-fresh` | metal__ckernel_sfpu_cumsum | SKIP_NOT_FEASIBLE: SFPTRANSP-bracketed replay cross-lane mechanism (ema/softmaxk precedent) |
| `reshufflerows-fresh` | metal__ckernel_sfpu_reshuffle_rows | SKIP_NOT_FEASIBLE: L1 uint8 mask side-input + cross-row scatter-add |
| `rope-fresh` | legacy__ckernel_sfpu_rope | SKIP_NOT_FEASIBLE: three L1 side-inputs + ADDR_MOD reprogramming choreography; no vehicle family |
| `maxpoolindices-fresh` | metal__ckernel_sfpu_max_pool_indices | SKIP_BLOCKED: TopK joint value+index pairing hazard (TOPK_TYPED_CONVERSION_BLOCKER.md) |
| `quant-fresh` | metal__ckernel_sfpu_quant | SKIP_BLOCKED: no int8-capable BH SFPU vehicle (harness format-matrix gap); header is the standing _REVIEWED_LLK_API_EXCEPTIONS entry |
| `logitsoftcap-fresh` | metal__ckernel_sfpu_logit_softcap | SKIP_NOT_FEASIBLE: TRISC_PACK-gated body (pack-thread SFPU); math-thread harness cannot reach it |
| `clampedsilu-fresh` | metal__ckernel_sfpu_clamped_silu | SKIP_NOT_FEASIBLE: TRISC_PACK-gated family (5 entry fns), same class |

## HELPER (18 ids — included by ops, not raceable alone)

| id | role |
|---|---|
| legacy__ckernel_sfpu_converter | bit-cast helper (Converter::as_float); consumed by measured fresh bodies + deepseek test |
| legacy__ckernel_sfpu_is_fp16_zero | predicate helper (metal sign/mask/comp consumers) |
| legacy__ckernel_sfpu_load_config | SFPU config-load helpers (_init_sfpu_config_reg_ used by reduce-sdpa test, topk_xl) |
| legacy__ckernel_sfpu_polyval | POLYVAL evaluator (silu + ~14 metal consumers) |
| legacy__ckernel_sfpu_cdf | value-fn library (_calculate_cdf_appx_/_pos_cdf_appx_, vFloat->vFloat, no tile-kernel entry; included by metal gelu.h; the cdf fns themselves are compiled dead code — zero callers repo-wide) |
| metal__ckernel_sfpu_conversions | float32_to_bf16_rne/_float_to_int32_positive_ helpers (binary/binop_with_unary/unary_power consumers) |
| metal__ckernel_sfpu_piecewise_rational | rational evaluator (erf/erfc/digamma/gelu consumers) |
| metal__llk_math_eltwise_{unary,binary,ternary}_sfpu_{init,macros} (6 ids) | dispatch/macro glue — the call layer itself |
| metal__llk_math_eltwise_unary_sfpu_topk_xl | API glue for the covered topk_xl op (skip row exists) |
| metal__llk_math_ema_sfpu_entry | API glue for the covered ema op |
| metal__llk_math_generic_moe_gate_topk_api | API glue for the covered moegatetopk op |
| metal__llk_math_softmax_k_api | API glue for the covered softmaxk op |
| metal__llk_math_welfords_sfpu_entry | API glue for the covered welford op |

## TWIN / SHADOWED (11 ids — the op is already raced via the other-surface header)

Manifest class D-SHADOWED (or the executed-through-consumer B-TRANSITIVE twins); racing
these headers would re-race an op the corpus already races, so no row is added — the
census is their machine-readable record:

| id | covered by |
|---|---|
| legacy__ckernel_sfpu_cast_fp32_to_fp16a | castfp32tofp16a row (metal header is the dispatched one) |
| legacy__ckernel_sfpu_comp | comp row (metal calculate_comp is the dispatched float path; the legacy header is compiled via a direct include but dispatched by nothing) |
| legacy__ckernel_sfpu_tanh | tanh/tanhlut-fresh rows (byte-identical pristine twin of the metal LUT tanh; included by NOTHING repo-wide) |
| metal__ckernel_sfpu_clamp | clamp row (legacy _calculate_clamp_ is the dispatched body; metal clamp is Quasar-harness-only) |
| metal__ckernel_sfpu_hardtanh | hardtanh row (legacy body dispatched) |
| metal__ckernel_sfpu_relu | relu row (legacy body dispatched; metal header includes the legacy one, dispatched by nothing) |
| metal__ckernel_sfpu_silu | silu row (legacy body dispatched, harness comment records the choice) |
| metal__ckernel_sfpu_log | log row (legacy _calculate_log_ dispatched; metal calculate_log_body raced transitively via Erfinv nodes) |
| metal__ckernel_sfpu_div_int32 | divint32floor row (DIV_INT32 re-pointed at calculate_div_int32_trunc; this header dispatched by nothing) |
| legacy__ckernel_sfpu_generic_moe_gate_topk_top16 | moegatetopk row (body EXECUTED via the consumer's :294 call; conversion refused by moegatetopk-fresh's cross-lane ruling) |
| legacy__ckernel_sfpu_generic_moe_gate_topk_top8 | moegatetopk row (same, :298) |

## Notes / staleness records

- The corpus manifest `sfpu_corpus_v2.tsv` (and the `sfpu_corpus.py` table) is a
  point-in-time audit at bb52e0fc33; its "zero test-source inclusion" notes for the 11
  converted ids are now historical — this lane's test-side selectors reference those
  entry functions.  The manifest's `mapping_state` fields are untouched (the
  test_sfpu_corpus assertions check the manifest record, not tree greps); a manifest v3
  refresh is follow-up work for whoever next regenerates it.
- QSR-only ids (headers with no BH/WH surface, e.g. legacy__ckernel_sfpu itself,
  legacy add/binary_comp QSR variants, metal quant's Quasar namesake) are outside this
  BH/WH census scope by charter, as before (laneED precedent).
- WH: converted ids on both arches except the BH-only ones — legacy experimental
  zero_pad + sparse_k_filter and metal experimental add_rsqrt (refused ids rope /
  logit_softcap / clamped_silu are BH-only too).  The conversion rows are BH
  (craq_archs=bh) per the ED/S4 precedent — WH legs ride the established wave-close
  WH re-enable path, not this lane.
- Expected merge conflict with lane ET's pushed `agent/e2e-metric`: `sweep_2x2_ops.tsv`
  only (v4 header + sem_class column + KERNEL-decided verdict rewrites vs these 20
  appended v3 rows).  Resolution: take ET's header/schema, re-append these rows with
  `sem_class=''`.  Every other file this lane touches is new or disjoint
  (sfpu_binary_test.cpp / eltwise_binary_sfpu_perf.cpp / test_sfpu_binary.py /
  perf_eltwise_binary_sfpu.py edits are additive blocks; ET's branch touches none of
  them except sweep_2x2.py/conf which this lane does not touch).
