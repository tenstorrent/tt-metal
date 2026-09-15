# BFP4/LoFi research checkpoint index

This directory is an experimental checkpoint, not a production SDPA API or a replacement for the four frozen choices in [the original Pareto checkpoint](../pareto-v1/README.md). The original BF16 input is the accuracy reference unless a diagnostic explicitly labels a different reference. The older [BFP4 numerical model](../bfp4-lofi-v1/REPORT.md) is not silently revised to fit a new device route.

## Read the evidence first

Run the standard-library, read-only [checkpoint validator](validate_research_checkpoint.py) from the repository root:

```sh
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_research_checkpoint.py
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_research_checkpoint.py --json
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/test_validate_research_checkpoint.py
```

`--json` prints the audit to stdout only. The validator does not import Torch/TTNN, execute producer code, modify sources/records, or rerun device kernels. It checks coverage, recorded gates, source manifests, and timing arithmetic. Exit codes are 0 for complete evidence/provenance PASS, 1 for evidence failure or files changing during the audit, and 2 for PENDING evidence or provenance warnings. Missing optional/planned files are PENDING, never a successful partial suite.

Evidence and provenance are separate axes: a run's `sources_unchanged` does not establish that today's sources match that run. Missing critical manifest entries and current-source drift are reported separately from numerical/integrity failures. The required-source policy lists important dependencies; it is not an exhaustive firmware/compiler/transitive dependency closure. Where older centering records omit explicit finite-output or source-stability booleans, the audit identifies the corresponding assertions in the producer only if its current hash matches the recorded hash. This is a disclosed producer-assertion witness, not a recheck of downloaded tensors.

Snapshot at index creation, 2026-09-15: 47 of 53 expected evidence files pass their recorded gates; six V8 constant-V records are pending. There are no observed evidence failures. Provenance warnings remain in 22 files, principally an older native 32K manifest/current producer drift, omitted `chain_link.hpp` pins, and V4 centering's omitted reference-helper pin. The validator output is authoritative as new records arrive; this snapshot is not a promise of complete provenance.

## Tested paths and bounded promotion evidence

| Area | Implementation / explanation | Evidence and interpretation |
|---|---|---|
| Native-exp LoFi FP32 streaming | [integration notes](EXP_NATIVE_INTEGRATION.md), [qualification driver](native_exp_qualification.py), [full-chip driver](fullchip.py), [private streaming header](streaming/compute_streaming.hpp) | [32K suite](native-suite-32k-v1.jsonl), [256K suite](native-suite-256k-v2.jsonl): 82 cases each, including LoFi B8/B4 and accurate controls. Finite outputs, replay identity and intended coverage are checked. These are correctness records, not 82 timing measurements or a universal accuracy-threshold pass. |
| Adaptive native BFP4 quantization | [primitive](adaptive_bfp4_round.py), [primitive contract](adaptive_bfp4_round/README.md), [full-chip integration](adaptive_fullchip.md) | Eight `adaptive_bfp4_round/adaptive-*-v1.json` primitive records check exact decoded hashes and FP32 oracle gates across search modes and input families. This qualifies the quantizer, not a universal attention improvement. Attention integration has separate evidence, outside the validator's core scope. |
| Identity-rescale batching | [resident driver](identity4_resident.py), [distinct/full-chip driver](identity4_streaming.py) | Five `identity4_streaming/identity4-*-v1.json` records compare off/on, require identical output hashes/values, and verify timing/useful-FLOP arithmetic. Repeated-KV resident coverage favors identity rescaling; distinct-KV and full-chip comparisons are mandatory and included. The optimization is modest, not evidence for 80% utilization. |
| V4/V8 centering | [V4 driver](value_centered_fullchip.py), [V8 driver](value_centered_b8_fullchip.py), [value-smoothing explanation](VALUE_SMOOTHING.md), [effective-value preprocessor](effective_v_preprocess/README.md) | `valuecenter-*.json` and `valuecenter-b8-*.json` compare none/original-mean/matched-mean on normal/common-V/constant-V at 32K and 256K. Six V8 constant-V files were pending at index creation. These are measured alternatives, not three equally recommended policies. Attention-only timing excludes the epilogue; combined timing includes it. Constant-V centered relative error is undefined and must not be turned into a numerical pass/fail. |
| Accurate K-centering attribution | [diagnostic driver](accurate_kcenter_diagnostic.py) | [1024 diagnostic](accurate-kcenter-1024-v1.jsonl), [32K diagnostic](accurate-kcenter-32k-v1.jsonl): normal/common-K off/on, immutable original inputs, BF16 preparation oracle, exact-shift invariance and prepared-input versus original-input references. Diagnostic only: no change to the frozen accurate default. |

The validator does not introduce a new global L2 cutoff. An integrity PASS means the expected records meet their explicit finite/replay/exactness or recorded numerical gates; it does not mean every stress distribution is acceptable for every application. Review L2, PCC, gain and centered-error fields together. Useful attention FLOPs count the attention problem, not additional compensation/preprocessing matmuls as useful work. Resident per-core measurements and end-to-end chip timings answer different questions and should not share an unlabeled Pareto axis.

## Important explanations and diagnostic controls

- [Numerical findings](NUMERICAL_FINDINGS.md) and [risk matrix](NUMERICAL_RISK_MATRIX.md): numerical mechanisms and remaining qualifications. [Progress](PROGRESS.md) is an intermediate narrative, not a current completeness manifest.
- [Q256 FAST correction bug](FAST_Q256_CORRECTION_BUG.md): private correction-state reset required for these experiments; the frozen implementations remain unchanged.
- [HiFi2 B8 attribution](HI2_B8_ATTRIBUTION.md): distinguishes high-level typecast from preprocessing pack routes. A B8 format name alone does not establish the same rounding pipeline.
- [FP32 matmul floor audit](matmul-fp32-floor-audit.md) and [CPU alignment model](matmul_fp32_alignment_model.py): a source-pinned model with device crosschecks; FP32 destination is not a claim of a strictly IEEE FP32 dot product. This is hardware attribution, not an additional attention qualification suite.
- [Scale-search research/model](bfp4-scale-search-report.md): exponent-constrained native BFP4 differs from NVIDIA's FP4 scale formats. Representation-MSE improvement does not guarantee attention-L2 improvement.
- [Residual error budget](RESIDUAL_ERROR_BUDGET.md), [Hadamard preparation](HADAMARD_PREPROCESS.md), [native rotated QK](QK_HADAMARD_NATIVE.md), and [Q centering/rotation](Q_CENTER_HADAMARD.md): targeted experiments; do not infer that a transformation is enabled in a native baseline.

## Layout/performance probes and non-promoted alternatives

- [Resident pack probe](pack_resident.py): measures copy/pack pipeline throughput, not attention TFLOPs. Native scalar BFP packing and BF16 blocked packing must not be conflated: the ordinary blocked B8 exact-control test failed. A faster invalid packing layout is not a result.
- [Separate P8](p8_streaming/README.md): independent intermediate-format experiment, scalar pack and matched denominator concerns. It is not automatically better because the payload is compressed.
- [Separate P16](p16_streaming/README.md): failed the device allocation budget for the requested unchanged chunks/input buffers. The 1,474,560-byte CB payload plus 111,616 reserved bytes exceeds 1.5 MiB by 13,312 bytes.
- [Padded P16](padded_p16_streaming/README.md) and [padded pack probe](padded_pack_probe/README.md): address-stride/alias and custom blocked-pack controls, with separate FIFO accounting and replay-resource constraints. Exact roundtrips establish the tested layout, not an attention speedup; scalar padded P16 did not beat the existing FP32 width-4 native baseline.
- [L1 denominator alternative](l1_denom_streaming/README.md): isolated tradeoff exploration, outside the core qualification manifest.

The active `exp_grid7_probe` and `exp_lut_macro_streaming` work is deliberately **not qualified by this checkpoint validator**. Likewise, an unlisted JSON, kernel, or README is not promoted merely because it resides here. Extend the explicit evidence contract only after reviewing its source pins, exactness/numerical gates, branch coverage and performance arithmetic.
