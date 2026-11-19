# Representative comparator endurance gate

The historical linear comparator `9d33bc14...` is no longer eligible. Its newly measured `v6_historical_9d33_layer34_seed154.json` fails 282 rows, with minimum contribution PCC 0.9126810470518099, despite its earlier layer 0 seed 913/42/154 passes. Latency or the layer 0 subset cannot restore eligibility.

`tests/check_fused_evidence.py` now requires two distinct sets of source-bound evidence for every linear comparator:

1. `comparator_correctness`: matched benchmark layer 0, seeds 913,42,154, each 128-token prefill + 8192 decode tokens + 33 continuation tokens.
2. `representative_comparator_correctness`: separate layer 34 seed 154, with the same complete lengths and comparator source hash.

Benchmark comparisons still match layer, kind, sequence length, context, device, software versions and repetition counts. Representative layer 34 evidence never masquerades as a layer 0 benchmark. Manifest `required_endurance_seeds` must explicitly include the hard-coded minimum; additional seeds are allowed, omission or reduction is rejected. Full-attention comparators retain layer 3 seed 913 as their minimum.

Each correctness artifact now also proves finite prefill/output/contribution PCC≥0.995, no failed rows, at least one complete repeat, exact per-repeat position coverage without duplicates, all 33 distinct continuation positions, one trace capture, deterministic replay, allocation tracking with program-cache inclusion, serial TTNN device execution and source identity. Ordinary stress harness and reference hashes must match the reviewed files. Explicit `oracle_assisted=true`, intervention harnesses, mismatched oracle parameters/source hashes, missing rows or inconsistent source identities are rejected.

Cached canonical HF outputs are permitted for comparator selection through the unchanged `stress_fused_decoder.py` / `stress_fused_snapshot.py` harnesses. This is CPU reference reuse only; it does not permit injecting HF recurrent state, selecting HF outputs, or accepting diagnostic repair substitutions. Final `check_stress` remains AST-identical: fresh HF reference, at least 3 repeats, exact current production source, original PCC, tracing and completeness requirements. `check_all` is likewise unchanged. This task changes eligibility checks, not quality thresholds.

## Schema and pending controls

`v6_best_correct_comparator_schema_example.json` gives the c823 linear / a478 full manifest structure with explicit `PENDING` filenames. It is documentation, not an eligible acceptance manifest. Paths resolve relative to the manifest.

- c823 candidate source: `v6_corrected_fma_full_v5_direct_coeff_cast_placement_fused_matmul_integration.py.snapshot`, SHA `c823e6f6f8480a9550276856a7aa660b9308ff598c6565a38c2bc54460440ece`.
- c823 matched benchmark: `v6_direct_coeff_l0_s256.json`.
- c823 representative evidence already passes the strict gate: `v6_direct_coeff_layer34_seed154.json`. Layer0 seed 913/42/154 controls still require parent-owned runs before complete comparator eligibility.
- Full comparator source remains `v4_padded_views.py.snapshot`, SHA `a478451bdf99a1b6a7a496cc17d071166465bca1b9dcba64ee4e1429fba261f4`.
- The historical full stress report has generic provenance `decoder="functional"`, although its explicit snapshot field identifies FusedDecoder. The final gate accepts **no exception**: parent must rerun with explicit `QWEN_DECODER=fused` and the same a478 snapshot. The original artifact remains unchanged and ineligible until replaced in the manifest by valid new evidence.

An intermediate attempt to reconcile that generic label was rejected by the parent. Its source and validation report are retained as `v6_comparator_label_exception_rejected.py.snapshot` and `v6_representative_comparator_existing_evidence.json`, explicitly superseded by `v6_representative_comparator_strict_validation.json`. The final checker has a strict fused provenance requirement for both ordinary and snapshot runs.

## Host validation and frozen sources

```bash
python_env/bin/python -m unittest models.autoports.qwen_qwen3_6_35b_a3b.tests.test_comparator_eligibility -v
```

All 5 tests pass, including 15 corrupted/intervened evidence cases, mandatory seed minima, separate layer 34 dispatch, missing/wrong representative evidence, ordinary cached-reference acceptance, and old full-label rejection. The real failed 9d33 report is rejected independently. Synthetic manifest fixtures exist only inside temporary test directories and are never used as device evidence. Neither Torch nor TTNN is imported by this validation.

- Previous checker preserved: `v6_before_representative_comparator_check_fused_evidence.py.snapshot`, SHA `d418729bffe6c8614726b1f9ae3273ea9185ba984268d9066683d78f9d34f914`.
- Final checker SHA `65a2763f23e275831cf74d81f2bc87a0f83389552ada039393d89f944ece072b`; exact copy `v6_representative_gate_check_fused_evidence.py.snapshot`.
- Test SHA `e9ab83a51a7968e5e4ce496313b4c8e2a369efae432087d4ff18e6bb4d099347`; exact copy `v6_representative_gate_test_comparator_eligibility.py.snapshot`.
- Strict host validation/provenance: `v6_representative_comparator_strict_validation.json`.

No production decoder, frozen comparator, device artifact or hardware state was modified. Complete comparator eligibility remains pending the required hardware controls and matched benchmark results.

## Completed comparator controls (parent-owned hardware)

After the checker froze, all c823 matched layer 0 seeds 913/42/154 completed, and the a478 full comparator rerun completed with explicit fused provenance. The strict helper validates all five source-bound evidence records, including the existing c823 layer 34 seed 154 run; exact hashes are in `v6_representative_comparator_completed_controls.json`. No generic-label exception is needed.

`v6_best_correct_comparators_ready_template.json` resolves every comparator path to the completed records and uses `v6_direct_coeff_l0_s256_10pf.json` for the linear benchmark. Both comparator benchmarks now have 10 prefill samples and 60 decode samples. The selected c71ea candidate still requires matching clean measurements before the complete comparison gate can pass; the existing three-prefill-sample diagnostic must not substitute. The template retains prospective `../final_0.json` / `../final_3.json` candidate paths and does not claim that they are current-source acceptance evidence.

Validated comparator timing artifacts (milliseconds, diagnostic comparison inputs): c823 prefill 1053.9965478237718 / traced decode 2.020052634179592; a478 prefill 9.004254825413227 / traced decode 0.5877036601305008. These are not a final before/after result without the matched selected-candidate measurements.
