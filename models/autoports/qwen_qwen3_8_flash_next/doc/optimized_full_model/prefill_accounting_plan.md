# Six-bucket prefill accounting plan

The prepared recipe profiles one warmed eager prefix chunk for each logical
length1/2/4/8/16/32 at position0, C262144 and B1 on the serialized1x4 mesh.
The body includes embeddings, real L0/L1/L3, final mixer/LM head and prefill
output copy. It excludes input upload, cache reset, profiler drain, output
finiteness reads, sampling, checkpoint save and scheduler orchestration.

`PREFILL_BUCKET_N`/`_END` delimit the entire reduced body. There are **no layer
markers**. Therefore the measured total cannot be weighted35/1/12, multiplied
by48/3, or reported as full48 prefill device time. Repeated generic GR kernels
also prevent assigning class membership from kernel names alone. The six
bucket rows are alternative shapes, not consecutive chunks to add together.

## Prepared CPU analyzer

`../../tests/analyze_full_model_prefill_profile.py` reads the completed final
CSV and six tt-perf-report CSVs. It checks coverage hashes, all four ranks,
no live traces, warmed programs, finiteness and positions; joins selected
merged-call IDs to raw geometry; emits per-bucket device sums, raw rank totals,
major kernel groups, advice and provenance. It refuses existing outputs.

```bash
D=models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model
python models/autoports/qwen_qwen3_8_flash_next/tests/analyze_full_model_prefill_profile.py \
  --profile-dir "$D/prefill_perf_v1" \
  --driver "$D/prefill_ops_v1.json" \
  --source-manifest "$D/final_sources_v2.json" \
  --resident "$D/exact_integrated_resident_v1.json" \
  --output-prefix "$D/prefill_profile_analysis_v1"
```

Use a newer final resident artifact if available. Source differences are
reported explicitly. TTFT remains a separate all48 public-generator measurement
with its own prompt length and reset/prefill/first-sample/read boundary. A
reduced eager kernel sum cannot be subtracted from TTFT to measure host overhead.
The analyzer sets `full48_prefill_device_estimate_ms` to null and leaves advice
review pending. Compilation, formatting and CLI help checks pass; **no analysis
has run yet because the six-bucket raw evidence is pending**.

## Required human reconciliation

For each bucket, examine the largest kernel groups, precise shapes, dtypes,
math fidelity, raw worker/shard geometry, and collective input/output sizes.
Separate gather transport from local reductions; distinguish generic operations
by native compute/data-movement source. Compare logical lengths using the same
source snapshot, without treating common32-row padding as identical semantics.
Keep aggregate source groups descriptive: a shared kernel may occur in multiple
layers or terminal work and is not itself a decoder-class measurement.

If per-class attribution is later necessary, derive exact source-specific
sequence boundaries from distinctive surrounding operations, independently
validate them for all six buckets, and prove a complete non-overlapping
partition into entry, L0, L1, L3 and terminal work. Preserve the explicit mapping
and raw call IDs. Until that proof exists, leave the weighted all48 field null.
A dedicated future layer-marked profile would be stronger evidence, but this
plan does not authorize another device job or modify the frozen profiler.

Disposition generic advice against actual evidence: tracing is already enabled
in public prefill, so eager gaps are not incremental trace savings; lower
fidelity/dtype changes conflict with the user's fixed policy; small-matmul
L1/block/subblock changes need exactness across logical lengths; compact-GR-down
remains rejected after integrated mixed-prefill failure despite isolated gains.
Record applicable geometry experiments, measured rejections and untried work
honestly. Reduced profile finiteness is not HF quality or full-model acceptance.

The authoritative optimization window is `stage_budget.json`, sourced from
`get_goal.createdAt=1794654561`: start2026-11-14 11:09:21UTC and four-hour deadline
15:09:21UTC. Finishing final evidence does not relax mandatory correctness or
the strict useful-weight/KV functional floor.
