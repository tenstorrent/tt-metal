# Live-band Q/K preparation: bounded follow-up and unapplied composition

Prepared CPU-only on 2026-11-13. No hardware/build/TTNN import or production edit. `live_row27.patch` is unapplied and defaults `recurrent_live_row=False`. The source before/after hashes and the complete queue manifest are in `live_row27_proposal.json`. Do not conflate the completed B1 diagnostic, these unrun queues, or the unrun composed decoder.

## Completed diagnostic reviewed

`recurrent_live_row_b1_smoke25_v2.json` reports PASS and normal mesh closure on TP4, B1/BF16/normal, watcher10 and allocation tracking. All four eager and traced steps are bit-exact for state and FP32 readout versus frozen cumulative-exp control. Rebinding, queued50 eager calls, queued50 changed-input trace calls, and post-eager trace reuse pass. Tracker rejects its deliberate live allocation; both traces have zero unsafe buffers before and after release, without exemptions. Independent FP64 prepared Q/K minimum head PCC is 0.9999999999999983. No timing was collected.

The corrected oracle maps Blackhole SFPLOAD/SFPSTORE slots to left-even, left-odd, right-even, right-odd columns, not four contiguous quarter-width strips. The current probe pins this mapping against `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h`'s even/odd load description. It retains the exact baseline comparison as well as the independent FP64 norm oracle. The v2 pass covers only the stated case; it does not establish wider state dtype/ring/padding coverage or speed.

## Prepared queues and receipt gates

`live_row27_quality.sh` contains14 bounded jobs /60 cases:

- B1/3/17/32 × BF16/FP32 state × normal, zero_qkv, zero_state, small, large, poison:48 cases.
- Ungrouped B3/B32 × both state dtypes × normal/poison:8 cases.
- One-core B3 user/head/ring wrap × both state dtypes × normal/poison:4 cases.

Each case uses four changed-input/state steps, distinct rebound input/state/output owners, queued50 eager and queued50 changed trace calls. Prepared Q/K has independently allocated/rebound owners and FP64 per-head normalization gates≥.995; complete recurrence retains **bit-exact finite** state/readout comparisons against frozen cumulative-exp arithmetic. Small/large scale only Q/K by2^-18/2^12; V and state stay bounded. Poison puts NaNs in dead Q/K/V/gate padding, including both ring slots; normal preparation also tests nonzero finite dead rows. B1 resolves grouped-QK false even with the requested grouped policy on.

`live_row27_perf.sh` first requires every quality receipt. Its8 jobs cover B1/3/17/32 × BF16/FP32 state with normal inputs, grouped policy on, default head ownership, warmup5,11 samples,100 replays/sample. Watcher/profiler/tracker are disabled for the target process; exact correctness checks still run. The timed boundary is the complete standalone packed recurrence and readout, not the prepared-QK diagnostic and not the whole decoder. The receipt checker writes `live_row27_timing.csv` after all8 succeed. Speedup ratios are descriptive medians, not a significance or keep/reject gate.

Both scripts use the existing leased `run_job.sh`, fresh filenames, STOP checks,1200-second per-job bounds, source pins, and existing command/native-build provenance. `live_row27_receipts.py` requires actual case coverage, minimum queues, FP64 oracle cardinality/scores, exact finite comparisons, effective tracker positive controls, watcher logs without error markers, and correct instrumentation regimes. It does not silently accept partial results. Resume a partially completed script by scheduling its remaining manifest jobs explicitly; scripts deliberately refuse existing output names.

Root-owned launch commands, after its active queue closes:

```bash
bash models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/live_row27_quality.sh
bash models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/live_row27_perf.sh
```

These scripts have not been executed by this agent. The source-only probe check was executed, producing `live_row27_source_cpu.json`. The receipt checker was exercised against the actual B1 v2 result and six negative mutations; results are `live_row27_cpu.json`.

## Proposed main hook and composition proof

The patch changes only `tt/multichip_decoder.py`. It exposes explicit `recurrent_live_row` boolean, off by default. Constructor guards require GDN, Blackhole, cumulative_exp_scalar, packed batched convolution including B1, and the existing legal column grains. This makes S1 prefill/tails/resumes use the same supported packed contract instead of encountering an unpacked runtime branch. Chunked S>1 recurrence is unchanged.

For the S1 recurrence source tuple, assembly is:

1. `recurrent_variants.sources(..., cumulative_exp_scalar, grouped=B>=3)`.
2. Optional `live_row_sources` changes the six pointwise calls inside `prepare_vector` and inserts their restricted-band SFPU wrappers/helper.
3. Optional existing `prediction_sources` changes only its previously reviewed ordered prediction loop/helper.
4. Optional existing `column_sources` transforms reader, compute ownership, and writer together; its normal grain1/2 wrapper receives all three sources. Without columns, the normal head-owner wrapper receives the reader/compute pair.

When live-row is off, existing prediction-only assembly produces the same source tuple as before. No global registration, AST/file reading, numerical-policy mutation, hidden fallback, new allocation, persistent buffer, collective, or inter-layer conversion is added to runtime.

The live transform leaves the reader byte-identical and does not alter the recurrence kernel main body, writer, CB descriptors, runtime arguments, tensor shapes, grouped head assignment, FP32 live readout, or BF16 persistence. `prepare_vector` still uses the native row reduction tree, original epsilon/Q-scale bits, pack/transposes and prepared-column replication. Only the minimum live four-row SFPU band is processed by pointwise operations; this is not a literal one-row SFPLOAD mode. The exact supported lane mapping is proven in the existing source-only check and measured only in the completed B1 case so far.

CPU composition checks cover grouped/ungrouped × poisoned/unpoisoned × prediction on/off. Live and prediction transformations commute byte-for-byte. Applying live before or after the existing column transformation yields the same reader/compute pair, with the column writer byte-identical. In both head and column families, live leaves `kernel_main` unchanged relative to its matching control. Thus it introduces no new state/readout ownership, CB lifetime, or cross-core dependency. Across384 batch/core/grain/grouped partition cases, each `(head,value-column)` has exactly one owner and every column is covered. The column wrapper already keeps the entire K-reduction dimension on one owner; live does not split that dimension or restore the old ownership. Eight CPU constructor cases include expected rejection of attention, non-Blackhole, wrong recurrence variant, missing B1 packed convolution, and a nonboolean flag.

These are **source composition checks**, not compiled/hardware equivalence of combined families. The standalone queue intentionally isolates live-row against the best cumulative control; it does not claim to measure the column/prediction combinations. Before keeping a composed policy, root must run unchanged full-decoder HF/TP4 gates and exact state/readout checks for live-only, live+prediction, live+column grain1/2, and live+prediction+columns as applicable, including B1/S1/resume and B32 stress, then compare complete warmed/traced decoder costs. First compile/API errors remain adaptation work. Any speed observed only in the isolated head-owner boundary cannot select the final composed default.

## Capacity and context

The hook changes no dtype, state/cache size, layout, physical output shape, persistent allocation, or public logical-length requirement. It does not justify a context-contract reduction. Its inter-layer residual remains the existing rank-local shard; no new collective exists. Source proof does not establish physical performance or stress safety: wider poison/ring/FP32 checks and final full-decoder validation remain pending.
