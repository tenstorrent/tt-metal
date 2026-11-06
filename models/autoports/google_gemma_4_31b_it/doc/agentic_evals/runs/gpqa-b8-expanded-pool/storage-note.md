# Retained phase storage scaling from existing evidence

Host-only extraction; no configuration changes or new runs. `tt/benchmark_phases.py:53–54` defaults GEMMA4_PHASE_MAX_FILE_BYTES to4GiB (4294967296 bytes). At lines281 onward, exhaustion records truncation/loss and omits subsequent events; fitting a compressed artifact does not avoid this raw-file cap. Coordinator reports the local diagnostic allowance as512MiB; that local allowance is not the CI default and must not be propagated to scored runs.

| Evidence | Raw JSONL bytes | Paired steps | Active decode tokens | Bytes/paired step | Bytes/active token | Retained token events |
|:--|--:|--:|--:|--:|--:|--:|
|Current C8 local capacity/exactness|2609893|1786|6705|1461.31|389.25|1262|
|Prior C2 scored10|24352264|29498|56042|825.56|434.54|0|
|Prior C4 scored10|16573914|19871|56042|834.08|295.74|0|
|Prior C2 scored20|156834512|186277|367509|841.94|426.75|0|

All four inspected exports report no errors/truncation. Older scored archives omit token IDs, so their smaller bytes/step are not a direct retained-runtime storage forecast. The local current export includes524 prefill steps plus1262 retained decode completions, substantial idle-frame/round overhead, and asynchronous8-row payloads. Its average is a measured mixed-workload proxy, not a universal byte constant.

Two explicit retained-runtime extrapolations use measured old20 work367509 active decode tokens and186277 paired steps, scaled by N/20, combined respectively with current local389.2458 bytes/active-token or1461.3063 bytes/paired-step. They deliberately expose sensitivity to active concurrency and local overhead; neither is a statistical range or upper bound.

| Cohort | Work-based projected raw bytes | Step-based projected raw bytes | Step-based fraction of4GiB |
|--:|--:|--:|--:|
|80|572205319|1088830993|25.35%|
|160|1144410639|2177661986|50.70%|
|198|1416208165|2694856708|62.74%|

These measured-work extrapolations do not predict4GiB exhaustion through198, but do not guarantee safety: unseen task work, longer tails, lower effective concurrency, larger numeric encodings and logging overhead may differ. At198 the step-based scenario leaves approximately1.60GB of raw cap headroom; a~1.59-fold increase over that scenario exhausts the cap. The local512MiB allowance would already be below both80 projections. Replace this historical-work basis with repaired20 retained evidence once collected; do not delay40 or require a new run for this estimate.

Raw JSONL bytes above are actual filesystem sizes, including snapshot metadata, before compression. Existing extracted archives do not provide a matching phase-only compressed-byte denominator; no compression ratio is claimed. CI tar/zip transfer size, duplicate staging copies, stdout/server response logs and dataset/model storage are separate and can overlap on disk. Stock response logging retains complete JSON bodies, including reasoning; its cost is not included in these phase-only projections. Existing first-agent-reply-observability confirmation gives an explicitly guessed response-log allowance (~650.8MB for80 capped outputs, before duplicate workflow copies), not a measured scored-log forecast. Actual available runner disk remains unknown:4GiB recorder headroom is not host-storage admission.

Sources: private current `bringup/artifacts/gpqa-b8-expanded-pool/candidate/phases.jsonl`; collected phase JSONL under runs/gpqa-current10-c2, gpqa-c4-scored10-evals and larger-gpqa20-c2-evals; benchmark_phases.py. No raw transcripts, tensors or credentials copied into this note.
