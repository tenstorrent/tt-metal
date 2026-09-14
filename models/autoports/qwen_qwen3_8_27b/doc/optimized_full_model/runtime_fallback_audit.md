# Final runtime fallback audit

Verdict: clean for the measured default TP4 split-sampling token-out path.
This verdict combines executable guards, full-model counters, native profiler
rows and source inspection; it is not inferred solely from constructor flags.

| Contract | Evidence |
| --- | --- |
| Fully traced steady decode | `after_full_final.json` deferred steady counters contain exactly127 model replays,127 sampling replays and127 history appends, with no capture, token/position/RoPE/page refresh or readback. `prefill_contract_full.json` checks nonblocking model/sample replay order and stable all-device addresses; `prefill_deferred_reduced.json` independently forbids host conversion/upload/sync during the hot loop. |
| Split greedy and sampled modes | Final benchmark declares split, no power-of-two padding, host_sampling=false. `contract_final_full_b32.json` and `prefill_contract_full.json` exercise greedy/top-k/top-p transitions, reseeding and recapture. All128 default benchmark tokens equal eager-prefill/immediate controls. |
| Persistent token feedback | `tt/generator.py` binds sampler output `tt_out_tok` to the next model token input; model/sample traces advance position and RoPE on device. Page tables are retained and uploaded only after contents change. Final steady counters and changed-page physical K/V tests verify these boundaries. |
| Actual native sampler | `perf_summary.json` partitions trace IDs and native rows from all4 devices. RoutePrep/LargeIndices/RouteFinish TopK uses81/110/2 cores; candidate BF16/UINT16 gathers grow32 candidates/rank to128, without full-vocabulary all-gather. No generic `TopKDeviceOperation`, force-argmax or host sampler appears in the selected token-out interval. |
| Frozen decoder and CCL | Native rows show BFP4/LoFi decoder projections, BF16 activations/residual/CCL, BFP8/HiFi2 head; persistent async Ring/two-link AR consumes the inherited L1 width-sharded B1 stream. Profile and full benchmark runtime hashes match. `full_path_checklist.md` links exact inherited rejection records. |
| Trace lifetime and shape safety | Full64 prefill checks cover1/31/32/33/4095/4096/4097, changed prompts/modes/pages, retained caller outputs and50 captures guarded against new program-cache entries. Final watcher/tracker covers representative layers on all4 devices without disabling Ethernet checks. |
| Batch/cache semantics | Full64 B32 contract covers slots31/0, mixed31/33 prompts, inactive rows, physical page remapping, history overflow before mutation, and continuation PCC.99921447. |

The request boundary intentionally uploads changed prompt/reset state and reads
the first sampled token inside TTFT. Deferred delivery reads the complete
persistent history allocation once after the loop, then returns its valid
prefix; both transfer and output construction are included in the final decode
rate. These request boundaries do not occur per token. History append currently
copies its allocation each step, so large retained capacity can increase cost.

Teacher forcing intentionally uploads reference feedback and reads sampled
outputs; its separate40.238 t/s metric is not the no-host-boundary path.
Callbacks/host compatibility retain explicit immediate behavior. Long prefill
above4096 uses the valid eager chunk path, while decode remains traced. These
are declared API modes, not silent fallbacks in the measured default loop.

The generic allocation and fabric advice warnings are classified with controls
in [anomaly_ledger.md](anomaly_ledger.md). There is no unresolved runtime failure
or automatic eager/CPU substitution in the measured path.
