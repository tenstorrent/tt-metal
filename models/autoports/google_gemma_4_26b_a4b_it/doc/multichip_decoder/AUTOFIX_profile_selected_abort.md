# AutoFix: selected sliding profiler abort

## Final status

**Controlled with a verified configuration workaround:** use
`--op-support-count 100000` for the selected TP4 sliding capture. The unchanged
4096-token prefill and 128 traced decode workload now closes normally, produces
its run JSON and both device reports, and retains every expected program in
every measured window. The parent confirmed outer process exit 0 (session
62915); the whole-layer summarizer also exited 0.

The underlying C++ large-buffer allocation bug remains unpatched, outside this
Python model-stage change. No assertion, marker validation, model computation,
trace window, or correctness threshold was bypassed. The failed 250000-count
capture remains excluded. This resolves this capture blocker; it does not
independently grant a complete model-stage pass.

## Starting evidence and diagnosis

The pre-experiment report is `AUTODEBUG_profile_selected_abort.md`.
`profile_selected_sliding.log` reached both performance signpost pairs and
`TP_DONE 4`, then aborted during mesh close, leaving no C++ device report.
The surviving host trace completed LAST_FD_READ on devices 3 and 2 and ended
in subsequent profiler readback. Missing CSV postprocessing was secondary.

`tt_metal/impl/profiler/profiler.cpp:2392–2394` takes two uint32_t arguments
and allocates with `size * num_dram_banks / sizeof(uint32_t)`. Multiplication
wraps before the size_t division. The failed 250000 count sets 12000000
**bytes** per RISC. With 5 RISCs, 8 banks, and the source-supported 14–20
rounded cores per bank, required per-device buffers range from 6.72 to 9.60GB;
the allocation wraps below 4GiB. `profiler.cpp:1359–1382` nevertheless copies
the complete bank data into this undersized vector. The concrete source defect
predicts memory corruption during the observed readback phase.

At 100000, the allocation is 2.688–3.840GB and does not wrap, including the
Blackhole architectural maximum of 140 TENSIX plus 14 Ethernet cores. The
110 logical compute/L1 banks in compile commands are a lower bound, not a
recovered exact SoC profiler core count. The diagnosis does not depend on
assuming the exact live Ethernet harvesting geometry.

The older fused-stage timestamp-rollover explanation has no matching marker
endpoints in this failed capture and is not used to justify this workaround.

## Focused experiment

Hypothesis: the oversized profiler host allocation corrupts memory during
close; choosing a supported count below its 32-bit boundary preserves the
same model workload and allows full validated collection.

From `/workspace/tt-metal`, the parent ran exactly:

```bash
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
python_env/bin/python -m tracy -r -p -v \
  --op-support-count 100000 --no-op-info-cache \
  --disable-device-data-dump-to-files --disable-device-data-push-to-tracy \
  -o models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/profile_selected_sliding_100k \
  -n tp4 -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder \
  --tp 4 --layer 0 --length 4096 --steps 128 --trace --profile \
  --prefill-timing-samples 1 \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/profile_selected_sliding_100k/run.json \
  > models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/profile_selected_sliding_100k.log 2>&1
```

No `--check-exit-code` flag was used. Child success is independently evidenced
by post-close run.json, normal device-close messages, and complete C++ reports;
the parent observed outer exit 0. Ordinary C++ postprocessing and final marker
checks remained enabled. No mid-run dump, SUM, accumulate, trace-only mode,
source patch, shared-cache deletion, or additional reset was part of this
focused control. Unbuffered/faulthandler environment flags improve diagnostics
without changing model arithmetic.

Verdict: **workaround verified; original cause strongly supported by a concrete
source defect and the controlled passing contrast.** An aborting native stack
was not recovered, so the specific allocator/abort mechanism is not claimed.

## Verification and artifact identity

The independent source-only agent performed CPU CSV/JSON checks; the parent
owned all device execution. Both raw and final report integrity checks pass:

| Per-device check | Result on devices 0, 1, 2, 3 |
| --- | --- |
| Total C++ program rows | 21634 each |
| Prefill host-expected / recovered programs | 2585 / 2585 each |
| Decode replay sessions | 128, IDs 1–128 |
| Unique programs per replay | 126 |
| Operation IDs | Identical set in every replay |
| Duplicate keys / missing or nonpositive FW spans | None |
| Missing host metadata / malformed JSON | None |
| Signposts | Ordered prefill start/end and decode start/end |
| Fatal, abort, full-buffer, dropped-marker warnings | None |

Checks are preserved as
`profile_selected_sliding_100k/capture_integrity.json` and
`profile_selected_sliding_100k/final_capture_integrity.json`.
The final exported CSV is
`profile_selected_sliding_100k/reports/tp4/2026_09_27_00_43_07/ops_perf_results_tp4_2026_09_27_00_43_07.csv`.
`profile_selected_sliding_100k/whole_layer.json` preserves one complete prefill
window and 128 complete decode windows. No timing from the failed capture is
used, and this report makes no performance-improvement claim.

Exact SHA256 values and sizes for the failed/passing logs, source files,
run JSON, raw device CSV, host capture/metadata, final CSV, integrity checks,
and whole-layer summary are in
`profile_selected_sliding_100k/abort_workaround_provenance.json`.
Key identities:

- Model runtime: `8b59370cda6f4ff88157de123123509036f2e91e8054000c809752e21f933175`.
- Runner: `06f0a0573dbd14e852be4a27c88b72faa610a51961b90cf33365025747898581`.
- run.json: `12cb1d92791a9007e5c7985c096fcb86edf518ec5c16a40ec1bda4e63339d2e1`.
- C++ device CSV: `6816d3678ca10621274fe6ef09e3ce3f7282874003b5be5eb52581e912a3cf74`.
- Final ops CSV: `cc943c26c9eeb36295990bc638f619dbf49179e7880418875a903c468cd71b0f`.

## Remaining general-runtime defect

A durable C++ repair should widen the multiplication before evaluating it,
for example `size_t(size) * num_dram_banks / sizeof(uint32_t)`, and include a
host regression covering a total above 4GiB without requiring giant hardware
allocations. It would require the repository-prescribed C++ build and relevant
runtime verification. That repair was not added to this Python stage.
The accepted 100000-count capture is fully validated and this profiler issue
is controlled for the current workload. No further hardware run is required
for this workaround's evidence.
