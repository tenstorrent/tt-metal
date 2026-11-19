# AutoFix — local reduction logical shapes

Diagnosis: [AUTODEBUG.md](AUTODEBUG.md), isolated inspection completed. The
native reducer builds an unpreallocated output from padded_shape. The decoder
incorrectly treated its spatial extents as the original logical extents.
Confirmed against fast_reduce_nc_device_operation.cpp and the optimized baseline.

Failing contract: indexed expert output must be `[1,1,2048]`, scan projection
must have one logical row, and recurrent reduction must preserve eight logical
heads. Full-attention prefill avoids the indexed branch, explaining why it ran.

Prioritized fixes:

1. Restore explicit logical/padded views at the singleton reductions (chosen).
2. Supply preallocated logical output tensors (valid alternative, unnecessary).
3. Change shared reducer or reshape validation (rejected: native contract is
   consistent and the user scoped this stage to model-local decoder/tests/docs).

Changes only in tt/multichip_decoder.py for this repair:

- Indexed expert reduction: logical `[1,1,2048]`, padded `[1,32,2048]`.
- Ordered projection reduction: logical `[1,1,width]`, padded `[1,32,width]`.
- Recurrent reduction: crop the spatial head extent to8 before its return permute.

The parent independently found and applied these three changes while AutoDebug
was still inspecting; its report explicitly identifies that revision boundary.

Validation:

- Python compile and Black passed.
- `smoke_l3_s32_v4.json`, exact command in its provenance sidecar: PASS, whole
  prefillPCC0.999930024, decode min0.999936756; written K/V PCC>0.999988.
- `smoke_l0_s32.json`: PASS, prefillPCC0.999988819, decode min0.999995943;
  convolution PCC~1 and recurrent state PCC1.0.
- Both runs complete eager decode, capture, repeated changing-input/position
  replay, determinism and device-only audit with TT_METAL_TRACE_ALLOC_TRACKING=1
  and TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0.
- No native op changes, threshold changes or fallback were used.

Resolution: the three shape defects are repaired and device-verified on the
initial shapes. This is not stage acceptance: boundary, batch, long-context,
watcher, stress, performance and review gates remain work.

Additional AutoDebug observations at that initial checkpoint (subsequently covered):

- Per-row PCC remains diagnostic, as explicitly required by the user; do not
  turn the report's suggested per-row aggregate gate into a stricter contract.
- B1 decode_batch now calls decode_forward, preserving its ordered linear
  router control. FP32 topk versus baseline BF16 selector is still a deliberate
  policy difference to validate on real activations and near ties.
- Add all-step eager/trace equality and live-trace prefill/continuation coverage;
  the existing baseline comparison already checks each replay output, but the
  stronger direct trace control will make stale-input diagnosis unambiguous.
- Larger numerical/stress coverage and profiler evidence remain pending.

## Replicated direct-AR prefill boundary repair

Observed: selected mixed-stack prefix33 reached concat with mixed L1-sharded
singleton outputs and DRAM aligned-chunk outputs. Native concat correctly
rejected this. Evidence preserved under failures/selected_stack_concat/.
The earlier sharded path always reduced to DRAM and did not expose this.
Fix: normalize each attention chunk and each MoE chunk to DRAM immediately
before multi-chunk prefill concatenation. Singleton decode retains L1 output.
Same logical lengths/cache updates; no padded token enters attention state.
Reset8 follows the failed multichip run. Selected mixed-stack rerun is required.

Selected mixed-stack retry passed after the prefill concat repair:
`mixed_stack_selected.json`, minimum whole-output PCC0.999707583.

## B32 sharded concat fan-in repair

Observed: natural B32 linear prefill passed; eager decode failed the native
sharded concat guard `num_input_tensors <= cb_dst_id` with32 inputs. Unlike
the earlier DRAM path, direct-AR leaves per-request attention in L1 shards.
Evidence: failures/natural_b32_concat/natural_l0_b32.log. Reset9 follows the
failed host-validation run; no hang/no killed process.
Fix: group attended rows into concatenations of at most16 inputs, then merge
the groups. This preserves L1 layout and logical row order without32 DRAM
conversions. Both selected natural B32 reruns passed: linear minimum decode
PCC 0.999902546 and full minimum decode PCC 0.999854728, with all-step eager
versus replay equality and deterministic repeated streams. B1 is unchanged.

## B32 Tracy fixture reduction

Contract: obtain complete four-rank prefill/decode CSVs and human reports for the
selected layer kinds without exhausting Tracy source locations. Diagnosis above
is source-backed; metadata caching was tested and refuted. No decoder/native
change is needed. First-choice repair is an unprofiled paired seed carrying exact
post-prefill physical states, then a narrow profiler fixture with one natural
request prefill and one B32 decode capture plus 16 replays. Second-choice fallback
is device-profiler collection/postprocessing if reduced Tracy still cannot fit.

`run_multichip_decoder.py --save-prefill-states` exports setup-only snapshots
outside measured windows. `run_multichip_profile.py` consumes the seeds. Complete
B32 correctness remains in final/natural artifacts; the narrower profile reports
its exact scope rather than claiming all 32 prefills were measured. The broad
32K-source-location failures are historical, preserved under
`failures/profiler_b32_no_cache/` and `failures/profiler_b32_cached/`.

Both narrow fixtures subsequently passed on the device and closed the mesh:
`perf_b32/tracy/layer_{0,3}/correctness.json`, their `.provenance.json` sidecars and
`correctness.pt` record the checks. Prefill is **one natural 32-token prompt**;
B32 decode uses 32 saved post-prefill states from the fully accepted
`seed_l{0,3}_b32.{json,pt}`. The reduced fixture resolves the source-location limit.

## CPU profiler report conversion memory repair

Historical failed attempt: after the successful linear device run, native pandas
conversion exited 137 while importing approximately 35 GiB of host timing rows;
the legacy device CSV is approximately 44 GiB and observed process memory reached
approximately 249 GiB. The process was already gone when inspected. No agent
killed it and no device reset was required. Evidence is retained at
`perf_b32/tracy/layer_0/capture.log` and `raw/.logs/` beneath the same directory.
This failure concerns CPU report construction, not the completed device checks.

The scoped repair is `tests/postprocess_multichip_profile.py`: export `TT_` host
roots from the saved trace, filter native-relevant host rows and deduplicate
device sub-device metadata with conflict rejection and a native lookup
equivalence check. Optional CompileProgram and HWCommandQueue_write_buffer
host child timings are excluded explicitly; device timing and op metadata are
unchanged. Write those inputs under
`perf_b32/tracy/layer_{0,3}/postprocess/.logs/`, preserving the
original raw logs. The authoritative device timings remain the original
`raw/.logs/cpp_device_perf_report.csv`; native `process_ops` report conversion is
unchanged. No decoder, kernel, precision or device execution changes are involved.

Validation status: native CPU CSV conversion passed for both layers. Each
`perf_b32/tracy/layer_{0,3}/` directory contains `ops.csv`, `ops_source.txt`,
`postprocess.log`, `host_times_compaction.json` and
`device_metadata_compaction.json`. The host exports retain 44,876 linear and
19,596 full-attention root rows in native order. Scanning 306,761,184 and
51,894,480 legacy device rows found no sub-device metadata in either capture;
the native lookup equivalence checks passed. Authoritative C++ timing is intact.

The first streaming attempt is also historical: its row-width assertion caught
malformed unfiltered native host CSV after more than 300 million rows. `Program_`
source-file fields contain unescaped commas/newlines. Preserve the failed
processing directory and log at
`perf_b32/tracy/layer_0/postprocess_malformed_host/` and
`perf_b32/tracy/layer_0/postprocess_malformed_host.log`; preserve the raw export.

The current repair invokes the native exporter with `-u -c -f TT_` on the saved
trace, producing `perf_b32/tracy/layer_0/raw/.logs/tracy_ops_root_times.csv`.
It then filters native-relevant rows. `host_times_compaction.json` records the
export filter and omitted optional host-child timings. Original
`raw/.logs/cpp_device_perf_report.csv` timing and `tracy_ops_data.csv` op metadata
are unchanged. Independent CPU audit confirmed that the omitted nonroot host-child
columns are optional and unused by device timing and `tt-perf-report`, and that
native root order is preserved. The tooling repair is resolved; human-readable
tables and final profiler summaries are complete, and all 26 acceptance gates pass.

`tests/collect_multichip_perf.sh` now rejects B32 with exit 2 and directs callers
to `tests/collect_multichip_profile.sh`, preventing repetition of the known broad
capture failure. CPU shell syntax and rejection-path checks passed. No decoder
change or reset was needed for the CPU conversion repair.

Final CPU audit also corrected the summary classifier's `pad` substring match
against `SdpaDecodeDeviceOperation`. SDPA is compute; kernel durations did not
change. Both batch summaries were regenerated and independently reconstructed
by the passing gate checker. MoE attribution validates the exact B32 output-split
tail (one Slice plus 31 untilize/slice/tilize triplets) outside its numerator.

## Review allocation-tracker control

Round 1 found that clean tracked decoder runs lacked the required deliberately
unsafe allocation rejection control. Added a bounded model-local 1x4 trace test
and required acceptance gate; the decoder source is unchanged. The first launch
stalled in mesh open with `TT_METAL_WATCHER=0`, which enables zero-interval
watcher polling (confirmed in runtime option parsing). Live stack/triage and
exit124 evidence are preserved under `failures/trace_control_watcher_zero/`.
Reset12 follows that timeout; retry unsets watcher. The runner now rejects the
incorrect watcher setting before TTNN import.

Resolution: reset12 and four-replica trivial-op smoke passed. Corrected positive
control passed with the expected buffer-specific RuntimeError and exact recovery
output on all four devices. The new gate requires rejection and recovery evidence;
no previous decoder measurement needed repeating. See
`trace_alloc_tracking/positive_control.{log,json}` and its provenance sidecar.

Reset and environment correction occurred together; these observations do not
isolate zero-interval polling as the sole cause of the original stall. Round-2
independent review accepts the corrected control and returns `clean-pass`.
