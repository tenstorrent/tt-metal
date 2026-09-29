# Gemma4 TSU experiment evidence

Status: local selected-default qualification in progress; no new remote result
is implied by these files. Decision log and complete caveats live in
`../../doc/tsu_optimization/work_log.md`.

## Attribution

All local device experiments use the existing immutable image at digest
`sha256:686c60b59bb8dc0da80e869981590522fa9920e4078f321a5ffadaa9088beb21`.
Compiled TT-Metal is919c110d3d4331b7753c1db78618e879905ae46d. Candidate Python
source is bind-mounted, working directory `/tmp`, with
`PYTHONPATH=/workspace/tt-metal:/home/container_app_user/tt-metal`.
The `baseline_sync` server imported unmodified image source. Launch manifests
record source hashes and intended argv; startup logs and `server_info.json`
attest the actual running configuration. Local bind-mounted measurements are
not qualification of a new immutable image.

Every serving configuration retains262144 maximum context,32 scheduler slots,
1GB trace region, FABRIC_1D, selected precision, and device sampling. Primary
client shape is4096 input/128 output/C1/four requests. TSU is1000/meanTPOT_ms,
not aggregate output throughput. Serving has no Tracy/device profiler.

## Evidence map

- `baseline_sync`: exact-image synchronous baseline, native ITLs and identical
  chat benchmark command arrays, raw JSON and summaries.
- `prefill8192_cwd_sync`: rejected long-prefill-trace candidate; full8K exceeds
  trace capacity. `prefill8192_sync` records an earlier import-precedence error
  and is not candidate performance evidence.
- `eager_sync`, `eager_async`: same decode-retention source and matched serving
  harnesses; scheduler comparison, short/long-output/32K/C32 guards. Sync
  contains the official six-prompt qualitative replay and seeded controls.
- `direct_paths.json`: unprofiled full30-layer generator, per-token readback and
  queued model/sampler paths. Synthetic inputs differ from serving; do not
  subtract timings as an exact host-overhead estimate.
- `profile_retry`: valid reduced real layers0/5 plus full terminal/sampler;
  complete decode-window CSV compressed losslessly, whole-window summary, scoped tt-perf-report.
  `profile.log.gz` is the invalid interrupted first capture, not timing evidence.
- `profile_short`: selected-source short-context reduced profile for SDPA
  attribution. Never an all-layer or serving profile.
- `head_bfp4_*.json`: isolated real-input head sweeps with exact geometry,
  precision, replay counts, legality failures and PCC/top1 checks.
- `head_full_paths.json`: full-model K4/K2/K1/K4 time-order controls, five steady
  repeats and one excluded warmup per block; exact output tokens/source hashes.
- `eager_watcher.json`: earlier31-row reduced Watcher/allocation-tracker pass.
  `final_watcher.json`: final31-row Watcher pass with model/sampler trace IDs.
  `final_tracking.json`: final11-row Watcher plus allocation-tracker pass.
  These diagnostics are correctness evidence, not performance measurements.
- CPU logs preserve initial fixture failures and passing retries. Do not treat
  all files whose names contain `tests` or `contracts` as passing by filename.

Large completed logs are committed as `.log.gz`; uncompressed originals remain
local. Runtime/JIT caches, huge profiler intermediate reports and Tracy captures
are ignored; the relevant raw CSV and scoped analysis are retained.

The repo limits individual files to500KB. Full profiler captures (including
unneeded warmup/prefill rows) remain local under ignored `reports/` and as
`raw_ops.csv.gz`. Committed `raw_decode_ops.csv.gz` preserves every column and
field inside the two decode signposts. Its manifest records original/full and
window hashes; `summary_window.json` reproduces the original `summary.json`
windows exactly. Recreate a CSV with `gzip -dk raw_decode_ops.csv.gz`.
`tests/filter_tsu_profile.py` performs only this row filtering. Benchmark JSON
received a final newline and human tables had trailing whitespace normalized
by repository hooks; metric fields and raw CSV field values are unchanged.
