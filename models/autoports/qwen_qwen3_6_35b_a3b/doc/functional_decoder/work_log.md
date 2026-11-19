# Functional decoder work log

## Contract and provenance

- Scope: functional decoder only; no later stages. Status: in progress; no acceptance gates claimed.
- HF: `Qwen/Qwen3.6-35B-A3B@345d467b117d1fd5eda9b879a5988de9ae5f47a7`.
- Initial checkout: `822d04806f292e2cf7e92de7f7be79fa8d83e5aa` (clean).
- Enabled skills: tt-model-bringup 0.1.23, tt-autodebug 0.1.8 (Codex plugin inventory and config).
- Environment: source `../lane-env.sh` after activating `python_env`; package roots resolved from enabled inventory. Persistent checkout, weights, Codex home and multigoal logs under `/home/ttuser/qwen36-35b-a3b`.
- Hardware: Blackhole P300c, physical chip 0, 1x1 mesh, serial jobs. `fuser /dev/tenstorrent/[0-9]*` found no users at startup. `timeout 60 tt-smi -ls --local` listed chips 0–3. No reset or process termination performed.
- AutoConfig maps this checkpoint to `Qwen3_5MoeTextConfig`: H=2048, 40 layers, three linear_attention followed by one full_attention; 256 experts, top 8, intermediate=512 plus shared expert=512. Full attention has Q=16, KV=2, D=256, output gate, 64 rotary dimensions. Linear attention has K heads=16, V heads=32, D=128, depthwise convolution width 4, FP32 recurrence. All RMS norms except the delta output norm use `1 + weight`.
- Advertised context is 262144. No reduced context is authorized or claimed.
- Reference code: installed `transformers/models/qwen3_5_moe/{configuration,modeling}_qwen3_5_moe.py`; direct layer-only state loading from indexed safetensor shards. The checkpoint was cached before this stage.

## Initial environment checks

- `timeout 90 env TT_VISIBLE_DEVICES=0 python ...` failed before device open: P300 topology treated the single visible die as CUSTOM and required a custom descriptor. Evidence: `logs/mesh_smoke.log`.
- Retried with all chips discoverable and `open_mesh_device(MeshShape(1,1), physical_device_ids=[0], trace_region_size=0)`. Arithmetic add and close pass: `logs/mesh_smoke_retry.log`. This uses only chip 0 for model execution. No reset needed.
- `pre-commit run --files <three initial Python files>`: exit 0, `logs/hooks_initial.log`.

## Runs

All device commands use bounded `timeout` through the installed `scripts/wait_for`. Check individual logs for current results; entries here are not substitute evidence.

- Initial real layer 0 prefill: `timeout 900 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.smoke`; `logs/linear_smoke_01.log`.

## Pending acceptance work

Both layer kinds, paged/continuation and non-aligned boundaries, advertised context, batch coverage, traced refreshed decode, determinism, watcher, fallback audit, warmed profiler reports, independent stage review, local stage commits. No stage completion or performance claim yet.

## Correctness evidence through the first long-context run

- `linear_smoke_01.log`: corrected environment API name to `init_device_compute_kernel_config`.
- `linear_smoke_02.log`: tiled FLOAT32 scatter unsupported. Kept router logits/top-k in FP32 and cast the selected weights to BF16 for scatter. `linear_smoke_03.log` passes at 0.999928796.
- `linear_boundary_tracking_01.log`: the longer chunk kernel requires rank-4 constant tiles. Fixed setup constant shapes to `[1,1,32,32]` / masks `[1,1,32,96]`. `linear_boundary_tracking_02.log` passes.
- `run_decoder --layer {0,3} --length 32`: real-weight prefill and traced decode pass. `linear_initial.json`, `full_initial.json`, logs `linear_trace_01.log`, `full_trace_01.log`.
- `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 ... run_decoder --layer {0,3} --length 257`: pass, `linear_boundary.json`, `full_boundary.json`. Tracking includes program-cache allocations. Three replay outputs are bit-equal.
- `timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.battery --layer {0,3} --output ...`: real-weight lengths 1,31,32,33,255,256,257,513,7 on the same instance; continuation at 32,33,256,512,528; batch 32 prefill/decode. Both pass in `linear_battery.json` / `full_battery.json`, logs `*_battery_01.log`. The latter batch initially used independent page pools; shared-pool and guard-page verification is an additional forthcoming run.
- Resume comparison uses every row's layer contribution (output minus residual input), against full HF prefill. Dropped-history negative controls fail the same 0.995 gate for both layer kinds. No weakened threshold.
- Runtime audit (`tests/runtime_audit.py`) rejects Torch dispatch, `from_torch`, and `to_torch` inside all battery prefill/decode paths. Guard passes. Setup weight conversion, reference and final readback are explicit test boundaries.
- `timeout 14400 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.long_context --layer 0 --output .../linear_long.json`: pass; `logs/linear_long_01.log`. Real HF layer oracle is streamed in 1024-token chunks with its DynamicCache. TTNN uses 256-token chunks with exact one-token tails. Prefill 262143/262144 PCC 0.999876320/0.999876326; traced decode position 262143 PCC 0.999923123, three bit-equal replays. These are correctness host-wall observations, not performance claims.
- API validation was tightened after the linear long run launched (explicit target config checks; required prepared plan). Tensor arithmetic was unchanged. Shared-pool allocation support affects only full-attention setup.
- Installed `pip==26.3` and `tt-perf-report==1.5.0` into the active source-build environment with `uv pip install --python python_env/bin/python tt-perf-report pip`; installer resolved matplotlib 3.10.9. `logs/install_perf.log`. The active environment initially lacked pip.
- Exact per-tensor checkpoint shape/dtype/mean/std are in `weight_stats.json`; software versions in `versions.json`. Synthetic tests use the unmodified target config copied to `hf/config.json` and these statistics.

## Anomaly ledger

- P300 single-visible-chip CUSTOM topology: setup-only failure; resolved by retaining board discovery and selecting physical chip 0 in the 1x1 mesh. No unhealthy-device result or reset.
- `Unknown motherboard B850M-C ... falling back to bus_id as tray_id`: topology naming warning in source `physical_system_discovery.cpp:159`; arithmetic smoke and both layer kinds pass. No remote chip computation is used.
- `Opening subset of mmio devices slows down UMD read/write to remote chips`: informational warning; measured model uses one local chip, no remote transfers.
- HF missing FLA/causal-conv fast path: the canonical CPU reference uses the installed Transformers Torch implementation. This is outside the TTNN forward and is not a TTNN runtime fallback.
- HF standalone ExpertsInterface warning: the layer-only oracle uses HF's eager expert implementation when `_experts_implementation` is unset. No substituted MoE math.

## Full-attention long run and AutoFix recovery boundary

- Exact command: `timeout 14400 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.long_context --layer 3 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/full_long.json`. Log: `logs/full_long_01.log`. HF completed in 751.83 s; no accepted TT long result was produced.
- Initial diagnostic: `timeout 180 tools/tt-triage.py --dev=0 --llm-output --llm-output-path models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/triage/full_long_snapshot.txt --triage-summary-path models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/triage/full_long_summary.txt`; exit 0. Detailed warnings remain in raw output. Crucially, this checkout intentionally leaves BH cores halted; script completion does not imply device usability.
- `py-spy dump --pid 2087868 --locals` twice and `--native` produced `triage/host_locals_{01,02}.txt`, `host_native.txt`. All were after the first triage capture. Earlier statements that these prove a pre-existing stall are retracted.
- Source-only fresh launcher: from this doc directory, `timeout 1200 env AUTODEBUG_SKIP_CHILD_SANDBOX=1 "$TT_AUTODEBUG_ROOT/skills/autodebug/scripts/autodebug.sh" --task autotriage --focus <repo>/tt_metal/impl/dispatch --focus <model-dir> -- 'Investigate preserved dispatch stall in AUTOTRIAGE_INPUT.md. Source/evidence only; no devices or code edits.'`; exit 0, `AUTOTRIAGE.md`, `logs/autotriage_launcher_retry.log`. Initial sandbox preflight rejected unavailable unprivileged namespaces (`logs/autotriage_launcher.log`). No automatic approval review rejected an action.
- GDB batch read-only attachments captured stack/disassembly/registers and host queue values; inferior was detached every time. Exact frame 4 expressions: `p/x *(unsigned char*)$r14`, `p/x *(unsigned int*)$r13`, `p/x *(unsigned int*)$rbx`, `p/x **(unsigned int**)($r15+0xe8)`, `p/x **(unsigned int**)($r15+0x100)`. CQ 0, credit address 0x1b7c0, fence/write pointer 0x1d33c. No inferior function calls or writes.
- Targeted triage commands: `timeout 120 tools/tt-triage.py --additional-scripts-directory=models/autoports/qwen_qwen3_6_35b_a3b/tests/triage_scripts --run=qwen_fetchqueue` and `--run=qwen_prefetchcursor`; logs `fetchqueue_capture_retry.log`, `prefetch_cursor_capture_retry.log`, both exit 0. Initial helper failures (missing Owner metadata; wrong coordinate constructor) happened before hardware reads and are retained. The successful cursor read found the core already halted. The script now explicitly rejects a running core. The ring JSON's original `translated_core` label means the input NOC0 string `16-2`; helper metadata is corrected to `noc0_core` for future captures.
- AutoFix agents independently audited dispatch and decoder address bounds: `autofix_dispatch_audit.md`, `autofix_decoder_audit.md`. No decoder bug or safe arithmetic repair is established. Corrected conclusion and exact recovery boundary: `AUTOFIX.md`.
- No reset, process termination, lock clearing, unsafe continue bypass or later-stage work. PID 2087868 remains under its original timeout; STOP_QUEUE holds later jobs. Requested user authorization to terminate this run and reset the affected chip, then prove recovery with list and arithmetic mesh smoke. Authorization pending.
- Prepared optional HF oracle caching in `tests/long_context.py`, with config/source/software/exact-weight hashes. Prepared profiler runtime guards. Host pre-commit checks pass; these additions have no post-recovery device evidence yet.
- Outstanding: full-attention advertised context; refreshed one-trace request coverage and watcher; mixed traced batch 32/shared-pool guards; synthetic pytest; partition control; warmed perf CSV and human tables; final source-provenance/host gates; stage review. No performance claim or stage pass.

## Local checkpoints

The implementation and evidence are being saved as an incomplete functional-decoder checkpoint. Host AST parsing and final pre-commit checks pass (`logs/hooks_checkpoint_final.log`). The following documentation commit records this checkpoint SHA. Nothing is pushed.

- `4261dec50699bf1141fd635d669c271991118cf5`: functional-decoder implementation, partial passing evidence, and corrected AutoFix investigation. Incomplete checkpoint; no push.

## Host-only progress while recovery authorization is pending

- Confirmed PID 2087868 and its original timeout remain live. No device recovery or additional device validation was attempted.
- Moved `hf_forward` unchanged into `tests/reference.py`; AST comparison against 3100c34f passes (`oracle_refactor_check.json`). This lets `tests/long_context.py --oracle-only` avoid importing TTNN entirely.
- Command: `timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.long_context --layer 3 --oracle-only --oracle-cache models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/oracle_cache/full_262144.pt --output models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/full_oracle.json`. Completed exit 0 in 951 s, with 939.898 s for the canonical HF calculation. All 262144 output rows finite; no TTNN import. `logs/full_oracle_only.log`, `full_oracle.json`, provenance sidecar. This is not a TTNN pass.
- Same command with timeout 120 and output `full_oracle_reload.json` loaded and verified the existing cache, exit 0 in 11 s; `logs/full_oracle_reload.log`. Both identity records agree. The 1,073,747,845-byte cache is ignored by Git; checksum: `full_oracle_cache_manifest.json`. Compact evidence is tracked. The initial generic provenance device label was transparently corrected to host-only, preserving original source hashes and timestamps; the recorder now distinguishes this mode.
- Added `tests/summarize_perf.py` and connected it to `collect_perf.sh`. Inspected installed tt-perf-report 1.5 source: CSV emits per-op raw microseconds under `Device Time [μs]`, no terminal total row. Summarizer requires actual reports and passing correctness/provenance, reports device time and gaps separately, and rejects overlapping operations rather than silently overcounting. No performance numbers have been produced.
- Added both mixed batch 32 and shared-pool traced runs to the serial validation queue with allocation tracking including program cache. STOP_QUEUE remains active. Shell syntax and Python AST checks pass. Final device gates and stage review still require recovery.

- `4d8a5f6750347504e21753b93b67aeac62dac26e`: full-context host-only oracle/cache evidence and remaining validation runners; recovery authorization still pending. No push.

## Authorized recovery and resumed validation

- Owner authorized terminating PID 2087868 and `tt-smi -r`. Verified exact long-context command and sole ownership of all four device handles, then sent SIGKILL. No unrelated process was terminated.
- Bounded `timeout 60 tt-smi -ls --local`, `timeout 180 tt-smi -r`, `timeout 60 tt-smi -ls --local`: all exit 0; chips 0..3 present. No second reset or lock clearing.
- `timeout 90 python` opened 1x1 physical chip 0, uploaded BF16 ones [32,32], added 2, asserted exact readback of 3, closed mesh; exit 0, `RECOVERY_ARITHMETIC_MESH_SMOKE_PASS`. Exact evidence: `recovery.json`, `logs/recovery_{list_before,reset,list_after,smoke}.log`. This is infrastructure recovery only.
- Resumed `timeout 14400 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.long_context --layer 3 --oracle-cache models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/oracle_cache/full_262144.pt --output models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/full_long.json`; `logs/full_long_recovered.log`. All acceptance gates unchanged.

## Long-context tail accuracy investigation

- Recovered full-attention run exited 1 at its per-row gate: prefill262143 aggregate PCC0.999269557, aligned row262110 PCC0.999531610, final single-token tail row262142 PCC0.945939623. Device closed cleanly. Preserved `logs/full_long_recovered.log`; this is a real correctness failure, not a device-health failure.
- Invoked installed AutoDebug launcher, source-only focus model and native SDPA; `logs/autodebug_long_tail.log`. AutoFix address hypothesis audit: `long_tail_address_audit.md`; uint32 indexing fits through the advertised context.
- Native decode fixes im_df/stats_df to BF16 even with FP32 destination accumulation. One core with k_chunk32 performs8192 serial rounded updates. `tests.sdpa_precision_probe` isolates this: original PCC0.77007/RMS ratio48.81; grid8x8 k32 PCC0.99784; grid8x8 k256 PCC0.99972. Exact `sdpa_precision_probe.json`/log.
- `tests.real_tail_probe` prepares full-context K/V directly from the canonical HF projections and uses real decoder weights: original last-token PCC0.84291, grid8x8 k32 PCC0.9998276 (contribution PCC0.9998272), grid8x8 k256 PCC0.9999453. This is a focused geometry comparison, not a substitute for end-to-end TT cache validation.
- Minimal repair changes only full-attention decode grid1x1 to8x8, retaining k_chunk32 so every page-table extent stays legal, including a single page. The sole request Q owner remains core(0,0); native workers distribute context. No thresholds/config/context reductions or native changes.
- Focused tracked real-weight `run_decoder --layer 3 --length 257 --output .../full_boundary_fixed.json` exits0: prefill PCC0.999902827, eager/traced decode0.999934527, three bit-equal replays with program-cache allocations included. `logs/full_boundary_fixed.log`.
- Re-running advertised context with unchanged cached HF oracle and repaired decoder: same long_context command, `logs/full_long_fixed.log`. All affected full-attention gates will be refreshed; earlier linear-attention arithmetic is unchanged.

- Installed AutoDebug completed exit0; report `long_tail_debug/AUTODEBUG.md`. Parent checked its BF16 storage claim against factory lines507-508 and the controlled device probes. The report describes the original grid1 run before the edit; later passing results supersede its runtime-unverified status. Original failed-run sidecar was overwritten by retry; `long_tail_failed_source.json` transparently records the historical source, while `full_long_grid_only.json.provenance.json` preserves the passing retry.
- Repaired full-context command exits0 in268s: prefill262143/262144 PCC0.999281097/0.999280941; final-position traced decode0.999833394, deterministic. Preserved first repair result in `full_long_grid_only.json`. Final64-row regression will strengthen coverage of the affected boundary.

- First watcher trace-reuse run passed all9 PCCs, repeated-input checks and deliberate allocation-hazard rejection, then failed JSON serialization of MeshTraceId. Changed only recorded trace_id to str(trace). Preserved `logs/trace_reuse_watcher_0_json_failure.log`, `logs/finish_queue_json_failure.log`, `watcher/linear_json_failure/`. Restarted complete queue for successful artifact output.

## Completed regression queue

- `bash models/autoports/qwen_qwen3_6_35b_a3b/tests/finish_validation.sh`: exits0 after reporting-only repair. `logs/finish_queue_recovered.log` records each bounded job. Both watcher trace-reuse commands use TT_METAL_WATCHER=10 with allocation tracking and program-cache allocations included, separately from profiling. `trace_reuse_{0,3}.json`: one capture, nine changed/repeated requests each, allocation hazard rejected, exact output acknowledgments only. `tests.audit_watcher` reports clean for both; `watcher_audit.json` has log hashes/dump counts.
- `full_resume_fixed.json`: all starts32/33/256/512/528 pass per-row contribution PCC (minimum0.999774843); corresponding dropped-history controls fail0.995.
- `linear_mixed_batch.json`, `full_shared_pool.json`: mixed starts0/1/32/33 and lengths1/31/32/33, 32 rows, full-batch trace with three bit-equal replays. Minimum recorded PCC0.999316625/0.999723984. Full attention also verifies64 logical K/V views and untouched unowned guard pages.
- `synthetic_pytest.xml`: 2 passed, no skips/failures/errors, target config shapes and recorded weight statistics. `logs/synthetic_pytest.log`.
- `partition_{0,3}.json`: same-instance boundary sweep and chunk32 versus unchunked256 control pass; logs `partition_{0,3}.log`.
- Installed `scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_6_35b_a3b --hf-model Qwen/Qwen3.6-35B-A3B --stage functional-decoder --require-contract --strict-caps`: exit0, full262144 context. `logs/context_contract_check.log`.
- Next sequential commands: `timeout 14400 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.long_tail_regression`, then `bash models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_perf.sh` only if that regression exits0. The wrapper preserves the original HF oracle identity and expands host readback checks to all64 trailing rows of both long lengths.

- Expanded full-context row check exits1: two chunked-prefill rows262105/262106 have PCC0.994884112/0.992146302; all31 single-token tails pass. Exact values: `long_tail_expanded_failure.json`, `logs/long_tail_regression.log`. The command chain correctly did not start profiling. Context acceptance is unset pending this stronger correctness gate.
- AutoDebug already identified BF16 running output buffers in chunked prefill (with FP32 denominator). Extended `tests.real_tail_probe` compares the same real-weight final256 input rows and directly prepared HF KV at key chunks32/256/512 to isolate accumulation depth. `logs/real_prefill_tail_probe.log`. No decoder change is accepted from prose alone.

## Full-attention accumulation repair and capacity padding

- Fixed-cache real-weight prefill comparison over final256 rows: K32 aggregate0.998515847, minimum row0.990282824; K256 aggregate0.999712409, minimum row0.998580746. `real_prefill_tail_probe.json`, `logs/real_prefill_tail_probe_fixed.log`. A K512 candidate failed native validation because start261888 is not divisible by512 (`logs/real_prefill_tail_probe.log`); device closed cleanly. K256 restores the required accuracy without changing the attention domain.
- Final model-local repair uses grid8x8 and K256 for both full-attention paths. Full prefill starts physical chunks only at256-aligned absolute positions; unaligned starts/tails use exact single-token decode until alignment. Public logical lengths stay arbitrary. Linear attention is unchanged.
- Page tables pad to an8-entry multiple by aliasing the last owned finite page; padding is causal read-only and not logical capacity. The default physical pool rounds up to that table length; explicit shared pools must be at least that length because native paged_update_cache validates table width <= physical page count. Extra default allocation is at most7 pages =458752 bytes for both K/V. No context reduction.
- First capacity probe exposed that native table-width validation when one physical page accompanied eight alias entries; fixed default physical allocation and explicit pool validation, then all capacities1/31/32/33/255/256/257/511/512/513 passed (`logs/page_extent_regression_fixed.log`). Added checks that extra unowned physical pages and future rows stay zero.
- Final serial chain: page_extent_regression, long_tail_regression, FUNCTIONAL_FULL_ONLY=1 finish_validation.sh, collect_perf.sh. Raw logs: `page_extent_guards.log`, `long_tail_regression_fixed.log`, kind-specific queue logs, and `profile_{0,3}.log`. Prior full watcher evidence is preserved in `watcher/full_grid_only/` and `trace_reuse_3_grid_only.json`.

- Source-only padding audit completed: `padding_contract_audit.md` finds no defect within the documented finite-cache, owned-page and valid-position contract; 87368 full-attention planning cases pass. Added its focused evidence recommendations to `tests.padded_trace_regression`: capacity33, one live tracked trace, two disjoint page assignments with all alias entries refreshed, position32, zero versus64-valued future rows, unowned-page sentinels, and repeated-request determinism. It is queued after boundary tests and before profiling.
- `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_source`: exit0; static runtime audit clean and28355 linear planning cases identical to checkpointa407e7d3. `source_audit.json` records changed functions and current decoder hash. Linear arithmetic/setup helpers are unchanged; prior linear full-context evidence remains applicable.

- Final `long_tail_regression` exits0 in1095s (`logs/long_tail_regression_fixed.log`). Prefill262143/262144 aggregate PCC0.9998874579/0.9998874323; all64 trailing rows pass, minimum0.9989460219/0.9986499443. Traced decode position262143 PCC0.9999472747, three bit-identical replays. `full_long.json` and its current-source provenance are final end-to-end evidence. Restored supported context262144 in `doc/context_contract.json`; no capability reduction. First pass spent953.997s including compilation/readback, second123.342s; these correctness host-wall observations are not performance measurements.

## Final affected-path queue

- `FUNCTIONAL_FULL_ONLY=1 bash models/autoports/qwen_qwen3_6_35b_a3b/tests/finish_validation.sh`: all jobs exit0 on the final K256 implementation. Watcher trace-reuse149s, resume17s, shared-pool batch32:63s, synthetic pytest42s, partition26s, padded trace7s. Per-job logs retain earlier attempts followed by the final run; JSON/provenance identifies the final source.
- Full watcher trace reuse: nine requests, minimum PCC0.999866143; one capture, changed positions/pages/inputs, repeated first request bit-equal, deliberate live-allocation hazard rejected. `tests.audit_watcher` now reports both kinds clean against current raw watcher hashes.
- Full resumed contribution minimum PCC0.999403081; every dropped-history control fails. Full batch32/shared-pool checks, all boundary lengths, and chunk-partition parity pass. `synthetic_pytest.xml`:2 passed, zero skipped/failures/errors;38.44s test time.
- `padded_trace_regression.json`: six runs, capacity33, one tracked capture, disjoint page orders[2,5]/[1,6], position32. Nonzero future-row sentinels produce exactly the zero-tail output, other-owner/unused pages remain unchanged, changed requests differ and repeated requests are bit-equal. Minimum PCC0.999905103. `logs/padded_trace_regression.log` exits0.
- Non-failing environment messages are classified: UMD's unknown B850M-C label falls back to PCIe bus/tray metadata; subset-of-MMIO warning describes open overhead in this required1x1 stage; Transformers' ExpertsInterface message explicitly permits standalone expert modules; optional FLA absence affects only the CPU HF oracle; pytest's two warnings concern SWIG type metadata deprecation. None reports device corruption or fallback inside the TTNN runtime. Watcher/error audit is clean.

## Profiler completeness repair

- Initial profiler commands completed, but raw audit found `Profiler DRAM buffers were full, markers were dropped!`. Linear decode's16 replay sessions contained106,106,106,95,...,39 operation rows instead of equal counts. Initial summary was rejected and moved to `tracy/overflow_summary.json` with `valid:false`; raw reports/logs are preserved under `tracy/overflow_layer_{0,3}/`, `logs/profile_{0,3}_overflow.log`. No incomplete timing is accepted.
- Changed only profiler orchestration: drain before measured windows, after prefill, and between each complete blocking trace replay. All16 forwards still execute the same captured device graph; profiler reads occur strictly between passes. Device-duration sums exclude these host measurement gaps. Added raw replay-session operation-count/content checks to the summarizer. Source hooks pass (`logs/hooks_profiler_fix_verify.log`). Re-running `bash tests/collect_perf.sh` in the model directory's module context; exact checkout-relative command is the earlier `bash models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_perf.sh`.

## Final performance and gate results

- Repaired `collect_perf.sh` exits0: profiler runs21s/15s, all real-weight PCC/determinism guards pass. Final console logs `logs/profile_{0,3}.log` contain no dropped-marker warnings. `profiler_completeness.json` verifies20 identical replay sequences per kind, including16 measured replays:144 operations each for linear attention,99 for full attention. The actual decoder is unchanged by profiler repair.
- Warmed256-token prefill device duration: linear69685.930us, full68881.854us. Traced warmed decode device duration/pass: linear60378.858us, full60137.201us, each averaged over16 complete replays at position256. Units are the filtered CSV's `Device Time [μs]`; sums exclude dispatch gaps and between-pass profiler drains. Human tables, CSVs, source CSV and provenance are under `tracy/layer_{0,3}/`; summary `functional_decoder.md`, `perf_summary.json`. These are device timings, not host latency or full-model throughput.
- `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.check_evidence`: `FUNCTIONAL_DECODER_HARD_GATES_PASS`. Both kinds retain context262144, no reduction; default0.995 acceptance unchanged. Runtime static/dynamic audit and watcher audit pass. Final profile loader's pandas mixed-column warnings are parsing notices; row completeness is independently checked. Initial AICLK1331MHz was within the firmware's requested1350MHz tolerance and belongs to the rejected overflow profile; no clock normalization or speedup claim is made.
- Tracy automatically started host-only GUI PID2684923. After both measurements completed, verified its exact serve_wasm.py command and stopped that server with SIGTERM; no device job was terminated.

## Pre-review self-check

- All model-directory pre-commit hooks pass: `logs/hooks_stage_final_verify.log`. The first pass normalized only trailing spaces in generated human tables and the final newline in JUnit XML; numerical CSV evidence is unchanged.
- Re-ran host source audit, profiler completeness/summarizer and functional hard gates after formatting; all pass. Context checker already passes with full262144 capability. No device writer remains active.
- `tests.archive_evidence` retains exact closed text bytes with deterministic gzip and SHA256 in `archive_manifest.json`. The larger linear raw CSV is split into ordered450000-byte compressed pieces to satisfy the repository's500KB limit; concatenate pieces before gzip decompression. All archived text hashes verify. The old checkpoint log manifest is explicitly historical.
- Only `models/autoports/qwen_qwen3_6_35b_a3b/` is changed. No later-stage implementation or push. Ready for fresh independent stage review; final acceptance and local stage commit wait for its verdict.

## Independent stage review and acceptance

Fresh fork-none xhigh subagent `/root/stage_review_round1` applied the installed stage-review skill after every run/gate completed. Verdict: **clean-pass**, round1, no Required Work. Report: `stage_review_round1.md`. It independently checked the decoder/HF source, provenance, full-context results, meaningful continuation controls, trace lifetimes/allocation positive controls, ownership/padding, watcher logs, all138 archives, and raw profiler sequences/units/totals. Reviewed decoder SHA256: `ff3307ea3a3a4dcc93400fdd2a79a47a32e216d9f1bc47626e875dc7b561f99b`.

The review's handoff notes are recorded without expanding this stage: later optimization can replace the dense expert bank and serial row loop; later generators must own centralized preparation and trace lifetimes; a future unified gate can incorporate the source/raw-evidence checks independently performed here. The original linear long run predates sidecars and is supported by its preserved log and unchanged arithmetic/planning, as the reviewer verified. None is Required Work or a capability waiver.

Functional-decoder requirements are satisfied at context262144 and the default0.995 bar. Only local stage-owned files will be committed on tt-metal branch `main`; no push, optimized decoder, multichip, full model or vLLM work.

## Accepted local checkpoint

- Repository: `/home/ttuser/qwen36-35b-a3b/tt-metal`; branch: `main`.
- Accepted implementation, tests, evidence and clean review commit: `2b0d17f5ddec89e91b5e2176e58e503a0dc15b66`. All commit hooks passed. No push.
- This follow-up documentation commit records that SHA; its own SHA is reported in the final handoff. Earlier checkpoint/documentation commits were4261dec50699bf1141fd635d669c271991118cf5,3100c34f,4d8a5f6750347504e21753b93b67aeac62dac26e,a407e7d3ec6a00693e1be369d8a395f63c9c2c62; all local.
