# Qwen3.6-27B final benchmark stage

Started 2026-11-17 from local tt-metal checkpoint c9ccabfb. This stage reports measurements to the bringup owner without an accuracy verdict. The original context contract remains unchanged (262144 tokens). Stages 9/10 used all 64 layers on P300x2 (four Blackhole chips, TP4, DP1), selected precision gdn_attn8_head8_mlp4_lofi, async scheduling, scheduler budget 512, trace mode all, seeded sampling policy request, prefix caching disabled because recurrent GDN state cannot be recovered from attention pages.

The enabled tt-model-bringup 0.1.24 and tt-autodebug 0.1.8 packages were verified in the active Codex configuration. Their runtime is loaded through the existing model environment. Installed workflow packages are read-only. The final benchmark runner has separate 7200-second client and 7200-second setup bounds for configured batches 1, 8, 16, 32. Server launches, compile and warmup are counted as setup. Pre-invocation installation, dataset preparation, initial server setup and speed probes are recorded separately below.

Initial device check: no fuser holders; bounded `timeout 60 tt-smi -ls --local` completed and listed chips 0–3 as Blackhole p300c. Stage 10 had already stopped its server. No reset was necessary. All server/device work is serialized on the existing bringup hardware; independent agents only prepare client/protocol and host-only phase collection code. No unrelated processes are stopped and no reservation is acquired or released by this stage.

## Commands and implementation

`server_control.py` launches the existing Stage 10 launcher with explicit checkpoint/tokenizer revision 8f33b98937c6031bb49e713dee3e8e185aafd26f, uses bounded server lifetime, records actual process argv and runtime registration/precision, and exercises completion boundaries before collector readiness. Each configuration snapshot retains the effective startup configuration, process ownership and hardware list. Shutdown uses the prior drain-checked stop helper and verifies process birth identity.

Phase accounting is prepared before accuracy. It uses opt-in host markers at existing readback completion boundaries, without new device synchronization or live profiler collection. Matching raw events, shapes and accounting are retained under run/. Source changes are committed and repository hooks run before device execution. Exact commits, client provisioning, speed probe results, frozen sample selection and attempts are recorded as they complete.

## Setup receipts

Client provisioning took 167.749 seconds (separate lm-eval 0.4.13/Transformers<5 and vLLM 0.26 CPU client environments). Exact client sources and package inventories are in setup/. GPQA is unavailable through the official gated dataset with the existing credentials, so MMLU-Pro and IFEval were selected before score observation. The native thinking recipe retains temperature 1, top_p .95, top_k 20, 32768 output cap, upstream 5-shot/0-shot tasks, structured chat and EOS. The harness sends seed 1234. See setup/PROTOCOL.md.

Initial event-query import failed because the build target did not deploy the copied extension. AutoDebug independently diagnosed the mismatch; AutoFix used CMake component tt_pybinds. A fresh import then passed and a bounded 1x4 mesh add/query/readback smoke passed (one nonblocking poll, about 0.56 ms to observed completion); devices closed and fuser returned no holders. Core checkpoint 21fddded; no reset or profiler used. Original failed evidence is retained.

Initial server setup: 314.251 seconds from hook invocation to health readiness (2026-11-17 03:00:09–03:05:23 UTC), plus the recorded 128/4 phase preflight. Actual imported adapter is the generated autoport path, all 64 layers, mesh IDs 0–3, observed compute grid 11×10 per chip, context 262144, 32 server slots. Source checkpoint 3e7473e6. A configuration-recorder spelling error was fixed without changing the running model; this setup-only server uses the inherited underscore CLI flags. The timed run will launch a new server with canonical flags. See AUTOFIX.md. Collector `check` passed before any accuracy request.

Speed probe: 18 frozen questions (MMLU-Pro 14, IFEval 4) completed in 347.057 seconds, all normal stops, 100% extracted answers, zero blanks at cap. Native tokenization verification passed all 18. Final samples are MMLU-Pro 196/12032 and IFEval 160/541, nested in the common 280/256 subsets with identical populations and few-shot hashes. The conservative forecast is 3861.01 seconds plus 1800 seconds reserved for performance/reporting; see setup/sample_sizing.json. Generation/thinking/output caps are unchanged.

Exact-token preflight completed 32/32 requests at exactly 4096/128 in 116.736 seconds; phase accounting matched all requests (prefill 69.478 s, decode 47.247 s). Original lossy-text preflight and all failed integration evidence are retained. Timed attempt 1 began at 2026-11-17 03:23:25 UTC using source checkpoint eeb42050d6563f711e136b918ee351fd614e04bc; no second timed attempt has been started. Final timings and completion receipts follow below.


## Completed timed attempt and handoff

Attempt 1 completed with both upstream tasks (196 MMLU-Pro, 160 IFEval) and all six performance rows. There were 356/356 normal accuracy stops, no token-limit or empty-final responses, and 100% extraction/nonempty finals. No accuracy verdict is applied. All 182/182 measured performance requests completed. Batch 1/8/16/32 output rates were 29.83/63.79/53.48/37.78 tokens/s; full-phase prefill and decode estimates are present for every batch. No configured batch was skipped.

Runner client time: 6299.825571 of 7200 seconds. Runner setup: 1533.025997 of 7200 seconds. Total runner wall time: 7832.851568 seconds. Pre-invocation setup was separately 2488.525134 seconds. Post-run validation/reporting is included in run/post_run_validation.json; the original runner summary is unchanged. Only one timed attempt was used.

The evidence checker exited 0. Final native template verification passed all 356 inputs, including exact token IDs and original prompt usage. Independent protocol/performance audits passed. The final context checker exited 0 with full target/supported context 262144, zero critical findings and historical advisory matches involving old process-return-code text. The original context-contract SHA is unchanged.

Every timed server used tt-metal eeb42050d6563f711e136b918ee351fd614e04bc. Earlier stage commits: 21fddded, 367a4c4c, 3e7473e6, 305a4735. Source revision evidence and the installed-wheel provenance caveat are in REPORT.md and server_package_provenance.json. Final local report/evidence commits are logged in final_commits.log. Nothing is pushed.

The final server was drained and stopped by the owned-process controller. API and engine PIDs exited, fuser found no device holders, no reset was used and reservation ownership was unchanged. A closeout assertion initially expected the ownership file to be deleted; inspection showed the controller deliberately preserves its stopped_unix receipt. The corrected check validates that timestamp and actual process exit; both original diagnostic and corrected receipt are retained under run/cleanup*.json. This did not affect execution or measurements.

The artifact manifest inventories evidence without altering raw files. Compact source, report, configuration, scores and compatible evidence files are committed. Files above the repository's 500-KB limit and immutable raw files incompatible with formatting hooks remain in this persistent checkout, including all raw responses and phase logs. No individual artifacts are compressed or archived. Unrelated prior-stage files are untouched.
