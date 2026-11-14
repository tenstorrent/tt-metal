# Stage Review

Verdict: clean-pass

Independent, scoped review of completed stage-12 extended GPQA semantic controls and selected-policy exact 4096-input/128-output B1/C1 vLLM headline. This is **not approval of stage 12 as a whole**. Extended SWE85 is live and excluded. Reviewed live worktree on v5-extended-run, launch/source HEAD d1e4c8a50935c0f084d7becfec656f3351fc0e3c.

## Required Work

None within this scope.

## Other Concerns

- The synthetic headline outputs are not a quality pass. `current/headline-head-bfp8-b1/perf-b1.json` contains visible repetition, including zero-based requests 5 (repeated isolated letter) and 7 (repeated phrase). Its command uses random input, exact-length truncation and `ignore_eos`; this is timing traffic. No control here proves these particular repetitions expected or attributes them to precision. The benchmark-model skill separately specifies long natural reasoning prompts for anomaly screening, and agentic-serving decides anomalies on real evaluation replies. Existing `doc/benchmark/anomaly_check.json` is that separate natural-prompt evidence, not a control for these exact synthetic outputs. The coordinator added an explicit `quality_scope` limitation to the summary during review; it preserves this distinction without changing measurements. The headline cannot support a text-quality claim.
- `precision_selection.json` retains historical “whole-list rerun pending” wording. Current completed rerun evidence supersedes that intermediate status. This is a provenance note, not missing execution.

## Hard-Check Gaps

- Semantic repetition can evade the mechanical detector. All eleven saved HF continuations (eight original seed42 samples, task147 seed43 original, and two extensions) were read directly. Detector=false is not used as a semantic pass.
- One selected-policy headline and its matched baseline do not establish run-to-run spread. Its comparison is appropriately described as a single matched measurement.

## Anomaly Ledger

- Observed anomaly: extended GPQA tasks 30,48,110,127,129,130,139,145,147 have repetitive reasoning or collapse.
  Evidence: `anomalies/gpqa-extended-precision/first-wave-served/`, `first-wave-full/`, and `conditional-hf-prep/` raw JSON text, metadata and adjudications.
  Affected path: selected served precision and full BF16/HiFi4 TT, on saved reconstructed suffix512 prompts; pinned BF16 HF CPU reference uses the identical saved diagnostic input IDs.
  Control or comparison: each case has three served and three full-TT continuations, with per-case semantic adjudication. Representative raw TT continuations for all nine were inspected, including progressing served task147 seed42 and repetitive seed43. Full precision explicitly covers weights, activations, head, CCL, KV and HiFi4; the HF loader is pinned and checks CPU/BF16.
  Likely subsystem: reference-model continuation behavior on these shortened diagnostic inputs; this does not establish native-trajectory causality.
  Investigation performed: directly inspected all HF outputs and runner source. Tasks30/48 repeat unchanged reaction/assertion cycles;110 repeats the same product assignment;127 repeats translations;130 cycles the same integration question;139 repeats reagent assumptions;145 collapses into a repeated phrase. Task129 original512 is preserved as inconclusive, while its1024 extension repeats complete unchanged wavelength calculations. Task147 original512 preserves initial candidate progress and remains inconclusive, while its1024 extension repeats the unchanged clue checklist. Independently compared case hashes and generated IDs: both extensions have exactly the original first512 IDs.
  Resolution: all nine have positive reference controls on the diagnostic input. No further precision change is supported by these controls. Original inconclusive attempts, diagnostic durations and limitations remain recorded. Neither equal incidence, exact native prompt equivalence nor exclusion of earlier precision drift is claimed. The selected64 teacher-forced positions are not whole-reply accuracy.

- Observed anomaly: original phase1 task195 precision collapse drove the selected policy.
  Evidence: `anomalies/gpqa-phase1-precision/precision_selection.json` and its four `timed-onset-*/task195.onset.json` raw results.
  Control or comparison: BFP4/LoFi and BFP4/HiFi2 each fail one of three trials; BFP8-head/LoFi and full BF16/HiFi4 pass the three-trial panel. Raw failing tails and passing output tails were inspected. BFP8-head traced diagnostic rate34.24475 tokens/s/user exceeds full-precision16.95436; both faster tried policies fail. Per-trial counters record1023 model/sample replays and zero captures across the1023-token timed decode.
  Resolution: fastest tested passing reproducer remains BFP8/LoFi head with the reduced-precision decoder. Diagnostic timing is explicitly separate from vLLM performance and does not establish population anomaly rates.

- Observed issue: raw benchmark `max_concurrent_requests=2` despite required C1.
  Evidence: exact command/server configuration and per-request starts, TTFTs and ITLs.
  Investigation performed: recomputed minimum next-start minus prior-last-token gap as0.000382408s; all seven gaps positive. Client and server both enforce1. The raw metric uses inclusive one-second buckets.
  Resolution: actual serial C1 execution is supported; the bucket metric does not contradict it.

## Scope Inspected

- Contracts/skills: `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`, supplied AGENTS/operator decisions; installed0.1.27 stage-review, agentic-serving, datatype-sweep and relevant benchmark-model anomaly requirements.
- GPQA artifacts: final control summary, original512 and extension1024 raw HF outputs, TT control results/summary, runtime precision metadata, conditional recipes and case provenance; corrected aggregate and report. Recounted198 tasks,157 solved,186 original results plus12 unconditional corrected replacements. This agrees with79.2929% and the reported Wilson interval containing84.3%. Detailed generic harness correction was not re-reviewed here.
- Headline artifacts: `current/headline-head-bfp8-b1/` raw performance/commands, summary, configuration, launch, readiness, logging check, warmup, phase window/raw events, compressed roofline evidence, shutdown. Recomputed755.631794ms mean TTFT and34.770637 decode tokens/s/user from mean TPOT28.759899ms. All eight inputs are4096 and outputs128; zero failed requests. Warmup and measured invocations are distinct. Registration resolves this autoport, all60 layers, selected stage12_head_bfp8_lofi, context262144, mesh1x4.
- Phase accounting: evidence contains8 host-compatibility intermediate-prefill,8 device final-prefill and1016 device decode steps;32768 executed prefill tokens and1016 delivered decode tokens. The5.938679s prefill plus29.328819s decode intervals equal35.267497s request window, distinct from35.267993s benchmark duration by wrapper boundary overhead. Work is modeled from executed shapes and selected head precision; no hardware memory-counter claim. Host-wall percentages12.7612/33.5324 are kept outside device fields.
- Code: `tools/precision_repro_hf.py`, `tests/generate_full_reference.py`, `tools/benchmark_work.py`, adapter completion hooks in `tt/generator_vllm.py`, `tt/benchmark_phases.py`, benchmark anomaly runner and pinned harness cache-salt wrapper. HF generation uses independent cache copies and seed resets, saved prompt IDs and pinned tokenizer/reference; the asynchronous phase completion hook observes the already completed readback rather than introducing synchronization.
- Reporting: `REPORT.md`, model `README.md`, telemetry packet `/home/mvasiljevic/v5-agentic/logs/telemetry/packets/11ae0f9c-df27-4fdd-95fe-cf14076f2e61.json`. Packet headline matches raw measurements; device durations and roofline fields are null with explicit reason. No live profiler is claimed. Report retains earlier results and investigation limits.
- Commands run: read-only cat/sed/rg and short Python JSON/gzip/hash/arithmetic inspections. No device, server, reset, network, CI or GitHub command. Only this review file was written.

## Residual Risk

Shortened reconstructed prompts change history and position origin; HF and TT RNG streams differ. Positive controls do not explain every native long-trajectory divergence or estimate anomaly incidence. Exact generated-token replay of the native trajectory is not established (task129 includes the recorded four-token reconstruction discrepancy; task147 removes18329 preceding IDs). These restrictions are disclosed consistently in final summary/report/telemetry. Synthetic forced-length timing output remains unsuitable as quality evidence. SWE85 outcomes, solution-fetch audit and final entire-stage acceptance remain pending outside this scoped verdict. Local-only/no-push operator policy applies.
