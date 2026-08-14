# Optimized vLLM work log

## Scope and starting evidence

Started from committed vLLM integration `3381c24c309` / documentation commit `1d1d40ffe15` and nested vLLM plugin commit `98d51d0`. The selected datatype configuration is `doc/datatype_sweep/selected_precision_config.json` (`bfp4_lofi_bfp8kv_bf16ccl`). `doc/context_contract.json` advertises and physically proves 32,768 tokens on TP4. The real measured route is the TT plugin model registered to `models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.generator_vllm:TTMistralSmall24BForCausalLM`.

Baseline artifacts were the completed integration's `readiness_vllm/vllm_benchmark.json` and `vllm_ci_serving_benchmark.json`: primary 128/128/1 at 18.927619 ms TPOT / 52.832848 token/s/user, and CI 100/100/32 at 19.618114 ms TPOT / 1026.973775 aggregate token/s.

## Commands

Production/final server shape:

```bash
TT_MISTRAL_TEXT_VER=mistral_small_24b_autoport \
PYTHONPATH="$PWD/vllm/plugins/vllm-tt-plugin/src:$PWD/vllm:$PWD" \
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/mistralai_mistral_small_24b_instruct_2501 \
  --hf-model mistralai/Mistral-Small-24B-Instruct-2501 \
  --mesh-device P300x2 --max-num-seqs 32 --block-size 32 \
  --max-model-len 32768 \
  --tt-config '{"sample_on_device_mode":"all","trace_region_size":200000000,"fabric_config":"FABRIC_1D"}'
```

The final benchmark/full-compatibility server added `MISTRAL_SMALL_24B_VLLM_HOST_SAMPLING_COMPAT=1`; this exposes host-only API compatibility without changing greedy benchmark routing. The 72-pass/1-skip full suite is labeled compatibility coverage and is not used as stochastic device-path proof. Attached checks:

```bash
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages sampling --sampling-profile full --server-url http://127.0.0.1:8000 \
  --model-dir models/autoports/mistralai_mistral_small_24b_instruct_2501 \
  --hf-model mistralai/Mistral-Small-24B-Instruct-2501 --max-num-seqs 32

python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages qualitative,benchmark --server-url http://127.0.0.1:8000 \
  --model-dir models/autoports/mistralai_mistral_small_24b_instruct_2501 \
  --hf-model mistralai/Mistral-Small-24B-Instruct-2501 --max-num-seqs 32

python models/common/readiness_check/check_degenerate_output.py \
  --hf-model mistralai/Mistral-Small-24B-Instruct-2501 \
  --missing-artifacts critical --scope vllm
```

Production stochastic proof used a separate server with the same shape and TT config, host compatibility disabled, and `TT_DEVICE_SAMPLING_AUDIT=1`. A 192-token chat prompt generated 128 tokens with temperature 0.7, top-p 0.9, top-k 32. Two subsequent waves each submitted four greedy and four supported stochastic requests, reversing request order in the second wave to force slot reuse. All 17 requests returned 200, `/health` remained 200, and the server log records `perform_device_sampling=True` for stochastic prefill/decode and mixed decode. The same server then ran the complete six-prompt qualitative stage on the production device sampler.

## Async/trace/runtime audit

- Plugin startup: `trace_mode=all`, `sample_on_device_mode=all`, `async_scheduling=True`.
- Adapter: `supports_async_decode=True`; production `decode_forward` returns the generator's device token tensor when `read_from_device=False`; deferred `read_decode_output(async_read=True)` uses nonblocking `ttnn.from_device`; host processing reads only sampled token IDs.
- Generator: persistent token/current-position/RoPE/page-table/sampler tensors are allocated once. `_replay_split_sampling` executes both model and sampler traces with `blocking=False`; sampler output aliases the token feedback tensor. Sampling parameters update at request boundaries. Page tables copy only after content change.
- Steady async overlap: `reset_batch=False` ignores stale host token/position and preserves device-authoritative feedback. Reset, fresh prefill, remap, changed page table, unchanged page table, and inactive-row cases are covered in `tests/test_vllm_adapter.py` and the canonical full sampling suite.
- Runtime fallback audit: host full-logit readback, eager host sampling, `untilize`, and host argmax occur only in explicit diagnostic/compatibility methods and are not reachable from supported `sampling_params` production serving. There is no adapter-local argmax or token-feedback reconstruction. No profiler environment was used.

## Candidate experiment

Hypothesis: Sampling1D replicates sampled IDs on all TP ranks, so reading only rank 0 at the async boundary might reduce host transfer/event overhead.

Experiment: change only `read_decode_output`/host formatting to schedule one rank's read; run the same production server, non-aligned check, qualitative suite, primary 128/128/1 benchmark, and CI 100/100/32 benchmark.

Result: correctness passed, but primary moved from 52.832848 to 52.800455 token/s/user (TPOT 18.927619 to 18.939231 ms) and CI moved from 50.973299 to 50.936766 token/s/user (TPOT 19.618114 to 19.632185 ms). Verdict: **refuted / rejected**. The implementation and test changes were reverted; artifacts are in `candidates/shard0_read/`.

The full-model trace is 54.451842 token/s/user. Final serving is 52.839005 token/s/user, so the serving gap is below 3%; no evidence justified reopening decoder math, LM-head topology, datatype, or collective families already closed by optimized-full-model and datatype-sweep.

## Final evidence

- Adapter unit contract: 18/18 before the candidate and 19/19 with the candidate-specific test; final source was restored and rerun after documentation.
- Host-only API compatibility sampling: 72 passed, 1 expected skip in 521.70 seconds.
- Production stochastic sampling: one 192/128 request at temperature 0.7 / top-p 0.9 / top-k 32 and two 8-request mixed greedy/stochastic slot-reuse waves passed with 17/17 HTTP 200, health 200, and live `perform_device_sampling=True` evidence. The six sampled qualitative completions also ran on device. See `final/production_device_sampling_evidence.json` and `final/server_production_device_sampling.log`.
- Non-aligned: logical prompt length 37 passed on both candidate and final default.
- Qualitative: six HF-declared chat prompts, greedy and sampled on the production device route, manually inspected; all coherent and on-topic. Automated degenerate-output check passed. The richer format artifact records the exact HF snapshot, tokenizer, regex fix, chat-template presence, prompt source, and exact-ID control. HF/optimized-full-model controls are retained under `doc/optimized_full_model/qualitative_suite/`.
- Primary final: 128/128/1, TTFT P50/P99 578.776/578.776 ms, mean/P99 TPOT 18.925/18.925 ms, ITL P50/P99 17.756/18.843 ms, output 42.915 token/s, decode 52.839 token/s/user, 1/1 complete.
- CI final: 100/100/32, TTFT P50/P99 1192.476/1193.635 ms, mean/P99 TPOT 19.887/24.216 ms, ITL P50/P99 17.939/76.613 ms, output 1018.207 token/s, 32/32 complete. Secondary only.
- Context: unchanged 32,768; max-num-seqs 32; no aligned-only path introduced.
- Cleanup: both owning runners terminated cleanly; no vLLM/API/EngineCore process remained; all four boards listed. `final/cleanup.log` preserves the final check.

## Stage-review remediation and AutoFix

The first independent review returned `more-work-needed` for two evidence gaps. First, compatibility mode routes all stochastic requests to the host, so its full-suite pass could not prove production Sampling1D. Second, its saved qualitative output contained `learning,,,`. Both findings were treated as blocking.

The production stochastic rerun above closes the first finding without changing the workload or TT configuration. Opt-in routing observability was added to the nested TT plugin; it de-duplicates signatures and is inert unless `TT_DEVICE_SAMPLING_AUDIT=1`. The old smoke's first target, `test_mixed_params_batch`, combines host-only top-k 100, explicit seeds, and penalties. With compatibility disabled, that batch selects the disabled host path and can terminate EngineCore; therefore the saved smoke fatal is not evidence that supported top-k-32 Sampling1D crashed. This classification is a source-backed inference because the overwritten original server log is unavailable. Separately, the remediation server's first startup reproduced the known device-0 ERISC heartbeat failure before model initialization. After one bounded reset and successful 1x4 mesh open/close, the identical server started and all supported stochastic checks passed.

`$autofix` was invoked for the punctuation finding. The fresh `$autodebug` runner was attempted but its isolated Codex process could not create a Bubblewrap loopback namespace (`RTM_NEWADDR: Operation not permitted`); the source-only report was completed from the parent environment and saved at repository root as `AUTODEBUG.md`. Focused experiments ran the exact prompt twice with `httpx`, twice with the OpenAI client, twice after the same greedy -> stochastic transition, and eight times concurrently. All were clean and deterministic. The full host-compatibility-disabled qualitative stage was also clean; the final, candidate, and readiness output SHA is `a4a2338f026e0baefcd69a40d41b51fc8889bcb1645fa6d878a5ac7a3c07f3f9`. The comma sequence tokenizes as two distinct IDs, not a stuck repeated token. Verdict: the production token/position/page-table/slot/async corruption hypotheses were refuted, no functional fix was justified, and the clean production artifact replaced the compatibility artifact.

## Hardware recovery

After the first production server stopped at expected unsupported smoke cases, an immediate relaunch hit device-0 active-Ethernet core 29-25 heartbeat timeout before model initialization. With server ownership absent, the bounded `tt-smi -ls --local`, `tt-smi -r`, `tt-smi -ls --local` sequence restored all four boards; a `FABRIC_1D` 1x4 mesh open/close printed `MESH_SMOKE_OK`. Final serving then passed. The same bounded reset and mesh smoke were used between the candidate and final server to avoid the known dirty-device transition. This is infrastructure recovery, not a model failure.

## Optimize checklist disposition

Serving-specific items are complete: real TT-plugin generator path; traced model and sampling; nonblocking replay; async split; persistent state; stale-input/page-table coverage; on-device greedy split sampling; no host fallback; identical before/after harness; final-default reproduction; 32-concurrency evidence; qualitative and degenerate-output gates; context and non-alignment preservation; clean teardown. Device-math topology, precision/fidelity, sharding, packed projections, DRAM-sharded matmuls, CCL families/buffers, LM-head geometry, and terminal sampling comparisons are inherited with evidence from `doc/optimized_multichip_decoder/`, `doc/optimized_full_model/`, and `doc/datatype_sweep/`. No live-serving Tracy, `tt-perf-report`, watcher, adapter profiler, device profiler, or `ReadDeviceProfiler` was attempted.

## Benchmark command comparability

Primary before/after used the same 128/128/1 `run_vllm_server` workload, greedy device sampling, max-num-seqs 32, max-model-len 32768, TP4 mesh, and TT config. For CI burst, the baseline command explicitly set `--max-concurrency 32`; the final command omitted the cap while submitting exactly 32 requests. Both raw results report observed concurrency 32. This command-level difference is disclosed because “same harness” would otherwise be too strong, although effective work is the same.

## Commits

Fresh independent stage rereview: `clean-pass` in `stage_review.md`.

- Nested `vllm` repository, branch `dev`: `6bd775d` (`Add opt-in device sampling route audit`).
- Root `tt-metal` repository, branch `mvasiljevic/fast-models/mistralai-mistral-small-24b-instruct-2501`: `7e14ba84874` (`Record optimized Mistral vLLM serving evidence`).
- SHA-record update: `cf1410271de` (`Log optimized vLLM checkpoint SHAs`).

No push was performed.
