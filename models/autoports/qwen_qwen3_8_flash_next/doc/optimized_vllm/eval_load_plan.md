# Preliminary final P4 load plan

CPU-only preparation, 2026-11-17. No target execution, server operation, or installed-plugin modification. This is a plan pending actual selected-budget results, not P4 acceptance or a measured capacity forecast. Keep the final server at resident2, C262144, TP4, selected precision, device-compatible seed policy, and the largest budget passing the33/17/9 interference sweep (alignment8, one-prefill/two-decode interleave under test). Run P5 timing before P4 as required by the installed workflow.

## EOS and comparison contract

The packaged [profile](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/eval_request_profiles.json:18) is thinking-enabled chat, T1/top-p.95/top-k40/seed42, with no `ignore_eos` setting. [Runner line251](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/run_long_answer_load.py:251) computes `profile_ignores_eos(profile) or not args.stop_at_eos`: its default forces EOS suppression, despite the reference prose saying suppression follows the profile. Therefore **use `--stop-at-eos` for the faithful main eval load**. It does not override a profile that explicitly sets `ignore_eos=True`, but this profile does not. The runner replaces the profile's32768 output cap with its own6000-token probe cap; it preserves T/p/k/seed and enables thinking through the served chat template. Natural short answers may produce answer-length/coverage advisories; retain those findings rather than silently forcing longer outputs or changing the eval profile.

The [comparison calls](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/run_long_answer_load.py:265) explicitly force `ignore_eos=True` for both profiles even with `--stop-at-eos`. They run plain first, then eval, each32 HTTP users and256 output tokens. Preserve these controlled comparison bursts and record their EOS override separately from the main load. Do not use `--no-compare` for the final run. The critical eval/plain median-TPOT ratio threshold is1.15; a passing ratio alone is not proof of device routing, so retain actual no-host route events for each arm.

## Command to execute after budget selection

Set `FINAL_SERVER_URL` and `FINAL_SERVER_LOG` to the actual final server. Derive `RESIDENT2_FUNCTIONAL_FLOOR_TPS` as10% of the selected full-model theoretical **per-active-user** decode rate at resident batch2, accounting for its actual expert union and TP4 layout. Do not substitute the historical single-layer/B1 roofline, measured throughput, or a theoretical resident32 model. If that derivation remains unavailable, omit the floor flag and carry the resulting missing-floor advisory explicitly; do not invent a passing value.

```bash
PYTHONPATH="/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime:$PWD" \
python_env/bin/python -m readiness_check.run_long_answer_load \
  --server-url "$FINAL_SERVER_URL" \
  --server-log "$FINAL_SERVER_LOG" \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --profile lm_eval_reasoning --plain-profile plain \
  --users 32 --max-tokens 6000 --stop-at-eos \
  --compare-tokens 256 --bucket-tokens 1000 \
  --request-timeout 36000 --settle-timeout 120 --metrics-interval 2 \
  --functional-floor-tps "$RESIDENT2_FUNCTIONAL_FLOOR_TPS" \
  --trace-capture-regex 'QWEN_VLLM_EVENT .*"kind":"capture"' \
  --ci-cap job
```

The36000-second timeout allows a long initial queue; it is a socket timeout, not an assertion about completion time. Do not run another load concurrently. Omit `--output` so the canonical `doc/eval_readiness/long_answer_load.json` is written where the checker expects it. Preserve earlier canonical results/raw directories before rerunning. The default plan remains explicitly a planning default; pass a model-owned `--eval-plan` only once actual release question counts/repeats/answer lengths are available. Changing its `concurrency` field to2 does **not** repair the installed forecast arithmetic.

## Capture window and health evidence

[Lines283–295](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/run_long_answer_load.py:283) take idle metrics and `log_start` **after both comparison bursts**, then time only the long load. `load_log` ends when that load returns; engine errors are read from `log_start` through settlement, while the zero-new-capture check uses only `load_log`. Thus comparison captures are warmup for this check and are not counted as load recaptures. The whole server log must nevertheless contain matching capture events or the runner records missing evidence.

The custom regex matches the adapter's actual compact [JSON event lines](../../tt/generator_vllm.py:46); the default regex does not reliably match `"kind":"capture"` followed by a `trace_id` field. Require zero capture events during the main load, and separately retain the mandatory mixed-shape capture/release audit. The P4 runner does not test release counts or all sampler-family switches by itself.

Inspect all32 completion records, live peak running/waiting, preemptions, return to zero occupants, KV return to idle, and engine health. `peak_running` should reflect resident capacity2;32 is offered HTTP concurrency. KV-fit arithmetic also assumes all32 users resident, so its preemption-fit interpretation needs qualification on this server. Do not manufacture missing metrics as zero.

## Correct capacity forecast

The installed [bucket rates](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/run_long_answer_load.py:85) exclude initial TTFT/queue wait, then report `median(per_user_tps) * len(records)` as aggregate. Its [forecast](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/run_long_answer_load.py:134) divides integrated per-answer decode seconds by `users=32`; `plan.concurrency` is merely reported. With only two simultaneous residents, these formulas can overstate ideal steady capacity by approximately32/2=16, before accounting for prefill and other stalls. They are not the final capacity denominator.

Preserve the raw probe output, but provide a separate corrected calculation using the measured main-load wall rate:

```
R_wall = output_tokens_total / load_s
H_eval = questions * repeats * expected_answer_tokens / (3600 * R_wall)
H_total = sum(H_eval)
fit_questions_i = floor(allocated_budget_seconds_i * R_wall
                        / (repeats_i * expected_answer_tokens_i))
```

The runner already records `wall_aggregate_tps` from this numerator/denominator ([line387 onward](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/run_long_answer_load.py:387)); `load_s` in JSON is rounded to0.1 seconds. This rate includes queueing, prefill and drain for the actual32-request burst. It is capacity evidence, never primary single-user t/s/u. Allocate the80%-of-CI-cap budget across evals so their allocations sum to that cap; the built-in per-eval fit suggestions each reuse the whole cap and cannot all be adopted together.

If using length-dependent decode buckets, the most an occupancy-proven resident2 approximation supports is `2 * r_bucket`, not `32 * r_bucket`; it still excludes prefill and requires explicit occupancy/overhead assumptions. The saved request summaries omit full token timestamps, so they cannot reconstruct exact bucket wall throughput afterward. Prefer `R_wall` for the first corrected forecast and label extrapolation beyond measured answer lengths. The default eval plan totals4,027,500 output tokens, yielding `1118.75 / R_wall` hours; those counts/lengths are planning defaults, not measured Qwen eval behavior. No numeric final forecast is possible until this load completes.

## Checklist follow-through

The current [optimization checklist](optimization_checklist.md) still describes several already-completed intermediate checks as pending and predates the k40 repair. Refresh it against final source/evidence before review. Specifically ensure:

- Canonical P4/P5 JSON and an explicit `scripts/check_eval_readiness.py --model-dir models/autoports/qwen_qwen3_8_flash_next --stage optimized-vllm` result. That checker requires both files; missing/malformed files cannot be replaced by a prose limitation. It checks recorded check statuses, not source freshness, profile faithfulness, resident-aware forecasts, or context identity.
- Final real-vLLM eval k40 and mixed greedy/k40 route evidence, supported non-aligned chunked-prefill traffic, trace retention/release counts and allocation safety after the sampler change. Reduced watcher and sampler-only tests do not close these gates.
- Final served-context and selected-policy identity; short/long/boundary timing; primary128/128/1 matched before/after, secondary CI100/100/32 capacity, and like-work full-model comparison. Preserve the required primary FUNCTIONAL roofline floor as well as P4's scoped floor.
- Current serial-prefill burst measurement and ratio/exclusion record. Existing historical budget256 burst and null candidate/serial ratios do not supply the requested final ratio; batched prefill remains optional and must not be claimed adopted.
- Explicit prefix-cache recurrent-state exclusion and batched-prefill contract rows, final serving qualitative/degeneracy and sampling checks, final cleanup of API/EngineCore processes, synchronized README/work log/perf summary, local stage-owned commits and independent stage review.

For real failed checks, preserve evidence and use the installed limitations schema only when its requirements are met: matching probe/check, nonempty reason, existing evidence, plus HF-same behavior or an actual two-hour fix attempt; non-waivable checks remain non-waivable. The user's requirement to continue failed gates through AutoFix remains in force. A corrected forecast does not turn a failed request or sampler fallback into acceptance.
