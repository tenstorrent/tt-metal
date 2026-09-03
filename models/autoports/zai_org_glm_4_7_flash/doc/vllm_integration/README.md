# GLM-4.7-Flash vLLM integration: stage report

Target: `zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`, 30.6B total / ~3.6B
active, 47 layers, vocab 154880), served through the Tenstorrent vLLM plugin on
**one Blackhole p150**, device 0, 1x1 mesh. Branch
`ttmodelmanager/glm47-flash-probe`.

Adapter: `tt/generator_vllm.py` (`GLM47FlashForCausalLM`), registered as
`TTGlm4MoeLiteForCausalLM` in `vllm-tt-plugin` (`9f2ec5d`, branch
`ttmodelmanager/glm47-flash-registration`). It is a translation layer only: every
forward/cache/sampling primitive is the full-model generator's, including the
canonical split-sampling traced decode path. No separate sampling path, no host
argmax, no full-logits readback on the measured path.

## Headline

All numbers below are from **one run at the real serving spec**: full
202752 context, `max_num_seqs=32`, on-device sampling
(`sample_on_device_mode=all`, enforced by the harness).

| | value | source |
|---|---|---|
| **TTFT, 128-token prompt** | **273.8 ms** | primary profile, `vllm_benchmark.json` |
| **Decode, batch 1 (headline t/s/u)** | **45.0 ms/token = 22.2 t/s/u** | TPOT p50/p99/mean all 45.0 ms |
| ITL (p50 / p99) | 45.0 / 45.5 ms | |
| End to end, 128 in / 128 out | 5.99 s | 1 request, `--max-concurrency 1` |
| **Serving burst, 100/100/32** | 137.1 tok/s output, 274.2 tok/s total | `vllm_ci_serving_benchmark.json` |
| burst TTFT p50 / TPOT | 14444 ms / 89.9 ms | admission-bound; **not** headline decode |
| burst e2e, 32 requests | 23.3 s | |

Raw: `readiness_vllm/vllm_result.json`, `vllm_ci_serving_result.json`.
Commands are recorded verbatim in each summary's `command_string`.

Server command for every stage below:

```
python -m models.common.readiness_check.run_vllm_server --stages serve \
  --model-dir models/autoports/zai_org_glm_4_7_flash \
  --hf-model zai-org/GLM-4.7-Flash \
  --mesh-device N150 --max-num-seqs 32 --max-model-len 202752 \
  --tt-config '{"trace_region_size": 350000000}'
```

Boot at the full 202752 context is clean: **zero** `TT_FATAL` / OOM / engine-core
failures, after the VS-006 prefill-chunk cap.

## The serving overhead worth acting on

The full model's own traced token-out decode is **22.994 ms/token (43.49 t/s/u)**
(`doc/full_model/README.md`). Through vLLM it is **45.0 ms/token (22.2 t/s/u)**.

That is **~22 ms/token, roughly 2x, of vLLM-path overhead** on an otherwise
identical traced model+sampling replay. TTFT moves the other way (273.8 ms
served vs 334.2 ms standalone), so prefill is not the cost. This is the single
largest optimisation target for stage 08 (optimized vLLM: async decode, trace
reuse, on-device sampling), and it is stated here as a lower-bound comparison
exactly as the stage requires, not as a model limitation.

## Stage results

| stage | result |
|---|---|
| serve (202752 ctx, 32 seqs) | healthy, 0 fatals |
| **sampling (smoke) -- the gated profile** | **3 passed, 1 skipped, 0 failed** |
| sampling (full) -- recorded, not gated | 11 failed, 62 passed, 1 skipped |
| qualitative | passed, 6 prompts, greedy + sampled |
| benchmark | both profiles completed |
| gate: `check_degenerate_output --scope all` | **exit 0**, "No degenerate output detected" |
| gate: `check_context_contract --stage vllm` | **exit 0**, target 202752 = supported 202752 |

Sampling evidence is kept as two files so neither claim is overstated:
`sampling_tests.log` / `sampling_tests_smoke_GATED.log` (the gated smoke run) and
`sampling_tests_full_RECORD.log` (the full profile, for the record).

Qualitative outputs are coherent, on-topic, English, with no repetition or
language drift. Degenerate-check metrics across all 12 completions: adjacent
duplication 0.0-0.015 (critical threshold 0.10), trigram-loop fraction
0.031-0.075 (advisory threshold 0.50). The visible reasoning scaffold
("1. **Analyze the Request:**") is this model's own style, not a decode defect.

## Known limitations

**1. Sampling status is `smoke-gated`, accepted by the project owner.**
The `$vllm-integration` skill allows recording the final status as `smoke-gated`
rather than the full profile when the full profile is impractical, with owner
acceptance, which was given explicitly. The smoke profile passes at spec; the
full profile does not, for the reason below. This is not presented as equivalent
to the full sampling gate.

**2. All 11 full-profile failures are one upstream serving-state defect:
[tenstorrent/tt-metal#55408](https://github.com/tenstorrent/tt-metal/issues/55408).**
Greedy (`temperature=0`) requests lose determinism in a mixed batch after a
long-lived server has served a long host-sampled request. Established on
hardware:

* Every failing test **passes when run alone against a freshly started server**.
* Bisected to two triggers with a canary, `test_bad_words` and `test_min_tokens`;
  the discriminator is request length (`max_tokens=100`), not the parameter, as
  three other host-sampled tests at `max_tokens=10` do not poison it.
* Per-request seeds are **not** implicated: all 10 request seeds derive identical
  device seed, counter and salt fresh vs poisoned.
* It **predates** this stage's fixes, and it affects the `tt_transformers`
  reference (SmolLM2-135M) on the same chip **worse**: that model fails the same
  canary at baseline on a fresh server, where this model passes.
* Nine hypotheses eliminated on hardware, including #48222 (`ttnn.sampling`
  matched `torch.argmax` on 256/256 rows at this model's shapes) and #50512
  (multi-device TP only). See work log VS-009.

It does not affect single-request correctness on a freshly started server and
does not touch either runner-side gate, both of which pass.

**3. Not exercised:** KV-cache migration (never enabled for this model), and
multi-host/multi-rank serving (single chip by design).

## Files

* `tt/generator_vllm.py`, `tests/test_generator_vllm_adapter.py`
* `readiness_vllm/` — both sampling logs, qualitative outputs, both benchmark
  summaries and raw results. The raw vLLM `server.log` (729 KB) is kept on disk
  but not committed: it exceeds the repo's 500 KB file limit and is a debug log,
  not stage evidence.
* `work_log.md` — VS-001..VS-009, the full investigation including corrections
