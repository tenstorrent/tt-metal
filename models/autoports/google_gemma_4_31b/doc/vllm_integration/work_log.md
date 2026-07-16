# Gemma 4 31B vLLM Integration Work Log

## 2026-07-15 to 2026-07-16

- Started from completed datatype-sweep checkpoint `d728e01c00f`.
- Added the thin `tt/generator_vllm.py` adapter and registered
  `TT_GEMMA4_TEXT_VER=gemma4_31b_autoport` in the TT vLLM plugin.
- Reused selected precision config `lm_head_bfp8_hifi2`: BFP8 attention and LM
  head, BFP4 MLP, BF16 activation/residual/prefill CCL, BFP8 decode CCL and KV,
  BF16 logits, LoFi attention/MLP, HiFi2 LM head, FP32 sampling gather, and no
  layer exceptions.
- Disabled standalone cache allocation in serving. The adapter requires and
  passes through vLLM-owned hybrid KV caches and page tables.
- Implemented the 50-sliding/10-global HMA cache spec, sharing ten physical K/V
  workspaces across six logical attention groups. Native decode geometry keeps
  fused paged update; cross-group geometry uses equal-volume non-fused updates.
- Delegated greedy prefill/decode to the full-model generator's canonical split
  sampler and traced token-feedback state. Dynamic active batches retain async
  decode and release both cooperating traces before host-only eager work.
- Added a direct non-aligned 149-token request, maximum-total-length request,
  exact repeated/cross-batch top-20 logit determinism, and adapter contracts for
  stale token, current position, page tables, cache ownership, and trace state.
- Added model-scoped `greedy_only` device sampling. Optional host compatibility
  is explicit for stochastic, penalty, allowlist, structured-output, and
  logprob tests; the benchmark path remains device greedy.
- Added raw-continuation qualitative mode because the base tokenizer has no
  chat template. Final artifacts record `prompt_format=raw_continuation`.

## Autofix record

- A reduced mixed-parameter run first failed with an empty pending-sample
  queue. The TT runner had masked an earlier `execute_model()` exception;
  matching upstream no-pending behavior exposed an undefined host-branch page
  table after the dynamic-batch refactor. The page table is now converted after
  synchronized trace release, and direct tests cover the branch.
- A reduced HMA run exposed full-attention decode cache geometry mismatch.
  Source inspection established the supported equal-volume non-fused paged
  update and ruled out storage-copying reshapes.
- Mixed eager/traced traffic exposed allocator-lifetime warnings. Transition
  allocations were moved after synchronized trace release. The one remaining
  warning is the inherited controlled registration of the cooperating sampler
  trace, bracketed and documented in `anomaly_ledger.md`.
- The first full sampling suite reported `68 passed, 1 skipped, 4 failed`.
  Autofix established four oracle defects involving Gemma control IDs, decoded
  characters versus token IDs, and an invalid bounded-presence-penalty
  expectation. The tests now inspect the correct observable; production
  sampling routing did not change for those failures.

## Physical context audit and repair

- `262144`, `22533` pool blocks: failed full-depth HMA KV allocation.
- `157696`, `13557` blocks: failed on physical KV buffer 19 of 20.
- Review rejected both an arbitrary reserve and startup plus a short prompt as
  maximum-context evidence. Real maximum-prompt attempts at `101888` and
  `100800` exposed prompt-sized concat/all-reduce peaks; the latter also proved
  the request checker had to reject HTTP-200 error envelopes.
- Normalized attention input is released immediately after QKV, and the full
  pre-FFN normalized input is released after its last chunk consumer.
- Later `108672`/`111488` probes isolated sliding/full attention concat,
  attention all-reduce, post-attention norm/residual, MLP residual, and global
  HMA cache-read geometry constraints.
- Autofix streams full/sliding SDPA and head concat, attention output projection
  and BF16 all-reduce, post-attention norm/residual, and long-prompt MLP
  norm/residual in 4096-row chunks. Global attention reads the equal-volume HMA
  cache through a zero-copy view with layer geometry.
- The final dominant peak is `4096*C + 4032*4096` bytes/bank. At `113280`, the
  `9740`-block pool leaves `483372736` post-KV bytes/bank; mandatory peak
  `480509952` plus page tables `1704960` leaves margin `1157824`.
- Adjacent aligned `113344` is source-proven short `148800` bytes/bank.
  `111488` passed as a conservative control, then `113280` passed a direct
  `113279`-input plus one-output request.

## Final validation

- Final runner command: the complete environment and command are recorded in
  `README.md`; it used stages
  `serve,sampling,qualitative,benchmark`, `sampling-profile=full`, raw prompts,
  `P150x4`, `max_num_seqs=32`, `max_model_len=113280`, block size 64, async
  scheduling, device sampling and tracing for all supported greedy traffic.
- Launched server command:

  ```bash
  /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3 \
    --block_size 64 --max_num_seqs 32 --port 8000 \
    --max_model_len 113280 \
    --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":268435456,"fabric_config":"FABRIC_1D","trace_mode":"all","enable_model_warmup":true}}' \
    --async-scheduling \
    --chat-template /localdev/odjuricic/tt-metal/models/autoports/google_gemma_4_31b/doc/vllm_integration/chat_template.jinja
  ```
- API ready; direct 149-token request passed.
- Exact top-20 logits and chosen token were stable across two runs and target
  batch positions 0/1/2; chosen token matched the standalone selected-TT
  control.
- Direct `113279` input + 1 output request passed at the advertised limit.
- Shared sampling suite: `72 passed, 1 skipped`.
- Six greedy and six sampled raw continuations saved and read; qualitative
  serving integrity passed with weak base-model instruction following
  documented. Degeneracy checker: zero findings, exit 0.
- Primary and secondary CI-burst benchmarks completed without missing output.
- Final runner terminated cleanly. Process audit found no live vLLM/API/runner/
  EngineCore holders; historical PID-1 zombie records only. All four P150b
  devices were healthy after cleanup.
- Firmware 19.9 emitted the expected warning that 19.5 is the latest fully
  tested Blackhole bundle; no firmware recovery or runtime fallback occurred.

## Final metrics

Primary single-user, 128 requested input / 128 output / 1 request / concurrency
1 / temperature 0 / ignore EOS:

- Raw: `readiness_vllm/vllm_result.json`
- Summary: `readiness_vllm/vllm_benchmark.json`
- Actual workload: 127 input tokens, 128 output tokens, one request.
- TTFT P50/P99 `992.586/992.586 ms`; TPOT mean/P99
  `38.023/38.023 ms`; ITL P50/P99 `29.348/29.739 ms`; output throughput
  `21.974 tok/s`; TPOT-derived decode `26.300 t/s/u`.

CI serving burst, 100 requested input / 100 output / 32 requests / burst
admission / temperature 0 / ignore EOS:

- Raw: `readiness_vllm/vllm_ci_serving_result.json`
- Summary: `readiness_vllm/vllm_ci_serving_benchmark.json`
- Actual workload: 99 input tokens/request, 100 output tokens/request,
  32-request burst with maximum observed concurrency 32.
- TTFT P50/P99 `8485.248/8488.457 ms`; TPOT mean/P99
  `77.373/127.442 ms`; ITL P50/P99 `55.807/687.715 ms`; aggregate output
  throughput `201.070 tok/s`; TPOT-derived `12.924 t/s/u`.
- This burst TPOT is secondary, not headline decode t/s/u, because admission
  and prefill scheduling affect it.

## Artifacts, cleanup, and provenance

- Final server/run: `readiness_vllm/server.log` and
  `doc/vllm_integration/evidence/final_full_113280_vllm_run.log`.
- Prompt gates: `non_aligned_prompt_check.json`,
  `max_context_prompt_check.json`, and `logit_determinism.json`.
- Sampling and quality: `sampling_tests.log`,
  `vllm_qualitative_outputs.json`, `degenerate_output_check.json`, and
  `qualitative_verdict.md`.
- Benchmarks: `vllm_result.json`, `vllm_benchmark.json`,
  `vllm_ci_serving_result.json`, and `vllm_ci_serving_benchmark.json`.
- No live processes were left holding devices; historical PID-1 zombies only.
- Main repository Stage 09 implementation/evidence commit:
  `e07e401794d5b34b61526d5c097e7c68e81189d3`.
- Nested vLLM plugin commit: `91c467d6fc18c4386eda14360baf0bee0e0f684c`.
- No push performed.
