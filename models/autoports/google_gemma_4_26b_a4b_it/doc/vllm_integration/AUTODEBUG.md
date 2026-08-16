# AutoDebug Report: Gemma4 vLLM first-prefill crash

## Scope and starting evidence

- Target: `google/gemma-4-26B-A4B-it`, vLLM integration stage.
- Failing run: the `run_vllm_server` invocation recorded in `readiness_vllm/server.log`, with `max_model_len=262144`, `max_num_seqs=1`, asynchronous scheduling, trace mode `all`, and `sample_on_device_mode=all`.
- Logs inspected:
  - `models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/server.log`
  - `models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/sampling_tests.log`
- Code inspected:
  - `models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py`
  - `models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py`
  - `/home/mvasiljevic/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py`
  - `/home/mvasiljevic/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py`
  - `/home/mvasiljevic/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/input_batch.py`
- No hardware or server command was run and no implementation file was edited.
- Initial worktree was already dirty: modified Gemma4 `generator.py` and readiness runner; untracked Gemma4 adapter/test plus unrelated untracked trees. These changes must be preserved and attributed before committing.

## Headline finding

The EngineCore crash is a verified adapter/plugin representation mismatch, not a device or model-compute failure. The TT plugin uses `SEED_NONE_SENTINEL=-1` internally and deliberately converts it back to Python `None` before calling the model. `Gemma4ForCausalLM._sampling_values()` then executes `int(v)` for every seed. A normal greedy request with no explicit seed therefore raises `TypeError` before sampling can run.

Evidence: `server.log` lines 841-859 show the first prefill reaching `submit_prefill -> generator_vllm.prefill_forward -> _sampling_values`, then failing at `tuple(int(v) for v in params.seed[:batch_size])` with `int(None)`. Lines 860-915 repeat the same root stack as the EngineCore exits. The four smoke pytest failures are downstream HTTP/engine-death symptoms (`response.choices is None`), not four independent sampling bugs.

## Hypothesis experiments

### H1 — verified: unset request seeds are not normalized at the adapter boundary

- Prediction: constructing `TTSamplingParams` with `seed=[None]` and calling `_sampling_values(..., 1)` raises the exact logged `TypeError`; an explicit integer seed does not.
- Focused experiment (CPU/source-only):

  ```bash
  python - <<'PY'
  from vllm_tt_plugin.model_input import TTSamplingParams
  from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import Gemma4ForCausalLM
  common = dict(temperature=[0.0], top_k=[1], top_p=[1.0])
  for seed in ([None], [42]):
      p = TTSamplingParams(**common, seed=seed)
      try:
          print(seed, Gemma4ForCausalLM._sampling_values(p, 1))
      except Exception as e:
          print(seed, type(e).__name__, str(e))
  PY
  ```

- Expected before fix: `[None]` reproduces `TypeError`; `[42]` succeeds.
- Smallest fix boundary: normalize `None` seeds in the adapter before calling the canonical generator sampler. Do not change the plugin's sentinel restoration, since it is shared behavior and preserves the semantic distinction between absent and explicit seeds. Select the fallback seed according to the shared TT sampler contract (at minimum a valid integer such as the generator default `0`), and add a unit test for both `None` and explicit seeds.
- Verification after fix: run the CPU experiment and adapter tests, then rerun only one greedy completion against a live server before the smoke sampling profile.

### H2 — likely immediate correctness issue: absent-seed normalization may accidentally make all unseeded sampled requests identical

- Evidence: `generator.py` requires integer seeds and keys captured sampling graphs by the full `SamplingSpec`, while vLLM represents an absent seed as `None`. Blindly mapping every `None` to constant zero removes the crash but can make nominally unseeded stochastic requests deterministic and correlated across request slots.
- Prediction: after the crash fix, greedy requests pass, but variety/isolation tests using nonzero temperature without explicit seeds may repeat exactly across requests or runs.
- Focused verify/refute experiment: first unit-test the chosen `None` policy and document it; then run the smallest live pair of identical non-greedy requests without explicit seeds and compare with an explicit same-seed pair. Explicit same seeds must reproduce; absent seeds should follow the shared plugin's intended semantics. If shared on-device sampling cannot express unseeded randomness, make that compatibility limitation explicit rather than silently treating `None` as a user-provided seed.

### H3 — likely next serving failure: chunked-prefill continuation is not actually implemented by the generator

- Evidence: plugin `model_runner.py` passes `input_tokens[:, :computed+chunk_len]`, `start_pos=input_positions`, and cumulative `prompt_lens=computed+chunk_len`. Its own comment says the generator should process `tokens[start_pos:prompt_lens]`. The adapter discards the supplied `start_pos` in `prefill_forward`, and `Gemma4Generator.prefill_forward` processes `tokens[:, :logical_len]` with positions starting at zero. Thus a continuation chunk will recompute the prefix and write positions from zero instead of filling only the new cache range.
- Prediction: prompts exceeding the plugin's 2048-token chunk budget (or resumed/APC-like prefills) corrupt/overwrite cache state or produce incorrect output, even though a short first-prefill request can pass.
- Focused experiment: a targeted runner/unit probe with synthetic `tokens`, `start_pos > 0`, and cumulative `prompt_lens`, instrumenting the low-level model call to assert that token slice and position IDs begin at `start_pos`. Follow with one live prompt just over 2048 tokens and compare output/cache position behavior with standalone full-model prefill. Fix at the adapter/generator prefill contract boundary; do not disable chunked prefill or reduce advertised context as a workaround.

### H4 — verified static contract gap for larger serving: request-to-state slot routing is discarded/rejected

- Evidence: `prefill_forward` receives `empty_slots` but immediately deletes it, so it assumes row index equals persistent cache/state slot. `decode_forward` raises `NotImplementedError` whenever the plugin supplies `slot_remap`. The plugin explicitly maintains persistent request slots and supplies these fields when layouts change. This is masked by the current `max_num_seqs=1` run but conflicts with required multi-request serving and async scheduling.
- Prediction: request churn, reordered batches, or concurrency greater than one will either raise on `slot_remap` or associate tokens/positions with the wrong cache row.
- Focused experiment: adapter-level fake-generator tests with non-identity `empty_slots` and `slot_remap`, asserting cache table rows, active mask, tokens, and positions follow persistent slots. Then run a two-request churn/isolation test before scaling to 32.

### H5 — likely performance/stability issue: page-table content changes force trace release/recapture

- Evidence: `_refresh_page_tables()` detects any host table content change; `decode_forward()` then calls `_release_decode_traces()` on `page_changed`. Normal block allocation can change table contents. The canonical generator trace keys stable device page-table object identities, so copying new contents into the same device tensors should not by itself require trace destruction.
- Prediction: crossing a new KV block boundary causes avoidable trace recapture, latency spikes, and possible allocator/capture hazards. At block size 64 this can recur during ordinary generation.
- Focused experiment: fake counters/unit test where host page-table content changes while target TT tensor identity stays fixed; assert a copy occurs but trace cache remains. On hardware, compare trace-capture counters before and after crossing token 64. Retain trace release only for shape/address/layout changes proven to invalidate capture.

### H6 — async read event ordering needs a focused proof

- Evidence: `read_decode_output(async_read=True)` calls `tt_out.cpu(blocking=False)` and only afterwards records an event on CQ0. This may be correct if the copy and event are enqueued on the same queue, but the adapter has no focused test proving the returned host object is materialized only after `event_synchronize`. `supports_async_decode=True` and overlap are already advertised.
- Prediction: under overlap, deferred output can occasionally observe incomplete/stale token data if the event does not fence the actual read queue.
- Focused experiment: a mock ordering test for the adapter call sequence plus the plugin stale-token/current-position overlap test under `sample_on_device_mode=all`. Keep both async capability flags only after that test passes with decode tracing enabled.

## Recommended repair order

1. Fix H1 and add a CPU unit regression for `seed=None`, explicit seeds, and greedy parameters.
2. Run one targeted greedy request, then smoke sampling. Classify any new root stack rather than treating the four current pytest failures independently.
3. Resolve H3 before long/non-aligned context evidence; the current code does not honor continuation `start_pos`.
4. Resolve H4 before claiming concurrency or request isolation.
5. Prove/refute H5 and H6 with focused counter/overlap tests before performance benchmarks or advertising async overlap.

## Final status

- Primary crash: root cause verified; smallest repair is clear but intentionally not implemented in this diagnosis-only pass.
- Current sampling smoke: invalid as correctness evidence because all tests ran after the single EngineCore crash.
- Hardware status: no device fault is implicated by this stack; no reset or hardware experiment is warranted for this failure alone.
- Remaining risk: fixing only `int(None)` should permit the first short prefill to advance, but chunked-prefill position handling and persistent slot routing are probable next gates and should be tested immediately.
