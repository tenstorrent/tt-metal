# Serving seed and reload investigation

Source inspection and host-only adjudication, 2026-09-11. No accelerator commands or model execution were performed by this investigation.

Initial checkout HEAD: `761f5e2ff138f173f80c4681bb087b77a08f7e08`, plus the staged native-prefill changes. Diagnosis line references describe that pre-fix source; the implementation below shifts adapter lines.

## Observations and scope

- The preserved initial run is `readiness_vllm/initial_sampling_attempt/sampling_tests.log`: mixed-parameter request isolation failed; top-k=1 greedy passed; min-p smoke passed; all-vocabulary chat logprobs skipped. The failure prints `temperature=0.5, seed=42` and different text, but omits the prompt, penalties, shuffled order, physical slot and token IDs. Three configurations have those printed sampling values. The text alone cannot identify the failing configuration or first divergent token.
- The tested plugin is `/tmp/qwen-ci-vllm-plugin`, commit `c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42`. Its `src/vllm_tt_plugin` implementation is the relevant serving boundary.
- Earlier checked-in `../../readiness_vllm/sampling_tests.log:47` reports this test passing, through a different plugin checkout path. This is historical evidence, not proof that current plugin/source semantics are identical. Current staged native-prefill changes do not modify the shared seed manager or the Qwen vLLM adapter.
- Full-model quality and generator latency reported by the parent do not exercise mixed request seeds, penalty history restoration or the plugin's asynchronous decode contract.

## 1. Proven: Qwen enables admission-order salting of independent request seeds

At investigation start, `tt/generator.py:27` defined `SamplingArgs` without `salt_duplicate_seeds`. Its constructor at `tt/generator.py:73` passed those arguments directly to `SamplingGenerator`. `models/common/sampling/generator.py:130` consequently constructed `SeedManager` with `getattr(args, "salt_duplicate_seeds", True)`.

The model adapter's prefill path calls `apply_prefill_state` with compact request seeds and physical slots (`tt/generator_vllm.py:291`). That helper calls `reset_seed` (`models/common/sampling/generator.py:211`), which assigns each same-seed request the smallest salt not already occupied (`:861`, `:881`, `:1065`). The seed hash includes this salt (`:39`, `:55`, `:851`). The canonical mixed test has six independent seed-42 requests and shuffles their submission order (`/tmp/qwen-ci-vllm-plugin/tests/tt/test_request_isolation.py:20`, `:74`). Admission order therefore changes their random streams even when logits and parameters are identical.

This difference reaches the device: `models/common/sampling/tt_sampling.py:1246` passes the per-slot seed tensor to `ttnn.manual_seed` immediately before sampling. The manual-seed kernel calls `rand_tile_init(seed)` for non-sentinel values (`ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/manual_seed_receive_all_data.cpp:30`); the sampling kernel consumes `rand_tile` (`ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp:27`). This is a different RNG input, not a numerical-precision hypothesis.

Host adjudication extracted the actual `SamplingArgs`, `SeedManager` and hash definitions with Python AST and replaced only device-library names with inert stubs. No Torch or TTNN library was imported. The result is `artifacts/serving_seed_host_adjudication.json`:

| Configuration | A then B: A seed | B then A: A seed |
| --- | ---: | ---: |
| Actual default, seed 42 | 275414, salt 0 | 62798, salt 1 |
| Salting disabled, seed 42 | 275414, salt 0 | 275414, salt 0 |

Counter 1 also changes under the default and remains 526445 with salting disabled. Assertions passed. This proves an order-dependent seed stream; the exact text failure remains a runtime confirmation.

Smallest intervention: add `salt_duplicate_seeds: bool = False` to Qwen's `SamplingArgs`, preserving shared demo defaults. The parent applied this independently after reviewing the host result. This is an existing Qwen/common-sampler integration issue, not introduced by the staged local-TopK or prefill-logit assembly edits. The common manager's own explanation at `:797` explicitly distinguishes independent vLLM requests from demos that replicate one seed across completions.

## 2. Proven: the adapter claims decode contract v1 but ignores its commands

`tt/generator_vllm.py:62` advertises `decode_input_update_contract = 1`. The canonical plugin therefore sends `reload_inputs`, `reload_page_table`, `reload_sampling_params`, and `reset_sampling_state` (`/tmp/qwen-ci-vllm-plugin/src/vllm_tt_plugin/async_decode.py:825`). It sends legacy `reset_batch` only for contract versions below 1 (`:834`).

At investigation start, Qwen `decode_forward` accepted only legacy `reset_batch=False` and swallowed all four new commands in `**_` (`tt/generator_vllm.py:330`). It called `apply_decode_state(reset_batch=reset_batch, refresh_sampling_params=sampling_changed)` (`:376`). Thus the plugin's request to restore prompt/output history was silently ignored. This matters because every prefill resets shared penalty prompt/output state (`models/common/sampling/generator.py:211`), and authoritative scheduler history is applied only when `reset_batch` is true (`:262`). Mixed penalty requests are exactly a consumer of that history.

The independent host proof invoked the extracted real Qwen decode method with a fake generator, stopping at the actual `apply_decode_state` boundary. With all relevant v1 transition commands true and unchanged sampling metadata it observed `reset_batch=False, refresh_sampling_params=False`. With legacy `reset_batch=True` it observed history reset true. See `artifacts/serving_reload_host_adjudication.json`; all assertions passed.

There is a second consequence of the same contract violation. The plugin permits intentionally stale host tokens/positions during overlap; only its reload commands establish authority (`model_input.py:24`). The existing adapter realigned seed counters to host positions on every decode (`tt/generator_vllm.py:397`), so a delayed host position can repeat a per-token seed. It also inferred recapture from page-table equality and local state instead of honoring the plugin's command. Those are consequences of the same missing contract implementation and should be repaired at that boundary, not patched as unrelated RNG defects.

Smallest complete intervention: explicitly consume the four v1 commands; restore penalty history only on `reset_sampling_state`; refresh parameters when requested; reload model inputs/align counters only on authoritative `reload_inputs`; update page tables without token reload on `reload_page_table`; keep active slots and advancing seed counters across steady overlap. Preserve legacy behavior when commands are omitted by direct adapter callers. A one-line mapping of `reset_sampling_state` to `reset_batch` alone leaves stale-host seed realignment unfixed.

## 3. Capability and test gaps

- The mixed test requests `top_k=100`, while Qwen advertises `max_device_top_k=32` (`tt/generator_vllm.py:68`) and configures `max_top_k=32` (`tt/generator.py:32`). The canonical plugin's `check_perform_device_sampling` does not inspect that capability or top-k (`model_runner.py:1920`). `format_sampling_params` silently clamps values above 32 (`models/common/sampling/generator.py:619`). Therefore the test exercises reproducibility of the clipped distribution, not top-100 semantics. This does not explain order dependence by itself.
- `test_min_p` explicitly checks only that requests return output (`tests/tt/test_host_only_params.py:13`, `:29`). Min-p routes through host sampling (`model_runner.py:1932`); its pass cannot validate device RNG or penalty bookkeeping.
- All-vocabulary logprob testing skips on an unsupported server response (`tests/tt/test_logprobs.py:84`, `:109`). A skip is no logprob correctness evidence.
- Top-k=1 is forced to `top_p=0` in plugin input preparation (`input_batch.py:375`), and greedy results do not depend on the seed. Its pass is consistent with both identified defects.

## Other candidates checked

- Compact prefill seeds are intentionally not scattered twice: `_format_slot_sampling` excludes the seed field, and `reset_seed` consumes request-order seeds plus explicit physical slots. No coordinate mismatch found at that boundary.
- The staged per-request assembly keeps rank-local padded vocabulary shards, slices each physical slot, and concatenates in slot order (`tt/generator.py:221`). The old path gathered/trimmed to host then replicated the entire vocabulary. The staged fix repairs that TP-width mismatch; no source evidence makes it the cause of the seed-42 order dependence.
- The staged TopK chunk path pads to 65536, splits into two 32768-wide local chunks, preserves physical chunk offsets, then reduces to 32 local candidates before gathering. No direct seed or request-identity change occurs there. A kernel issue remains possible in principle but ranks below the demonstrated host state defects.
- Prefill resets model-owned linear state for assigned slots before narrowing execution (`tt/generator_vllm.py:290`). The lower recurrence's validated numerical quality is not evidence about sampler lifecycle, but there is no demonstrated recurrence defect needed to explain this failure.
- Trace setup performs an eager sampling warmup, so restored penalty history must survive that execution too. The repair restores authoritative history after setup, before the real replay; otherwise the warmup token is counted as generated output (`tt/generator.py:485`, `:494`; `models/common/sampling/generator.py:333`).

## Follow-up and acceptance

1. Keep the seed-salting fix and a host regression proving order-independent seeds with the actual Qwen arguments.
2. Implement the explicit reload contract with host fake-generator tests for transition/history restoration, unchanged sampling metadata after prefill, steady overlap with stale host positions, page-table-only updates, and legacy direct calls.
3. Restart once and rerun the original canonical mixed-parameter test unchanged. Also retain exact prompt, seed, penalty values, request order and all outputs so a future failure identifies the request rather than only temperature/seed.
4. If it still fails, compare max_tokens=1 with longer generation, unique seeds with duplicate seeds, and penalties disabled with enabled. A first-token failure narrows to prefill/logits/seed assignment; a later-only failure points to decode history/counter/slot transitions. These are proposed experiments, not observations.

The repo AutoDebug runner was launched in `/tmp/qwen38-serving-seed-autodebug` with source-only constraints. Its nested workspace-write sandbox repeatedly failed shell reads with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. At the parent's request it was stopped with SIGINT and exited; no report was produced. Tool outputs are preserved in `/tmp/qwen38-serving-seed-autodebug/runner_tool_outputs.log`, with compact status in `artifacts/serving_autodebug_runner_status.json`. This report uses independently checked source and the two successful host AST experiments; no findings are borrowed from an unavailable nested report.

## Applied repair and host validation

After reviewing the two host experiments, the parent authorized the Qwen-only repair. The parent set `SamplingArgs.salt_duplicate_seeds=False`. This investigation changed `tt/generator_vllm.py` to consume explicit reload commands, retain active slots and seed counters during overlap, update only page tables when commanded, and restore scheduler-owned penalty history after trace warmup. Calls without explicit commands retain the legacy parameter cache and setup decisions.

`tests/test_vllm_decode_reload.py` uses the actual AST-loaded adapter method, generator capture method, seed manager, hash and Qwen argument class with fake device boundaries. Nine tests passed, covering equal-parameter refresh after a transition, warmup history restoration, stale-host overlap, page-table-only update, slot remap/new prefill, legacy calls, invalid commands, duplicate seed admission order, and recapture without an authoritative history reset. Stale host inputs raise if consumed, making the overlap check sensitive to the former unconditional position read.

Checks run:

```text
python3 -m unittest discover -s models/autoports/qwen_qwen3_6_27b/tests -p test_vllm_decode_reload.py -v
python_env/bin/python -m black --check models/autoports/qwen_qwen3_6_27b/tt/generator_vllm.py models/autoports/qwen_qwen3_6_27b/tests/test_vllm_decode_reload.py
python3 -m py_compile models/autoports/qwen_qwen3_6_27b/tt/generator_vllm.py models/autoports/qwen_qwen3_6_27b/tests/test_vllm_decode_reload.py
git diff --check -- models/autoports/qwen_qwen3_6_27b/tt/generator_vllm.py models/autoports/qwen_qwen3_6_27b/tests/test_vllm_decode_reload.py
python_env/bin/pre-commit run --files models/autoports/qwen_qwen3_6_27b/tt/generator.py models/autoports/qwen_qwen3_6_27b/tt/generator_vllm.py models/autoports/qwen_qwen3_6_27b/tests/test_vllm_decode_reload.py
```

All passed. The installed Python environment has no standalone `isort` module, but the pre-commit environment's isort hook passed. No build is required for these Python/documentation changes. The parent's new `readiness_vllm/sampling_tests.log` records the original mixed-parameter isolation test passing, along with greedy and min-p smoke: **3 passed, 1 skipped in 30.29 seconds**. This confirms the combined seed/reload repair on the previously failing canonical serving path; it does not separately attribute the text failure to one of the two proven host defects.

Independent review reproduced the valid abstract combination `reload_inputs=True, reset_sampling_state=False` and legacy page-table-triggered recapture losing history through warmup. The final generator API adds `preserve_sampling_history=True` to setup/capture. With active penalties it clones the three persistent output-history buffers, performs the original counting warmup, restores and deallocates the snapshots before beginning model trace capture. Keeping counting enabled warms the scatter/count operations before a model trace owns device addresses. The adapter disables preservation only when it will immediately restore authoritative scheduler history after setup. Thus the canonical c9 plugin transition path adds no history snapshots; public/legacy recapture preserves history without requiring host output reconstruction. The actual capture-method host regression checks all three buffers and confirms restore/deallocation precede trace capture.

Snapshots temporarily total approximately 48 MB per device for the fixed 32-row, TP4 vocabulary layout when preservation is requested with active penalties (two vocabulary-sharded int32 buffers plus one replicated int32 buffer). The reduced device probe below verified this branch. The canonical server was loaded before this final edge-path extension; its normal transition behavior is unchanged because it already restores authoritative history and skips these snapshots.

## Separate follow-up: unseeded sampled text quality

The new shared chat suite (`readiness_vllm/vllm_qualitative_outputs.json`) exhibits a different failure boundary after request isolation passed. Greedy outputs are coherent. Several unseeded `temperature=0.7, top_p=0.9` outputs contain malformed words, missing words and broken code: prompt 2's story contains `glmed` and `Elia's eyes leened`; prompt 1 loses nouns and merges `Examplesupervised`; prompt 5 begins `Herepython`, assigns `seq`, then refers to undefined `fib`. These are concrete quality concerns, not explained by normal truncation at 256 tokens. The French final translation remains correct, so this is not a claim that every sampled output is unusable.

Prompt-format metadata at `artifacts/qualitative_prompt_format.json` pins Qwen3.8 revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, its chat template, rendered prompts and token IDs. The shared runner uses `/v1/chat/completions` and alternates greedy and sampled requests sequentially (`models/common/readiness_check/run_vllm_server.py:420`). This is correctly formatted chat evidence, unlike earlier raw continuation diagnostics. No matched HF stochastic control has yet established whether the remaining difference belongs to the checkpoint, quantized logits, device sampling, or token feedback. The quality verdict remains **more work needed**.

Ranked controls and hypotheses:

1. **Device sampling versus host sampling on otherwise identical parameters.** Use the shared supervised/unsupervised prompt, `max_tokens=160, temperature=0.7, top_p=0.9, top_k=20, seed=42`; repeat it twice. Repeat with `logprobs=True, top_logprobs=1`, which forces host sampling on TP4 (`/tmp/qwen-ci-vllm-plugin/src/vllm_tt_plugin/model_runner.py:1949`) without deliberately changing the sampling distribution. This is a cleaner boundary control than `min_p=0.01`, which also changes the token distribution. A host-only quality pass narrows the fault to the device-sampling/feedback path; it does not by itself distinguish them.
2. **Unseeded internal sampling trace versus explicit-seed eager sampling.** Repeat the same device request with seed omitted. `SamplingGenerator.sample` uses its internal trace only when no explicit request seed is active (`models/common/sampling/generator.py:461`). Explicit seeds also reinitialize the RNG per token, whereas the unseeded manager transitions to `MAX_UINT32` and then stops writing seeds (`:1097`); the manual-seed kernel skips RNG initialization for that sentinel. Thus a seed-42 quality pass changes both RNG initialization and internal sampling-trace selection. Neither should be declared the sole cause without a second control.
3. **Cross-rank token agreement is a required TP4 invariant.** Each TP rank receives gathered candidate logits, independently samples into its local persistent `_trace_token`, and feeds that token into the next model replay (`tt/generator.py`, `_capture_token_out_trace` and `token_out_decode_step`). The host adapter emits only the first rank's token vector (`tt/generator_vllm.py`, `process_decode_output_host`). Compare all four rank-local sampled token vectors at every stochastic step, including unseeded trace replay and explicit-seed eager sampling. Any disagreement is a decisive feedback defect regardless of linguistic quality. This can be added to a small four-layer device probe without a full-model load.
4. **Sampler distribution/candidate pairing.** If all ranks agree but device quality is uniquely bad, use identical saved BF16 logits and exactly `top_k=20, top_p=0.9, temperature=0.7` for TT and Torch. First compare local/global candidate `(token_id, value)` pairs; then compare empirical rank frequencies/nucleus membership over a fixed seed set. Cover logits dominated by each of the four vocabulary shards and both 32768-wide local TopK chunks. A frequency test is preferable to demanding bit-identical Torch/TT RNG outputs. This isolates values/indices, temperature and top-p math from the model and serving scheduler. No current source fact proves those stages wrong.
5. **Readback ordering is lower priority without contrary evidence.** The adapter enqueues `.cpu(blocking=False)` then records a CQ0 event; the plugin synchronizes that event before processing host output (`generator_vllm.py`, `read_decode_output`; plugin `async_decode.py:880`, `:915`). That visible ordering weakens a generic 'async read races token overwrite' story. A direct sampler/generator control remains smaller than another server rebuild solely to disable overlap.

One attractive explanation was specifically challenged: an older Python comment says intervening SFPU work can disturb PRNG/LREG state, but the current Blackhole `ckernel_sfpu_rand.h` reconstructs its lane salt and reloads mutable LREGs on every call (`make_lane_salt`, `rand`). Therefore LREG clobber alone is **not proven** to explain unseeded rank divergence. The per-rank token comparison, rather than that comment, must adjudicate the invariant.

The parent subsequently ran the six-request HTTP matrix, preserved in `artifacts/serving_quality_controls.json`. Both unseeded device requests have malformed phrases (`you give examples computer examples that correct answers`, `structureings`). Both seeded device requests are coherent and byte-identical. Both seeded host-logprob requests are also coherent and byte-identical within that mode; their text differs from the device-mode pair, as different RNG implementations can. These matched-format controls substantially weaken a general model-graph or tokenizer explanation and support continuing at the unseeded/internal-trace sampling boundary.

`validate_penalty_recapture.py --sampling-consistency-steps 16` includes a four-case device control after the history test: unseeded/traced, unseeded/eager, seed42/traced and seed42/eager. It keeps B32 allocation and four real model layers, with one active prompt. A test-local override of the sampler's trace-selection predicate isolates trace choice while leaving real seed registration, counters and device seed uploads intact. Every step records all four rank-local active token IDs, all-rank seed-vector equality and the actual explicit-seed-active flag.

## Reduced device result

After the parent shut down serving and handed over exclusive device ownership, the combined probe completed successfully under a 600-second timeout, without watcher/profiler. `artifacts/penalty_recapture_b32.json` and `penalty_recapture_b32.log` preserve results and runtime configuration. The run used pinned Qwen3.8, native flat prefill/fused convolution, the selected precision policy, four layers, B32, context 128 and TP4. Device close completed at 2026-09-11 14:46:52 UTC.

- All three buffers were nonzero on every rank before recapture. Setup produced zero mismatches for `output_mask`, `output_counts`, and `output_counts_gathered` on all four devices.
- One actual model/sampler trace replay added exactly the sampled token to each of 32 rows; expected masks, sharded counts and replicated counts all matched exactly.
- All four rank-local seed vectors and active token IDs agreed in all 64 decode observations across the 2×2 control. Seed42 traced and eager token sequences were also identical to each other.

This closes the new penalty-preservation device branch. It **does not reproduce** the full-64-layer unseeded text defect, and therefore refutes any claim that rank divergence or a generic sampling-trace defect was established by the smaller probe. The next decisive observation must preserve the full failing execution envelope or inspect exact candidate distributions/feedback at the first malformed continuation; the successful reduced control must not be promoted to a full-model sampled-quality pass.

## Full-model reproduction: unseeded rank divergence

The subsequent full64/B32/TP4 probe reproduced the malformed output with the exact saved supervised/unsupervised chat prompt (67 logical tokens, physical prefill128), context512, native prefill and the selected precision policy. It generated up to160 decode tokens per arm, stopping at EOS, and synchronized/read every rank after each step. This removes serving overlap from the reproduction. Evidence is in `artifacts/sampling_rank_full64.json` and `sampling_rank_full64.log`; the device closed normally at 2026-09-11 14:56:17 UTC. Exit1 is the probe's functional rank-agreement assertion, not a device failure.

| Case | Decode steps | Rank-divergent steps | Text |
| --- | ---: | ---: | --- |
| Unseeded, traced sampling | 160 | 36 | Malformed phrases and missing words |
| Unseeded, eager sampling | 137 | 33 | Malformed phrases and duplicated words |
| Seed42, traced sampling | 148 | 0 | Coherent |
| Seed42, eager sampling | 148 | 0 | Coherent; token-identical to seeded trace |

Every arm's first prefill token agreed across ranks. Both unseeded arms first diverged at zero-based decode step1, while all four device seed vectors were identical and contained the `4294967295` skip sentinel. The active token IDs were `[4087,310,310,4087]` for traced sampling and `[310,4087,4087,310]` for eager sampling. Identical seed tensors therefore do not establish identical live rank-local RNG state. The malformed unseeded responses include `you’re the computer examples with are have correct answers` and `being told the the groups mean`; both fixed-seed arms produced the same coherent explanation.

This establishes a full-model unseeded TP feedback defect and excludes internal sampling trace selection or asynchronous host readback as necessary causes. The precise lower-level operation that makes rank-local RNG state differ is not yet identified. In particular, this evidence does not override the current Blackhole rand implementation's documented LREG reconstruction. The smallest next intervention is to refresh a replicated non-sentinel unseeded seed vector immediately before every sampled token, preserving unseeded trace selection, and rerun the same full-model control before changing production behavior.

## Isolated reseeding intervention and production policy

The intervention kept the full64 model, prompt, precision and trace/eager choices fixed and set only the test manager's reset flag before each unseeded `get_new_values` call. That reused the existing bounded entropy-seed upload path without assigning an explicit request seed. Both previously failing arms passed: traced sampling produced a coherent 135-token response (134 decode steps), eager sampling a coherent 142-token response (141 decode steps), with zero token or seed-vector rank mismatches. Every step retained `actual_explicit_seed_active=False`. Both responses ended at EOS. See `artifacts/sampling_rank_full64_refresh.json` and `sampling_rank_full64_refresh.log`; rc0, normal device close at 2026-09-11 15:02:06 UTC.

After this isolated result the parent authorized the smallest configurable repair. `SamplingGenerator` forwards optional `reseed_unseeded_each_step` to `SeedManager`; its shared default is false. Qwen's `SamplingArgs` opts in. With that policy, the all-unseeded branch sends fresh ordinary per-user device seeds on every `get_new_values` call instead of entering the skip state. Prefill already uses the same method, so it is covered. Explicit and mixed batches retain their existing per-step seed generation, and the explicit-seed flag/trace-selection predicate is unchanged. The change adds the existing 32-entry seed upload to unseeded steps and no host token readback; serving overhead still requires measurement.

Three additional host tests exercise the real `SamplingGenerator` constructor, `apply_prefill_state`, Qwen argument class and `SeedManager` with only device boundaries replaced. They verify Qwen's fresh prefill/decode seed uploads without explicit-seed activation, the unchanged shared default init/skip/steady sequence, and identical explicit/mixed streams plus transition back to unseeded. All 12 host tests, targeted pre-commit hooks, syntax checks and `git diff --check` passed. Full-model controls using the production policy without the test override and the final HTTP suite remain required before serving readiness can pass.

## Final production-policy device validation

The final full64/B32/TP4 run used the production policy without `--refresh-unseeded-each-step`. All four arms passed across 574 decode observations: unseeded traced 131, unseeded eager 147, seed42 traced 148 and seed42 eager 148. Every sampled token and seed vector agreed across all ranks, all four responses were coherent and ended at EOS, and unseeded requests retained `actual_explicit_seed_active=False`. Both seeded sequences were identical to each other **and to the pre-policy baseline**, directly confirming that the repair preserved those explicit streams. The forced seeded-trace arm remains an experimental control; normal explicit requests still use their existing eager path.

Evidence: `artifacts/sampling_rank_full64_production.json`, `sampling_rank_full64_production.log`, and the exact source hashes in `artifacts/sampling_rank_full64_production_source.json`. The run exited 0 and all devices closed normally at 2026-09-11 15:13:48 UTC. Hardware ownership was then returned to the parent for final serving validation.

The measured host duration of unseeded `get_new_values` was median 0.112/0.125 ms and p95 0.187/0.193 ms for traced/eager arms. This measures seed generation/upload submission on the host, not isolated device latency or end-to-end serving overhead. The final serving benchmarks remain authoritative. Warm synchronized prefill was 321–332 ms, prefill sampling about 2.0–2.1 ms, and decode setup 506–512 ms in this B32/context512 envelope; first-arm cold work was higher.

The optional empty `single_slot_prefill_view(0)` control took 20.20, 20.20 and 20.14 ms after one warmup. All 448 before/after rank-local hashes matched for the tensors that the context manager touches (linear conv/recurrent caches and full-attention batch indices). This rules out attributing the entire roughly 200 ms generator-versus-serving prefill gap to the slot-view roundtrip alone. No optimization or trace-reuse experiment was integrated into this final graph.

The isolated reproduction, single-policy intervention, host regressions and final production device validation close the unseeded TP feedback repair. The parent must still validate the final HTTP shared qualitative suite, unchanged canonical sampling tests and serving benchmarks on this source before declaring serving readiness complete.
