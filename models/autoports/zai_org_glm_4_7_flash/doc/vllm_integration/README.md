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
| primary-profile aggregate output throughput | 21.371 tok/s | `vllm_benchmark.json` (batch-1, so this is the same rate as the decode t/s/u above, not an independent throughput figure) |

Raw: `doc/vllm_integration/readiness_vllm_stage09/vllm_result.json`,
`vllm_ci_serving_result.json`. Commands are recorded verbatim in each summary's
`command_string`.

**Where this stage's artifacts live.** They were produced into
`models/autoports/zai_org_glm_4_7_flash/readiness_vllm/`, and every path in this
report originally named that directory. The optimized-vLLM stage (stage 10) then
ran its own `run_vllm_server` arms into the same place, so this stage's copies
were preserved at `doc/vllm_integration/readiness_vllm_stage09/` first and every
path here now points there. The live `readiness_vllm/` holds stage 10's
after-arm evidence, whose numbers are different by design; see
`doc/optimized_vllm/README.md`.

Server command that produced the serve/sampling(smoke)/qualitative/benchmark
evidence in one pass (no `--stages`, so the default
`serve,sampling,qualitative,benchmark` pipeline ran):

```
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/zai_org_glm_4_7_flash \
  --hf-model zai-org/GLM-4.7-Flash \
  --mesh-device N150 --max-num-seqs 32 --max-model-len 202752 \
  --sampling-profile smoke \
  --tt-config '{"trace_region_size": 350000000}'
```

The recorded, not-gated `--sampling-profile full` run (`sampling_tests_full_RECORD.log`)
was a follow-up attach to that same live server (that run's `server.log`, which
was never committed, showed a single continuous `APIServer` session spanning
both sampling runs, 19:20-19:37), using
the runner's documented attach-to-a-running-server pattern
(`--stages sampling --sampling-profile full --server-url http://localhost:8000`
plus the same `--model-dir`/`--hf-model`); the exact flags for that specific
invocation were not preserved verbatim in this stage's own logs, which is
recorded as a provenance gap under Known Limitations rather than reconstructed
with false confidence.

Boot at the full 202752 context is clean: **zero** `TT_FATAL` / OOM / engine-core
failures, after the VS-006 prefill-chunk cap.

**Non-aligned prompt length through serving:** all six qualitative prompts
were served through the live server above with real (non-synthetic) tokenized
lengths of 13, 20, 28, 14, 19, and 15 tokens after chat-template rendering
(verified by re-tokenizing each prompt with the model's own tokenizer) --
none a multiple of the model's internal tile (32), paged-cache block (64), or
this stage's prefill-chunk cap (1024). All six completed successfully with
coherent output (see Qualitative below), so this is direct, through-serving
non-aligned-length evidence, not only the reduced-model inner-loop tests in
`tests/test_generator_vllm_adapter.py` (which additionally cover larger
non-aligned lengths -- 37, 65, 137 tokens -- but are explicitly *not* final
serving evidence per `$vllm-integration`'s minimum-surface-loop guidance).

**Post-evidence-collection correctness fix (VS-011, found by `$stage-review`)
-- none of the numbers above are affected:** `allocate_kv_cache` sized this
adapter's per-request page-table width as `num_blocks // max_batch_size` (an
equal-share quota), instead of `cdiv(max_seq_len, block_size)` (the actual
bound on how wide any one request's block list can legitimately be in a
shared vLLM pool). At this stage's measured `num_blocks=7362`, that silently
capped every request's addressable context at 230 blocks = **14,720 tokens**
while still advertising `max_model_len=202752` -- a real, un-evidenced
capability reduction, not a hard physical limit. Fixed to derive the width
from `max_seq_len` alone (independent of `num_blocks`), and
`_write_page_table_rows` now raises rather than truncates if a table is ever
wider than that. Every measured request above (128-token benchmark,
100-token burst, 13-28-token qualitative prompts) was always far below both
the old wrong cap and the corrected one, so nothing in Headline changes. The
fix is proven by a new hardware-verified regression test
(`test_blocks_per_user_is_max_seq_len_derived_not_pool_derived` in
`tests/test_generator_vllm_adapter.py`) using a deliberately non-equal-share
pool size, but **has not been re-verified through a live vLLM server request
above the old 14,720-token cap** -- doing so would be a new hardware serving
run, out of this review round's budget. See work log VS-011.

## The serving overhead worth acting on

The full model's own traced token-out decode is **22.994 ms/token (43.49 t/s/u)**
(`doc/full_model/README.md`). Through vLLM it is **45.0 ms/token (22.2 t/s/u)**.

That is **~22 ms/token, roughly 2x** on an otherwise identical traced
model+sampling replay. TTFT moves the other way (273.8 ms served vs 334.2 ms
standalone), so prefill is not the cost. This is the single largest optimisation
target for the optimized-vLLM stage, and it is stated here as a lower-bound
comparison exactly as the stage requires, not as a model limitation.

> **Corrected by the optimized-vLLM stage (stage 10).** Calling this "vLLM-path
> overhead" was wrong, and reading it that way would have sent the next stage
> hunting for adapter and engine cost that does not exist. Driving this exact
> adapter with the same async split and no vLLM engine at all measures 45.208
> ms/token (doc/optimized_vllm/adapter_decode_floor_before.json), against 45.0
> served: the engine contributes nothing measurable. The
> whole gap was the model doing different work -- `build_generator` defaults to
> `max_batch_size=1`, so the full-model harness builds a one-physical-row decoder
> that takes the compact indexed MoE path, while the serving adapter builds 32
> rows and took the union path. Stage 10 closed most of it in the model
> (45.218 -> 29.496 ms/token served) and quantified the rest. See
> `doc/optimized_vllm/README.md` and its work log OV-001.

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
duplication 0.0-0.0273 (critical threshold 0.10), trigram-loop fraction
0.027-0.075 (advisory threshold 0.50). The visible reasoning scaffold
("1. **Analyze the Request:**") is this model's own style, not a decode
defect -- confirmed against `doc/full_model/qualitative/`'s HF-control and
full-model-TT completions on the identical six prompts, which show the same
scaffold and the same per-prompt voice.

## Known limitations

**1. Sampling status is `smoke-gated`, accepted by the project owner.**
The `$vllm-integration` skill allows recording the final status as `smoke-gated`
rather than the full profile when the full profile is impractical, with owner
acceptance, which was given explicitly. The smoke profile passes at spec; the
full profile does not, for the reason below. This is not presented as equivalent
to the full sampling gate.

**2. All 11 full-profile failures are determinism breaks at full (32-row)
occupancy, filed upstream as
[tenstorrent/tt-metal#55408](https://github.com/tenstorrent/tt-metal/issues/55408);
the exact mechanism is not fully pinned down, and this section was corrected
by `$stage-review` after the first version overstated it.**

What the **committed** `sampling_tests_full_RECORD.log` actually shows,
re-derived directly from that log rather than restated from memory:

* **Not "greedy-only, seeded incidental."** Two of the eleven
  (`test_seeding`, `test_topk[32]`) fail on a greedy-subset-of-a-mixed-batch
  assertion. The other nine (`test_same_seeds_reproduce_across_batches`,
  `test_specific_seed_reproducible[42]`/`[999]`, both
  `test_uniform_seed_deterministic[32-*]`, `test_mixed_params_batch`, all
  three `*_penalty_mixed_batch`) are genuine seeded-reproducibility or
  mixed-parameter assertions at `temperature=1.0` -- there is no greedy row in
  those batches at all (verified against the test sources in
  `vllm-tt-plugin/tests/tt/test_seeding_and_variety.py` and
  `test_tt_penalties.py`). The original wording here conflated the two.
* **Full occupancy (32 concurrent rows) is necessary but not sufficient.**
  Every failure in the committed log is at `batch_size=32`
  (`test_uniform_seed_deterministic[32-1]`/`[32-0]` FAIL,
  `[10-*]`/`[1-*]` PASS; `test_topk[32]` FAILS, `[15]`/`[19]` PASS). But it is
  not a clean threshold either: `test_specific_seed_reproducible` runs at a
  fixed batch size for all four of its `seed` parametrizations and still
  alternates FAIL(42)/PASS(123)/FAIL(999)/PASS(0) -- so whatever the real
  discriminator is, it is not "batch==32" alone, and it is not simply "runs
  after the first failure poisons everything forever" either (passing runs
  are interleaved between failing ones). This stage did not re-derive the
  precise discriminator, and does not claim to.
* Every failing test **passes when run alone against a freshly started
  server** (established on hardware, this run).
* Nine hypotheses were eliminated on hardware, including #48222 (`ttnn.sampling`
  matched `torch.argmax` on 256/256 rows at this model's shapes) and #50512
  (multi-device TP only) -- see work log VS-009 for each experiment.
* The work log's own "remaining suspect" is code this stage added:
  `TTSampling.reset_params` rewrites all 32 sampling lanes and ignores
  `empty_slots`, so an admitting request's prefill can overwrite a
  concurrently-decoding request's `k`/`p`/`temp` until the next
  `reset_batch` repairs it. This has **not** been isolated with an A/B (revert
  the lane-broadcast, re-run `[32-*]` vs `[10-*]` on a fresh server) -- that
  is the next concrete step for whoever picks this up, and until it runs, the
  upstream-vs-adapter attribution is unconfirmed either way.
* The claims that this predates this stage's own changes and that the
  `tt_transformers` reference (SmolLM2-135M) fails the same canary at baseline
  **worse** are recorded in work log VS-009, but the specific logs that would
  let a reader re-derive them (the pre-VS-008 full-profile run, the SmolLM2
  cross-check run) were not committed alongside this stage's evidence -- see
  the provenance note below.

None of this affects single-request correctness on a freshly started server,
and none of it touches either runner-side gate (both pass on the committed
artifacts, re-verified independently by `$stage-review`). The smoke profile
-- the actually gated profile -- is unaffected and passes clean.

**3. Two warnings in `server.log` during warmup are disclosed, not fully
classified.** 1,777 occurrences of `ttnn.split: L1 budget exceeded ... DRAM
downgrade` (a performance fallback during prefill/MoE routing ops, consistent
with the already-disclosed ~2x vLLM-path decode overhead) and one
`Allocating device buffers is potentially unsafe due to the existence of an
active trace` warning at 19:23:08, during the `warmup_model_prefill`/
`warmup_model_decode` window. Qualitative and degenerate-output checks are
clean and the serving run completed normally, so there is no observed
correctness symptom; a `TT_METAL_TRACE_ALLOC_TRACKING=1` probe (the
model's own existing procedure for this exact hazard class, per
`GLM47FlashModel.prepare_cache_reset`'s comments) would be needed to fully
rule the second one out, and was not run for this stage (would be a new
hardware measurement, out of this review's and this stage's remaining
budget).

**4. Provenance gap:** the pre-VS-008 full-sampling-profile log and the
SmolLM2/`tt_transformers` baseline cross-check log that VS-009's "predates /
affects the reference worse" claims depend on were not committed under
`readiness_vllm/` or `probe/`. The claims are recorded in work log VS-009 as
prose only. Re-generating them was judged out of scope for this review round
(would be a new hardware evidence sweep); a future session that wants to
close this out should commit both logs rather than re-describe them.

**5. Not exercised:** KV-cache migration (never enabled for this model), and
multi-host/multi-rank serving (single chip by design).

## Files

* `tt/generator_vllm.py`, `tests/test_generator_vllm_adapter.py`
* `doc/vllm_integration/readiness_vllm_stage09/` — this stage's runner output:
  both sampling logs, qualitative outputs, both benchmark summaries and raw
  results. (Produced into `readiness_vllm/`; preserved here before stage 10
  overwrote that directory — see the note under Headline.) The raw vLLM
  `server.log` (729 KB) was kept on disk but not committed: it exceeds the
  repo's 500 KB file limit and is a debug log, not stage evidence.
* `work_log.md` — VS-001..VS-011, the full investigation including corrections
