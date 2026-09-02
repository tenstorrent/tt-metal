# GLM-4.7-Flash full-model stage: work log

Branch `ttmodelmanager/glm47-flash-probe`, starting commit `ba10cee4e60`
(optimized decoder). Target: one Blackhole p150-class chip, device 0, 1x1 mesh.
Host: AMD Ryzen 7 9700X (8c/16t), 249 GiB RAM. Board: `p300c` card, one chip
used; 31.5 GiB allocatable DRAM measured (FM-001), which is the p150-class
32 GB budget the goal specifies.

All commands below are literal and rerunnable from the repo root with
`./python_env/bin/python`.

---

## FM-001: device capacity baseline

Before designing anything, measure what "32 GB" actually means to the
allocator: 512 MiB DRAM tensors on device 0 until OOM.

```
allocated GiB 28.0
STOP at 32256 MiB = 31.5 GiB: Out of Memory ... bank size 4272341376 B (8 banks)
```

**31.5 GiB allocatable.** Every capacity claim in this stage is against that
number, not against the 32 GB nameplate.

## FM-002: terminal-path probe before writing the wrapper

`probe/full_model_head_probe.py`: embedding output rank, `plus_one` dtypes, and
the LM-head matmul geometry at vocab 154880, M = 1 tile.

| LM head config | bf8 | bf4 |
|---|---|---|
| wide-1D mcast, 110 cores, in0_block_w 2/4/8 | 871 / 881 / 897 us | 628 / 633 / 662 us |
| wide-1D mcast, 88 cores | 908 us | 667 us |
| wide-1D mcast, 64 cores | CB clash | 683 us |
| **default (no program config)** | **15310 us** | **14928 us** |

The default matmul config is a **17x** regression at this N, so the explicit
program config is mandatory, not a tuning nicety. 110 cores / `in0_block_w=4`
kept; bf8 kept over bf4 for LM-head accuracy (bf4 is a datatype-sweep
candidate, recorded, not taken here).

`ttnn.embedding` accepts a rank-4 index tensor and returns `[1, T, hidden]`,
so the decode token input can stay the rank-4 `[1, 1, 1, 32]` tensor
`ttnn.sampling` needs as a preallocated output. `ttnn.plus_one` works on int32
`[B]` and uint32 `[1, B]` / `[1, 1, 1, B]`, and `skip_negative_entries=True`
leaves a `-1` inactive-slot sentinel alone.

## FM-003: sampler selection

Read both common implementations before writing any token-out code
(`$full-model` requires the comparison). Full report is in `README.md`; the
decisive facts:

* `TTSampling.num_single_device_vocab_splits(154880)` returns **4** chunks of
  38720, each inside `ttnn.topk`'s 65536 practical width
  (`models/common/sampling/tt_sampling.py:80-92`).
* `Sampling1D` hardcodes a **2-way** single-device split
  (`models/common/modules/sampling/sampling_1d.py:570`, offsets at `:773-777`),
  which would hand `ttnn.topk` a 77440-wide input. The split factor is not
  configurable, and no in-tree 1x1 user or test exceeds vocab 128256.

`models/common/sampling` selected; `Sampling1D` rejected.

## FM-004: first wrapper, reduced probe

`tt/model.py` + `tt/generator.py` written against the optimized decoder, then
debugged on a **2-layer reduced probe** (HF layer 0 dense + layer 1 moe, real
embedding / final norm / LM head / sampler / paged-cache shapes), built in
5.8 s instead of 175 s:

```
python models/autoports/zai_org_glm_4_7_flash/tests/dev_full_model.py smoke
python models/autoports/zai_org_glm_4_7_flash/tests/dev_full_model.py pcc
python models/autoports/zai_org_glm_4_7_flash/tests/dev_full_model.py trace
```

`pcc` compares the reduced TTNN stack against an fp32 torch reference built
from the same two HF layers plus the real embedding/norm/LM head:
prefill logits PCC 0.99753, prefill top-5 100%, teacher-forced decode top-1
6/8 and top-5 8/8. (A 2-layer truncated stack has very flat logits, so top-1
flips are expected there; the 47-layer numbers are in `README.md`.)

## FM-005: full 47-layer build fits, and is slower than it should be

First all-layer run: weights 17.383 GiB + 202752-token cache 5.431 GiB =
22.814 GiB of 31.5 GiB. Correct output, but **42.2 ms/token**, against a
~24 ms expectation from the optimized-decoder per-layer numbers.

Bisected by rebuilding the 2-layer probe at two cache sizes:

| `max_seq_len` | 2-layer traced model decode |
|---|---|
| 4096 | 1.790 ms |
| 202752 | 2.573 ms |

+0.39 ms **per layer** purely from the advertised context. Two candidate ops:

1. `probe/decode_cache_scaling_probe.py`: `paged_update_cache` and
   `paged_flash_multi_latent_attention_decode` at cache sizes 4096 / 16384 /
   65536 / 202752 tokens, decode position 130, with both a full-width and a
   64-block page table:

   ```
         ctx  blocks  update us   flash us  flash(pt64) us
        4096      64        5.6       32.3            32.2
      202752    3168        6.2       32.4            32.5
   ```

   **Refuted**: neither op scales with the allocated cache or page-table size.

2. The RoPE cos/sin lookup. `RopeSetup.decode_mats` does
   `ttnn.embedding(rot_idxs, cos_matrix)` on a `[1, 1, max_pos, 64]` **TILE**
   table, twice per layer:

   ```
     max_pos  TILE tbl us    RM tbl us
        4096         24.9         15.5
       16384         28.1         15.5
       65536         73.7         15.6
      202752        209.5         16.0
   ```

   **Verified**: the TILE-table lookup scales with table height; a ROW_MAJOR
   table is flat. 94 lookups/step x 209.5 us = 19.7 ms/token, more than the
   whole decoder stack.

Fix (two independent parts, both value-preserving):

* every layer already shares one `RopeSetup` (`_rebind_rope`, which also saves
  2.4 GiB of byte-identical cos/sin copies at this context), so the lookup is
  hoisted to **one per decode step** in `GLM47FlashModel._decode_rope_lookup`
  and consumed by `SharedRopeDecoder._decode_rope_mats`;
* the hoisted lookup reads model-owned **ROW_MAJOR** copies of the same tables
  (+103 MiB) instead of the prefill TILE tables.

2 x 16 us instead of 94 x 209.5 us. Reduced-probe decode 2.573 -> 1.762 ms;
the generated token sequence is bit-identical before and after, and
`test_shared_rope_matches_per_layer_lookup` asserts the two tables return
equal values.

## FM-006: prefill program-shape explosion

Cold `generate()` on an unseen prompt length took 13.5 s and the readiness
teacher-forcing TTFT read 16.7 s, while the warmed TTFT was 289 ms. The decoder
pads prefill to the 64-token paged block, so **every distinct prompt length is
its own program set**, and the first request at a new length paid for building
them. (FM-011 later quantified this properly against a genuinely cold JIT
cache: `compile_cost.json`, 71.4 s to warm the five bucket shapes at
construction and +1338 ms on the first request at an un-warmed shape (both
figures re-measured in FM-015 with a synchronized timer). The
13.5 s here was one whole cold `generate()` - kernel build, first trace capture
and first prefill together - not compile alone.)

Fix: `GLM47FlashModel.prefill_physical_len` buckets the physical prefill
length to `(128, 256, 512, 1024, 2048)`, and a longer prompt becomes whole
2048-token chunks plus a bucketed tail, so the distinct chunk shapes are
`{2048} | buckets`, six in total, however long the prompt is. Padded positions
are never attended (prefill attention is causal; decode writes its own cache
row before reading it), and the logical prompt length is untouched: the
public API still accepts 1, 17, 63, 65, 129, 154, 1057, 2049, 2600 and slices
the logits back. `build_generator` then compiles the five single-chunk bucket
shapes at construction (`warmup_prefill`: 4.6 s warm cache, 71.4 s cold), so
no request pays for them.

First bucketing attempt rounded long prompts to whole chunks (3000 -> 4096).
Replaced with chunks + bucketed tail (3000 -> 3072, 2049 -> 2176), which is
strictly less padding for the same six shapes. Bit-identical outputs before
and after on the reduced probe.

## FM-006b: proving the bucket padding is invisible

Bucketing is only legitimate if a padded position can never influence a real
one. `tests/test_prefill_padding.py` (reduced 2-layer probe, so two models with
different bucketing fit at once; 11 passed in 43 s):

* **sharp test, shape held fixed**: same prompt, same physical prefill length,
  pad token 0 vs 12345 -> prefill logits **bit-identical**, max |delta| 0.0;
  and 24 generated tokens identical, so the pad rows the bucketed prefill wrote
  into the KV cache are never attended by a later decode step either;
* **cross-shape**, bucketed vs block-aligned at the same logical length
  (17/65/129/154/700/2049/2600): PCC 1.0 on most, 0.99998 at 129/154, with 0-2
  argmax disagreements per prompt, each a **1-3 bf16 ULP tie** between the same
  two candidates. A different physical length is a different matmul M and a
  different flash-prefill K extent, hence a different accumulation order; the
  bit-identical fixed-shape test above is what rules out leakage.

The first version of the cross-shape test asserted exact argmax equality and a
2e-3 *relative* gap bound; both were the wrong criterion (a 1-ULP bf16 tie at
|logit| 11.9 is a 0.5% relative gap). Replaced with an explicit bf16-ULP tie
classification. The free-running greedy cross-shape comparison was also
replaced by the pad-token-invariance decode test: a greedy chain amplifies one
tie flip into total divergence, so it could never have been a clean signal.

## FM-007: `ttnn.zeros(device=...)` is a host upload

`generate()` at prompt 128 / 128 tokens took 16.15 s wall while its own TTFT +
decode accounting summed to 3.2 s. The missing 13 s is `reset()`:

```
per-layer ttnn.zeros + copy: 293.1 ms/layer -> 13.78 s for 47
shared zeros + copy:           0.6 ms/layer ->  0.03 s for 47
in-place multiply(0):         39.5 ms/layer
ttnn.fill:                    46.3 ms/layer
```

`ttnn.zeros(..., device=...)` builds the tensor on the host and uploads it
(118 MiB per layer cache at ~0.4 GB/s). `reset_kv_cache` now keeps one
device-resident zero buffer and issues 47 device-to-device copies, and
`allocate_kv_cache` fills layers 1..46 from layer 0's cache instead of
building 47 host zero tensors (cache allocation 12.8 s -> 0.4 s). Same
generated tokens; per-prompt qualitative generation 16.0 s -> 3.2 s.

## FM-008: shared-harness fixes (one additive, one changes a shared default)

* `models/common/readiness_check/generate.py`: transformers 5.x returns a
  `BatchEncoding` from `apply_chat_template(tokenize=True)`, which the harness
  fed straight into `int()`
  (`ValueError: invalid literal for int() with base 10: 'input_ids'`). Added
  `_flatten_token_ids`, which also handles a batch dimension and torch tensors.
  Without this no chat-template reference can be generated on this transformers
  version.
* `models/common/readiness_check/mesh_device.py`: `open_readiness_mesh_device`
  opened the device with ttnn's defaults, i.e. `trace_region_size=0`, while
  `contract.py` *requires* the teacher-forcing runner to drive traced decode.
  Added `--trace-region-size` / `--l1-small-size` and threaded them through
  the three runners. **This one is not additive**: the new defaults (90 MB and
  32 KiB, matching the largest entry in
  `models/model_trace_region_sizes.yaml`) apply to every model that uses the
  shared harness, so any other model's readiness run now opens its device with
  a trace region and a non-zero L1 small allocation where it previously got
  neither. The direction is the safe one (the harness could not honour its own
  traced-decode requirement before, and a larger trace region cannot break a
  model that does not trace) but it is a shared-default change, not a
  model-local one, and it should be reviewed as such.

## FM-008b: greedy strategy benchmark

`$tt-enable-tracing` says force-argmax is a candidate to be measured, never a
default. `probe/greedy_sampler_probe.py` measures both `TTSampling` greedy
paths on a real `[1, 1, 32, 154880]` bf16 logits tensor:

```
split_topk_greedy  traced 1.108 ms  eager 1.129 ms  32/32 == torch argmax  out [1,1,1,32]
force_argmax       traced 1.084 ms  eager 1.060 ms  32/32 == torch argmax  out [1,1,1,32]
```

Force-argmax wins by 0.024 ms = 0.1% of the 23.0 ms token-out step, and is
rejected on trace behaviour, not speed: it is greedy-only, and flipping it
makes `reset_sampling_params` call `reset_trace()`, so a mixed
greedy/sampled workload would release and recapture the sampling trace on
every mode change. Note the run also corrects an earlier assumption: with a
preallocated rank-4 buffer, force-argmax *did* write `[1, 1, 1, 32]`
correctly, because `ttnn.argmax` does not validate the preallocated output
shape. The gemma4 writeback hazard is therefore latent here, not immediate,
and the README says so.

## FM-009: evidence runs

Reference (fresh; no GLM-4.7-Flash reference existed in this checkout):

```
python -m models.common.readiness_check.generate \
  --hf-model zai-org/GLM-4.7-Flash --prompt-source aime24 --chat-template \
  --gen-len 100 --top-k 100 \
  --output models/autoports/zai_org_glm_4_7_flash/readiness_aime24_chat.refpt
```

154 chat-template prompt tokens, 100 HF continuation tokens, top-100 per
position. Provenance in `readiness_aime24_chat.meta.json`.

Gates (each in its own process; the 17.4 GiB of weights means one full model
at a time):

```
python -m models.common.readiness_check.run_prefill_check   --model-dir models/autoports/zai_org_glm_4_7_flash --reference .../readiness_aime24_chat.refpt --mesh-device N150 --trace-region-size 350000000 --l1-small-size 32768
python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/zai_org_glm_4_7_flash --reference .../readiness_aime24_chat.refpt --mesh-device N150 --trace-region-size 350000000 --l1-small-size 32768
python -m models.common.readiness_check.run_autoregressive  --model-dir models/autoports/zai_org_glm_4_7_flash --hf-model zai-org/GLM-4.7-Flash --prompt-file doc/full_model/autoregressive_prompt_chat.txt --mesh-device N150 --trace-region-size 350000000 --l1-small-size 32768 --max-new-tokens 256
python models/common/readiness_check/check_degenerate_output.py --model-dir models/autoports/zai_org_glm_4_7_flash --missing-artifacts critical --scope autoregressive
python .agents/scripts/check_context_contract.py --model-dir models/autoports/zai_org_glm_4_7_flash --hf-model zai-org/GLM-4.7-Flash --stage full-model --require-contract

pytest models/autoports/zai_org_glm_4_7_flash/tests/test_full_model.py -x -q -s -p no:randomly
pytest models/autoports/zai_org_glm_4_7_flash/tests/test_full_model_perf.py -x -q -s -p no:randomly
GLM47_FM_BATCH=32 GLM47_FM_BATCH_SEQ=8192 pytest models/autoports/zai_org_glm_4_7_flash/tests/test_full_model_batch.py -x -q -s -p no:randomly
python models/autoports/zai_org_glm_4_7_flash/tests/run_qualitative_suite.py --max-new-tokens 128
python -m tracy -r -p -v -m pytest models/autoports/zai_org_glm_4_7_flash/tests/test_full_model_profile.py -q -s -p no:randomly
tt-perf-report --arch p150 <ops_csv> --start-signpost PERF_FM_<W> --end-signpost PERF_FM_<W>_END --csv doc/full_model/tracy/<w>_perf_report.csv
TT_METAL_WATCHER=2 python models/autoports/zai_org_glm_4_7_flash/tests/dev_full_model.py {smoke,trace,capacity}
```

`run_prefill_check` / `run_teacher_forcing` / `run_autoregressive` were rerun
on the final code after FM-005..FM-007 so the committed logs match the shipped
implementation.

## FM-010: anomaly ledger

| observed | evidence | resolution |
|---|---|---|
| full model 42.2 ms/token vs ~24 ms expectation | FM-005 | fixed: per-layer RoPE table lookup scaling with the advertised context |
| 16.7 s readiness TTFT | FM-006 | fixed: prefill bucketing + construction-time warmup; warmed TTFT 388.7 ms at prompt 128 |
| `generate()` wall 16.15 s vs 3.2 s of accounted work | FM-007 | fixed: `ttnn.zeros(device=...)` host upload in cache reset |
| model trace (21.75 ms) *below* the 23.03 ms layer-stack lower bound | `perf.json` | expected, not an error: the decoder-stage per-layer figure is a single-layer traced replay measured through its own harness, so its per-replay dispatch overhead is counted 47 times in the naive sum. The full model amortises it into one 47-layer trace. Both terminal costs (LM head, sampler) are measured separately and are on top of the stack figure. |
| `SparseMatmulDeviceOperation rows without numeric nnz` warning from `tt-perf-report` | `doc/full_model/tracy/*.txt` | expected: the report cannot know how many experts a sparse matmul activated, so it omits DRAM/FLOP utilisation for those rows. Row timings are unaffected. |
| `tt-perf-report` "Unclassified operation" warnings (TopK*, Sampling, ManualSeed, PlusOne, UntilizeCodegen) | same | cosmetic: those ops are not in the tool's category table, so they land in "other". Timings are reported. |
| `Profiler DRAM buffers were full, markers were dropped!` (295 lines in the final tracy run, all before the first signpost) | `logs/tracy_profile_run.log.gz` | controlled: they come from the un-flushed wall-clock loops that run *before* the signposted windows. Each signposted decode window drains the profiler every iteration and captures all 8 steps - the once-per-step LM head appears exactly 8 times (1264 / 1600 rows), and `perf_report_summary.json` records that `anchor_calls_in_window` so a truncated capture cannot be mistaken for a fast one. Before the fix the same run dropped 1100 markers and captured 17 of 32 steps, and the summary divided by the assumed 32 (FM-011 P1). |
| `ttnn.split: L1 budget exceeded ... DRAM downgrade` + `migrating L1 input (9912320 B)` | every run log | controlled and disclosed: the sampler-ready logits are produced in L1, so `TTSampling`'s first `ttnn.split` migrates them. Removing it by producing DRAM logits is measured 34 us *slower* end to end with identical tokens (`logits_memory_ab.json`), so L1 is kept and the fallback is named in the runtime fallback audit. `decode_logits_in_dram=True` reproduces the other arm. |
| `Allocating device buffers is potentially unsafe due to the existence of an active trace` (one line per process) | every run log; `trace_alloc.json` | **was mis-classified as controlled; measured and fixed in FM-016.** The earlier resolution read the single log line as "one unsafe allocation, at capture". Both halves were wrong: `allocator.cpp` suppresses repeats behind a `thread_local static bool`, so the line count carries no information, and `mesh_device.cpp` registers a trace as active at `end_mesh_trace` and keeps allocations unsafe until it is released, so the window is the whole life of the retained traces. Measured with `TT_METAL_TRACE_ALLOC_TRACKING=1`, which found the cache-reset zero buffer (a real hazard, fixed by allocating it before capture) and post-capture program compilation (fixed by `recapture_decode_traces`). All shipped paths now report zero unsafe buffers. |
| free-running TT and HF greedy diverge after 8-45 tokens | `doc/full_model/qualitative/` | expected for a 30.6B MoE with bf4 routed experts: the teacher-forced top-1 agreement is 85% per step, so a free-running greedy chain separates within tens of tokens. Both sides stay coherent, on-topic and same-language; the degeneracy check is clean. No control shows TT-specific wrongness. |
| `preds[:16]: [278, 77, 944, 312, 64, 501, 502, 503, ...]` in `logs/watcher_full_model.log` - an 11-step +1 token-id ramp, which reads like stale token feedback | that log | controlled: `tests/dev_full_model.py capacity` builds its prompt as `list(range(500, 500 + seq))`, i.e. raw token ids that happen to be an arithmetic ramp and decode to gibberish. Ids 501-511 decode to exactly `prompt[1:12]`, so the model is copying the prompt (induction), not repeating itself. Counter-controls: the coherent 256-token chat completion, the six coherent suite completions with adjacent duplication 0.000, teacher-forced top-5 1.000 over 100 positions, and `test_split_sampling_trace_feedback`. The synthetic prompt is a debug convenience, not evidence. |
| `generate()` measures a slightly *faster* ms/token than the isolated `decode_step_traced + read_decode_tokens` microbenchmark, although the loop does strictly more work | `perf.json`, `decode_position_scaling.json` | explained, and the direction is conservative because the README headline quotes the slower microbenchmark. It is decode position, not dispatch: `bench()` runs three 65-replay windows back to back, so the token-out window sits ~200 positions further into the cache than `generate`'s decode window, and `decode_position_scaling.json` shows the traced step growing with `cur_pos`. That accounts for roughly half the gap; the rest is inside the run-to-run spread. An earlier entry here blamed "the tighter dispatch pattern", which was self-contradictory: the tighter pattern is the slower number. |
| a repeated prefill of the same prompt looked systematically SLOWER than the first one after construction (218.0 then 4x 339.9 ms at prompt 128, 5388.3 then ~6469 ms at prompt 3000) | `compile_cost.json`, `compile_cost_warm.json` | **retracted in FM-015: measurement error, not a device effect.** `prefill_forward_last_logits_device` returns a *device* tensor and `ttnn.deallocate` does not block, so `timed_prefill` stopped its clock while the device was still draining: the first call returned early and every later call was device-bound behind it. With `ttnn.synchronize_device` bracketing the timer the ordering is the expected one at both prompt lengths (cold 7817.2 then 4x ~6475.2 ms at prompt 3000; 313.9 then 4x 313.8 ms at prompt 128), repeats are stable to 0.0-0.1%, and `first_minus_repeat_mean_ms` becomes a usable compile measurement (+1342.0 ms cold vs +3.8 ms warm at prompt 3000). Every prefill number reported before FM-015 was host enqueue time; the README table is rebuilt from the synchronized run. |
| prefill throughput 383-433 tokens/s at short prompts, 90.7 tokens/s at the full context | `perf.json`, `doc/full_model/tracy/prefill_perf_report.*` | disclosed, not fixed here. The two sparse expert matmuls are 48.4% of the reduced-profile prefill window (no bandwidth figure is claimed for them: `tt-perf-report` omits DRAM utilisation for sparse rows); the prefill sparse geometry was tuned for 1024-token chunks in the optimized-decoder stage and the flat prefill projections deliberately keep default configs below 10 M-tiles. That is optimized-full-model (stage 07) work; see README "Limitations". |

## FM-011: stage review round 1 (`more-work-needed`) and the fixes

An independent `$stage-review` subagent returned `more-work-needed` with one P1
and six P2 findings. Each is resolved below; nothing was argued away.

**P1: the reduced Tracy capture dropped half its markers.** The profiled run
logged 1100 x `Profiler DRAM buffers were full, markers were dropped!` and the
token-out CSV held 17 LM-head rows for a 32-iteration window (the LM head runs
exactly once per step). `perf_report_summary.json` divided by the *assumed* 32
iterations, so every `us_per_step` in it was ~2x low, and the README inherited
that (LM head "759-875 us", topk "311 us").
Fix: `DECODE_ITERS` 32 -> 8 with an explicit `ttnn.ReadDeviceProfiler` after
each iteration, and the wall clock moved ahead of the signposted windows so the
flushes cannot distort it. New `tests/summarize_perf_report.py` normalizes by
the number of LM-head calls actually captured, not by an assumed iteration
count, and records `anchor_calls_in_window` so a truncated capture is visible.
Recaptured (the committed capture, taken on the shipped L1-logits default):
8/8 steps in both decode windows, LM head x1.0/step at 873.1 us,
`TopkLargeIndicesDeviceOperation` x4.0/step at 665.0 us, device 1709.4 us/step
(model) and 2752.8 us/step (token-out) - now consistent with the 1.886 / 2.979 ms
wall clock in `perf_reduced_decode.json` instead of half of it. (An intermediate
recapture taken while the DRAM-logits arm was briefly the default read 917 /
665.8 / 1754.6 / 2789.8; it is superseded.) The independent reviewer's own CSV
segmentation of the pre-fix capture (4 topk calls, 664 us, 868 us LM head)
agrees with the recapture.

**P2: undisclosed `ttnn.split` L1 -> DRAM downgrade in the sampling graph.**
Every run logged `ttnn.split: L1 budget exceeded (need ~9945088 B, have
1248256 B for 4 chunks); DRAM downgrade` + `migrating L1 input (9912320 B)`.
`9912320 = 32 x 154880 x 2` is the sampler-ready logits tensor, produced in L1
by `lm_head_decode`. A/B measured (`logits_memory_ab.json`, reduced probe):
DRAM logits 1.813 / 2.937 ms (model / token-out, 0 warnings) vs L1 logits
1.768 / 2.903 ms, identical tokens. Removing the fallback is **34 us slower**,
so L1 is kept, `decode_logits_in_dram` exposes the other arm, and the fallback
is now named in the runtime fallback audit and the anomaly ledger instead of
being invisible. The prefill last-position logits were also moved onto the same
knob so the two paths agree.

**P2: unclassified `Allocating device buffers is potentially unsafe due to the
existence of an active trace`.** Bisected with flush markers: it fires exactly
once, inside `SamplingGenerator.capture_trace`, i.e. while the sampling trace
is being recorded with the model trace already captured. The allocations are
the sampling trace's own intermediates. Control added:
`test_traced_decode_matches_eager_decode` generates 16 tokens through the
captured traces and 16 through a fully eager model+sampler and asserts the
streams are identical - if a model-trace replay were clobbering a buffer the
sampling trace owns, they would part. Also covered by 98-replay bitwise
determinism and 100 teacher-forced steps at top-5 1.000.

**P2: README numbers with no artifact.** "8241 ms" and "warmed in 17 s" were
from runs whose logs had since been overwritten; the committed logs show
`prefill programs warmed in 4.6s`. Replaced with the logged values, and
`tests/measure_cold_compile.py` now produces `compile_cost.json` for the
warm- and cold-JIT-cache prefill compile cost so the prefill-compile
limitation has an artifact.

**P2: `ttnn.plus_one(rot_idxs)` had no inactive-row handling.** `cur_pos` used
`skip_negative_entries=True` but the separate uint32 RoPE index was
unconditionally incremented, so a permanently inactive slot's index grew without
bound and would eventually index past the cos/sin table. Fixed structurally: the
separate tensor is gone and `GLM47FlashModel.decode_rope_indices` derives the
index from `cur_pos` on device every step (`clamp -> uint32 -> [1, B]`).
Verified value-identical (same generated tokens before and after) and covered by
`test_decode_rope_index_derived_from_position` (positions 0 / 7 / 202751 / -1)
and a rewritten `test_batch_inactive_rows` that now runs the *traced* path and
asserts the derived index stays 0 for inactive slots across three steps.
`test_shared_rope_matches_per_layer_lookup` also grew from one index to eight
(0, 1, 31, 32, 33, 12345, 65535, 202751).

**P2: post-capture rebinding was silently accepted.** `bind_decode_state` now
raises if it is called after `capture_decode_trace` with a different cache or
page-table tensor, and traced `decode_forward` raises rather than silently
ignoring a caller-supplied device page table that is not the bound one. The
dead `pt = page_table if ... else page_table` line is gone and
`_page_table_torch` is kept in sync so `only_if_changed` diffs against real
state.

**P2: pad-invariance only covered single-chunk prefill.** The bit-identical
pad-token test is now parameterized over 154 (single chunk), 2049 and 2600
(multi-chunk: 2048 + a bucketed tail, where the tail meets the
chunk-offset-dependent RoPE slice, `chunked_flash_mla_prefill(chunk_start_idx)`
and the per-chunk `paged_fill_cache` page-table slice), and the decode-side
pad-invariance test moved to 2600. Cross-shape parameterization extended to
2049 and 2600. 11 passed.

Smaller items from the same review, all fixed: `test_batch_inactive_rows`' no-op
`seq if user in active else seq`; `batch 32 x 8192` stated as 7.2 GiB in
`context_contract.json` against 7.023 measured; the stale "rejected on the
shape/trace grounds" sentence in Performance accounting (the greedy benchmark
had already shown force-argmax writes `[1,1,1,32]` correctly into a
preallocated buffer); the 32-row sampler width named as the stage-07 lever;
`check_context_contract` output committed as a log; unused imports and `black`
formatting; and `tests/summarize_perf_report.py` added so
`perf_report_summary.json` is regenerable from the tree.

## FM-012: stage review round 2 (`more-work-needed`) and the fixes

Second independent `$stage-review` pass. It confirmed round-1 P1 (the Tracy
recapture) as genuinely fixed and raised five P2s plus a set of consistency
items. Resolutions:

**P2: no full-model evidence near the advertised context.** Correct and the
most substantive finding: the longest logical prompt ever run through the
47-layer public generator was 3000, while the contract advertises 202752.
Added `tests/test_full_context.py`, which prefills **202751** tokens (the
longest valid non-aligned length, physical 202752) through the real 47-layer
stack and then takes traced decode steps at position 202751, the last valid
position. (That prompt length was wrong and took the device down; FM-013 has
the root cause and the fix, and FM-015 shortened it again to make room for the
needle query. The shipped length is 202733.) It is checkable without an HF reference because the prompt is an
exactly periodic token stream: a model whose cache, page table and positions
are healthy at that depth continues the period. Results in
`doc/full_model/full_context.json`.

**P2: the shipped generator postdated every evidence run.** True at the time:
a late defensive edit (the caller-owned page-table guard) landed after the
gate logs. Fixed procedurally - all source edits were completed first, then the
whole evidence set was re-run in one sweep with no edits in between, and the
logs committed from that sweep.

**P2: round-1's RoPE fix was in the code but not in the docs.** The module
docstrings of `tt/model.py` and `tt/generator.py` and the README trace-contract
section still described the two-tensor `plus_one` design and quoted the old
`ttnn_decode_forward` signature, and the README counter line said
`rope_index_refreshes 2` where every artifact says 0. All corrected.

**P2: the prefill-spread limitation and `compile_cost.json` rested on numbers no artifact
supports.** The "5387.7 / 6469.0 ms, 18% spread" pair came from a superseded
run. `tests/measure_cold_compile.py` now takes several warm repeats per prompt
and records every sample plus `repeat_spread_pct`, so the spread claim is an
artifact rather than a memory, and the README quotes it from there.

**P2: stale counts and superseded numbers.** `test_prefill_padding` 7 -> 11
passed, `test_full_model.py` 26 -> 35 tests, sampler 665.8 -> 665.0 us and
1.121 -> 1.125 ms in the sampler-rows limitation (1.122 ms as re-measured in
FM-015), FM-011's P1 recapture numbers replaced with
the committed ones, and a markdown table row that had swallowed the following
paragraph.

Code holes the review found by inspection (unreachable from the tests, but on
the exact surface a vLLM adapter drives), all now closed:

* `bind_decode_state`'s post-capture guard treated *any* torch page table as
  "same binding", so a post-capture call with a torch table while a
  caller-owned device table was bound slipped through, wrote into the caller's
  buffer, and silently flipped ownership. The guard now keys on
  `_page_table_caller_owned`.
* `refresh_page_table` did not honour `_page_table_caller_owned` despite the
  comment saying it must; it now raises.
* `_ensure_owned_state` filled `_page_table_torch` with the *identity* default
  when a caller had bound a device page table, which would have let a later
  `only_if_changed` diff compare against a table that was never on the device.
  It no longer invents one.

Smaller items also fixed: the DRAM-capacity probe is now a committed script
(`probe/dram_capacity_probe.py`) instead of a quoted snippet, so
`capacity.json`'s allocatable figure has provenance; `compile_cost.json`'s
`build` block no longer contains 47 degenerate `key == value` entries; the
expert dtype table splits decode LoFi from prefill HiFi2 (the prefill rows
really do read `HiFi2 BF16 x BFP4`, which is the optimized decoder's
`prefill_expert_fidelity`, not a policy break); the unsupported "~74 GB/s" for
the prefill sparse matmuls is gone (`tt-perf-report` omits DRAM utilisation for
sparse rows, so only the share is claimed, 48.3% in the FM-016 run); the "no trace churn" claim is
restated as structural rather than test-demonstrated; the 290 residual
dropped-marker warnings are named in the README next to the "captures are
complete" claim; and the L1-vs-DRAM logits margin is labelled as the
single-sample tie-break it is.

## FM-013: the 202752-context run, and a device wedge it exposed

Round 2's headline finding was that nothing had run the *whole* model near the
advertised context. Building that test found a real robustness bug.

First two attempts hung. Both times the host ended up blocked in
`ttnn::Tensor::cpu` -> `enqueue_read_tensor` -> `wait_for_outstanding_reads`
(native backtrace via `gdb -p`), i.e. a tiny readback waiting behind device
work that never completed, and `tools/tt-triage.py` could not even attach:
`DeviceTimeoutError: MMIO per-op timeout: 4B load took 220639 us`. The board
needed `tt-smi -r` to come back; `tt-smi -ls` and a mesh open/close smoke both
looked healthy *before* the reset, so device listing is not a sufficient health
check after this failure.

Isolating it: `probe/decode_position_scaling_probe.py` times one decode step at
a ladder of positions against an allocated-but-unfilled cache (the flash MLA
decode reads the cache regardless, so no 40-minute prefill is needed):

```
{"position": 128,    "eager_ms": 13.5, "traced_ms": 1.81}
{"position": 2048,   "eager_ms":  3.4, "traced_ms": 1.89}
{"position": 8192,   "eager_ms": 3.33, "traced_ms": 2.03}
{"position": 32768,  "eager_ms":  3.3, "traced_ms": 2.61}
{"position": 65536,  "eager_ms": 3.85, "traced_ms": 3.40}
{"position": 131072, "eager_ms": 5.46, "traced_ms": 4.96}
{"position": 202751, "eager_ms":  7.1, "traced_ms": 6.64}
```

(Those are the round-2-era numbers, quoted as they were read at the time. The
committed `decode_position_scaling.json` was re-measured in FM-015 and reads
1.82 -> 6.67 ms traced across the same ladder; the shape of the curve, which is
what mattered here, is unchanged.)

Decode at the last valid position is fine and scales smoothly (2-layer probe;
+4.8 ms from position 128 to 202751, which is per-layer cache-read growth, not
a cliff). So the stall was not depth.

Root cause: **the test asked for decode positions past the end of the context.**
It prefilled 202751 tokens and then took 8 decode steps at 202751..202758, but
the paged cache and page table only represent [0, 202752). Step 2 onward drove
`paged_update_cache` with a page-table index past the table, and that does not
raise - it wedges the device, after which every read hangs behind the wedged
queue. The third attempt reproduced it exactly at decode step 6 with per-step
logging, which is what pinned it down.

Two fixes:

* the test now prefills `202752 - 8 = 202744` tokens (still non-aligned to the
  tile, block, bucket and chunk, still a full 202752-token physical prefill) so
  the eight decode steps land on 202744..**202751**, the last valid position;
* `GLM47FlashGenerator.set_decode_positions` now rejects any position outside
  `[-1, max_seq_len)`. This is the real deliverable of the episode: a serving
  adapter that lets a request run one token past its context would otherwise
  take the card down rather than get an error.
  `test_decode_position_past_context_is_rejected` covers it.

## FM-014: final evidence sweep

All source edits were completed before this sweep, and nothing was edited
during or after it, so every committed log matches the committed source. One
script, `/tmp/final_sweep.sh`, ran in order: DRAM-capacity probe, warm-cache
compile cost, cold-cache compile cost, `test_full_model.py` +
`test_prefill_padding.py`, `test_full_model_perf.py`,
`test_full_model_batch.py` at batch 32, `run_prefill_check`,
`run_teacher_forcing`, `run_autoregressive`, `check_degenerate_output`,
`check_context_contract`, the qualitative suite, the Tracy profile plus
`tt-perf-report` and `summarize_perf_report.py`, and three watcher runs.
`test_full_context.py` ran separately just before it (40 minutes on its own).

Every step exited 0; all three watcher runs reported 0 faults. Results are in
`doc/full_model/logs/` and the JSON artifacts, and the README numbers are taken
from them.

Two things changed after that sweep, both at commit time and both re-verified:

* the repo's pre-commit hooks removed dead `import torch` statements from
  `tt/generator.py` (inside `generate`), `tt/model.py` (inside
  `from_pretrained`) and `probe/decode_position_scaling_probe.py`, and isort
  collapsed one import statement in `tt/generator.py`. Removing an unused name
  binding cannot change behaviour, and it was checked anyway: the main suite
  was rerun (`logs/pytest_full_model.log`, 46 passed) and the reduced smoke
  reproduced the same eight generated tokens after each hook pass;
* the `prefer-expect-error` hook required `tests/test_full_context.py` to use
  the repo's `expect_error` fixture instead of `pytest.raises`. Rerun with
  `-k rejected`: that run's log (`logs/pytest_full_context_guard.log`) was
  superseded and removed in FM-016, when the guard tests moved into the full
  `test_full_context.py` session; `logs/pytest_full_context.log` is the
  current record. It reported 1 passed at the time. The 40-minute
  `test_full_context_prefill_and_decode` was not rerun for a test-only change
  in a sibling test; its evidence is `full_context.json` plus
  `logs/pytest_full_context.log` (2 passed).

## FM-015: stage review round 3 (`more-work-needed`) and the fixes

Round 3 returned no correctness findings against the model or the generator.
One P1 and a set of P2s were all about provenance: numbers in the report that
no committed artifact contained, a timing method that measured the wrong thing,
and a traced path that could still walk out of the context. Resolutions:

**P1: the headline table came from a run that was not the committed
`perf.json`.** True. `perf.json` was dirty relative to the commit and the
headline quoted figures (`21.756`, `45.96`, `329.1`) that existed in no
committed artifact, because evidence arms had been re-run after the commit to
check unrelated things. Fixed at the root rather than by editing numbers:

* one sweep, `/tmp/final_sweep2.sh`, ran with the source **already committed**
  (`84b11d86639`) and with no stage source changes in the tree. The sweep's
  first and last acts are `git rev-parse HEAD` plus
  `git status --porcelain` over the stage directory, recorded in
  `logs/sweep_provenance.log`, so the claim is checkable instead of asserted;
* every generated JSON now carries `source_manifest`: sha256 prefixes of
  `tt/model.py`, `tt/generator.py`, the three decoder modules and the script
  that produced the file (`GLM47FlashModel.source_manifest`). An artifact is
  now attributable to exact source rather than to a timestamp;
* every number in the README headline, the performance-accounting block and
  the compile-cost table is taken from that one sweep's artifacts. The two
  artifacts that predate `source_manifest`
  (`greedy_sampler_benchmark.json`, `logits_memory_ab.json`) and the reused CPU
  HF control (`qualitative/hf_control.json`) are named in the README instead of
  being covered by a blanket claim.

The only files edited after the sweep are this work log and the README, neither
of which any run reads, so `sweep_provenance.log`'s closing `git status` lists
exactly those two.

**P1/P2: the cold-compile probe measured host enqueue, not prefill.**
`prefill_forward_last_logits_device` returns a *device* tensor and
`ttnn.deallocate` does not block, so `timed_prefill` stopped its clock while
the device was still draining. The first call therefore returned early and
every later call was measured while device-bound behind it, which is where the
"repeats are systematically slower than the first call" anomaly came from, and
it made the cold-cache penalty (`+1391 ms`) unusable. Fixed with
`ttnn.synchronize_device` on both sides of the timed region and both arms
re-run. The conclusion changes:

| | before (unsynchronized) | after |
|---|---|---|
| prompt 3000, cold, first call | 6779.6 ms | 7817.2 ms |
| prompt 3000, warm, first call | 5388.3 ms | 6479.1 ms |
| cold-cache penalty at an un-warmed shape | +1391 ms | **+1338 ms** |
| prompt 3000 repeats vs first call | +1081 ms (slower, unexplained) | **-1342 ms** (faster, as expected) |
| repeat spread | 0.0% | 0.0-0.1% |
| prefill warmup at construction, cold / warm | 71.2 / 4.6 s | 71.4 / 4.6 s |

The anomaly is retracted in FM-010, and the compile penalty is now readable two
independent ways that agree: across arms at the same call index (+1338 ms) and
within the cold arm against its own repeat mean (+1342.0 ms, against +3.8 ms
for the warm arm).

**P2: a traced decode loop could still walk out of the context.** FM-013 added
`set_decode_positions` rejection, but a *captured* trace advances the device
position itself, so a loop that started legally inside the context and kept
replaying never called back into that guard. Past the end,
`paged_update_cache` indexes off the page table and wedges the device instead
of failing. Fixed with a host mirror of the positions:
`_advance_host_positions` refuses the replay that would step out, and every
full-model trace replay now goes through the new
`GLM47FlashGenerator.replay_decode_trace` so the guard cannot be bypassed by
calling `ttnn.execute_trace` directly. `decode_step_traced` delegates to it.
`test_traced_decode_loop_stops_at_context_end` covers the traced loop,
including that an inactive (`-1`) slot is never out of range.

**P2: a needle was worth planting at the full context.** The periodic
continuation gate proves the cache, page table and positions are healthy at
position 202751, but it does not prove anything is *retrievable* from the far
end of the cache. `test_full_context.py` now plants a distinctive sentence at
position 1024 and teacher-forces a query for it over the last decode positions,
recording the full distribution at the answer position. It is deliberately
**recorded, not gated**: whether a 30.6B model with bfloat4_b routed experts
succeeds at 200k-distance recall is a property of the checkpoint, not of this
port, so gating on it would make an unrelated model property block the stage.
What is gated is that the deep-cache read produces a sane, peaked distribution.
The prompt shortened from 202744 to **202733** tokens to make room for the
11-token query, and the run still ends exactly on max_valid_position.

**Smaller items, all fixed.** `bind_decode_state` dropped its `batch` kwarg,
which could not take effect after the trace inputs were allocated; the eager
`decode_forward` argmax path now increments `host_argmax_calls` instead of
silently skipping it; `summarize_perf_report.py` marks `op_to_op_gap` as
instrumentation, since the profile test flushes the device profiler between
iterations; the unsupported "~74 GB/s" for the prefill sparse matmuls is gone
from both this log and the README (FM-011 said it had been removed but it was
still in two places); the stale uncompressed `logs/tracy_profile_run.log` was
deleted in favour of the `.gz` the README cites; the duplicated README
limitation number (`5` twice) is renumbered; headline row 3 is relabelled
"model trace only, logits out" because it is not teacher forcing; the
`_ensure_owned_state` cache-adoption behaviour is documented as a caveat in
the runtime fallback audit, since it means `reset()` can zero a caller-owned
cache; FM-008's shared-harness claim is restated, because the
`mesh_device.py` change alters device-open **defaults for every model** using
the harness rather than being additive; and the em dashes are gone from both
documents.

**The sweep itself.** `/tmp/final_sweep2.sh`, one script, in order: DRAM
capacity probe, warm-cache compile cost, cold-cache compile cost,
`test_full_model.py` + `test_prefill_padding.py`, `test_full_model_perf.py`,
`test_full_model_batch.py` at batch 32, the decode-position ladder,
`run_prefill_check`, `run_teacher_forcing`, `run_autoregressive`,
`check_degenerate_output`, `check_context_contract`, the qualitative suite, the
Tracy profile plus `tt-perf-report` and `summarize_perf_report.py`, three
watcher runs, and finally `test_full_context.py`. Every step exited 0. All
three watcher runs reported 0 faults. 81 minutes wall, of which the
full-context test is 40.

Results that moved:

| | FM-014 | FM-015 |
|---|---|---|
| TTFT, prompt 128 warmed | 389.0 ms | 388.7 ms |
| traced model-only decode | 21.756 ms/token | 21.753 ms/token (45.97 t/s/u) |
| traced token-out decode | 23.010 ms/token | 22.982 ms/token (43.51 t/s/u) |
| sampler / token readback | 1.124 / 0.133 ms | 1.122 / 0.107 ms |
| resident DRAM | 23.022 GiB | 23.022 GiB (byte-identical) |
| prefill / teacher-forced top-1, top-5, top-100 | 0.880, 0.850 / 1.000 / 1.000 | unchanged |
| full-context prompt | 202744 | 202733 (11 tokens now go to the needle query) |
| full-context periodic continuation | 9/9 | 9/9 |
| full-context prefill | 90.7 tok/s | 90.7 tok/s (2236.4 s) |
| decode at position 202751, 47 layers | 136.3 ms/token | 136.3 ms/token |
| 2-layer decode ladder, 128 -> 202751 | 1.81 -> 6.64 ms | 1.82 -> 6.67 ms |
| main suite / batch 32 / perf / profile | 46 / 5 / 2 / 2 passed | 46 / 5 / 2 / 2 passed |
| full-context suite | 2 passed | 3 passed (the traced-loop guard test is new) |

Nothing regressed, and the capacity accounting came out byte-for-byte
identical, which is the expected result for a stage whose source changes were
guards and instrumentation.

One byte changed after the sweep, at commit time: the repo's
`end-of-file-fixer` hook appended a trailing newline to
`degenerate_check.json`, because `check_degenerate_output.py` writes its JSON
without one. Disclosed rather than papered over; the verdict inside the file
(`findings: []`, `exit_code: 0`, adjacent duplication 0.0, trigram loop
fraction 0.0246) is untouched, and `logs/check_degenerate_output.log` is the
run's own record of it.

**The needle result, since it is the one genuinely new measurement.** The
answer position's top-1 is ` jade`, the correct token, from a plant 201727
positions earlier, and its top-5 is
`[" jade", " seventeen", " \"", " eighteen", " seven"]`: two of the five
candidates are words from the planted sentence and two more are near-misses on
"seventeen". The absorbed-MLA paged latent cache is therefore not merely
mechanically healthy at 200k, it is carrying retrievable content. Recorded, not
gated, for the reason above.

## FM-016: stage review round 4 (`more-work-needed`) and the fixes

Round 4 confirmed round 3's provenance work held (all stamped manifests match
the committed source, the cold-compile timer is synchronized, the traced-loop
context guard is real and tested) and raised one P1 plus three P2s.

**P1: the trace-active unsafe-allocation warning was classified from prose, not
measured.** FM-010 resolved

    Allocating device buffers is potentially unsafe due to the existence of an
    active trace

as `controlled`, describing it as firing "once per process" and concluding it
came only from `SamplingGenerator.capture_trace`. Both halves were wrong:

* `tt_metal/impl/allocator/allocator.cpp:118-133` emits the line at most once
  per host thread for the process lifetime, behind a `thread_local static bool
  warning_generated`. One log line is equally consistent with one unsafe
  allocation and with a thousand, so the line count carried no information;
* `tt_metal/distributed/mesh_device.cpp:1421-1424` registers a trace as active
  at `end_mesh_trace`, and
  `tt_metal/impl/allocator/trace_allocation_tracker.cpp:76-91` keeps
  `allocations_unsafe_` true while any trace is registered. The window is the
  whole life of the retained model and sampling traces, not the capture.

Metal ships the accounting: `TT_METAL_TRACE_ALLOC_TRACKING=1` makes
`ttnn.execute_trace` verify before every replay and raise if a buffer allocated
in that window is still alive (`ttnn/ttnn/unsafe_allocation_tracker.py`). Run
it and the answer is specific. It found one outright bug and one class of them.

*The cache-reset zero buffer.* The first run under the tracker refused the very
first replay of `dev_full_model.py smoke` with exactly one survivor:

    Buffer 2501 [op: ttnn.to_device]
      allocated at: ... reset() -> reset_kv_cache -> _cache_zeros -> ttnn.zeros
    Buffer 2501: found in 1 Python reference(s)
      'self.model._cache_zeros_buf[1]', shape=Shape([64, 1, 64, 576])

`_cache_zeros` built that buffer lazily on the first `reset()`, which is after
`build_generator` captures the traces. It is the *source* of all 47
cache-zeroing copies, so a replay landing on its address would make `reset()`
fill every layer cache with garbage instead of zeros. Why no test saw it: the
paged cache is only ever read at positions prefill or decode has already
written, so garbage in the untouched tail is invisible. `prepare_cache_reset`
now materializes the buffer before any capture (it was already accounted for as
persistent in `capacity.json`), and
`test_reset_zeroes_cache_rows_the_request_never_wrote` reads a far untouched
block after a traced generation plus a reset, which is the observable form of
the bug.

*Post-capture program compilation.* With that fixed, the tracker's next verdict
was **16 unsafe buffers per trace** (32 across the model and sampling traces),
one per program newly cached during a prefill, spanning six op types, and the
program-cache counter corroborates it exactly: 469 entries against 453 at
capture. (An earlier run of the probe, before the whole-tile terminal slice
landed, read 18 per trace over more op types; the committed
`trace_alloc.json` is the 16/32/6 run and is what this log now quotes.) A newly cached program keeps a device buffer for the
process lifetime, so compiling one while a trace is live leaves a permanently
unsafe buffer. This is not only the multi-chunk case the README already
disclosed: the terminal slice/pad depends on `seq mod 32`
(`s0 = 32 * ((seq - 1) // 32)`), so *any* logical length not warmed at
construction trips it, and `warmup_prefill` only warms the bucket lengths.
Enumerating every shape is not an option (99 chunk offsets times 32 residues at
the full context), so the fix is to re-capture instead:

* `recapture_decode_traces()` releases both traces and captures them again, so
  the trace intermediates are allocated after the program buffers exist;
* `_maybe_recapture_after_compile()` decides with an exact signal rather than a
  heuristic: `mesh_device.num_program_cache_entries()` against its value at
  capture time;
* it is wired into `generate`, the low-level `prefill_forward` and
  `_prefill_and_sample_first`, so every prefill entry point through the
  generator is covered, and called explicitly in `test_full_context.py`, which
  drives the model-level entry point for its per-layer progress callback;
* `capture_decode_trace` gained `warm_at` for it. The warm pass writes one
  cache row, and mid-request the only row it may write is the one the next
  decode step is about to overwrite anyway.

Cost: **175 ms** measured on the 47-layer model, once per new prefill shape,
against the >1.3 s the compile that triggered it already cost. It is timed
separately in `perf.json` (`first_use_shape_trace_recapture_ms`) and excluded
from both TTFT and the decode rate, because it is setup for that shape.

`probe/trace_alloc_probe.py` is the artifact. Four arms, `get_unsafe_tracked_ids`
per trace id:

| arm | unsafe buffers | replay |
|---|---|---|
| shipped path, warmed single-chunk prompt | 0 | ok |
| shipped path, first-use multi-chunk prompt | 0 (hook fired) | ok |
| the same compile with the hook bypassed | 32 (16 per trace) | **refused by the tracker** |
| after `recapture_decode_traces()` | 0 | ok |

`logs/trace_alloc_full_model.log` is the same gate over the full 47-layer
build, prefill and 128-token generate with the tracker on.

**P2: one prompt length was reported with two different prefill numbers.** TTFT
at prompt 128 read 388.7 ms while the compile-cost table read 313.8 ms for the
same prompt length and the same physical shape, and the report never named the
boundary. Two separate causes, both now measured rather than argued:

* `reset()` enqueued 47 device-to-device copies of a 118 MiB zero buffer and
  returned without draining, so the request's TTFT absorbed work that belongs
  to the request boundary. `reset()` now synchronizes before returning, which
  also makes its documented contract ("the cache is zeroed") true on return,
  and `generate` reports `reset_s` separately. Measured at **28.3 ms**;
* the rest is **prompt content**. MoE prefill routes to different experts for
  different text, and the two probes used different prompts. The perf test now
  measures both texts at the same shape: its own prompt prefills in 333.3 ms
  and `measure_cold_compile.py`'s in **313.6 ms**, which is
  `compile_cost.json`'s number to the decimal. So the residual gap is workload,
  not measurement, and it is 6.3% at prompt 128.

`perf.json` now carries `ttft_breakdown_ms`: prefill alone between device
syncs, the untraced prefill sampler, the token readback, and the unattributed
remainder, which comes out under 1 ms at both prompt lengths.

**P2: the `source_manifest` claim was broader than the mechanism.** The README
said every generated JSON carries one; four did not. `accuracy.json`,
`dram_capacity.json` and `perf_report_summary.json` now do. The helper moved to
`tt/provenance.py`, which imports neither ttnn nor torch, so the pure-CSV
summarizer can stamp its output, and it hashes itself as well. The remaining
exceptions are named in a table in the README rather than covered by a blanket
claim, along with the two logs that predate the sweep and the four files the
`end-of-file-fixer` hook rewrites at commit time.

**Also fixed, from Other Concerns.**

* A generator whose *first* capture happened in host-sampling mode kept
  sampling untraced for the rest of the process: `capture_decode_trace` skips
  the sampling capture when `host_sampling` is set, and nothing captured it
  later, so switching back to on-device sampling gave correct tokens, silently
  slower, with no error. `_ensure_sampling_trace` captures on demand, through
  the recapture path rather than calling `precompile` directly, because
  `precompile` allocates and doing that under a live trace is the very hazard
  above. `test_sampling_trace_is_captured_on_demand_if_capture_skipped_it`
  reproduces the state and asserts identical tokens.
* `set_decode_positions` required exactly `max_batch_size` entries while
  `set_decode_tokens` padded. It now pads the missing rows inactive, which is
  the fixed-slot contract the rest of the API already had.
* FM-010's row on `generate()` measuring a slightly faster ms/token than the
  isolated microbenchmark blamed "the tighter dispatch pattern", which is
  self-contradictory, since the tighter pattern is the slower number. Restated:
  it is decode position. `bench()` runs three 65-replay windows back to back,
  so the token-out window sits ~200 positions deeper into the cache than
  `generate`'s, and `decode_position_scaling.json` shows the traced step
  growing with `cur_pos`.
* The `_pad_to_sampler_rows` docstring named `fill_implicit_tile_padding`; the
  code calls `ttnn.pad`.
* The README's "six distinct prefill shapes" claim covered the decoder stack
  only. The terminal slice/pad adds one program pair per
  `(bucket, seq mod 32)`, which is what the recapture hook exists to handle.

**The sweep.** Same script as FM-015 plus the two tracker arms, run against
`13148176475` with no stage source changes in the tree
(`logs/sweep_provenance.log`; the two documents were mid-edit, and no run reads
them). Every step exited 0, all three watcher runs reported 0 faults, 85
minutes wall of which the full-context test is 40.

| | FM-015 | FM-016 |
|---|---|---|
| TTFT, prompt 128 warmed | 388.7 ms | **334.0 ms** (the reset drain came out of it) |
| of which request-boundary reset | folded in, unmeasured | 28.3 ms, drained before the clock |
| prefill only, prompt 128, this report's prompt | not separated | 333.0 ms |
| prefill only, prompt 128, `measure_cold_compile.py`'s prompt | 313.6 ms in another artifact | 313.7 ms, same run, same shape |
| traced model-only decode | 21.753 ms/token | 21.752 ms/token (45.97 t/s/u) |
| traced token-out decode | 22.982 ms/token | 23.014 ms/token (43.45 t/s/u) |
| sampler / token readback | 1.122 / 0.107 ms | 1.123 / 0.139 ms |
| end to end, prompt 128 + 128 tokens | 3.267 s | 3.239 s (39.51 tok/s) |
| resident DRAM | 23.022 GiB | 23.022 GiB (byte-identical again) |
| prefill / teacher-forced top-1, top-5, top-100 | 0.880, 0.850 / 1.000 / 1.000 | unchanged |
| readiness TTFT, prompt 154 | 615.9 ms | 763.6 ms first request, 583.4 ms second |
| cold-cache penalty at an un-warmed shape | +1338 ms | +1336 ms |
| generator construction, cold / warm | 270.9 / 180.2 s | 264.2 / 180.6 s |
| full context: prompt, continuation, prefill, decode | 202733, 9/9, 90.7 tok/s, 136.3 ms/token | identical |
| needle top-1 at 201727 positions of reach | ` jade` | ` jade` |
| main suite / batch 32 / perf / profile / full context | 46 / 5 / 2 / 2 / 3 passed | **50** / 5 / 2 / 2 / 3 passed |
| unsafe buffers alive at replay, shipped paths | not measured | **0** (`trace_alloc.json`) |

Two rows moved for reasons worth naming rather than filing as noise. TTFT at
prompt 128 dropped 54.7 ms because the request-boundary reset is now drained
before the clock starts instead of inside it; the work did not get faster, the
measurement got honest, and the reset is reported at 28.3 ms. The readiness
TTFT at prompt 154 rose because that runner issues exactly one request in a
fresh process, so it pays this length's first-use cost; `first_use_ttft.json`
puts a number on it (175.0 ms, essentially all recapture) and on the steady
state behind it (583.4 ms). The recapture is placed immediately after the
prefill, which is the conservative choice: it can only make a first-request
TTFT larger, never smaller.

## FM-017: stage review round 5 (`more-work-needed`) and the fixes

Round 5 confirmed the batch-1 measured path, every headline number, the
qualitative outputs, the watcher evidence and all 14 stamped manifests, and
returned two P1s plus six P2s. Both P1s were in round 4's own recapture work.

**P1: the terminal-shape warmup was never called.** `warmup_terminal_shapes`
was written, documented in the README and asserted in
`context_contract.json`, and nothing called it: the edit meant to add the call
to `warmup_prefill` did not land, and I did not check. So the claim "a prompt
inside one chunk needs no first-use compile" was false, and the stage's own
artifacts said so: `first_use_ttft.json` recorded `trace_recaptures: 1` and a
175 ms penalty at prompt 154, which is a single-chunk prompt, and
`test_first_use_prompt_shape_recaptures_traces_and_stays_correct` *asserted*
`recaptures >= 1` at 173, also single-chunk. A capability contract asserting a
property the code does not deliver is worse than not claiming it.

The call is in now, and the evidence is an assertion rather than prose.
`test_single_chunk_prompt_shape_does_not_recapture` reads
`num_program_cache_entries()` across a prompt whose length is not a bucket and
requires it not to move; `test_first_use_multi_chunk_shape_recaptures_and_stays_correct`
keeps the honest half (17 new programs at 4300, one recapture, identical tokens
on the second request). Measured on the full model in a fresh process:

| prompt | new programs | recaptures | first request | second request | penalty |
|---|---|---|---|---|---|
| 154 (one chunk, bucket 256) | 0 | 0 | 583.9 ms | 583.8 ms | **0.1 ms** |
| 4300 (two chunks + 256 tail) | 17 | 1 | 10279.6 ms | 10098.8 ms | 180.8 ms |

That is the round-4 readiness-TTFT regression undone: prompt 154 was 763.6 ms
in FM-016 and is 583.9 ms here, because the 175 ms it was paying was the
recapture that no longer happens.

**P1: the recapture's warm pass wrote a KV row for every slot.**
`capture_decode_trace`'s warm pass runs one eager decode of token 0, and the
decoder issues `paged_update_cache` per row, so the pass writes one cache row
per slot at the warmed position. The recapture inherited it with the *new
request's* prompt length as that position, and the docstring justified it with
"the only row it may write is the one the next decode step is about to
overwrite anyway", which is true at batch 1 and false at batch > 1: prefilling
a short request while another slot sits deeper writes a bogus row into that
slot's own blocks, inside its causal read window, with no error and no counter.
No test could see it, because all four batch tests prefill every slot in one
call, so the warmed position was always at or beyond every live position.

Skipping the warm pass looked like the fix and is not one: Metal refuses to
capture a program that is not already in the cache
("Cannot load new binaries during trace capture. This program is not yet in
program cache."), which is exactly what the first attempt hit, and it left the
capture open so every later op in that process failed too. The pass also runs
the sampler pre-compile, which has to happen while no trace is live.

The fix is to warm with every slot **inactive**. `-1` is the marker the whole
decode path already honours (`plus_one(skip_negative_entries=True)`, the
derived RoPE index pinned at 0), and `paged_update_cache` skips it, so the same
programs compile and nothing is written. Both halves are asserted rather than
argued:

* `test_traced_replay_with_all_slots_inactive_writes_no_cache_row` reads three
  cache blocks back after three inactive traced steps and requires them zero;
* `test_recapture_mid_decode_leaves_a_deeper_slot_untouched` runs batch 32 with
  a recapture injected mid-decode and requires bit-identical tokens against a
  control without one.

**P2: a prefill at exactly the supported context raised.** The recapture used
to derive its warm position from the prompt length, so `prefill_logits` at
`max_seq_len` asked `set_decode_positions` for 202752, one past the last
representable position, and raised *after* completing the prefill. The warm
position is now always -1, and
`test_prefill_at_exactly_the_supported_context` covers the boundary on a small
reduced model, because the property is the arithmetic and not the depth.

**P2: FM-016 quoted trace-allocation numbers the artifact does not contain.**
18 per trace / 36 total / 23 op types came from a probe run that predated the
whole-tile terminal slice; the committed `trace_alloc.json` is 16 / 32 / 6 and
its program-cache counter agrees exactly (469 against 453 at capture). FM-016
now quotes the committed run and says where the other numbers came from.

**P2: the `source_manifest` exception set was still incomplete and the README
contradicted itself.** `qualitative_outputs.json` now carries one, and the
Artifacts block points at the Tests-section exception table instead of
restating a narrower set four sections later.

**P2: the sweep-provenance `git status` was blind to new files.**
`.git/info/exclude` lists `models/autoports/`, so a plain `git status` over the
stage directory never reports an untracked file, which is exactly the class of
change that happened inside the round-4 window (three artifacts and one probe).
The sweep is now a committed script, `tests/run_evidence_sweep.sh`, whose
provenance step records `git ls-files --others` (deliberately without
`--exclude-standard`) plus a sha256 of every stage source file, runs the
first-use TTFT probe inside the sweep instead of after it, and keeps a
standalone `test_full_model.py` log next to the combined one.

**P2: the dtype table was wrong for the shared expert at prefill.** The table
said bfloat4_b for the shared expert in both passes; the committed
`tracy/prefill_perf_report.csv` shows the down projection at prefill as
`HiFi2 BF16 x BFP8`, because the optimized decoder re-uploads only the decode
copy at bf4 and leaves the prefill interleaved copy at `weight_dtype`. That is
the inherited decoder contract, so the runtime is right and the table was
wrong: the row is split into gate/up and down, and
`test_deployment_dtype_policy_preserved` now asserts the prefill copy's dtype
so it cannot drift from the profiler rows again.

**P2: the prefill-throughput range was stale.** "329-431 tok/s" used the
round-3 TTFT that FM-016 replaced. `perf.json` says 383.2 and 432.9, and the
full-context figure is 90.7; all three are now in the limitation.

**Also fixed, from Other Concerns.** `prefill_forward` takes `user_ids=` so an
adapter can name the slot each row fills, instead of that only being reachable
by handing in a re-rowed page table while a stray `user_id=` disappeared into
`**kwargs` and prefilled slot 0. `bind_decode_state` raises on the one
ownership transition that used to flip the caller-owned flag and write into the
caller's device page table. `reset()` marks every slot inactive rather than
position 0, so a single-user `generate` on a 32-slot model stops driving 31
phantom rows through the cache update and the RoPE lookup every step.
`bind_decode_state` also pre-materializes the cache-reset zero buffer, so a
rebind cannot reintroduce the post-capture allocation FM-016 removed.
`summarize_perf_report.py` counts the `Bound == SLOW` rows per window, so the
stage-07 geometry target is a number in an artifact. A missing readiness
reference now fails `test_readiness_reference_is_present` instead of silently
skipping both accuracy tests. The `set_decode_positions` padding branch moved
to the batch module, where batch > 1 makes it meaningful; the batch-1 module
keeps the too-many-positions case. And `test_full_model_batch.py` prints
whether `BATCH`/`BATCH_SEQ` came from the environment or the defaults, which
the log could not previously distinguish.

**The sweep.** `tests/run_evidence_sweep.sh` (now committed) against
`ca0b1330d38`, clean stage tree, with the provenance step recording HEAD,
tracked changes, untracked files and a sha256 of every stage source file at
both ends. Every step exited 0, all three watcher runs reported 0 faults.

| | FM-016 | FM-017 |
|---|---|---|
| TTFT, prompt 128 warmed | 334.0 ms | 334.5 ms (382.7 tok/s) |
| **readiness TTFT, prompt 154** | 763.6 ms | **591.0 ms** (the first-use penalty is gone) |
| first-use penalty at prompt 154 | 175.0 ms | **0.0 ms** (583.4 ms first and second request) |
| first-use penalty at prompt 4300 (multi-chunk) | not measured | 182.5 ms, one recapture, 17 new programs |
| traced model-only decode | 21.752 ms/token | 21.760 ms/token (45.96 t/s/u) |
| traced token-out decode | 23.014 ms/token | 23.017 ms/token (43.45 t/s/u) |
| sampler / token readback | 1.123 / 0.139 ms | 1.123 / 0.134 ms |
| end to end, prompt 128 + 128 tokens | 3.239 s | 3.242 s (39.48 tok/s) |
| resident DRAM | 23.022 GiB | 23.022 GiB (byte-identical, fourth sweep) |
| prefill / teacher-forced top-1, top-5, top-100 | 0.880, 0.850 / 1.000 / 1.000 | unchanged |
| cold-cache penalty at an un-warmed shape | +1336 ms | +1341 ms |
| generator construction, cold / warm | 264.2 / 180.6 s | 270.2 / 181.8 s |
| unsafe buffers at replay, shipped paths | 0 | 0, and a single-chunk prompt needs no recapture at all |
| `Bound == SLOW` rows, prefill / decode / token-out | not counted | 17 of 110 (21.1%) / 112 of 1264 (15.0%) / 112 of 1600 (9.4%) |
| main / padding / combined / perf / batch / profile / full context | 39 / 11 / 50 / 2 / 5 / 2 / 3 passed | **41 / 13 / 54 / 2 / 9 / 2 / 3 passed** |

The readiness TTFT row is the one that matters: 763.6 ms was round 4's
regression, caused by the recapture that the uncalled terminal warmup made
unavoidable at every new prompt length. With the warmup actually running it is
591.0 ms, below the 615.9 ms this stage reported before any of the
trace-allocation work, because the drained `reset()` also came out of it.

Nothing else moved beyond run-to-run spread, and the capacity accounting came
out byte-identical for the fourth sweep running.

## FM-011b: checkpoint

Repo: `tt-metal`, branch `ttmodelmanager/glm47-flash-probe`, no push.

| commit | contents |
|---|---|
| `ba10cee4e60` | parent: optimized-decoder stage |
| `59cd8b4204a` | full-model stage: `tt/model.py`, `tt/generator.py`, six test modules, five probes, `doc/full_model/` (62 artifacts), the updated `doc/context_contract.json`, the fresh AIME24 reference and `readiness_autoregressive/`, and the two `models/common/readiness_check/` fixes (one of which changes a shared default, see FM-008) |
| `a1f8f235fb2` | records the SHA above, plus the round-2 documentation fixes |
| `84b11d86639` | round-3 source fixes (FM-015): the traced-loop context guard and `replay_decode_trace`, the synchronized cold-compile timer, `source_manifest`, the needle phase in `test_full_context.py`, and the smaller items |
| `88c33f005a6` | the FM-015 evidence sweep and the report rewritten from it: all `doc/full_model/` artifacts and logs, `README.md`, `work_log.md`, `doc/context_contract.json` |
| `3107c5661d1` | records the SHA above |
| `13148176475` | round-4 source fixes (FM-016): the cache-reset zero buffer moved before capture, `recapture_decode_traces` plus the exact program-cache trigger, the whole-tile terminal slice and `warmup_terminal_shapes`, the drained `reset()`, the TTFT breakdown, `tt/provenance.py`, the on-demand sampling-trace capture, and two new probes |
| `7037afbba94` | the FM-016 evidence sweep and the report rebuilt from it |
| `ae92cd90c78` | records the SHA above |
| `ca0b1330d38` | round-5 source fixes (FM-017): the terminal warmup actually called, the recapture warmed inactive, the supported-context boundary, `user_ids=`, the page-table ownership guard, inactive slots after `reset()`, SLOW-row accounting, the committed evidence-sweep script |
| `8b5f776276c` | the FM-017 evidence sweep and the report rebuilt from it |
| (this commit) | records the SHA above |

The source and the evidence are deliberately in separate commits: the sweep ran
against `13148176475` (round 4) and `54960f76eda` (round 5) with a clean
stage tree, which is what
`logs/sweep_provenance.log` records and what every artifact's `source_manifest`
hashes.

No other repo was touched: `vllm` is out of scope for this stage. The only
files outside `models/autoports/zai_org_glm_4_7_flash/` are in
`models/common/readiness_check/`: `generate.py` (the transformers 5.x
`BatchEncoding` fix, additive) and `mesh_device.py` plus the three runners (the
`--trace-region-size` / `--l1-small-size` plumbing, which changes device-open
defaults for every model using the harness, as FM-008 states). Unrelated dirty
files in the checkout (`.env`, `model_cache/`, `tt_cache/`,
`models/tt_dit/...`, various HTML/notes at the repo root) were left untouched
and are not in the commit.
