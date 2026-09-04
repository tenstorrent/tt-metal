# GLM-4.7-Flash optimized vLLM serving: stage report

Target: `zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`, 30.6B total / ~3.6B
active, 47 layers, 64 routed experts top-4, vocab 154880), served through the
Tenstorrent vLLM plugin on **one Blackhole p150**, device 0, 1x1 mesh, full
202752 context, `max_num_seqs=32`, on-device sampling
(`sample_on_device_mode=all`). Branch `ttmodelmanager/glm47-flash-probe`.

The measured path is real vLLM TT-plugin serving through `tt/generator_vllm.py`.
No Tracy, no `tt-perf-report`, no live-server device profiler, no
serving-adapter profiler, no `ReadDeviceProfiler` was collected or attempted;
both `$optimize` and `$vllm-integration` require that of a vLLM stage, and
`perf_summary.json` records it as intentional with `decode_ms_per_token_device:
null`.

**Status at a glance:** sampling smoke (the gated profile) **3 passed, 1
skipped, 0 failed**; qualitative **6/6 prompts coherent and byte-identical to
the before arm's greedy output**; both runner gates exit 0; context contract
unchanged at 202752.

## Headline: primary single-user serving

Workload: 128-token input, 128-token output, 1 prompt, `--max-concurrency 1`,
greedy `--temperature 0.0`, `max_num_seqs=32`, `max_model_len=202752`, N150
mesh, `trace_region_size=350000000`.

| | before | **after** | delta |
|---|---|---|---|
| **TTFT, p50 / p99** | 273.88 / 273.88 ms | **274.07 / 274.07 ms** | +0.1% (noise) |
| **Decode, TPOT mean -> t/s/u** | 45.218 ms = 22.115 t/s/u | **29.496 ms = 33.903 t/s/u** | **-34.8% latency, +53.3% t/s/u** |
| TPOT p99 | 45.218 ms | 29.496 ms | -34.8% |
| ITL p50 / p99 | 45.210 / 45.584 ms | 29.492 / 29.695 ms | -34.8% / -34.9% |
| aggregate output throughput | 21.273 tok/s | 31.838 tok/s | +49.7% |
| total token throughput | 42.547 tok/s | 63.677 tok/s | +49.7% |
| end to end (128/128) | 6016.6 ms | 4020.1 ms | -33.2% |

Batch-1 is the headline, so aggregate output throughput here is the same rate as
the decode t/s/u, not an independent figure.

## Secondary: CI serving-burst

Workload: 100-token inputs, 100-token outputs, 32 prompts, no
`--max-concurrency`, greedy. Capacity and vLLM-nightly-parity evidence. **Not**
the headline decode t/s/u: burst admission dominates its TTFT and affects TPOT.

| | before | **after** | delta |
|---|---|---|---|
| TTFT p50 / p99 | 14320.1 / 14321.7 ms | **9132.9 / 9133.9 ms** | **-36.2%** |
| TPOT mean -> t/s/u | 90.157 ms = 11.092 t/s/u | 90.159 ms = 11.091 t/s/u | unchanged, by design (see below) |
| ITL p50 / p99 | 91.193 / 94.532 ms | 91.168 / 94.338 ms | -0.0% / -0.2% |
| aggregate output throughput | 137.649 tok/s | **177.185 tok/s** | **+28.7%** |
| total token throughput | 275.298 tok/s | 354.370 tok/s | +28.7% |
| end to end (32 requests) | 23245.6 ms | 18058.6 ms | -22.3% |
| completed / missing output tokens | 32 / 0 | 32 / 0 | |

## How the arms were run, and which change did what

Both arms are the **same harness, same workload, same config, same commit**, and
the source difference between them is two independent env knobs this stage
added, so the whole comparison is reproducible from the committed tree and each
change is attributable:

```bash
# before: stage-entry behaviour (both changes off)
GLM47_VLLM_MOE_COMPACT=0 GLM47_VLLM_PREFILL_SLOT_WARM=0 \
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/zai_org_glm_4_7_flash \
  --hf-model zai-org/GLM-4.7-Flash \
  --mesh-device N150 --max-num-seqs 32 --max-model-len 202752 \
  --sampling-profile smoke \
  --tt-config '{"trace_region_size": 350000000}'

# attribution arm: per-slot prefill warm only
GLM47_VLLM_MOE_COMPACT=0 python -m ... --stages serve,benchmark   # same flags otherwise

# after: both knobs at their defaults (on)
python -m ...                                                      # same flags, no env vars
```

Per-change attribution, measured (`before_after_summary.json`):

| | primary TPOT | burst TTFT p50 | burst output throughput |
|---|---|---|---|
| before (both off) | 45.218 ms | 14320.1 ms | 137.649 tok/s |
| + per-slot prefill warm only | 45.213 ms (no change) | 9126.2 ms | 177.254 tok/s |
| + compact decode MoE (shipped) | **29.496 ms** | 9132.9 ms | 177.185 tok/s |

So the single-user decode win is entirely the compact MoE, and the burst TTFT
and throughput win is entirely the per-slot prefill warm. Neither is claimed for
the other.

The attribution arm ran `--stages serve,benchmark`, so unlike the other two its
server had not served the sampling and qualitative stages first. That is why it
is an attribution arm and not a third headline: it isolates the knob but does not
carry the other arms' accumulated server state. The before/after headline pair
above is the like-for-like comparison; both ran the full
`serve,sampling,qualitative,benchmark` pipeline.

Raw: `readiness_vllm_before/`, `attribution_warm_only/`, `readiness_vllm_after/`
(benchmark JSONs, raw `vllm bench serve` results, sampling logs, qualitative
outputs, the measured runs' `server.log`); normalized three-arm table in
`before_after_summary.json`; skill-shaped reconciliation in `perf_summary.json`.
Each summary's `command_string` records the exact `vllm bench serve` invocation.
`../../readiness_vllm/` holds the after arm as the live serving state: its
tracked files are byte-identical to `readiness_vllm_after/`. Its `server.log` is
untracked (that directory's `server.log` is git-ignored), so every `server.log`
claim in this report is against the committed
`readiness_vllm_{before,after}/server.log`, which are the measured runs.

## Change 1: compact (indexed) batch>1 decode MoE, bucketed by live-row count

The adapter always builds the decoder with 32 physical rows, because vLLM's
`max_num_seqs` is 32 and the decoder pins its per-slot shard grids at
construction. Any `max_batch > 1` decode therefore took
`OptimizedDecoder._moe_decode_union`, whose `sparse_matmul` scans all `E = 64`
expert groups and zero-fills the ones the union skipped, so its output group
axis is 64 wide and every post-matmul op (two slices, the fused SiLU multiply,
the routing-weight multiply, the final reduction) ran over all 64 groups no
matter how few experts the live rows actually selected. The batch-1 path already
avoided this with `sparse_matmul`'s INDEXED/GATHER mode, whose output group axis
is compact.

`kc` is the compact expert-axis width and it is a **hard correctness bound**, not
a heuristic: with the inactive-row routing mask (added here, derived on device
from the current-position tensor), only live rows contribute selected experts,
so the union has at most `live_rows * top_k` members. A `kc` below the real union
silently drops the lowest-scoring selected experts, measured at PCC 0.60 for
kc=16 with 32 live rows (`moe_compact_layer.json`), which is why the bound is
enforced rather than tuned.

`kc` is part of the captured program's shapes, so the generator captures **one
decode trace per bucket** and picks the cheapest legal one per step from its own
live-row count. Buckets: `(4, 16, 24, 32, union)`. Whole-model token-out decode at 32
physical rows, real 47 layers, distinct tokens and positions per row
(`adapter_decode_floor_{before,after,kc64}.json`):

| live rows | before (union) | after | kc used | delta |
|---|---|---|---|---|
| 1 | 45.208 ms | **29.513 ms** | 4 | **-34.7%** |
| 2 | 48.436 ms | **41.586 ms** | 16 | -14.1% |
| 3 | 51.283 ms | **41.593 ms** | 16 | -18.9% |
| 4 | 53.397 ms | **41.603 ms** | 16 | -22.1% |
| 5 | 55.377 ms | **49.745 ms** | 24 | -10.2% |
| 6 | 56.749 ms | **49.756 ms** | 24 | -12.3% |
| 7 | 58.949 ms | **57.525 ms** | 32 | -2.4% |
| 8 | 60.213 ms | **57.537 ms** | 32 | -4.4% |
| 12 | 65.802 ms | 65.825 ms | union | +0.0% |
| 16 | 71.616 ms | 71.620 ms | union | +0.0% |
| 32 | 78.447 ms | 78.426 ms | union | -0.0% |

Every live-row count where the bucket choice changes, plus points inside each
bucket's range. That matters: an earlier revision of this stage swept only
1 / 4 / 8 / 32 -- the counts where each bucket's bound is *saturated*, i.e. each
bucket's best case -- and shipped a rule that was a **+2.1 ms/token (+3.9%)
regression at 5 live rows and +0.8 ms at 6**, because a bucket's cost is flat
across the rows it serves while the union path's grows with the real union
width, so within a bucket's range the compact form starts behind and crosses
over. `COMPACT_KC_MIN_ROWS` in `tt/generator.py` records the measured crossover for
every bucket (kc=4 from 1 live row, kc=16 from 2, kc=24 from 5, kc=32 from 7),
and the lookup is strict, so adding a bucket without measuring its crossover
raises rather than silently defaulting to "use it from one row up".

Rows 5-6 then had no bucket that paid, so a **kc=24** bucket was measured and
added: `ttnn.topk` accepts a non-power-of-two `k` (checked on device), and 24
covers the bound at both 5 and 6 rows. It is worth 5.6 and 7.0 ms/token there
against the union trace. Worst delta anywhere in the swept range is now
**+0.023 ms/token**, i.e. noise. `$stage-review` derived the original 5-row
regression from the union-width and union-cost curves before it was measured;
the sweep above confirmed it and the kc=24 bucket removed it.

`kc = n_experts` is deliberately never captured. The compact form pays for all
`kc` experts unconditionally while the union form only pays for the experts the
batch really selected, so at row counts whose bound forces full width the union
trace is faster. That refusal is what makes this a no-regression change at every
batch size rather than a trade, and it is measured, not assumed: the rejected
variant is a committed, re-runnable arm
(`probe_scripts/adapter_decode_floor.py kc64` ->
`adapter_decode_floor_kc64.json`), and it is **13.3 ms/token slower at 32 live
rows** (91.77 vs 78.43), and worse still at 12 and 16 live rows (+24.4 and
+18.7 ms/token).

The mechanism is measured too, not just asserted. The bound
(`live_rows * top_k`) is an upper bound on the union, and on real activations it
is a loose one at large batches (`moe_union_width.json`, mean distinct experts
over the 46 MoE layers on the real 47-layer model):

| live rows | bound | real union (mean) | real union (median / range) |
|---|---|---|---|
| 1 | 4 | 4.00 | 4 / 4-4 |
| 2 | 8 | 6.41 | 6 / 4-8 |
| 4 | 16 | 10.41 | 11 / 6-15 |
| 8 | 32 | 16.65 | 17 / 11-22 |
| 16 | 64 | 23.43 | 24 / 15-31 |
| 32 | 64 | 32.17 | 32 / 20-48 |

At one live row the bound is exact, which is why the narrow bucket is a pure win.
At 32 live rows the bound is 64 but the real union is ~32, so a fixed-width
compact trace does about twice the expert work the union path does -- and that is
the whole story of the rejection.

**Do not read the single-layer probes for this comparison.** They drive routing
from synthetic `torch.randn` activations, which select **52** of 64 experts at 32
rows against the real model's 32.2 (measured the same way, same file). That is
enough to invert the sign: `moe_compact_layer.json` and
`moe_union_vs_compact.json` both show compact `kc=64` beating the union path at
32 rows, which the whole-model arm refutes. Those files are stamped with the
caveat; their per-layer comparisons **at a fixed active count** are still valid
and are what they are used for here.

### The buckets do not change the numbers

On the real 47-layer model, forcing each captured bucket over identical
persistent inputs gives **bitwise identical** logits, identical argmax and
identical repeats (`bucket_numerics.json`). Every shipped compact bucket is
covered, each at the live-row count where its bound is exactly **saturated** --
kc=4 at 1 row, kc=16 at 4, kc=24 at 6, kc=32 at 8 -- which is each bucket's
zero-slack case and therefore the one that most needs checking. The probe
derives those row counts from the shipped bucket table rather than hard-coding
them, so a bucket cannot be added without this evidence covering it. The reduced 2-layer
model agrees at 1, 2, 4, 5, 6, 8, 16 and 32 live rows, to a row checksum delta of
exactly 0.0 (`compact_decode_equivalence.json`). At the output level, all six
qualitative greedy completions are byte-identical between the arms
(`qualitative_before_after.json`).

### The routing prologue, split and then optimized

The compact path's per-layer prologue is not all new work. Measured on one real
MoE layer at B=32, kc=4, one live row (`moe_prologue_ablation.json`):

| | ms/layer | ms/token over 46 MoE layers |
|---|---|---|
| router + top-k + normalize + mask (both paths pay this) | 0.1202 | 5.53 |
| union path's own tail (+ max + to_layout) | 0.1234 | 5.68 |
| compact tail, first implementation (repeat + `ttnn.gather`) | 0.1904 | 8.76 |
| compact tail, candidate (one-hot + small matmul) | 0.1678 | 7.72 |
| **compact tail, shipped (`ttnn.embedding` over an `[E, B]` table)** | **0.1599** | **7.36** |

So the compact-specific cost is 0.1599 - 0.1234 = 0.0365 ms/layer = 1.68
ms/token, not the ~7.8 ms an earlier draft of this report implied by treating
the whole prologue as addressable. Two adapted alternatives to the first
implementation were measured and the fastest shipped: extending
`_moe_decode_indexed`'s own `ttnn.embedding` trick to a per-row `[E, B]` table
(the batch-1 path uses a `[E, 1]` table for the same lookup) recovered **1.40
ms/token** against `repeat` + `ttnn.gather`. `ttnn.gather` does not broadcast its
index against its input, so the `repeat` is mandatory in that form; that is the
op contract that made it the loser.

## Change 2: per-slot prefill programs warmed at startup

A served prefill compiled exactly one new program the first time each decode slot
was used, slot-keyed rather than length-keyed
(`prefill_recapture_probe_before.json`: slot 7 compiles one program at prompt
length 100 and then none at 200 or 400; repeating a slot compiles nothing; with
`set_program_cache_misses_allowed(False)` a fresh slot throws). That compile
happened **while the decode traces were live**, so
`_maybe_recapture_after_compile` correctly released and re-captured every decode
trace right after it: 22 recaptures inside the 100/100/32 burst, one per admitted
request.

`GLM47FlashGenerator.warmup_prefill_slots` now compiles one prefill per slot at
the shortest warmed bucket during server warm-up. After:
`prefill_recapture_probe_after.json` shows 0 programs compiled and 0 recaptures
for fresh slots. Recaptures over a whole serving run went **31 -> 2**, burst TTFT
14320.1 -> 9126.2 ms and burst output throughput 137.65 -> 177.25 tok/s (the
warm-only attribution arm). Startup cost: 31 extra short prefills, off every
measured path.

Accounting: 22 recaptures x ~0.24 s each is ~5.2 s of the 14.3 s before-arm
burst TTFT. The other ~9.1 s is the irreducible cost of 32 sequential prefills,
which is exactly what the after arm's TTFT now is.

## Serving contract, from the live server

`tt/generator_vllm.py` logs one counter line per 100 traced decode steps into
`server.log`, as deltas. The after arm's run has 33 such windows. **32 of the 33**
show:

```
model_trace_replays=100 sampling_trace_replays=100
eager_decode_steps=0 eager_sampling_steps=0
full_logits_readbacks=0 host_argmax_calls=0
```

and 98-100 of every 100 decode calls skipped the page table
(`page_table_calls_skipped`; four windows are at 100). The refresh counters are not uniformly zero and
should not be quoted as if they were: about half the windows show
`token_input_refreshes=2 position_refreshes=1 token_readbacks=1
page_table_refreshes=1..3`, which are request boundaries (`reset_batch=True`
steps, where vLLM's scheduler layout genuinely changed) and page-table growth as
requests cross 64-token block boundaries. Windows entirely inside one request's
steady state do show zero of all of them, for example:

```
model_trace_replays=100 sampling_trace_replays=100 eager_decode_steps=0
eager_sampling_steps=0 full_logits_readbacks=0 host_argmax_calls=0
token_input_refreshes=0 position_refreshes=0 page_table_refreshes=0
trace_recaptures=0 decode_trace_bucket_switches=0 token_readbacks=0
page_table_calls_written=0 page_table_calls_skipped=100
| kc_replays(total)={'None': 13, '32': 4, '16': 24, '4': 2967}
```

That is the contract as a fact from the running server:

* every step replayed both the model decode trace and the split-sampling trace;
* zero eager decode steps and zero eager sampler steps on the measured path;
* zero full-logits readbacks and zero host argmax calls;
* token feedback and the position advance are device-owned inside the trace, and
  the page table is re-uploaded only when vLLM's block list actually changes;
* `kc_replays` shows the primary benchmark ran on the `kc=4` bucket and the burst
  on the union trace, which is the bound working as designed.

The one window that does not is the first (`readiness_vllm_after/server.log`,
during the smoke sampling stage): `eager_sampling_steps=19
full_logits_readbacks=19`. Both come from paths that are deliberately not the
measured path and are named here rather than left to be discovered:

* `models/common/sampling` runs a **seeded** draw eagerly by construction, so a
  replay cannot observe stale seed state -- that is `eager_sampling_steps`;
* the smoke profile's `test_min_p` asks for logprobs, and on this single-chip
  mesh vLLM itself forces the whole step to host-sample
  (`vllm_tt_plugin/model_runner.py`'s `check_perform_device_sampling`). The
  adapter's `sampling_params is None` branch then returns logits for vLLM's own
  host sampler -- that is `full_logits_readbacks`. It still drives the traced
  model graph; only the on-device split sampler is skipped.

Neither appears in any benchmark window. Every window from either benchmark
profile shows zero of both.

Async decode is real and exercised: `decode_forward(read_from_device=False)`
returns the persistent device token tensor, `read_decode_output(async_read=True)`
issues `cpu(blocking=False)` plus an event, `process_decode_output_host` does the
`ttnn.to_torch` after the caller synchronizes, and the model trace replays with
`ttnn.execute_trace(..., blocking=False)`. vLLM ran with asynchronous scheduling
enabled (`server.log`: "Asynchronous scheduling is enabled"), so the plugin's
`submit_async_non_dp_decode` path drove every measured decode step.

## Checks

| check | result |
|---|---|
| serve, 202752 ctx, 32 seqs | healthy, **0** `TT_FATAL` / OOM / engine-core failures in either arm |
| **sampling (smoke) -- the gated profile** | **3 passed, 1 skipped, 0 failed** |
| sampling (full) -- recorded, not gated | before: 5 failed, 68 passed, 1 skipped; after: 8 failed, 65 passed, 1 skipped |
| the after arm's failures re-run alone on a fresh server | **8 passed, 0 failed** |
| qualitative, 6 chat-template prompts (`prompt_mode: chat`), greedy + sampled | passed; greedy byte-identical to the before arm |
| gate: `check_degenerate_output --scope all` | **exit 0**, "No degenerate output detected"; worst trigram-loop fraction 0.0865 (worst greedy 0.075) against a 0.50 advisory threshold, worst adjacent duplication 0.0273 against a 0.10 critical threshold |
| gate: `check_context_contract --stage optimized-vllm` | **exit 0**, target 202752 = supported 202752 |
| non-aligned prompt lengths through the live server | 37, 129, 777, 1039, 2051 tokens all served, exact `prompt_tokens` echoed; 1039 and 2051 cross the 1024 serving prefill-chunk cap |
| concurrency | 32 concurrent requests completed, 0 missing output tokens |
| reduced adapter suite | **25 passed** |
| full-model batch-32 suite (real 47 layers) | **10 passed**; **10 passed** again at `GLM47_FM_BATCH=8`, which takes the union fallback |
| full-model batch-1 suite (real 47 layers) | **47 passed** |
| watcher (`TT_METAL_WATCHER=10`) | **0 faults**, see `watcher/summary.json` |
| batch-1 full-model traced decode (previous stage's headline) | 21.830 ms/token token-out (mean, matching the recorded figure's statistic), vs 23.013 recorded: no regression |
| full-model accuracy, re-derived under this stage's source | prefill top-1/5/100 **0.880 / 1.000 / 1.000**, teacher-forced **0.850 / 1.000 / 1.000** -- identical to `doc/full_model/accuracy.json`'s recorded values |

### The full sampling profile: 5 failed (before) -> 8 failed (after)

Sampling status is `smoke-gated`, carried forward from the vLLM-integration
stage where the project owner accepted that coverage. The full profile is
recorded for both arms, and the after arm has three more failures than the
before arm. That is chased, not waved away:

* **The failing set varies run to run, and always from one pool.** Five
  measurements of the same suite against the same server config now exist:
  the vLLM-integration stage 11, this stage's before arm 5, and three after-arm
  runs at 8, 7 and (the committed final one) **8**. The before arm's five
  (`test_mixed_params_batch`, `test_topk[19]`, the three
  `*_penalty_mixed_batch`) are a subset of every after-arm set; the extras vary
  run to run with no code change touching them -- `test_top1_is_greedy` fails in
  the final run and passed in the one before it. Every failure in every run is
  an `assert_deterministic` failure at full or near-full occupancy.
* **They pass alone against a freshly started server.** The eight failing node
  ids -- the same set as the committed final run's -- were re-run against a
  fresh server: **8 passed** (`logs/sampling_isolated_after.log`). That is the
  discriminator the vLLM-integration stage established for this defect.
* **The decode logits are bitwise identical whichever bucket runs**
  (`bucket_numerics.json`), so the compact path cannot move a sampling result.
  The worry was real and worth testing: the compact path reduces expert outputs
  in union-score order while the union path reduces them in expert-id order, and
  float addition is not associative.
* **The convenient shortcut does not hold, and is not used here.** Most of the
  failing tests run at 15-32 concurrent, where the bound forces the union trace,
  i.e. unchanged code. But `test_top1_is_greedy` runs a batch of **4**, which
  takes the compact `kc=16` bucket. So "all the failures are on the unchanged
  path" would be false. What rules the change out is the bitwise identity at 4
  live rows plus the isolated pass, not the row count.

This is the upstream full-occupancy determinism defect filed as
[tenstorrent/tt-metal#55408](https://github.com/tenstorrent/tt-metal/issues/55408).
Its signature is worth restating more precisely than "reproducibility": under
greedy decode at full occupancy, N-1 rows return the correct completion and one
row returns garbage. It remains this model's most significant serving
limitation. This stage did not narrow it and does not claim to.

## Serving vs the model's own traced decode

| | ms/token | t/s/u |
|---|---|---|
| vLLM single-user serving, 32-row build, 1 live request | **29.496** | 33.90 |
| adapter only, same 32-row build, 1 live row, no vLLM engine | 29.513 | 33.88 |
| full-model traced token-out, batch-1 **build** (1 physical row) | 21.830 | 45.81 |

The vLLM engine costs **-0.02 ms/token** against the adapter driven directly with
the same async split, i.e. nothing measurable. Serving decode is as fast as the
model's own traced decode for comparable work. The remaining 7.68 ms/token is not
serving overhead and splits into two measured parts:

* **1.68 ms/token** is the compact path's own routing tail (union max, top-k,
  index prep, embedding lookup) -- already optimized twice this stage, from 3.08
  ms/token down to 1.68;
* **~6.00 ms/token** is the cost of driving 32 physical decode rows rather than
  one through attention, the paged cache update, the norms and the shared expert.
  Narrowing that needs a decoder whose per-slot shard grids are built for fewer
  rows, which is a construction-time change
  (`GLM47FlashGenerator.bind_decode_state`'s own docstring: "a narrower decode
  batch needs a differently-constructed model") and would trade away the 32-seq
  capability unless several decoders shared weights. Measured, quantified, and
  deferred rather than claimed.

## Rejected and deferred, with the measurement

* **`kc = n_experts` (64) as the top compact bucket.** Rejected on a committed,
  re-runnable arm: 91.77 vs 78.43 ms/token at 32 live rows
  (`adapter_decode_floor_kc64.json`), because the real union at 32 live rows is
  ~32 of 64 experts, not 64 (`moe_union_width.json`). Those row counts replay the
  union trace instead.
* **`repeat` + `ttnn.gather` for the compact routing weights** (the first
  implementation): 0.1904 ms/layer vs the shipped 0.1599
  (`moe_prologue_ablation.json`).
* **One-hot + small matmul for the compact routing weights**: 0.1678 ms/layer,
  also beaten by the shipped embedding form.
* **One compact bucket per reachable row count** (`kc = live_rows * top_k` for
  rows 1..8: eight compact traces plus the union trace). Measured, not argued
  away: it is faster everywhere it differs -- 8.0 / 3.9 / 4.1 / 3.8 ms/token
  better at 2 / 3 / 5 / 7 live rows -- and loses nothing
  (`adapter_decode_floor_kcexact.json`). **Deferred on a measured resource cost,
  not on latency:** it uses **78.4%** of the 350 MB trace region against the
  shipped set's 43.5%, leaving 9.0 MiB per bank against 23.6 MiB for the
  additional traces `models/common/sampling` captures per sampling mode.
  Adopting it should come with a multi-sampling-mode serving run proving the
  region still fits. `COMPACT_KC_BUCKETS` / `COMPACT_KC_MIN_ROWS` in
  `tt/generator.py` are the knobs, and the crossover lookup is strict, so a new
  bucket must be measured before it can be used.
* **kc=32 for 5-6 live rows** (what the first crossover-free rule selected):
  57.5 ms/token against the union trace's 55.4 / 56.7. Rejected;
  `COMPACT_KC_MIN_ROWS[32] = 7`.
* **Narrowing the physical decode batch below 32 rows.** Would recover the ~6.00
  ms/token above; construction-time change, deferred (see previous section).
* **A/B in one process with two live generators.** Two generators holding live
  traces on one device corrupt each other's replays (measured: max absolute
  logits difference ~1e20 between arms that are bitwise identical when run in
  separate processes). Every A/B here runs one arm per process. Recorded because
  it is a trap for the next person, not a model property. `work_log.md` OV-003.

## Warnings in the serving logs, classified

* **`ttnn.split: L1 budget exceeded ... DRAM downgrade`**, 163 (before) / 106
  (after) occurrences. Already classified, in `tt/model.py`'s constructor
  comment: it is `TTSampling`'s split of the 9.9 MB sampler-ready logits tensor
  in L1, an op-internal DRAM migration inside the captured sampling graph. The
  full-model stage measured both arms (`doc/full_model/logits_memory_ab.json`):
  producing the logits in DRAM removes the fallback but is 34 us/token slower
  end to end, and the tokens are identical. L1 is therefore the deliberate
  default and the fallback is disclosed rather than paid for. Not a prefill-path
  or QKV-layout problem, and not new here.
* **`Allocating device buffers is potentially unsafe due to the existence of an
  active trace`**, exactly 1 per arm, during warm-up. This is the hazard
  `recapture_decode_traces` exists for, and the model's own
  `probe/trace_alloc_probe.py` is the procedure for auditing it. It is unchanged
  in count by this stage (1 before, 1 after) despite going from one decode trace
  to four, and no correctness symptom follows it: qualitative output is
  byte-identical between arms, both gates pass, and the bucket logits are bitwise
  identical.

## Limitations

1. **CI serving-burst TPOT is unchanged (90.15 ms).** At 9 or more live rows the
   correctness bound forces `kc = n_experts`, which measured slower than the
   union path, so those steps replay the union trace unchanged. The burst's gain
   is TTFT (-36.2%) and throughput (+28.7%) from the recapture fix. Disclosed
   rather than hidden in the aggregate throughput number.
2. **Sampling is `smoke-gated`**, inherited; the full profile is recorded for
   both arms and analysed above. Not presented as equivalent to a green full
   profile.
3. **Every remaining decode-trace recapture is several times more expensive**,
   because there are five decode traces to release and re-warm instead of one.
   Derived from the burst TTFT arms at the four-trace point: ~1.0 s against
   ~0.24 s with one trace; the fifth bucket was added afterwards and was not
   separately timed, so treat ~1.0 s as a lower bound. The per-admission trigger
   is gone, but two triggers remain in production -- the first multi-chunk
   prompt (18 programs compiled once) and a sampling-mode change -- and each is
   now a ~1 s stall on the request that causes it.
4. **Trace-region occupancy rose** from one decode trace to five plus the shared
   logits buffer: 43.5% of the 350 MB reservation in use, 23.6 MiB per bank free
   (`doc/context_contract.json` `optimized_vllm.trace_region_measured`).
   `models/common/sampling` captures one further trace per active sampling mode
   into the same region, so the reservation is no longer trivially oversized. The
   reservation itself is unchanged, so no other DRAM budget line moves.
5. **Full-model watcher coverage is partial**: two of three targeted batch-32
   tests completed under watcher; the third hit pytest-timeout's 300 s cap on
   watcher overhead during a 32x96-token batched prefill and passes without
   watcher. Watcher itself reported zero faults. `watcher/summary.json`.
6. **Two oversize logs are not committed** (repo 500 KB file limit): the 3.5 MB
   full-model watcher log, and the ~690 KB `server.log` of each arm's separate
   `--sampling-profile full` record run. The commands to regenerate both are
   recorded and the watcher fault count was checked before deleting. Each arm's
   *measured* run `server.log` -- the one with the boot configuration and the
   decode-counter windows quoted above -- is committed.
7. **Compaction is off entirely for a decode batch that is not a whole tile.**
   The compact path's per-row routing lookup builds an `[E, B]` `ttnn.embedding`
   table, so `B` must be a multiple of 32. Serving always builds 32 rows;
   any other caller (for example `GLM47_FM_BATCH=8`) falls back to the
   batch-agnostic union path rather than failing. Supporting a non-tile batch is
   a measurement away, not a redesign, and nothing has needed it.
8. **Which measurements are on the shipped source, precisely.** Re-measured
   after the last source edit: the `after` serving arm (both benchmark profiles,
   smoke sampling, qualitative, and the full sampling record),
   `adapter_decode_floor_{before,after,kc64,kcexact}.json`,
   `prefill_recapture_probe_{before,after}.json`,
   `full_model_batch1_regression.json`, both runner gates, the adapter suite
   (25 passed), the adapter suite under watcher, and the full-model batch-1 (47)
   and batch-32 (10, plus 10 at `GLM47_FM_BATCH=8`) suites, which also
   regenerated `doc/full_model/accuracy.json` with a matching source manifest.
   **Not** re-measured: the `before` and `attribution_warm_only` serving arms
   (both run `GLM47_VLLM_MOE_COMPACT=0`, so `_kc_buckets` returns no buckets and
   none of the compact-path edits are reachable in them),
   `bucket_numerics.json` and `compact_decode_equivalence*.json` (the edits
   since are bucket-*selection* and docstrings; these probes force a bucket
   directly and do not consult the selection table), `moe_union_width.json`,
   `moe_prologue_ablation.json`, `moe_compact_layer.json`,
   `moe_union_vs_compact.json` and `non_aligned_serving.json` (single-layer or
   serving-request probes that do not touch bucket selection), and the
   full-model batch-32 watcher run (labelled in `watcher/summary.json`).
9. Not exercised: prefix caching (still `False`), KV-cache migration, multi-host
   or multi-rank serving (single chip by design).

## Files

* `tt/optimized_decoder.py` (`_moe_decode_compact`, `_routing_weights_decode`),
  `tt/model.py` (`decode_active_mask`, `allocate_decode_logits`, `moe_kc` /
  `logits_out` plumbing), `tt/generator.py` (kc buckets, per-bucket traces,
  shared logits buffer, `warmup_prefill_slots`), `tt/generator_vllm.py`
  (both A/B knobs, page-table diff, counter logging).
* `tests/test_generator_vllm_adapter.py` -- 10 new tests covering the kc bound,
  bucket selection, the measured crossover below which a bucket must not be used,
  the union fallback, one-trace-per-bucket, the shared logits buffer replayed per
  bucket, token feedback across a bucket switch, page-table skipping, the
  per-slot prefill warm, and the non-tile-batch fallback.
* `probe_scripts/` -- every measurement in this report, re-runnable, including
  the rejected `kc64` arm.
* `readiness_vllm_before/`, `attribution_warm_only/`, `readiness_vllm_after/` --
  the three arms' runner output.
* `work_log.md` -- OV-001..OV-013, including the wrong turns and all four
  stage-review responses.
