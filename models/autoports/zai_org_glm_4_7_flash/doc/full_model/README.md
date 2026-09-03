# GLM-4.7-Flash full model: stage report

Target: `zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`, 30.6B total /
~3.6B active params, 47 decoder layers, vocab 154880, advertised context
202752), **one Blackhole p150-class chip**, device 0, 1x1 mesh, 11x10 compute
grid, 8 DRAM banks, **31.5 GiB allocatable DRAM measured**. Branch
`ttmodelmanager/glm47-flash-probe`, from optimized-decoder commit `ba10cee4e60`.

Implementation: `tt/model.py` (`GLM47FlashModel`) + `tt/generator.py`
(`GLM47FlashGenerator`, `build_generator`). The decoder stack is the
optimized-decoder deployment policy, unchanged.

## Headline

| | value | how measured |
|---|---|---|
| **TTFT, prompt 128, prefill + first token** | **334.2 ms** (383.0 tok/s prefill) | warmed shape, `perf.json`, `tests/test_full_model_perf.py`. Prefill + the untraced prefill sampler + the one-word token readback. The request-boundary cache reset is **excluded**: it is drained before the clock starts and reported separately at 28.3 ms, so the request-boundary-inclusive figure is **362.5 ms** |
| **TTFT, prompt 154, request-boundary inclusive** (AIME reference) | **590.8 ms** | `logs/run_teacher_forcing.log`, the first request in a fresh process. This row **includes** the reset, because that is what the shared harness measures, so it is not directly comparable with the row above; `first_use_ttft.json` splits it (28.3 ms reset + 555.4 ms prefill and first token = 583.7 ms) and shows **0 new programs compiled** on either the first or the second request, because every terminal program family for a single-chunk prompt is warmed before capture |
| **Traced decode, batch 1 (model trace only, logits out)** | **21.758 ms/token = 45.96 t/s/u** | model decode trace only (no sampling, no token feedback) |
| **Traced decode, batch 1 (token-out)** | **22.994 ms/token = 43.49 t/s/u** | model trace + sampling trace + the one caller-visible token read |
| Token-out decode as the readiness harness drives it | 44.07 t/s/u | `run_teacher_forcing`, `generate(..., enable_trace=True)` |
| End to end, prompt 128 / generate 128 | 3.241 s (39.49 tok/s) | includes cache reset, prefill, 127 traced steps |

The two decode numbers differ by the sampler (1.124 ms) plus the token
readback (0.112 ms); 21.758 + 1.124 + 0.112 = 22.994, which is exactly the
token-out step `perf.json` records. Both paths replay the
same captured model trace. Every number in this table comes from the single
committed `perf.json` (and, for the two harness rows, from
`logs/run_teacher_forcing.log`) produced by the sweep in work log FM-016.

| accuracy vs the fresh AIME24 chat-template HF reference (100 positions) | top-1 | top-5 | top-100 |
|---|---|---|---|
| prefill (`run_prefill_check`) | 0.880 | **1.000** | **1.000** |
| traced teacher-forced decode (`run_teacher_forcing`) | 0.850 | **1.000** | **1.000** |

Bar is top-5 >= 0.98 and top-100 = 1.00. Both clear it.

## Capability contract

`doc/context_contract.json` and `capacity.json`.

| | bytes | GiB |
|---|---|---|
| decoder layers (47) | 17,589,521,664 | 16.382 |
| token embedding (bf16, ROW_MAJOR) | 634,388,480 | 0.591 |
| LM head (bf8) | 337,018,880 | 0.314 |
| shared RoPE tables (TILE prefill + ROW_MAJOR decode) | 103,813,120 | 0.097 |
| cache-reset zero buffer | 124,084,224 | 0.116 |
| final norm | 4,096 | ~0 |
| **weights + persistent scratch** | **18,788,830,464** | **17.498** |
| paged latent KV cache, batch 1 @ 202752, bf8 | 5,831,958,528 | 5.431 |
| `TTPenalties` int32 `[32, 154880]` buffers (5, allocated unconditionally) | 99,123,200 | 0.092 |
| **resident total** | **24,719,912,192** | **23.022** |
| trace region reserved | 350,000,000 | 0.326 |
| headroom of the measured 31.5 GiB (`dram_capacity.json`) | 8,752,955,264 | 8.152 |

* **Supported context: 202752 = the full HF-advertised context, no reduction.**
  The cache is really allocated at that size in every full-model test, and
  `tests/test_full_context.py` runs the whole 47-layer stack there: a
  **202733-token** prefill (non-aligned to the tile, paged block, prefill
  bucket and prefill chunk; 202752 physical) followed by eight traced decode
  steps and an 11-token teacher-forced query, the last of which lands on
  **202751**, the last valid position. The prompt is an exactly periodic token
  stream, so the continuation is checkable without an HF reference at that
  depth: the model reproduced it **9/9** exactly
  (`" fox jumps over the lazy dog. Tenstor"`), the last prompt position's top-1
  logit sat 27.3 above the row mean, and those eight steps ran with zero host
  token/position/page-table refreshes and zero full-logits readbacks. Prefill
  measured 90.6 tok/s (2236.6 s) and decode 136.3 ms/token at that depth
  (`full_context.json`).

  The same run also reads a **needle** back out of the far end of the cache. A
  sentence containing "the vault passphrase is jade lantern seventeen" is
  planted at position 1024 and queried over the last decode positions, so the
  query attends **201727 positions** back. The answer position's top-1 is
  ` jade`, the correct token, and its top-5 is
  `[" jade", " seventeen", " \"", " eighteen", " seven"]`, i.e. two of the five
  candidates are words from the planted sentence, with the top-1 sitting 20.0
  above the row mean. This is **recorded, not gated**: 200k-distance recall is
  a property of the checkpoint and of the bfloat4_b routed experts rather than
  of this port, so gating on it would let an unrelated model property block the
  stage. What is gated is that the deep-cache read produces a finite, peaked
  distribution at all.
* **A decode position past the context is rejected on the host.** The paged
  cache and page table only represent `[0, 202752)`; driving
  `paged_update_cache` past the page table wedges the device instead of
  failing, so `set_decode_positions` refuses any position outside
  `[-1, max_seq_len)` (`test_decode_position_past_context_is_rejected`, work
  log FM-013). Decode cost itself scales smoothly to the end of the context:
  1.81 -> 6.66 ms per traced step from position 128 to 202751 on the 2-layer
  probe (`decode_position_scaling.json`).
* KV cache cost is 612 B/token/layer = **28,764 B/token** across 47 layers
  (paged compressed-MLA latent, width 576 = kv_lora_rank 512 + qk_rope 64, one
  KV head, block size 64). An MHA-shaped cache would be 962,560 B/token and
  would cap context near 13k. The latent-cache contract inherited from the
  functional stage is what makes 202752 reachable at all.
* **Batch contract.** Batch 1 is the primary latency target and keeps the full
  202752 context. Batch and context trade directly against DRAM:
  `batch x context x 28764 B` must fit what is left after the weights.
  Batch **32 tested** at 8192 tokens/user (17.440 + 7.022 = 24.462 GiB,
  `tests/test_full_model_batch.py`, 10 passed). Batch 32 at 202752 would need
  174 GiB of cache, which is a hard physical limit rather than a design
  choice.
* Prompt length is a logical input. Tested through the public generator at
  1, 17, 63, 65, 129, 154, 1057, 2049 and 2600. None of those is a multiple
  of the tile (32), the paged block (64), a prefill bucket, or the prefill
  chunk (2048). Padding, cache fill, position handling and output slicing are
  internal; `prefill_logits` returns exactly `[1, prompt_len, vocab]`.

## What the full model adds around the decoder

The decoder stack is unchanged: same op graph, same DRAM-sharded decode
matmuls, same sparse indexed MoE at batch 1 / union walk at batch > 1, same
paged latent cache ops, same inter-layer residual layout (each layer both
returns and accepts the width-sharded L1 residual, so no layer boundary
gathers). Carried-forward dtype/fidelity policy, asserted by
`test_deployment_dtype_policy_preserved`:

| group | dtype / fidelity |
|---|---|
| activations, residual, norms | bfloat16, norms HiFi4 + fp32 acc |
| attention decode copies (`wqkv_a_ds`, `wq_b_ds`, `wo_ds`) + absorbed `w_uk`/`w_uv` | **bfloat4_b**, LoFi |
| attention prefill flat copies | bfloat8_b, HiFi2 + fp32 acc |
| shared expert, gate/up | **bfloat4_b** both passes: LoFi at decode, HiFi2 + fp32 acc at prefill (`tracy/prefill_perf_report.csv` row `128 x 2048 x 1536` = `HiFi2 BF16 x BFP4`) |
| shared expert, down | **bfloat4_b** at decode (`32 x 1536 x 2048` = `LoFi BF16 x BFP4`), **bfloat8_b** at prefill. Only the decode copy is re-uploaded at bf4 by the optimized decoder; the prefill interleaved copy stays at `weight_dtype`. Carried forward as-is, and now asserted in `test_deployment_dtype_policy_preserved` so the table cannot drift from the rows again |
| dense MLP | bfloat8_b (bf4 measured and rejected at the decoder stage) |
| routed experts | **bfloat4_b**, the deployment contract that fits 30.6B on one card; LoFi at decode, HiFi2 + fp32 acc at prefill (the optimized decoder's `prefill_expert_fidelity`, visible in `tracy/prefill_perf_report.csv` as `HiFi2 BF16 x BFP4`) |
| router gate | float32, HiFi4 + fp32 acc |
| paged latent KV cache | **bfloat8_b** (bfloat16 still supported) |

New model-level pieces:

1. **Token embedding.** `ttnn.embedding` over a `[1, 1, 154880, 2048]`
   ROW_MAJOR bf16 table. The decode token input stays a rank-4
   `[1, 1, 1, 32]` device tensor because that is exactly the preallocated
   output shape `ttnn.sampling` requires, so the sampler can write the next
   token straight into it.
2. **One shared `RopeSetup`** for all 47 layers, plus ROW_MAJOR copies of the
   cos/sin tables and a single per-step lookup hoisted to model level. Two
   separate wins, both value-preserving and both large at this context: 47
   private table sets would cost **2.4 GiB** of byte-identical data, and the
   per-layer TILE-table `ttnn.embedding` lookup **scales with table height**
   (26.2 us at 4096, 209.2 us at 202752; `cache_scaling.json`) so running it 94 times per decode step
   cost **19.7 ms/token**, more than the entire decoder stack. See work log
   FM-005; `test_shared_rope_matches_per_layer_lookup` asserts the ROW_MAJOR
   and TILE tables return equal values.
3. **Final RMS norm.** Width-sharded on the decoder's residual grid at decode
   (the shard the decoder already returns), interleaved at prefill.
4. **LM head.** One wide-1D mcast matmul over the full 11x10 grid,
   `per_core_N` 44 of the 4840 vocab tiles, bf8, HiFi2. Measured at M = 1 tile:
   878 us here against **2476 us** for the default program config, i.e. 2.8x
   (`head_probe.json`, from `probe/full_model_head_probe.py`; an earlier
   revision of this report quoted 15310 us, which the committed artifact does
   not support)
   (`probe/full_model_head_probe.py`), so the explicit config is required.
5. **Device-side decode state.** One persistent position tensor. The captured
   graph advances it with `ttnn.plus_one(..., skip_negative_entries=True)`, so a
   `-1` inactive slot is left alone, and it *derives* the RoPE index from it
   each step (`clamp(cur_pos, 0) -> uint32 -> [1, B]`). Deriving rather than
   carrying a second incremented tensor makes position coherence structural and
   pins an inactive slot at RoPE index 0 instead of letting its index walk past
   the end of the 202752-tall cos/sin table
   (`test_decode_rope_index_derived_from_position`,
   `test_batch_inactive_rows`).
6. **Prefill length bucketing.** Physical prefill lengths are bucketed to
   `(128, 256, 512, 1024, 2048)`, and longer prompts become whole 2048-token
   chunks plus a bucketed tail. Six distinct *decoder-stack* prefill shapes
   instead of one per prompt length; compiling one 47-layer prefill shape costs
   ~13 s, so without this every new prompt length paid it inside its TTFT
   (measured 16.7 s readiness TTFT before the fix). `build_generator` compiles
   the five single-chunk bucket shapes at construction.

   Two smaller program families are *not* bounded by the bucketing, and both
   compile on first use. A *multi-chunk* prompt needs
   chunk-offset-dependent programs (the RoPE-table `ttnn.slice` offsets are
   compile-time constants). And the terminal path slices the tile holding the
   last prompt position, `s0 = 32 * ((seq - 1) // 32)`, then pads to 32 rows,
   so there is one small program pair per `(bucket, seq mod 32)`. These are
   cheap to compile but they compile *after* the decode traces are captured,
   which is a trace-allocation hazard rather than only a latency question: see
   the trace-lifecycle bullet in the runtime fallback audit. Both costs are
   measured in `compile_cost.json`, cold and warm
   (`tests/measure_cold_compile.py`; the cold run's log records
   `JIT cache stats: 0/1244 hits (0.0%)`, the warm one `1244/1244 (100.0%)`):

   | | cold JIT cache | warm JIT cache |
   |---|---|---|
   | generator construction | 273.7 s | 182.7 s |
   | of which prefill warmup (buckets + every terminal program family) | 71.8 s | 4.5 s |
   | of which trace capture | 21.8 s | 0.3 s |
   | first request at an **un-warmed** shape (prompt 3000) | 7833.5 ms | 6481.1 ms |
   | steady state at that shape (mean of 4 repeats) | 6477.5 ms | 6477.1 ms |
   | first request at a **warmed** shape (prompt 128) | 313.8 ms | 313.9 ms |
   | steady state at that shape (mean of 4 repeats) | 313.7 ms | 313.7 ms |

   Read the request rows across, not down: the cold JIT cache costs an
   un-warmed prefill shape **+1352 ms** inside its first request
   (7833.5 - 6481.1) and costs a warmed shape nothing (313.8 vs 313.9, inside
   the 0.1% call-to-call spread), which is exactly what construction having
   already built those kernels should look like. The cold arm reaches the same
   conclusion without leaving its own data: its first call sits +1355.9 ms
   above its own repeat mean (`first_minus_repeat_mean_ms`), against +4.0 ms
   for the warm arm. Every timed prefill in this table brackets the call with
   `ttnn.synchronize_device`; an earlier unsynchronized version of this probe
   measured host enqueue instead and produced both a smaller penalty and an
   apparent "repeats slower than the first call" anomaly, retracted in the
   anomaly ledger.

   Padded positions are never attended:
   prefill attention is causal, and every decode step writes its own cache row
   before reading it. **Proven, not argued** (`tests/test_prefill_padding.py`,
   13 passed): at a fixed physical length, changing the pad token from 0 to
   12345 leaves the prefill logits *bit-identical* (max |delta| = 0.0) and
   leaves 24 generated tokens identical. Comparing a bucketed prefill against a
   block-aligned one at the same logical length gives PCC 0.99998-1.0, and each
   of the 0-5 argmax disagreements per prompt is a **0-3 bf16 ULP tie** between
   the same two candidates (a different physical length is a different matmul M
   and a different flash-prefill K extent, so the accumulation order differs).

## Sampler: chosen and rejected

Both common implementations were read before any token-out code was written.

**Chosen: `models/common/sampling` (`SamplingGenerator` + `TTSampling`).**

* It is the only one whose single-device vocab split adapts to 154880.
  `TTSampling.num_single_device_vocab_splits` (`tt_sampling.py:80-92`) returns
  **4** chunks of 38720, each inside `ttnn.topk`'s 65536 practical width, and
  raises rather than silently misbehaving if no legal cut exists.
* 1x1 meshes with a larger vocab are directly tested for it
  (`models/common/tests/test_sampling.py:1149-1200`, `(1,1)` at 151936 and
  256000, asserting the sampled token is the row maximum).
* Greedy determinism: `_adjust_values_for_tiebreak` (`tt_sampling.py:663-806`)
  boosts the lowest-global-index tied maximum for `k == 1` rows, working around
  the unreliable stable top-k (#33492).
* `tt_out_tok` decode feedback is a production feature of this path
  (`models/tt_transformers/tt/generator.py:1919-1934`, `:2177-2186`).
* It owns its own trace (`capture_trace` / `precompile` / `reset_trace`), which
  is exactly the second half of the split-sampling contract.
* `tt_ccl=None` is safe at 1x1: every CCL call sits behind a
  `num_devices > 1` guard.

**Rejected: `models/common/modules/sampling/sampling_1d.py` (`Sampling1D`).**

* It hardcodes a **2-way** single-device vocab split
  (`sampling_1d.py:570`, offsets `:773-777`), which for this model means two
  77440-wide `ttnn.topk` calls, past the multi-core bitonic 16-bit width
  bound (`topk_constants.hpp:26`). The factor is not configurable, and the largest
  1x1 vocab any in-tree user or test exercises is 128256.
* No greedy tie-break and it never passes `stable=`, so bf16 ties over 154880
  candidates would flip greedy output run to run.
* No seed management: the default seeds buffer is a static `arange(B)` reseeded
  on every `_sample_topk` call, so sampled decoding would redraw the same
  stream each token.
* `Penalties1D` is not wired into `Sampling1D` or any model.
* No in-tree decode `tt_out_tok` user; the TTTv2 runtime passes
  `tt_out_tok=None` and closes the loop with an explicit `ttnn.copy`.

No custom sampler code was written.

**Greedy is semantically greedy split sampling, not force-argmax.** The top-k
stage always gathers `max_top_k = 32` candidates per vocab chunk (4 chunks =
128 candidates, Wt = 4, a power of two as `ttnn.sampling` requires) and the
draw runs with `k=1, p=0, temp=1`.

Both on-device greedy strategies were **measured** on this chip on a real
`[1, 1, 32, 154880]` bf16 logits tensor
(`probe/greedy_sampler_probe.py`, `greedy_sampler_benchmark.json`):

| greedy strategy | traced | eager | agrees with torch argmax |
|---|---|---|---|
| split top-k (`k=1, p=0, temp=1`), shipped | **1.108 ms** | 1.129 ms | 32/32 |
| force-argmax (untilize + `ttnn.argmax`) | **1.084 ms** | 1.060 ms | 32/32 |

Force-argmax is 0.024 ms faster, i.e. 2% of the sampler and **0.1% of the
22.99 ms token-out step**. It is not selected, for two reasons that outweigh
that margin: it is greedy-only, so any top-k/top-p request flips
`force_argmax_sampling` and `reset_sampling_params` then calls `reset_trace()`
(`models/common/sampling/generator.py:270-272`), releasing and recapturing
every sampling trace; and its in-place `tt_out_tok` writeback has a documented
reliability problem under async scheduling
(`models/demos/gemma4/tt/generator.py:1141-1148`). With a preallocated rank-4
buffer it *did* write correctly here, since `ttnn.argmax` does not validate the
preallocated output shape, so that hazard is latent rather than immediate. `test_topk_topp_sampling_runs_and_greedy_still_works`
shows the shipped path serving `temperature=0.8, top_k=20, top_p=0.9` and then
returning bit-identical greedy tokens. It asserts token equality, not trace
ids; the no-churn property is structural - `_SamplingArgs` supplies no
`model_config`, so `_allow_force_argmax_sampling` is False, `force_argmax_sampling`
never flips, and `reset_sampling_params` therefore never reaches `reset_trace()`.

## Split-sampling trace contract

Token-out decode is two cooperating traces over persistent device tensors:

1. the **model** decode trace: `embedding -> 47 layers -> final norm -> LM head`,
   deriving the RoPE index from the current position on device on the way in and
   ending with `ttnn.plus_one(cur_pos, skip_negative_entries=True)`, returning
   sampler-ready logits `[1, 1, 32, 154880]`;
2. the **sampling** trace: `SamplingGenerator.capture_trace(logits=<that exact
   tensor>, tt_out_tok=<the persistent decode token tensor>)`.

Both are captured once at `build_generator` time, on dummy state (token 0 at
position 0 of a fresh cache), with the sampler pre-compiled first while no
trace is live.

Evidence (`test_split_sampling_trace_feedback`, `test_unchanged_and_changed_page_table`,
`test_no_host_fallback_during_traced_decode`):

```
trace feedback inputs: [6724, 773, 279, 45681]  positions: [80, 81, 82, 83]
              outputs: [773, 279, 45681, 18611]
```

* the sampled token of step N is byte-for-byte the token input read by step
  N+1, with no host reconstruction;
* the current position advances on device, and the RoPE index the layers
  consume - recomputed from it inside the same graph - tracks it exactly
  (asserted per step against `decode_rope_indices`);
* over a measured 128-token generation the host-work counters are
  `model_trace_replays 127, sampling_trace_replays 127, eager_decode_steps 0,
  token_input_refreshes 2, position_refreshes 2, rope_index_refreshes 0,
  page_table_refreshes 0, device_synchronizations 1, token_readbacks 128,
  full_logits_readbacks 0, host_argmax_calls 0, kv_cache_resets 1,
  trace_recaptures 0` (`perf.json`). The two token/position refreshes are the
  request boundary (reset + the prefill handoff), the one synchronization is
  `reset()` draining the cache zeroing, and `trace_recaptures 0` says this
  prompt shape was already warmed. None of them scales with generation length,
  and inside the decode loop the counters are exactly 0;
* unchanged page table: zero page-table copies across the steady-state loop.
  Changed page table: remapping every physical block through
  `refresh_page_table` and re-prefilling produces the identical token sequence
  from the same captured trace;
* tripwires on `ttnn.from_torch / to_torch / as_tensor /
  copy_host_to_device_tensor` fire **zero** times inside a traced decode step,
  and exactly one `to_torch` for the caller-visible token read.

Both are re-captured, in place, if a prefill compiles a program while they are
live: a newly cached program keeps a device buffer for the process lifetime, and
Metal treats anything allocated while a trace is registered as unsafe for that
trace. `_maybe_recapture_after_compile` compares
`mesh_device.num_program_cache_entries()` against its value at capture time, so
the trigger is exact rather than a guess, and it is wired into every prefill
entry point. It costs **178.6 ms** once per new prefill shape
(`first_use_ttft.json` at prompt 4300; `perf.json` measures 175.8 at prompt
3000) and is timed apart from both TTFT and the decode rate. Verified with `TT_METAL_TRACE_ALLOC_TRACKING=1`
in `trace_alloc.json`; work log FM-016.

A `sampling_params` change is the other way a capture could happen at the
wrong moment, and it needs its own rule. `SamplingGenerator` keys its trace on
`(penalties, log_probs, force_argmax, bucket)` and its `reset_trace` clears
every key, so **at most one sampling mode is captured at a time** and any
change - in either direction, including back to greedy - selects a mode with
no trace. Asking `sample` to trace that mode would start a capture with the
model decode trace still live. Asking it to sample eagerly forever would be a
silent permanent slowdown. So the step samples eagerly **once**, counts it as
`eager_sampling_steps`, and then captures the new mode through
`recapture_decode_traces`, which releases the model trace first. That one
eager step is load-bearing: it compiles the mode's programs with
`count_tokens=True`, which the captured body needs and
`SamplingGenerator.precompile` (`count_tokens=False`) does not compile, so
capturing before it fails with `Cannot load new binaries during trace
capture`. Measured on the 47-layer model, four steps per direction:
`eager_sampling_steps 1, sampling_trace_replays 3, trace_recaptures 1` each
way, with the greedy tokens bit-identical across the round trip
(`test_sampling_mode_change_never_captures_under_a_live_trace`); re-run with
`TT_METAL_TRACE_ALLOC_TRACKING=1`, where every post-recapture replay is
accepted. The shipped measured path never enters it:
`perf.json` reports `eager_sampling_steps 0` over the 128-token generation.
A mode the recapture cannot capture is remembered and never retried, so a
failure costs one eager step rather than one per step.

`SamplingParams.seed` is **refused**, not honoured and not ignored.
`set_sampling_params` raises `ValueError` for a non-`None` seed, because this
generator applies params through `reset_sampling_params`, which never reaches
`seed_manager.reset_seed`; only `apply_prefill_state` does. A seed set here
would leave `has_active_request_seed()` false and the draw unseeded, which a
caller asking for a reproducible stream cannot detect
(`test_request_seed_is_refused_rather_than_silently_ignored`). Wiring
`apply_prefill_state` into the request boundary is stage-07 work; work log
FM-023.

Because the trace advances the position itself, the loop also has to know where
the context ends: the paged cache and page table only represent
`[0, 202752)`, and driving `paged_update_cache` past that wedges the device
instead of raising (work log FM-013). Replays therefore go through
`GLM47FlashGenerator.replay_decode_trace`, which keeps a host mirror of the
positions and refuses the step that would leave the range, with an inactive
(`-1`) slot never counting as out of range. `decode_step_traced` delegates to
it, and no full-model test or probe calls `ttnn.execute_trace` directly, so the
guard cannot be stepped around
(`test_traced_decode_loop_stops_at_context_end`).

## Performance accounting

```
layer-stack lower bound   46 x 0.491 (moe) + 1 x 0.447 (dense) = 23.033 ms/token
traced model decode                                              21.758 ms/token
  + sampling trace                                              + 1.124 ms
  + token readback                                              + 0.112 ms
  = token-out decode                                              22.994 ms/token
```

The full model runs **below** the naive layer-stack lower bound. That is
expected and not an error: the 0.491 ms per-layer figure is a single-layer
traced replay measured through the decoder-stage harness, so its per-replay
dispatch overhead is counted 47 times in the sum, while the full model pays it
once for a 47-layer trace. Both full-model-only terminal costs are measured on
top of the stack figure and are small:

* **LM head** 872.4 us device per step in the reduced profile (x1.0 call/step,
  51.1% of that 2-layer window, 3.8% of the 47-layer token-out step). Roofline
  for its 337 MB bf8 weight read at 512 GB/s is 658 us, i.e. 75% of that
  ceiling; `tt-perf-report`'s own column reads 70.69-71.10% (361.9-364.1
  GB/s) across the eight LM-head rows of the decode-model window, and 70.99%
  for the single row in the prefill window, because it models the p150
  differently. Both denominators are quoted so neither is implied. bf4 measures 628 us but is held back for
  LM-head accuracy (a datatype-sweep candidate).
* **Two reduced-probe wall clocks differ by 5.9%.**
  `perf_reduced_decode.json` measures the same shipped 2-layer configuration
  at 1.890 / 2.979 ms while `logits_memory_ab.json`'s L1 arm measures
  1.788 / 2.911 ms, outside both recorded spreads. The difference is the
  process: `perf_reduced_decode.json` is written by the Tracy run, so its wall
  clock is taken inside a device-profiler-enabled process, whereas the A/B
  probe and `decode_position_scaling.json` (1.81 ms at the same position and
  cache) are not. The two un-profiled measurements agree with each other, so
  read the profiled pair as device-op attribution and the un-profiled ones as
  latency.
* **Sampler** 1.124 ms wall clock, **4.9%** of the token-out step. Its device
  footprint is the whole difference between the two profiled windows, which is
  the honest way to state it: **1046.1 us/step**, 38.0% of the 2-layer
  token-out window (2754.4 minus 1708.3 us/step). Of that, 861.0 us is in
  named sampler ops (`TopkLargeIndicesDeviceOperation` 664.5 us over 4 calls,
  one per vocab chunk; `TopkRouteFinish` 82.2; `TopkRoutePrep` 68.6;
  `SamplingDeviceOperation` 27.3; `ManualSeedDeviceOperation` 18.4) and the
  remaining ~185 us is the sampling graph's support and fallback traffic that
  the op names do not label: `SliceDeviceOperation` +55.6 us over 4 extra
  calls (the 4-chunk `ttnn.split`), a `CopyDeviceOperation` 40.4 us that
  appears only in this window (19.8 MB at ~490 GB/s, i.e. the logged
  `migrating L1 input (9912320 B) to DRAM`), plus `BinaryNg` +36.7,
  `Typecast` +32.6 and small Reduce/Untilize/Concat/Unary deltas. 1046.1 us is
  also the figure consistent with the 1.124 ms wall clock; an earlier revision
  of this report quoted the 861 us subtotal as the whole footprint, which left
  the rest unexplained. Two stage-07 levers follow from it: the per-step
  `ManualSeed` is pure waste in a greedy (`k=1, p=0, temp=1`) trace, and the
  `CopyDeviceOperation` is the L1 migration that `logits_memory_ab.json`
  measures both ways. It does
  not dominate token-out decode, and the alternative (force-argmax) was
  benchmarked and is 0.024 ms faster - rejected on trace behaviour, not shape
  (see the Sampler section).
* `ttnn.plus_one`, the RoPE-index derive (`clamp` + `typecast` + `reshape`),
  the token slice and the 31-row logits pad are single-digit microseconds
  each in the reduced profile, below the resolution of the 47-layer
  wall-clock measurement.

`tt-perf-report` evidence uses the **reduced** profiling variant (HF layers 0
dense + 1 moe, real embedding / final norm / LM head / sampler / cache / page
table): `doc/full_model/tracy/{decode_model,decode_tokenout,prefill}_perf_report.{txt,csv[.gz],png}`
and `perf_report_summary.json`, from `tests/test_full_model_profile.py` under
Tracy. The all-layer stack is deliberately not profiled (≈3200 device
ops/step, multi-GB dumps).

The profiled windows run 8 iterations with an explicit `ttnn.ReadDeviceProfiler`
after each, because the device marker buffer overflows otherwise; the captures
are complete, with the once-per-step LM head appearing exactly 8 times in each
decode window (1264 and 1600 rows, device 1708.3 and 2754.4 us/step,
consistent with the 1.888 / 2.982 ms wall clock in `perf_reduced_decode.json`).
 The run log still contains 295
`Profiler DRAM buffers were full, markers were dropped!` lines: they all come
from the *un-flushed* wall-clock loops that run before the first signpost
(log lines 47-341, ahead of `PERF_FM_DECODE_MODEL` at line 342), so no
reported window is truncated. `perf_report_summary.json` is regenerated by
`tests/summarize_perf_report.py`, which normalizes by that anchor count rather
than by an assumed iteration count and records `anchor_calls_in_window` so a
truncated capture cannot be read as a fast one. Wall clock for the same windows
is measured *before* them with no profiler flushes in the loop, and the
per-window `op_to_op_gap` in the summary is dominated by those flushes - it is
instrumentation, not the real dispatch gap.

Prefill is the weak side, and it degrades with depth: 383.0 tok/s at prompt 128
(334.2 ms), 433.0 tok/s at prompt 3000 (6932.6 ms, physical 3072), and 90.6 tok/s
at the full 202733-token context (2236.6 s). Decode degrades far more gently -
1.81 -> 6.66 ms per traced step from position 128 to 202751 on the 2-layer probe
(`decode_position_scaling.json`), and 136.3 ms/token measured at full context on
the 47-layer model.
The reduced-profile prefill window shows the two sparse expert matmuls at
**48.3%** of device time (3047.0 + 1983.1 of 10411.9 us/step). `tt-perf-report`
omits DRAM utilisation for those rows (it cannot know the active expert count,
hence its `nnz` warning), so no bandwidth figure is claimed for them. See
Limitations.

## Qualitative evidence

The checkpoint ships a chat template ending in `<|assistant|><think>`, so it is
treated as an instruct/reasoning model and **every** prompt in this stage is
rendered with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`;
the HF control uses the identical token ids. Prompt-format metadata:
`qualitative/qualitative_prompt_format.json`.

* `run_autoregressive` (chat-rendered prompt, 256 tokens, HF control on CPU):
  `readiness_autoregressive/{hf_completion.txt,tt_completion.txt,autoregressive_meta.json}`.
  TT output is a clean, well-structured reasoning trace, same language, same
  register, same list structure as HF, no repetition and no prompt echo.
* Shared qualitative suite (`models/common/readiness_check/vllm_prompts.txt`,
  6 prompts, 128 tokens each, HF control + TT):
  `qualitative/qualitative_side_by_side.txt`, `qualitative_outputs.json`.
  All six TT completions are coherent, on-topic and in English; greedy prefix
  agreement with HF runs 8-45 tokens before the chains separate.
* `check_degenerate_output.py --scope autoregressive --missing-artifacts critical`:
  **clean**, with adjacent duplication 0.0 and trigram loop fraction 0.0246.

**Divergence read.** Free-running greedy TT and HF separate after 8-45 tokens.
That follows directly from the measured 85% per-step top-1 agreement (a greedy
chain has probability 0.85^n of staying identical), and the divergence is
benign: both continuations stay on-topic, well-formed and in the same language,
and the branch points are ordinary near-tie word choices ("Genre" vs "Tone",
"Analogy 2: A GPS" vs "Analogy 2: A classroom"). No wrong-language drift, no
early collapse, no doubled tokens, no control-token leakage.

## Runtime fallback audit

* **Traced decode**: monkeypatched tripwires on `ttnn.from_torch`,
  `ttnn.to_torch`, `ttnn.as_tensor`, `ttnn.copy_host_to_device_tensor` record
  **zero** calls during `decode_step_traced()`, and exactly one `to_torch` for
  the token read (`test_no_host_fallback_during_traced_decode`).
* **Static**: `tt/model.py` and `tt/generator.py` have no module-level torch
  import; torch appears only inside weight loading, host tensor staging and
  the explicit host-sampling mode (`test_static_no_torch_in_runtime_modules`).
* **Cache ownership**: both modes are supported and tested. The high-level API
  owns a cache and page table (`prefill_logits`, `generate`); the low-level API
  takes the caller's (`prefill_forward`, `decode_forward`,
  `bind_decode_state`, `refresh_page_table`), which is what a vLLM adapter will
  drive. Ownership mistakes fail loudly rather than silently decoding against
  the wrong buffers: `bind_decode_state` raises if it is called after
  `capture_decode_trace` with a different cache or page-table tensor, traced
  `decode_forward` raises on a caller-supplied device page table that is not
  the bound one, and passing a torch page table while a caller-owned device
  table is bound also raises. A generator-owned torch page table is copied only
  when it actually changed, so the steady-state loop performs no page-table
  copies.

  One caveat for a caller that mixes the two levels: if a cache is already
  bound through `bind_decode_state`, the high-level entry points *adopt* it
  rather than allocating a second one (`_ensure_owned_state`), because a second
  full-context cache costs another 5.4 GiB. The consequence is that `generate`,
  `prefill_logits`, `warmup_prefill` and `reset` then read, write and zero the
  **caller's** buffers, and `reset()` in particular clears a cache the caller
  owns. A serving adapter should pick one level per generator instance and stay
  on it. Adopting is the right default for a single-chip 32 GB budget, but it
  is not a copy.
* **Host-logit boundary**: the only full-logits readbacks are the explicitly
  host-side paths: `prefill_logits` (the readiness prefill gate), the
  `return_logits=True` low-level `decode_forward`, and the host-sampling
  compatibility mode. The measured token-out path reads one uint32 word per
  token. Counters distinguish them (`full_logits_readbacks`,
  `host_argmax_calls`).
* **Trace lifecycle**: measured, not assumed. A prefill that compiles a
  program while the traces are live leaves that program's device buffer on the
  unsafe side of them, so the generator re-captures
  (`_maybe_recapture_after_compile`, triggered by an exact
  `num_program_cache_entries()` comparison rather than a guess). Two things
  keep the cost off the common path. Every terminal program family is keyed
  on the *bucketed physical* prefill length rather than the logical one (the
  whole-tile slab of the token-out path, and the tile-aligned walk of the
  host-logits path that `prefill_logits` and the low-level `prefill_forward`
  use), and `warmup_terminal_shapes` compiles all of them for the five buckets
  before capture. So **a prompt inside one prefill chunk compiles nothing and
  re-captures nothing on any of the three entry points**, which
  `test_host_logits_paths_compile_nothing_at_an_unaligned_length` asserts
  against `num_program_cache_entries()`:
  `first_use_ttft.json` measures 583.7 ms then 583.5 ms at prompt 154 with
  **0 new programs compiled** on either request (a 0.2 ms "penalty", i.e.
  nothing outside run-to-run spread), and
  `test_single_chunk_prompt_shape_does_not_recapture` asserts the
  program-cache counter does not move. A prompt past one chunk still compiles
  its chunk-offset programs on first use (**18** of them at 4300 tokens,
  recorded by the probe) and pays one recapture, measured at **178.6 ms**, for
  a 184.1 ms first-use penalty once per new chunk depth. Bounding
  that too needs the terminal path to slice the last chunk before the tile,
  which is stage-07 work.
* **The recapture is non-destructive by construction.** Its warm pass runs
  with every slot at position `-1`, the inactive marker the decode path already
  honours, because at position 0 it would write one KV row per slot and corrupt
  any slot sitting elsewhere, which at batch > 1 with mixed prompts is the
  normal case. Skipping the warm pass is not an option (Metal refuses to
  capture an uncached program). Both halves are asserted:
  `test_traced_replay_with_all_slots_inactive_writes_no_cache_row` reads three
  cache blocks back after three inactive traced steps, and
  `test_recapture_mid_decode_leaves_a_deeper_slot_untouched` runs batch 32 with
  a recapture injected mid-decode and requires bit-identical tokens. Metal registers a trace as
  active from `end_mesh_trace` until it is released and flags every allocation
  made in that window as unsafe, because such a buffer can land on an address
  the trace's own freed intermediates used and a replay then writes over it.
  Run with `TT_METAL_TRACE_ALLOC_TRACKING=1`, `ttnn.execute_trace` refuses a
  replay when such a buffer is still alive, and every shipped path here passes:
  `trace_alloc.json` (four arms) and `logs/trace_alloc_full_model.log` (the
  full 47-layer build, prefill and 128-token generate). Two things had to
  change for that: the cache-reset zero buffer is allocated before capture
  instead of on first `reset()`, and a prefill that compiles programs after
  capture triggers `recapture_decode_traces()` (178.6 ms, once per new
  prefill shape, timed separately from TTFT, and never for a prompt inside one
  chunk). Work log FM-016 has the measurements,
  including what the untreated hazard looks like.
* **Host-sampling compatibility mode**: `generate(..., host_sampling=True)` or
  `build_generator(..., host_sampling=True)`. It selects the *same* tokens as
  the on-device sampler (`test_host_sampling_compatibility_mode`), is flagged
  in the counters, and restores the previous mode on exit so it cannot leak
  into a measured run. A generator whose *first* capture happened in
  host-sampling mode used to keep sampling untraced for the rest of the
  process, correct but slower and silent about it;
  `_ensure_sampling_trace` now captures on demand
  (`test_sampling_trace_is_captured_on_demand_if_capture_skipped_it`).
* **Reset**: `reset()` zeroes all 47 caches in place (buffer addresses
  preserved, so captured traces stay valid), clears the token/position state
  and the penalty counters, and keeps weights, device buffers and traces.
  `test_reset_clears_cache_and_state` re-runs a prompt after an unrelated one
  and gets identical tokens.
* **One op-internal device fallback, disclosed and measured.** `TTSampling`
  starts by running `ttnn.split` over the 9.9 MB sampler-ready logits tensor.
  With the logits in L1 that op logs
  `L1 budget exceeded (need ~9945088 B, have 1248256 B for 4 chunks); DRAM
  downgrade` and `migrating L1 input (9912320 B) to DRAM before slice fallback`
  and takes its slice path, inside the captured sampling graph. This is a
  device-side op fallback, not a host one, and it is on the measured token-out
  path. Producing the logits in DRAM instead removes it entirely (0 warnings)
  and the profiler attributes a 40.4 us/step `CopyDeviceOperation` to the L1
  arm that the DRAM arm cannot need, so the two are not obviously ordered.
  `logits_memory_ab.json` measures both, with 64 repeats per window and each
  arm's own spread recorded next to its mean, from the committed
  `probe/logits_memory_ab_probe.py`; the first version of this measurement was
  one sample per arm and 34 us apart, which is not enough to choose on. Read
  the deltas there against those spreads. What the decision actually rests on
  is that both arms produce **identical tokens**, which the probe asserts: L1
  is kept as the pre-existing behaviour, the fallback is disclosed rather than
  paid for, and
  `GLM47FlashModel.from_pretrained(decode_logits_in_dram=True)` reproduces the
  other arm in one argument. Stage 07 should re-run the A/B if it adds L1
  pressure around the terminal path.
* **Watcher**: `TT_METAL_WATCHER=2` over the reduced prefill + traced-decode
  smoke, the reduced traced-decode benchmark (131 trace replays at the full
  202752-token cache), and a full 47-layer build + prefill + 128-token
  generate. **No watcher exceptions, asserts or sanitize errors**
  appeared in any of the three (`logs/watcher/*.log.gz`). Watcher and profiler runs
  were kept separate.

## Determinism

* greedy generation repeats identically for the same prompt
  (`test_greedy_generation_deterministic`);
* `prefill_logits` is bit-identical across runs
  (`test_prefill_logits_bitwise_reproducible`);
* the same (token, position, cache prefix) yields the same decode token after
  an unrelated prompt in between (`test_decode_logits_independent_of_cache_history`);
* a user placed in batch slot 7 predicts the same first token as the same
  prompt run alone in slot 0 (`test_batch_slot_isolation_matches_single_user`).

## Tests

| suite | result |
|---|---|
| `tests/test_full_model.py` (batch 1, all 47 layers) | **47 passed** (251 s, `logs/fm023/pytest_full_model_only.log`) |
| `tests/test_full_model_perf.py` | **2 passed** (216 s, `logs/fm023/pytest_full_model_perf.log`) |
| `tests/test_full_model_batch.py` (`GLM47_FM_BATCH=32`, `GLM47_FM_BATCH_SEQ=8192`, echoed into the log) | **10 passed** (307 s) |
| `tests/test_prefill_padding.py` (bucket-padding non-leakage, the supported-context boundary, the inactive-slot cache proof) | **13 passed** (46 s, `logs/pytest_prefill_padding.log`) |
| `tests/test_full_context.py` (202733-token prefill, decode to 202751, needle read) | **3 passed** (40 min), periodic continuation 9/9, needle top-1 correct |
| `test_full_model.py` + `test_prefill_padding.py` in one session (`logs/fm023/pytest_full_model_and_prefill_padding.log`) | **60 passed** (294 s) |
| `tests/test_full_model_profile.py` under Tracy | **2 passed** |
| `run_prefill_check` / `run_teacher_forcing` / `run_autoregressive` | pass, above bar |
| `check_degenerate_output.py` | clean (exit 0) |
| `.agents/scripts/check_context_contract.py --stage full-model` | OK, target = supported = 202752 |

Every row above comes from one sweep, `tests/run_evidence_sweep.sh` (committed,
so the ordering and flags are reproducible rather than reconstructable), run
against a committed source tree with no stage source changes in it - with three
exceptions, all from FM-023, which added two tests to
`tests/test_full_model.py` after that sweep: the three rows citing
`logs/fm023/` were re-run afterwards, on the same device, one suite at a time.
Their sweep-run counterparts are still in `logs/` (`pytest_full_model_only.log`
at 45 passed, `pytest_full_model.log` at 58) and are the pre-FM-023 record.
`logs/fm023/` also holds the same sampling tests re-run under
`TT_METAL_TRACE_ALLOC_TRACKING=1`, which is the unsafe-allocation gate for the
mid-loop recapture FM-023 added. That is
recorded rather than asserted, and the recording is deliberately not just
`git status`: `.git/info/exclude` lists `models/autoports/`, so a plain
`git status` over this directory reports no untracked file at all.
`logs/sweep_provenance.log` therefore captures, at the start and end of the
sweep, `git rev-parse HEAD`, the tracked changes, `git ls-files --others`
(without `--exclude-standard`) and a **sha256 of every source file the sweep
depends on**: the stage's `tt/`, `tests/` and `probe/`, the sweep script
itself, `models/common/readiness_check/` and the context-contract checker. In
the run behind this report those hashes are identical at both ends. The
untracked-file list is *not*: the opening block lists the earlier-stage watcher
and Tracy leftovers under `doc/{functional,fused,optimized}_decoder/`, and the
closing block adds the artifacts the sweep itself wrote for the first time.
That is the mechanism working, not a problem, and it is what the previous
revision of this paragraph got wrong twice: it asserted the lists were
identical. `logs/sweep_run.log` is tracked, so it appears in neither list.

Two of the sweep's own steps read this report, the work log and the capability
contract: the context-contract check and `tests/check_report_numbers.py`. They
run last, and they are re-run after the documentation pass, so
`logs/check_context_contract.log` and `logs/check_report_numbers.log` are from
that re-run rather than from the middle of the sweep. Nothing else is edited
after a sweep. **The in-sweep figure check is expected to fail on a sweep that
moved any number**, and `logs/sweep_run.log` records that: its `reportnums`
step is what tells the stage owner which figures to regenerate. The committed
`check_report_numbers.log` is the clean pass afterwards.

`check_report_numbers.py` is the mechanical answer to eight rounds of stale
figures, and it is an *absence* check, not a presence check. Its first version
took values out of the artifacts and required the string to appear somewhere,
which cannot see a contradiction: that is how "584.0 ms" survived in one
paragraph while the correct "583.6 ms" satisfied the check in another. It now
scans every measurement-shaped literal in this report, the work log, the
capability contract and the `tt/*.py` module docstrings, and requires each to
resolve to a value some committed artifact contains. Anything else has to be
named in its `ALLOWED` table with a reason, which `--list-allowed` prints; most
of those are figures from earlier sweeps that the work log quotes in
comparison columns on purpose.

On top of that, every generated JSON carries a `source_manifest` block with
sha256 prefixes of `tt/*.py` and of the script that produced it, so any single
artifact can be tied to exact source without trusting the sweep log. All three
`TT_METAL_WATCHER=2` runs reported 0 faults.

Not every file in `doc/full_model/` is stamped, and the exceptions are named
rather than glossed:

| file | why it has no `source_manifest` |
|---|---|
| `greedy_sampler_benchmark.json` (FM-008b) | predates the mechanism; from a probe untouched since it ran, and it exercises `models/common/sampling` rather than `tt/*.py` |
| `degenerate_check.json` | written by the shared `models/common/readiness_check/check_degenerate_output.py`, which this stage does not own. Its own `logs/check_degenerate_output.log` is the run record |
| `qualitative/hf_control.json` | the CPU HF control, reused across sweeps with `--skip-hf` because it is a property of the checkpoint, not of the port. The TT side (`qualitative_outputs.json`) is regenerated every sweep and *is* stamped |
| `doc/context_contract.json` | hand-maintained, not generated. Its `full_model.provenance` block names the commit and points here |

One log is older than the sweep: `logs/generate_aime24_reference.log`, the
one-time CPU generation of the committed AIME24 reference.
`logs/pytest_prefill_padding.log` used to be older too; the sweep now runs that
suite standalone as well as in the combined session, so it is a sweep product
like the rest.

Finally, the repo's `end-of-file-fixer` hook appends a trailing newline to four
files whose writers omit it, which happens at commit time and therefore after
the sweep: `degenerate_check.json` and the three
`readiness_autoregressive/` artifacts. The change is one byte per file and the
run logs corroborate the contents.

## Limitations and follow-ups (all disclosed, none blocking this stage)

1. **Prefill throughput 383-433 tok/s at short prompts, 90.6 tok/s at the
   full context** (`perf.json`, `full_context.json`). The reduced-profile prefill window is
   48% two sparse expert matmuls, and no bandwidth figure is claimed for them
   because `tt-perf-report` omits DRAM utilisation for sparse rows. The prefill
   sparse geometry was
   tuned for 1024-token chunks at the optimized-decoder stage, and the flat
   prefill projections deliberately keep default program configs below 10
   M-tiles, so short prompts get the untuned path. This is the main
   optimized-full-model (stage 07) target.
2. **Prefill program compilation** is a construction-time cost that a serving
   deployment must keep paying up front: 71.8 s on a cold JIT cache, 4.5 s on
   a warm one, for the five bucket shapes and every terminal program family. A
   prompt whose *physical* shape was
   not warmed (any multi-chunk length) adds **+1352 ms** to its first TTFT on
   a cold cache (7833.5 vs 6481.1 ms at prompt 3000), plus a one-time
   **178.6 ms** trace recapture that the post-capture compile makes necessary
   (see the trace-lifecycle bullet in the fallback audit); at prompt 4300 the
   two together are a 184.1 ms first-use penalty over 18 newly compiled
   programs. A prompt inside one prefill chunk pays neither: every terminal
   program family is warmed at construction, and `first_use_ttft.json` records
   **0 new programs** and a 0.2 ms penalty at prompt 154, i.e. nothing. `build_generator(warmup_prefill_lens=[...])` takes
   the exact logical lengths a deployment expects and pre-pays the multi-chunk
   case too, before the traces are captured. Bucketing is what bounds this to `{2048} | buckets` shapes
   instead of one per prompt length; without it the first request at each new
   length paid it (measured 16.7 s readiness TTFT before the fix).
3. **LM head at bf8** costs 878 us/token at the shipped `in0_block_w` of 4;
   `in0_block_w` 2 is 10 us/token faster and was **measured and refused**,
   because it costs prefill top-1 0.880 -> 0.830 and teacher-forced top-5
   1.000 -> 0.990 (work log FM-021). bf4 measures 627 us
   (`head_probe.json`), a 251 us/token saving, i.e. 1.1% of the token-out
   step. Deferred to the datatype sweep
   (stage 08) **with no accuracy datapoint yet**: this stage carries the
   optimized decoder's dtype policy forward unchanged and does not re-select
   it, so the honest statement is "not measured", not "measured and worse".
   The gate it has to clear is the one this stage already uses: top-5 >= 0.98
   and top-100 = 1.00 against `readiness_aime24_chat.refpt`, on real weights.
   `lm_head_dtype=bfloat4_b` is a `from_pretrained` argument, so the
   measurement is one readiness run once stage 08 owns the decision.
4. **On-device logprobs are unavailable at 1x1**:
   `LogProbsCalculator._is_supported` requires 8 or 32 devices
   (`tt_log_probs.py:419-425`). Logprobs must be computed on host until that
   changes.
5. **`TTPenalties` is constructed unconditionally** by `SamplingGenerator`,
   allocating five `[32, 154880]` int32 buffers (99,123,200 B) even for a
   greedy-only run. Accounted for in the budget; a lazy construction would
   return it.
6. **The sampler always processes 32 logits rows**, even at batch 1: `ttnn.sampling`
   takes one row per user and `TTSampling` floors its batch to a full tile, so
   the four 38720-wide `ttnn.topk` calls (664.5 us/step) do 32x the work a
   single active user needs. That is the concrete stage-07 lever on the 1.124 ms
   sampler cost, and it needs a `TTSampling` change rather than a model-side one.
7. **Batch 32 is tested at 8192 tokens/user**, not at the full context. The
   limit is physical: 174 GiB of cache would be needed. Recorded in
   `context_contract.json`.
8. **A fifth of the profiled prefill window is marked `SLOW`.**
   `tt-perf-report`'s `Bound` column says those rows are neither compute nor
   bandwidth bound, i.e. the geometry leaves cores idle, and
   `perf_report_summary.json` now counts them per window so the number is in
   an artifact rather than a CSV column: **17 of 110 prefill rows, 21.2% of
   that window**, 112 of 1264 decode-model rows (15.0%) and 112 of 1600
   token-out rows (9.4%), all matmuls, and all of them carried-forward decoder
   matmuls rather than full-model-only ops (the LM head and the sampler are
   not among them). This is the concrete
   matmul-geometry target for stage 07, alongside the two sparse expert
   matmuls in item 1; nothing here is fixed in this stage.
9. The `tt-perf-report` warnings about `SparseMatmulDeviceOperation` nnz and
   unclassified TopK/Sampling ops are tool-side classification gaps; row
   timings are unaffected (work log FM-010).
10. **Prefill call-to-call spread is small, but read it from the artifact.**
   With the timer bracketed by `ttnn.synchronize_device`, `compile_cost.json`
   records `repeat_spread_pct` of 0.0-0.1% at both prompt 128 and prompt 3000,
   and `perf.json`'s back-to-back 3000-token pair (6932.6 / 6928.0 ms) differs
   by 0.07%. That is tight enough to compare single calls, which the
   cold-versus-warm rows above rely on. It was not true of the earlier
   unsynchronized probe, so treat any prefill figure from before FM-015 as
   host-enqueue time rather than prefill time. The decode numbers are averages
   over 64 replays and are stable to <1%. Prefill cost *is* prompt-content
   dependent, though, because MoE routing is: at the same prompt length and the
   same physical shape, this report's prompt prefills in 332.8 ms and
   `measure_cold_compile.py`'s in 313.7 ms, 6.1% apart
   (`perf.json: ttft_breakdown_ms`). Compare prefill numbers across probes only
   at the same prompt text.

## Known limitations

Recorded rather than fixed. Every item here is a freshness, provenance,
coverage or documentation-polish issue, not a correctness defect: the
functional gates in the Tests table are green, and ten independent
`$stage-review` rounds have been run against this stage (work log FM-011,
FM-012, FM-015 through FM-023). Rounds 1 through 7 each found at least one
correctness defect, each of which was fixed with a regression test; round 8
found none; round 10 found one, the mid-loop sampling-mode capture, which is
fixed with two regression tests and recorded in FM-023 along with the silently
ignored request seed that diagnosing it turned up. The stage's review budget
caps further rounds, because each evidence sweep costs 90 minutes of hardware
and the remaining findings do not change what the model does.

* **The figure check cannot tell "quoted as history" from "stale".**
  `tests/check_report_numbers.py` requires every measurement in this report,
  the work log, the capability contract and the `tt/*.py` docstrings to resolve
  to a committed artifact value, and it currently clears 545 of them. The
  exceptions live in an `ALLOWED` table with a mandatory reason
  (`--list-allowed`), and that table has grown to 318 entries, almost all
  superseded figures the work log quotes in comparison columns on purpose. A
  literal that is allowlisted today and becomes wrong tomorrow would not be
  caught. Making the allowlist position-aware (a value permitted only inside
  the section that labels it as history) is the obvious next step and is not
  done.
* **Each artifact comes from one run.** The 202733-token full-context run (40
  minutes) and the batch-32 residency (24.462 GiB) each rest on a single
  measurement. The periodic-continuation gate is 9/9 in that one run and the
  needle read is one sample. Nothing here is averaged across sweeps except the
  decode figures, which are 64 replays inside one run.
* **Most artifacts carry the pre-FM-023 `tt/generator.py` hash.** FM-023
  changed that file after the sweep, so every stamped artifact's
  `source_manifest` went stale and `logs/sweep_provenance.log` is a
  pre-change record. Four artifacts were regenerated, the ones the change
  could plausibly move: `trace_alloc.json` (verdict still `clean`),
  `accuracy.json` and `capacity.json` (**bit-identical values**, only the
  hashes moved) and `perf.json` (run-to-run noise, and the report quotes the
  fresh values). The rest were not, because re-running them is the 90-minute
  sweep the review budget rules out; none of them exercises a
  `sampling_params` change, which is the only path FM-023 alters, and
  `perf.json`'s `eager_sampling_steps 0` is the direct evidence that the
  measured path is unchanged. The three re-run pytest logs live in
  `logs/fm023/` and the Tests table names them.
* **Two artifacts are not from the recorded sweep.**
  `logs/trace_alloc_full_model.log` was re-run on its own after the sweep,
  because a stray signal from a previous session's cleanup terminated the
  sweep's own run of it (work log FM-022); the log says so at the bottom. And
  `trace_alloc.json` with its `logs/trace_alloc_probe.log` is the FM-023
  re-run described above. Everything else in `doc/full_model/` is from the
  single run recorded in `logs/sweep_provenance.log`.
* **A request seed can be refused but not honoured.**
  `set_sampling_params` raises on `SamplingParams.seed` rather than dropping
  it silently, which is the honest half. Actually honouring it means driving
  `SamplingGenerator.apply_prefill_state` at the request boundary, so the
  seed manager is reset and `has_active_request_seed()` becomes true, and
  that is a serving-integration change this stage does not make. Until then a
  caller wanting reproducibility gets it from greedy params, not from a seed
  (work log FM-023).
* **The mid-loop mode-change cost is counted, not timed.** Changing
  `sampling_params` inside a decode loop spends one eager sampler step plus
  one `recapture_decode_traces` on a single token. The counters say it
  happened (`eager_sampling_steps 1`, `trace_recaptures 1`) and the recapture
  is measured elsewhere at 175.8-178.6 ms per shape, but no artifact times
  that particular token, so the latency spike is stated as its parts rather
  than measured end to end.
* **The shared readiness harness now opens devices differently for every
  model.** `models/common/readiness_check/mesh_device.py` gained
  `--trace-region-size` / `--l1-small-size` with defaults of 90 MB and 32 KiB,
  where ttnn previously defaulted to 0 and 0 (work log FM-008). The direction
  is the safe one and the harness could not honour its own traced-decode
  requirement before, but it is a cross-model default change made from a
  single-model stage, no other autoport has been re-run against it, and this
  stage's own runs pass 350 MB explicitly, so the new default is never
  exercised by this evidence.
* **The bf4 LM head has no accuracy datapoint.** It is 251 us/token faster on
  the largest single op in the decode step and is deferred to the datatype
  sweep (stage 08) with the gate named (top-5 >= 0.98, top-100 = 1.00 on
  `readiness_aime24_chat.refpt`) and the knob exposed (`lm_head_dtype=`). The
  honest statement is "not measured", not "measured and worse". FM-021 is a
  warning about exactly this class of change: an LM-head program-config change
  turned out to be an accuracy change.
* **`in0_block_w` 16, 32 and 64 have a blocker but not an analysis.** They fail
  with a static circular buffer / L1 clash (`program.cpp:1875`), which is
  recorded in `head_probe.json` as the op-contract blocker, but no L1 budget
  was computed to say whether a different core count or output subblock would
  make them expressible.
* **The L1-versus-DRAM logits decision rests on a delta near its own spread.**
  `logits_memory_ab.json` runs 64 repeats per arm and the arms differ by about
  one spread. What the choice actually rests on is that both arms produce
  identical tokens, which the probe asserts; the ordering is not a strong
  result and stage 07 should redo it if it changes L1 pressure around the
  terminal path.
* **One wall-clock pair is profiler-contaminated.**
  `perf_reduced_decode.json` is written by the Tracy run, so its wall clock is
  taken inside a device-profiler-enabled process and reads 5.9% slower than the
  two un-profiled measurements of the same configuration
  (`logits_memory_ab.json`'s L1 arm and `decode_position_scaling.json`). The
  Performance accounting section says which is which; the profiled pair is
  still what the device-op attribution is normalised against.
* **The cache-adoption caveat has no test.** `_ensure_owned_state` adopts a
  caller-bound cache rather than allocating a second full-context one, so
  `generate`, `prefill_logits`, `warmup_prefill` and `reset` then operate on
  the caller's buffers. It is documented in the runtime fallback audit and it
  is the right default for a 32 GB budget, but no test drives the mixed-level
  sequence it warns about.
* **Counters are silent for direct device-helper calls.**
  `full_context.json`'s end-of-run counters report `eager_decode_steps 0` and
  `full_logits_readbacks 0` although the needle read runs one eager
  `_decode_logits_device` and reads a full logits row: the counters only fire
  inside `decode_forward` and `generate`. The artifact says the assertions use
  the periodic phase, whose counters are the meaningful ones.
* **Two pre-artifact figures survive as labelled history.** FM-002's
  "15310 us / 17x" LM-head default-config pair and FM-005's bring-up RoPE
  table were taken before those probes wrote artifacts and do not reproduce
  (`head_probe.json` measures 2476 us and 2.8x; `cache_scaling.json` shows the
  ROW_MAJOR lookup is not flat). Both are kept in the work log with an explicit
  retraction next to them rather than deleted, because the journal is the
  record of what was believed when a decision was made.

## Artifacts

```
tt/model.py, tt/generator.py                     implementation
tests/test_full_model.py                         batch-1 correctness suite (47 tests)
tests/test_full_model_batch.py                   batch-32 suite (own pytest session)
tests/test_full_model_perf.py                    wall-clock perf + capacity
tests/test_full_model_profile.py                 reduced Tracy variant
tests/test_prefill_padding.py                     prefill bucket-padding non-leakage
tests/test_full_context.py                        202733-token prefill + decode to position 202751, needle recall
tests/summarize_perf_report.py                    regenerates perf_report_summary.json from the CSVs
tests/measure_cold_compile.py                     cold/warm JIT-cache prefill compile cost
tests/dev_full_model.py                          fast reduced debug probe
tests/run_qualitative_suite.py                   shared qualitative suite, HF + TT
probe/full_model_head_probe.py                   embedding / plus_one / LM-head sweep
probe/decode_cache_scaling_probe.py              cache-size scaling refutation
probe/decode_position_scaling_probe.py           decode cost vs decode position, to 202751
probe/dram_capacity_probe.py                     allocatable DRAM measurement
probe/greedy_sampler_probe.py                    split-topk vs force-argmax greedy benchmark
probe/trace_alloc_probe.py                       unsafe-allocation accounting per trace id
probe/first_use_ttft_probe.py                    first vs second request at an unwarmed prompt length
probe/logits_memory_ab_probe.py                  L1 vs DRAM sampler-ready logits, both arms with repeats
tt/provenance.py                                 source_manifest, ttnn-free so CSV tools can stamp too
readiness_aime24_chat.refpt                      fresh AIME24 chat-template reference
readiness_aime24_chat.meta.json                  its provenance (tokenizer, template, top-k, transformers)
readiness_autoregressive/                        HF vs TT free-running completions
doc/full_model/accuracy.json                     top-1/5/100, prefill and decode
doc/full_model/perf.json                         TTFT, both decode metrics, counters
doc/full_model/capacity.json                     exact DRAM byte accounting
doc/full_model/perf_reduced_{decode,prefill}.json
doc/full_model/perf_report_summary.json          per-op device time from tt-perf-report
doc/full_model/greedy_sampler_benchmark.json     both on-device greedy strategies, measured
doc/full_model/logits_memory_ab.json             L1 vs DRAM sampler-ready logits, both arms
doc/full_model/compile_cost{,_warm}.json         prefill program-compile cost, cold and warm JIT cache
doc/full_model/full_context.json                 the 202733 -> 202751 full-context run, incl. the needle read
doc/full_model/decode_position_scaling.json      decode ms/token vs position
doc/full_model/dram_capacity.json                measured allocatable DRAM
doc/full_model/trace_alloc.json                  unsafe buffers per trace id, four arms
doc/full_model/first_use_ttft.json               first-use TTFT penalty and new-program count per shape
doc/full_model/head_probe.json                   LM-head geometry sweep, incl. the default program config
doc/full_model/cache_scaling.json                decode cost vs *allocated* cache, and the RoPE table lookup
doc/full_model/tracy/                            tt-perf-report txt/csv(.gz)/png per window
doc/full_model/qualitative/                      prompt format, HF control, side by side
doc/full_model/degenerate_check.json             degeneracy verdict
doc/full_model/logs/                             every run above, incl. watcher/*.log.gz
doc/full_model/logs/fm023/                       the three suites re-run after FM-023, plus the tracker gate
doc/full_model/logs/sweep_provenance.log         HEAD, tracked + untracked files, sha256 of every stage source file, at both ends
doc/full_model/logs/sweep_run.log                the sweep's own stdout: every step's exit code and watcher fault count
tests/run_evidence_sweep.sh                      the sweep itself
doc/context_contract.json                        updated with the full_model section

Every JSON above carries a `source_manifest` block (sha256 prefixes of
`tt/*.py`, of `tt/provenance.py` itself and of the producing script) except the
files named in the exception table in the Tests section. That table, not this
sentence, is the authoritative list.
```
