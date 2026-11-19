# AutoDebug — Stage 10 RW3 / P4 batch32 floor

Inspection date: 2026-11-17. Checkout HEAD: `8fbe38e7ce079a19153b1baad80f270f3d0c3025`, with pre-existing working-tree changes. Scope is **RW3 only**. This investigation read source and saved artifacts and ran standard-library static calculations. It did not run the model, import TTNN, open devices, run a server/profiler, edit source, reset hardware, or relaunch AutoDebug. Three independent xhigh investigators checked evidence, attention, and MoE. The only investigation write is this report; root `AUTODEBUG.md` is preserved.

## Findings

**The batch32 failure is real. Two source-supported scaling defects/opportunities deserve a new, batch32-specific repair campaign: (1) combine's batch assignment sends every nonempty expert to a single untilizer, leaving three available untilizers without expert data work; (2) the 30 linear-attention layers still serialize 32 complete singleton attention paths.** The second includes 960 input normalizations and 1,920 ordered projection scans per batch step. These are accelerator graph costs, even though the graph is captured once.

The strongest small intervention is the combine ownership correction described below. The next bounded graph repair is batching the existing ordered input-normalization sequence, preserving its arithmetic. Neither has a measured saving here. **No source-only result proves that these changes recover the approximately 214 ms / 50.5% step-time reduction required.** A precise kernel attribution and a sufficient repair remain unverified.

Do not rerun the previous single-user experiment and call it P4 remediation. Do not weaken the useful-work convention, change context 262144, precision selection, expert arithmetic or canonical device sampling. The owner's extended wall allowance supersedes the historical optional-search stop notes for this required work; it supplies no performance waiver.

Throughout, `M` means `models/autoports/qwen_qwen3_6_35b_a3b`, `D=M/doc/optimized_vllm`, `R=M/readiness_vllm/optimized_acceptance_eval_final`, and `N=ttnn/cpp/ttnn/operations/experimental/deepseek_prefill`. File/line anchors refer to the inspected tree.

## Direct observations and the requirement

| Observation | Evidence / consequence |
|---|---|
| First output bucket: 2.36 tokens/s/user; aggregate 75.67 | `R/long_answer_load.json`, `data.buckets[0]`. Aggregate is 32 times the unrounded median, not exactly 32 times 2.36. |
| All 32 requests produced 6000 tokens, finish reason `length`, no errors; maximum prompt 74 | `M/doc/eval_readiness/long_answer_load/requests.jsonl`. Total 192000 outputs. Logged prompt-digest multisets match the bound's manifest. |
| Necessary minimum 4.764924390924285 tokens/s/user | `D/eval_batch32_necessary_floor.json`; re-summed below. This is a necessary bound, not actual measured useful traffic or a sufficient passing criterion. |
| P4 exits 2; `functional_floor` is critical and failed | `R/records/p4_long_answer.execution.json`, `R/long_answer_load.json`. Completion, health, resource release, trace stability and no-preemption checks pass separately. |
| Plain/eval comparison TPOT: 421.53 / 421.20 ms | Same result, `data.host_sampling`; ratio 0.999. All 6509 P4 comparison/load decodes use device sampling; none use host sampling. |
| Independent first-1000 server decode-start cadence: 424.4336759 ms/step, 2.3560807185 steps/s | Parsed long-load `decode` events in `R/server/server.log`. Corroborates client 2.36 without relying on the client stream parser. These are host event intervals, not device kernel timings. |

Recomputed necessary-bound arithmetic:

```text
30 * 267,028,640 + 10 * 259,920,000
  + 135,086,080 + 32,768 + 8,192
    = 10,745,186,240 maximum logical useful bytes/rank/batch step
0.10 * 512,000,000,000 / 10,745,186,240
    = 4.764924390924285 tokens/s/user
allowed batch step = 209.86691875 ms
```

The inherited logical convention is `M/tests/multichip_functional.py:12-49`. The bound prices the **maximum** union of experts (256 global, 64 resident per rank) and upper first-bucket position 1074. It does not assert the requests actually touched every expert. Overpricing possible useful bytes relaxes the necessary rate. Each step emits one token for each active user, so step rate already is per-user rate; dividing by 32 would be wrong. Physical padding, repeated reads, intermediate tensors and CCL must not be added to this numerator.

The client bucket measures 999 intervals between output token 1 and token 1000, excluding initial-token wait (`runtime/readiness_check/run_long_answer_load.py:86-114` in installed bringup 0.1.24). Rounded 2.36 implies about 423.729 ms/step and a 50.4714% latency reduction; the independent server cadence implies about 214.567 ms to remove. These slightly different statistics should not be treated as identical precision measurements.

## Actual execution and completion boundaries

The recorded server is full40, TP4/EP4 on P300x2, capacity 32, context 262144, block 32, max batched tokens 2048, async scheduling and decode-only tracing (`R/server/vllm_run_summary.json`). The launch helper sets audit=1, allocation tracker=0, skip-program-cache=0 and removes profiler/Watcher variables (`D/continue_eval_sweep.py:19-39`). Audit overhead is present; a tracked trace walk is not the explanation for this recorded run.

The long load has one 32-row prefill, all starts zero, total 1200 prompt tokens, then **5999 decodes with all 32 slots active** (`R/server/server.log:6185` and following events). Every long-load step advances one `model_fixed` replay and one `sampling_64` replay. The full P4 ledger is 255 plain-comparison + 255 eval-comparison + 5999 long-load = 6509 device decodes. There is no later prefill in the long load.

`tt/generator.py:623-647` selects B32 by active-slot extent and sampling width64 for temperature1/top-k40. `tt/serving.py:386-397` enqueues the model trace once, sampler trace once and an output snapshot on CQ0. It does **not** issue 32 whole-model forwards. The row serialization occurs inside the captured layer graph. Eval parameters remain thinking-on, temperature1, top-p0.95, top-k40, seed42.

Both trace executions use `blocking=False`. `tt/serving_state.py:506-525` snapshots output on the same queue, submits `.cpu(blocking=False)` and records a completion event. The external plugin waits on that event in `../vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:1653-1655`; the adapter's host-processing hook skips a duplicate wait (`tt/generator_vllm.py:302-309`). Thus a `decode_submitted` event proves enqueue, not completion. Tokens still cross the device/host boundary (`serving_state.py:518-545`); device-only sampling does not mean zero readback.

Across 5999 long decodes, `decode` to `decode_submitted` averages 17.1639 ms (median24.7189, maximum58.3315). Submitted-to-next-decode averages 414.9974 ms. Audit page comparison happens before the `decode` event. These boundaries do not isolate device execution from event waits, scheduler work or all audit costs. They support model/queued-execution investigation, not an additive component timing budget.

The evidence excludes eval-profile **host sampling** on this run, reduced batch occupancy, repeated prefill/admission, preemption and an adapter loop over whole-model forwards. Server cadence makes client-only pacing a poor explanation for the twofold gap. Stable first-to-last bucket rate (2.36 to2.28) and the short prompts also make long-prefill pauses an unsuitable explanation for the sustained first-bucket result. Residual scheduler/audit cost is not quantitatively ruled out.

## Finding 1 — combine's work assignment degenerates at B32

### Caller, dimensions and selected native path

`tt/multichip_decoder.py:1675-1789` performs rank-local dispatch, grouped expert GEMMs, combine and postreduce on all 40 layers. At B32:

- Hidden input is replicated BF16 TILE `[1,32,2048]`; native top-k chooses 8 distinct IDs per live row (`:1594-1606`).
- The fused integer route plan counts only active, rank-owned IDs and computes `offset[e] = prefix(ceil(count[e]/32)*32)` (`tt/moe_route_mask.py:85-105`). Its inactive duplicate skip IDs are excluded, so they do not inflate counts.
- Each rank owns 64 experts (`multichip_decoder.py:1056-1068`); therefore `0 <= count[e] <= 32`, at most64 nonempty experts, and at most2048 occupied dispatch rows. The conservative allocated capacity is2240 (`dispatch_capacity`); no capacity clamp is reached for this valid B32 case.
- Combine receives tiled BF16 `[1,1,2240,2048]`, counts/offsets for256 global expert slots, `dispatch_group_size=1`, `experts_per_chip=256`, top-k8, `seq_len_per_chip=32`, `cluster_axis=0`, `init_zeros=True` (`:1755-1766`). The 256-slot descriptor does not make 256 experts resident.
- On mesh1x4, axis0 is singleton. `N/combine/combine.cpp:37-58` lowers this to local-only, zero fabric links. The factory still creates one sender and an untilizer group (`combine_program_factory.cpp:213-222`), taking the first worker row (`:275-284`) and capping the group at four (`:326-340`). Selected hardware has enough cores for that cap. Read-batch size is tile height32 (`:417-421`).

### The exact imbalance

All three kernels independently use this assignment **anew for each expert**:

```cpp
actual_batches = ceil(expert_tokens / 32);
for (batch_idx = untilizer_global_pos;
     batch_idx < actual_batches;
     batch_idx += total_untilizers) { ... }
```

References: `N/combine/device/kernels/dataflow/reader_untilize.cpp:186-192`, `kernels/compute/untilize_combine.cpp:120-125`, and `kernels/dataflow/writer_untilize.cpp:247-257`.

Since every nonempty B32 expert has `actual_batches=1`, **only global position0 processes every nonempty expert**. Other untilizers still do initialization/control/sentinel work, but no expert payload work. The comments promise distribution across all untilizers; restarting the round-robin at each expert defeats it for decode's small expert batches. This is a concrete work-partition problem, not a speculative bank-bandwidth diagnosis.

Per nonempty expert the reader loads64 BF16 tiles =131072 bytes, compute untilizes a full32-row batch, and writer emits only `count[e] * 4096` valid payload bytes (`reader_untilize.cpp:238-254`, `writer_untilize.cpp:269-285`). With64 nonempty experts, 8MiB of tiled source reads per rank/layer passes through this one untilizer. Actual expert coverage is not logged; 8MiB is an instantiated legal upper case, not measured traffic.

Static ownership simulation, with four untilizers:

| Count fixture | Existing batches/core | Prefix-balanced batches/core |
|---|---|---|
| Two local experts with count1 (sparse single-user contrast) | `[2,0,0,0]` | `[1,1,0,0]` |
| B32, each of256 global experts selected once; each rank has64 count1 experts | `[64,0,0,0]` | `[16,16,16,16]` |
| B32, 64 local experts each count4 | `[64,0,0,0]` | `[16,16,16,16]` |
| Larger prefill contrast: one expert count128 | `[1,1,1,1]` | `[1,1,1,1]` |

The B32 fixtures are legal distinct-top-k routing. This predicts more lost parallelism as expert union grows, whereas the previous B1 mailbox experiment had very few active experts. That experiment changed grouped GEMM mailbox handling, **not this combine assignment**; its negligible B1 gain does not adjudicate this repair.

### Smallest discriminating repair

Use the existing cumulative tile-row offset to rotate batch ownership, consistently in reader, compute and writer:

```text
prefix_batches = start_page_tiled / tiles_per_batch
first_batch = (global_pos + G - (prefix_batches % G)) % G
for batch_idx = first_batch; batch_idx < actual_batches; batch_idx += G
```

`read_batch_size == tile_height == 32`, so that tile-row prefix is the cumulative batch ordinal. Unlike `expert_id % G`, it balances arbitrary sparse expert sets. Initial prefix calculation, empty experts, capacity clamp and prefix advancement must remain identical across all three kernels. A shared helper prevents later drift. Limit the first experiment to local-only32-token decode with the current tiled BF16 path, retaining the old path elsewhere if needed to contain validation scope. Encode any static selection in program identity.

This changes ownership only. Metadata still targets unique `(token,topk_slot)` pages, so payload write order does not change arithmetic. Per assigned batch, metadata c9 pushes32 pages; reader c0 produces64 tiles in its existing blocks; compute consumes them and c2 pushes32 rows; writer waits/pops32 c9/c2 pages after writing only `batch_count` rows and completing its write barrier. Empty assignments push/pop none. All untilizers, including idle ones, retain zero initialization, credit drain, counter reset and job-done sentinel (`writer_untilize.cpp:332-358`). Sender polling already waits for every untilizer's sentinel (`reader_combine.cpp:302-334,368-380`). Do not alter ACKs, output-zeroing or semaphore protocol to obtain a speedup.

Require exact all-chip meaningful output/metadata behavior for count0/1/31/32, holes in expert IDs, all routes local/off-rank, one/two/many nonempty experts, adversarial expert indices congruent modulo4, changed-input cache hits and changed/unchanged traced replay. Include count33/64 and capacity boundaries as fallbacks/ownership regressions. Compare BF16 output bits through integer views, including signed zeros, and require finite inactive cells in the initialized **combine output**. Unused dispatch/expert-region padding and unused metadata tails are undefined (`moe_fused_swiglu_common.hpp:120-146`); do not require those to be finite or equal. Require producer/consumer completion; a host assertion before synchronization is insufficient.

The data stage has at most a four-way parallelism opportunity; initialization, metadata scans and control remain. Even a fourfold improvement of combine alone is insufficient to halve the whole step unless its old share exceeds roughly67%. No current artifact measures that share. Treat this as the smallest well-grounded discriminator, not a promised floor fix.

## Finding 2 — linear attention is a captured sequence of 32 singleton graphs

`tt/multichip_decoder.py:1943-1994` batches the full-attention input projection but loops over every linear-attention row at `:1973-1986`, executing its norm and entire attention path before moving to the next. All32 P4 rows are live, so the solo-slot0 shortcut cannot help. Defaults `batch_attention_reduce=True`, `batch_full_projections=True`, `pack_linear=True` and the full-model helper installation are present (`:883-886,1258-1268`).

| Per full40 batch step | B2, ordinary | B4 | B32 |
|---|---:|---:|---:|
| Linear singleton input norms | 60 | 120 | 960 |
| Ordered linear input + output scans | 120 | 240 | 1920 |
| Convolution / QK-L2 graph / decay / recurrence / norm-tail / SiLU, each | 60 | 120 | 960 |
| Full-attention input + output projections | 20 | 20 | 20 |
| Paged SDPA calls | 20 | 40 | 320 |
| Attention + MoE reductions | 80 | 80 | 80 |

These are per-operation/helper invocation counts, not equal-cost kernels. B2 solo0 differs from the ordinary B2 column. Full-attention row work is at `:1949-1971,2063-2157`. Two reductions/layer are already batched (`:1992,2004`); direct all-reduce supports B<=32 (`:1444-1482`). Replacing a hypothetical 32-fold collective is therefore not a current repair.

### First bounded graph repair: batch the existing ordered input norm

`tt/mesh_compensated_norm.py:213-242` preserves the exact single-row reduction and refined inverse. It executes three multiplies, two native reductions, one terminal scan and one epilogue: seven computational dispatches/row, excluding reshape/layout movement. The two reductions follow from `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.cpp:37-56`, which lowers dims[0,1] as dim1 then dim0. B32 linear input norms thus contribute6720 such dispatches across30 layers.

The existing multirow norm specifies the corresponding row-independent arrangement (`tt/fused_decoder.py:1032-1049`):

```text
square [1,B,2048]
→ reshape [1,B,64,32]
→ permute (0,2,1,3)
→ reshape [4,16,B,32]
→ reduce dim1, then dim0
→ groups [B,1,4,8]
```

The permute is essential; directly reshaping the squared rows to `[4,16,B,32]` mixes rows. `NormTerminalScan` already accepts independent B rows and partitions them over cores (`fused_decoder.py:3206-3247`), as does `CompensatedNormEpilogue` (`mesh_compensated_norm.py:149-173`). The current selector restricts the latter optimized norm to singleton rows (`multichip_decoder.py:1498-1505`).

Lowered reduction check: the native factory uses **padded** spatial dimensions (`fast_reduce_nc_program_factory.cpp:48-65,107-121`). Both B1 and B32 arrangements have `Ht=Wt=1`, with outer reduction counts16 then4 and input granularities8 then4. The selected FP32-destination branch sums corresponding elements of successive whole tiles, initializing from the first operand (`device/kernels/reduce_nc.cpp`); it does not reduce across the B row lanes. This supports the row-independence invariant. Packing/permutation, broadcasts and final helper outputs still need exact-bit validation.

Small intervention: concatenate the linear residual rows once, reproduce the above grouping, reuse those two terminal helpers and the separate FP32 normalization/BF16 gamma multiplies, then split normalized rows for the **unchanged** attention bodies. Target computational dispatch count is6720→210, before packing/splitting costs. First compare intermediate squares, reduction groups, inverse and final physical bits against32 independent `normalized_singleton` calls on distinct real/edge rows. Preserve original memory configuration, reduction order, sqrt/reciprocal refinements, BF16 rounding and transport canonicalization. Ordinary batched RMSNorm is not a proven substitute. Extending the scalar epilogue to the already-batched post-attention norm is a separate, smaller test, not evidence for the input-norm change.

### Projection work is also multiplied; batching it is a larger repair

Decode uses `attention_combined` and ordered output projection (`multichip_decoder.py:442-457,613-621,714-736`), lowered to `CachedCoefficientScan` (`:1265-1268`). Per row/layer:

- Input weight `[4,1,512,3136]`: 392 runs ×16 height tiles =6272 BF16 tile reads.
- Output weight `[2,1,512,2048]`: 128×16 =2048 reads.
- Total8320 tiles =17,039,360 bytes; B32×30 layers =**16,357,785,600 physical weight-read bytes/rank/step**, with7,987,200 multiply/cumsum tile pairs.

The reader caches activation coefficient tiles, not weights (`tt/mesh_cached_scan.py:43-79`). Each weight read has its existing barrier; compute keeps ordered cumsum carry across height tiles (`:99-115`). These are source-derived implementation traffic/work counts, not useful-floor bytes or measured DRAM bandwidth. Their ideal512GB/s payload time is31.9488ms; that is neither a measured contribution nor an upper bound on possible efficiency gains.

A multirow scan could reuse a weight tile across rows while preserving independent per-row ordered arithmetic; removing31/32 repeated payload would remove15,846,604,800 bytes. A naive interleave is unsafe because cumsum carry must belong to the correct row. A conventional GEMM, changed projection K split, precision relaxation or replacing grouped EP experts with indexed B1 arithmetic does not discharge this contract. This is a larger fallback after the small norm/combine discriminators, not a minimal established fix.

### Native Q/K is already enabled; recurrence pairing remains a hypothesis

No supported missing-switch bug was found. `tt/generator_vllm.py:49-61` enables conditional state commit and native Q/K for capacity>=2; masked row decode reaches `multichip_decoder.py:524-548` and passes the flag. Shapes Q/K `[1,1,4,128]` FP32, V `[1,1,8,128]` BF16, beta/decay `[1,1,8]` FP32, state `[1,8,128,128]` FP32 and persistent activity match `StateSlabRecurrence.supports_native` (`tt/mesh_state_slab_recurrence.py:348-392`). The unmasked fallback omits the flag, but it is not the reported masked B32 route.

The state-slab helper uses32 workers (`mesh_state_slab_recurrence.py:315-332`, `mesh_direct_recurrent_sum.py:122-128`). Pairing two independent rows on disjoint32-worker groups could change960 recurrence dispatches into480 on a grid with>=64 available workers, preserving each row's existing reader/compute/writer and state/activity addresses. This requires reorganizing row phases and validating core/CB capacity, program identity, state isolation and trace bindings. It can at best halve the recurrence contribution; there is no evidence that contribution is sufficient. Keep it below normalization and combine in intervention priority.

## Other potential issues / secondary opportunities

- **Postreduce remains one core at B32.** `N/post_combine_reduce/device/post_combine_reduce_program_factory.cpp:76-92` partitions by32-token chunks:32 tokens→one chunk→one core. It scans eight expert slots/token and preserves ordered BF16 accumulation. The current model already selects packed weights and `valid_token_rows=32` (`multichip_decoder.py:1770-1788`), so repeating the previous B2 padding-prune or score-layout repair removes no B32 rows. Parallelizing independent token/width work needs a deliberate output-tilization/ownership design; merely increasing the core count breaks the32-token-chunk contract. No measured attribution makes this a headline sufficient repair.
- **Page-only reloads normalize full context-width tables.** In the5999 long steps there are3561 page-only reloads,59980 table uploads,5995 newly allocated row pages; no-upload calls have median0.3776ms while uploading calls average about28.45–29.16ms. `_validated_pages` rebuilds10×32×8192 INT32 entries on reload (`tt/serving.py:154-179,273-277`); `_bind_pages` compares full rows and copies only changed rows (`tt/generator.py:311-328`). Each upload is32KiB:1,965,424,640 logical host payload bytes across the load, before replication. Resident unchanged calls bypass model-side normalization. A page-delta or canonical-table cache preserving suffix-prefetch values and all10 layer mappings is a candidate, not a context shrink. Existing timing boundaries cannot justify subtracting all17ms average submission time, much less claiming a214ms cure. No host/device position refresh may be inferred from stale async metadata.
- **Duplicate full-attention residual concat.** `multichip_decoder.py:1946` builds the residual and `:1993` builds it again without intervening mutation. Reusing it removes30 concat API calls across10 full layers at B32 (`_bounded_concat`, `:42-52`, uses three calls for32 inputs/fan-in16). Small transport repair; no floor-sized attribution.
- **Embedding is gathered separately32 times.** `tt/model.py:450-452` calls `embed` per slot, and `embed` calls all-gather (`:309-325`). Packing token indices once could use one gather and then extract rows, with exact BF16/padding/slot proof. The high-level gather canonicalizes dim-1; BF16 TILE local width512 is tile-aligned and interleaved, selecting the ordinary minimal path, not a hidden padded-gather composite (`ttnn/cpp/ttnn/operations/experimental/ccl/composite_common.cpp:286-305`, `ccl/all_gather_async/all_gather_async.cpp:163-187`). This once-per-step opportunity is lower priority than the repeated layer work. No incorrect operand/axis mapping was established.
- **Existing performance contrasts have limited scope.** Native31.5064ms vs queued adapter31.4450ms in `D/native_qk_zero_fill_metrics.json` is one active user, physical solo0/B2, greedy sampling. It neither measures B32 nor rules out all B32 host overhead. Prior mailbox/ND failures should not be repeated unchanged; neither tests combine ownership or32-active graph packing.

## Discriminating validation and exact existing commands

These are **follow-up commands, not commands run by this investigator**. Serialize hardware ownership, follow the existing owner cleanup workflow, use fresh outputs and preserve existing evidence. No profiler is required. Run only affected gates after a changed implementation or a genuinely discriminating new control; do not repeat unchanged P4 solely to reconfirm the known failure.

1. Implement one bounded candidate at a time. First add/extend its direct component control. For combine, compare the old and prefix assignments on the count/routing matrix above, exact outputs, cache hits and changed/unchanged traces, measuring synchronized elapsed time over repeated changed-input replays. For normalization, compare all32 distinct rows and intermediates against the current singleton path. Native timing must include completion, not just trace enqueue.
2. Existing local-combine test is a useful transport/cache smoke, **not the required Qwen B32 microgate**: its constants are S128/E64/K6/H2560. Extend/add companion B32/K8/H2048/E256-with64-owned fixtures before claiming coverage. Existing selected-family model commands below likewise do not yet cover32 live attention states.

```bash
# From the checkout, with exclusive hardware ownership already established.
set -euo pipefail
source python_env/bin/activate
source ../lane-env.sh
export VLLM_TT_PLUGIN_ROOT="$REPO/../vllm-tt-plugin"
unset TT_METAL_DEVICE_PROFILER TT_METAL_ENABLE_PROFILER TT_METAL_WATCHER
timeout 1800 python -m pytest -q \
  models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_dispatch_combine_local.py \
  -k local-1x4

export MODEL_DIR=models/autoports/qwen_qwen3_6_35b_a3b
export MOD=models.autoports.qwen_qwen3_6_35b_a3b.tests
export NEW_RUN="$MODEL_DIR/readiness_vllm/p4_repair_candidate_01"
test ! -e "$NEW_RUN"
mkdir -p "$NEW_RUN"

TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
timeout 7200 python -m "$MOD.run_vllm_bucket_probe" \
  --four-row-bucket --route-mask --solo-attention --route-plan \
  --conditional-state-commit --hidden-row-mask --packed-postreduce-weights \
  --native-qk-reader --prune-postreduce-padding \
  --output "$NEW_RUN/buckets.json"

TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
timeout 14400 python -m "$MOD.run_vllm_logit_determinism" \
  --mode standalone --four-row-bucket --route-mask --solo-attention --route-plan \
  --conditional-state-commit --hidden-row-mask --packed-postreduce-weights \
  --native-qk-reader --prune-postreduce-padding --serial-output-tokens 96 \
  --compare-full-logits "$MODEL_DIR/readiness_vllm/optimized_full_control/logits.pt" \
  --output "$NEW_RUN/full40.json"
```

Coverage gap: these are existing **bounded regression controls**, not full-context serving measurements. Bucket-probe defaults use context1024 and reduced layers; the full40 standalone logit harness explicitly uses cache/context1024 (`run_vllm_logit_determinism.py:144-158,411`). Served context must remain262144 in the new matched timing and real P4. `run_vllm_bucket_probe.py:476-486` exercises at most six active owners; its all32 tuple is only an integer route-mask fixture. `run_vllm_logit_determinism.py:266-272` forces B32 with **one** active fixture and uses three active fixtures for reorder controls. Before adopting a multirow attention helper, extend the direct control to32 distinct active states, ragged positions, remapped slots, dormant→live transitions and two pending output leases. Compare complete all-chip state and full-vocabulary logits against a saved **pre-change** selected-source reference; do not let both reference and candidate silently use a globally changed native kernel. Existing PCC-only decoder checks are not an exactness substitute. No invented new CLI in this report purports to supply that missing coverage.

A source-matched B32 native/model-plus-sampler vs adapter control would be a discriminating timing addition. The current `D/../vllm_integration/sampling_debug/run_matched_b2_overhead.py:217-250` hardcodes slot0 prefill, `[128,-1]` positions and127 steps. Its capacity32 label does not make it such a control. Extend its workload explicitly before use; retain the exact served cache geometry, full40/context262144,32 distinct live rows, top-k40/seed42, model_fixed/sampling_64 and representative first-bucket positions. Measure native completion, serial adapter and queued adapter separately with drained outputs; validate tokens/state before timing. Do not fabricate a `--batch32` option on this existing script.

After component and model correctness, launch the same P4 server configuration into the fresh run directory (owner terminal):

```bash
# Owner terminal: repeat setup if this is a separate shell.
source python_env/bin/activate
source ../lane-env.sh
export NEW_RUN="$MODEL_DIR/readiness_vllm/p4_repair_candidate_01"
export SERVER_LOG="$NEW_RUN/server/server.log"
QWEN_VLLM_ASYNC=1 QWEN_VLLM_AUDIT=1 \
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
PYTEST_ADDOPTS='-p vllm' \
bash "$MODEL_DIR/tests/run_vllm_server.sh" "$NEW_RUN/server" 32 2048 --stages serve
```

The wrapper retains pinned HF/model/tokenizer revision, full context262144 and the selected TT adapter. `HF_MODEL`, `HF_REVISION`, `REPO` and `TT_MODEL_BRINGUP_ROOT` come from the existing lane environment. Audit1 matches the observed P4; an additional audit0 control may quantify overhead but must be labeled separately.

The packaged P4 probe overwrites canonical `doc/eval_readiness/long_answer_load.json` **and its raw subdirectory**. In the client terminal, preserve both before a candidate run, then run the exact affected gate:

```bash
# Client terminal: exports from the owner terminal do not propagate here.
set -euo pipefail
source python_env/bin/activate
source ../lane-env.sh
export NEW_RUN="$MODEL_DIR/readiness_vllm/p4_repair_candidate_01"
export SERVER_LOG="$NEW_RUN/server/server.log"
mkdir -p "$NEW_RUN/prior_eval_readiness"
cp -a "$MODEL_DIR/doc/eval_readiness/long_answer_load.json" \
  "$MODEL_DIR/doc/eval_readiness/long_answer_load" "$NEW_RUN/prior_eval_readiness/"

p4_exit_code=0
python "$MODEL_DIR/doc/optimized_vllm/run_probe.py" \
  --name p4_long_answer --server-log "$SERVER_LOG" \
  --server-url http://localhost:8000 --timeout 14400 \
  --output-dir "$NEW_RUN/records" -- \
  "$REPO/python_env/bin/python" -m readiness_check.run_long_answer_load \
    --server-url http://localhost:8000 --model-dir "$MODEL_DIR" \
    --server-log "$SERVER_LOG" --hf-model Qwen/Qwen3.6-35B-A3B \
    --revision 345d467b117d1fd5eda9b879a5988de9ae5f47a7 \
    --profile lm_eval_reasoning --users 32 --max-tokens 6000 \
    --functional-floor-tps 4.764924390924285 \
    --eval-plan "$TT_MODEL_BRINGUP_ROOT/runtime/readiness_check/eval_plan_defaults.json" \
    --trace-capture-regex 'QWEN_TRACE_EVENT.*"event"\s*:\s*"capture"' \
  || p4_exit_code=$?
printf '%s\n' "$p4_exit_code" > "$NEW_RUN/p4_exit_code.txt"
cp -a "$MODEL_DIR/doc/eval_readiness/long_answer_load.json" \
  "$MODEL_DIR/doc/eval_readiness/long_answer_load" "$NEW_RUN/"

stage_check_exit_code=0
python "$TT_MODEL_BRINGUP_ROOT/scripts/check_eval_readiness.py" \
  --model-dir "$MODEL_DIR" --stage optimized-vllm \
  || stage_check_exit_code=$?
printf '%s\n' "$stage_check_exit_code" > "$NEW_RUN/stage_check_exit_code.txt"
```

The captured nonzero exit codes are failures to inspect, not passes; saving them allows evidence collection and the stage check to continue under `set -e`. Retain default256-token plain/eval comparison,6000 outputs/user,1000-token buckets and ignore-EOS behavior. Check all32 prompt lengths<=74, all32 occupancy, actual model_fixed/sampling_64 use, no recapture, canonical device sampling, errors/preemptions/idle resource release, and the unchanged floor. A rate above4.764924 alone only clears this relaxed necessary check; validate the actual-work floor under the governing convention before claiming acceptance. Run the owned idle-server cleanup from `D/stop_idle_server.py` with its verified runner PID before the next device owner.

## Revision consistency, limitations and claim recheck

The saved full40 reference `M/readiness_vllm/optimized_full_qk_zero_fill/standalone.json` lists73 source hashes. A fresh rehash finds72 matches and one mismatch: `tt/model.py`, expected `83236afcf4462616ff6a34bf1fcb47d74d6b885cbefda63b0b1c441ac8286206`, current `3112ccd1d5430a2b3c4067f47d77de5c245ba84806256b75408630653ce00b75`. The current diff changes only the RW1 fused-mask selection from `len(slots)>1` to `len(slots) in (2,4,32)`, plus comments. B32 follows the same branch. The byte-model file still matches `c458af3035c7ff76172d38517639eff27c7edb3d36139191a525bb7521f251a5`. Combine files are outside that73-file manifest; their revision consistency is supported separately by an empty `git diff 8fbe38e7ce079a19153b1baad80f270f3d0c3025 -- ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine`. Do not restate historical `acceptance_source_match.json` as a fresh all73 match or claim new runtime evidence for the working tree.

P4 records the historical commit and diff stat rather than executable hashes for every runtime/plugin file. External plugin/runtime line references describe the installed current analogous source; their exact historical hashes were not recorded. The model/native structural findings use revision-matching files, and the only model delta leaves B32 selection unchanged.

The probe saves request summaries, not complete raw SSE/token timestamp arrays, and no raw plain/eval request records. Therefore the exact client median cannot be recomputed independently from retained timestamps. `served_probe.py:221-230` can spread content-chunk arrival times over the usage token count; the artifacts lack enough mapping to audit every such token boundary. Independent server cadence corroborates the gross2.36 result but does not restore those missing timestamps.

After drafting, each headline was checked against reachability and the lowered contracts:

| Candidate/story | Adjudication | Smallest falsifier / remaining uncertainty |
|---|---|---|
| Combine resets assignment per expert; B32 experts fit one batch | Proven source work imbalance on the selected path; counted offsets fit capacity; all three consumer loops agree | Synchronized old/candidate B32 combine timing could show negligible contribution; actual expert union is unknown. |
|32 singleton linear paths | Proven current graph structure and physical-work multiplier, not a measured212ms attribution | Exact batch-norm control plus native B32 timing; packing overhead or changed reduction bits could reject it. |
| Native Q/K flag omitted in serving B32 | Refuted by masked conditional branch and supported shapes | A runtime guard mismatch would require new contrary evidence; unmasked source omission alone is not it. |
| Host sampling / repeated prefill /32 model forwards in adapter | Refuted for this saved P4 by route/replay/prefill ledgers | Do not generalize beyond this run; token read/event wait remains. |
| Postreduce, page-table, concat or embedding alone solves floor | No sufficient attribution; secondary candidates | Measure a controlled change; retain exact state, output ownership and numerical contracts. |
| Previous B1 AutoFix discharges P4 | False scope match | Only new batch32 repair evidence or applicable contract disposition can close RW3. |

No mathematical correctness defect, wrong-device expert mapping, hang or output corruption was established. No change is marked implemented, hardware-validated, sufficient, waived or accepted. RW3 remains open pending a correct, source-matched batch32 repair and its affected validation.
