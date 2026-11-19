# AutoDebug: Qwen3.6 single-user serving misses the inherited functional floor

Inspection: 2026-11-17. Repository HEAD: `5e2dca2e09485be6c4f18c688a1b7b532e45a99b`.
`M/` below means `models/autoports/qwen_qwen3_6_35b_a3b/`; native paths are repository-relative.

## Findings

**The current single-user result fails the inherited gate: 53.796970693 ms versus a maximum of 21.63129875 ms/token.** The comparable useful-work denominator remains **1,107,522,496 bytes/rank**. B2's inactive row, MoE padding, extra routes, intermediate traffic and repeated reads do not enlarge it. This is 4.0209% of theoretical useful bandwidth versus the required 10%, requiring a 59.7909% latency reduction. Native/adapter agreement does not waive the gate.

**The leading source-backed explanation is the model's multirow execution graph, especially its padded grouped-EP MoE, rather than a large vLLM adapter cost.** Solo serving selects a physical B2 graph. Every one of 40 layers pads its MoE input to 32 rows *before routing*. The 30 synthetic zero rows receive real top-eight routes and enter dispatch/counts; the dormant physical row is also not excluded from routing. Each nonempty expert then executes at least one 32-token tile on an 8×8 grid. This is concrete wasted work, not a measured allocation of the 53.8 ms among components. B2 also computes attention for two physical rows and takes different normalization/router/expert paths from historical B1. Inspection cannot prove which component dominates the current critical path or promise that one small change reaches the gate.

**The B4-only candidate cannot improve the primary solo benchmark by construction.** It addresses extents 3/4 that currently select B32; solo slot 0 still selects B2. It is a useful mixed-concurrency experiment, but cannot be the single-user floor repair.

**The smallest credible model-owned next intervention is coherent device-side suppression of inactive MoE routes while keeping the existing grouped arithmetic and completion barrier.** A valid off-rank expert ID can encode a skipped inactive route using the existing `table[id] == -1` contract. Use the same modified IDs consistently for counts, dispatch and post-combine, with persistent device activity and fixed 32-row carriers. This requires exactness and timing experiments; it is not a verified fix. If insufficient, specialize inactive attention and local routing transport without changing the active row's numerical family. Ordinary B1 substitution is not such a specialization.

The unconditional native reader ACK drain is required and must remain. Neither chronology nor the existing probes establishes that it accounts for the historical/current latency difference.

## Scope, revision and evidence discipline

Inspection only: read source, existing JSON/Markdown and cheap CPU bookkeeping; no TTNN/model import, device execution, profiling, server requests, reset, process inspection/termination or implementation edits. Three isolated xhigh source reviewers examined floor accounting, native MoE, and serving exactness. Only this report was written. The previous root report was already preserved verbatim at `M/doc/optimized_vllm/AUTODEBUG_previous_stage.md`; equality was checked before replacement.

The prepared tree changed externally during this investigation. Initial source and measured baseline used adapter buckets2/32; during draft review the owner changed `generator_vllm.py` to2/4/32. Its initial hash below is therefore a snapshot, not its final hash. Model, generator and decoder hashes remain the inspected values. Evidence from earlier stages is identified explicitly rather than treated as a current execution. In particular, `revision=345d467...` in model artifacts is the **HF checkpoint revision**, not the TT source revision (`M/tests/run_full_model_performance.py:322,341`).

Inspected SHA-256 values:

| Source | SHA-256 |
| --- | --- |
| `M/tt/multichip_decoder.py` | `5b5696d6d2028f3ac5ccc5f0349f35916915d43130010ef9f835f7921da873fe` |
| `M/tt/model.py` | `79d18c123ac508d0e75cf09777c0f3c13cc262175b6b038bf8347378863dbf78` |
| `M/tt/generator.py` | `034fd1a2d2e823c5f1454ef6299309e9808aa2d0ca635f2c5b7bbe87602351dd` |
| `M/tt/generator_vllm.py`, initial2/32 | `05c4488a8c672047e50e4634d028afb5a1b726be63c1eb6dde0c4a475f6c0af4` |
| `M/tt/generator_vllm.py`, later2/4/32 | `37e1a5d0f1ee93901382c2019e3d0d25f1a4a140943564ea52324e25dd04fc0b` |
| `M/tt/precision.py` | `d892fbd51e696e605101d5a19d48a53f197d16a3dcb4b7963a0d2f8b9d23ac18` |
| native `moe_fused_swiglu_reader.cpp` | `24123b33d520b495a43a4f8b23286387e00bfaecb1bf8fac6b6adb11ed7359fb` |

During review, `M/readiness_vllm/optimized_b4_probe/result.json` completed PASS on real layers0/3, context1024, capacity32, tracking1/skip0:21 greedy/seeded/native64 cases, five retained traces,8601 prepared program entries,204 model/sampling replays, zero workload releases and five shutdown releases. Its snapshots synchronize, so this is neither async-overlap performance nor a full40/server result. `optimized_b4_logits/standalone.json` was still RUNNING when inspected; no pass is inferred. The reduced artifact has a stale limitation saying only top-k8 was tested despite the recorded seeded64 cases. None of this changes the soloB2 selection or establishes the floor.

### Direct observations

| Evidence | Observation | Limit |
| --- | --- | --- |
| `M/readiness_vllm/optimized_before/vllm_benchmark.json:38–79` | Fresh greedy 128-input/128-output/concurrency1: TPOT 53.796970693 ms, TTFT 3035.459136 ms, decode 18.588407 t/s/u | One primary request; no per-op attribution. Overall output throughput 12.971173 t/s includes prefill. |
| `M/readiness_vllm/serving_overhead/comparison.json:2–15` | Repaired native serial 53.806492 ms; adapter serial 53.937979 ms; async2 queued 53.750487 ms. All 18 trajectories exact | Canonical32 historical source; sequential small samples, different timing boundaries. Async2 is throughput, not serial latency. |
| `M/readiness_vllm/optimized_before/matched_b2_http.json:2,2202–2231` | Completed fresh matched-fixture HTTP PASS, TPOT P50 53.747926 ms, P99 53.748012 ms | Two measured requests after warmup; matches saved native fixture, not a same-run current native decomposition. |
| `M/doc/optimized_full_model/perf_summary.json:59–191` | Historical B1 queued token-out 21.416868 ms; observed delivery 21.461293 ms; floor 21.631299 ms | Different source/graph/context/fixture; predates ACK repair. |
| `M/readiness_vllm/logit_determinism/standalone.json` | Historical all40 PASS: three nonaligned fixtures 129/130/131, solo B2/forced B32 and forward/reverse three-row cases, exact full logits at prefill and one decode | Context1024; no sampling replays. Current model/decoder match, but later generator/adapter/serving hashes differ. Not a fresh final64/B4 acceptance. |

Interpretation: the matched native result and HTTP result strongly disfavor a roughly 33 ms wrapper explanation. They do not prove zero residual overhead, negative HTTP overhead, or a numerical split between attention, MoE, ACK completion and other model work. The native after-control is canonical32; current final64-capable source has later generator/serving changes. The topology audit already records this caveat (`M/doc/optimized_vllm/topology_audit.md:24–31`). Old documents that say HTTP comparison is still pending are stale chronology; completed HTTP evidence exists.

## The useful-work gate remains 21.63129875 ms

`M/tests/multichip_functional.py:3–8,12–49` defines a logical useful-byte model, excluding padding, redundant reads, intermediates and collectives. `M/tests/run_full_model_performance.py:737–760` applies it over **127 positions 128…254**, adds terminal work and enforces 10% of 512 GB/s/rank. The byte function matches the historical recorded source hash.

| Useful term, per rank per emitted token | Bytes |
| --- | ---: |
| One linear layer, weights | 25,462,944 |
| One linear layer, recurrent/convolution traffic | 1,073,152 |
| One linear layer total | 26,536,096 |
| One full-attention layer, weights | 17,436,800 |
| One full-attention layer, mean KV traffic | 197,632 |
| One full-attention layer total | 17,634,432 |
| `30 × linear + 10 × full` | 972,427,200 |
| Vocabulary-parallel LM head `2048 × 62080 × 17/16` | 135,086,080 |
| Local embedding row `512 × 2` | 1,024 |
| Terminal norm `2048 × 4` | 8,192 |
| **Total** | **1,107,522,496** |

`ideal_ms = 1,107,522,496 / 512,000,000 = 2.163129875`.
`maximum_ms = ideal_ms / 0.10 = 21.63129875` (at least 46.229309 t/s/u).
At 53.796970693 ms: useful bandwidth is 20.587079 GB/s/rank; efficiency is 4.020914%; latency is **2.486997×** the allowed ceiling. The excess is 32.165672 ms/token.

The equivalent aggregate calculation divides four ranks' useful bytes by four ranks' bandwidth. Do not sum rank latencies or replace useful bytes with the busiest EP rank's padded physical traffic. The bound includes required state/cache traffic; “weight bandwidth” is shorthand. Context262144 is capacity, not 262144 live KV tokens read at position128. Allocated pages, inactive rows, 2240 workspace rows and canonical sampling width are not additional user work.

There is a small real representation qualification, which does not rescue performance: historical linear B1 uses the ordered BF16 router scanner; current B2 defaults to the ordinary FP32 router. Chain: `M/tt/multichip_decoder.py:1751–1754,1863–1867`, `M/tt/fused_decoder.py:688–709`, `M/tt/functional_decoder.py:80`; the historical formula prices BF16 at `M/tests/multichip_functional.py:25–29`. A *different physical-path stored-weight estimate* would add `30×2048×256×(4−2)=31,457,280` bytes/rank, giving 22.24569875 ms at10% for that difference alone. This is not permission to change the inherited common-useful-work gate. It illustrates why physical implementation traffic and the comparison numerator must be reported separately. Neither denominator comes remotely close to accepting 53.8 ms.

The historical 128-position convention 128…255 gives 21.63139875 ms, only 0.0001 ms higher; it is not the recorded 127-interval gate (`M/doc/optimized_full_model/topology_audit.md:61`). Stage10 itself explicitly retains the 21.63129875 ms gate (`M/doc/optimized_vllm/work_log.md:43`).

## Concrete execution and work ledger

### Dispatch from serving to native kernels

1. The measured baseline adapter capacity32 advertises buckets `{2,32}`; the later candidate adds4 (`M/tt/generator_vllm.py:25–49`, original cache construction `:115–121`). Generator disallows B1 as a multirow-capacity bucket (`M/tt/generator.py:61–70`). Selection uses **highest active physical slot plus one**, not live request count (`:578–584`). Thus slot0 alone uses B2; a lone slot17 owner still uses B32.
2. `QwenModel.decode` embeds and prepares positions for every physical row in that graph, masks hidden rows, executes each layer's `decode_batch`, then masks again (`M/tt/model.py:369–411`). Activity protects state and residuals; it does not erase the captured attention graph or remove routes.
3. For B>=2, linear attention runs independently for every physical row, followed by batched reduction. Full attention batches normalization/projections but invokes paged attention separately per row (`M/tt/multichip_decoder.py:1807–1867`). Inactive position/state controls remain necessary. A physical inactive intermediate entering MoE must not be assumed zero solely because the previous layer boundary was masked.
4. `_moe` pads to a multiple of32 **before** router linear/top8/softmax; no activity argument reaches its routing (`:1541–1566`). B2 therefore inserts30 exact synthetic zero rows; B4 inserts28; B32 inserts none. Zero activation is not a skipped route: top8 still emits8 in-range expert IDs. Actual tied IDs and their rank placement were not saved in the reviewed performance evidence; do not assume IDs0…7 or rank0 concentration.
5. B>=2 calls `_grouped_moe(..., stochastic_rounding=False)` through `batched=True`. Scatter builds a selected-expert mask; sum produces counts; local expert mask drops off-rank experts; counts align to32 and a matrix prefix produces offsets (`:1612–1640`). Physical inactive/synthetic rows are currently counted when their IDs are local.
6. Each rank owns64 **whole** experts, IDs `r*64 … (r+1)*64`, rather than a quarter of each expert. Weight upload/dispatch table are `:1018–1044`. Dispatch group size1 routes only locally owned experts from replicated input; dispatch/combine are chip-local despite the4-chip mesh because axis0 has size1 and their wrappers select zero fabric links (`dispatch/dispatch.cpp:41–62`, `combine/combine.cpp:37–58`). The final model reduction crosses the mesh. Native call uses `experts_per_chip=256` for global lookup while the fused operation correctly derives local64 from its weight lists (`:1645–1675`; native wrapper `moe_fused_swiglu.cpp:34–41`). This is not a 256-versus64 contract bug.
7. `capacity = ((8*32 +31*64 +31)//32)*32 = 2240` rows/rank for B2/B4/B32 (`M/tt/multichip_decoder.py:829–833`). Fused input is logically `[2240,2048]` but `input_m_tiles=1`, `core_grid=(8,8)`, offsets direct-write, BF16 output preallocated (`:1658–1675,1699–1709`). **Capacity is not a loop count of2240 live tokens.**
8. Combine initializes zeros and reconstructs `[32,8,2048]` BF16 contributions, then post-combine weights/reduces in top-k order and the model reduces EP partials across ranks (`:1677–1697,1863–1867`). The full contribution carrier is `32×8×2048×2 = 1,048,576` logical bytes/rank, including inactive rows. This excludes extra scratch/padding and is not measured DRAM traffic.

### Fused expert work: passing and current configurations

The public wrapper honors the supplied compute/output/grid; it does not canonicalize B2 into the indexed path (`moe_fused_swiglu.cpp:54–88,96–119`). Factory setup derives grid8×8, embedding2048, hidden512 and actual input/output/weight formats, then passes them to reader/compute/writer (`device/moe_fused_swiglu_program_factory.cpp:372–413,574–609`). Selected weights are BFP8, compute LoFi/BF16 DEST, output BF16. Reader and writer use their dedicated default NoCs.

Geometry derives 64 embedding tiles split over8 K groups (`KR_PAD=8`),16 hidden tiles split over8 H groups (`HN_PAD=2`, two GU chunks), and one output embedding tile/core. `M_BLOCK=8` tile rows is a maximum block; `M_EFF_MIN=1` is the small-work minimum (`device/moe_fused_swiglu_geometry.hpp:15–24`; `geometry.cpp:94–124,154–179`).

| State | Reader count and loop derivation | Consequence |
| --- | --- | --- |
| Historical B1 token | Does not call this grouped kernel; uses indexed sparse TP matmuls | Cannot subtract its time to measure grouped reader/barrier cost. |
| Current B2, any local expert count1…32 | `m_t=ceil(count/32)=1`; `m_blocks=ceil(1/8)=1`; `m_eff=1` | Executes32-row tile arithmetic even for one useful token. |
| B4 or B32, same count range | Same values as B2 | Smaller physical bucket alone does not shrink grouped MoE below a32-row tile. |
| Local expert count0 | `m_t=m_blocks=0` | Publishes/consumes mailbox, then skips weight/compute/collective block traffic. Required epilogue still executes. |

Reader validates counts/offset regions and publishes matching scalar work to compute/writer (`reader.cpp:564–627`); the block loop skips count0 (`:671–677`). Compute consumes one mailbox for each local expert (`compute.cpp:524–551`). Therefore the kernel scans64 descriptors per rank, **not64 full expert FFNs regardless of activity**. Count0 is already optimized below the descriptor loop. The 32-row arithmetic minimum is not256 rows from `M_BLOCK=8`.

One full expert's three BFP8 matrices have `3×2048×512×17/16 = 3,342,336` logical weight bytes. The useful selected8 aggregate is26,738,688 bytes/layer, equivalent to6,684,672 mean bytes/rank, independent of TP or EP ownership. Indexed B1 also executes one32-row M tile; this is not32× arithmetic relative to B1. With eight live experts alone, both families have the same aggregate24,576 weight tiles/tile products, before their different partial reductions and transport. EP's rank distribution, serial nonempty-expert sequence and tile arithmetic differ from TP's balanced quarter-expert execution. Extra zero-row routes can add nonempty experts and increase a rank's critical workload; they cannot be counted as more useful experts.

No saved route-to-rank mapping proves how many extra experts or how much rank skew occurs in the primary run. If all inactive rows, including the dormant physical row, share one eight-expert set, the live/inactive union has at most16 distinct experts and can shrink to8. If the dormant physical row routes differently from the30 synthetic rows, that union can reach24 under the same synthetic-set assumption. These are conditional examples, **not a universal2× critical-path bound**: concentration on one rank can make the busiest-rank improvement larger than the aggregate ratio. Conversely route overlap can make weight savings small. Both stories require actual mapped IDs/counts; neither is a timing result.

## Historical exactness and completion evidence constrain the repair

`M/doc/full_model/AUTODEBUG_MIXED.md:71–89,154–171` documents two separate historical boundaries. With grouped stochastic rounding, equal recorded target inputs could produce different grouped outputs when other requests changed counts. A later SR-off-labelled run restored exact solo/mixed logits for slots1/2 but exposed a B1-versus-grouped slot0 mismatch; its generator changed too and kernel hashes were absent. Preserve those provenance limits. Current code explicitly disables grouped SR for batched decode (`M/tt/multichip_decoder.py:1671–1675`; native `compute.cpp:451–459`). Do not re-enable SR or accept PCC/argmax agreement as a replacement for exact full-logit invariance.

B1 and B2 differ beyond MoE: singleton delegation, compensated norm/shape-specific full norm, batched full projections, ordered router flag/selector, indexed TP versus grouped EP, shared projection configuration and reductions (`M/tt/multichip_decoder.py:1383–1403,1473–1500,1541–1612,1751–1754,1807–1873`). Selected policy itself specifies FP32 DEST for indexed experts but BF16 DEST for grouped EP (`M/tt/precision.py:45–80`). Merely declaring bucket1, or adding grouped padding after ordinary `decode_forward`, does not preserve B2/B32 logits. Running indexed arithmetic consistently across every batch would define a different baseline and need full model/precision/capacity validation; it is not an equality-preserving adapter tweak.

The native reader now ends with unconditional `noc.async_write_barrier()` (`device/kernels/moe_fused_swiglu_reader.cpp:1365–1367`). `M/doc/vllm_integration/NOC_COMPLETION_AUDIT.md:7–33,37–48` explains why sent-only metadata/H flushes and writer NoC1 completion cannot drain reader NoC0 nonposted ACKs. Dedicated-mode next-program initialization can rebase counters while old ACKs remain. Every normal reader exit reaches the repair, including all-empty experts.

The completed synthetic controls each pass5000 replays: up to245 ACKs remain at the undrained final observation versus0 after drain; neither reproduces next-kernel overshoot. Both actual native baseline/drained fixtures pass1000 empty plus1000 nonempty replays. Later repaired full40 quality control completes12 outputs/3072 tokens and preserves the two completed prerepair trajectories (`M/doc/vllm_integration/work_log.md:170–180`). This supports retaining the repair. It does not isolate the exact original hang counter pair or measure a32ms barrier penalty.

Historical Stage7 B1 does not execute the grouped reader during decode, so “it predates the ACK drain” is chronologically true but is not causal evidence for the performance regression. Do not remove/conditionalize the drain, change ACK equality, reset expected counters, skip empty-path completion, or move the wait to the writer.

## Minimal model-owned route and ranked discriminators

These are proposed follow-ups for the owning agent, **not requests for a device run in this investigation**, and none is represented as passing evidence. Keep B4 validation as its independently scoped mixed-concurrency experiment; its reduced PASS above is not a floor result.

1. **First: remove inactive routing coherently, retaining grouped arithmetic.** Thread the persistent per-slot device activity into the decode MoE path. Construct a fixed32-row activity carrier: physical rows read their runtime activity; synthetic pad rows are always false. Before scatter/counts, create rank-local IDs that keep every active ID unchanged and replace each inactive route with a *valid in-range nonlocal ID*: rank0 can use64; ranks1/2/3 can use0. The existing rank-local dispatch/reduce tables mark those IDs `-1` (`M/tt/multichip_decoder.py:1031–1044`). Use these same substituted IDs in selected/counts/local-mask, dispatch and post-combine; keep active scores unchanged. Local masking makes the inactive fake expert count0; dispatch skips it before reading/updating offsets.

   This exploits a proved lower-level contract: `dispatch/.../reader_worker_dispatch.cpp:358–387` reads `table[id]`, then skips `-1` before offsets/route emission. `post_combine_reduce/.../deepseek_moe_post_combine_reduce_writer.cpp:154–165` and `..._compute.cpp:95–108` implement the paired nonlocal skip and zero initialization. Metadata follows the surviving route plan. An ID65535/256 sentinel is unsafe because table lookup precedes a range check; masking counts alone is unsafe because original payload routes would still advance offsets. Skip IDs must be rank-local, not replicated from rank0. This candidate needs no shared native change and leaves the required barrier intact.

   Retain32-row MoE/router/shared/terminal carriers, existing64 expert weights, selected precision, SR off, grid8×8, output buffers, B2/B32 ownership and canonical sampler. Prepare any new mask/ID programs, constants and buffers before captures in a fresh generator lifetime. Read activity on device during replay; a Python capture-time activity branch would freeze the wrong workload. With all32 rows active, this route substitution must be identity.

   **Discriminator:** equal-input, all-chip expert-output and complete active-vocabulary logits against the unchanged grouped reference, before autoregressive feedback. Compare expert outputs by logical `(slot, expert ID, top-k position)` through each variant's offsets/metadata; raw packed-buffer element correspondence is not valid after compaction. Include solo slots0/1/2/3/17/31, inactive holes, expert-sharing live rows and active rows preceded by inactive ones. Removal changes compacted expert lane/region offsets; deterministic SR eliminates the known stochastic coupling but is not a proof of bitwise lane invariance. Restore/reference state and compare recurrent/conv/live logical KV too. Reject any active logit difference. Then a clean, no-profiler full40 matched primary timing decides whether it reaches21.63129875ms. This discriminates excess inactive-route work: a timing miss rejects this intervention as sufficient, while leaving intrinsic active-expert tiling, fixed routing/combine and other costs unresolved. The whole-step reduction required is59.7909%: a2× improvement of the affected component cannot suffice alone even if it were all current latency; a5× component improvement requires at least74.7386% of current latency in that component. No current component timing or mapped route distribution establishes that precondition. Amdahl limits prevent claiming this minimal candidate will necessarily meet the floor.

2. **Second: if needed, eliminate inactive independent attention while preserving numerical carriers.** A bounded solo-slot0 signature can omit the other row's independent linear-attention/SDPA work, materializing its partial without altering active attention, then use the existing B2 normalization/projection/postnorm/router/shared/reduction/terminal family. Keep masked state semantics and the grouped route fix. Preserve batched full-projection shapes; do not enter ordinary B1. A universal per-slot optimization needs runtime-safe device predicates or explicit stable slot indirection; changing Python `if active` inside capture is insufficient. A new retained signature must fit the unchanged trace reservation alongside required model/sampling signatures; no capacity is assumed. This separates duplicate attention cost from grouped work. It offers no source-only guarantee of meeting the floor.

3. **Third: specialize the model-owned local grouped routing transport if residual routing cost remains.** With replicated activations, local-only table and dispatch group1, a bounded decode-only device program can build coherent local counts/offsets/packed rows and metadata for at most32×8 routes, using preallocated storage. Start by retaining native fused expert compute and its barrier plus existing combine/post-reduce; subsequently replace transport only if exactness and timing justify it. A custom combine must preserve current BF16 boundaries, top-k accumulation order, per-rank partials and cross-rank reduction. The existing row-indexed post-reduce variant is not a direct replacement: its table input requires row-major BF16 while fused expert output is tiled (`post_combine_reduce/device/post_combine_reduce_device_operation.cpp:99–100`; `moe_fused_swiglu/device/moe_fused_swiglu_device_operation.cpp:341,422`). Selective untilization/gather or equivalent model-owned consumption would be needed; blindly untilizing all2240 rows moves8.75MiB/rank. This is more work than the off-rank mask, so it ranks later. Zeroing and skipped-output initialization cannot be deleted without supplying equivalent correctness.

4. **Only after those contrasts: tune/fuse expert scheduling without changing its numerical contract.** Native64-descriptor traversal, serial nonempty experts, whole-expert EP skew and 32-row minimum remain potential costs. Different grids/K partitions, indexed TP, changed accumulator precision or activation packing can change active results and require a new numerical proof; they are not safe flags. A model-owned small-work kernel must reproduce grouped per-row reduction/rounding and drain its own outstanding traffic. There is no evidence yet that such a kernel is necessary, or that a pure Python configuration can meet the floor.

The earliest useful experiment is item1, not another adapter rewrite, more HTTP confirmations, a drain-removal timing A/B, or another unchanged B2/B32 accuracy run. B4 tests a different question: whether extents3/4 can avoid B32 attention while retaining exactness and lifetime.

### Acceptance conditions shared by every candidate

- Exact active full-vocabulary logits against the current grouped reference across B2/B4/B32 and activity changes; exact dormant conv/recurrent state and correctly mapped live K/V. Use teacher-forced equal inputs before testing sampled trajectories. Historical top-logprob HTTP comparisons do not replace full logits.
- Keep checkpoint-selected `baseline_bfp8`, LoFi expert families, BF16 KV/CCL/logits and existing ordered linear state. No silent precision reselection or tolerance relaxation.
- Canonical split device sampling remains: native32/native64 draws, persistent tokens/seeds, active-row merge and dormant-wide width selection (`M/tt/generator.py:494–502,555–600`). No force-argmax, host logits or host sampling in the measured path. A dormant top-k>32 still requires native64.
- Prepare before capture; retain trace IDs/buffers, `blocking=False` model/sampler replays, stable program entries, zero workload recaptures/releases, immutable two-output leases and valid output-event waits (`M/tt/generator.py:504–600,724–731`; `M/tt/serving_state.py`). Do not “optimize” by deleting completion/snapshot ownership.
- Keep context262144, aggregate paged-cache budget, capacity32/all physical owners, page32, nonaligned prompt/continuation support and checkpoint defaults. Cover129/130/131 plus boundary tails and required page growth; no truncation/alignment rejection to reduce work.
- After a reduced exactness discriminator, prove all40 exactness and actual plugin route/bucket selection, nonblocking resident stale-input behavior, changed/unchanged page tables and two outstanding outputs. Then compare clean matched128/128/concurrency1 and the fixed mixed workload. The floor remains a separate pass/fail field even if benchmark execution succeeds.

## Hard contradictions and lower-ranked possibilities

| Claim or intervention | Adjudication |
| --- | --- |
| AddB4 and primary solo will become fast | False: primary still selectsB2; `_moe` still pads to32. No causal path from this declaration to a smaller solo graph. |
| Count32 padded rows /2240 workspace rows /all64 experts as useful work | False: contradicts the defined numerator and actual count0 skip. |
| NativeB2≈HTTP means optimized or floor passed | False: localizes away a large wrapper gap, while the unchanged gate still fails by2.487×. |
| Stage7B1 proves current grouped decoder can do21.4ms, or ACK barrier costs32ms | False: different model graph, numerical paths, context and revisions; B1 decode bypasses this native reader. |
| Restore B1 only for solo while mixed keeps current grouped math | Contradicts the known cross-batch exactness problem; not fixed by matching greedy output on a few prompts. |
| Disable grouped SR /skip empty experts | Already present for batched decode /already present below the descriptor loop. These are not fresh fixes. |
| Zero routing weights or counts alone to suppress inactive rows | Insufficient or unsafe: counts/dispatch/offsets/metadata must describe the same surviving routes; zero weights do not prevent expert work. |
| Remove reader drain because writer/dispatch completed | False ownership/completion assumption; readerNoC0 ACKs require their own completion guarantee. |
| All remaining time is sampler, host logging, max-context KV reads or generic TopK | Unproven. Sampling widths/physical rows and context capacity do not imply those costs; matched evidence bounds wrapper effects, not individual device operations. |

Other potential costs include repeated row masks/layout copies, router setup, generic local dispatch/combine/zeroing, replicated shared work, descriptor traversal, per-row attention and EP rank imbalance. These are source-supported work opportunities without current wall-time attribution. LM-head geometry and persistent sampler transport have historical tuning evidence, but no present observation promotes them above the direct grouped/inactive-work contrast. Prefill transport and singleton tails affect TTFT and mixed traffic; they do not explain the127-interval solo decode metric by themselves.

## Report review

The draft was checked again against Python selection, C++ wrapper/factory, reader/compute mailbox loops and dispatch/post-combine consumers, with fresh adversarial native/accounting reviews. External B4 source/artifact changes were separately recorded without splicing them into the2/32 baseline. Claims were narrowed where necessary:30 synthetic B2 zeros rather than assuming31; no assumed tied expert IDs/rank; no all64 full-expert execution; no2240 live-token loop; one32-row tile rather than `M_BLOCK`256; no measured barrier penalty; canonical32 native evidence distinguished from current final64; completed HTTP evidence distinguished from stale pending labels; numerical equality and speedup of route suppression remain unverified. The proposed intervention preserves the completion contract and makes count/route agreement explicit. The last review also required logical expert-output correspondence through changed offsets, and narrowed a failed timing contrast to rejection of sufficiency rather than rejection of all grouped costs. No experiment was run to turn those remaining uncertainties into a performance or correctness claim.
