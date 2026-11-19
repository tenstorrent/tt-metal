# AutoDebug: stage-5 TP4 B1 decoder performance

Inspection date: 2026-11-13. Scope: `models/autoports/qwen_qwen3_6_35b_a3b` only; single-decoder B1 on the true Blackhole 1×4 mesh. No implementation changes, target execution, hardware access, resets, or AutoDebug relaunch were performed. Only this report was written. Below, `M/` denotes that model directory; other source paths are repository-relative.

## Findings

**Both actual mesh FUNCTIONAL runs exist and fail the 10% floor. Linear requires substantial additional kernel work, not another packing flag. Full attention has a plausible smaller repair sequence, but no inspected change is proven to close its floor.** The current defaults already select packed linear input projections, separate full-attention projection, and the batch optimizations. B1 bypasses the B32 attention path.

There is also a **concrete theoretical-byte discrepancy**: linear B1 reads BF16 ordered-router weights, but `tests/multichip_functional.py` charges FP32 router weights. Correcting it makes the linear target stricter. It does not explain away either failure.

The strongest repair priorities are:

1. Full attention: normalize the replicated incoming B1 tensor using a suitable sharded layout; then specialize the FP32 256-to-8 selector and its score/index output tail.
2. Linear: fuse the two compensated norm scalar tails while retaining their arithmetic; reduce repeated SiLU-buffer initialization and then address projection coefficient movement or recurrence with new component evidence.
3. Obtain a current packed-path B1 profile before assigning current percentage shares or accepting a combined speedup. The preserved profile is excellent historical operator evidence, but not a current 128/128 latency trace.

**No single small repair identified here is enough to promise linear acceptance.** Reusing the earlier fused-AGMM intervention, enabling already-selected flags, changing `sorted=False`, or retuning CCL alone does not account for the required reduction.

## Direct observations and measurement contract

The artifacts are `M/doc/optimized_multichip_decoder/functional_l{0,3}.json` and their provenance sidecars. Both match the runner, byte-model, fused-decoder and mesh-decoder source inspected at the start of this investigation. That measured mesh hash is `44554422bd0838de1954d6b9d145a14ab21e8c46acbdd88867a6b49f27f0a60d`; mesh line references below describe that version.

During the final review, a concurrent edit added three setup lines releasing superseded single-chip scan weights after constructing the packed mesh weight (`install_linear_tp4`); the mesh hash became `f84bc5fbe494fda51ffb31f06e2c2858b5d3a50601a2a7d015fe5b6cce96328a`. This shifts later line references by three and does not change the inspected forward arithmetic. The saved timings are evidence for the measured hash, not a new performance/correctness result for that edit. Fresh acceptance must match the final source. Other files in this active checkout can change independently as well.

| Actual 128-in/128-out B1 run | Linear, layer 0 | Full attention, layer 3 |
|---|---:|---:|
| Mean blocking replay, ms/token | 1.007534997 | 0.397953289 |
| Recorded useful-byte floor, ms | 0.053876813 | 0.034443250 |
| Recorded floor / latency | 5.3474% | 8.6551% |
| Recorded 10% latency ceiling, ms | 0.538768125 | 0.344432500 |
| Required reduction, ms | 0.468766872 | 0.053520789 |
| Required latency reduction | 46.526% | 13.449% |
| Required speedup | 1.8701× | 1.1554× |
| Prefill PCC | 0.999991977 | 0.999969297 |
| Minimum decode PCC, 128 steps | 0.999158705 | 0.999777332 |
| Minimum saved state PCC | 1.0 | 0.999998648 |
| Top-level / FUNCTIONAL passed | false / false | false / false |

All three repeats consistently fail. Their per-repeat means are approximately 1.009175/1.005609/1.007821 ms for linear and 0.397934/0.398543/0.397383 ms for full. Saved eager/trace equality, repeated bit equality and output-replica equality pass. These are saved observations, not reproductions performed here.

`M/tests/run_multichip_decoder.py:240-341` supplies the important completion boundaries:

- It saves post-prefill state, produces an independent evolving eager stream, restores state before capture, and captures one decode invocation.
- The FUNCTIONAL loop restores state once per repeated sequence, refreshes the input and position for each step 128–255, and synchronizes before timing.
- Each timed replay uses `execute_trace(..., blocking=True)`. Readback and equality checking against that step's eager result happen afterward. State continues through all 128 steps.
- Three sequences yield 384 samples, whose arithmetic mean determines the floor. Ordinary `decode_traced_warmed_ms` instead reports the median of restored step-zero replays (`:293-301`); that median cannot replace the FUNCTIONAL mean.
- `:518-522` folds the FUNCTIONAL failure into overall `passed` before asserting. `tests/check_optimized_multichip_evidence.py:118-188` requires the exact true-1×4 contract, current runner/model identity, changing positions, repeated equality and ≥10%. There is no acceptance loophole from a passing ordinary PCC result.

One small measurement discrepancy deserves correction in a subsequent measurement: the timer surrounds entry/exit of `device_only()` (`runner:313-316`), whose implementation constructs a Torch dispatch mode and two `mock.patch` contexts (`tests/runtime_audit.py:18-24`). Thus “blocking trace replay only” includes audit-context setup/teardown and blocking host dispatch. Move the unchanged audit context outside the clock if replay-only latency is the intended contract, and rerun an apples-to-apples control. Do not subtract guessed overhead from these saved samples or present such a correction as a kernel optimization. Refresh, presynchronization, readback and sequence reset are already outside timing.

## The byte model: one real mismatch, no mesh factor-four credit

`M/tests/multichip_functional.py:12-52` uses useful per-rank bytes / 512 GB/s. Its current totals are 27,584,928 B for linear and mean 17,634,944 B for full. Sharded weights are divided by four; each full-attention rank owns one of the two KV heads, with duplication across rank pairs. The replicated router is charged on every rank. The B1 expert path still computes all eight selected experts on every rank's tensor-parallel weight slices.

Most accounting is intentionally appropriate: it excludes the padded A/B channels, shared-scalar padding, rereads, intermediate buffers, and CCL. Charging all four ranks' bytes against four times the bandwidth leaves exactly the same floor. Single-chip efficiency, aggregate mesh work, and per-user B32 latency provide no extra factor-four credit.

### Proven linear-router dtype mismatch

The exact selected call chain is:

1. `tt/multichip_decoder.py:1512-1513`: B1 calls `decode_forward`, not the batched path.
2. `:1438-1455`: `_forward` passes `ordered_router=True` for public decode.
3. `:1269-1272`: `_moe` requests FP32 **logits**, with `terminal_scan=True`.
4. The router key is absent from mesh projection weights, so the MRO reaches `tt/fused_decoder.py:688-709` and reads `scan_router_weight`.
5. `fused_decoder.py:342-345,564`: that weight is uploaded with the BF16 default. `DirectCoefficientStreamingScan:2796` explicitly requires BF16 weight and activation.

The model's `router_fp32=2048*256*4` is therefore wrong for linear B1. The actual unique weight bytes are `2048*256*2`: **1,048,576 fewer bytes/rank**. Full attention uses its native FP32 router; its denominator does not receive this correction. An FP32 policy label or FP32 output cannot establish input-weight storage precision.

There is a second, tiny stale term: `output_norm_bf16=128*2`. The selected TP4 weighted tail (`multichip_decoder.py:547,749`) consumes the FP32 gamma constructed in `fused_decoder.py:2386-2398`, resident in L1. Counting unique useful weights across tiers gives 512 B for gamma; a strict useful-DRAM model excludes its resident gamma entirely. Do not count its expanded 32-row copies as additional useful weight traffic.

| Linear accounting after correction | Unique useful bytes/rank | Floor, ms | 10% ceiling, ms |
|---|---:|---:|---:|
| Unique weights across tiers, FP32 gamma | 26,536,608 | 0.0518293125 | 0.518293125 |
| Useful DRAM only, exclude resident gamma | 26,536,096 | 0.0518283125 | 0.518283125 |

Either honest convention gives approximately 5.1441% and requires approximately 1.944× speedup, or 48.559% lower latency. Resolve the model's stated memory-tier convention explicitly, retain the original failed artifacts, and collect new evidence after any model correction. This is not a means to raise the numerator and pass.

### What this floor leaves out

A useful-byte floor is not a full compute roofline. Linear input/output projections alone contain `2048*3088 + 1024*2048 = 8,421,376` useful coefficients per rank; the selected implementation performs FPU products followed by ordered FP32 scans. Recurrent state updates, norm refinements and selection add distinct arithmetic and dependencies. This can explain poor bandwidth-normalized utilization without implying a bandwidth saturation defect.

A separate compute model could use independently substantiated FPU/SFPU throughputs and the useful arithmetic in the model. No such throughput-derived computation is established by these artifacts. Padding, rereads, register/CB traffic, measured operator durations and the serialization of this particular implementation must not be added to the theoretical numerator. The required bandwidth-based gate remains failed under both the reported model and the stricter corrected model.

## Exact operators and source boundaries

### Linear input/output scans: packing has already spent its easy gain

Selected entry: `M/tt/multichip_decoder.py:404-419,647-669`. Input packs state width 1600 and output-only width 1536 into width 3136. Of these, 3088 channels are useful; 48 A/B channels are padding. Decode partitions K into 512-element chunks. The output projection uses local K1024/N2048, also split into K512 chunks.

Lowering does not select an ordinary matmul factory:

- Python helper: `M/tt/fused_decoder.py:2782-2879`, `DirectCoefficientStreamingScan`.
- Data mover: inline `DIRECT_COEFFICIENT_READER`, `:2779`.
- Compute: inline `STREAMING_SCAN_COMPUTE`, `:2558`.
- Writer: inline `STREAMING_SCAN_WRITER`, `:2560`.
- Hardware math: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h:15-97`.

Each run reads BF16 weight tiles and BF16 activation tiles, converts the activation's valid row into a pair-swapped coefficient column, forms FPU products, then calls Default cumsum in increasing tile order. The LLK scans columns in eight face passes and carries running sums in LREG4–7. The scan resets at each K512 block; `fast_reduce_nc` then combines block outputs. Immutable weight rows were pair-swapped at load time (`mesh:686,702,719`). All of those boundaries matter numerically.

The historical 110-core scheduling explains the modest packing gain:

| Input scan schedule | Runs | Maximum runs/core | Ktiles/run | Critical sequential Ktile iterations |
|---|---:|---:|---:|---:|
| Old state width1600 | 4×50=200 | 2 | 16 | 32 |
| Old output-only width1536 | 4×48=192 | 2 | 16 | 32 |
| Current packed width3136 | 4×98=392 | 4 | 16 | 64 |

Old sequential calls total 64 critical tile iterations; the packed call still has 64. It saves launches and edges, not this arithmetic sequence. Saved short-run medians improve approximately 1.015803→0.999910 ms, consistent with a modest improvement, not the further 0.469 ms required by the recorded gate.

The reader also rereads a full 2 KiB padded activation tile for every weight tile. Packed input has 6,272 tiles: 12.25 MiB of weight reads and another 12.25 MiB of source-issued padded activation reads, for just 4 KiB of unique activations. Caching/reusing coefficients, or reading two aligned 32-byte half rows, is a concrete movement hypothesis. It must preserve pair order, first-tile carry behavior and Blackhole address alignment. More short transactions or a coefficient-construction adapter can outweigh saved bytes; this is not proof of DRAM saturation or a predicted speedup. Those rereads do not belong in the useful-byte denominator.

### Linear norms: the strongest narrow fusion target

`mesh:1215-1225` calls the inherited norm twice on replicated linear input. The reachable implementation is `M/tt/fused_decoder.py:1014-1104`:

- FP32 squares; the existing two-stage `fast_reduce_nc` and terminal scan preserve the selected reduction organization.
- `NormTerminalScan` (`:3194-3295`) does the two ordered scans and exact packed-bit terminal selection.
- `:1050-1101` constructs shifted variance, a corrected sqrt and reciprocal using BF16 high/low splits and explicit FP32 products/subtractions.
- Only then do the wide activation and gamma multiplies produce BF16 output.

The scalar tail dispatches 24 small operations per norm. In the old rank-0 profile their combined kernel work is **109.1825 µs** (52.698+56.485); entire norms total **177.015 µs**. These depend on little data and are natural fusion candidates.

The smallest intervention is one scalar epilogue after the existing terminal scan, preserving every split, subtraction association, sqrt/reciprocal primitive and FP32 rounding boundary. Keep final wide/gamma multiplies initially. An alternative is to append that epilogue to `NormTerminalScan`. Do not replace the whole expression with an approximate rsqrt, contract it into different FMAs, or substitute native RMSNorm on the evidence of one short passing run: sensitive norm/projection substitutions have prior long-stream/layer-34 rejections (`M/doc/fused_decoder/README.md:20,44`). Fusion is promising, but its cost must remain below the removed chain; 109 µs is historical work available, not guaranteed wall savings.

### Linear recurrent and output tail

The current recurrent path is `mesh:363-402`: FP32 decay, explicit SFPU products and state updates, plus two `_recurrent_sum` reductions. The reduction reshapes to `(8,16,8,128)`, reduces dims `[0,1]`, and extracts the valid eight heads (`:355-361`). Public decode rounds state on consumption; the live unrounded state participates in the current output before the next boundary. Replacing it with a generic fused recurrence can narrow FP32 through FPU/TF32 inputs or change reduction grouping.

The historic recurrence adapter experiments are negative evidence, not reusable wins: `M/doc/fused_decoder/endurance/v6_recurrent_fusion_rejection_summary.md` records bit-exact six-stage adapters at 5.30×/5.59× native traced latency. A new implementation needs a different, justified work distribution and a complete input→state/output→next-consumer timing. The old B1 recurrent section contains approximately 106.808 µs kernel work; even deleting it cannot supply the linear deficit.

The local SiLU reader has a small concrete inefficiency. `fused_decoder.py:2883` clears two 1024-word temporary buffers on each of four tiles, although only the valid row is subsequently overwritten. TP4 allocates two pages per buffer (`mesh:155-163`). Initializing both pages once per invocation reduces initialization from 8192 to 4096 uint32 stores/core without changing values or lookup arithmetic. Initialize both pages before reuse/publication; preserve low-six-bit DRAM alignment and the compact LUT's NaN/signed-zero behavior. The CB loop must still push/pop four input pairs and four output tiles.

A larger alternative distributes the 32 `(head,tile)` tasks over 32 workers instead of eight workers doing four tiles each. It needs a compatible width-sharded output and charges the downstream projection conversion. Historical SiLU is approximately 38.500 µs; the ideal four-to-one body scaling offers at most about 28.9 µs before fixed and layout costs. The once-per-page initialization is the smaller first experiment. Neither gives a floor-closing gain by itself.

### Full-attention B1 input norm and selector

The incoming test tensor is DRAM-interleaved. `mesh:1217-1225` uses native sharded RMSNorm only when input is already sharded, so the initial full-attention norm selects the one-core interleaved path. The later residual is sharded and uses the faster branch. Historical rank-0 means are **24.713 µs initial norm** versus **6.114 µs post-attention norm**. The initial norm is followed by an approximately 1.51 µs interleaved-to-sharded conversion for projection anyway.

A small scoped candidate converts the incoming full-attention B1 tensor to the existing projection input memory config before normalizing. `M/tt/optimized_decoder.py:549` builds a 16-core width shard; using it also allows the projection conversion at `mesh:1146` to become a no-op. Alternatively use the 32-core residual shard and explicitly charge its reshard. The measured norm difference supplies an approximately 18.6 µs historical comparison; 16-core norm cost and the net conversion cost remain unmeasured. A roughly 15–19 µs net target is plausible, not a result. Recheck norm numerics, exact paired KV equality and all route decisions. This is an isolated-decoder boundary opportunity; do not assume every composed layer receives interleaved input.

Both B1 kinds then call `ttnn.topk` on **FP32 `[1,1,1,256]` logits**, requesting eight (`mesh:1269-1280`). Exact lowering:

- `ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:46-47`: K rounds to the minimum supported tile quantity, 32.
- `.../topk/device/topk_device_operation.cpp:106-151`: small width/height selects `TopKSingleCoreProgramFactory`; `device/topk_constants.hpp:21` sets the low-height multicore width threshold to 1024, above this width256.
- `.../topk/device/topk_single_core_program_factory.cpp:393-410`: one-tile output, full FP32 value unpack/destination. The compute arguments include largest/stable/rank-stamped behavior, not the Python `sorted` option.
- `.../topk/device/kernels/compute/topk.cpp:276,446`: Wt8 executes seven insertion rounds, each sorting/merging 64 candidates and retaining 32, despite requiring eight useful outputs.

Historical TopK is **43.473 µs** for full and **43.512 µs** for linear. It is the largest individual full B1 operator. `sorted=False` does not bypass that selected compute sequence. The inherited `InlineSelector` is also not a drop-in repair: `fused_decoder.py:1925` compares `a>>16` BF16 keys, and its reader consumes uint16 values. Likewise the existing BF16 generalized gate would change this path's FP32 comparisons.

The useful candidate is a **full-FP32 256-to-8 specialization**, optionally fusing stable softmax, BF16 score rounding, and direct UINT16 ROW_MAJOR index output. Preserve all 256 comparisons, value/ID pairing, cutoff/tie behavior, NaN behavior and eight selected experts; do not use fixed IDs, BF16-truncated comparisons or dense all-expert work. Current `stable` is not explicitly enabled: a replacement must define and validate the selected tie behavior rather than silently assume lowest-index ties.

Beyond TopK, the historical full tail spends 5.650 µs in softmax, 1.258 in score cast, 1.122 in ID cast, and 1.127 in untilize: **9.156 µs total**. A selector target of 20–35 µs savings, 3–7 µs from the output tail, and 15–19 µs from initial norm gives a **38–61 µs hypothesis envelope**. Only its upper portion exceeds the actual 53.521 µs deficit; none of those ranges are measured gains. Use these as experiment budgets. Keep a margin and measure the combined implementation before declaring full acceptance.

### Sparse experts and collectives are not an untried flag fix

The selected B1 sparse path is `mesh:1282-1322`: eight active experts, GU K2048/N256 on eight cores with Kblock64; down K128/N2048 on 64 cores with Kblock4. BF16 inputs, BFP8 weights, LoFi, FP32 accumulation and 1×1 subblocks are explicit. It is tensor-parallel indexed sparse execution, not grouped prefill dispatch.

The native path is `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:191` → `sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:111,330,578,660,688,846` → `device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp`. Its data movers are `reader_bmm_tile_layout_in0_sender_padding_metal2.cpp`, `reader_bmm_tile_layout_in0_receiver_metal2.cpp`, and `reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp`. The latter loops over the eight supplied IDs, addresses each selected expert by `group_id*KtNt`, and emits compact slot `bB` (`:360-379`). There is no remaining 256-expert scan to remove. Expert-level parallelism would require a larger kernel/work-distribution change.

Historical GU/down costs are approximately 36.108/11.576 µs. Existing local screens already measured and rejected L1 GU-input and wider-subblock layouts (`M/doc/multichip_decoder/local_geometry_audit.md`, `sparse_geometry/search.json`). Generic perf-report advice is not evidence those simple changes remain untried.

Full-attention QKVG/output use the DRAM-sharded matmul factory (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:1548,1632`) and `reader_dram_sharded_multicore.cpp` / `bmm_dram_sharded_multicore.cpp`. Initial interleaved norm lowers through `layernorm_op_multi_core.cpp:867,1026` to `layernorm.cpp`; the sharded branch uses `sharded_layernorm_factory_helpers.cpp:498-503` and `layernorm_sharded.cpp`. These are native families, unlike the linear inline scans.

Paged SDPA (`mesh:1647`) uses `k_chunk_size=256`; `sdpa_decode_program_factory.cpp:927-941` selects `reader_decode_all.cpp`, `writer_decode_all.cpp` and `sdpa_flash_decode.cpp`. Positions128–255 fit one 256-token chunk; workers with no assigned chunk exit (`sdpa_flash_decode.cpp:203-217`). The configured 64-core count is not 64 useful workers. Smaller chunk geometry is a bounded secondary experiment; the entire historical SDPA operator is only about 14.486 µs.

Historical full CCL totals 20.926 µs; linear totals 25.827 µs. Deleting all CCL is not a realizable repair and still falls far below the required reductions. Current separate gather/projection avoids the earlier duplicated-KV fused-AGMM issue. Reintroducing that candidate would require fixing its rank-relative accumulation contract and revalidating exact pair equality; it is not the smallest B1 floor repair.

## Historical profile ledger and limits on gain claims

Source for this ledger is `M/doc/multichip_decoder/tracy/layer_{0,3}/ops.csv`, restricted by `PERF_DECODE` signposts, rank0, and each of 16 replay-session IDs. Each replay has 177 linear or 79 full operations. `fused_decoder.py` matches the old profile and new FUNCTIONAL provenance; `multichip_decoder.py` differs because the packing/batch work changed. Inline source kernels are mapped using operand shapes and call order; the CSV does not name those helpers.

| Historical linear section, rank0 mean | Kernel work, µs |
|---|---:|
| Input + post-attention ordered norms | 177.015 |
| Two input scans including reduction/cast | 123.947 |
| Convolution, QK normalization, gate preparation | 94.004 |
| Recurrent update/output | 106.808 |
| Output normalization and SiLU | 77.025 |
| Output ordered scan | 57.704 |
| Router, selector, score/index preparation | 84.502 |
| Remaining routed/shared MoE work | 118.641 |
| CCL/residual remainder | 29.646 |
| Total | 869.292 |

These are sums of kernel durations, **not wall-latency bounds**. Gaps, instrumented host work and overlap differ from clean blocking replay. Never sum four chips as latency, subtract these values directly from the new mean, or turn them into current percentages. They show why multiple changes are needed: the entire historical scalar norm tails plus entire SiLU operator plus entire recurrence sum to only about 254.5 µs of kernel work, with nonzero replacement costs, versus the recorded 468.8 µs deficit (about 489.2 µs after correcting linear bytes).

Full B1 individual targets are smaller: TopK43.473, sparse GU36.108, router25.092, initial norm24.713, QKVG15.920, SDPA14.486, sparse down11.576 µs. TopK deletion alone would not supply the actual 53.521 µs target even under an optimistic direct-additive assumption.

Linear prefill is a separate large issue. The old profile spends approximately 24.428 ms in eight corrected-matmul recurrence calls and 3.572 ms in triangular recurrence, out of a 33.643 ms critical-rank kernel sum. This traces through `mesh:504` to `fused_decoder.py:2325,2053,2641,3378`; increasing-K corrected arithmetic and feedback packing dominate. Fixing prefill does not satisfy the decode-token floor. No full-model or serving work is proposed here.

## Discriminating next experiments, in priority order

These are proposed device follow-ups for the parent, not actions performed in this investigation.

| Priority | Single-variable intervention | Fact it tests | Keep / reject criterion |
|---|---|---|---|
| 0 | Preserve the two failed FUNCTIONAL runs; make router-byte accounting describe the selected weights; clarify timing/audit scope | Reporting matches actual storage and timing | Keep old evidence; regenerate provenance and bounds. No denominator inflation or estimated subtraction. |
| 1, full | Convert initial B1 full-attention input to the existing projection shard before native norm | One-core interleaved norm is avoidable at net cost | Compare complete norm→projection chain, including conversion; retain ≥0.995 and exact paired KV/output replica checks. |
| 2, both | Same FP32 logits into specialized selector, then separately integrate score/index tail | General top-32 sorting and conversions dominate 256-to-8 routing | All selected IDs/value pairs and scores remain valid, including close cutoff/ties; eight gate-selected experts; complete MoE latency improves. |
| 1, linear | Fuse only the compensated scalar norm epilogue; retain the reduction | Tiny scalar dispatch chain is avoidable without altering norm arithmetic | Check shifted variance, corrected root/inverse and BF16 normalized output against current path; then long/changing-input and state gates. |
| 2, linear | Initialize both SiLU CB pages once per invocation | Repeated zero loops cause avoidable reader work | Exact output and padding/CB behavior retained; complete tail→projection improves. |
| 3, linear | Change scan coefficient staging/read geometry only | Redundant activation reads limit the ordered scan | Products and terminal FP32 partials unchanged; K512 order and final reduction unchanged; complete conversion-inclusive time improves. |
| 4, linear | New fused recurrence work distribution, not a rejected adapter | Native product/reduction/layout dispatch is avoidable | Exact public state boundary and repeated trace behavior; complete next-consumer chain beats native. |

For every accepted component, rerun the same 128/128 FUNCTIONAL sequence with refreshed inputs/positions, state reset only between repeats, blocking completion, no instrumentation, and unchanged mean-based gate. Also retain the existing broader PCC/state/mesh-replica gates; one passing aggregate PCC is insufficient. Record all four ranks and the actual packed-path operator geometry in any new profile.

## Other potential issues and rejected explanations

- `fused_decoder.py:910-912` contains an unreachable `linear_attention` condition inside `if self.kind == "full_attention"`. This is dead duplicated code, not the selected linear path, which exists at `:1014`. It does not cause these timing failures.
- Inherited precision metadata such as `expert_memory="l1"` or single-chip `sparse_cores` does not describe every mesh override. Mesh uploads place indexed expert weights in DRAM (`mesh:889-916`) and override geometry. Use actual tensor/config metadata for diagnosis; do not infer storage from a broad policy label.
- Distributed norm, fused AGMM, B32 attention batching and grouped dispatch are not the selected B1 default branches. Earlier observations from them cannot be spliced into a current B1 causal chain.
- No source contract violation demonstrating dropped experts, missing collective contributions, invalid state ownership, or a queue deadlock was found on the current B1 path. The evidence supports performance bottlenecks and the byte-accounting bug; it does not establish a current kernel correctness fault.

## Draft review

The first draft was checked against the current B1 MRO, actual functional JSON and provenance, selected BF16 router weight construction, FP32 top-k factory, concrete K512 scan counts, LLK carry semantics, SiLU two-page buffers, timing completion boundaries, and prior rejected recurrence/layout candidates. Three independent xhigh investigators reviewed linear arithmetic, full-attention/MoE lowering, and the harness/bound. This review rejected a presumed FP32 linear-router weight, a `sorted=False` speedup, another packing-only floor fix, and reuse of fused-AGMM or exact recurrence adapters as demonstrated wins. All projected gains remain explicitly unverified on hardware. Only `AUTODEBUG.md` was changed by this investigation.
