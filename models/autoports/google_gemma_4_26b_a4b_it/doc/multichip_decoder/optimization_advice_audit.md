# Optimization advice audit

Source/CPU inspection only, 2026-09-26. No hardware execution, runtime edit,
new performance measurement or final-path acceptance is made by this audit.
The audit follows the optimize skill's per-role advice and topology review.

## Evidence boundary

The native advice comes from the historical sliding-layer `profile_v0/`
[decode CSV](profile_v0/decode_perf_report.csv),
[prefill CSV](profile_v0/prefill_perf_report.csv), and human tables. Its
[provenance](profile_v0/provenance.json) records 4096 prefill tokens and **one**
traced decode, against `runtime_v0.py.txt`. It is not a native profile of the
current hybrid/shared-policy candidate. Native row times below identify old
costs, not current row times or the final 4096/128 layer duration.

The strongest completed paired correctness evidence inspected here is
[sliding_shared_policy.json](sliding_shared_policy.json) and
[full_shared_policy.json](full_shared_policy.json), runtime `5f75aa5f...`.
Both pass 4096/128 and all-rank cache checks with EP prefill, indexed TP decode,
fused tail and matching shared decode policy. Their TP4 host medians are
95012.64/817.75 us sliding and 80566.95/812.31 us full, for prefill/decode.
The latest boundary controls use `d3dee9c5...`. These host intervals cannot
replace a current native profile or establish current per-op bottlenecks.

## Native advice mapped to experiments

| Historical role and native evidence | Advice and source interpretation | Tested response | Remaining targeted work |
| --- | --- | --- | --- |
| Sparse prefill gate/up, 128 rows of advice; IDs beginning 27373; 623978.511 us summed old row time | Runtime `nnz` is unknown in CSV. Old gate K-block 1 was a material implementation cost. | TP v1 changes gate K-block to 11; EP4 then assigns 32 complete experts per rank. Hybrid retains the faster EP prefill with active routing. | Current native EP profile; measure actual per-chunk union counts if reporting sparse utilization. Do not insert fixed `nnz=8` into prefill or label every 32-token union as eight experts. |
| Sparse prefill down, 128 rows beginning 27392; 113086.780 us summed old row time | Same runtime sparsity caveat; width-192 TP down and output-core selection affect efficiency. | TP v1 moves down from 8 to 88 workers; EP uses width 704 and down K-block 22. | Inspect updated native rows before further prefill tuning; retain separate TP/EP precision and local-count ledgers. |
| Indexed sparse decode gate/up, ID 37877, 54.053 us; M32,K2816,N384; 12 workers, K44, subblock1x1 | Try a larger output subblock. Accuracy-oriented fidelity advice is not proof LoFi is wrong. Increasing only the grid cannot create more than 12 output tiles. | Indexed top-8 TP decode remains; pure EP decode was measured slower, so hybrid uses TP. Gate weights retain sliding BFP8/full BFP4 and LoFi. | A legal 1x2 variant needs `out_block_w=per_core_N=2`, six working output blocks, and a compatible grid. Compare K44 versus K88 with L1 accounting. Fewer workers may lose; do not assume the larger subblock wins. |
| Indexed sparse decode down, ID 37882, 53.093 us; M32,K192,N2816; only 8 workers in v0 | Larger subblock and appropriate work distribution. | v1/current TP path uses 88 workers, K6, per-core N1. Pure EP's wider down path loses in complete decode. | Compare a 44-worker, per-core N2, subblock1x2 variant at K6 under the same sparse/indexed contract. Current kernel time must be measured; v0's 53 us is not its current cost. |
| QKV decode, ID 37920, 55.650 us; K2816,N2048; HiFi4, K2, subblock1x1 | Larger K blocks/subblocks and lower fidelity where correctness permits. | v2 uses packed QKV, K22 and HiFi2 sliding/LoFi full with FP32 output/accumulation; paired correctness passes. | For sliding, compare wider per-core N2/4 against current N1 under the same precision. DRAM-sharded QKV and fused AGMM are separate coherent candidates described below. |
| WO decode, ID 37976, 9.656 us; K1024,N2816; HiFi4, K8, subblock1x1 | Larger output subblock; HiFi2 is a candidate, not a guaranteed accuracy-preserving replacement. | Explicit output grid/block/output-L1 path is present and measured cumulatively with v2. FP32 partials are reduced before norm. | Precision-locked N2 or N4 subblock/grid control; separate HiFi2 control if this row remains material. DRAM-sharded row projection or adapted fused-CCL decomposition remains open. |
| Shared gate/up decode, ID 38028, 41.000 us; K2816,N1088; BF16, HiFi2, K2, subblock1x1 | Larger K/subblocks; generic suggestion to increase fidelity targets accuracy, not latency. | Matching TP1's sliding BFP8/full BFP4, LoFi decode policy fixes the stack failure. No geometry win is established by that correctness fix. | Root's shared-geometry A/B: K44/N1 versus K88/N2, followed by independent DRAM-sharded and packed-versus-split candidates if material. |
| Shared down decode, ID 38032, 24.678 us; K544,N2816; DRAM input, K1, subblock1x1 | Move input to L1; increase the legal K block and subblock. | Shared-policy path writes BF16 intermediates in L1 and uses explicit compute policy; the selected automatic program is not yet a measured geometry choice. | Root's K17/N2 explicit down program versus automatic baseline; DRAM-sharded alternative must handle the prime 17-tile K dimension. |
| Prefill WO, IDs 27247/29484/31719/33940, 896.259 us total; K-block1 | Larger K block and phase-appropriate projection program. | Current `configure_prefill_output(... minimal=True, minimal_block_w=8, fidelity=LoFi)` replaces the old path. | Verify native rows on current code. Do not credit all whole-layer prefill improvement to this one change. |
| Prefill QKV/shared/router dense rows | Generic “place input in L1”; QKV/router/shared-GU blocks are already described as reasonable by the tool. | Chunked DRAM prefill is retained; explicit minimal QKV/WO programs exist. Dominant sparse-prefill work was addressed first. | If rows remain material, compare total producer/projection cost and L1 live allocation. Moving 1024-row FP32 H2816 input wholesale into L1 costs 11534336 bytes across the grid; it needs a legal distribution and is not a generic single-core placement. |
| Router decode, ID 37714, 13.281 us; FP32 x BF16, K22, 4 workers | Subblock1x2 would require fewer output blocks/workers. The advice text says BFP8 despite the same row recording BF16 weights. | Native centered top-8 gate plus indexed expert decode is already used; layer-specific HiFi4/LoFi projection matches the baseline policy. | If material in the new profile, compare a two-worker N2 projection with identical routing tests. Retain score margins and expert-ID agreement; do not apply the row's mismatched BFP8 recommendation blindly. |

The historical v0-to-v1 sliding host medians move from 819988.08/938.33 us to
225144.43/898.61 us; v2 is 224246.27/864.36 us. These are cumulative candidate
runs with passing PCC, not isolated native-op speedup attribution. EP-only
prefill/decode is 100370.47/936.39 us sliding and 85972.09/935.38 us full.
Hybrid plus fused tail is 94791.51/817.06 us sliding and 80611.13/811.42 us full.
Sources are the corresponding `sliding_headline_v*.json`, `*_ep_v2.json` and
`*_hybrid_tail.json`. The results support phase-specific experts; they do not
establish that further geometry or topology work cannot improve the path.

## Collective, residual and normalization topology

The source has three row-parallel reduction boundaries: attention WO, routed
experts and shared MLP. With replicated residuals each uses reduce-scatter
followed by all-gather. The old decode table records three pairs at IDs
38117/38118, 38164/38165 and 38172/38173, with summed pair row durations of
50.798, 38.697 and 39.410 us. These are per-row diagnostics; their sum excludes
other operations and is not a complete-layer duration.

| Dataflow | Work already performed | Remaining candidate/evidence |
| --- | --- | --- |
| Attention partial -> RS+AG -> post norm/residual | Explicit FP32 projection and current replicated reduction pass; an earlier full-attention length65 hidden-sharded residual path also passes. | Parent's Ring/sharded-residual AGMM experiment must include next-layer-compatible distributed norms, prefill, cache and full-layer timing. The old short sharded run is not a target-workload rejection. |
| Local WO -> fused matmul+RS -> sharded norm/residual | Linear fused MMRS stalled; source-backed Ring retry succeeds. Tuned component MMRS loses to an equally tuned separate component for both kinds. | The measured component family is not a win. Other program/layout families remain unmeasured; no universal MMRS or sharded-residual rejection follows. |
| Sharded normalized input -> gather -> QKV | Ring AGMM component passes and saves 5.736 us sliding/11.788 us full against its tuned separate component. | Real-weight integration is pending. Preserve actual sliding HiFi2/full LoFi and logical S1 shapes; component used synthetic weights and physical M32. Root owns this A/B. |
| Routed and shared partials -> two reductions -> separate branch norms | Current branches reduce independently, then normalize independently. | Separately assigned combined-CCL candidate may pack both outputs into one collective, unpack and apply the same two norms. Adding branches before their norms would change model semantics. No result is available here. |
| Repeated native reductions/gathers | Async CCL and persistent semaphore ownership are already present. | Persistent output/intermediate buffers, worker/channel tuning and lower payload dtype require separate same-topology controls. Semaphore persistence alone is not evidence that tensor outputs are preallocated. |
| Norm/residual/tail conversions | Fused-tail option reproduces baseline sharded branch norms, residual-input norm and fused BF16 residual/scalar output; paired target and stack pass. | Current native rows must quantify remaining sharded/interleaved and DRAM/L1 transitions. Consider carrying compatible working shards through adjacent consumers; do not remove necessary cross-layer layout conversions from measured windows. |

Component evidence, exact topology constraints and recoveries are in
[AUTOFIX_fused_ccl.md](AUTOFIX_fused_ccl.md). Parent-owned AGMM, shared geometry
and combined-CCL experiments remain pending in this audit. No component
saving is added to the complete-layer host results.

## Dense DRAM-sharding applicability by role

DRAM sharding here means bank placement **within each ASIC**, independently of
TP/EP mesh placement. The ordinary dense DRAM-sharded program requires TILE
weights width-sharded in DRAM, width-sharded row-major L1 input/output with
matching storage-grid contract, exactly one physical M tile, and K plus
input-shard K divisible by `in0_block_w`. Source:
`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:1302-1375`.
Changing the program config on existing interleaved weights is insufficient.

The following eight-bank, one-reader starting geometries are source-consistent
candidates, not measured configurations. Read the actual DRAM bank count from
the device at setup. Logical decode M1 is padded to a 32-row tile; the decode
backend does not replace long prefill. Every trial must include required
input/output reshards, slicing, next consumer and collective costs.

| Dense role | Per-rank K,N | Bank padding and working shard candidate | Status/constraints |
| --- | --- | --- | --- |
| Shared gate/up | 2816,1088 | Pad each rank's packed N to1280; eight L1 storage cores give K352 (11 tiles), N160 (5 tiles); K-block11. | Detailed implementation plan in [shared_geometry_audit.md](shared_geometry_audit.md). Crop each rank back to1088 before its up544/gate544 split. Global trailing padding would corrupt rank ownership. Preserve BFP8 sliding/BFP4 full, LoFi policy. |
| Shared down | 544,2816 | N already divides eight banks. One activation storage core keeps K17 tiles and permits K-block17; the eight bank readers remain distinct from that storage grid. | Valid source design, untested. Eight storage cores instead require K padding544->768 and matching activation zeros, adding 41.18% weight tiles. Include conversion and CCL cost. |
| QKV sliding | 2816,2048 | No N padding at eight banks; eight storage cores give K11 tiles, N8 tiles, allowing K-block11. | Preserve FP32 normalized inputs/output, BFP8 weights, HiFi2/FP32 accumulation and exact local Q4/KV2 packing. Compare to interleaved K22 and the complete AGMM family; DRAM sharding can force a smaller K block. |
| QKV full | 2816,3072 | No N padding; eight storage cores give K11 tiles, N12 tiles, allowing K-block11. | Preserve local Q4/KV1 and duplicated tied K/V packing, full LoFi/FP32 policy, head split and cache behavior. This TP4 packed width includes both K and V. |
| WO sliding | 1024,2816 | No N padding; eight storage cores give K4 tiles, N11 tiles, allowing K4. Four storage cores give K8,N22 and allow K8. | Row-parallel weights remain mesh-sharded in K while locally bank-sharded in N. Preserve BF16 SDPA input, BFP8 weights, FP32 partial output and reduce-before-norm. |
| WO full | 2048,2816 | Eight storage cores give K8,N11; four give K16,N22, allowing K16. | Same contract; include head-concat and working-shard conversion cost. More cores are not automatically better if they shrink legal K blocks. |

For QKV, a four-storage-core variant would allow K22 and N16/24 tiles, while
bank-reader assignment remains separate; it needs live-L1 validation. The
same principle permits wider WO working shards without changing residual
mesh ownership. The factory supports storage-output resharing separately from
bank reader workers; see
`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`.

The imported Gemma4 attention and shared-MLP loaders deliberately disable their
older DRAM-sharded paths for this MoE model after historical PCC regressions
(`models/demos/gemma4/tt/attention/weights.py:161-166`,
`models/demos/gemma4/tt/shared_mlp.py:143-150`). That is evidence against blindly
turning on those old helper paths, not a blanket inability of dense
DRAM-sharded matmul to implement these roles. A new candidate must preserve
current packing and precision, use exact real local shapes, and pass
projection, full-layer, cache, stack and replay checks.

## Sparse API exclusions and advice limitations

- The active expert path uses `ttnn.sparse_matmul`, not dense `linear`.
  `SparseMatmulDeviceOperation::select_program_factory` selects its sparse
  multicast factory, which explicitly extracts
  `MatmulMultiCoreReuseMultiCast1DProgramConfig`
  (`ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:60`).
  The current sparse backend therefore has no drop-in dense DRAM-sharded
  program-config variant. The broad parameter variant in the API is not proof
  every dense factory is supported. Generic dense/batched DRAM-sharded advice
  cannot justify replacing runtime-selected experts with dense all-expert work.
- Expert batching/indexing is essential. Indexed TP decode obtains eight
  device-resident expert IDs, passes those IDs to both sparse projections, and
  gathers matching weights for the mix. EP prefill uses dynamic local unions;
  partitions can be empty. A DRAM-sharded expert replacement needs a new
  routed weight-access implementation or proven compatible sparse backend,
  and is outside a Python program-config-only change.
- Sparse `nnz` is an exact kernel loop-count contract, not a performance hint.
  Mismatches can deadlock; indexed mode instead derives loop count from IDs
  and forbids a simultaneous `nnz`
  (`sparse_matmul_device_operation.cpp:223-268`). Instrument route unions for
  utilization accounting rather than guessing eight active experts per
  prefill chunk. The tool's `--active-experts` modeling flag does not authorize
  changing runtime `nnz`.
- Sparse subblock changes remain legal candidates within the supported 1D
  family, but output subblock must divide output block and per-core N. The
  validator explains the former invalid combination's stalled CB producer
  (`sparse_matmul_device_operation.cpp:353-390`). Merely setting subblock W2
  while leaving block/core N1 is invalid.
- The old CSV's router advice misidentifies the recorded BF16 weight as BFP8.
  Its fidelity text is generic; use lowered dtype/compute policy and measured
  routing correctness. Likewise, promoting already-passing LoFi expert/shared
  arithmetic is not a performance optimization solely because the advice
  suggests more accurate BF16 multiplication.
- “Input in L1” is phase- and footprint-dependent. Large prefill tensors may
  need DRAM interleaving; test the full producer/consumer placement and L1
  budget. Decode placement is a stronger candidate because logical M1 pads
  to one tile. Avoid converting a complete shared input separately for every
  consumer when one compatible working shard can serve the group.
- The old report leaves several native operations unclassified, including
  generalized routing, cache update and async CCL. Its reported modeled-op
  DRAM percentage and merged per-op row sums are not the complete-layer
  roofline. Final accounting must use the native whole-layer window and the
  consistent denominator in [roofline_basis_audit.md](roofline_basis_audit.md).

## Remaining evidence order

Complete the parent-owned precision-locked shared-geometry, adapted AGMM and
combined-CCL comparisons first. Keep one change per A/B and compare to the
strongest correct cumulative path. Use a new bounded native profile to verify
which roles remain material, then test the applicable dense DRAM-sharding,
role-specific subblock/K-block, packed/split shared-GU and persistent-CCL
candidates above. A source-compatible design is a candidate; a failed helper
or generic report hint is not a family-wide rejection.

Final stage acceptance still requires selected-default paired native target
measurements, complete capability/trace/cache/stack coverage, Watcher and an
independent clean-pass review. This audit closes the historical advice
inventory, not those experiments or overall optimization acceptance.

## Follow-up experiments (current resume)

The inventory above records the initial audit, not the final status of each
experiment. These subsequent matched-policy whole-layer controls are complete:

- Shared geometry1/2 and DRAM-bank copies: geometry1 beats automatic, geometry2
  and DRAM copies. Both geometry1/2 use the selected BFP8 sliding/BFP4 full
  weights and LoFi, avoiding mixed-precision geometry comparisons.
- Grouped shared/routed RS+AG retains branch identity through independent norms
  and is faster with identical PCC vectors.
- Adapted Ring/sharded AGMM with both layers' compatible distributed residual
  chains passes real4096/128 but loses to the replicated layer path. Tuned
  producer-to-sharded-norm MMRS components also lose their matched controls.
- Attention QKV/WO decode LoFi passes individually and together. BF16 attention
  CCL passes both layer kinds, mixed stack and heterogeneousB32 checks. BF8
  candidates expose replay/stack failures and remain under AutoFix investigation.
- Four separate attention DRAM-bank role/kind trials pass but lose whole-layer
  latency; packing, current dtypes/fidelity, conversion and collective costs
  are included. FullQKV also reduces PCC to .995221.
- EP-only decode remains slower under the same updated shared/projection/CCL
  policies; hybrid retains gate-selected indexed TP decode and EP prefill.

Artifacts: shared_dram_grouped_results.md, AUTOFIX_fused_ccl.md,
attention_ccl_dtype_results.md, attention_dram_results.md,
stack_selected_bf16.json and batch32_{sliding,full}_selected.json.
Current-source native profiles are still required before closing remaining
material role-specific advice. No new per-op cost is inferred from oldv0rows.

## Current candidate native capture (100k profiler buffers)

`profile_selected_sliding_100k` has complete 4096-token prefill and 128 advancing
traced decode sessions on four devices. See `capture_integrity.json`,
`final_capture_integrity.json`, `whole_layer.json`, `perf_report_provenance.json`
and preserved `*_table.txt.gz` / `*_perf_report.csv.gz`. The 250k profiler buffer
configuration overflowed a native uint32 allocation; the 100k configuration
closes normally without changing model execution or bypassing assertions.

The whole-layer windows are 338681.719us prefill and mean800.823us decode.
The tool-selected per-op subtotals are 92202.056us and 679.812us respectively;
these are diagnostic subtotals, not roofline denominators. Heavy prefill host
instrumentation introduces device gaps (same-run host340208us, ordinary warm
unprofiled candidate about94ms). Whole-window rooflines retain those gaps.
Native per-device spans do not assume synchronized chip clocks.

| Current operation group | Decode us/replay, tool-selected ops | Remaining action |
|---|---:|---|
| BinaryNg | 110.525 | Distinguish expert elementwise work from normalization/residuals; all decode expert intermediates already L1 |
| TilizeWithValPadding / untilize / repeat | 68.565 / 20.444 / 7.308 | Try sharded native decode RoPE broadcasting to avoid repeated head-table construction |
| Sparse gate/up and down | 54.07 / 13.52 | Indexed active8 replay AutoFix first; then legal role-specific N2/K candidates |
| AG / RS | 50.786 / 44.072 | Grouped branch reduction selected; compatible Ring/sharded fused families already measured slower |
| QKV / WO | about24.93 / 9 | Try legal output subblocks beyond1; same LoFi/BFP8 policy; DRAM copies already rejected by whole-layer measurement |
| Router | about13 | Legal N2 output geometry; do not follow erroneous generic BFP8 fidelity advice for actual BF16 weights |
| RotaryEmbeddingHf | 29.755 | Sharded decode candidate, preserve FP32 and current compute config |

Prefill sparse projections consume about50.079ms of92.202ms op subtotal;
active32-owner union computation is retained. Generic minimal-matmul missing
config advice is not evidence that its existing explicit K/M settings are
absent. Shared prefill down K1 remains a legal K17 candidate. Candidate selection
is pending expert repair and these material matched-policy measurements.

The final grouped MoE collective is still BF16, independently of attention CCL
policy. After replay repair, a separate BF8 grouped-MoE payload trial remains
applicable: cast the paired shared/routed buffer before RS/AG, then restore BF16
before independent branch norms. Keep gate-selected sparse work and attention
policy fixed. This compares the MoE communication boundary under the selected
replicated/grouped topology; the attention BF8 experiment does not answer it.

## 2026-09-27 integrated router109 update

The preceding pending labels describe historical work. BF8 attention CCL now
passes ordinary paired4096/128 for both kinds and a33-token mixed stack after
router placement at(10,9). Native mechanism remains unproven; restored and
reallocated core1 controls fail. See AUTOFIX_full_router1_bfp8.md.

Current matched sparse geometry candidates N2/K44, N2/K88 and downN2/K6 all
pass exact replay/cache/PCC but are slower for both kinds. Artifacts carry
`*_router109_sparse_*` names. DenseN2/N4 tests for QKV, WO and router all pass;
slidingQKV N2 saves~6µs (731.48 vs737.56µs). FullQKV N4 and wider routers lose;
WO changes are marginal (fullN2 723.57 vs724.29µs). See router109_dense_results.json.
These are whole-layer warmed host intervals, not isolated operation/device time.

Native sharded decodeRoPE was implemented as an off-by-default candidate and
failed key-cache PCC, preserving valuecache accuracy. AutoFix is actively
checking table/operand formats and destination-register limits, with a fresh
source audit and exact-width components. Initial failure is not a family-wide
rejection. GroupedMoE BF8 payload, larger projectionK blocks, direct sliding
expertBF4 quantization and packed/split projection controls remain prepared
but unmeasured. Final defaults/profiles/contract/Watcher/review remain pending.

## Final-policy matched controls (a12a913c)

All six real4096/128 controls pass output/cache PCC and exact replicas/replay. Selected ordinary stress decode host medians are650.445us/sliding and724.702us/full. Wider sparse gate N2/K44 and N2/K88 yield681.535/683.980us, down N2/K6 yields652.010us under the final BFP4 weights/BFP8 input/LoFi policy. Mixed GU4/down8 shared DRAM yields666.306us. The complete compatible Ring + sharded residual + fused AGMM path consumes the sharded stream through its norm/residual/next projections; it yields810.761us sliding and877.898us full. These whole-layer measurements reject those coherent alternatives; boundary gathers are outside measured decoder time. See final_policy_alternatives.json and each command JSON. Prior dtype-incompatible trials are not used for these rejections.
