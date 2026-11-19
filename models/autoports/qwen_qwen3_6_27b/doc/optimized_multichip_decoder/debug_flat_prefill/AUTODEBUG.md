# AutoDebug: TP4 prefill repeatedly discards its compact token representation

## Finding

The reported performance defect is supported by both source and the matching profiles: token-independent helpers repeatedly turn compact token rows back into separately padded users, and the next helper immediately compacts them again. At B32/S33 these are real tile data rearrangements. They are not cheap views, and no native reshape correctness defect is needed to explain the measured cost.

The smallest broad fix is a **flat representation for each existing prefill chunk**, retained through hidden residuals, hidden normalization, column and row projections, CCL, gating, and the entire MLP. Carry the original logical batch and chunk length separately. Restore user/time structure at the existing GDN and full-attention temporal boundaries, then compact their valid output rows before the output projection. Restore the public batched residual at the end of each chunk. Do not flatten requests into a single temporal sequence.

**A shape-only patch is unsafe.** In particular, flat storage silently changes the default full-attention hidden-normalization route, and changes several small-tail dispatch decisions. These must use logical metadata rather than the storage shape.

Confidence is high in the source diagnosis and boundary requirements. This investigation did not run devices or implement a candidate, so net latency improvement and numerical equivalence of an implementation remain unmeasured.

## Scope and evidence provenance

- Inspected checkout HEAD: 8502f42bca20dcad1507fbc8cebf7032220b7d8e.
- Inspected accepted multichip decoder SHA256: 082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44.
- Inspected optimized decoder SHA256: 315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22.
- The requested “082c82fa” is the multichip **file hash prefix**, not an available Git revision. Both files match the hashes in the accepted B32/S33 profile run records.
- Exact profile artifacts: [GDN accounting](../../multichip_decoder/perf/delivery_profile_retry_l0_b32/accounting.json), [attention accounting](../../multichip_decoder/perf/delivery_profile_retry_l3_b32/accounting.json), corresponding prefill_perf_report.csv and compressed raw ops CSV parts; run records are delivery_profile_retry_l0_b32.json and delivery_profile_retry_l3_b32.json in doc/multichip_decoder.
- Their fourteen ReshapeViewDeviceOperation entries total **4042.521 µs/GDN** and **3686.299 µs/full attention**, matching [profile_accounting.md](../../multichip_decoder/profile_accounting.md). Nearby final_safe_profile artifacts have slightly different timings and were not substituted for these totals.

Source references below use **M** for [tt/multichip_decoder.py](../../../tt/multichip_decoder.py) and **O** for [tt/optimized_decoder.py](../../../tt/optimized_decoder.py), with line numbers from the accepted files at Git HEAD 8502f42b. Native source paths are relative to the repository root.

During final review, concurrent workspace edits added optional packed_qkvg/direct_compact_attention code to multichip_decoder.py, changing its working-tree hash to fc3dadfa512cf28d4db859611198a993a59652595071afc32c87d65170cf08f2 and shifting line numbers. A fresh git-show hash check confirmed that HEAD still supplies exactly the accepted 082c82fa file. This report diagnoses that accepted snapshot; it does not combine its profiles with the concurrent candidate's behavior or evaluate that candidate.

This investigation and its reviewers wrote only this report. We did not modify model, test, or kernel source or operate devices; the concurrent changes were left untouched.

## Direct observations: the fourteen reshape kernels

Each row below names a physical pack or unpack between batched and compact token storage. Timings are µs from the accepted per-operation report; each report selects its slowest-rank operation. They are overhead accounting, not a prediction of additive wall-clock savings.

| GDN movement, in execution order | µs | Source cause |
|---|---:|---|
| BF16 residual 1280: pack | 95.010 | M:420, first hidden norm |
| BF16 normalized hidden 5120: unpack | 508.793 | M:432 |
| BF16 hidden 5120: pack for primary projection | 348.479 | O:499–503 |
| FP32 primary 4160: unpack | 434.219 | O:513/609 |
| BF16 hidden 5120: pack for correction projection | 346.826 | O:499–503 |
| FP32 correction 2624: unpack | 287.765 | O:513/609 |
| BF16 gated GDN output 1536: pack | 99.590 | M:362 |
| BF16 projected hidden 1280: unpack | 134.638 | M:399 |
| BF16 residual 1280: pack | 94.049 | M:420, post-attention norm |
| BF16 normalized hidden 5120: unpack | 508.701 | M:432 |
| BF16 MLP input 5120: pack | 348.205 | M:439 |
| BF16 SwiGLU output 4352: unpack | 434.090 | M:457 |
| BF16 SwiGLU output 4352: pack | 268.499 | M:362 |
| BF16 projected hidden 1280: unpack | 133.657 | M:399 |
| **Total** | **4042.521** | |

| Full-attention movement, in execution order | µs | Source cause |
|---|---:|---|
| BF16 residual 1280: pack | 94.073 | M:420 |
| BF16 normalized hidden 5120: unpack | 508.655 | M:432 |
| BF16 hidden 5120: pack for QKV | 349.057 | O:503, called by M:499 |
| BF16 QKV 2048: unpack | 206.841 | O:513/609, before M:502 |
| BF16 hidden 5120: pack for gate | 351.470 | O:503, called by O:1304 |
| BF16 gate 1536: unpack | 158.576 | O:513/609 |
| BF16 gated attention output 1536: pack | 98.961 | M:362 |
| BF16 projected hidden 1280: unpack | 134.192 | M:399 |
| BF16 residual 1280: pack | 95.156 | M:420 |
| BF16 normalized hidden 5120: unpack | 508.648 | M:432 |
| BF16 MLP input 5120: pack | 347.163 | M:439 |
| BF16 SwiGLU output 4352: unpack | 433.151 | M:457 |
| BF16 SwiGLU output 4352: pack | 266.432 | M:362 |
| BF16 projected hidden 1280: unpack | 133.924 | M:399 |
| **Total** | **3686.299** | |

The raw/report join uses Global Call Count and device identity. The selected reshape entries are cached native operations on 110 cores with DRAM-interleaved input and output. MLP's 4352-wide unpack/repack alone costs 702.589 µs/GDN and 699.583 µs/attention. GDN also independently packs the same normalized hidden for its primary and residual-correction projections and expands both FP32 outputs before their BF16 combine (O:495–497).

The existing flat_ccl and flat_norm policies operate *inside* helpers. M:352 and M:432 restore the caller's batched shape, so these flags do not provide persistent flat storage. TP4 constructs its own policy dictionary (M:53–86); it does not inherit the single-chip DEFAULT_POLICIES. Consequently, the single-chip flat_prefill_correction/chunk_add_prefill_correction helpers are not enabled by TP4 defaults.

## Concrete storage geometry

For the **current chunk**, let B be logical users, n be valid tokens per user, R=B*n, P=32*ceil(n/32), and RP=32*ceil(R/32). Row order must be:

    flat[0, b*n+t, c] = batched[b, t, c]   for 0 <= t < n

For standard 32×32 TILE storage with tile-aligned width C:

| Representation | Logical shape | Padded shape |
|---|---|---|
| Public/current-chunk residual on each TP rank | [B,n,1280] | [B,P,1280] |
| Compact residual on that rank | [1,R,1280] | [1,RP,1280] |
| Compact normalized hidden, replicated over ranks | [1,R,5120] | [1,RP,5120] |
| Internal rank-4 CCL/norm form | [1,1,R,C] | [1,1,RP,C] |

At B32/n33: P=64, R=RP=1056. Batched storage contains 2048 physical rows, while compact storage contains 1056. User 1 token 0 is compact row 33 but batched physical row 64. User 16 token 0 is compact row 528 but batched physical row 1024. A tile in compact storage can contain the end of one user's sequence and the start of the next user's sequence; this is safe only for independent-row operations.

This also proves why padding-volume equality is insufficient: B2/n31 uses 64 padded rows in either representation, but user 1 starts at compact row 31 versus batched row 32.

The public reshape wrapper only takes its metadata-view path when the last width and placement/shardedness match, and the penultimate logical dimensions match or both are tile-aligned (ROW_MAJOR is a separate case): ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:757–770. Otherwise this TILE case lowers to reshape_tiled (:807–816), which invokes prim::reshape_view (:501–538), i.e. the profiled operation. Public view validates the same tile-boundary constraint in data_movement/view/view.cpp:24–46.

Thus, for this standard layout and unchanged width/placement, [B,n,C] ↔ [1,B*n,C] is a data-preserving metadata view when B=1 or n is a multiple of 32. B32/n33 does not qualify even though R is tile-aligned. Adding/removing the leading singleton for [1,R,C] ↔ [1,1,R,C] is harmless.

## Minimal safe intervention

Implement this as an explicit TP4 prefill path, retaining the existing public API and time-chunk scheduler. The conservative initial scope is the accepted default sharded-residual configuration; leave replicated controls and incompatible policy overrides on their existing path unless their dispatch is preserved explicitly.

1. Keep O:1147–1209's list handling, length validation, per-user chunk slicing, absolute positions, and recurrent-state publication. After selecting chunk=[B,n,hidden_width], compact it once to [1,R,hidden_width]. Keep B and n as metadata.
2. Apply the same hidden-normalization route to the compact residual and return compact replicated hidden [1,R,5120]. The distributed path's existing x4/work_shape already has this representation; omit its final batched restore. Preserve the route-selection rules described below.
3. Perform column projections on compact hidden. For GDN, compute primary and correction in the same flat rows, combine in FP32 into BF16 [1,R,4160], then bridge to the temporal layout. For full attention, project QKV to [1,R,2048], restore [B,n,2048] before the existing head splitter, and retain flat hidden for the separate gate projection.
4. Keep the existing temporal algorithms and state/cache writes. Compact each user's valid attention/GDN output to [1,R,1536] before gating/output projection. Gate and Z multiplication can operate in flat rows.
5. Let the row projection return flat output. Preserve its existing MMRS admission and ≤128-row pieces. _reduce must return [1,R,1280] for sharded residual addition, whether the projection was fused or used separate matmul/RS.
6. Keep the first residual add, post-attention norm, minimal SwiGLU, down projection/reduction, and final residual add flat. In M:437–457, the activation can stay [1,R,4352]; remove the restore at :457. Both residual operands must have identical flat token mapping.
7. Restore each finished chunk to [B,n,1280] before the existing output concat along time. This preserves public layer boundaries and chunk ordering. Keeping storage flat across public layer calls is a separate API change and is unnecessary for this fix.

Conceptually:

    chunk[B,n,1280] -> residual_flat[1,R,1280]
      -> same_hidden_norm -> hidden_flat[1,R,5120]
      -> flat_in_projection -> per_user_temporal_bridge -> existing_temporal_ops
      -> compact_valid_output[1,R,1536] -> flat_gate_and_out_projection
      -> reduce_if_needed -> residual_add[1,R,1280]
      -> same_hidden_norm -> flat_SwiGLU[1,R,4352]
      -> flat_down_projection -> reduce_if_needed -> residual_add
      -> public_chunk[B,n,1280]

A conservative GDN bridge can unflatten the whole **combined BF16** packed tensor [1,R,4160] once, reuse the existing temporal body, and flatten its gated [B,n,1536] output once. This already avoids the two FP32 expansions and repeated hidden/MLP conversions. A narrower bridge can unflatten only QKV and beta/g, retain Z flat, and gate after compacting recurrent output; that requires additional dtype-specific boundary handling and is a subsequent refinement.

The existing corrected_projection_flat_prefill (O:2279–2287) provides a small independent first step: one hidden pack, two flat FP32 projections, one BF16 combine, one BF16 temporal unpack. Its output is still batched, so enabling it alone does not fix the whole representation lifetime. Existing prefill_unflatten_bf16 (O:4625–4660) implements valid-row scatter and individual tail zeroing, but accepts only interleaved BF16 TILE with its asserted shapes. It cannot unpack FP32 decay or correction partials. Ordinary reshape is the minimal existing bridge; selecting a custom movement helper needs separate validation.

For B32/n33, a conservative broad implementation needs approximately four batched/compact representation crossings per layer: residual entry pack, temporal input unpack, temporal result pack, and public residual exit unpack. This counts explicit representation boundaries, not every kernel: native head splitting/concatenation, pad/trim, and temporal data movements remain. GDN introduces a BF16 packed-input unpack where two FP32 projection unpacks existed. Full attention retains the existing QKV unpack and output pack. The fourteen measured reshape costs are therefore a useful removal budget, **not a claim that all 4.043/3.686 ms disappear**; necessary bridges and scheduling/memory effects remain.

## Boundary invariants

### Hidden residual, normalization, CCL, and MLP

- “Sharded residual” means hidden-dimension ownership across the four mesh ranks, not necessarily an L1-sharded tensor. Rank r retains its own 1280 contiguous hidden channels. Do not gather residuals just to implement flat storage (M:89,255–256).
- Distributed RMSNorm still consumes the same four 1280-wide hidden shards. M:418–432 uses BF16 pre-all-gather statistics, post-all-gather RMSNorm with the rank's affine shard, then hidden all-gather. Flat storage must not reduce over tokens or users, or replace this with local-only normalization.
- CCL still uses dim=3, cluster_axis=1, the same rank order, dtype policy, topology, links, and semaphore manager (M:326–352). It gathers/reduces width, never the token axis.
- At B32/n33, distributed norm operates on [1,1,1056,1280]; replicated normalized hidden is [1,1056,5120]. Paired SwiGLU weights are [5120,8704] per rank and produce [1,1056,4352] (M:210–219,437–457).
- Preserve BFP8 weights, compute configs, SwiGLU fusion/rounding, activation/output dtypes, and K/N geometry. Removing representation restores does not justify changing these numerical choices.
- Row projections are conditional: for sharded defaults and 512<R≤8192, native MMRS returns width1280; other sizes produce width5120 rank-local partials and require the existing reduce-scatter. At R=1056 use eight 128-row pieces plus one 32-row piece. Preserve R-based thresholds, padding, trim, and the NoC safety workaround at M:354–399. Do not replace these pieces with an unvalidated tall native plan.

### GDN

At B32/n33 the packed BF16 projection is [1,1056,4160]. Its local column ranges are QKV [0,2560), Z [2560,4096), beta [4096,4108), and decay input [4128,4140); the unused gate columns remain padded. Correction has logical width2624, with QKV then padded BA, not Z. Physical primary/correction weight widths are normally 4608/3072. Preserve true tensor strides, never infer them from logical widths (M:148–162,263–279,1138–1190).

Before temporal processing:

- Each user's convolution input is [1,64,2560], selected from that user's 33 valid rows and explicitly zero-padded. It receives that same user's ROW_MAJOR history [1,3,2560], with actual_start=0 and **actual_end=33**, and returns that user's updated carry (M:647–667).
- Concatenated Q/K are [32,64,512], V [32,64,1536]. Beta/g are [32,64,12], with gates for t≥33 explicitly set to zero **after sigmoid/decay** (M:668–708). Padding a projection with zeros is insufficient: sigmoid(0)=0.5 and decay generally is nonzero.
- Initial/final recurrent state remains [32,12,128,128], with 4 local key heads and 12 local value heads, and three adjacent value heads per key head. Chunk size remains 32.
- Preserve live FP32 recurrence and the caller's persistence dtype: the default BF16 recurrent state is widened for prefill and published back to BF16, while an incoming FP32 state remains FP32. Preserve BF16 conversion before the existing per-head RMSNorm, affine/gating order, and BF16 residual arithmetic (M:709–719; O:1169–1171,1205–1208). The packed projection still uses FP32 primary/correction and exactly one BF16 combine; do not replace add/round boundaries with a standalone cast.

These are lowered contracts, not conventions inferred only from Python. Native convolution requires [1,T,W], T tile-aligned, and one matching carry (experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_device_operation.cpp:115–120,154–170). Its history-exchange kernel assembles the last valid rows and retains older history when fewer than three valid tokens arrived (device/kernels/dataflow/history_exchange_qkv_causal_conv1d_silu.cpp:83–109).

Native chunk_gated_delta_rule reads B and T from Q, validates beta/g as [B,T,HV], and reshapes state using B*HV (transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp:145–200,213–230,281–290). Its “flat QKV” option flattens **heads into feature width**, retaining [B,T,...]; the reader addresses a separate padded batch base and maps value head hv to key head hv/3 (device/kernels/dataflow/reader_chunk_gdn_prep.cpp:133–171). Passing [1,B*n,...] merges users into time and violates the state contract.

### Full attention

For B32/n33, expand token-major projection [1,1056,2048] to [32,33,2048] before the existing split. Local Q is [32,6,33,256], K/V [32,1,33,256]. Norm and RoPE retain these user/head axes; the temporal Q/K/V inputs are independently padded to [32,6,64,256] and [32,1,64,256] (M:496–514; O:1247–1279,1383–1409).

- Each user keeps its own cosine/sine rows. Position is start_pos+t, never start_pos+b*n+t. RoPE rotates 64 channels and leaves the remaining 192 unchanged.
- KV remains BFP8 [physical_pages,1,32,256] on each rank, with a replicated page table and cache fill batch_idx equal to the logical user. Keep existing page IDs, page slicing, casts, and finite per-user tail padding.
- Native chunked SDPA always uses causality (transformer/sdpa/sdpa.cpp:121–131), and requires Q.batch == page_table.rows (device/sdpa_device_operation.cpp:265–305). Flat Q with batch1 cannot consume the original 32-row page table.
- After SDPA, crop every user's token axis to n before concatenating heads/compacting; [32,33,1536] maps back to [1,1056,1536]. Project the separate gate on flat hidden; preserve fused sigmoid multiplication and the row output projection.
- Splitting heads while still flat produces [1,H,R,D]. A direct reshape to [B,H,n,D] generally interchanges user/head ordering. The required mapping is Q[b,h,t,d]=Qflat[0,h,b*n+t,d]. Unflatten token-major QKV before splitting for the minimal safe implementation; a post-split alternative needs factor/permute/repack.

## Shape-dependent dispatch must remain semantic

| Decision | Existing behavior | Flat-path requirement |
|---|---|---|
| Full-attention gather_norm='single' | M:410–416 tests logical B==1 in prefill | Use original B. At B32/S33 retain distributed RMS pre/post all-gather; physical batch1 must not select gather-then-RMSNorm. |
| GDN one-token temporal branch | M:636–689 selects s==1 and uses logical B | Use n and B explicitly. A [1,B,C] tail is B users, not B sequential tokens. |
| Small GDN correction L1 combine | O:471–494 requires batch1 and seq≤32 | Flat B32/n1 would newly enable it. Preserve the prior logical-batch predicate, or validate this as a separate memory/dispatch change. |
| Replicated hidden norm / gather overrides | O:612–625 selects SFPU norm for batch1 and seq≤32; O:630–634 has a different sum route | Do not let flat storage enable a different norm. Retain the old path for replicated controls/unsupported overrides or explicitly select the same route. |
| CCL temporary placement | M:328 chooses L1 when its input seq≤32 | Separate row-reduction inputs B>1,n≤32,R>32 would change placement after flattening. Carry the original extent for this decision if preserving the route. Norm collectives already receive flat work_shape; do not apply a blanket n-based override to all collectives. |
| Native MMRS / ordinary row projection | M:358–399 uses B*seq | Continue using R=B*n, and preserve separate paths for R≤512 and R>8192. |
| Per-head attention norm, compact head decode | Operate on actual user/head shapes | Preserve the existing temporal layout and logical n==1 conditions; do not infer them from compact hidden. |

These are concrete hazards for a proposed optimization, not evidence of an existing numerical error in the accepted B32/S33 implementation.

## Arbitrary lengths, continuations, and output order

Retain the existing scheduler before making a chunk flat. In particular:

| Request / chunk | Compact rows | Temporal rows per user | Existing dispatch to preserve |
|---|---:|---:|---|
| B32/S33 at start0 | 1056 | 64 | Intermediate-row MMRS pieces; normal batched temporal prefill |
| B1/S33 | 33 (physical64) | 64 | Batch flatten itself is already a view; preserve B1 norm policy |
| B32/S513, first chunk n512 | 16384 | 512 | Large-row separate matmul/RS |
| B32/S513, final chunk n1 | 32 | 32 physical | Small-row separate matmul/RS; GDN one-token batched conv/local recurrence |
| B2/S31 | 62 (physical64) | 32 | Equal padded volume does not remove the user-boundary rearrangement |

O:1174–1199 selects a one-token full-attention update while absolute position is not aligned to _attention_chunk(state), then uses normal prefill. The alignment is 32 or 512, determined from page-table capacity (O:1237–1245). Keep this logic; padded cache fills must not overwrite an existing prefix.

Do not flatten a full S513 request and then take a single contiguous 512*B-row slice. User b's source row is b*513+offset+t, whereas the current compact chunk's destination row is b*n+t. Similarly, concatenating flat chunk outputs along their row axis gives chunk-major order, not the required batch-major public tensor. Unflatten each finished chunk before the existing time concat for the smallest fix.

Keep list-based unequal-length requests on their independent state/start/rope paths (O:1155–1162), and retain the exact FP32 live recurrent state across chunks before publication. Conv end descriptors only cover n=1..512 (M:184–188), reinforcing that the flattened row count cannot become its temporal length.

## Review, rejected alternatives, and remaining validation

The first-draft claims were checked against source, lowered validation/readers, and exact profile identity. The leading diagnosis explains why B32/S33 is dominated by reshapes while B1/S512 has no corresponding batch compaction cost, why the GDN projection adds FP32 expansion traffic, and why repeated MLP/norm transitions contribute to both layer kinds. It does not claim that all large-context latency is reshape overhead: S513 uses different chunk sizes and row-projection dispatch, and its profiles also contain substantial matmul, slice, and CCL cost.

Other potential issues / rejected explanations:

- Replacing reshape with view or forcing padded metadata does not compact arbitrary logical lengths. Public view rejects this S33 case; a low-level alias cannot implement the missing data mapping.
- Keeping [1,B*P,C] with padding gaps is a different representation. It can support independent-row math with explicit b*P+t indexing, but changes executed M, thresholds, tail handling, and required gather/scatter boundaries. It is not a drop-in optimization for [1,B*n,C].
- ROW_MAJOR flattening can be a view, but untilize/tilize adds physical movement; it does not establish an improvement.
- Changing precision, RMSNorm algorithm, temporal GDN, causality, page ownership, or the proven MMRS safety partition is unnecessary to explain or remove this overhead. No additional current correctness bug is established here.

Completed inspection-only checks:

- Recomputed both source hashes, matched accepted profile metadata, parsed both Python modules with ast.parse, and checked the fourteen-op totals.
- Independently traced normalization/MLP contracts, temporal boundaries, and native reshape lowering with three isolated xhigh source reviewers; cross-checked the critical findings against the files.
- A standard-library-only row-index check covered B in {1,2,3,32} and S in {1,17,31,32,33,63,64,65,257,511,512,513,1025}: 52 cases passed compact/scatter mapping and per-chunk output-order reconstruction. This verifies index arithmetic, not TTNN execution.

Implementation follow-up must validate on hardware; none of the following is reported as completed:

1. Compare all valid output rows, residual contributions, per-user PCC, and conv/recurrent/KV state against the accepted implementation and existing numerical thresholds (run_multichip_decoder.py:190–220 uses ≥0.995 and finite checks). Include distinct per-user inputs near flat rows31/32/33 and user15/16/31 to expose accidental user mixing.
2. Cover both layer kinds, B1/B2/B3/B32, n1/17/31/32/33/63/64/65, total513/1025, and the existing 256/257, 512/513 and 8192-row MMRS boundaries; test non-tile R as well as tile-aligned R. Check fresh and continued states, unaligned attention starts, permuted physical pages, unequal-length lists, and sharded multi-layer chaining.
3. Check small tails for norm/correction/CCL route parity. Preserve trace output/state equality, no runtime host tensor conversion, and the existing watcher/resource checks; inspect allocations because representation lifetime and aliases change even when arithmetic does not.
4. Profile matched warmed runs with explicit completion. The current harness synchronizes before/after timed prefill (run_multichip_decoder.py:330–340); Python enqueue completion alone is not evidence. Confirm fewer reshape kernels and report necessary bridge cost plus uninstrumented latency, rather than subtracting the old operator totals from wall time.

