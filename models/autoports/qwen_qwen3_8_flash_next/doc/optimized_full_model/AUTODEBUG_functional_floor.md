# AutoDebug: exact-math GR/GDN performance diagnosis

2026-11-14. Inspection only: no hardware access, model execution, resets, target tests, implementation edits, or new profiling. Only AUTODEBUG.md was written. Three independent xhigh source investigations covered accurate reduction, projections/residuals, and collectives. The parent retains device ownership and all48 validation.

Prepared HEAD: de432e04c21c3ea53ffe13d9c6e78a96f7acb704, with existing stage7 working-tree changes retained.

Inspected focus SHA256: fused_decoder.py = d638f10da2929e6898f63bac36151590c64eb5feb1ad184c3942053f59ae1b5b; multichip_decoder.py = ef79bc16c8a9c7fe31f285a633b2aaca85446b525bdfbfd245a80dc6b661683c. Current model.py = 39fa776d03e3ea0af91f2f0415000734ce346c0b443db1dd0a8e86be4637ebf9.

## Leading findings

1. **FusedPreciseProjection still materializes full FP32 products.** Current GR down/up and GDN output paths write an output-by-K product to DRAM, launch an accurate SFPU sum, reshape, then narrow to BF16. Despite the name/docstring, this is not streaming multiply/reduce fusion. It is a source-backed structural bottleneck, not a newly demonstrated numerical bug.
2. **GR down's three-core reduction follows the factory's work partition.** Physical96 output rows become three32-row tasks, each traversing320 K tiles. A bigger grid creates no more tasks. Fusing into that same partition can regress because the previously110-core product moves onto three cores. Finer independent output-row work and coordinated output assembly are needed.
3. **Exact fusion must preserve the selected SFPU order.** The W reduction accumulates matching lanes across K tiles first, then applies the Blackhole intra-tile row tree. Per-tile scalar sums, ordinary split-K partials, native matmul/FMA, and the existing balanced QKV reducer are different arithmetic.
4. **CCLsum has removable local preparation around unchanged transport.** The generic FP32 outer-axis sum clones/fills token padding before HC transpose/H-reduce/transpose-back. Its selected HC writer already skips original token padding and fills the new rank padding. A narrowly guarded clone/fill removal is a strong smaller candidate; a custom post-gather reducer is a larger overlapping candidate.
5. **There is no measured basis for promising50ms.** Historical GR down/up and GDN-output multiply+reduce rows total20.796698ms when weighted across the stack. Their broader local chains plus compact down assembly and GR/final-mixer post-gather reduction total31.338994ms of disjoint historical kernel time. These are optimistic zero-cost ceilings in an older eager profile, not trace-speedup forecasts. The selected trace needs49.736174ms improvement.

Prioritize precise projection fusion and complete-layer validation. Preserve the selected scorer, LM-head20/3/2, synchronous CCL, precision, rounding boundaries, and one whole residual stream/rank.

## Observations, interpretations, and provenance

Artifact observations, recomputed using Python standard-library CSV/JSON arithmetic:

- [selected_resident_v1.json](models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/selected_resident_v1.json) records113.88145404772496ms/token and8.78106104599722tokens/s/user for all48 resident token-out. No reproduction was performed here.
- [functional_floor_audit_v1.json](models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/functional_floor_audit_v1.json) counts3,281,879,040 prepared major-weight bytes plus2,359,296 useful selected-KV bytes/rank at512GB/s/rank:6.414528ms, or5.632636% of the recorded latency. The10% threshold is64.145280ms,15.589612tokens/s/user. Required reduction:49.73617404772496ms.
- **Do not add avoidable FP32 product/scorer/intermediate traffic to that denominator.** Implemented-graph traffic is useful for diagnosing wasted work, not for relaxing this FUNCTIONAL bar.
- [model_perf_report.csv](models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_final_perf_v1/model_perf_report.csv) has1253 merged model rows; SHA256152f944b22d00b966eb45e39ad936871cab1266dc3bf7afbf23eac418f1508c8.
- [ops_perf_results.csv](models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_final_perf_v1/ops_perf_results.csv) has21494 records including signposts,21488 device-operation records; SHA256055fbc6cbcada681661804279df2f26dd3187978b0eee42280543b9cf87f8854. Architecture is Blackhole.
- [coverage.json](models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_final_perf_v1/coverage.json) identifies one eager reduced0/1/3 profile,1253 model ops/rank. The report merges ordinary ops by maximum rank duration and collectives by mean rank duration. Its sum is neither one rank's wall time nor traced token latency.

All row intervals below are zero-based, half-open data-row indices, excluding the header. Device Time is microseconds; raw kernel durations are nanoseconds.

| Historical region | Rows | Merged kernel us |
|---|---:|---:|
| Entry/embedding | 0:5 | 39.06975 |
| Complete L0 | 5:298 | 2022.538 |
| Complete PLE/L1 | 298:795 | 3043.04075 |
| Complete QSA/L3 | 795:1204 | 5066.12250 |
| Terminal/row assembly | 1204:1253 | 1086.03650 |
| L0 first GR read | 5:58 | 407.49450 |
| L0 GDN | 58:161 | 670.82325 |
| L0 first residual write | 161:169 | 31.508 |
| L0 second GR read | 169:222 | 400.734 |
| L0 MoE | 222:290 | 482.30025 |
| L0 second residual write | 290:298 | 29.678 |

35*L0 +L1 +12*L3 +entry +terminal =135.750447ms of historical merged eager kernel time, before the separate split sampler. It is not a decomposition of the selected113.881454ms trace. Eager operation gaps, profiling overhead, and per-rank/merged durations cannot be substituted for clean trace time.

**Revision consistency:** historical QSA includes the obsolete capacity scorer and historical head. Current [MeshQSA._selected_ids](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L1029) uses causal_indexer_scores(...packed_decode=True); current [head policy](models/autoports/qwen_qwen3_8_flash_next/tt/model.py#L127) is20 cores/3 readers/block2. Exclude removed key casts/products and the old2.387150ms/layer scorer section from remaining savings. The selected scorer still has terminal materialization, which is outside this GR/GDN budget.

Historical fused_decoder.py matches the inspected hash. Historical multichip_decoder.py/model.py differ from current stage7, but relevant GR/GDN/Collectives/FinalMixer operation sequences remain unchanged. Current fused_decoder.py, multichip_decoder.py and model.py match selected_sources_v1.json. This supports a structural diagnosis, not transfer of timings between execution regimes.

Some BinaryNg annotations are inconsistent: row32 repeats a down-projection shape although its following reduction is [2560,320]. Source and adjacent reduction metadata establish runtime dimensions. Do not infer product FLOPs/bandwidth from those anomalous annotations.

Interpretation: remaining explicit materialization, coarse reduction work, and local format preparation are credible optimization targets. Their realizable speedups, resource fit and contribution to the selected trace remain unmeasured.

## Current projection and residual contracts

[FusedPreciseProjection.__call__](models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py#L79) widens the activation, creates products at line95, calls accurate sum at114–120, then reshapes/concatenates/typecasts at124–126:

    BF16 input -> FP32 widening/reshape
      -> broadcast SFPU multiply against resident weight
      -> FP32 product tensor in DRAM
      -> accurate W sum -> physical reshape -> block concat -> BF16 typecast

[MeshExactProjection](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L124) inherits this body, defaults balanced_tree=False and uses widen_in_unpack=True. GR weights are already FP32; BF16 GDN weights widen in the binary unpack. Do not credit a nonexistent runtime weight cast.

| Per-rank role | Useful /physical output N | K | Weight /product-sum dtype | Decode W tasks |
|---|---:|---:|---|---:|
| GR down | 80 /96 | 10240 | FP32 /FP32 | 3 |
| GR up | 2560 /2560 | 320 | FP32 /FP32 | 80 |
| GDN output | 640 /640 | 6144 | BF16 /FP32 | 20 |
| Router, separate secondary scope | 128 /128 | 2560 | BF16 /FP32 | 4 |

Sources: [GR setup/read/write](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L195), [GDN setup](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L234), [GDN output](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L367). There are96 GR pairs plus one final-mixer pair and36 GDN output projections. Their current product write+read subtotal is:

    2*4*[97*(96*10240 +2560*320) +36*640*6144]
    =2,531,000,320 bytes/rank/token

These are avoidable implementation bytes, not a denominator addition or latency prediction. Router products are excluded.

GDN QKV is already [FusedExactQKV](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L250), with its separate preserved balanced tree. Z/A/B already use [compact BF16/HiFi4 projection](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L258), logical1560 outputs/physical1568. GR injection remains its selected prepared BF16 matmul, not an arithmetic alias of exact down. Old generic QKV, per-role ZAB padding and obsolete injection topology earn no new credit.

For Q>1 the inherited implementation processes32-output down blocks,256-output up blocks,32-output GDN-output blocks; Q>32 recursively chunks tokens. A Q1 fast path must preserve fallback until broader signatures pass.

Each rank owns one complete2560-feature BF16 residual stream. Norm covers all2560 features locally. GR gathers four normalized streams into10240; down emits BF16 local96, gathers384, compacts four80-column pieces to320, then scales/activates. Up emits local2560 mixing values. The BF16 mixing*normalized product is gathered in BF16, widened and summed in FP32, scaled0.25, then narrowed.

At [MeshResidual.write](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L225), block*injection is widened/evaluated in FP32 and rounded to BF16; the residual addition then widens and rounds again. Both rounds are mandatory. GDN also preserves its BF16 conv/activation/beta/recurrence-output/normalization/gate boundaries before output projection. Math equivalence over real numbers does not authorize removing them.

## Exact accurate-W order

The executable path is Python sum -> generic_reductions.cpp -> reduce_op.cpp -> ReduceMultiCoreWProgramFactory -> universal reader + compute/reduce.cpp -> compute_kernel_lib::reduce -> Blackhole SFPU LLK.

- [generic_reductions.cpp:745](ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp#L745) dispatches sum and forwards W/configuration.
- [reduce_op.cpp:198](ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp#L198) selects accurate SFPU for FP32, precise mode, non-Quasar, FP32 DEST.
- [W factory:153](ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp#L153) distributes NC*Ht complete tile rows; line477 configures direct FP32 unpack.
- [reader:31](ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp#L31) streams ascending tile IDs.
- [helper:108](ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl#L108) copies the first tile directly and SFPU-adds subsequent tiles. [Lines521–555](ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl#L521) finish all K folds before the intra-tile reduction.

For each output row, p[t,c] is the separately rounded FP32 product at K index32*t+c:

    a[c] = p[0,c]
    a[c] = FP32_ADD(a[c], p[t,c]) for t=1,...,K/32-1
    result = Blackhole_row_tree(a[0:32])

There are32 independent ascending chains. No scalar per-K-tile sum or independent split-K partial exists. Seed with the first product, rather than adding to zero, to retain special-value behavior.

Blackhole [load_row_group:544](tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h#L544) loads left even/odd and right even/odd columns. Its [recorded FP32 chain:1368](tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h#L1368), followed by [mask/rotate merges:638](tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h#L638), implies this ordinary-finite-value tree:

    q[j] = a[2j] + (a[2j+1] + (a[16+2j] + a[17+2j])), j=0..7
    L = (q[0]+q[7]) + (q[6]+q[5])
    R = (q[4]+q[3]) + (q[2]+q[1])
    result = L+R

Each plus is a separate FP32 SFPU operation. Literal final operands reverse for some row groups; this simplified expression does not promise NaN-payload identity. Reusing the LLK preserves that detail and its instruction hazards. Top/bottom face pairs are independent. Only column zero is a defined reduced result: retain pack-mask semantics or assemble only those values. Wormhole differs.

BinaryNg's nullopt compute_config does not imply BF16 multiply. Its factory derives FP32 DEST from operand/output dtype and direct FP32 unpack (binary_ng_program_factory.cpp:1134–1146,1374–1381). [Compute:58](ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp#L58) reconfigures A/B formats; [Blackhole multiply:221](tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h#L221) performs a separate FP32 product. BF16 software narrowing is disabled for FP32 DEST. Preserve separate multiply and addition; matmul/mul_tiles/FMA is not equivalent.

Blackhole BinaryNg disables a particular LLK/MOVB2D FP32-DST row-broadcast path (factory:1450–1465), replicating the activation row in dataflow instead. A custom SFPU row-broadcast primitive is a different possible path and needs product-bit gates.

Final BF16 narrowing uses [explicit integer finite-value RNE](tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h#L273): add0x7FFF plus retained bit, mask low16 bits, store. The literal primitive also defines nonfinite payload behavior; a generic IEEE conversion is not a substitute for that behavior. [ordered_projection.py:155](models/autoports/qwen_qwen3_8_flash_next/tt/ordered_projection.py#L155) records midpoint differences from direct BF16 pack. Do not replace typecast with an unverified pack.

## Ranked interventions

### Streaming precise projection and finer output work

Smallest first implementation: preserve resident weights, mesh ownership, external gathers and signatures. Stream K tiles in ascending order into correctly configured FP32 DEST, generate separate SFPU products, seed/fold exactly, and invoke the existing SFPU W finalizer. Initially emit the same FP32 reduced tensor and retain standalone BF16 typecast. After exactness passes, integrate final-layout assembly and the explicit typecast primitive. Keep physical96 down output initially.

This removes full product DRAM write/read, not weight scans or arithmetic. It is a source-supported design, not a verified kernel/speedup.

For down, retaining three whole-tile-row tasks can lose110-core product parallelism. Assigning independent logical output dots more finely preserves each dot's K order. Q1 down offers80 useful scalar-dot tasks/rank, not110 or96 useful tasks. It needs aligned reads across weight-tile faces or setup repacking; repeatedly reading full tiles for each scalar row can multiply traffic.

Sub-tiling must also preserve each output's original within-tile row position if claiming literal special-value bit identity. Moving an original B/C/D row into scratch row0 changes the LLK's final R+L into L+R. Ordinary finite, nonexceptional results agree, but NaN-payload identity does not follow. Preserve the original row-position class or explicitly constrain and test that claim.

Multiple workers must not pack overlapping pieces of a final BF16 tile. Use a unique tile assembler and bounded scratch/L1 handoff, counting assembly, NoC and barriers in timing. Up/out provide more output rows but retain the same ownership problem. Splitting K into independently reduced partials changes association; exact ordered product production or full lane-state handoff adds communication and is not the first target.

Potential SFPU row broadcast must first match baseline FP32 products for FP32 GR and BF16 GDN weights. Reinitialize reduction after multiply/fold: reduction init owns replay slots/masks, and another SFPU initializer between reduction init and calculation violates its state contract.

Existing [QKV balanced column reducer](models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py#L722) and [outer half-tree](models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py#L778) are framework references, not an accurate-W drop-in.

### Down gather compaction

[GatheredProjection](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L149) gathers physical96-column results, slices80 from each rank and concatenates. Historical row26 costs30.354us on one core.

First preserve the BF16 collective shape and replace only local slice/unaligned-concat assembly with a parallel copy into320 logical columns. Preserve rank order and BF16 bits, with one owner per output tile. A projection writer alone cannot delete post-gather assembly because each rank only computed80 values. Changing wire packing/collective geometry is a larger experiment.

### Local CCLsum preparation/reducer

[Collectives.sum](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L116) performs synchronous BF16 all_gather(dim1), FP32 typecast, generic dim1 sum. [generic_reductions.cpp:639](ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp#L639) clones/fills H padding for T<32. FP32 bypasses fast_reduce_nc; [outer-axis lowering](ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp#L209) transposes HC, reduces H and transposes back.

The selected HC writer [copies logical T lines and skips padding](ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp#L95), then [initializes new rank padding](ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp#L160). Whole tiles are physically read, but the writer skips original token-padding lines, so they never reach live transposed outputs or the reduction. A narrow interleaved FP32 [1,4,T,2560], dim1 exemption can plausibly remove clone/fill without changing arithmetic/CCL. Do not generalize the guard removal without auditing other layouts/routes.

A specialized post-AG reducer can additionally read four BF16 rank planes directly, widen to FP32 DEST, reproduce the selected H tree and apply caller scale/typecast. For ordinary finite values, the rank tree is r0+(r1+(r2+r3)) with padded zero additions. Preserve the full selected sequence for bitwise zero/nonfinite claims. Existing experimental.fast_reduce_nc accepts FP32 but uses an ascending left fold: it is not an exact replacement.

The rank-tree source chain is [H tile-fold/finalize](ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl#L629), [four-row LLK loads](tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h#L104), [face add/transpose](tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h#L245), then the [right-nested replay additions](tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h#L1366). The simplified expression assumes nonexceptional intermediates; retain padded zero operations when testing full bit identity.

For T<=32, local BF16 payload is80*2048=163840bytes/rank, gathered output655360bytes, ideal received payload491520bytes/rank. Logical B1 does not shrink tiled payload. Raw historical attributes show1x4/ring/two links; omitted Python topology/link arguments are inferred by runtime. A local reducer earns no network-byte saving.

Clone/fill removal and custom post-AG reduction overlap completely; never add their savings.

### Smaller follow-ons

- **GR normalization:** fuse selected square/mean/epsilon/rsqrt/two multiplies/BF16 result while preserving order and the2560-feature domain ([decoder_components.py:27](models/autoports/qwen_qwen3_8_flash_next/tt/decoder_components.py#L27)). Native RMSNorm is not automatically exact.
- **Residual write:** adapt the existing [two-round SFPU writer](models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py#L210) to mesh one-stream geometry. Its reader hard-codes320 tiles/four streams; it cannot bind unchanged to a mesh80-tile stream. Retain both BF16 rounds.
- **GDN recurrence layout:** compact_recurrence_io/fuse_recurrence_layout are true, but the [conditional](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L351) selects the same mesh recurrence in both branches. That function still [permutes/repeats/round-trips inputs](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L398) and [untilizes/permutes/retilizes output](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L511). Flags do not prove layout fusion. Direct reader indexing and consumer-layout output are plausible exact transformations; retain the12-head SFPU recurrence arithmetic, state lifetime and rounding. In-place state aliasing requires a separate lifetime audit.

## Nonoverlapping historical latency ceilings

Use role-specific rows per layer kind. For a projection reduction index i, product+reduce is [i-1,i+1); local chain [i-2,i+3) contains input cast/product/reduce/reshape/output cast. The output cast remains logically mandatory even if merged.

| Role | L0 reduction rows | L1 | L3 | Terminal |
|---|---|---|---|---|
| Down | 18,182 | 515,679 | 808,1088 | 1217 |
| Up | 33,197 | 530,694 | 823,1103 | 1232 |
| GDN out | 157 | 654 | none | none |

Rows18/33/157 have global call IDs4096000/4111361/4238336. First-path L0 costs are down36.573us product +71.393us reduce (3cores); up33.587+12.733us (80cores); GDN out95.021+68.527us (20cores). GDN [154,161) totals200.65425us, including two gathers excluded from its local-chain budget.

The following sets are disjoint, weighted L0/L1/L3 by35/1/12 and terminal once:

| Candidate scope | L0 us | L1 us | L3 us | Terminal us | Weighted ms |
|---|---:|---:|---:|---:|---:|
| Down local projection chain | 236.947 | 236.897 | 233.321 | 119.630 | 11.449524 |
| Up local projection chain | 108.781 | 107.645 | 125.044 | 53.140 | 5.468648 |
| GDN out local chain | 175.725 | 184.026 | 0 | 0 | 6.334401 |
| Four slices +compact concat | 69.502 | 70.718 | 72.366 | 35.387 | 3.407067 |
| GR/final post-AG cast/sum/scale/cast | 93.231 | 93.258 | 106.488 | 45.155 | 4.679354 |
| **Primary expanded budget** | | | | | **31.338994** |
| Entire GR/final local normalization | 121.646 | 118.931 | 140.420 | 60.207 | 6.121788 |
| Both residual-write chains | 61.186 | 67.606 | 66.303 | 0 | 3.004752 |
| GDN recurrence pure layout | 42.738 | 43.188 | 0 | 0 | 1.539018 |
| **All listed scopes, zero-cost ceiling** | | | | | **42.004552** |

Reproduce with GR read starts L0 [5,169], L1 [502,666], L3 [795,1075], terminal [1204]. Relative to s: normalization [s,s+10), compact assembly [s+17,s+22), post-AG chain [s+36,s+44). Residual writes: L0 [161,169),[290,298); L1 [658,666),[787,795); L3 [1067,1075),[1196,1204). Pure recurrence layout: L0 [119,125),[126,131),[132,135); L1 [616,622),[623,628),[629,632). These exclude exp, state copy and final typecast.

Nested budgets, **not additional credits**:

- Product+reduce only: down10.367852ms, up4.536640ms, GDN out5.892206ms; total20.796698ms.
- Compact concat alone2.944359ms, inside3.407067ms.
- CCLsum clone+fill1.911939ms, inside4.679354ms; including both transposes gives3.067186ms, also inside it.
- Finer output partition and streaming fusion act on the same projection time.

Every replacement still reads operands, computes and writes outputs, and pays dispatch/assembly overhead. The42.004552ms total is not a predicted current saving and does not prove50ms is impossible. It demonstrates that a50ms promise cannot be supported by adding these named historical rows. Measure source-matched wins and then identify the remaining critical path. Never add old scorer work or eager gaps as extra credits.

## Focused repro and acceptance gates for the parent

No commands below were run. Device work remains serialized under the parent's ownership. Prioritize complete-layer validation over further isolated tuning.

1. **Freeze correspondence:** baseline/candidate model and low-level hashes, actual grid/mesh, weight format/dtype, output geometry and role binding. Reference the selected accurate sum, not an old half-tree or ordinary matmul.
2. **Projection bits before packing:** diagnostic FP32 output; compare product bits, reduced FP32 bits, then BF16 output bits, all ranks, using real down/up/GDN-out operands. BF16 equality alone hides reassociation. Include cancellation/exponent-gap/midpoint/zero cases and explicit nonfinite policy; actual K320/6144/10240 plus32/64/96 controls. If supporting nonaligned K, reproduce padding masks; production K here is aligned.
3. **Partition/shape:** output rows across four-row/face/tile boundaries, especially4/5/8/12/16/17/31/32/33/80/96; Q1/2/3/7/17/31/32/33 and prefill tails. Check physical padding, unique output ownership and fallback signatures.
4. **CCL:** all T1..32, distinguishable/permuted rank operands, cancellation and midpoint cases. Compare FP32 and BF16 bits. Poison token padding with finite garbage/NaN/Inf; require stable live outputs and input immutability. This adjudicates the narrow clone/fill exemption.
5. **Complete L0/L1/L3:** both GR reads/writes, MoE routes, GDN conv/recurrent state, QSA KV/indexer, PLE, next-layer behavior. Claimed exact paths must match selected baseline, including ordered IDs/weights/ties and state. Retain whole-output HF PCC>=0.995; per-row PCC stays diagnostic. PCC alone cannot admit an exact-fusion claim.
6. **Trace/state:** warm signatures, capture, change real inputs/positions/pages, compare eager/replay outputs and state, stable output addresses, request order, resume, fixed/inactive slots and diverse B32 routing. Synchronize/read at correctness boundaries so enqueue completion is not mistaken for device completion. Allocation tracking includes program-cache allocations; watcher is separate from clean timing/profiling.
7. **Measure the intervention boundary:** clean warmed complete-layer Q1/Q32 timing, including assembly/sharding/collectives; fresh reduced0/1/3 eager profile/advice showing intended product/materialization removal. Reconcile35/1/12 layers plus entry/terminal/split sampler with clean all48 token-out. No all48 operation profile, no eager-gap trace credit.
8. **Integrate after gates:** the parent's currently running all48 validation establishes the present baseline, not a future fused kernel. A modified kernel requires fresh full-model accuracy/autoregression/qualitative/resume/B32/context evidence under the existing contract. No hardware pass or stage acceptance is claimed here.

Useful templates: [probe_fused_projection_cast.py](models/autoports/qwen_qwen3_8_flash_next/tests/probe_fused_projection_cast.py), [probe_parallel_ordered_projection.py](models/autoports/qwen_qwen3_8_flash_next/tests/probe_parallel_ordered_projection.py), [probe_multichip_decoder.py](models/autoports/qwen_qwen3_8_flash_next/tests/probe_multichip_decoder.py), [test_sum.py](tests/ttnn/unit_tests/operations/reduce/test_sum.py). These cover only parts of widening, FP32 comparison, trace/state and poisoned-padding contracts. The residual-topology probe explicitly uses an ordinary WO producer, so it is not an exact GDN replacement gate.

## Preserved policy and rejection ledger

Preserve BF16 activation/KV/CCL, FP32 exact GR/GDN projection, selected fidelities including QSA SDPA HiFi2, routed BFP4/LoFi as inherited physical-fit exception, and one whole stream/rank. Keep head20/3/2 and selected synchronous sampler/decoder CCL.

[Inherited rejection ledger](models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/operation_topology_audit.md) records async/persistent/global BFP8 decoder CCL regressions, native-AR numerical differences, hidden-residual RS/AG-MM B32 regressions, rejected BFP4 attention, retained BF16 KV and projection fidelities. Newer sampling_ccl_v1.json also rejects sampler async-barrier gather for latency. These do not reject a new exact local post-AG reducer, whose intervention boundary differs.

## Other potential issues and remaining uncertainty

- **Provenance caveat, not scorer defect:** current causal_indexer_scores.py SHA256 is37ce40237cf425e0b2cea28c86a7a3ab14c02987c2f462a04b304ba2aa92a8bc; selected_sources_v1.json recordsba8d32ae291338883bfc07b70f150145208025addb477f9acd0a131145049d03. Reconcile content before extending older validation to the entire current checkout. GR/GDN focus-source hashes match that manifest.
- fast_reduce_nc and existing QKV/indexer writers have different reduction/rounding contracts; availability is not proof of equivalence.
- Recurrence-layout flags and stale error text (16 rather than4 Q/K heads,48 rather than12 cores) are source discrepancies. Actual constants/validation/runtime assignment agree on4/12/12; messages do not explain the gap or prove a correctness defect.
- Resource fit, faster output scheduling, exact replay, state lifetime and actual critical-path latency remain hardware questions. This audit cannot certify the64.145280ms goal.

## Review and launcher limits

The draft was revisited against Python, C++ factories, dataflow, compute helpers and Blackhole LLK, and disjoint row budgets were recomputed. Split-K exactness, free direct BF16 packing, QKV-tree reuse and a guaranteed50ms saving were rejected. The clone/fill finding is restricted to the selected padding-aware interleaved HC path.

The earlier autodebug_functional_floor_launch.log records a pre-execution missing code-mode host at /home/ttuser/.local/bin/codex-code-mode-host; that attempt could not inspect files or create a new report. This session's command bridge and three source subagents worked. No AutoDebug launcher was invoked again. No launcher timeout, sandbox recovery, device availability or hardware validation outcome was tested here.
