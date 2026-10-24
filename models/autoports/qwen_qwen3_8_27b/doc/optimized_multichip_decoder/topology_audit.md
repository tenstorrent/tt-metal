# Initial topology audit

Source checkpoint ad9ed5d3c46; tt/multichip_decoder.py; stage04 tracy/final_profile_l{0,3}/decode_perf_report.txt. Planned work, not completed experiments.

|Boundary|Current sequence|Candidate/action|Constraints/evidence|
|---|---|---|---|
|Input projections|norm, packed mix|retain packed; compare split|BFP4/LoFi, real weights/recorded activations|
|Attention output|local-input full-output MM,RS,AG|persistent CCL; sharded residual; gathered-input local-output fused AG-MM|BF16 ring TP4; full attention18/14/16us respectively|
|MLP|norm,packed gate/up,slices,SiLU multiply,down|split fused gate/up; working shard grids|gate/up77us,down49us|
|Residual|replicated DRAM decode;sharded prefill|coherent L1 replicated and hidden-sharded distributed norm|carry across next consumer; no immediate compatibility restore as rejection|
|Prefill|distributed norm stats AG,hidden AG,2D MM,RS|geometry/minimal MM;CCL dtype/placement|prior8326/10629us whole-layer envelopes|

Cross residual contract, collective placement, packed/separate, activation/CCL dtype and persistent buffers. Prior stage04 fused AG+RS/RS-only slower paths require exact artifact inspection before reuse. Norm and matmul boundaries have repeated DRAM/L1 conversions. No between-layer wrapper conversion exists today; default decode residual is replicated. Persistent buffers must be unique to live uses and prepared before capture. Maintain internal padding/slicing for nonaligned inputs.

Timeline: projections/attention → row collective → norm/MLP → row collective. Device envelopes include gaps. Host refresh/readback is outside the layer. Serving cache/scheduler absent at this stage; no full-model claims.

## Executed family comparisons

All times below are host-timed warmed traced whole-layer medians at4096+128 B1 on TP4; they are selection evidence, not device-time roofline inputs. Separate whole-layer device profiles exist for the before path and will be collected for the final default. Complete policies, PCC, commands and logs are in the named sweep artifacts.

|Family/action|Residual at layer boundaries|Movement/precision|Linear/full decode us|Evidence/decision|
|---|---|---|---|---|
|RS then AG, persistent outputs|Replicated DRAM|BF16; two row reductions each followed by gather|483.25 /402.57|topology_persist; improves allocator overhead, not best family|
|Same with sharded L1 residual/adds|Replicated L1|BF16; retains CCL intermediates L1|480.69 /397.13|topology_l1_residual; valid coherent producer/consumer layouts|
|RS and distributed norm|Hidden1280/chip, DRAM|BF16, gathers hidden only at column projection|588.84 /507.11|topology_sharded; no replicated residual restore|
|RS and distributed norm, L1|Hidden1280/chip, L1|BF16; carries sharded residual to next norm|541.20 /461.31|topology_sharded_l1; improved but slower|
|Fused AG-MM|Hidden1280/chip|1D tuned fused input gather/projection|603.53 /518.40|topology_fused_ag|
|Fused MM-RS|Hidden1280/chip|Output projection/down fused with RS|687.25 /599.62|topology_fused_rs|
|Fused AG-MM and MM-RS|Hidden1280/chip|Both boundaries fused; no compatibility restore|680.29 /598.86|topology_fused_both|
|Repartition output/down weights, AG-MM|Hidden1280/chip|Gather localK, compute only localN;2D→1D adaptation plusBF8 payload|539.14 /460.20|adapted_output_ag_1d_bf8; slower despite adaptation|
|Direct async AR|Replicated L1|BF16; eliminates separate RS/AG decomposition|472.59 /390.12|adapted_ar_l1|
|Direct async AR, BF8 payload|Replicated L1|BF8 communication, BF16 output/add/norm|463.50 /381.21|adapted_ar_l1_bf8; strongest initial family|
|Shared two-slot AR scratch|Replicated L1|Same BF8 payload;1.39MB/device across owners|463.75 /380.77|pool_pooled; constant storage, same latency/PCC|

For each row projection the logical B1 output is5120 elements/chip, physically160 tiles:327680B BF16 or174080B BFP8 including headers. Direct AR staging is canonical four-rank capacity; the pair is1,392,640B/device atBF8. A hidden-sharded residual holds1280 channels/chip but requires distributed norm statistics plus hidden input gather for column projections. The alternate output-AG family gathers6144 or17408 global input channels instead of reducing5120 output channels; actual weight packing and matmul/layout costs are measured, not inferred from tensor byte counts alone. CCL uses mesh axis1, ring, two physical links.

Repeated same-input projection trials: packed mix versus real separate weight groups; packed gate/up versus two matmuls with fused gate SiLU; fused minimal SwiGLU with two legal K/output configurations. All splitting, activation and movement are inside the measured layer. Unsupported/catastrophic failures are not used as blanket rejection evidence. Remaining tuning experiments are recorded separately while the stage remains in progress.

## Cumulative follow-up evidence

The search is precision-locked to BFP4/LoFi for every dominant projection while testing geometry. `final_geometry` and `geometry_confirm` cross8–80 input-storage cores and non-power-of-two K blocks; mix8/K20, out8/K6 and down8/K17 improve the combined layer. Gate/up80/K2 remains the fastest tested family after wider-shard K4/8/16/20 alternatives. Down K68 exceeds physical L1 and K136 overlaps live storage; K34 is legal but slower. These are per-role outcomes, not a global core-count default.

Prefill is separately optimized:4096 chunks with K4 across projections and full-attention output subwidth5 win.4096/K8 initially exceeds L1; setting explicit output block height1 makes K8 and the larger-K family legal, but slower.2048 and512 chunks,8x8/8x4 grids, minimal fused SwiGLU, and larger K alternatives have measured rows. Nonaligned inputs remain internally padded and sliced; final acceptance must exercise4095/4096/4097.

SDPA512's initial NaN correlations were diagnosed as undefined page IDs, not accepted as a numerical-policy rejection. Explicit owned padding passes but loses to256; a static mapped-capacity guard prevents invalid reads for external tables.1024 remains above physical L1 after FP16/padded adaptation. HiFi2/FP16 passes native context; final LoFi/FP16 versus HiFi2 comparison is recorded in sdpa_final and sdpa_final_native.

Native DRAM-sharded weights in the fused CCL factories are a separate experiment from valid interleaved fused weights. Both AG K10/K20 hang and RS K1/K2 yield NaN correlations. Triage, bounded recovery and AutoDebug source diagnosis are retained; same-program interleaved controls all pass, isolating native reader defects. Native RS is repaired by matching out_block_w20 to its per_core_N20 descriptor; this passes and competes at B32 (AUTOFIX_fused.md). The compatible sharded-residual family is evaluated without restoring the old replicated boundary.

## Batch32 coherent-family closure

The B1 winner does not imply the same residual contract at B32. `compact_rows32`, `row_family_final32`, `compact_boundaries32`, `compact_movement32`, `fused_compact32` and `column_family32` compare complete B32 layers at4096+128. Row reduction consumes compact `[1,B,5120]` partial sums and expands only the1280-channel residual. Column gathers retain compact `[1,B,5120]` outputs for matmul; reshape precedes the payload memory move. Norm statistics and column gathers are transient L1 allocations, prepared by warmup before capture. No sharded candidate restores the replicated residual between layers.

|Family|Linear/full traced host us|Outcome/evidence|
|---|---:|---|
|Replicated residual control|3119.57 /2058.27|compact_rows32|
|Replicated, compact row reduction|2884.57 /1820.79|compact_rows32|
|Sharded, compact rows, DRAM RS output|2730.43 /1662.03|compact_rows32; transient L1 AG|
|Sharded, compact row and column boundaries|2557.21 /1487.14|compact_movement32 control|
|Same, compact before AG memory move|2547.18 /1478.68|compact_movement32 compact_move|
|Same, BFP8 CCL|2558.86 /1485.49|compact_move_bf8; passes but loses|
|Native fused RS, L1 output, transient gathers|2540.81 /1474.66|compact_movement32 native_l1_output|
|Native fused RS, DRAM output, L1 residual|2562.50 /1490.27|compact_movement32 native_dram_output; passes but loses|

Native fused RS precision alternatives (HiFi2, BFP8 activation, BFP8 output, both) pass but lose in fused_rs_precision32. Compact-before-input-move native RS (`fused_compact32`) preserves2541.18/1474.21us. Its two per-layer L1 outputs need27.5MiB/device across64 layers under bank-granular allocation;28MiB reservation passes,32MiB fails. This is counted separately from shared AR scratch and DRAM weights/cache. The final column_family32 test uses interleaved column weights with native DRAM-sharded row weights, so fused AG is compared without discarding the repaired RS family.

## Per-user convolution launch removal

Beyond projections/CCLs, B32 linear decode contained32 same-kernel convolution launches on separately padded user rows. The API's B1 restriction is not a semantic blocker: temporal four-row windows preserve each user's history. `temporal_conv_review.md` rederives kernel row indexing; `packed_conv8/16/32` measures the complete layer including row-major packing/unpacking. B32 drops2541.08→2002.41us, B16 1488.59→1253.32us, B8 879.10→779.36us with unchanged PCC. The selected default owns padding for non-multiple-of-eight batches too. Final runtime audit should see one convolution call per ASIC atB32, with exact intermediate/state proof and final stack/watcher/mixed-batch acceptance.

## Final-profile prefill collective audit

The final prefill profile exposes a previously unclosed material family: two hidden AGs plus two RSs account for2263us on device0 (31% of the7251us whole-layer envelope). Earlier CCL flags/fusion candidates were decode-gated and cannot reject this prefill family. The initial L1 matmul-input advice trial still gathered through DRAM; it exposed gate/up L1 capacity (buffer starts910976 versus CB end1105536), and H1/K2/2048-chunk adaptations pass but lose. Per-role placement is recorded separately.

New coherent prefill adapter `tests/run_prefill_ccl_candidate.py` preserves hidden-sharded residuals and unchanged decode. It directly places AG output in L1 for its consuming matmul, retains L1 RS results through residual/norm, compares BF16/BFP8 communication/activation, transient/persistent buffers and worker counts. L1 residual trials return L1 rather than restoring DRAM at the measured boundary. Fused AG/MM uses rank4 token-M, interleaved BFP4 column weights; fused MM/RS starts with native weights, output width20, grid8x8 and CCL offset0,8, bounded output blocks. Reviewer source inspection confirms M4096 is legal in principle; each actual candidate still requires device evidence and any failure requires adaptation. Completed family results, repairs and paired selections are in prefill_ccl_findings.md, prefill_selection.json and prefill_ccl_candidates.csv. Production selects BFP8 column activations/CCL and native prefill MM/RS; the repaired coherent L1 and AG-fusion alternatives are rejected by passing whole-layer comparisons.

## Selected closure

The completed prefill families select8workers/BFP8 hidden CCL and column inputs, full-attention mix directL1 gathering, and preallocated nativeRS for B1 workingrows4096. Current native prefill row geometry is outK48/downK17,H4,sub2x3,11x8/N15; see prefill_rs_geometry.md for isolated larger-K adaptations. The repeated same-input/packed, residual-placement, collective-placement, activation/fidelity and persistence families above remain the selection evidence; final acceptance and runtime rows are reported in validation_summary.json, runtime_contract.json and batched_runtime_contract.json. Historical phase descriptions above describe their measured configurations, not the final default.

V4 precision-matched column audit changes only the MLP prefill family: separate gate/up share the existing BF8 AG; gate fuses SiLU, product is BF16, native downRS and hidden-sharded residual continue unchanged. Compared packed versus separate at actual BF8 input, largerK/grid/subblocks, direct sharedAG-to-L1 consumed by both projections, and fusedAG+gate/SiLU returning the same gathered tensor to up. All families have passing adapted comparisons; native separate wins paired means6002.47/6271.89us. Native multi-N reader failures were repaired through exact grid-padded interleaved weights before larger-K rejection. No extra inter-layer movement. See prefill_projection_geometry.md, prefill_projection_selection.json, AUTODEBUG_column.md and AUTOFIX_column.md.

## V5 final-profile movement and grid closure

|Observed topology/cost|Candidate and constraints|Action and evidence|
|---|---|---|
|Native prefillRS8x8, full-width readerN20|10x8/N16 and11x8/N15 with outputW=perN, same precision/buffers/CCL placement; role-K/H/subblock adaptations|Select11x8/N15/H4/sub2x3,K48/17. Forward/reverse whole-layer comparisons and capacity evidence in prefill_rs_grid_selection.json and prefill_rs_geometry.md.|
|Native decodeRS8x1,N20|10x1/N16 and11x1/N15; role-isolated outK24/48,downK34; larger endpoints bounded by native CB capacity|Select10x1/N16/sub1x4,outK6linear/K24full,down17. Full comparison includes selected packedRoPE, preserving best coherent family. decode_rs_grid_selection.json and decode_rs_geometry.md.|
|FullB32 decode64 per-user RoPE operations with202 slices|Fold logical users into independent sequence rows, remove physical tile padding before packing, share transient cos/sin acrossQ/K|SelectB>=4. First L1 output API rejection repaired to original DRAM Q/K consumer boundary. Distinct/permuted position exact proofs and B2/4/8/16/32 comparisons in rope_optimization.md; B2 retains original loop.|
|FullB32 prefill64 per-user RoPE operations and concatenation|Tile-aligned user/sequence folding, zero-copy reshape plus DRAM permutations; same final DRAM boundary|SelectB>=8 after B32 forward/reverse and B8/B16 comparisons (B16 tie). Uniform-FP16 exact packing proof separates index identity from source core-group precision assignment; production original policy remains and whole-layer PCC passes. No persistent table cache.|

These changes affect internal working tensors only. DecoderStack still passes residual directly to the next layer with one shared owner, without an inter-layer gather/reshard/all-reduce. Native grids and packed RoPE add no persistent weights, KV/cache or CCL buffers. Final default verification is source-hashed separately from all candidate receipts.
