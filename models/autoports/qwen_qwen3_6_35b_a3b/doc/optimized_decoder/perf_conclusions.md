# Optimized decoder performance conclusions

This report describes the delivered decoder source `99e2a2fa1d4cb78f4dd0eb95079cc8288e8f7e0ebd92e6ffbc44ffb3be05cb00`. `hard_gates.json` passes its current-source checks. These are individual decoder layers, not full-model throughput or generation quality measurements. Independent stage review is recorded separately.

## Clean before/after and profiler accounting

Clean S256/context512 measurements use ten warmed prefills and sixty blocking traced decodes, with Watcher, profiler and allocation tracking disabled. Whole-output PCC remains above .995.

| Layer | Prefill fused → selected, ms | Decode fused → selected, ms | Prefill gain | Decode gain | Prefill PCC fused → selected | Decode PCC fused → selected |
|---|---:|---:|---:|---:|---:|---:|
| 0, linear | 478.105574 → 468.971433 | 1.447121 → 1.346515 | 1.91% | 6.95% | .999993613 → .999975917 | .999996072 → .999934383 |
| 3, full | 8.975175 → 5.876420 | .583035 → .431675 | 34.53% | 25.96% | .999890040 → .999779540 | .999923261 → .999902656 |

Evidence: `baseline_{0,3}.json`, `final_{0,3}.json`. `latency_comparison.csv` includes matched lengths127/128/129/256/257/4095/4096/4097; nonaligned lengths use internal padding/chunking without public divisibility restrictions. Final timing supersedes candidate screening numbers. The selected decode policy screened at .4309295ms; final .431675ms differs by .17%, within the recorded sample variation, and remains below the best different residual layout (.4320665ms).

The following is **same-run** accounting from the final Tracy captures. Decode contains sixteen complete replay sessions; kernel sums divide by sixteen. The host samples surround blocking trace execution and exclude explicit profiler drains. Kernel sums exclude dispatch gaps and cannot simply be substituted for clean latency.

| Layer | Prefill device ops / kernel sum ms | Decode device ops / kernel sum ms | Profiled host decode ms | Same-run host minus kernel ms | Separate clean host ms |
|---|---:|---:|---:|---:|---:|
| 0 | 11920 / 454.499884 | 165 / 1.205704 | 1.537843 | .332140 | 1.346515 |
| 3 | 104 / 5.451319 | 72 / .391295 | .614596 | .223301 | .431675 |

Sources: `perf_summary.json`, `tracy/layer_{0,3}/ops.csv`, `correctness.json`, `{prefill,decode}_op_groups.csv`, `{prefill,decode}_perf_report.csv` and corresponding human-readable reports. The profiled host difference includes instrumentation and execution overhead; it is not a clean runtime regression.

Clean changing-input128/128 runs average1.346337ms (linear) and .428768ms (full). Optimistic logical-byte bandwidth floors at512GB/s are .203109ms and .120238ms averaged over the actual positions, giving15.09% and28.04% of theoretical. Both exceed the10% floor. The full S256 decode floor is instead .120496ms because its KV read length differs. Logical byte estimates are not measured DRAM traffic or a compute roofline.

## Actual internal prefill shapes

The S256 profile executes four outer64-token chunks for linear attention and one256-token chunk for full attention. The table is extracted from actual rows between `PERF_PREFILL` and `PERF_PREFILL_END`, not inferred from the public chunk hint. Kernel times are sums over the full S256 pass; mean divides by call count. Grouped MoE uses routed/padded row counts, so its M must not be interpreted as the public sequence length.

| Layer / outer chunk | Actual kernel role | Physical M | K → N | Calls | Sum / mean, µs |
|---|---|---:|---|---:|---:|
| Linear / 64 | Output projection, HiFi4 | 64 | 4096 → 2048 | 4 | 321.783 / 80.446 |
| Linear / 64 | Router, HiFi4 | 64 | 2048 → 256 | 4 | 178.129 / 44.532 |
| Linear / 64 | Shared packed, HiFi2 | 64 | 2048 → 1056 | 4 | 79.935 / 19.984 |
| Linear / 64 | Shared down, HiFi2 | 64 | 512 → 2048 | 4 | 42.721 / 10.680 |
| Linear / 64 | Grouped expert gate/up, LoFi | 2528 | 2048 → 65536 | 16 | 6426.840 / 401.678 |
| Linear / 64 | Grouped expert down, LoFi | 2528 | 512 → 131072 | 16 | 3546.746 / 221.672 |
| Full / 256 | Packed QKVG, HiFi2 | 256 | 2048 → 9216 | 1 | 128.079 / 128.079 |
| Full / 256 | Output projection, HiFi2 | 256 | 4096 → 2048 | 1 | 57.056 / 57.056 |
| Full / 256 | Router, HiFi4 | 256 | 2048 → 256 | 1 | 61.032 / 61.032 |
| Full / 256 | Shared packed, HiFi2 | 256 | 2048 → 1056 | 1 | 20.067 / 20.067 |
| Full / 256 | Shared down, HiFi2 | 256 | 512 → 2048 | 1 | 13.811 / 13.811 |
| Full / 256 | Grouped expert gate/up, LoFi | 4064 | 2048 → 65536 | 4 | 2144.110 / 536.028 |
| Full / 256 | Grouped expert down, LoFi | 4064 | 512 → 135168 | 4 | 1304.682 / 326.171 |

Linear prefill is dominated by inherited FP32/HiFi4 recurrence: two leading GenericOp groups alone total284.942ms. Full prefill grouped experts total3.449ms of5.451ms device time. Full decode expert gate/up and down cost64.614µs and33.527µs; packed attention44.636µs, output26.972µs, router24.699µs and SDPA22.505µs. These explain why precision/layout changes improve full attention much more than linear prefill.

## Advice and measured decisions

The advice strings below occur in the final reports; timings cited for decisions are clean screening rows from the linked searches, not profiler latency.

| Advice / opportunity | Action and evidence |
|---|---|
| Place input0/intermediates in L1 | Expert L1 .510937ms versus DRAM .524893ms at BFP8/HiFi2 (`sparse_search/search.json`). Retained L1 and width-sharded residual across norms/adds. Explicit MoE-output reshard .430930ms beats implicit boundary .432067ms (`residual_search/search.json`). |
| Increase small output subblocks | Sparse16/32-core subblock2 .506132ms and8/16-core subblock4 .533254ms lose to32/64-core subblock1 with block16 .499339ms at HiFi2. LoFi geometry also measured (`sparse_search{,2}/search.json`). Keep measured faster1x1; larger dense-prefill subblocks remain a stop-rule handoff. |
| Increase small K block | Sparse block16/32/64 compared; LoFi16 .497876ms selected. DRAM dense block4 selected against8/16/32; block64 produced a concrete L1 allocation limit (2,774,976 requested versus1,572,864 available). First projection API failure was repaired and the family rerun. |
| Use DRAM-sharded program config | Applied to shared/full decode. Shared DRAM .435223ms versus .453343ms without it. Reader controls1/2/3 at block8 measured .510528/.441510/.441416ms; selected three readers and block4 (`projection_search_retry/search.json`). |
| No explicit prefill program config | Auto versus1D/2D/minimal variants measured. Auto5.813ms versus2D/block8 screen5.600ms; final selected5.876ms supersedes screen. Remaining report advice on inherited operations does not imply all model matmuls use auto (`prefill_search/search.json`). |
| Try HiFi2 or retain HiFi4 for accuracy | Attention BFP8/HiFi2 .453343ms versus BF16/HiFi4 .501300ms; LoFi .453368ms gave no gain. Shared HiFi2 .510707ms versus LoFi .510616ms did not justify reducing fidelity. Expert LoFi and geometry measured separately; final real-weight/natural-input gates pass. Sensitive recurrence/router exceptions remain unchanged. |
| Packed versus separate same-input projections | Packed full .435223ms versus legal separate .472955ms; shared packed-GU-plus-scalar .454545ms and fully split .461058ms lose. Fresh expert split controls at16/8cores and block16/32 all pass; best split .458517ms loses to packed control .430870ms and final .431675ms (`expert_split_search/search.json`). |
| BFP4/LoFi candidate | Measured real-weight sparse candidate .468039ms versus BFP8 .498592ms at matched block32 in the sparse search. Candidate profiler verifies actual BFP4/LoFi rows; grouped prefill remains BFP8 due API support. Stage8 accuracy/precision probe decides adoption; production stays BFP8. |

## Remaining runtime movement

There is no runtime torch/from_torch/to_torch fallback in the measured decoder. This is **not** a claim that the trace contains no layout operations. Full decode contains five reshards totaling2.490µs: one changes residual width-shard geometry for DRAM-sharded projection, four bridge head-creation height shards through Q/K normalization width shards back to attention height shards. These are consumer contracts of the selected kernels. The single `TilizeWithValPadding` costs1.934µs: the fused MoE gate emits row-major scores, its1×8 score slice is tiled for weighted expert combination. IDs remain row-major for indexed sparse execution. DRAM/L1 transfers also bridge norm, attention and MoE contracts; carrying the residual layout and omitting the final DRAM return were measured, while the faster explicit MoE reshard was retained.

Linear prefill retains substantial tilize/untilize traffic inside inherited ordered FP32 recurrence and routed prefill processing. Those operations are visible in `tracy/layer_0/prefill_op_groups.csv`, including512 large untilize calls totaling9.767ms. Removing them would require replacement kernels/layout contracts, not merely deleting conversions. Their necessity is relative to the retained ordered implementation; no claim of globally minimal movement is made. New recurrence/composite fusion and alternate phase-specific layouts are untried stop-rule handoffs, not measured failures.

## Stop-rule handoff

The last two kept residual changes gained .716% and .263%, triggering optimize stop rule2. Further per-role dense split tuning, dense subblock sweeps, exhaustive physical-M64/128/256 kernel families, larger internal prefill chunks, grouped-expert repartitioning, alternative selector/softmax fusion, recurrence/layout replacements, row-parallel output decomposition, and reduced KV dtype remain untried; expected gains are unknown. KV dtype/layout remain BF16 and unchanged, with32K smoke and the existing262144 context contract. No maximum-capacity reduction is claimed. CCL, LM head, sampling and full-model/serving work are outside this stage. Required correctness and independent review are not waived by the stop rule.

Boundary tradeoff: full-attention S127 warmed prefill increases from41.864672
to43.940496ms (+4.96%); all three optimized samples exceed the baseline
samples. This is a measured regression, not dismissed as timing noise. The
retained configuration wins the headline S256 prefill and traced-decode
comparison. The inherited planner executes S127 as one96-token call plus31
singleton calls; S129 as128+1; S4095 as15×256+224+31×1. The latency CSV
records actual call counts/maxima, including these tails. Further tail-specific
config tuning is a stop-rule handoff.
