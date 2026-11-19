# Selected d810 graph-pattern reconciliation

Selected source: `v6_transport_conv_norm_decay_transpose_interleaved_tail_integration.py.snapshot`, also adopted byte-for-byte as `tt/fused_decoder.py`. This supersedes topology descriptions in the ab28 audit, while retaining all historical measurements under their original sources. The owner stopped tuning at the stage time/token cap. Final capture reconciliation is appended below when available.

The composed source audit restores the preceding reviewed graphs through exact inverses: the transpose-reader change preserves the corrected increasing-K arithmetic, and the weighted-tail reader changes transport while preserving native sum, BF16 rounding and compact SiLU. Its 47 reachable runtime helpers and 77 negative controls pass. Public planning covers 80,784 cases and 168 dispatch cases. See `v6_interleaved_tail_d810_source_audit.{json,md}` and the current root `fallback_audit.json`.

## All 29 skill patterns

| Pattern | Selected disposition and evidence |
|---|---|
| Activation recognition | Dedicated/native activations remain where equivalent. Linear convolution uses the exact fused convolution/SiLU/carry adapter; gated output uses the exhaustive compact SiLU lookup. `v6_decode_conv_complete_all_device.json`, `v6_silu_compact_lookup_device.json`. |
| Softmax | Native stable softmax after exact linear singleton selection; full attention retains its generalized gate. No spelled-out exp/div graph remains. |
| RMSNorm | Full attention retains native RMSNorm. Linear input/postnorm require ordered arithmetic; native substitutions and weight folding fail recorded rounding/endurance checks. Exact terminal scan and weighted-tail fusion are retained. `v6_norm_terminal_scan_device.json`, `v5_inputnorm_weight_fold.decision.json`. |
| Distributed RMSNorm | Not applicable to this single-chip decoder; no distributed statistics. |
| SDPA | Full attention retains native paged decode/chunked prefill SDPA. Native recurrent/GDN alternatives for linear attention fail numerical or complete-adapter performance checks; they cannot replace its corrected recurrence. |
| Split QKV and heads | Full packed projections/native head creation retained. Linear ordered QZ/KVAB producers preserve separate accumulation contracts. |
| Decode QKV heads | Full native helper retained. Adapted linear native-head alternative was slower: `linear_native_heads_0.json`. |
| Decode concat heads | Full native helper retained. Linear compact SiLU writes the required packed WIDTH layout directly. |
| Prefill concat heads | Full native helper retained; linear corrected output retains its necessary head permutation and exact normalization. |
| RoPE | Full attention applies native RoPE to 64 rotary dimensions and preserves 192 unrotated dimensions. Linear attention has no RoPE. |
| TopK | Full generalized gate retained. Linear singleton selection uses a device implementation preserving CPU tie membership; prefill retains native TopK. Native singleton alternatives fail actual layer34 streams. |
| RepVGG conv-sum | Not applicable: no parallel spatial convolution branches. The temporal four-tap convolution is separately fused. |
| Shared-LHS matmuls | Expert/shared packing retained. Ordered linear projections preserve required accumulation; native QZ/output ablations fail actual streams (`v6_native_qz_corrected_state_layer34_seed154.json`, `v6_native_out_corrected_state_layer34_seed154.json`). |
| Spatial mean reshape | Not applicable: no spatial mean. Last-axis normalization reductions are assessed separately. |
| Permute/reshape cancellation | Removed selector-index tilize/untilize and four postnorm placement transfers with exact complete-adapter evidence (`v6_transport_cleanup_device.json`). Remaining recurrence layout changes encode broadcasting/reduction axes; attempted direct transport fusion loses latency. |
| Conv+bias | Not applicable: temporal convolution has no bias. |
| Conv+scale | No separate adjacent output scale to fold; learned tap factors already participate in convolution. |
| Conv+activation | Exact model-local convolution/SiLU/carry fusion retained after repairing native signed-zero/tiny-value behavior. Native fused activation was not numerically interchangeable. |
| Matmul+activation | Packed gate/up followed by binary SiLU wins over split fused matmuls (`matmul_activation_0.json`, `shared_native_swiglu_config_0.json`). Ordered attention outputs retain required BF16 boundaries. |
| Input activation+binary | Expert/shared gating and compatible scalar/square/reciprocal chains retain their merged binary forms. No unsupported reassociation of ordered normalization. |
| Matmul+bias | No learned attention bias. The earlier synthetic dt-bias matmul is replaced by exact tables. |
| Transpose+matmul | **Retained:** corrected prefill matmul reader directly consumes transposed operands, removing three producer transposes while preserving arithmetic. All actual sites, broadcasts, signed-zero/tiny cases and complete zero/nonzero-prefix adapters pass. `v6_corrected_matmul_transpose_matched_timing_device.json`, `v6_chunk64_transpose_output_{zero,nonzero}_device.json`; matched whole-decoder queue confirms prefill benefit. |
| Slice after matmul | Packed logical columns are all consumed. Slices separate consumers; physical tile padding is not discarded logical work. |
| BatchNorm+conv | Not applicable: no BatchNorm. |
| Pad+pool/conv | No pooling. Temporal history is persistent state. Internal chunk64 padding handles logical tails; no public alignment restriction is introduced. |
| Stable softmax | Native `numeric_stable=True`; no separate max/subtraction chain. |
| Reduction+reshape | Native sums retain dimensions. Exact terminal normalization scans eliminate terminal crops/layout adapters while preserving accumulation order. |
| Scaled sum→mean | Native mean changes sensitive reduction ordering. Exact same-order custom terminal scans are retained instead; arbitrary reassociation is rejected. |
| Decode RoPE layout/token index | Full native decode `token_index=0` preserves the established head/token layout without an extra permutation. |

## Additional transport fusions and rejected follow-up

Direct decay lookup eliminates transpose/index/gather adapters and preserves all 65,536 BF16 codes across 32 heads (`v6_decay_direct_lookup_rawcode_all_device.json`). The d810 weighted-tail reader removes two input reshards with identical native reduction and BF16 boundaries (`v6_weighted_norm_interleaved_device.json`). It replaces 20KiB transient shards with 20KiB CB storage and adds no persistent tensors; current linear and full maximum-context gates pass without capacity reduction.

The six-dispatch recurrence adapter and blocked-transport adaptation are both bit-exact across 48 cases, changing-input updates and traced/restarted trajectories. Their traced adapter latency is respectively 5.30× and 5.59× native, so neither is integrated. Fewer dispatches or 16× fewer state requests do not qualify as a win. `v6_recurrent_fusion_rejection_summary.{json,md}` preserves the compile repair, both comparisons and unresolved bottleneck hypotheses. Remaining native recurrence layout changes therefore have a tested, rejected replacement rather than an assumed removal benefit.

Further selector/softmax fusion, direct native-reduction producers and alternate recurrence parallelization are hand-off ideas, not measured wins. Preserve exact tie membership, pack points, signed-zero behavior and repeated-state accuracy in any follow-up. The owner explicitly replaced the earlier eight-hour/70%-roofline condition with the stage wall/time/token cap. These ideas are handed off without further tuning; no roofline or eight-hour claim is made. Final selected-source evidence is indexed by ../final_manifest.json.

## Final selected-source profile reconciliation

The single final capture per layer passes strict completeness, source identity and correctness. Full-kind validation initially rejected four native intermediate placement warnings; the capture-specific reviewed validator then proved44final score/index outputs have the requested DRAM layout/dtype/shape, with three rejecting metadata mutations. The initial failure and fresh raw logs remain intact. No hardware rerun or blanket warning allowance was used.

| Kind | Phase | Ops/pass | Device kernel us/pass |
|---|---|---:|---:|
| linear | prefill | 11920 | 465412.496000 |
| linear | decode | 161 | 1306.808687 |
| full | prefill | 104 | 8732.648000 |
| full | decode | 68 | 549.994937 |

Compared with the historical ab28 warmed S256 profile (11988/207 linear ops), selected d810 has11920/161ops. Linear prefill kernel sum is465412.496us (ab28:465714.035us); decode1306.808688us (ab28:1413.651us). These are operation-duration sums, not clean latency or independent additive speedups. Full attention retains104/68ops; its small timing variation is not attributed to a topology change. The clean matched benchmark, not kernel counts, selects d810.

Linear decode contains **no tilize or untilize operations**. The remaining placement operations are the exact selector input shard atstep134 and score/index outputs at136/141. They bind the device selector to its single-core scalar nth-element contract and feed native softmax/sparse-matmul consumers; the redundant index tilize/untilize roundtrip is absent. Remaining permutations/transposes at66/67/69/74/77/83/86/87 arrange key/head/recurrent reduction axes. Both adapted complete six-dispatch recurrence replacements preserve outputs but lose5.30x/5.59x traced latency; retain the faster native path. Projection/head/expert slices split distinct consumed fields, rather than discard unused logical columns.

Linear prefill retains layout conversions in the ordered projection/normalization and grouped routed-combine adapters. Its dominant custom operation class is299298.588us across44ops, followed by binary arithmetic53996.181us and ordered accumulation44300.683us. These preserve the proved arithmetic boundaries; native projection/norm replacement and folded-pack variants fail actual target-stream or exact-boundary checks recorded in the pattern inventory. Direct coefficient reading, terminal scans, transpose-address corrected GEMM, and direct weighted-tail reading have removed their redundant adapters. Grouped-down outputs still convert for native routed-combine; adapted alternative packing/pruning was measured and rejected. Remaining native head/RoPE/paged-cache adapters in full attention satisfy their consumers; its decode operation sequence is unchanged.

No measured forward uses torch/from_torch/to_torch or a host functional fallback: the runtime prohibition is enforced by the unchanged device-only harness and47-helper source audit/77negative mutations. Device profile sequences and dtype/layout operands are retained in each capture directory and root tracy/layer_0,3. Future combined selector/softmax or custom reduction producers could absorb additional consumer boundaries but are unimplemented handoffs under the owner wall cap, not unexplained inverse-only movement.
