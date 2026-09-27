# Final device profiling and optimization evidence

The exact target is TP4 on a 1×4 mesh, 4096 input tokens, 128 advancing traced decode steps, batch 1 and one request. Each device window spans the first firmware start through the last firmware end of the whole layer, including all operations and gaps; the mesh result uses the maximum rank span. Host input refresh and comparison outside replay are excluded. Different layer kinds are never averaged.

| Layer | Baseline prefill device µs | Final prefill device µs | Baseline decode device µs | Final decode device µs | Prefill useful FLOPs % | Decode estimated DRAM % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding_attention | 325830.533 | 329194.676 | 705.420 | 710.513 | 0.101000 | 9.293628 |
| full_attention | 309683.707 | 339373.403 | 766.584 | 744.896 | 0.134930 | 8.969920 |

Baseline device windows come from Stage04, using the same whole-layer accounting. These instrumented measurements do not establish a prefill speedup. Prefill device windows include dispatch gaps under profiling and are much longer than ordinary warmed host medians; neither host time nor summed matmul time is substituted into a device roofline denominator. Decode device measurements also differ from ordinary host medians because profiling is a separate instrumented execution.

Useful prefill work counts eight selected experts and logical tokens, excluding padding and inactive expert-union work from the numerator only. Decode DRAM is estimated from native operand metadata across all four ASICs, including BFP exponents and chunk-rounded local KV reads. It is not a controller measurement. Collective operands reside in L1; their device time remains included while their L1 traffic is excluded from DRAM bytes. Internal repeated reads, NoC and scratch passes are not modeled. The theoretical reference is four × 120 Tensix × 4096 FLOP/cycle × 1.35 GHz for LoFi, and four × 512 GB/s DRAM. Actual kernels mix fidelity. No percentages are clamped.

## Advice disposition

| Cost or advice | Action and evidence |
| --- | --- |
| Repeated Q/K/V and gate/up inputs | Packed projections retained; separate controls measured in candidate comparisons and final-policy matrix. |
| Collective latency and placement | Async RS/AG with persistent L1 outputs; worker/chunk/buffer controls, full-grid semaphore repair, Watcher endpoint repair. See candidate comparisons and AUTODEBUG_ccl_semaphore_grid.md. |
| Fused gather-matmul and matmul-reduce-scatter | Adapted local-width704 residual, distributed norms and carry-forward layout; fixed K22 readiness contract. Six repaired_sharded_* controls all pass and lose to defaults. Harness gather stays outside timing. |
| Lower-movement L1 residual | Repaired mixed-precision add through supported SFPU path; compatible whole-layer family remains slower. AUTOFIX_l1_residual.md. |
| DRAM-sharded decode projections | Fixed physical-device worker-placement query; tested real shapes, padding and reader counts1/2/3, including final BF8 sliding policy. AUTOFIX_dram_mesh.md. |
| Sparse gate/down subblock1×1 | N2 and K-block controls are slower. The actual active-expert8 path stays indexed. |
| Precision/fidelity | Per-group BF4/BF8, LoFi/HiFi2/HiFi4, activation and collective controls; actual adjacent-stack accuracy selects sliding BF8 gate/up and BF16 attention CCL. Full WO BF4 fails B32. |
| Norm, tilize, RoPE and layout costs | Carried-shard and L1 residual families measured coherently. Full sharded-RoPE512 adapted path passes but is slower; sliding retains its selected sharded RoPE. |
| MinimalMatmul missing config advice | Metadata limitation: source supplies explicit minimal program configs. Native policy rows identify selected geometry. |
| Router BFP8 fidelity advice | Inapplicable to actual BF16 router weights; inspect native operand dtype. |
| Prefill sparse nnz advice | Dynamic32-token expert unions do not have fixed nnz8. Decode explicitly executes the selected8 experts. |
| Prefill producer L1 placement | All four real-shape roles tested for both kinds, then15-sample repeats and a combined30-sample full-layer control. Initial tiny gains reverse; combined change is0.061% with overlapping samples. Retain DRAM; see prefill_producer_summary.json. |

Native geometry and dtype details are preserved in each profile directory’s native_decode_policy_rows.json. Human-readable tt-perf-report advice tables, CSVs, command records, whole-layer windows, capture integrity and hashes accompany both profiles. Compact compressed copies preserve ignored table/CSV/log artifacts for the local commit.

## Final producer-placement decision

Retain DRAM producers. Initial0.4–0.5% apparent gains do not repeat consistently: second full control78540.505us vs QKV78543.679, output78558.617, router78586.753. Sliding router92717.347 vs control92690.445 in first repeat reverses the initial gain; second repeat changes ordering again. Combined full placement78487.685 vs78535.423us control is only0.061%, within overlapping30-sample distributions, while decode702.372 vs701.530us. No reproducible material whole-layer gain is established. Shared-input L1 loses2.3–2.4%. All24 cases pass identical per-kind output PCC; source intercepts verify actual L1 producer placement. No API failure was used as rejection.
