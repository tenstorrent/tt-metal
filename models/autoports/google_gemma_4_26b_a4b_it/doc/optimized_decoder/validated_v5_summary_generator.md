The v5 evidence generator completed successfully for runtime `169c0d97d7d0e9f35d97633f133305f1088b987ef625693d3faa100d25c3e67b`: all 18 commands, 16 public contracts, four pytest cases, both 1025/512 stress streams, and both Watcher runs passed. The canonical outputs are [validation summary](validated_v5_validation_summary.json) and [Watcher summary](validated_v5_watcher_summary.json).

Run from the checkout root to regenerate these CPU-only summaries:

```sh
python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/summarize_validated_v5.py
```

The generator exits 0 for a complete pass, 1 for failed or contradictory evidence, and 2 for incomplete evidence. It does not poll or access a device. `--output-dir /tmp/gemma_v5_summary_check` writes a separate snapshot. Every original v4 check is retained, with v5 artifact names and runtime provenance: heterogeneous B32, prefix/cache preservation, request reuse, BF16-cache compatibility, headline trace/replay checks, all 291 sampled actual-input rows at both 262144 and 262143, end-query/decode gates, four raw pytest reports, paired HF/direct stress comparisons and hashes, and clean Watcher lifecycle/device logs separate from profiling.

The selected v5 policies are checked explicitly: minimal prefill QKV on grid 11×8 with K8 sliding/K16 full, HiFi4, BFP8 weights and FP32 accumulation/output; full prefill output uses minimal matmul K8, while sliding output retains K16. The saved [driver](run_validated_v5.py) must match its hash in the [captured environment](validated_v5_environment_snapshot.json), and that capture must name this persistent driver. Historical v4 and older e810 reports are excluded from v5 completion. The earlier e810 aggregate/end-query gates are not reinterpreted as strict sampled-row passes.

Each maximum/near-maximum run passes all 291 sampled rows. Minimum row PCC is **0.995130377831** sliding and **0.996084490295** full at both lengths. The 512-step optimized HF minima are **0.995555061903** and **0.996252903034**; direct fused-comparison minima are **0.995158301497** and **0.996327176492**. These are sampled layer checks, not all-token/full-model accuracy claims.

[CPU checks](validated_v5_summary_generator_checks.json) cover stale runtime, wrong selected projection geometry/backend, driver hash mismatch, a numeric row failure despite a passing flag, rejection of v4 journal evidence, command planning, and native minimal-config parsing. Syntax and Black checks passed. Multi-GB fixtures are bound through recorded run/manifest hashes; saved compact oracles and stress tensors are rehashed. No runtime or test acceptance was edited.
