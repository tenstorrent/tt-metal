# Decode projection precision-cost ablations

These standalone diagnostic snapshots start from corrected-FMA explicit-norm source `8391717476ae224ee9b82c203d7dec21ceab69632c5edd0abd89bf8cc4daeb03`. The parent reports that source passes the layer34/seed154 long stream and continuation checks. The ablations ask whether ordered Q/Z and output projections remain necessary after correcting prefill state; they do not assume the answer or claim a speed improvement.

| Ablation | Snapshot | SHA256 |
|---|---|---|
| Native Q/Z projection only | `v6_corrected_fma_decode_native_qz.py.snapshot` | `e448bf91b3314fb198710c9b72a1eac0a9f57e4b9735d3ed7e53e054ceef0d32` |
| Native attention output projection only | `v6_corrected_fma_decode_native_out.py.snapshot` | `a228b46e04dd9019e9d32487dc7ef99feec40e9e4a5984db781ece30ed387587` |
| Both | `v6_corrected_fma_decode_native_qz_out.py.snapshot` | `ff4cbe9173bb2d587f5bf53dd466832659c01b7c30481e17ed295291d9c1d651` |

The Q/Z selector is `not (decode and cache_boundary)`. It bypasses both ordered output_only_packed branches, allowing the existing singleton `decode_linear` configuration, weights, dtype, and placement to execute. The attention_packed ordered path remains enabled. The output-projection ablation changes only the final decode return's `ordered_out` flag from cache_boundary to False. Prefill returns earlier; singleton prefill already passes cache_boundary=False. Consequently every prefill path, including public S=1 and internal tails, retains its prior arithmetic. Router ordering, norms, corrected chunk graph, and state boundaries remain unchanged.

`tests/v6_build_decode_projection_ablation.py` pins the base hash, proves complete-module AST identity after restoring the selectors, compiles each emitted source without importing it, and records the selector truth table. Only `_linear_attention` changes for native_out; `linear` and `_linear_attention` change for Q/Z variants. Adjacent provenance and exact builder snapshots accompany each source. Formatting and Python compilation pass. No hardware or existing-source edits occurred during preparation.

Reproduce using fresh artifact paths:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_decode_projection_ablation --variant native_qz --output NEW_QZ_SNAPSHOT --report NEW_QZ_REPORT
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_decode_projection_ablation --variant native_out --output NEW_OUT_SNAPSHOT --report NEW_OUT_REPORT
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_decode_projection_ablation --variant native_qz_out --output NEW_COMBINED_SNAPSHOT --report NEW_COMBINED_REPORT
```

Parent-owned full-stream accuracy must decide each independently. Only passing candidates should enter complete warmed traced-decode comparisons. A failure identifies precision still required with the corrected initial state; previous failures with inaccurate state do not settle this experiment.
