# Known issues: mimo_v2_6_d_p_2x2

Model-specific lessons moved from the shared file. Shared lessons: `models/demos/common/bringup/knowledge/known_issues.md`.

- **[M22-1] A 2x2 contract cache without sequence parallelism.** Symptom: gpt_oss_d_p's KV helpers give a half-length cache per chip; `update_padded_kv_cache` offsets block-cyclically on `cluster_axis`. Cause: they assume sequence over rows, heads over columns. Fix: allocate the full cache, write with `cluster_axis=0, kv_actual_global=2 * start`, all_gather the input (`dim=-1, cluster_axis=0`). Detail: known_issues_detail.md. Applies to: models/demos/mimo_v2_6_d_p_2x2/tt/runners, ttnn.experimental.deepseek_prefill.update_padded_kv_cache. Seen at: d8b010327de. Found: mimo_v2_6_d_p_2x2 K.1.
