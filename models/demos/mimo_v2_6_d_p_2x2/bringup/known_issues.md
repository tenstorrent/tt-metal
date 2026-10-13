# Known issues: mimo_v2_6_d_p_2x2

Model-specific lessons moved from the shared file. Shared lessons: `models/demos/common/bringup/knowledge/known_issues.md`.

- **Contract cache on a 2x2 mesh with no sequence parallelism.** Symptom: gpt_oss_d_p's KV cache helpers give a half-length cache per chip on 2x2, and `update_padded_kv_cache` always offsets block-cyclically along `cluster_axis`. Cause: they assume sequence over rows, heads over columns. Fix: allocate the full cache yourself, write with `cluster_axis=0, kv_actual_global=2 * start`, gather the input with `ttnn.all_gather(dim=-1, cluster_axis=0)` (`mimo_v2_6_d_p_2x2/tt/runners/`). Detail: known_issues_detail.md. Found: mimo_v2_6_d_p_2x2 K.1.
