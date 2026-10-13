# Known issues: gemma4_a4b_d_p

Model-specific lessons moved from the shared file. Shared lessons: `models/demos/common/bringup/knowledge/known_issues.md`.

- **Gemma-4 global attention semantics.** Symptom: wrong global-layer value state if V is copied from K. Cause: with `attention_k_eq_v` there is no v_proj; V is the unscaled RMS norm of the raw k_proj output, K = RoPE(k_norm(k_proj)). Gemma-4 RMSNorm is `x * w` (not `1 + w`) and the attention scale is 1.0. Fix: store K and V separately from one projection. Applies to: method (Gemma-4 attention semantics). Found: gemma4_a4b_d_p R.2.
