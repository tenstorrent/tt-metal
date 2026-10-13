# Repo map: mimo_v2_6_d_p

| Need | Where it is |
|---|---|
| noaux_tc router with a large correction bias (selection precision-bound) | `models/demos/mimo_v2_6_d_p/tt/router.py:TtRouter` (fp32 linear -> sigmoid -> add bias -> `ttnn.topk` -> gather / renorm -> bf16 scatter). moe_grouped_topk sorts TF32 keys unless `fp32_selection=True`, see known issues. |
| Routed experts at a chosen fidelity without a ttnn change | `models/demos/mimo_v2_6_d_p/tt/experts.py:TtExperts._loop_experts` (dispatch -> per-expert `deepseek_prefill.extract` -> `ttnn.linear` SwiGLU -> `insert` -> combine/reduce); `LazyExpertWeights` dequantizes mxfp4 experts one at a time into the TtRoutedExpert bfp8 cache |
| All-device model with fp8/mxfp4 weights, dense + MoE graphs, shared module builders | `models/demos/mimo_v2_6_d_p/tt/model.py` (`build_*` used by both the component hooks and `TtMiMoBlock`, so hybrid and all-device paths build identical modules) |
| Serving KV contract with a layer subset and per-layer KV head counts (1 or 2 heads of 192 per chip, V padded) | `models/demos/mimo_v2_6_d_p/tt/runners/kv_contract.py` (384-wide per-chip slab, pad columns checked zero on read-back) + `tt/runners/adapter.py` (`served_layers`, `num_kv_cache_layers`, on-device pad-id clamp) |
