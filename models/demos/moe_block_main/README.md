# moe_block_main

Marko Staletovic's reworked MoE block (prefill), the Python side of `ttnn.bringup.moe_block_main`
(ttnn/ttnn/bringup/moe_block_main_ttnn): a separate copy next to models/demos/moe_block (the block this branch has run
since DeepSeek-V4.1 P.17), never a replacement of it.

Source: `mstaletovic/moe-block-main` @ `0c0bbe59d98` (on origin/main `f93403fa228`, 2026-11-06):
- `tt/moe_block.py`, `tt/flat_gather.py`, `tests/` <- `models/demos/moe_block/{tt,tests}`
- `tt/moe_impl.py` <- `models/demos/deepseek_v3_d_p/tt/moe_impl.py` (the mesh's L1_SMALL / fabric payload for the block)
- `tt/k3_moe_block_ffn.py` <- `models/demos/deepseek_v3_d_p/tt/kimi_k3/moe_block_ffn.py` (GLM-5.3 / Kimi-K3's layer on
  the block; the tests' model dims and weight loading)

Renamed only: `ttnn.experimental.moe_block` -> `ttnn.bringup.moe_block_main`, `models.demos.moe_block` ->
`models.demos.moe_block_main`, the two deepseek_v3_d_p modules -> the two files above, the op tests' path
(ttnn/ttnn/bringup/moe_block_main_ttnn/tests). Not ported: the source commit's deepseek_v3_d_p model integration
(GLM-5.3 / Kimi-K3 block.py, transformer, MTP, the prefill runner's mesh open, CI lists, perf baselines) and its
`glm_moe_reference` bias change (the GLM-5.3 cases of tests/test_moe_block_models.py take this tree's reference).

DeepSeek-V4.1 runs it with `DS41_MOE_BLOCK_SRC=main` (models/demos/deepseek_v41_flash_d_p/tt/moe_block_main_adapter.py).
