# Kimi K2.5: Disaggregated prefill

This directory contains the implementation of the prefill stage for the Kimi
K2.5 text decoder. The architecture mirrors `models/demos/deepseek_v3_d_p`
1:1 because Kimi K2.5 reuses the DeepSeek V3 modeling code (its
`KimiK25ForConditionalGeneration` wraps a `DeepseekV3ForCausalLM` for the
text path); only the model hyper-parameters and a few code paths differ.

Vision tower / multimodal projector are out of scope for this prefill
bring-up.

## Differences vs DeepSeek V3 (`deepseek_v3_d_p`)

Source of truth: [`reference/config.json`](reference/config.json) (`text_config`).
Mirror constants in [`reference/kimi_k25_config.py`](reference/kimi_k25_config.py).

| Field                        | DeepSeek V3 | Kimi K2.5   |
| ---------------------------- | ----------- | ----------- |
| `vocab_size`                 | 129280      | **163840**  |
| `hidden_size`                | 7168        | 7168        |
| `intermediate_size` (dense)  | 18432       | 18432       |
| `moe_intermediate_size`      | 2048        | 2048        |
| `num_hidden_layers`          | 61          | 61          |
| `first_k_dense_replace`      | 3           | **1**       |
| `num_attention_heads`        | 128         | **64**      |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | 1536 / 512 |
| head dims (qk_nope/qk_rope/v_head) | 128/64/128 | 128/64/128 |
| `n_routed_experts`           | 256         | **384**     |
| `num_experts_per_tok`        | 8           | 8           |
| `n_group`                    | 8           | **1**       |
| `topk_group`                 | 4           | **1**       |
| `routed_scaling_factor`      | 2.5         | **2.827**   |
| `max_position_embeddings`    | 4096        | **262144**  |
| `rope_theta`                 | 10000.0     | **50000.0** |
| `rope_scaling.type`          | (none)      | **yarn**    |
| YaRN `factor` / `original_max_pos` | -- | **64 / 4096** |

The MoE gate has an explicit ungrouped fast path (`n_expert_groups == 1`)
in [`tt/moe/tt_moe_gate_prefill.py`](tt/moe/tt_moe_gate_prefill.py) that
skips the per-group reshape and group-level top-k, going directly to a
flat top-k over all 384 routed experts.

## Environment Variables

- **`KIMI_K25_HF_MODEL`** — Path to Kimi K2.5 weights directory. Falls back
  to `models/demos/kimi26_d_p/reference/`, then
  `/proj_sw/user_dev/moonshotai/Kimi-K2.5`, then downloads from
  `moonshotai/Kimi-K2.5` on Hugging Face.
- **`TT_KIMI_PREFILL_TTNN_CACHE`** — Directory for cached TTNN weight
  tensors (`.tensorbin` files). First run writes cache, subsequent runs
  load directly. Defaults to
  `{model_path}/tensor_cache_{arch}_{num_devices}dev/`.
- **`TT_KIMI_PREFILL_HOST_REF_CACHE`** — Directory for cached host
  reference snapshots used in PCC validation. Defaults to
  `/tmp/kimi_k25_transformer_ref_cache`.
- **`TT_KIMI_PREFILL_INFINITEBENCH_CACHE`** — Directory for cached
  InfiniteBench prompt data. Defaults to
  `/tmp/kimi_k25_transformer_inputs`.
- **`TT_KIMI_PREFILL_DEBUG_TOKEN_COUNT`** — Enable debug output for
  per-expert token counts in MoE forward pass. Set to `1`, `true`, or
  `yes` to enable. Defaults to disabled. Warning: enabling this adds
  device-to-host transfer overhead on every MoE layer forward.

## Weight Loading and TTNN Cache

See [tt/WEIGHTS_AND_CACHE.md](tt/WEIGHTS_AND_CACHE.md) for the weight
loading contract that every TT module implements.

## Bring-up order

Run mirrored tests in this dependency order:

1. **Primitives** — `tests/pcc/test_parallel_embedding.py`,
   `tests/pcc/test_rmsnorm.py`, `tests/pcc/test_lm_head.py`,
   `tests/op_unit_tests/test_rope_prefill.py` (validates Kimi YaRN at
   `seq128k` and `seq256k_kimi_longctx`),
   `tests/op_unit_tests/test_mla_matmuls.py`.
2. **MLA** — `tests/test_mla.py`, `tests/test_mla_disaggregation.py`,
   `tests/cache/test_mla_cache.py`. The MLA reference is shared from
   `models/demos/deepseek_v3_d_p/reference/mla_reference.py` (the same
   `DeepseekV3Attention` wrapper, fed Kimi's text config with 64 heads
   and YaRN rope).
3. **Dense FFN** — `tests/pcc/test_shared_expert.py`,
   `tests/pcc/test_ffn.py` (+ corresponding cache tests).
4. **MoE kernels** — `tests/op_unit_tests/test_moe_grouped_topk.py`
   (parametrized to exercise both the DS 8-group routing and the Kimi
   1-group degenerate routing),
   `tests/pcc/test_moe_gate_prefill2d.py` and the bf16/bfp8 quantization
   variants, `tests/pcc/test_masked_bincount.py`,
   `tests/pcc/test_offset_cumsum.py`,
   `tests/pcc/test_moe_routing_setup.py`.
5. **Dispatch / experts / combine / reduce** —
   `tests/pcc/test_single_routed_expert.py`,
   `tests/pcc/test_ttnn_routed_expert.py`,
   `tests/pcc/test_prefill_dispatch.py`,
   `tests/pcc/test_prefill_combine.py`,
   `tests/op_unit_tests/test_dispatch_combine_l1_small_semaphores.py`,
   `tests/pcc/test_ttnn_dispatch_combine.py`,
   `tests/pcc/test_reduce.py`.
6. **Stitched MoE** — `tests/pcc/test_ttnn_moe.py` +
   `tests/cache/test_moe_cache.py` + `tests/torch/test_moe.py`.
7. **Block** — `tests/test_prefill_block.py`.
8. **Full transformer** — `tests/test_prefill_block_loop.py`,
   `tests/test_prefill_transformer.py`.

## Open follow-ups

- **MLA matmul calibration** — `tt/mla/mla_config.py` program-configs
  were inherited from DeepSeek's 128-head shapes. With Kimi's 64 heads
  the per-head Q/K/V projection widths halve along the head dimension;
  re-tune via `tests/op_unit_tests/test_mla_matmuls.py`.
- **Long-context** — Kimi `max_position_embeddings = 262144` exceeds
  what the DS pipeline has been exercised at. Validate end-to-end via
  `tests/op_unit_tests/test_rope_prefill.py[seq256k_kimi_longctx]` and
  long-seq cases in `tests/test_prefill_block_loop.py`.
- **Expert sharding** — `tt_prefill_transformer.py` defaults
  `experts_per_chip = 12` (= 384 / 32). For other mesh sizes pass an
  explicit value (e.g. 384/8 = 48 on an 8-chip dispatch group).
- **Quantization** — Kimi's `text_config.quantization_config` advertises
  int4 (group_size=32, packed) for routed-expert weights. The current
  loader relies on HF runtime dequantization to bf16; a native TTNN int4
  loader is not implemented.
- **Vision tower** — The multimodal projector and ViT3D vision tower in
  `reference/modeling_kimi_k25.py` are intentionally out of scope.
